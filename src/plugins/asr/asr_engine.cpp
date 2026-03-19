// asr_engine.cpp — Qwen3-ASR 推理引擎实现
//
// 加载权重 + 编排 encoder/decoder + transcribe 接口

#include "asr_engine.h"
#include "audio_utils.h"
#include "audio_ops.h"
#include "mel_gpu.h"
#include "engine/safetensors.h"
#include "engine/tokenizer.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen_thor {
namespace asr {

// ============================================================================
// Special token IDs (Qwen3-ASR vocab)
// ============================================================================

static constexpr int IM_START     = 151644;
static constexpr int IM_END       = 151645;
static constexpr int AUDIO_START  = 151669;
static constexpr int AUDIO_END    = 151670;
static constexpr int AUDIO_PAD    = 151676;
static constexpr int ENDOFTEXT    = 151643;
static constexpr int ASR_TEXT_TOKEN = 151704;  // <asr_text> — force text-only output mode

// ============================================================================
// ASREngine implementation
// ============================================================================

ASREngine::ASREngine() = default;

ASREngine::~ASREngine() {
    for (auto ptr : device_weights_) cudaFree(ptr);
    if (mel_gpu_) cudaFree(mel_gpu_);
    if (encoder_out_) cudaFree(encoder_out_);
    if (input_embeds_) cudaFree(input_embeds_);
    if (logits_) cudaFree(logits_);
    if (position_ids_) cudaFree(position_ids_);
    if (token_id_gpu_) cudaFree(token_id_gpu_);
    if (prompt_tokens_gpu_) cudaFree(prompt_tokens_gpu_);
    if (mel_staging_gpu_) cudaFree(mel_staging_gpu_);
    // Batch decode buffers
    if (batch_logits_) cudaFree(batch_logits_);
    if (batch_token_ids_) cudaFree(batch_token_ids_);
    if (batch_position_ids_) cudaFree(batch_position_ids_);
    if (batch_result_ids_) cudaFree(batch_result_ids_);
}

// ============================================================================
// Greedy argmax on GPU — single CTA reduction, result in managed memory
// Replaces CPU version that copied 496KB per step
// ============================================================================

// (greedy_decode removed — using audio_ops::invoke_argmax + token_id_gpu_)

// ============================================================================
// Build prompt token IDs
// ============================================================================

void ASREngine::build_prompt(int encoder_out_len, std::vector<int>& token_ids,
                            const std::string& language) {
    token_ids.clear();

    // Official Qwen3-ASR prompt format (empty system content):
    // <|im_start|>system\n<|im_end|>\n
    token_ids.push_back(IM_START);
    token_ids.push_back(8948);   // system
    token_ids.push_back(198);    // \n
    token_ids.push_back(IM_END);
    token_ids.push_back(198);    // \n

    // <|im_start|>user\n
    token_ids.push_back(IM_START);
    token_ids.push_back(872);    // user
    token_ids.push_back(198);    // \n

    // <|audio_start|><|audio_pad|>×N<|audio_end|>
    token_ids.push_back(AUDIO_START);
    for (int i = 0; i < encoder_out_len; i++) {
        token_ids.push_back(AUDIO_PAD);
    }
    token_ids.push_back(AUDIO_END);

    // <|im_end|>\n
    token_ids.push_back(IM_END);
    token_ids.push_back(198);    // \n

    // <|im_start|>assistant\n
    token_ids.push_back(IM_START);
    token_ids.push_back(77091);  // assistant
    token_ids.push_back(198);    // \n

    // Force text-only mode: "language Chinese<asr_text>"
    // This matches official qwen-asr behavior when language is specified
    if (!language.empty()) {
        // "language" = [11528], " Chinese" = [8453], <asr_text> = [151704]
        // For other languages, token IDs differ — but Chinese covers our main use case
        token_ids.push_back(11528);  // language
        token_ids.push_back(8453);   //  Chinese
        token_ids.push_back(ASR_TEXT_TOKEN);  // <asr_text>
    }
}

// ============================================================================
// Load weights from safetensors
// ============================================================================

void ASREngine::load_weights(const std::string& model_dir) {
    using namespace qwen_thor::io;

    std::string dir = model_dir;
    if (dir.back() != '/') dir += '/';

    // Open both shards simultaneously
    SafetensorsLoader shard1(dir + "model-00001-of-00002.safetensors");
    SafetensorsLoader shard2(dir + "model-00002-of-00002.safetensors");

    // Helper: load tensor from whichever shard contains it
    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        SafetensorsLoader* loader = nullptr;
        if (shard1.has_tensor(name))      loader = &shard1;
        else if (shard2.has_tensor(name)) loader = &shard2;
        else {
            fprintf(stderr, "[ASR] WARNING: tensor '%s' not found in any shard\n", name.c_str());
            return nullptr;
        }
        auto tensor = loader->get_tensor(name);
        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, tensor->nbytes());
        cudaMemcpy(d_ptr, tensor->data(), tensor->nbytes(), cudaMemcpyHostToDevice);
        device_weights_.push_back(d_ptr);
        return reinterpret_cast<__nv_bfloat16*>(d_ptr);
    };

    // --- Encoder weights ---

    // Conv2D frontend
    auto c1w = load_tensor("thinker.audio_tower.conv2d1.weight");
    auto c1b = load_tensor("thinker.audio_tower.conv2d1.bias");
    auto c2w = load_tensor("thinker.audio_tower.conv2d2.weight");
    auto c2b = load_tensor("thinker.audio_tower.conv2d2.bias");
    auto c3w = load_tensor("thinker.audio_tower.conv2d3.weight");
    auto c3b = load_tensor("thinker.audio_tower.conv2d3.bias");
    auto cow = load_tensor("thinker.audio_tower.conv_out.weight");
    encoder_->set_conv_weights(c1w, c1b, c2w, c2b, c3w, c3b, cow);

    // Encoder transformer layers (0-23)
    for (int i = 0; i < config_.encoder_layers; i++) {
        std::string prefix = "thinker.audio_tower.layers." + std::to_string(i) + ".";
        EncoderLayerWeights elw;
        elw.self_attn_layer_norm_w = load_tensor(prefix + "self_attn_layer_norm.weight");
        elw.self_attn_layer_norm_b = load_tensor(prefix + "self_attn_layer_norm.bias");
        elw.q_proj_w = load_tensor(prefix + "self_attn.q_proj.weight");
        elw.q_proj_b = load_tensor(prefix + "self_attn.q_proj.bias");
        elw.k_proj_w = load_tensor(prefix + "self_attn.k_proj.weight");
        elw.k_proj_b = load_tensor(prefix + "self_attn.k_proj.bias");
        elw.v_proj_w = load_tensor(prefix + "self_attn.v_proj.weight");
        elw.v_proj_b = load_tensor(prefix + "self_attn.v_proj.bias");
        elw.o_proj_w = load_tensor(prefix + "self_attn.out_proj.weight");
        elw.o_proj_b = load_tensor(prefix + "self_attn.out_proj.bias");
        elw.final_layer_norm_w = load_tensor(prefix + "final_layer_norm.weight");
        elw.final_layer_norm_b = load_tensor(prefix + "final_layer_norm.bias");
        elw.fc1_w = load_tensor(prefix + "fc1.weight");
        elw.fc1_b = load_tensor(prefix + "fc1.bias");
        elw.fc2_w = load_tensor(prefix + "fc2.weight");
        elw.fc2_b = load_tensor(prefix + "fc2.bias");
        encoder_->set_layer_weights(i, elw);
    }

    // Encoder post-processing
    auto ln_post_w = load_tensor("thinker.audio_tower.ln_post.weight");
    auto ln_post_b = load_tensor("thinker.audio_tower.ln_post.bias");
    auto proj1_w = load_tensor("thinker.audio_tower.proj1.weight");
    auto proj1_b = load_tensor("thinker.audio_tower.proj1.bias");
    auto proj2_w = load_tensor("thinker.audio_tower.proj2.weight");
    auto proj2_b = load_tensor("thinker.audio_tower.proj2.bias");
    encoder_->set_post_weights(ln_post_w, ln_post_b, proj1_w, proj1_b, proj2_w, proj2_b);

    // --- Decoder weights ---

    // Shared embeddings
    auto embed_w = load_tensor("thinker.model.embed_tokens.weight");
    auto lm_head_w = load_tensor("thinker.lm_head.weight");
    this->embed_tokens_w_ = embed_w;

    // All 28 decoder layers
    for (int i = 0; i < config_.decoder_layers; i++) {
        std::string prefix = "thinker.model.layers." + std::to_string(i) + ".";
        DecoderLayerWeights dlw;
        dlw.input_layernorm_w = load_tensor(prefix + "input_layernorm.weight");
        dlw.q_proj_w = load_tensor(prefix + "self_attn.q_proj.weight");
        dlw.k_proj_w = load_tensor(prefix + "self_attn.k_proj.weight");
        dlw.v_proj_w = load_tensor(prefix + "self_attn.v_proj.weight");
        dlw.o_proj_w = load_tensor(prefix + "self_attn.o_proj.weight");
        dlw.q_norm_w = load_tensor(prefix + "self_attn.q_norm.weight");
        dlw.k_norm_w = load_tensor(prefix + "self_attn.k_norm.weight");
        dlw.post_attention_layernorm_w = load_tensor(prefix + "post_attention_layernorm.weight");
        dlw.gate_proj_w = load_tensor(prefix + "mlp.gate_proj.weight");
        dlw.up_proj_w = load_tensor(prefix + "mlp.up_proj.weight");
        dlw.down_proj_w = load_tensor(prefix + "mlp.down_proj.weight");
        decoder_->set_layer_weights(i, dlw);
    }

    // Final norm
    auto final_norm_w = load_tensor("thinker.model.norm.weight");
    decoder_->set_embed_weights(embed_w, lm_head_w, final_norm_w);

    fprintf(stderr, "[ASR] Loaded %zu tensors from 2 shards\n", device_weights_.size());
}

// ============================================================================
// load_model: config + weights + initialize
// ============================================================================

void ASREngine::load_model(const std::string& model_dir) {
    auto t0 = std::chrono::steady_clock::now();

    // 1. Load config
    if (!config_.load_from_model_dir(model_dir)) {
        fprintf(stderr, "[ASR] ERROR: failed to load config from %s\n", model_dir.c_str());
        return;
    }
    fprintf(stderr, "[ASR] Config: encoder %dL d=%d, decoder %dL d=%d, vocab=%d\n",
            config_.encoder_layers, config_.encoder_d_model,
            config_.decoder_layers, config_.decoder_hidden_size,
            config_.vocab_size);

    // 2. Save model_dir and load tokenizer
    model_dir_ = model_dir;
    if (!tokenizer_.load(model_dir)) {
        fprintf(stderr, "[ASR] WARNING: failed to load tokenizer from %s\n", model_dir.c_str());
    }

    // 3. Create encoder and decoder
    encoder_ = std::make_unique<AudioEncoder>(config_);
    // max_seq_len for decoder: prompt + encoder output + max generation
    int max_decoder_seq = 2048;  // conservative max
    decoder_ = std::make_unique<TextDecoder>(config_, max_decoder_seq);

    // 4. Load weights
    load_weights(model_dir);

    // 5. Initialize encoder and decoder
    encoder_->initialize(stream_);
    decoder_->initialize(stream_);

    // 5.5. Optimize decoder weights (QKV merge + RMSNorm centered transform)
    decoder_->prepare_optimized_weights(stream_);

    // 6. Allocate GPU buffers
    // Max mel frames: with center padding, T_max audio (120s) at 16kHz
    // produces (T_max*16000 + n_fft - n_fft) / hop + 1 = T_max*100 + 1 frames.
    // Add margin for rounding. 12100 frames supports up to ~120s audio.
    max_mel_frames_ = 12100;
    cudaMalloc(&mel_gpu_, (size_t)config_.num_mel_bins * max_mel_frames_ * sizeof(__nv_bfloat16));

    // F32 staging buffer for GPU mel→BF16 conversion
    cudaMalloc(&mel_staging_gpu_, (size_t)config_.num_mel_bins * max_mel_frames_ * sizeof(float));

    // Max encoder output tokens
    int max_enc_tokens = config_.max_source_positions;
    cudaMalloc(&encoder_out_, (size_t)max_enc_tokens * config_.output_dim * sizeof(__nv_bfloat16));

    // Max prompt length: ~20 prompt tokens + encoder output + ~20 suffix
    max_prompt_len_ = max_enc_tokens + 50;
    cudaMalloc(&input_embeds_, (size_t)max_prompt_len_ * config_.decoder_hidden_size * sizeof(__nv_bfloat16));

    // Logits buffer (single token)
    cudaMalloc(&logits_, config_.vocab_size * sizeof(__nv_bfloat16));

    // Position IDs for MRoPE [3, max_seq]
    cudaMalloc(&position_ids_, 3 * max_decoder_seq * sizeof(int));

    // Token ID for decode step (device memory + cudaMemcpy, no managed memory)
    cudaMalloc(&token_id_gpu_, sizeof(int));

    // Pre-allocated prompt token buffer (avoids per-call cudaMalloc/Free)
    cudaMalloc(&prompt_tokens_gpu_, max_prompt_len_ * sizeof(int));

    // 7. Pre-compute mel filterbank and Hann window
    init_mel_cache();

    auto t1 = std::chrono::steady_clock::now();
    float load_time = std::chrono::duration<float>(t1 - t0).count();
    fprintf(stderr, "[ASR] Model loaded in %.1fs, %zu weight tensors (%.1f MB)\n",
            load_time, device_weights_.size(),
            [&]() {
                size_t total = 0;
                // Estimate from number of weights
                // Encoder: ~300M params, Decoder: ~1.4B params, total ~1.7B
                // At BF16: ~3.4 GB
                return 3400.0;  // approximate
            }());
    loaded_ = true;
}

// ============================================================================
// init_mel_cache: 预计算 mel filterbank + Hann window
// ============================================================================

void ASREngine::init_mel_cache() {
    cached_n_fft_ = 400;
    cached_n_mels_ = 128;
    cached_sample_rate_ = 16000;

    // Try loading precomputed mel filterbank from model directory
    // (exported from WhisperFeatureExtractor with Slaney mel scale)
    int n_freqs = cached_n_fft_ / 2 + 1;  // 201
    int fb_size = cached_n_mels_ * n_freqs;
    std::string fb_path = model_dir_ + "/mel_filters.bin";
    FILE* f = fopen(fb_path.c_str(), "rb");
    if (f) {
        cached_mel_fb_.resize(fb_size);
        size_t read = fread(cached_mel_fb_.data(), sizeof(float), fb_size, f);
        fclose(f);
        if ((int)read == fb_size) {
            fprintf(stderr, "[ASR] Loaded precomputed mel filterbank from %s (%d values)\n",
                    fb_path.c_str(), fb_size);
        } else {
            fprintf(stderr, "[ASR] WARNING: mel_filters.bin incomplete (%zu/%d), falling back to computed\n",
                    read, fb_size);
            cached_mel_fb_ = audio::build_mel_filterbank(cached_n_mels_, cached_n_fft_, cached_sample_rate_);
        }
    } else {
        cached_mel_fb_ = audio::build_mel_filterbank(cached_n_mels_, cached_n_fft_, cached_sample_rate_);
        fprintf(stderr, "[ASR] Using computed mel filterbank (no %s found)\n", fb_path.c_str());
    }

    cached_hann_window_ = audio::build_hann_window(cached_n_fft_);
    fprintf(stderr, "[ASR] Mel filterbank cached: %d mels, %d FFT, %d Hz\n",
            cached_n_mels_, cached_n_fft_, cached_sample_rate_);

    // Initialize GPU Whisper mel (cuFFT-accelerated)
    gpu_whisper_mel_ = std::make_unique<GpuWhisperMel>();
    if (gpu_whisper_mel_->init(cached_mel_fb_.data())) {
        fprintf(stderr, "[ASR] GPU Whisper mel initialized\n");
    } else {
        fprintf(stderr, "[ASR] WARNING: GPU Whisper mel init failed, using CPU fallback\n");
        gpu_whisper_mel_.reset();
    }
}

// ============================================================================
// transcribe: PCM float → text
// ============================================================================

std::string ASREngine::transcribe(
    const float* samples, int num_samples, int sample_rate,
    float temperature, int max_new_tokens, bool suppress_early_eos)
{
    if (!loaded_) {
        fprintf(stderr, "[ASR] ERROR: model not loaded\n");
        return "";
    }

    // Check for lingering CUDA errors from previous chunk
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        fprintf(stderr, "[ASR] WARNING: pre-existing CUDA error: %s\n",
                cudaGetErrorString(prev_err));
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Resample to 16kHz if needed
    std::vector<float> audio_buf;
    const float* audio_16k = samples;
    int audio_len = num_samples;
    if (sample_rate != 16000) {
        std::vector<float> input_vec(samples, samples + num_samples);
        audio::resample(input_vec, sample_rate, audio_buf, 16000);
        audio_16k = audio_buf.data();
        audio_len = (int)audio_buf.size();
    }

    // 2. Compute mel spectrogram
    int mel_frames = 0;
    bool used_gpu_mel = false;

    if (gpu_whisper_mel_) {
        // GPU path: cuFFT-accelerated mel (saves ~488ms/segment vs CPU Bluestein FFT)
        auto mel_result = gpu_whisper_mel_->compute(audio_16k, audio_len);
        gpu_whisper_mel_->sync();
        mel_frames = mel_result.num_frames;

        if (mel_frames > 0 && mel_result.d_mel) {
            if (mel_frames > max_mel_frames_) {
                fprintf(stderr, "[ASR] WARNING: audio too long (%d frames > %d max), truncating\n",
                        mel_frames, max_mel_frames_);
                mel_frames = max_mel_frames_;
            }
            // mel_result.d_mel is [128, T] F32 on GPU → convert to BF16
            int mel_count = 128 * mel_frames;
            audio_ops::invoke_f32_to_bf16(mel_gpu_, mel_result.d_mel, mel_count, stream_);
            used_gpu_mel = true;
        }
    }

    if (!used_gpu_mel) {
        // CPU fallback path
        audio::MelConfig mel_cfg;
        mel_cfg.n_fft = 400;
        mel_cfg.hop_length = 160;
        mel_cfg.n_mels = 128;
        mel_cfg.sample_rate = 16000;

        std::vector<float> mel_f32;
        audio::compute_mel_cached(audio_16k, audio_len, mel_cfg,
                                  cached_mel_fb_, cached_hann_window_,
                                  mel_f32, mel_frames);

        if (mel_frames > max_mel_frames_) {
            fprintf(stderr, "[ASR] WARNING: audio too long (%d frames > %d max), truncating\n",
                    mel_frames, max_mel_frames_);
            mel_frames = max_mel_frames_;
        }

        int mel_count = 128 * mel_frames;
        cudaMemcpyAsync(mel_staging_gpu_, mel_f32.data(),
                        mel_count * sizeof(float), cudaMemcpyHostToDevice, stream_);
        audio_ops::invoke_f32_to_bf16(mel_gpu_, mel_staging_gpu_, mel_count, stream_);
    }

    // 4. Encode audio
    int encoder_out_len = 0;
    encoder_->forward(mel_gpu_, mel_frames, encoder_out_, encoder_out_len, stream_);

    auto t_encode = std::chrono::steady_clock::now();
    float encode_ms = std::chrono::duration<float, std::milli>(t_encode - t0).count();

    if (encoder_out_len == 0) {
        fprintf(stderr, "[ASR] ERROR: encoder produced 0 tokens\n");
        return "";
    }
    fprintf(stderr, "[ASR] Encoder: %d mel frames → %d tokens (%.1f ms)\n",
            mel_frames, encoder_out_len, encode_ms);

    // 5. Build prompt and embed
    std::vector<int> prompt_tokens;
    build_prompt(encoder_out_len, prompt_tokens);  // default: "Chinese"
    int prompt_len = (int)prompt_tokens.size();

    if (prompt_len > max_prompt_len_) {
        fprintf(stderr, "[ASR] ERROR: prompt too long (%d > %d)\n", prompt_len, max_prompt_len_);
        return "";
    }

    // Embed all prompt tokens (using pre-allocated GPU buffer)
    int h = config_.decoder_hidden_size;
    cudaMemcpyAsync(prompt_tokens_gpu_, prompt_tokens.data(),
                    prompt_len * sizeof(int), cudaMemcpyHostToDevice, stream_);
    audio_ops::invoke_embedding_lookup(
        input_embeds_, prompt_tokens_gpu_, embed_tokens_w_,
        prompt_len, h, stream_);

    // 6. Replace AUDIO_PAD embeddings with encoder output
    // Find the start index of audio_pad tokens in the prompt
    int audio_pad_start = -1;
    for (int i = 0; i < prompt_len; i++) {
        if (prompt_tokens[i] == AUDIO_PAD) {
            audio_pad_start = i;
            break;
        }
    }
    if (audio_pad_start >= 0) {
        cudaMemcpyAsync(
            input_embeds_ + (size_t)audio_pad_start * h,
            encoder_out_,
            (size_t)encoder_out_len * h * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream_);
    }

    // 7. Construct position IDs for MRoPE
    // For pure text, all 3 dimensions use the same sequential positions
    std::vector<int> pos_ids(3 * prompt_len);
    for (int d = 0; d < 3; d++) {
        for (int i = 0; i < prompt_len; i++) {
            pos_ids[d * prompt_len + i] = i;
        }
    }
    cudaMemcpy(position_ids_, pos_ids.data(),
               3 * prompt_len * sizeof(int), cudaMemcpyHostToDevice);

    // 8. Prefill decoder
    decoder_->reset_cache();
    decoder_->forward_prefill(input_embeds_, position_ids_, prompt_len, logits_, stream_);

    auto t_prefill = std::chrono::steady_clock::now();
    float prefill_ms = std::chrono::duration<float, std::milli>(t_prefill - t_encode).count();
    fprintf(stderr, "[ASR] Prefill: %d tokens (%.1f ms)\n", prompt_len, prefill_ms);

    // 9. Autoregressive decode (GPU argmax — no D2H logits transfer)
    //
    // EOS 抑制 (可选): 根据音频时长设置最小输出 token 数。
    // 语音停顿可能导致模型过早生成 <|im_end|>, 在最小 token 数之前
    // 将 EOS logits 设为 -inf, 强制模型继续转录后续内容。
    int min_new_tokens = 0;
    if (suppress_early_eos) {
        float audio_duration_s = (float)audio_len / 16000.0f;
        min_new_tokens = std::max(1, (int)(audio_duration_s * 0.4f));
        fprintf(stderr, "[ASR] EOS suppression: audio=%.1fs, min_new_tokens=%d\n",
                audio_duration_s, min_new_tokens);
    }

    std::vector<int> output_tokens;

    // 首个 token: 若需要抑制 EOS 则先抑制再 argmax
    if (min_new_tokens > 0) {
        audio_ops::invoke_suppress_eos(logits_, IM_END, ENDOFTEXT, stream_);
    }
    audio_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
    int next_token;
    cudaMemcpyAsync(&next_token, token_id_gpu_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    int current_pos = prompt_len;

    while (next_token != IM_END && next_token != ENDOFTEXT
           && (int)output_tokens.size() < max_new_tokens
           && current_pos < max_prompt_len_ + max_new_tokens) {
        output_tokens.push_back(next_token);

        // Inline repetition check every 50 tokens — early stop on severe repetition
        if (output_tokens.size() >= 50 && output_tokens.size() % 50 == 0) {
            auto& ot = output_tokens;
            int n = (int)ot.size();
            bool found_rep = false;
            // Check pattern lengths 1-32 for 5+ consecutive repeats in last 200 tokens
            int check_start = std::max(0, n - 200);
            for (int pl = 1; pl <= 32 && !found_rep; ++pl) {
                for (int s = check_start; s + pl * 5 <= n && !found_rep; ++s) {
                    int reps = 0;
                    while (s + pl * (reps + 1) <= n) {
                        bool match = true;
                        for (int j = 0; j < pl; ++j) {
                            if (ot[s + j] != ot[s + pl * (reps + 1) + j]) { match = false; break; }
                        }
                        if (!match) break;
                        reps++;
                    }
                    if (reps >= 5) {
                        fprintf(stderr, "[ASR] Inline repetition at %d tokens (pat=%d reps=%d), stopping.\n",
                                n, pl, reps + 1);
                        found_rep = true;
                    }
                }
            }
            if (found_rep) break;
        }

        // Position IDs for decode step: [3, 1] all same position
        int step_pos[3] = {current_pos, current_pos, current_pos};
        cudaMemcpyAsync(position_ids_, step_pos, 3 * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);

        decoder_->forward_decode(next_token, position_ids_, logits_, stream_);

        // EOS 抑制: 未达到最小 token 数时禁止生成 EOS
        if ((int)output_tokens.size() < min_new_tokens) {
            audio_ops::invoke_suppress_eos(logits_, IM_END, ENDOFTEXT, stream_);
        }
        audio_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
        cudaMemcpyAsync(&next_token, token_id_gpu_, sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        current_pos++;
    }

    auto t_decode = std::chrono::steady_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(t_decode - t_prefill).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_decode - t0).count();
    fprintf(stderr, "[ASR] Decode: %d tokens (%.1f ms, %.1f tok/s)\n",
            (int)output_tokens.size(), decode_ms,
            output_tokens.empty() ? 0 : output_tokens.size() / (decode_ms / 1000.0f));
    fprintf(stderr, "[ASR] Total: %.1f ms (encode %.0f + prefill %.0f + decode %.0f)\n",
            total_ms, encode_ms, prefill_ms, decode_ms);

    // 10. Detect and fix token repetitions (port from official qwen-asr)
    // Look for patterns of 1-32 tokens repeating 5+ times consecutively
    if (output_tokens.size() >= 10) {
        for (int pat_len = 1; pat_len <= 32 && pat_len * 5 <= (int)output_tokens.size(); ++pat_len) {
            for (int start = 0; start + pat_len * 5 <= (int)output_tokens.size(); ++start) {
                int repeats = 0;
                while (start + pat_len * (repeats + 1) <= (int)output_tokens.size()) {
                    bool match = true;
                    for (int j = 0; j < pat_len; ++j) {
                        if (output_tokens[start + j] != output_tokens[start + pat_len * (repeats + 1) + j]) {
                            match = false;
                            break;
                        }
                    }
                    if (!match) break;
                    repeats++;
                }
                if (repeats >= 5) {
                    int keep = start + pat_len; // keep only 1 occurrence
                    fprintf(stderr, "[ASR] Repetition detected: pattern_len=%d repeats=%d, "
                            "truncated %zu→%d tokens\n", pat_len, repeats + 1,
                            output_tokens.size(), keep);
                    output_tokens.resize(keep);
                    goto rep_done;
                }
            }
        }
        rep_done:;
    }

    // 11. Decode tokens to text — skip "language XXX <asr_text>" header
    std::string result;
    if (!output_tokens.empty()) {
        // Debug: print token IDs
        fprintf(stderr, "[ASR] Token IDs (%zu):", output_tokens.size());
        for (auto id : output_tokens) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");

        // Find LAST <asr_text> marker (token 151704) and only decode tokens after it.
        // The model may output a double header pattern for no-speech segments:
        //   11528 2240 151704 151644 11528 2240 151704
        // Using the last marker skips the spurious "language None" prefix.
        static constexpr int ASR_TEXT_TOKEN = 151704;
        size_t text_start = 0;
        for (size_t i = 0; i < output_tokens.size(); i++) {
            if (output_tokens[i] == ASR_TEXT_TOKEN) {
                text_start = i + 1;  // keep updating → ends on last occurrence
            }
        }

        if (text_start > 0 && text_start < output_tokens.size() && tokenizer_.is_loaded()) {
            // Decode only text tokens after <asr_text> marker
            std::vector<int> text_tokens(output_tokens.begin() + text_start,
                                          output_tokens.end());
            result = tokenizer_.decode(text_tokens);
        } else if (text_start > 0) {
            // <asr_text> found but no text tokens after it → no speech detected
            result = "";
        } else if (tokenizer_.is_loaded()) {
            // No <asr_text> marker at all → decode everything as fallback
            result = tokenizer_.decode(output_tokens);
        } else {
            result = "[" + std::to_string(output_tokens.size()) + " tokens]";
        }
    }

    return result;
}

// ============================================================================
// transcribe_file: WAV file → text
// ============================================================================

std::string ASREngine::transcribe_file(
    const std::string& wav_path,
    float temperature, int max_new_tokens, bool suppress_early_eos)
{
    // Load WAV file
    audio::AudioData wav;
    if (!audio::load_wav(wav_path, wav)) {
        fprintf(stderr, "[ASR] ERROR: failed to load WAV file: %s\n", wav_path.c_str());
        return "";
    }

    return transcribe(wav.samples.data(), (int)wav.samples.size(), wav.sample_rate,
                      temperature, max_new_tokens, suppress_early_eos);
}

// ============================================================================
// transcribe_batch: multiple audio chunks → batch decode (GEMV→GEMM)
//
// Flow:
//   For each chunk: mel → encode → build prompt → embed → prefill (batch KV)
//   Then batch decode: all chunks generate tokens simultaneously
// ============================================================================

std::vector<std::string> ASREngine::transcribe_batch(
    const std::vector<AudioChunk>& chunks,
    int sample_rate,
    bool suppress_early_eos)
{
    int B = (int)chunks.size();
    if (B == 0) return {};
    if (B == 1) {
        return { transcribe(chunks[0].samples, chunks[0].num_samples,
                            sample_rate, 0.0f, 448, suppress_early_eos) };
    }

    if (!loaded_) {
        fprintf(stderr, "[ASR] ERROR: model not loaded\n");
        return std::vector<std::string>(B, "");
    }

    auto t0 = std::chrono::steady_clock::now();

    // Allocate batch GPU buffers if needed
    if (B > max_batch_allocated_) {
        if (batch_logits_) cudaFree(batch_logits_);
        if (batch_token_ids_) cudaFree(batch_token_ids_);
        if (batch_position_ids_) cudaFree(batch_position_ids_);
        if (batch_result_ids_) cudaFree(batch_result_ids_);
        cudaMalloc(&batch_logits_, (size_t)B * config_.vocab_size * sizeof(__nv_bfloat16));
        cudaMalloc(&batch_token_ids_, B * sizeof(int));
        cudaMalloc(&batch_position_ids_, 3 * B * sizeof(int));
        cudaMalloc(&batch_result_ids_, B * sizeof(int));
        max_batch_allocated_ = B;
    }

    // Initialize batch mode in decoder
    decoder_->initialize_batch(B, stream_);
    decoder_->reset_batch(B);

    // ====================================================================
    // Phase 1: Sequential mel → encode → prefill for each chunk
    // ====================================================================
    std::vector<int> prompt_lens(B);
    std::vector<int> max_gen_tokens(B);
    std::vector<int> min_gen_tokens(B);
    std::vector<int> first_tokens(B, 0);

    // NOTE: Use file-level IM_END, ENDOFTEXT, AUDIO_PAD constants — do NOT redefine here

    auto t_encode_start = std::chrono::steady_clock::now();

    for (int i = 0; i < B; i++) {
        const float* audio = chunks[i].samples;
        int audio_len = chunks[i].num_samples;

        // Resample to 16kHz if needed
        std::vector<float> audio_buf;
        const float* audio_16k = audio;
        int len_16k = audio_len;
        if (sample_rate != 16000) {
            std::vector<float> input_vec(audio, audio + audio_len);
            audio::resample(input_vec, sample_rate, audio_buf, 16000);
            audio_16k = audio_buf.data();
            len_16k = (int)audio_buf.size();
        }

        // GPU mel spectrogram
        int mel_frames = 0;
        bool used_gpu = false;
        if (gpu_whisper_mel_) {
            auto mel_result = gpu_whisper_mel_->compute(audio_16k, len_16k);
            gpu_whisper_mel_->sync();
            mel_frames = mel_result.num_frames;
            if (mel_frames > 0 && mel_result.d_mel) {
                if (mel_frames > max_mel_frames_) mel_frames = max_mel_frames_;
                audio_ops::invoke_f32_to_bf16(mel_gpu_, mel_result.d_mel,
                                               128 * mel_frames, stream_);
                used_gpu = true;
            }
        }
        if (!used_gpu) {
            audio::MelConfig mel_cfg;
            mel_cfg.n_fft = 400; mel_cfg.hop_length = 160;
            mel_cfg.n_mels = 128; mel_cfg.sample_rate = 16000;
            std::vector<float> mel_f32;
            audio::compute_mel_cached(audio_16k, len_16k, mel_cfg,
                                      cached_mel_fb_, cached_hann_window_,
                                      mel_f32, mel_frames);
            if (mel_frames > max_mel_frames_) mel_frames = max_mel_frames_;
            int mel_count = 128 * mel_frames;
            cudaMemcpyAsync(mel_staging_gpu_, mel_f32.data(),
                            mel_count * sizeof(float), cudaMemcpyHostToDevice, stream_);
            audio_ops::invoke_f32_to_bf16(mel_gpu_, mel_staging_gpu_, mel_count, stream_);
        }

        // Encode
        int encoder_out_len = 0;
        encoder_->forward(mel_gpu_, mel_frames, encoder_out_, encoder_out_len, stream_);
        if (encoder_out_len == 0) {
            fprintf(stderr, "[ASR Batch] WARNING: chunk %d encoder produced 0 tokens\n", i);
            first_tokens[i] = IM_END;  // mark as finished
            prompt_lens[i] = 0;
            continue;
        }

        // Build prompt and embed
        std::vector<int> prompt_tokens;
        build_prompt(encoder_out_len, prompt_tokens);  // default: "Chinese"
        int prompt_len = (int)prompt_tokens.size();
        if (prompt_len > max_prompt_len_) {
            fprintf(stderr, "[ASR Batch] ERROR: chunk %d prompt too long (%d)\n", i, prompt_len);
            first_tokens[i] = IM_END;
            prompt_lens[i] = 0;
            continue;
        }

        int h = config_.decoder_hidden_size;
        cudaMemcpyAsync(prompt_tokens_gpu_, prompt_tokens.data(),
                        prompt_len * sizeof(int), cudaMemcpyHostToDevice, stream_);
        audio_ops::invoke_embedding_lookup(input_embeds_, prompt_tokens_gpu_,
                                            embed_tokens_w_, prompt_len, h, stream_);

        // Replace AUDIO_PAD with encoder output
        int audio_pad_start = -1;
        for (int j = 0; j < prompt_len; j++) {
            if (prompt_tokens[j] == AUDIO_PAD) { audio_pad_start = j; break; }
        }
        if (audio_pad_start >= 0) {
            cudaMemcpyAsync(input_embeds_ + (size_t)audio_pad_start * h,
                            encoder_out_,
                            (size_t)encoder_out_len * h * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream_);
        }

        // Position IDs for MRoPE [3, prompt_len]
        std::vector<int> pos_ids(3 * prompt_len);
        for (int d = 0; d < 3; d++)
            for (int j = 0; j < prompt_len; j++)
                pos_ids[d * prompt_len + j] = j;
        cudaMemcpy(position_ids_, pos_ids.data(),
                   3 * prompt_len * sizeof(int), cudaMemcpyHostToDevice);

        // Prefill into batch KV cache
        decoder_->forward_prefill_batch_item(i, input_embeds_, position_ids_,
                                              prompt_len, logits_, stream_);
        prompt_lens[i] = prompt_len;

        // Compute generation limits
        float audio_dur_s = (float)len_16k / 16000.0f;
        max_gen_tokens[i] = std::min(512, std::max(40, (int)(audio_dur_s * 5.0f)));
        min_gen_tokens[i] = suppress_early_eos
            ? std::max(1, (int)(audio_dur_s * 0.4f)) : 0;

        // Extract first token from prefill logits
        if (min_gen_tokens[i] > 0)
            audio_ops::invoke_suppress_eos(logits_, IM_END, ENDOFTEXT, stream_);
        audio_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
        cudaMemcpyAsync(&first_tokens[i], token_id_gpu_, sizeof(int),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    auto t_encode_end = std::chrono::steady_clock::now();
    float encode_ms = std::chrono::duration<float, std::milli>(t_encode_end - t_encode_start).count();
    fprintf(stderr, "[ASR Batch] Phase 1: %d chunks encode+prefill in %.1f ms\n", B, encode_ms);
    // ====================================================================
    // Phase 2: Batch decode — all chunks generate tokens simultaneously
    // ====================================================================
    std::vector<std::vector<int>> output_tokens(B);
    std::vector<bool> finished(B, false);
    std::vector<int> positions(B);

    for (int i = 0; i < B; i++) {
        positions[i] = prompt_lens[i];
        if (first_tokens[i] == IM_END || first_tokens[i] == ENDOFTEXT || prompt_lens[i] == 0) {
            finished[i] = true;
        } else {
            output_tokens[i].push_back(first_tokens[i]);
        }
    }

    int max_steps = 512;
    std::vector<int> cur_tokens(B);
    std::vector<int> h_pos_ids(3 * B);
    std::vector<int> h_result_ids(B);

    auto t_decode_start = std::chrono::steady_clock::now();

    for (int step = 0; step < max_steps; step++) {
        // Count active sequences
        int active = 0;
        for (int i = 0; i < B; i++) if (!finished[i]) active++;
        if (active == 0) break;

        // Prepare token IDs and position IDs
        for (int i = 0; i < B; i++) {
            cur_tokens[i] = finished[i] ? 0 : (output_tokens[i].empty() ? first_tokens[i] : output_tokens[i].back());
            for (int d = 0; d < 3; d++)
                h_pos_ids[d * B + i] = positions[i];
        }
        cudaMemcpyAsync(batch_token_ids_, cur_tokens.data(),
                        B * sizeof(int), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(batch_position_ids_, h_pos_ids.data(),
                        3 * B * sizeof(int), cudaMemcpyHostToDevice, stream_);

        // Batch decode forward
        decoder_->forward_decode_batch(batch_token_ids_, batch_position_ids_,
                                        B, batch_logits_, stream_);

        // Argmax per sequence + EOS check
        for (int i = 0; i < B; i++) {
            if (finished[i]) continue;

            // Suppress EOS if below minimum
            if ((int)output_tokens[i].size() < min_gen_tokens[i]) {
                audio_ops::invoke_suppress_eos(
                    batch_logits_ + (size_t)i * config_.vocab_size,
                    IM_END, ENDOFTEXT, stream_);
            }
            audio_ops::invoke_argmax(
                batch_logits_ + (size_t)i * config_.vocab_size,
                batch_result_ids_ + i, config_.vocab_size, stream_);
        }

        // D2H transfer all results at once
        cudaMemcpyAsync(h_result_ids.data(), batch_result_ids_,
                        B * sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Update state
        decoder_->increment_batch_seq_lens(finished);
        for (int i = 0; i < B; i++) {
            if (finished[i]) continue;
            positions[i]++;

            int tok = h_result_ids[i];
            if (tok == IM_END || tok == ENDOFTEXT ||
                (int)output_tokens[i].size() >= max_gen_tokens[i]) {
                finished[i] = true;
            } else {
                output_tokens[i].push_back(tok);
            }
        }
    }

    auto t_decode_end = std::chrono::steady_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
    int total_tokens = 0;
    for (int i = 0; i < B; i++) total_tokens += (int)output_tokens[i].size();
    fprintf(stderr, "[ASR Batch] Phase 2: %d tokens in %.1f ms (%.1f tok/s, B=%d)\n",
            total_tokens, decode_ms,
            total_tokens > 0 ? total_tokens / (decode_ms / 1000.0f) : 0.0f, B);

    // ====================================================================
    // Phase 3: Decode tokens to text
    // ====================================================================
    static constexpr int ASR_TEXT_TOKEN = 151704;
    std::vector<std::string> results(B);

    for (int i = 0; i < B; i++) {
        auto& ot = output_tokens[i];
        if (ot.empty()) continue;

        // Post-process: repetition detection
        if (ot.size() >= 10) {
            for (int pl = 1; pl <= 32 && pl * 5 <= (int)ot.size(); ++pl) {
                bool found = false;
                for (int s = 0; s + pl * 5 <= (int)ot.size() && !found; ++s) {
                    int reps = 0;
                    while (s + pl * (reps + 1) <= (int)ot.size()) {
                        bool match = true;
                        for (int j = 0; j < pl; ++j) {
                            if (ot[s + j] != ot[s + pl * (reps + 1) + j]) { match = false; break; }
                        }
                        if (!match) break;
                        reps++;
                    }
                    if (reps >= 5) {
                        ot.resize(s + pl);
                        found = true;
                    }
                }
                if (found) break;
            }
        }

        // Find last <asr_text> marker and decode
        size_t text_start = 0;
        for (size_t j = 0; j < ot.size(); j++) {
            if (ot[j] == ASR_TEXT_TOKEN) text_start = j + 1;
        }
        if (text_start > 0 && text_start < ot.size() && tokenizer_.is_loaded()) {
            std::vector<int> text_tokens(ot.begin() + text_start, ot.end());
            results[i] = tokenizer_.decode(text_tokens);
        } else if (tokenizer_.is_loaded()) {
            results[i] = tokenizer_.decode(ot);
        }
        fprintf(stderr, "[ASR Batch] Chunk %d text: \"%s\"\n", i, results[i].substr(0, 100).c_str());
    }

    auto t_end = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
    fprintf(stderr, "[ASR Batch] Total: %.1f ms (encode+prefill %.0f + decode %.0f)\n",
            total_ms, encode_ms, decode_ms);

    return results;
}

} // namespace asr
} // namespace qwen_thor
