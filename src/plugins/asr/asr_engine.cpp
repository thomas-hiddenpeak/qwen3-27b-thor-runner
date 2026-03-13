// asr_engine.cpp — Qwen3-ASR 推理引擎实现
//
// 加载权重 + 编排 encoder/decoder + transcribe 接口

#include "asr_engine.h"
#include "audio_utils.h"
#include "audio_ops.h"
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
}

// ============================================================================
// Greedy argmax on GPU — single CTA reduction, result in managed memory
// Replaces CPU version that copied 496KB per step
// ============================================================================

// (greedy_decode removed — using audio_ops::invoke_argmax + token_id_gpu_)

// ============================================================================
// Build prompt token IDs
// ============================================================================

void ASREngine::build_prompt(int encoder_out_len, std::vector<int>& token_ids) {
    token_ids.clear();

    // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    token_ids.push_back(IM_START);
    // "system\n" → tokenize
    // Hardcode known IDs for common tokens (from Qwen3 vocab):
    // "system" = [8948], "\n" = [198]
    token_ids.push_back(8948);   // system
    token_ids.push_back(198);    // \n
    // "You are a helpful assistant."
    // Hardcode: [2610, 525, 264, 10950, 17847, 13]
    token_ids.push_back(2610);   // You
    token_ids.push_back(525);    // are
    token_ids.push_back(264);    // a
    token_ids.push_back(10950);  // helpful
    token_ids.push_back(17847);  // assistant
    token_ids.push_back(13);     // .
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

    // 6. Allocate GPU buffers
    // Max mel frames for ~120s audio at 16kHz: 120 * 100 + margin
    max_mel_frames_ = 12000;
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

    // Token ID for decode step (managed memory, GPU argmax writes here)
    cudaMallocManaged(&token_id_gpu_, sizeof(int));

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
    cached_mel_fb_ = audio::build_mel_filterbank(cached_n_mels_, cached_n_fft_, cached_sample_rate_);
    cached_hann_window_ = audio::build_hann_window(cached_n_fft_);
    fprintf(stderr, "[ASR] Mel filterbank cached: %d mels, %d FFT, %d Hz\n",
            cached_n_mels_, cached_n_fft_, cached_sample_rate_);
}

// ============================================================================
// transcribe: PCM float → text
// ============================================================================

std::string ASREngine::transcribe(
    const float* samples, int num_samples, int sample_rate,
    float temperature, int max_new_tokens)
{
    if (!loaded_) {
        fprintf(stderr, "[ASR] ERROR: model not loaded\n");
        return "";
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

    // 2. Compute mel spectrogram (CPU, cached filterbank)
    audio::MelConfig mel_cfg;
    mel_cfg.n_fft = 400;
    mel_cfg.hop_length = 160;
    mel_cfg.n_mels = 128;
    mel_cfg.sample_rate = 16000;

    int mel_frames = 0;
    std::vector<float> mel_f32;
    audio::compute_mel_cached(audio_16k, audio_len, mel_cfg,
                              cached_mel_fb_, cached_hann_window_,
                              mel_f32, mel_frames);

    if (mel_frames > max_mel_frames_) {
        fprintf(stderr, "[ASR] WARNING: audio too long (%d frames > %d max), truncating\n",
                mel_frames, max_mel_frames_);
        mel_frames = max_mel_frames_;
    }

    // 3. Convert mel to BF16 on GPU (avoids CPU loop + H2D copy of BF16 data)
    {
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
    build_prompt(encoder_out_len, prompt_tokens);
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
    std::vector<int> output_tokens;
    audio_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
    cudaStreamSynchronize(stream_);
    int next_token = *token_id_gpu_;
    int current_pos = prompt_len;

    while (next_token != IM_END && next_token != ENDOFTEXT
           && (int)output_tokens.size() < max_new_tokens
           && current_pos < max_prompt_len_ + max_new_tokens) {
        output_tokens.push_back(next_token);

        // Position IDs for decode step: [3, 1] all same position
        int step_pos[3] = {current_pos, current_pos, current_pos};
        cudaMemcpyAsync(position_ids_, step_pos, 3 * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);

        decoder_->forward_decode(next_token, position_ids_, logits_, stream_);
        audio_ops::invoke_argmax(logits_, token_id_gpu_, config_.vocab_size, stream_);
        cudaStreamSynchronize(stream_);
        next_token = *token_id_gpu_;
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

    // 10. Decode tokens to text — skip "language XXX <asr_text>" header
    std::string result;
    if (!output_tokens.empty()) {
        // Debug: print token IDs
        fprintf(stderr, "[ASR] Token IDs (%zu):", output_tokens.size());
        for (auto id : output_tokens) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");

        // Find <asr_text> marker (token 151704) and only decode tokens after it
        static constexpr int ASR_TEXT_TOKEN = 151704;
        size_t text_start = 0;
        for (size_t i = 0; i < output_tokens.size(); i++) {
            if (output_tokens[i] == ASR_TEXT_TOKEN) {
                text_start = i + 1;
                break;
            }
        }

        if (text_start < output_tokens.size() && tokenizer_.is_loaded()) {
            std::vector<int> text_tokens(output_tokens.begin() + text_start,
                                          output_tokens.end());
            result = tokenizer_.decode(text_tokens);
        } else if (tokenizer_.is_loaded()) {
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
    float temperature, int max_new_tokens)
{
    // Load WAV file
    audio::AudioData wav;
    if (!audio::load_wav(wav_path, wav)) {
        fprintf(stderr, "[ASR] ERROR: failed to load WAV file: %s\n", wav_path.c_str());
        return "";
    }

    return transcribe(wav.samples.data(), (int)wav.samples.size(), wav.sample_rate,
                      temperature, max_new_tokens);
}

} // namespace asr
} // namespace qwen_thor
