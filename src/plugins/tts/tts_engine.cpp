// tts_engine.cpp — Qwen3-TTS Engine Implementation
//
// Load weights, orchestrate Talker, provide synthesize() interface

#include "tts_engine.h"
#include "tts_ops.h"
#include "tts_tokenizer_decoder.h"
#include "../asr/audio_ops.h"
#include "engine/safetensors.h"
#include "engine/tokenizer.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace qwen_thor {
namespace tts {

TTSEngine::TTSEngine() = default;

TTSEngine::~TTSEngine() {
    for (auto ptr : device_weights_) cudaFree(ptr);
}

// ============================================================================
// Weight loading from safetensors
// ============================================================================

void TTSEngine::load_weights(const std::string& model_dir) {
    using namespace qwen_thor::io;

    std::string dir = model_dir;
    if (dir.back() != '/') dir += '/';

    // TTS model uses a single safetensors file
    SafetensorsLoader loader(dir + "model.safetensors");

    // Helper: load tensor → cudaMalloc → cudaMemcpy H2D
    auto load_tensor = [&](const std::string& name) -> __nv_bfloat16* {
        if (!loader.has_tensor(name)) {
            fprintf(stderr, "[TTS] WARNING: tensor '%s' not found\n", name.c_str());
            return nullptr;
        }
        auto tensor = loader.get_tensor(name);
        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, tensor->nbytes());
        cudaMemcpy(d_ptr, tensor->data(), tensor->nbytes(), cudaMemcpyHostToDevice);
        device_weights_.push_back(d_ptr);
        return reinterpret_cast<__nv_bfloat16*>(d_ptr);
    };

    // ======= Talker Shared Weights =======

    auto text_embed = load_tensor("talker.model.text_embedding.weight");
    auto codec_embed = load_tensor("talker.model.codec_embedding.weight");
    auto final_norm = load_tensor("talker.model.norm.weight");
    auto codec_head = load_tensor("talker.codec_head.weight");

    auto tp_fc1_w = load_tensor("talker.text_projection.linear_fc1.weight");
    auto tp_fc1_b = load_tensor("talker.text_projection.linear_fc1.bias");
    auto tp_fc2_w = load_tensor("talker.text_projection.linear_fc2.weight");
    auto tp_fc2_b = load_tensor("talker.text_projection.linear_fc2.bias");

    talker_->set_text_embedding(text_embed);
    talker_->set_codec_embedding(codec_embed);
    talker_->set_text_projection(tp_fc1_w, tp_fc1_b, tp_fc2_w, tp_fc2_b);
    talker_->set_final_norm(final_norm);
    talker_->set_codec_head(codec_head);

    // ======= Talker Layers (28) =======

    for (int i = 0; i < config_.talker.num_hidden_layers; i++) {
        std::string prefix = "talker.model.layers." + std::to_string(i) + ".";
        TalkerLayerWeights lw;
        lw.input_layernorm_w = load_tensor(prefix + "input_layernorm.weight");
        lw.q_proj_w = load_tensor(prefix + "self_attn.q_proj.weight");
        lw.k_proj_w = load_tensor(prefix + "self_attn.k_proj.weight");
        lw.v_proj_w = load_tensor(prefix + "self_attn.v_proj.weight");
        lw.o_proj_w = load_tensor(prefix + "self_attn.o_proj.weight");
        lw.q_norm_w = load_tensor(prefix + "self_attn.q_norm.weight");
        lw.k_norm_w = load_tensor(prefix + "self_attn.k_norm.weight");
        lw.post_attention_layernorm_w = load_tensor(prefix + "post_attention_layernorm.weight");
        lw.gate_proj_w = load_tensor(prefix + "mlp.gate_proj.weight");
        lw.up_proj_w = load_tensor(prefix + "mlp.up_proj.weight");
        lw.down_proj_w = load_tensor(prefix + "mlp.down_proj.weight");
        talker_->set_talker_layer_weights(i, lw);
    }

    // ======= CodePredictor Weights =======

    // Projection + final norm
    auto cp_proj_w = load_tensor("talker.code_predictor.small_to_mtp_projection.weight");
    auto cp_proj_b = load_tensor("talker.code_predictor.small_to_mtp_projection.bias");
    auto cp_norm = load_tensor("talker.code_predictor.model.norm.weight");
    talker_->set_code_predictor_projection(cp_proj_w, cp_proj_b);
    talker_->set_code_predictor_final_norm(cp_norm);

    // CodePredictor layers (5)
    for (int i = 0; i < config_.code_predictor.num_hidden_layers; i++) {
        std::string prefix = "talker.code_predictor.model.layers." + std::to_string(i) + ".";
        CodePredictorLayerWeights cw;
        cw.input_layernorm_w = load_tensor(prefix + "input_layernorm.weight");
        cw.q_proj_w = load_tensor(prefix + "self_attn.q_proj.weight");
        cw.k_proj_w = load_tensor(prefix + "self_attn.k_proj.weight");
        cw.v_proj_w = load_tensor(prefix + "self_attn.v_proj.weight");
        cw.o_proj_w = load_tensor(prefix + "self_attn.o_proj.weight");
        cw.q_norm_w = load_tensor(prefix + "self_attn.q_norm.weight");
        cw.k_norm_w = load_tensor(prefix + "self_attn.k_norm.weight");
        cw.post_attention_layernorm_w = load_tensor(prefix + "post_attention_layernorm.weight");
        cw.gate_proj_w = load_tensor(prefix + "mlp.gate_proj.weight");
        cw.up_proj_w = load_tensor(prefix + "mlp.up_proj.weight");
        cw.down_proj_w = load_tensor(prefix + "mlp.down_proj.weight");
        talker_->set_code_predictor_layer_weights(i, cw);
    }

    // CodePredictor per-group lm_heads (15) and codec_embeddings (15)
    int num_groups = config_.talker.num_code_groups - 1;  // 15
    for (int g = 0; g < num_groups; g++) {
        std::string lm_name = "talker.code_predictor.lm_head." + std::to_string(g) + ".weight";
        std::string ce_name = "talker.code_predictor.model.codec_embedding." + std::to_string(g) + ".weight";
        talker_->set_code_predictor_lm_head(g, load_tensor(lm_name));
        talker_->set_code_predictor_codec_embedding(g, load_tensor(ce_name));
    }

    fprintf(stderr, "[TTS] Loaded %zu tensors from model.safetensors\n", device_weights_.size());
}

// ============================================================================
// Build text token IDs from user text
// ============================================================================

std::vector<int> TTSEngine::build_text_tokens(const std::string& text) {
    // TTS chat template (Python reference: _build_text):
    //   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
    //
    // The Talker extracts:
    //   role = [<|im_start|>, assistant, \n]  (3 tokens)
    //   text_tokens = tokenized text
    //   closing = [<|im_end|>, \n, <|im_start|>, assistant, \n]  (5 tokens)
    //
    // We build the full sequence and pass it to build_prefill which deconstructs it.

    std::vector<int> tokens;

    // <|im_start|>assistant\n
    tokens.push_back(config_.im_start_token_id);   // 151644
    tokens.push_back(config_.assistant_token_id);   // 77091 "assistant"
    tokens.push_back(198);    // "\n"

    // Tokenize the user text
    if (tokenizer_.is_loaded()) {
        auto text_tokens = tokenizer_.encode(text);
        tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    } else {
        fprintf(stderr, "[TTS] WARNING: tokenizer not loaded, cannot encode text\n");
        return tokens;
    }

    // <|im_end|>\n
    tokens.push_back(config_.im_end_token_id);    // 151645
    tokens.push_back(198);    // "\n"

    // <|im_start|>assistant\n
    tokens.push_back(config_.im_start_token_id);   // 151644
    tokens.push_back(config_.assistant_token_id);   // 77091
    tokens.push_back(198);    // "\n"

    return tokens;
}

// ============================================================================
// load_model
// ============================================================================

void TTSEngine::load_model(const std::string& model_dir) {
    auto t0 = std::chrono::steady_clock::now();

    // 1. Load configs
    std::string dir = model_dir;
    if (dir.back() != '/') dir += '/';

    if (!config_.load_from_json(dir + "config.json")) {
        fprintf(stderr, "[TTS] ERROR: failed to load config from %s\n", model_dir.c_str());
        return;
    }
    config_.load_generation_config(dir + "generation_config.json");

    // Load speech tokenizer config
    std::string st_config = dir + "speech_tokenizer/config.json";
    config_.load_tokenizer_config(st_config);

    fprintf(stderr, "[TTS] Config: talker %dL h=%d, code_predictor %dL h=%d, "
                    "vocab=%d, groups=%d\n",
            config_.talker.num_hidden_layers, config_.talker.hidden_size,
            config_.code_predictor.num_hidden_layers, config_.code_predictor.hidden_size,
            config_.talker.vocab_size, config_.talker.num_code_groups);

    // 2. Load tokenizer (reuse main engine tokenizer BPE)
    model_dir_ = model_dir;
    if (!tokenizer_.load(model_dir)) {
        fprintf(stderr, "[TTS] WARNING: failed to load tokenizer from %s\n", model_dir.c_str());
    }

    // 3. Create Talker
    int max_seq_len = 8192;  // max generation length
    talker_ = std::make_unique<Talker>(config_, max_seq_len);

    // 4. Load weights
    load_weights(model_dir);

    // 5. Initialize Talker
    talker_->initialize(stream_);

    // 6. Load Speech Tokenizer Decoder
    std::string st_dir = dir + "speech_tokenizer/";
    st_decoder_ = std::make_unique<SpeechTokenizerDecoder>();
    if (st_decoder_->load_weights(st_dir, config_.tokenizer_decoder)) {
        st_decoder_->initialize(stream_);
        fprintf(stderr, "[TTS] Speech tokenizer decoder loaded\n");
    } else {
        fprintf(stderr, "[TTS] WARNING: speech tokenizer decoder not loaded, WAV output disabled\n");
        st_decoder_.reset();
    }

    // 7. Load Speaker Encoder (Base model only — for voice clone)
    if (config_.tts_model_type == "base") {
        load_speaker_encoder_weights(model_dir);

        // Initialize voice manager with voices directory next to model
        std::string voices_dir = dir + "voices/";
        voice_manager_.set_directory(voices_dir);
    }

    auto t1 = std::chrono::steady_clock::now();
    float load_time = std::chrono::duration<float>(t1 - t0).count();
    fprintf(stderr, "[TTS] Model loaded in %.1fs (%zu weight tensors)\n",
            load_time, device_weights_.size());
    loaded_ = true;
}

// ============================================================================
// Synthesize: text → codec tokens
// ============================================================================

std::vector<std::vector<int>> TTSEngine::synthesize(
    const std::string& text,
    const std::string& speaker,
    const std::string& language,
    int max_new_tokens)
{
    if (!loaded_) {
        fprintf(stderr, "[TTS] ERROR: model not loaded\n");
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Tokenize text
    auto text_tokens = build_text_tokens(text);
    if (text_tokens.empty()) {
        fprintf(stderr, "[TTS] ERROR: empty text tokens\n");
        return {};
    }
    fprintf(stderr, "[TTS] Text: %d tokens\n", (int)text_tokens.size());

    // 2. Build prefill embeddings
    talker_->reset();
    talker_->set_max_new_tokens(max_new_tokens);
    int prefill_len = talker_->build_prefill(
        text_tokens.data(), (int)text_tokens.size(),
        nullptr, 0,
        speaker, language, stream_);

    auto t_build = std::chrono::steady_clock::now();
    float build_ms = std::chrono::duration<float, std::milli>(t_build - t0).count();
    fprintf(stderr, "[TTS] Prefill built: %d tokens (%.1f ms)\n", prefill_len, build_ms);

    // 3. Run prefill
    talker_->forward_prefill(stream_);

    auto t_prefill = std::chrono::steady_clock::now();
    float prefill_ms = std::chrono::duration<float, std::milli>(t_prefill - t_build).count();
    fprintf(stderr, "[TTS] Prefill done: %.1f ms\n", prefill_ms);

    // 4. Autoregressive decode
    std::vector<std::vector<int>> all_codes;
    int num_groups = config_.talker.num_code_groups;
    std::vector<int> codec_step(num_groups);

    int step = 0;
    while (step < max_new_tokens) {
        int ret = talker_->forward_decode_step(codec_step.data(), stream_);
        if (ret < 0) {
            // EOS
            fprintf(stderr, "[TTS] EOS at step %d\n", step);
            break;
        }
        all_codes.push_back(codec_step);
        step++;

        // Progress report every 100 steps
        if (step % 100 == 0) {
            auto t_now = std::chrono::steady_clock::now();
            float elapsed_ms = std::chrono::duration<float, std::milli>(t_now - t_prefill).count();
            fprintf(stderr, "[TTS] Decode: %d steps (%.1f ms, %.1f steps/s)\n",
                    step, elapsed_ms, step / (elapsed_ms / 1000.0f));
        }
    }

    auto t_decode = std::chrono::steady_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(t_decode - t_prefill).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_decode - t0).count();
    fprintf(stderr, "[TTS] Decode: %d steps (%.1f ms, %.1f steps/s)\n",
            step, decode_ms, step > 0 ? step / (decode_ms / 1000.0f) : 0.0f);
    fprintf(stderr, "[TTS] Total: %.1f ms (build %.0f + prefill %.0f + decode %.0f)\n",
            total_ms, build_ms, prefill_ms, decode_ms);

    // At 12 Hz, each step = 1/12 second of audio
    float duration_s = step / 12.0f;
    fprintf(stderr, "[TTS] Generated %.1fs of audio (%d codec frames at 12 Hz)\n",
            duration_s, step);

    return all_codes;
}

// ============================================================================
// Sampling parameter overrides
// ============================================================================

void TTSEngine::set_sampling(float temperature, int top_k, float top_p, float rep_penalty) {
    if (talker_) talker_->set_sampling(temperature, top_k, top_p, rep_penalty);
}

void TTSEngine::set_sub_sampling(float temperature, int top_k, float top_p) {
    if (talker_) talker_->set_sub_sampling(temperature, top_k, top_p);
}

// ============================================================================
// Build instruct-only token IDs
// Format: <|im_start|>user\n{instruct}<|im_end|>\n
// These tokens are embedded as pure text-track in build_prefill
// ============================================================================

std::vector<int> TTSEngine::build_instruct_tokens(const std::string& instruct)
{
    std::vector<int> tokens;

    // <|im_start|>user\n
    tokens.push_back(config_.im_start_token_id);   // 151644
    auto user_tokens = tokenizer_.encode("user");
    tokens.insert(tokens.end(), user_tokens.begin(), user_tokens.end());
    tokens.push_back(198);    // "\n"

    // instruct text
    auto instruct_toks = tokenizer_.encode(instruct);
    tokens.insert(tokens.end(), instruct_toks.begin(), instruct_toks.end());

    // <|im_end|>\n
    tokens.push_back(config_.im_end_token_id);     // 151645
    tokens.push_back(198);    // "\n"

    return tokens;
}

// ============================================================================
// Synthesize to WAV (end-to-end)
// ============================================================================

std::vector<float> TTSEngine::synthesize_to_pcm(
    const std::string& text,
    const std::string& speaker,
    const std::string& language,
    const std::string& instruct,
    int max_new_tokens)
{
    if (!loaded_) {
        fprintf(stderr, "[TTS] ERROR: model not loaded\n");
        return {};
    }

    if (!st_decoder_ || !st_decoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: speech tokenizer decoder not loaded\n");
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Tokenize text (always use build_text_tokens for the spoken text)
    std::vector<int> text_tokens = build_text_tokens(text);

    // Build instruct tokens separately (for CustomVoice emotion or VoiceDesign style)
    std::vector<int> instruct_tokens;
    if (!instruct.empty() || config_.tts_model_type == "voice_design") {
        instruct_tokens = build_instruct_tokens(instruct);
    }

    if (text_tokens.empty()) {
        fprintf(stderr, "[TTS] ERROR: empty text tokens\n");
        return {};
    }

    // 2. Build prefill + decode → codec tokens
    talker_->reset();
    talker_->set_max_new_tokens(max_new_tokens);
    talker_->build_prefill(text_tokens.data(), (int)text_tokens.size(),
                           instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                           (int)instruct_tokens.size(),
                           speaker, language, stream_);
    talker_->forward_prefill(stream_);

    int num_groups = config_.talker.num_code_groups;
    std::vector<std::vector<int>> all_codes;
    std::vector<int> codec_step(num_groups);
    int step = 0;
    while (step < max_new_tokens) {
        int ret = talker_->forward_decode_step(codec_step.data(), stream_);
        if (ret < 0) break;
        all_codes.push_back(codec_step);
        step++;
    }

    auto t_talker = std::chrono::steady_clock::now();
    float talker_ms = std::chrono::duration<float, std::milli>(t_talker - t0).count();
    fprintf(stderr, "[TTS] Talker done: %d steps (%.1f ms, %.1f steps/s)\n",
            step, talker_ms, step > 0 ? step / (talker_ms / 1000.0f) : 0.0f);

    if (all_codes.empty()) {
        fprintf(stderr, "[TTS] ERROR: no codec tokens generated\n");
        return {};
    }

    // 3. Reshape codes to [num_groups, num_frames]
    int T = (int)all_codes.size();
    std::vector<int> codes_flat(num_groups * T);
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < num_groups; g++) {
            codes_flat[g * T + t] = all_codes[t][g];
        }
    }

    // 4. Decode codec tokens → PCM
    fprintf(stderr, "[TTS] Decoding %d frames → PCM...\n", T);
    auto pcm = st_decoder_->decode(codes_flat.data(), num_groups, T, stream_);

    auto t_decode = std::chrono::steady_clock::now();
    float decode_ms = std::chrono::duration<float, std::milli>(t_decode - t_talker).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_decode - t0).count();
    fprintf(stderr, "[TTS] Decoder done: %zu samples (%.1f ms, total %.1f ms)\n",
            pcm.size(), decode_ms, total_ms);

    return pcm;
}

// ============================================================================
// Streaming synthesis: generate codec tokens and decode in chunks
// First audio chunk arrives after chunk_frames * step_time + decode_time
// instead of waiting for full generation + decode
// ============================================================================

int TTSEngine::synthesize_streaming(
    const std::string& text,
    const std::string& speaker,
    const std::string& language,
    const std::string& instruct,
    int max_new_tokens,
    int chunk_frames,
    PcmCallback pcm_callback)
{
    if (!loaded_ || !st_decoder_ || !st_decoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: model not loaded\n");
        return 0;
    }
    if (!pcm_callback) return 0;

    auto t0 = std::chrono::steady_clock::now();

    // 1. Tokenize text (always use build_text_tokens for the spoken text)
    std::vector<int> text_tokens = build_text_tokens(text);

    // Build instruct tokens separately (for CustomVoice emotion or VoiceDesign style)
    std::vector<int> instruct_tokens;
    if (!instruct.empty() || config_.tts_model_type == "voice_design") {
        instruct_tokens = build_instruct_tokens(instruct);
    }
    if (text_tokens.empty()) return 0;

    // 2. Prefill
    talker_->reset();
    talker_->set_max_new_tokens(max_new_tokens);
    talker_->build_prefill(text_tokens.data(), (int)text_tokens.size(),
                           instruct_tokens.empty() ? nullptr : instruct_tokens.data(),
                           (int)instruct_tokens.size(),
                           speaker, language, stream_);
    talker_->forward_prefill(stream_);

    int num_groups = config_.talker.num_code_groups;
    std::vector<std::vector<int>> chunk_codes;
    std::vector<int> codec_step(num_groups);
    int total_steps = 0;
    int total_pcm_samples = 0;
    bool aborted = false;

    // Left context for decoder continuity (eliminates chunk boundary artifacts)
    const int left_ctx_frames = config_.tokenizer_decoder.left_context_size; // 25
    const int upsample_rate = config_.tokenizer_decoder.decode_upsample_rate; // 1920
    code_history_.clear();  // Reset for new utterance

    // 3. Generate + decode in chunks with left context
    auto decode_and_send = [&]() {
        if (chunk_codes.empty() || aborted) return;
        int T_new = (int)chunk_codes.size();
        int ctx = std::min(left_ctx_frames, (int)code_history_.size());
        int T_total = ctx + T_new;

        std::vector<int> codes_flat(num_groups * T_total);
        // Fill left context from history
        int hist_start = (int)code_history_.size() - ctx;
        for (int t = 0; t < ctx; t++) {
            for (int g = 0; g < num_groups; g++) {
                codes_flat[g * T_total + t] = code_history_[hist_start + t][g];
            }
        }
        // Fill new codes
        for (int t = 0; t < T_new; t++) {
            for (int g = 0; g < num_groups; g++) {
                codes_flat[g * T_total + (ctx + t)] = chunk_codes[t][g];
            }
        }

        auto pcm = st_decoder_->decode(codes_flat.data(), num_groups, T_total, stream_);
        if (!pcm.empty()) {
            // Skip left context PCM samples
            int ctx_samples = ctx * upsample_rate;
            int valid = (int)pcm.size() - ctx_samples;
            if (valid > 0) {
                total_pcm_samples += valid;
                if (!pcm_callback(pcm.data() + ctx_samples, valid)) {
                    aborted = true;
                }
            }
        }

        // Append new codes to history, keep only last left_ctx_frames
        for (auto& c : chunk_codes) code_history_.push_back(c);
        if ((int)code_history_.size() > left_ctx_frames) {
            code_history_.erase(code_history_.begin(),
                code_history_.begin() + ((int)code_history_.size() - left_ctx_frames));
        }
    };

    while (total_steps < max_new_tokens && !aborted) {
        chunk_codes.clear();

        // Generate chunk_frames codec frames
        bool eos = false;
        for (int f = 0; f < chunk_frames && total_steps < max_new_tokens; f++) {
            int ret = talker_->forward_decode_step(codec_step.data(), stream_);
            if (ret < 0) { eos = true; break; }
            chunk_codes.push_back(codec_step);
            total_steps++;
        }

        // Decode accumulated frames (including partial chunk on EOS)
        decode_and_send();

        if (eos) break;
    }

    auto t_end = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
    float audio_s = total_pcm_samples / (float)config_.tokenizer_decoder.output_sample_rate;
    float rtf = audio_s > 0.0f ? audio_s / (total_ms / 1000.0f) : 0.0f;
    fprintf(stderr, "[TTS] Streaming done: %d steps, %d samples (%.1fs audio), %.1f ms total (%.1fx realtime)\n",
            total_steps, total_pcm_samples, audio_s, total_ms, rtf);

    return total_pcm_samples;
}

// ============================================================================
// Continue synthesis (preserve talker KV cache for voice consistency)
// ============================================================================

std::vector<float> TTSEngine::continue_to_pcm(
    const std::string& text,
    int max_new_tokens)
{
    if (!loaded_ || !st_decoder_ || !st_decoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: model not loaded for continuation\n");
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();

    // Tokenize new text (same format as normal synthesis)
    auto text_tokens = build_text_tokens(text);
    if (text_tokens.empty()) {
        fprintf(stderr, "[TTS] ERROR: empty continuation text tokens\n");
        return {};
    }

    // Inject new text without resetting KV cache
    talker_->inject_continuation_text(text_tokens.data(), (int)text_tokens.size(), stream_);
    talker_->set_max_new_tokens(max_new_tokens);

    // Continue decoding from current talker state
    int num_groups = config_.talker.num_code_groups;
    std::vector<std::vector<int>> all_codes;
    std::vector<int> codec_step(num_groups);
    int step = 0;
    while (step < max_new_tokens) {
        int ret = talker_->forward_decode_step(codec_step.data(), stream_);
        if (ret < 0) break;
        all_codes.push_back(codec_step);
        step++;
    }

    auto t_talker = std::chrono::steady_clock::now();
    float talker_ms = std::chrono::duration<float, std::milli>(t_talker - t0).count();
    fprintf(stderr, "[TTS] Continue talker: %d steps (%.1f ms)\n", step, talker_ms);

    if (all_codes.empty()) return {};

    // Reshape and decode to PCM
    int T = (int)all_codes.size();
    std::vector<int> codes_flat(num_groups * T);
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < num_groups; g++) {
            codes_flat[g * T + t] = all_codes[t][g];
        }
    }

    auto pcm = st_decoder_->decode(codes_flat.data(), num_groups, T, stream_);

    auto t_decode = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_decode - t0).count();
    fprintf(stderr, "[TTS] Continue done: %zu samples (%.1f ms total)\n", pcm.size(), total_ms);

    return pcm;
}

// ============================================================================
// Continue streaming: inject new text, preserve KV cache, decode in chunks
// ============================================================================

int TTSEngine::continue_streaming(
    const std::string& text,
    int max_new_tokens,
    int chunk_frames,
    PcmCallback pcm_callback)
{
    if (!loaded_ || !st_decoder_ || !st_decoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: model not loaded for continue_streaming\n");
        return 0;
    }
    if (!pcm_callback) return 0;

    auto t0 = std::chrono::steady_clock::now();

    auto text_tokens = build_text_tokens(text);
    if (text_tokens.empty()) return 0;

    talker_->inject_continuation_text(text_tokens.data(), (int)text_tokens.size(), stream_);
    talker_->set_max_new_tokens(max_new_tokens);

    int num_groups = config_.talker.num_code_groups;
    const int left_ctx_frames = config_.tokenizer_decoder.left_context_size;
    const int upsample_rate = config_.tokenizer_decoder.decode_upsample_rate;

    std::vector<std::vector<int>> chunk_codes;
    // code_history_ persists from synthesize_streaming (provides cross-sentence left context)
    std::vector<int> codec_step(num_groups);
    int total_steps = 0;
    int total_pcm_samples = 0;
    bool aborted = false;

    auto decode_and_send = [&]() {
        if (chunk_codes.empty() || aborted) return;
        int T_new = (int)chunk_codes.size();
        int ctx = std::min(left_ctx_frames, (int)code_history_.size());
        int T_total = ctx + T_new;

        std::vector<int> codes_flat(num_groups * T_total);
        int hist_start = (int)code_history_.size() - ctx;
        for (int t = 0; t < ctx; t++)
            for (int g = 0; g < num_groups; g++)
                codes_flat[g * T_total + t] = code_history_[hist_start + t][g];
        for (int t = 0; t < T_new; t++)
            for (int g = 0; g < num_groups; g++)
                codes_flat[g * T_total + (ctx + t)] = chunk_codes[t][g];

        auto pcm = st_decoder_->decode(codes_flat.data(), num_groups, T_total, stream_);
        if (!pcm.empty()) {
            int ctx_samples = ctx * upsample_rate;
            int valid = (int)pcm.size() - ctx_samples;
            if (valid > 0) {
                total_pcm_samples += valid;
                if (!pcm_callback(pcm.data() + ctx_samples, valid)) aborted = true;
            }
        }

        for (auto& c : chunk_codes) code_history_.push_back(c);
        if ((int)code_history_.size() > left_ctx_frames)
            code_history_.erase(code_history_.begin(),
                code_history_.begin() + ((int)code_history_.size() - left_ctx_frames));
    };

    while (total_steps < max_new_tokens && !aborted) {
        chunk_codes.clear();
        bool eos = false;
        for (int f = 0; f < chunk_frames && total_steps < max_new_tokens; f++) {
            int ret = talker_->forward_decode_step(codec_step.data(), stream_);
            if (ret < 0) { eos = true; break; }
            chunk_codes.push_back(codec_step);
            total_steps++;
        }
        decode_and_send();
        if (eos) break;
    }

    auto t_end = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
    float audio_s = total_pcm_samples / (float)config_.tokenizer_decoder.output_sample_rate;
    float rtf = audio_s > 0.0f ? audio_s / (total_ms / 1000.0f) : 0.0f;
    fprintf(stderr, "[TTS] Continue streaming done: %d steps, %d samples (%.1fs audio), %.1f ms total (%.1fx realtime)\n",
            total_steps, total_pcm_samples, audio_s, total_ms, rtf);

    return total_pcm_samples;
}

// ============================================================================
// Synthesize to WAV (end-to-end)
// ============================================================================

bool TTSEngine::synthesize_to_wav(
    const std::string& text,
    const std::string& output_path,
    const std::string& speaker,
    const std::string& language,
    const std::string& instruct,
    int max_new_tokens)
{
    auto pcm = synthesize_to_pcm(text, speaker, language, instruct, max_new_tokens);
    if (pcm.empty()) return false;

    bool ok = SpeechTokenizerDecoder::write_wav(
        output_path, pcm, config_.tokenizer_decoder.output_sample_rate);

    if (ok) {
        float duration_s = (float)pcm.size() / config_.tokenizer_decoder.output_sample_rate;
        fprintf(stderr, "[TTS] WAV written: %s (%.1fs audio)\n",
                output_path.c_str(), duration_s);
    }
    return ok;
}

// ============================================================================
// Speaker Encoder: load weights from model.safetensors
// ============================================================================

void TTSEngine::load_speaker_encoder_weights(const std::string& model_dir) {
    using namespace qwen_thor::io;

    std::string dir = model_dir;
    if (dir.back() != '/') dir += '/';

    SafetensorsLoader loader(dir + "model.safetensors");

    // Collect all speaker_encoder.* tensors as (name, (bf16_ptr, num_elements))
    std::vector<std::pair<std::string, std::pair<const uint16_t*, size_t>>> se_weights;

    for (const auto& name : loader.get_tensor_names()) {
        if (name.rfind("speaker_encoder.", 0) == 0) {
            auto tensor = loader.get_tensor(name);
            if (tensor) {
                const uint16_t* data = reinterpret_cast<const uint16_t*>(tensor->data());
                size_t num_elements = tensor->nbytes() / 2;  // BF16 = 2 bytes
                se_weights.push_back({name, {data, num_elements}});
            }
        }
    }

    if (se_weights.empty()) {
        fprintf(stderr, "[TTS] No speaker_encoder weights found (not a Base model)\n");
        return;
    }

    speaker_encoder_ = std::make_unique<SpeakerEncoder>();
    speaker_encoder_->set_config(config_.speaker_encoder);
    if (!speaker_encoder_->load_weights(se_weights)) {
        fprintf(stderr, "[TTS] ERROR: failed to load speaker encoder weights\n");
        speaker_encoder_.reset();
    }
}

// ============================================================================
// Voice Clone: extract speaker embedding
// ============================================================================

std::vector<float> TTSEngine::extract_speaker_embedding(
    const float* audio, int num_samples, int sample_rate)
{
    if (!speaker_encoder_ || !speaker_encoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: speaker encoder not loaded\n");
        return {};
    }
    return speaker_encoder_->extract(audio, num_samples, sample_rate);
}

// ============================================================================
// Voice Clone: register/delete/list voices
// ============================================================================

bool TTSEngine::register_voice(const std::string& name,
                                const float* audio, int num_samples, int sample_rate)
{
    auto embedding = extract_speaker_embedding(audio, num_samples, sample_rate);
    if (embedding.empty()) return false;
    return voice_manager_.register_voice(name, embedding);
}

bool TTSEngine::register_voice_embedding(const std::string& name,
                                          const std::vector<float>& embedding)
{
    return voice_manager_.register_voice(name, embedding);
}

bool TTSEngine::delete_voice(const std::string& name)
{
    return voice_manager_.delete_voice(name);
}

std::vector<std::string> TTSEngine::list_clone_voices() const
{
    return voice_manager_.list_voices();
}

bool TTSEngine::has_clone_voice(const std::string& name) const
{
    return voice_manager_.has_voice(name);
}

// ============================================================================
// Voice Clone: synthesize using registered voice name
// ============================================================================

std::vector<float> TTSEngine::synthesize_voice_clone(
    const std::string& text,
    const std::string& voice_name,
    const std::string& language,
    int max_new_tokens)
{
    const auto* emb = voice_manager_.get_embedding(voice_name);
    if (!emb) {
        fprintf(stderr, "[TTS] ERROR: clone voice '%s' not found\n", voice_name.c_str());
        return {};
    }
    return synthesize_voice_clone_embedding(text, *emb, language, max_new_tokens);
}

// ============================================================================
// Voice Clone: synthesize using raw embedding vector
// ============================================================================

std::vector<float> TTSEngine::synthesize_voice_clone_embedding(
    const std::string& text,
    const std::vector<float>& speaker_embedding,
    const std::string& language,
    int max_new_tokens)
{
    if (!loaded_) {
        fprintf(stderr, "[TTS] ERROR: model not loaded\n");
        return {};
    }
    if (!st_decoder_ || !st_decoder_->is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: speech tokenizer decoder not loaded\n");
        return {};
    }
    if (speaker_embedding.empty()) {
        fprintf(stderr, "[TTS] ERROR: empty speaker embedding\n");
        return {};
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Tokenize text
    std::vector<int> text_tokens = build_text_tokens(text);
    if (text_tokens.empty()) {
        fprintf(stderr, "[TTS] ERROR: empty text tokens\n");
        return {};
    }

    // 2. Build prefill with external speaker embedding (voice clone mode)
    talker_->reset();
    talker_->set_max_new_tokens(max_new_tokens);
    talker_->build_prefill_clone(text_tokens.data(), (int)text_tokens.size(),
                                 speaker_embedding.data(), (int)speaker_embedding.size(),
                                 language, stream_);
    talker_->forward_prefill(stream_);

    // 3. Decode
    int num_groups = config_.talker.num_code_groups;
    std::vector<std::vector<int>> all_codes;
    std::vector<int> codec_step(num_groups);
    int step = 0;
    while (step < max_new_tokens) {
        int ret = talker_->forward_decode_step(codec_step.data(), stream_);
        if (ret < 0) break;
        all_codes.push_back(codec_step);
        step++;
    }

    auto t_talker = std::chrono::steady_clock::now();
    float talker_ms = std::chrono::duration<float, std::milli>(t_talker - t0).count();
    fprintf(stderr, "[TTS] Voice clone talker: %d steps (%.1f ms, %.1f steps/s)\n",
            step, talker_ms, step > 0 ? step / (talker_ms / 1000.0f) : 0.0f);

    if (all_codes.empty()) return {};

    // 4. Decode to PCM
    int T = (int)all_codes.size();
    std::vector<int> codes_flat(num_groups * T);
    for (int t = 0; t < T; t++) {
        for (int g = 0; g < num_groups; g++) {
            codes_flat[g * T + t] = all_codes[t][g];
        }
    }

    auto pcm = st_decoder_->decode(codes_flat.data(), num_groups, T, stream_);

    auto t_decode = std::chrono::steady_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t_decode - t0).count();
    fprintf(stderr, "[TTS] Voice clone done: %zu samples (%.1f ms total)\n",
            pcm.size(), total_ms);

    return pcm;
}

} // namespace tts
} // namespace qwen_thor
