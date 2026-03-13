// tts_talker.cu — Qwen3-TTS Talker + CodePredictor CUDA implementation
//
// Talker: 28L GQA transformer (16Q/8KV, head_dim=128, SwiGLU)
//   - Dual-track embedding: text_projection(text_embedding) + codec_embedding
//   - MRoPE (sections [24,20,20]) — degenerates to 1D for TTS
//   - Per-head Q/K RMSNorm
//   - codec_head for group-0 logits
//
// CodePredictor: 5L GQA transformer (16Q/8KV, head_dim=128)
//   - Standard 1D RoPE, per-head Q/K RMSNorm
//   - Generates codec groups 1-15

#include "tts_talker.h"
#include "tts_ops.h"
#include "../asr/audio_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>

namespace qwen_thor {
namespace tts {

// ============================================================================
// cuBLAS helpers
// ============================================================================

static void cublas_gemm(
    cublasHandle_t handle,
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int M, int K, int N,
    cudaStream_t stream)
{
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 weight, CUDA_R_16BF, K,
                 input, CUDA_R_16BF, K,
                 &beta,
                 out, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ============================================================================
// Talker implementation
// ============================================================================

Talker::Talker(const TTSConfig& config, int max_seq_len)
    : config_(config)
    , max_seq_len_(max_seq_len)
    , talker_layer_weights_(config.talker.num_hidden_layers)
    , talker_k_cache_(config.talker.num_hidden_layers, nullptr)
    , talker_v_cache_(config.talker.num_hidden_layers, nullptr)
    , cp_layer_weights_(config.code_predictor.num_hidden_layers)
    , cp_k_cache_(config.code_predictor.num_hidden_layers, nullptr)
    , cp_v_cache_(config.code_predictor.num_hidden_layers, nullptr)
    , cp_lm_heads_(config.talker.num_code_groups - 1, nullptr)
    , cp_codec_embeddings_(config.talker.num_code_groups - 1, nullptr)
{
    temperature_ = config.temperature;
    top_k_ = config.top_k;
    top_p_ = config.top_p;
    rep_penalty_ = config.repetition_penalty;
    sub_temperature_ = config.sub_temperature;
    sub_top_k_ = config.sub_top_k;
    sub_top_p_ = config.sub_top_p;
}

Talker::~Talker() {
    for (auto p : talker_k_cache_) if (p) cudaFree(p);
    for (auto p : talker_v_cache_) if (p) cudaFree(p);
    for (auto p : cp_k_cache_) if (p) cudaFree(p);
    for (auto p : cp_v_cache_) if (p) cudaFree(p);
    for (auto p : merged_weight_allocs_) if (p) cudaFree(p);
    if (workspace_) cudaFree(workspace_);
    if (prefill_embeds_) cudaFree(prefill_embeds_);
    if (position_ids_) cudaFree(position_ids_);
    if (logits_) cudaFree(logits_);
    if (cp_logits_) cudaFree(cp_logits_);
    if (trailing_text_hidden_) cudaFree(trailing_text_hidden_);
    if (tts_pad_embed_) cudaFree(tts_pad_embed_);
    if (past_hidden_) cudaFree(past_hidden_);
    if (sampled_token_) cudaFree(sampled_token_);
    // Decode pre-allocated buffers
    if (decode_all_embeds_) cudaFree(decode_all_embeds_);
    if (decode_input_embeds_) cudaFree(decode_input_embeds_);
    if (decode_token_gpu_) cudaFree(decode_token_gpu_);
    if (decode_pos_gpu_) cudaFree(decode_pos_gpu_);
    if (rep_ids_gpu_) cudaFree(rep_ids_gpu_);
    if (codec_out_gpu_) cudaFree(codec_out_gpu_);
    if (cp_input_buf_) cudaFree(cp_input_buf_);
    if (cp_hidden_buf_) cudaFree(cp_hidden_buf_);
    if (cp_embed_buf_) cudaFree(cp_embed_buf_);
    if (cp_decode_hidden_) cudaFree(cp_decode_hidden_);
    if (cp_pos_gpu_) cudaFree(cp_pos_gpu_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

// ===== Weight Binding =====

void Talker::set_text_embedding(__nv_bfloat16* w) { text_embedding_w_ = w; }
void Talker::set_codec_embedding(__nv_bfloat16* w) { codec_embedding_w_ = w; }
void Talker::set_text_projection(__nv_bfloat16* fc1_w, __nv_bfloat16* fc1_b,
                                  __nv_bfloat16* fc2_w, __nv_bfloat16* fc2_b) {
    text_proj_fc1_w_ = fc1_w; text_proj_fc1_b_ = fc1_b;
    text_proj_fc2_w_ = fc2_w; text_proj_fc2_b_ = fc2_b;
}
void Talker::set_final_norm(__nv_bfloat16* w) { final_norm_w_ = w; }
void Talker::set_codec_head(__nv_bfloat16* w) { codec_head_w_ = w; }
void Talker::set_talker_layer_weights(int i, const TalkerLayerWeights& w) {
    talker_layer_weights_[i] = w;
}
void Talker::set_code_predictor_projection(__nv_bfloat16* w, __nv_bfloat16* b) {
    cp_projection_w_ = w; cp_projection_b_ = b;
}
void Talker::set_code_predictor_final_norm(__nv_bfloat16* w) { cp_final_norm_w_ = w; }
void Talker::set_code_predictor_layer_weights(int i, const CodePredictorLayerWeights& w) {
    cp_layer_weights_[i] = w;
}
void Talker::set_code_predictor_lm_head(int i, __nv_bfloat16* w) { cp_lm_heads_[i] = w; }
void Talker::set_code_predictor_codec_embedding(int i, __nv_bfloat16* w) {
    cp_codec_embeddings_[i] = w;
}

void Talker::set_sampling(float temp, int topk, float topp, float rep_pen) {
    temperature_ = temp; top_k_ = topk; top_p_ = topp; rep_penalty_ = rep_pen;
}
void Talker::set_sub_sampling(float temp, int topk, float topp) {
    sub_temperature_ = temp; sub_top_k_ = topk; sub_top_p_ = topp;
}

// ============================================================================
// Initialize: allocate KV caches and workspace
// ============================================================================

void Talker::initialize(cudaStream_t stream) {
    if (initialized_) return;

    cublasCreate(&cublas_handle_);

    const auto& tc = config_.talker;
    const auto& cp = config_.code_predictor;
    int h = tc.hidden_size;
    int kv_dim = tc.num_kv_heads * tc.head_dim;
    int q_dim = tc.num_attention_heads * tc.head_dim;
    int ffn = tc.intermediate_size;
    int num_talker_layers = tc.num_hidden_layers;
    int num_cp_layers = cp.num_hidden_layers;

    // Talker KV cache: [max_seq_len, kv_heads, head_dim] per layer
    size_t talker_kv = (size_t)max_seq_len_ * kv_dim;
    for (int i = 0; i < num_talker_layers; i++) {
        cudaMalloc(&talker_k_cache_[i], talker_kv * sizeof(__nv_bfloat16));
        cudaMalloc(&talker_v_cache_[i], talker_kv * sizeof(__nv_bfloat16));
    }

    // CodePredictor KV cache: max length per talker step = 2 (prefill) + 14 (decode) = 16
    int cp_max_len = tc.num_code_groups;  // 16
    int cp_kv_dim = cp.num_kv_heads * cp.head_dim;
    size_t cp_kv = (size_t)cp_max_len * cp_kv_dim;
    for (int i = 0; i < num_cp_layers; i++) {
        cudaMalloc(&cp_k_cache_[i], cp_kv * sizeof(__nv_bfloat16));
        cudaMalloc(&cp_v_cache_[i], cp_kv * sizeof(__nv_bfloat16));
    }

    // Workspace: enough for the largest layer (talker prefill)
    // For prefill T up to max_seq_len_:
    //   norm_buf: T*h, q_buf: T*q_dim, k_buf: T*kv_dim, v_buf: T*kv_dim,
    //   attn_out: T*h, gate_buf: T*ffn, up_buf: T*ffn
    workspace_size_ = (size_t)max_seq_len_ * h
                    + (size_t)max_seq_len_ * q_dim
                    + (size_t)max_seq_len_ * kv_dim
                    + (size_t)max_seq_len_ * kv_dim
                    + (size_t)max_seq_len_ * h
                    + (size_t)max_seq_len_ * ffn
                    + (size_t)max_seq_len_ * ffn
                    + 4096;
    cudaMalloc(&workspace_, workspace_size_ * sizeof(__nv_bfloat16));

    // Prefill embeddings buffer
    cudaMalloc(&prefill_embeds_, (size_t)max_seq_len_ * h * sizeof(__nv_bfloat16));

    // Position IDs [3, max_seq_len]
    cudaMalloc(&position_ids_, 3 * max_seq_len_ * sizeof(int));

    // Logits buffers
    cudaMalloc(&logits_, tc.vocab_size * sizeof(__nv_bfloat16));
    cudaMalloc(&cp_logits_, cp.vocab_size * sizeof(__nv_bfloat16));

    // Trailing text hidden
    // Max text length = max_seq_len_ (generous)
    cudaMalloc(&trailing_text_hidden_, (size_t)max_seq_len_ * h * sizeof(__nv_bfloat16));

    // TTS pad embed [hidden_size]
    cudaMalloc(&tts_pad_embed_, h * sizeof(__nv_bfloat16));

    // Past hidden [1, hidden_size]
    cudaMalloc(&past_hidden_, h * sizeof(__nv_bfloat16));

    // Sampled token (managed memory for GPU→CPU)
    cudaMallocManaged(&sampled_token_, sizeof(int));

    // ===== Pre-allocated Decode Buffers =====
    int num_groups = tc.num_code_groups;
    cudaMalloc(&decode_all_embeds_, (size_t)num_groups * h * sizeof(__nv_bfloat16));
    cudaMalloc(&decode_input_embeds_, h * sizeof(__nv_bfloat16));
    cudaMalloc(&decode_token_gpu_, sizeof(int));
    cudaMalloc(&decode_pos_gpu_, 3 * sizeof(int));
    cudaMalloc(&rep_ids_gpu_, max_seq_len_ * sizeof(int));  // max tokens
    cudaMalloc(&codec_out_gpu_, num_groups * sizeof(int));  // GPU-resident codec tokens
    // CodePredictor
    int talker_h = tc.hidden_size;
    int cp_h_sz = cp.hidden_size;
    int cp_max_seq = num_groups;  // max CP context length = 16
    cudaMalloc(&cp_input_buf_, 2 * talker_h * sizeof(__nv_bfloat16));
    cudaMalloc(&cp_hidden_buf_, 2 * cp_h_sz * sizeof(__nv_bfloat16));
    cudaMalloc(&cp_embed_buf_, talker_h * sizeof(__nv_bfloat16));
    cudaMalloc(&cp_decode_hidden_, cp_h_sz * sizeof(__nv_bfloat16));
    cudaMalloc(&cp_pos_gpu_, 3 * cp_max_seq * sizeof(int));

    initialized_ = true;

    // Report memory usage
    float talker_kv_mb = (float)num_talker_layers * talker_kv * 2 * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float cp_kv_mb = (float)num_cp_layers * cp_kv * 2 * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float ws_mb = workspace_size_ * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    fprintf(stderr, "[TTS Talker] initialized: talker %dL, code_predictor %dL\n",
            num_talker_layers, num_cp_layers);
    fprintf(stderr, "  talker KV cache: %.1f MB, CP KV: %.1f MB, workspace: %.1f MB\n",
            talker_kv_mb, cp_kv_mb, ws_mb);

    // Merge QKV and GateUp weights for both Talker and CodePredictor
    merge_weights(stream);
}

// ============================================================================
// Merge QKV and GateUp projections for reduced kernel launch overhead
// ============================================================================

void Talker::merge_weights(cudaStream_t stream) {
    const auto& tc = config_.talker;
    const auto& cp = config_.code_predictor;

    // Helper: allocate merged weight, copy sub-weights into it
    auto merge_row_major = [&](const __nv_bfloat16* w1, size_t rows1,
                               const __nv_bfloat16* w2, size_t rows2,
                               const __nv_bfloat16* w3, size_t rows3,
                               size_t cols) -> __nv_bfloat16* {
        size_t total_rows = rows1 + rows2 + rows3;
        size_t total_bytes = total_rows * cols * sizeof(__nv_bfloat16);
        void* merged = nullptr;
        cudaMalloc(&merged, total_bytes);
        merged_weight_allocs_.push_back(merged);
        auto* dst = reinterpret_cast<__nv_bfloat16*>(merged);
        cudaMemcpyAsync(dst, w1, rows1 * cols * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(dst + rows1 * cols, w2, rows2 * cols * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        if (rows3 > 0) {
            cudaMemcpyAsync(dst + (rows1 + rows2) * cols, w3, rows3 * cols * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream);
        }
        return dst;
    };
    auto merge2_row_major = [&](const __nv_bfloat16* w1, size_t rows1,
                                const __nv_bfloat16* w2, size_t rows2,
                                size_t cols) -> __nv_bfloat16* {
        return merge_row_major(w1, rows1, w2, rows2, nullptr, 0, cols);
    };

    int talker_q_dim  = tc.num_attention_heads * tc.head_dim;
    int talker_kv_dim = tc.num_kv_heads * tc.head_dim;
    int cp_q_dim  = cp.num_attention_heads * cp.head_dim;
    int cp_kv_dim = cp.num_kv_heads * cp.head_dim;

    size_t merged_bytes = 0;

    // Merge Talker layer weights
    for (int i = 0; i < tc.num_hidden_layers; i++) {
        auto& lw = talker_layer_weights_[i];
        // QKV: [q_dim, h] + [kv_dim, h] + [kv_dim, h] → [q_dim+2*kv_dim, h]
        lw.qkv_proj_w = merge_row_major(lw.q_proj_w, talker_q_dim,
                                         lw.k_proj_w, talker_kv_dim,
                                         lw.v_proj_w, talker_kv_dim,
                                         tc.hidden_size);
        merged_bytes += (size_t)(talker_q_dim + 2 * talker_kv_dim) * tc.hidden_size * sizeof(__nv_bfloat16);

        // GateUp: [inter, h] + [inter, h] → [2*inter, h]
        lw.gate_up_proj_w = merge2_row_major(lw.gate_proj_w, tc.intermediate_size,
                                              lw.up_proj_w, tc.intermediate_size,
                                              tc.hidden_size);
        merged_bytes += (size_t)(2 * tc.intermediate_size) * tc.hidden_size * sizeof(__nv_bfloat16);
    }

    // Merge CodePredictor layer weights
    for (int i = 0; i < cp.num_hidden_layers; i++) {
        auto& cw = cp_layer_weights_[i];
        cw.qkv_proj_w = merge_row_major(cw.q_proj_w, cp_q_dim,
                                         cw.k_proj_w, cp_kv_dim,
                                         cw.v_proj_w, cp_kv_dim,
                                         cp.hidden_size);
        merged_bytes += (size_t)(cp_q_dim + 2 * cp_kv_dim) * cp.hidden_size * sizeof(__nv_bfloat16);

        cw.gate_up_proj_w = merge2_row_major(cw.gate_proj_w, cp.intermediate_size,
                                              cw.up_proj_w, cp.intermediate_size,
                                              cp.hidden_size);
        merged_bytes += (size_t)(2 * cp.intermediate_size) * cp.hidden_size * sizeof(__nv_bfloat16);
    }

    cudaStreamSynchronize(stream);
    fprintf(stderr, "[TTS Talker] merged QKV+GateUp weights: %.1f MB (%d talker + %d CP layers)\n",
            merged_bytes / (1024.0f * 1024.0f), tc.num_hidden_layers, cp.num_hidden_layers);
}

void Talker::reset() {
    talker_cache_len_ = 0;
    cp_cache_len_ = 0;
    generation_step_ = 0;
    prefill_len_ = 0;
    trailing_text_len_ = 0;
    generated_tokens_.clear();
}

// ============================================================================
// Inject continuation text (no reset, preserves KV cache for voice consistency)
// ============================================================================

void Talker::inject_continuation_text(const int* text_ids_cpu, int text_len, cudaStream_t stream) {
    const auto& tc = config_.talker;
    int h = tc.hidden_size;

    // text_ids layout: <|im_start|> assistant \n <text> <|im_end|> \n <|im_start|> assistant \n
    int actual_text_start = 3;
    int actual_text_end = text_len - 5;
    int actual_text_len = actual_text_end - actual_text_start;
    if (actual_text_len < 1) actual_text_len = 1;

    // Use workspace for temporaries (workspace is pre-allocated and large enough)
    // Layout: [text_ids_gpu: text_len ints] [text_embed: actual_text_len * h bf16]
    //         [eos_text_embed: h bf16]
    int* text_ids_gpu = reinterpret_cast<int*>(workspace_);
    size_t ids_bf16 = (text_len * sizeof(int) + sizeof(__nv_bfloat16) - 1) / sizeof(__nv_bfloat16);
    __nv_bfloat16* text_embed = workspace_ + ids_bf16;
    __nv_bfloat16* eos_text_embed = text_embed + actual_text_len * h;
    __nv_bfloat16* proj_ws = workspace_ + ids_bf16 + (actual_text_len + 1) * h;

    cudaMemcpyAsync(text_ids_gpu, text_ids_cpu, text_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Compute text embeddings → text_projection → trailing_text_hidden_
    audio_ops::invoke_embedding_lookup(text_embed, text_ids_gpu + actual_text_start,
                                        text_embedding_w_, actual_text_len, h, stream);
    text_projection_forward(trailing_text_hidden_, text_embed, actual_text_len, proj_ws, stream);

    // Append tts_eos_embed at the end
    int tts_eos_id = config_.tts_eos_token_id;
    cudaMemcpyAsync(text_ids_gpu, &tts_eos_id, sizeof(int), cudaMemcpyHostToDevice, stream);
    audio_ops::invoke_embedding_lookup(eos_text_embed, text_ids_gpu,
                                        text_embedding_w_, 1, h, stream);
    text_projection_forward(trailing_text_hidden_ + actual_text_len * h,
                            eos_text_embed, 1, proj_ws, stream);

    trailing_text_len_ = actual_text_len + 1;  // text tokens + tts_eos

    // Reset generation state for new segment
    generated_tokens_.clear();
    cp_cache_len_ = 0;

    // ---- Bootstrap forward pass: generate fresh logits ----
    // Build input: tts_pad_embed_ (neutral codec) + trailing_text_hidden_[0] (first text)
    // This mirrors the "last prefill position" pattern: text_proj(text) + codec_embed
    __nv_bfloat16* hidden = workspace_;
    cudaMemcpyAsync(hidden, tts_pad_embed_, h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    invoke_add(hidden, trailing_text_hidden_, h, stream);

    // Position = current cache length
    int pos = talker_cache_len_;
    int pos_ids[3] = {pos, pos, pos};
    cudaMemcpyAsync(decode_pos_gpu_, pos_ids, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Forward through all talker layers (T=1 decode)
    __nv_bfloat16* layer_ws = hidden + h;
    for (int layer = 0; layer < tc.num_hidden_layers; layer++) {
        talker_layer_forward_decode(layer, hidden, decode_pos_gpu_, layer_ws, stream);
    }

    // RMSNorm → past_hidden + codec_head → logits
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, hidden, final_norm_w_,
                               tc.rms_norm_eps, 1, h, stream);
    cublas_gemm(cublas_handle_, logits_, norm_out, codec_head_w_,
                1, h, tc.vocab_size, stream);
    invoke_suppress_tokens(logits_, tc.vocab_size - 1024, tc.vocab_size,
                           tc.codec_eos_token_id, stream);
    cudaMemcpyAsync(past_hidden_, norm_out,
                    h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    talker_cache_len_++;
    generation_step_ = 1;  // consumed first text token in bootstrap

    cudaStreamSynchronize(stream);

    fprintf(stderr, "[TTS Talker] continuation injected: %d text tokens, bootstrap cache_len=%d\n",
            trailing_text_len_, talker_cache_len_);
}

// ============================================================================
// Text Projection: output = Linear2(SiLU(Linear1(input)))
// ============================================================================

void Talker::text_projection_forward(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int num_tokens,
    __nv_bfloat16* ws,
    cudaStream_t stream)
{
    int d = config_.talker.text_hidden_size;  // 2048
    int d_out = config_.talker.hidden_size;    // 2048

    // ws layout: fc1_out [num_tokens, d]
    __nv_bfloat16* fc1_out = ws;

    // fc1: [num_tokens, d] → [num_tokens, d], with bias
    cublas_gemm(cublas_handle_, fc1_out, input, text_proj_fc1_w_,
                num_tokens, d, d, stream);
    invoke_add_bias(fc1_out, text_proj_fc1_b_, num_tokens, d, stream);

    // SiLU activation
    invoke_silu(fc1_out, fc1_out, num_tokens * d, stream);

    // fc2: [num_tokens, d] → [num_tokens, d_out], with bias
    cublas_gemm(cublas_handle_, output, fc1_out, text_proj_fc2_w_,
                num_tokens, d, d_out, stream);
    invoke_add_bias(output, text_proj_fc2_b_, num_tokens, d_out, stream);
}

// ============================================================================
// cuBLAS GEMM wrappers
// ============================================================================

void Talker::gemm_bf16(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                       int M, int N, int K, cudaStream_t stream) {
    cublas_gemm(cublas_handle_, C, A, B, M, K, N, stream);
}

void Talker::gemm_bf16_bias(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                            const __nv_bfloat16* bias, int M, int N, int K, cudaStream_t stream) {
    cublas_gemm(cublas_handle_, C, A, B, M, K, N, stream);
    invoke_add_bias(C, bias, M, N, stream);
}

// ============================================================================
// Build Prefill: construct dual-track embeddings for CustomVoice mode
// ============================================================================

int Talker::build_prefill(
    const int* text_ids_cpu, int text_len,
    const std::string& speaker,
    const std::string& language,
    cudaStream_t stream)
{
    const auto& tc = config_.talker;
    int h = tc.hidden_size;

    // Determine language_id (codec token for language)
    int language_id = -1;  // -1 = auto (no language token)
    if (!language.empty() && language != "auto") {
        auto it = tc.codec_language_id.find(language);
        if (it != tc.codec_language_id.end()) {
            language_id = it->second;
        } else {
            fprintf(stderr, "[TTS] Warning: unknown language '%s'\n", language.c_str());
        }
    }

    // Determine speaker_id (codec embedding index)
    int speaker_id = -1;  // -1 = no speaker
    if (!speaker.empty()) {
        auto it = tc.spk_id.find(speaker);
        if (it != tc.spk_id.end()) {
            speaker_id = it->second;
        } else {
            fprintf(stderr, "[TTS] Warning: unknown speaker '%s'\n", speaker.c_str());
        }
    }

    // ======= Build Codec Track =======
    // Codec prefill sequence:
    //   With language:    [think, think_bos, language_id, think_eos]
    //   Without language: [nothink, think_bos, think_eos]
    // Then: [speaker_embed] (if speaker)
    // Then: [pad, bos]

    std::vector<int> codec_prefix;
    if (language_id >= 0) {
        codec_prefix = {tc.codec_think_id, tc.codec_think_bos_id,
                        language_id, tc.codec_think_eos_id};
    } else {
        codec_prefix = {tc.codec_nothink_id, tc.codec_think_bos_id,
                        tc.codec_think_eos_id};
    }

    std::vector<int> codec_suffix = {tc.codec_pad_id, tc.codec_bos_id};

    // Total codec sequence length (including speaker embed slot)
    int codec_prefix_len = (int)codec_prefix.size();
    bool has_speaker = (speaker_id >= 0);
    int codec_total = codec_prefix_len + (has_speaker ? 1 : 0) + (int)codec_suffix.size();

    // Upload codec IDs to GPU and lookup embeddings
    int* codec_ids_gpu;
    cudaMalloc(&codec_ids_gpu, (codec_prefix_len + 2) * sizeof(int));

    // Lookup codec prefix embeddings
    cudaMemcpyAsync(codec_ids_gpu, codec_prefix.data(),
                    codec_prefix_len * sizeof(int), cudaMemcpyHostToDevice, stream);
    __nv_bfloat16* codec_prefix_embed;
    cudaMalloc(&codec_prefix_embed, codec_prefix_len * h * sizeof(__nv_bfloat16));
    audio_ops::invoke_embedding_lookup(codec_prefix_embed, codec_ids_gpu,
                                        codec_embedding_w_, codec_prefix_len, h, stream);

    // Lookup speaker embedding (single codec embedding)
    __nv_bfloat16* speaker_embed = nullptr;
    if (has_speaker) {
        cudaMalloc(&speaker_embed, h * sizeof(__nv_bfloat16));
        int spk = speaker_id;
        cudaMemcpyAsync(codec_ids_gpu, &spk, sizeof(int), cudaMemcpyHostToDevice, stream);
        audio_ops::invoke_embedding_lookup(speaker_embed, codec_ids_gpu,
                                            codec_embedding_w_, 1, h, stream);
    }

    // Lookup codec suffix embeddings [pad, bos]
    cudaMemcpyAsync(codec_ids_gpu, codec_suffix.data(), 2 * sizeof(int), cudaMemcpyHostToDevice, stream);
    __nv_bfloat16* codec_suffix_embed;
    cudaMalloc(&codec_suffix_embed, 2 * h * sizeof(__nv_bfloat16));
    audio_ops::invoke_embedding_lookup(codec_suffix_embed, codec_ids_gpu,
                                        codec_embedding_w_, 2, h, stream);

    // Assemble full codec embedding: [codec_prefix, speaker, codec_suffix]
    // codec_embed_all: [codec_total, hidden]
    __nv_bfloat16* codec_embed_all;
    cudaMalloc(&codec_embed_all, codec_total * h * sizeof(__nv_bfloat16));
    int offset = 0;
    cudaMemcpyAsync(codec_embed_all + offset * h, codec_prefix_embed,
                    codec_prefix_len * h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    offset += codec_prefix_len;
    if (has_speaker) {
        cudaMemcpyAsync(codec_embed_all + offset * h, speaker_embed,
                        h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        offset++;
    }
    cudaMemcpyAsync(codec_embed_all + offset * h, codec_suffix_embed,
                    2 * h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    // ======= Build Text Track =======
    // Text structure: <|im_start|> assistant \n <text_tokens> <|im_end|> \n <|im_start|> assistant \n
    // text_ids[0:3] = <|im_start|> assistant \n  → role tokens
    // text_ids[3:text_len-5] = actual text tokens
    // text_ids[text_len-5:] = <|im_end|> \n <|im_start|> assistant \n  → ignored for TTS

    int role_len = 3;  // <|im_start|>, assistant, \n
    int actual_text_start = 3;
    int actual_text_end = text_len - 5;  // exclude closing tokens
    int actual_text_len = actual_text_end - actual_text_start;
    if (actual_text_len < 1) actual_text_len = 1;  // at least 1 text token

    // Upload text IDs to GPU
    int* text_ids_gpu;
    cudaMalloc(&text_ids_gpu, text_len * sizeof(int));
    cudaMemcpyAsync(text_ids_gpu, text_ids_cpu, text_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Compute text embeddings and apply text_projection
    // Role tokens: text_ids[0:3] → text_embedding → text_projection
    __nv_bfloat16* role_embed;
    cudaMalloc(&role_embed, role_len * h * sizeof(__nv_bfloat16));
    __nv_bfloat16* role_text_embed;
    cudaMalloc(&role_text_embed, role_len * h * sizeof(__nv_bfloat16));
    audio_ops::invoke_embedding_lookup(role_text_embed, text_ids_gpu,
                                        text_embedding_w_, role_len, h, stream);
    text_projection_forward(role_embed, role_text_embed, role_len, workspace_, stream);

    // Compute TTS special embeddings: tts_pad, tts_bos, tts_eos
    int tts_special_ids[3] = {config_.tts_pad_token_id, config_.tts_bos_token_id, config_.tts_eos_token_id};
    int* tts_ids_gpu;
    cudaMalloc(&tts_ids_gpu, 3 * sizeof(int));
    cudaMemcpyAsync(tts_ids_gpu, tts_special_ids, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);
    __nv_bfloat16* tts_special_text_embed;
    cudaMalloc(&tts_special_text_embed, 3 * h * sizeof(__nv_bfloat16));
    audio_ops::invoke_embedding_lookup(tts_special_text_embed, tts_ids_gpu,
                                        text_embedding_w_, 3, h, stream);
    __nv_bfloat16* tts_special_embed;
    cudaMalloc(&tts_special_embed, 3 * h * sizeof(__nv_bfloat16));
    text_projection_forward(tts_special_embed, tts_special_text_embed, 3, workspace_, stream);

    // tts_pad_embed = tts_special_embed[0], cache for decode
    cudaMemcpyAsync(tts_pad_embed_, tts_special_embed, h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
    __nv_bfloat16* tts_bos_embed = tts_special_embed + h;
    __nv_bfloat16* tts_eos_embed = tts_special_embed + 2 * h;

    // ======= Construct Prefill Embedding (streaming mode) =======
    // Layout: [role_embed(3)] + [_talker_input_embed(codec_total-1)] + [first_text_token(1)]
    //
    // _talker_input_embed = tts_pad * (codec_total-2) + tts_bos
    //                       + codec_embed_all[0:codec_total-1]  (additive)
    //
    // first_text_token = text_projection(text_embedding(text_ids[3]))
    //                    + codec_embed_all[codec_total-1]  (additive)

    int prefill_total = role_len + (codec_total - 1) + 1;  // role + codec_aligned + first_text
    prefill_len_ = prefill_total;

    // Copy role embed to prefill buffer
    cudaMemcpyAsync(prefill_embeds_, role_embed,
                    role_len * h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);

    // Build _talker_input_embed (dual-track additive)
    // Text track: tts_pad repeated (codec_total-2) times, then tts_bos
    __nv_bfloat16* dual_start = prefill_embeds_ + role_len * h;
    for (int i = 0; i < codec_total - 2; i++) {
        cudaMemcpyAsync(dual_start + i * h, tts_pad_embed_,
                        h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    }
    cudaMemcpyAsync(dual_start + (codec_total - 2) * h, tts_bos_embed,
                    h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    // Add codec track: codec_embed_all[0:codec_total-1]
    invoke_add(dual_start, codec_embed_all,
               (codec_total - 1) * h, stream);

    // First text token (additive with last codec embed)
    __nv_bfloat16* first_text_pos = prefill_embeds_ + (role_len + codec_total - 1) * h;
    // text_projection(text_embedding(text_ids[3]))
    __nv_bfloat16* first_text_raw;
    cudaMalloc(&first_text_raw, h * sizeof(__nv_bfloat16));
    audio_ops::invoke_embedding_lookup(first_text_raw, text_ids_gpu + actual_text_start,
                                        text_embedding_w_, 1, h, stream);
    text_projection_forward(first_text_pos, first_text_raw, 1, workspace_, stream);
    // + codec_embed_all[codec_total-1]
    invoke_add(first_text_pos, codec_embed_all + (codec_total - 1) * h, h, stream);

    // ======= Build Trailing Text Hidden (streaming text injection) =======
    // trailing = text_projection(text_embedding(text_ids[4:text_len-5])) + tts_eos_embed
    int trail_text_len = actual_text_len - 1;  // exclude first text token already in prefill
    if (trail_text_len < 0) trail_text_len = 0;
    trailing_text_len_ = trail_text_len + 1;  // +1 for tts_eos

    if (trail_text_len > 0) {
        __nv_bfloat16* trail_text_embed;
        cudaMalloc(&trail_text_embed, trail_text_len * h * sizeof(__nv_bfloat16));
        audio_ops::invoke_embedding_lookup(trail_text_embed,
                                            text_ids_gpu + actual_text_start + 1,
                                            text_embedding_w_, trail_text_len, h, stream);
        text_projection_forward(trailing_text_hidden_, trail_text_embed,
                                trail_text_len, workspace_, stream);
        cudaFree(trail_text_embed);
    }
    // Append tts_eos_embed at the end
    cudaMemcpyAsync(trailing_text_hidden_ + trail_text_len * h, tts_eos_embed,
                    h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    // ======= Build Position IDs (MRoPE, all 3 dims same for TTS) =======
    std::vector<int> pos_ids(3 * prefill_total);
    for (int d = 0; d < 3; d++) {
        for (int i = 0; i < prefill_total; i++) {
            pos_ids[d * prefill_total + i] = i;
        }
    }
    cudaMemcpyAsync(position_ids_, pos_ids.data(),
                    3 * prefill_total * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Reset generation state
    generation_step_ = 0;
    talker_cache_len_ = 0;
    generated_tokens_.clear();

    // Cleanup temporaries
    cudaStreamSynchronize(stream);
    cudaFree(codec_ids_gpu);
    cudaFree(codec_prefix_embed);
    if (speaker_embed) cudaFree(speaker_embed);
    cudaFree(codec_suffix_embed);
    cudaFree(codec_embed_all);
    cudaFree(text_ids_gpu);
    cudaFree(role_embed);
    cudaFree(role_text_embed);
    cudaFree(tts_ids_gpu);
    cudaFree(tts_special_text_embed);
    cudaFree(tts_special_embed);
    cudaFree(first_text_raw);

    fprintf(stderr, "[TTS Talker] prefill built: %d tokens, trailing_text=%d tokens\n",
            prefill_len_, trailing_text_len_);
    return prefill_len_;
}

// ============================================================================
// Talker layer forward (prefill, T > 1)
// ============================================================================

void Talker::talker_layer_forward_prefill(
    int layer_idx,
    __nv_bfloat16* hidden,
    const int* pos_ids,
    int seq_len,
    __nv_bfloat16* ws,
    cudaStream_t stream)
{
    const auto& lw = talker_layer_weights_[layer_idx];
    const auto& tc = config_.talker;
    int h = tc.hidden_size;
    int q_dim = tc.num_attention_heads * tc.head_dim;
    int kv_dim = tc.num_kv_heads * tc.head_dim;
    float eps = tc.rms_norm_eps;

    __nv_bfloat16* norm_buf = ws;
    __nv_bfloat16* q_buf    = norm_buf + (size_t)seq_len * h;
    __nv_bfloat16* k_buf    = q_buf    + (size_t)seq_len * q_dim;
    __nv_bfloat16* v_buf    = k_buf    + (size_t)seq_len * kv_dim;
    __nv_bfloat16* attn_out = v_buf    + (size_t)seq_len * kv_dim;
    __nv_bfloat16* gate_buf = attn_out + (size_t)seq_len * h;
    __nv_bfloat16* up_buf   = gate_buf + (size_t)seq_len * tc.intermediate_size;

    // Self-Attention (separate QKV for prefill — merged output is interleaved for T>1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.input_layernorm_w,
                               eps, seq_len, h, stream);
    cublas_gemm(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
    cublas_gemm(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
    cublas_gemm(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);

    // Per-head QK RMSNorm
    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w, eps,
                                        seq_len, tc.num_attention_heads, tc.head_dim, stream);
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w, eps,
                                        seq_len, tc.num_kv_heads, tc.head_dim, stream);

    // MRoPE
    audio_ops::invoke_mrope(q_buf, k_buf, pos_ids,
                             seq_len, tc.num_attention_heads, tc.num_kv_heads, tc.head_dim,
                             tc.mrope_sections[0], tc.mrope_sections[1], tc.mrope_sections[2],
                             tc.rope_theta, stream);

    // Write KV to cache
    audio_ops::invoke_write_kv_cache(talker_k_cache_[layer_idx], talker_v_cache_[layer_idx],
                                      k_buf, v_buf, 0, seq_len,
                                      tc.num_kv_heads, tc.head_dim, stream);

    // Causal GQA prefill attention
    audio_ops::invoke_causal_gqa_prefill(
        attn_out, q_buf,
        talker_k_cache_[layer_idx], talker_v_cache_[layer_idx],
        seq_len, tc.num_attention_heads, tc.num_kv_heads, tc.head_dim, stream);

    // O projection + residual
    cublas_gemm(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, seq_len * h, stream);

    // MLP (SwiGLU) — separate gate/up for prefill (merged output interleaved for T>1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.post_attention_layernorm_w,
                               eps, seq_len, h, stream);
    cublas_gemm(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w,
                seq_len, h, tc.intermediate_size, stream);
    cublas_gemm(cublas_handle_, up_buf, norm_buf, lw.up_proj_w,
                seq_len, h, tc.intermediate_size, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, tc.intermediate_size, stream);
    cublas_gemm(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w,
                seq_len, tc.intermediate_size, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, seq_len * h, stream);
}

// ============================================================================
// Talker layer forward (decode, T=1)
// ============================================================================

void Talker::talker_layer_forward_decode(
    int layer_idx,
    __nv_bfloat16* hidden,
    const int* pos_ids,
    __nv_bfloat16* ws,
    cudaStream_t stream)
{
    const auto& lw = talker_layer_weights_[layer_idx];
    const auto& tc = config_.talker;
    int h = tc.hidden_size;
    int q_dim = tc.num_attention_heads * tc.head_dim;
    int kv_dim = tc.num_kv_heads * tc.head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    float eps = tc.rms_norm_eps;

    __nv_bfloat16* norm_buf = ws;
    __nv_bfloat16* qkv_buf  = norm_buf + h;      // merged QKV output
    __nv_bfloat16* q_buf    = qkv_buf;            // alias into merged output
    __nv_bfloat16* k_buf    = qkv_buf + q_dim;
    __nv_bfloat16* v_buf    = qkv_buf + q_dim + kv_dim;
    __nv_bfloat16* attn_out = qkv_buf + qkv_dim;
    __nv_bfloat16* gateup_buf = attn_out + h;     // merged GateUp output
    __nv_bfloat16* gate_buf = gateup_buf;          // alias
    __nv_bfloat16* up_buf   = gateup_buf + tc.intermediate_size;

    // Self-Attention: fused QKV projection (3 GEMV → 1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.input_layernorm_w,
                               eps, 1, h, stream);
    cublas_gemm(cublas_handle_, qkv_buf, norm_buf, lw.qkv_proj_w, 1, h, qkv_dim, stream);

    // Per-head QK RMSNorm
    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w, eps,
                                        1, tc.num_attention_heads, tc.head_dim, stream);
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w, eps,
                                        1, tc.num_kv_heads, tc.head_dim, stream);

    audio_ops::invoke_mrope(q_buf, k_buf, pos_ids,
                             1, tc.num_attention_heads, tc.num_kv_heads, tc.head_dim,
                             tc.mrope_sections[0], tc.mrope_sections[1], tc.mrope_sections[2],
                             tc.rope_theta, stream);

    audio_ops::invoke_write_kv_cache(talker_k_cache_[layer_idx], talker_v_cache_[layer_idx],
                                      k_buf, v_buf, talker_cache_len_, 1,
                                      tc.num_kv_heads, tc.head_dim, stream);

    audio_ops::invoke_causal_gqa_decode(
        attn_out, q_buf,
        talker_k_cache_[layer_idx], talker_v_cache_[layer_idx],
        1, tc.num_attention_heads, tc.num_kv_heads, tc.head_dim,
        talker_cache_len_ + 1, stream);

    cublas_gemm(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, 1, q_dim, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, h, stream);

    // MLP: fused GateUp projection (2 GEMV → 1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.post_attention_layernorm_w,
                               eps, 1, h, stream);
    cublas_gemm(cublas_handle_, gateup_buf, norm_buf, lw.gate_up_proj_w,
                1, h, 2 * tc.intermediate_size, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, tc.intermediate_size, stream);
    cublas_gemm(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w,
                1, tc.intermediate_size, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, h, stream);
}

// ============================================================================
// CodePredictor layer forward (prefill/decode)
// ============================================================================

void Talker::cp_layer_forward_prefill(
    int layer_idx,
    __nv_bfloat16* hidden,
    int seq_len,
    __nv_bfloat16* ws,
    cudaStream_t stream)
{
    const auto& lw = cp_layer_weights_[layer_idx];
    const auto& cp = config_.code_predictor;
    int h = cp.hidden_size;
    int q_dim = cp.num_attention_heads * cp.head_dim;
    int kv_dim = cp.num_kv_heads * cp.head_dim;
    float eps = cp.rms_norm_eps;

    __nv_bfloat16* norm_buf = ws;
    __nv_bfloat16* q_buf    = norm_buf + (size_t)seq_len * h;
    __nv_bfloat16* k_buf    = q_buf    + (size_t)seq_len * q_dim;
    __nv_bfloat16* v_buf    = k_buf    + (size_t)seq_len * kv_dim;
    __nv_bfloat16* attn_out = v_buf    + (size_t)seq_len * kv_dim;
    __nv_bfloat16* gate_buf = attn_out + (size_t)seq_len * h;
    __nv_bfloat16* up_buf   = gate_buf + (size_t)seq_len * cp.intermediate_size;

    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.input_layernorm_w, eps, seq_len, h, stream);
    cublas_gemm(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
    cublas_gemm(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
    cublas_gemm(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);

    // Per-head QK RMSNorm
    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w, eps,
                                        seq_len, cp.num_attention_heads, cp.head_dim, stream);
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w, eps,
                                        seq_len, cp.num_kv_heads, cp.head_dim, stream);

    // Standard 1D RoPE for CodePredictor (position = 0..seq_len-1)
    // Use position_ids from the CP's perspective
    // Build simple sequential position IDs on stack
    std::vector<int> cp_pos(3 * seq_len);
    for (int d = 0; d < 3; d++)
        for (int i = 0; i < seq_len; i++)
            cp_pos[d * seq_len + i] = cp_cache_len_ + i;
    cudaMemcpyAsync(cp_pos_gpu_, cp_pos.data(), 3 * seq_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Use invoke_rope_1d or invoke_mrope with all sections equal
    // Since CodePredictor uses standard 1D RoPE and head_dim=128:
    // Use mrope with sections that cover full head_dim/2 = 64 in first section
    audio_ops::invoke_mrope(q_buf, k_buf, cp_pos_gpu_,
                             seq_len, cp.num_attention_heads, cp.num_kv_heads, cp.head_dim,
                             64, 0, 0, cp.rope_theta, stream);

    audio_ops::invoke_write_kv_cache(cp_k_cache_[layer_idx], cp_v_cache_[layer_idx],
                                      k_buf, v_buf, cp_cache_len_, seq_len,
                                      cp.num_kv_heads, cp.head_dim, stream);
    audio_ops::invoke_causal_gqa_prefill(
        attn_out, q_buf,
        cp_k_cache_[layer_idx], cp_v_cache_[layer_idx],
        seq_len, cp.num_attention_heads, cp.num_kv_heads, cp.head_dim, stream);

    cublas_gemm(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, seq_len * h, stream);

    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.post_attention_layernorm_w, eps, seq_len, h, stream);
    cublas_gemm(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w, seq_len, h, cp.intermediate_size, stream);
    cublas_gemm(cublas_handle_, up_buf, norm_buf, lw.up_proj_w, seq_len, h, cp.intermediate_size, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, cp.intermediate_size, stream);
    cublas_gemm(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, seq_len, cp.intermediate_size, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, seq_len * h, stream);
}

void Talker::cp_layer_forward_decode(
    int layer_idx,
    __nv_bfloat16* hidden,
    const int* pos_ids,
    __nv_bfloat16* ws,
    cudaStream_t stream)
{
    const auto& lw = cp_layer_weights_[layer_idx];
    const auto& cp = config_.code_predictor;
    int h = cp.hidden_size;
    int q_dim = cp.num_attention_heads * cp.head_dim;
    int kv_dim = cp.num_kv_heads * cp.head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    float eps = cp.rms_norm_eps;

    __nv_bfloat16* norm_buf = ws;
    __nv_bfloat16* qkv_buf  = norm_buf + h;      // merged QKV output
    __nv_bfloat16* q_buf    = qkv_buf;            // alias
    __nv_bfloat16* k_buf    = qkv_buf + q_dim;
    __nv_bfloat16* v_buf    = qkv_buf + q_dim + kv_dim;
    __nv_bfloat16* attn_out = qkv_buf + qkv_dim;
    __nv_bfloat16* gateup_buf = attn_out + h;     // merged GateUp output
    __nv_bfloat16* gate_buf = gateup_buf;
    __nv_bfloat16* up_buf   = gateup_buf + cp.intermediate_size;

    // Fused QKV projection (3 GEMV → 1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.input_layernorm_w, eps, 1, h, stream);
    cublas_gemm(cublas_handle_, qkv_buf, norm_buf, lw.qkv_proj_w, 1, h, qkv_dim, stream);

    // Per-head QK RMSNorm
    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w, eps,
                                        1, cp.num_attention_heads, cp.head_dim, stream);
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w, eps,
                                        1, cp.num_kv_heads, cp.head_dim, stream);

    // 1D RoPE — position IDs passed in from caller (set once per group step)
    audio_ops::invoke_mrope(q_buf, k_buf, pos_ids,
                             1, cp.num_attention_heads, cp.num_kv_heads, cp.head_dim,
                             64, 0, 0, cp.rope_theta, stream);

    audio_ops::invoke_write_kv_cache(cp_k_cache_[layer_idx], cp_v_cache_[layer_idx],
                                      k_buf, v_buf, cp_cache_len_, 1,
                                      cp.num_kv_heads, cp.head_dim, stream);
    audio_ops::invoke_causal_gqa_decode(
        attn_out, q_buf,
        cp_k_cache_[layer_idx], cp_v_cache_[layer_idx],
        1, cp.num_attention_heads, cp.num_kv_heads, cp.head_dim,
        cp_cache_len_ + 1, stream);

    cublas_gemm(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, 1, q_dim, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, h, stream);

    // Fused GateUp projection (2 GEMV → 1)
    audio_ops::invoke_rmsnorm(norm_buf, hidden, lw.post_attention_layernorm_w, eps, 1, h, stream);
    cublas_gemm(cublas_handle_, gateup_buf, norm_buf, lw.gate_up_proj_w, 1, h, 2 * cp.intermediate_size, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, cp.intermediate_size, stream);
    cublas_gemm(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, 1, cp.intermediate_size, h, stream);
    audio_ops::invoke_add_residual(hidden, norm_buf, h, stream);
}

// ============================================================================
// Run CodePredictor: generate codec groups 1-15 from past_hidden + group_0
// ============================================================================

void Talker::run_code_predictor(
    const __nv_bfloat16* talker_hidden,
    int group_0_id,
    int* codec_out,
    cudaStream_t stream)
{
    const auto& tc = config_.talker;
    const auto& cp = config_.code_predictor;
    int talker_h = tc.hidden_size;         // 2048
    int cp_h = cp.hidden_size;             // 1024
    int num_groups = tc.num_code_groups - 1; // 15

    // Reset CodePredictor KV cache
    cp_cache_len_ = 0;

    // Build CodePredictor prefill: [past_hidden, codec_embedding(group_0)]
    // Copy past_hidden (talker's last hidden state)
    cudaMemcpyAsync(cp_input_buf_, talker_hidden,
                    talker_h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    // Lookup group_0 embedding — use GPU-resident token from codec_out_gpu_[0]
    audio_ops::invoke_embedding_lookup(cp_input_buf_ + talker_h, codec_out_gpu_,
                                        codec_embedding_w_, 1, talker_h, stream);

    // Project: [2, talker_h] → [2, cp_h]
    if (cp_projection_w_) {
        cublas_gemm(cublas_handle_, cp_hidden_buf_, cp_input_buf_, cp_projection_w_,
                    2, talker_h, cp_h, stream);
        invoke_add_bias(cp_hidden_buf_, cp_projection_b_, 2, cp_h, stream);
    } else {
        cudaMemcpyAsync(cp_hidden_buf_, cp_input_buf_,
                        2 * cp_h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
    }

    // Prefill through CodePredictor layers
    for (int layer = 0; layer < cp.num_hidden_layers; layer++) {
        cp_layer_forward_prefill(layer, cp_hidden_buf_, 2, workspace_, stream);
    }
    cp_cache_len_ = 2;

    // RMSNorm + lm_head[0] on last token → logits for group 1
    __nv_bfloat16* last_hidden = cp_hidden_buf_ + cp_h;
    __nv_bfloat16* cp_norm_out = workspace_;
    audio_ops::invoke_rmsnorm(cp_norm_out, last_hidden, cp_final_norm_w_,
                               cp.rms_norm_eps, 1, cp_h, stream);
    cublas_gemm(cublas_handle_, cp_logits_, cp_norm_out, cp_lm_heads_[0],
                1, cp_h, cp.vocab_size, stream);

    // Sample group 1 — GPU-resident, no sync (result → codec_out_gpu_[1])
    static std::mt19937_64 cp_rng(42);
    invoke_gpu_sample_top_k_top_p(cp_logits_, cp.vocab_size,
                                   sub_top_k_, sub_top_p_, sub_temperature_,
                                   codec_out_gpu_ + 1, cp_rng(), stream);

    // Decode groups 2-15 autoregressively — fully GPU-resident, zero intermediate syncs
    for (int g = 1; g < num_groups; g++) {
        // Lookup embedding from GPU-resident previous token (codec_out_gpu_[g])
        audio_ops::invoke_embedding_lookup(cp_embed_buf_, codec_out_gpu_ + g,
                                            cp_codec_embeddings_[g - 1], 1, talker_h, stream);

        // Project to cp_h
        if (cp_projection_w_) {
            cublas_gemm(cublas_handle_, cp_decode_hidden_, cp_embed_buf_, cp_projection_w_,
                        1, talker_h, cp_h, stream);
            invoke_add_bias(cp_decode_hidden_, cp_projection_b_, 1, cp_h, stream);
        } else {
            cudaMemcpyAsync(cp_decode_hidden_, cp_embed_buf_,
                            cp_h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        }

        // Set position IDs once per group step (not per layer)
        int pos = cp_cache_len_;
        int cp_pos[3] = {pos, pos, pos};
        cudaMemcpyAsync(cp_pos_gpu_, cp_pos, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Forward through CodePredictor layers (decode T=1)
        for (int layer = 0; layer < cp.num_hidden_layers; layer++) {
            cp_layer_forward_decode(layer, cp_decode_hidden_, cp_pos_gpu_, workspace_, stream);
        }
        cp_cache_len_++;

        // RMSNorm + lm_head[g]
        audio_ops::invoke_rmsnorm(cp_norm_out, cp_decode_hidden_, cp_final_norm_w_,
                                   cp.rms_norm_eps, 1, cp_h, stream);
        cublas_gemm(cublas_handle_, cp_logits_, cp_norm_out, cp_lm_heads_[g],
                    1, cp_h, cp.vocab_size, stream);

        // GPU-resident sampling — result → codec_out_gpu_[g+1]
        invoke_gpu_sample_top_k_top_p(cp_logits_, cp.vocab_size,
                                       sub_top_k_, sub_top_p_, sub_temperature_,
                                       codec_out_gpu_ + g + 1, cp_rng(), stream);
    }

    // Async copy — let caller's sync handle completion
    cudaMemcpyAsync(codec_out, codec_out_gpu_ + 1, num_groups * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
}

// ============================================================================
// Forward Prefill: run prefill through talker transformer
// ============================================================================

void Talker::forward_prefill(cudaStream_t stream) {
    if (!initialized_ || prefill_len_ == 0) {
        fprintf(stderr, "[TTS Talker] ERROR: not initialized or no prefill built\n");
        return;
    }

    const auto& tc = config_.talker;
    int h = tc.hidden_size;
    int seq_len = prefill_len_;

    // hidden_states starts as prefill_embeds_, modified in-place
    // Copy to workspace first
    __nv_bfloat16* hidden = workspace_;
    cudaMemcpyAsync(hidden, prefill_embeds_,
                    (size_t)seq_len * h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);

    __nv_bfloat16* layer_ws = hidden + (size_t)max_seq_len_ * h;

    // Process all talker layers
    for (int layer = 0; layer < tc.num_hidden_layers; layer++) {
        talker_layer_forward_prefill(layer, hidden, position_ids_, seq_len, layer_ws, stream);
    }

    // Final RMSNorm on last token
    __nv_bfloat16* last_hidden = hidden + (size_t)(seq_len - 1) * h;
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, last_hidden, final_norm_w_,
                               tc.rms_norm_eps, 1, h, stream);

    // codec_head: [1, h] → [1, vocab_size]
    cublas_gemm(cublas_handle_, logits_, norm_out, codec_head_w_,
                1, h, tc.vocab_size, stream);

    // Suppress special tokens: [vocab_size-1024, vocab_size) except codec_eos
    invoke_suppress_tokens(logits_, tc.vocab_size - 1024, tc.vocab_size,
                           tc.codec_eos_token_id, stream);

    // Save past_hidden for CodePredictor
    cudaMemcpyAsync(past_hidden_, norm_out,
                    h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    talker_cache_len_ = seq_len;
    generation_step_ = 0;

    cudaStreamSynchronize(stream);

    fprintf(stderr, "[TTS Talker] prefill done: %d tokens, cache_len=%d\n",
            seq_len, talker_cache_len_);
}

// ============================================================================
// Forward Decode Step: sample + CodePredictor + combine + forward
// ============================================================================

int Talker::forward_decode_step(int* codec_out, cudaStream_t stream) {
    const auto& tc = config_.talker;
    int h = tc.hidden_size;
    int num_groups = tc.num_code_groups;

    // Step 1: Apply repetition penalty to logits
    if (rep_penalty_ != 1.0f && !generated_tokens_.empty()) {
        int n_rep = (int)generated_tokens_.size();
        cudaMemcpyAsync(rep_ids_gpu_, generated_tokens_.data(),
                        n_rep * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        invoke_repetition_penalty(logits_, rep_ids_gpu_, n_rep,
                                   rep_penalty_, stream);
    }

    // Step 2: Sample group 0 on GPU (vocab=3072 fits in SMEM)
    static std::mt19937_64 talker_rng(12345);
    invoke_gpu_sample_top_k_top_p(logits_, tc.vocab_size,
                                   top_k_, top_p_, temperature_,
                                   codec_out_gpu_, talker_rng(), stream);

    // Single sync + D2H copy for EOS check
    cudaStreamSynchronize(stream);
    int group_0_id;
    cudaMemcpy(&group_0_id, codec_out_gpu_, sizeof(int), cudaMemcpyDeviceToHost);
    codec_out[0] = group_0_id;

    // Check EOS
    if (group_0_id == tc.codec_eos_token_id) {
        return -1;  // EOS
    }

    // Track for repetition penalty
    generated_tokens_.push_back(group_0_id);

    // Step 3: Run CodePredictor to generate groups 1-15 (GPU-resident, single sync)
    run_code_predictor(past_hidden_, group_0_id, codec_out + 1, stream);

    // Step 4: Combine all 16 group embeddings using GPU-resident tokens
    // Group 0: use codec_out_gpu_[0]
    audio_ops::invoke_embedding_lookup(decode_all_embeds_, codec_out_gpu_,
                                        codec_embedding_w_, 1, h, stream);

    // Groups 1-15: use codec_out_gpu_[1..15] — no H2D roundtrips
    for (int g = 0; g < num_groups - 1; g++) {
        audio_ops::invoke_embedding_lookup(decode_all_embeds_ + (g + 1) * h,
                                            codec_out_gpu_ + g + 1,
                                            cp_codec_embeddings_[g], 1, h, stream);
    }

    // Sum all 16 embeddings → decode_input_embeds_ [1, hidden_size]
    invoke_sum_embeddings(decode_input_embeds_, decode_all_embeds_, num_groups, h, stream);

    // Step 5: Add trailing text hidden (streaming text injection)
    if (generation_step_ < trailing_text_len_) {
        invoke_add(decode_input_embeds_, trailing_text_hidden_ + generation_step_ * h, h, stream);
    } else {
        invoke_add(decode_input_embeds_, tts_pad_embed_, h, stream);
    }

    // Step 6: Set up position IDs for decode (current position)
    int pos = talker_cache_len_;
    int pos_ids[3] = {pos, pos, pos};
    cudaMemcpyAsync(decode_pos_gpu_, pos_ids, 3 * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Step 7: Forward through talker transformer (decode T=1)
    __nv_bfloat16* hidden = workspace_;
    cudaMemcpyAsync(hidden, decode_input_embeds_, h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);

    __nv_bfloat16* layer_ws = hidden + h;
    for (int layer = 0; layer < tc.num_hidden_layers; layer++) {
        talker_layer_forward_decode(layer, hidden, decode_pos_gpu_, layer_ws, stream);
    }

    // Final RMSNorm
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, hidden, final_norm_w_,
                               tc.rms_norm_eps, 1, h, stream);

    // codec_head → logits for next step
    cublas_gemm(cublas_handle_, logits_, norm_out, codec_head_w_,
                1, h, tc.vocab_size, stream);

    // Suppress special tokens: [vocab_size-1024, vocab_size) except codec_eos
    invoke_suppress_tokens(logits_, tc.vocab_size - 1024, tc.vocab_size,
                           tc.codec_eos_token_id, stream);

    // Save past_hidden for next CodePredictor call
    cudaMemcpyAsync(past_hidden_, norm_out,
                    h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);

    talker_cache_len_++;
    generation_step_++;

    cudaStreamSynchronize(stream);
    return 0;  // success
}

} // namespace tts
} // namespace qwen_thor
