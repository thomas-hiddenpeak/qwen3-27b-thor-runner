// asr_decoder.cu — Qwen3-ASR Text Decoder 实现
//
// 28-layer GQA decoder with MRoPE, per-head Q/K RMSNorm, SwiGLU MLP
// 支持 prefill (T>1) 和 decode (T=1) 两条路径

#include "asr_decoder.h"
#include "audio_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace qwen_thor {
namespace asr {

// ============================================================================
// cuBLAS BF16 linear: out = input @ weight^T (no bias for decoder)
// ============================================================================

static void cublas_linear_nobias(
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
// TextDecoder implementation
// ============================================================================

TextDecoder::TextDecoder(const ASRConfig& config, int max_seq_len)
    : config_(config)
    , max_seq_len_(max_seq_len)
    , layer_weights_(config.decoder_layers)
    , k_cache_(config.decoder_layers, nullptr)
    , v_cache_(config.decoder_layers, nullptr) {}

TextDecoder::~TextDecoder() {
    for (int i = 0; i < config_.decoder_layers; i++) {
        if (k_cache_[i]) cudaFree(k_cache_[i]);
        if (v_cache_[i]) cudaFree(v_cache_[i]);
    }
    if (workspace_) cudaFree(workspace_);
    if (token_id_gpu_) cudaFree(token_id_gpu_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

void TextDecoder::set_embed_weights(
    __nv_bfloat16* embed_tokens_w,
    __nv_bfloat16* lm_head_w,
    __nv_bfloat16* final_norm_w)
{
    embed_tokens_w_ = embed_tokens_w;
    lm_head_w_ = lm_head_w;
    final_norm_w_ = final_norm_w;
}

void TextDecoder::set_layer_weights(int layer_idx, const DecoderLayerWeights& weights) {
    layer_weights_[layer_idx] = weights;
}

void TextDecoder::initialize(cudaStream_t stream) {
    if (initialized_) return;

    cublasCreate(&cublas_handle_);

    int h = config_.decoder_hidden_size;      // 2048
    int kv_dim = config_.decoder_kv_dim();     // 1024
    int ffn = config_.decoder_intermediate_size; // 6144
    int num_layers = config_.decoder_layers;    // 28

    // Allocate KV cache: [max_seq_len, num_kv_heads, head_dim] per layer
    size_t kv_per_layer = (size_t)max_seq_len_ * kv_dim;
    for (int i = 0; i < num_layers; i++) {
        cudaMalloc(&k_cache_[i], kv_per_layer * sizeof(__nv_bfloat16));
        cudaMalloc(&v_cache_[i], kv_per_layer * sizeof(__nv_bfloat16));
    }

    // Workspace layout for prefill (max T = max_seq_len_):
    //   norm_buf:     max_seq * h
    //   q_buf:        max_seq * q_dim (= h = 2048)
    //   k_buf:        max_seq * kv_dim (= 1024)
    //   v_buf:        max_seq * kv_dim
    //   attn_out:     max_seq * h
    //   gate_buf:     max_seq * ffn (= 6144)
    //   up_buf:       max_seq * ffn
    //   logits_buf:   vocab_size (only last token)
    int q_dim = config_.decoder_q_dim();  // 2048
    workspace_size_ = (size_t)max_seq_len_ * h          // norm_buf
                    + (size_t)max_seq_len_ * q_dim       // q_buf
                    + (size_t)max_seq_len_ * kv_dim      // k_buf
                    + (size_t)max_seq_len_ * kv_dim      // v_buf
                    + (size_t)max_seq_len_ * h           // attn_out
                    + (size_t)max_seq_len_ * ffn          // gate_buf
                    + (size_t)max_seq_len_ * ffn          // up_buf
                    + (size_t)config_.vocab_size           // logits (1 token)
                    + 1024;  // alignment padding

    cudaMalloc(&workspace_, workspace_size_ * sizeof(__nv_bfloat16));
    cudaMemset(workspace_, 0, workspace_size_ * sizeof(__nv_bfloat16));

    // Token ID for decode step (managed memory for CPU/GPU access)
    cudaMallocManaged(&token_id_gpu_, sizeof(int));

    cache_seq_len_ = 0;
    initialized_ = true;

    float kv_mb = (float)num_layers * kv_per_layer * 2 * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float ws_mb = workspace_size_ * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    fprintf(stderr, "[ASR Decoder] initialized: %d layers, max_seq=%d, KV cache %.1f MB, workspace %.1f MB\n",
            num_layers, max_seq_len_, kv_mb, ws_mb);
}

void TextDecoder::reset_cache() {
    cache_seq_len_ = 0;
}

// ============================================================================
// decoder_layer_forward_prefill: single layer, T > 1
// ============================================================================

void TextDecoder::decoder_layer_forward_prefill(
    int layer_idx,
    __nv_bfloat16* hidden_states,
    const int* position_ids,
    int seq_len,
    __nv_bfloat16* workspace_base,
    cudaStream_t stream)
{
    const auto& lw = layer_weights_[layer_idx];
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int num_q_heads = config_.decoder_num_attention_heads;
    int num_kv_heads = config_.decoder_num_kv_heads;
    int head_dim = config_.decoder_head_dim;
    float eps = config_.rms_norm_eps;

    // Workspace pointers
    __nv_bfloat16* norm_buf  = workspace_base;
    __nv_bfloat16* q_buf     = norm_buf  + (size_t)seq_len * h;
    __nv_bfloat16* k_buf     = q_buf     + (size_t)seq_len * q_dim;
    __nv_bfloat16* v_buf     = k_buf     + (size_t)seq_len * kv_dim;
    __nv_bfloat16* attn_out  = v_buf     + (size_t)seq_len * kv_dim;
    __nv_bfloat16* gate_buf  = attn_out  + (size_t)seq_len * h;
    __nv_bfloat16* up_buf    = gate_buf  + (size_t)seq_len * config_.decoder_intermediate_size;

    // === Self-Attention ===

    // 1. RMSNorm (plain weight)
    audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.input_layernorm_w,
                               eps, seq_len, h, stream);

    // 2. Q/K/V projections (no bias)
    cublas_linear_nobias(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
    cublas_linear_nobias(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
    cublas_linear_nobias(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);

    // 3. Per-head Q/K RMSNorm
    // Q: [seq_len, num_q_heads * head_dim] → reinterpret as [seq_len * num_q_heads, head_dim]
    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w,
                                        eps, seq_len, num_q_heads, head_dim, stream);
    // K: [seq_len, num_kv_heads * head_dim]
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w,
                                        eps, seq_len, num_kv_heads, head_dim, stream);

    // 4. MRoPE (half-rotation, interleaved sections)
    // Q: [seq_len, num_q_heads, head_dim], K: [seq_len, num_kv_heads, head_dim]
    audio_ops::invoke_mrope(q_buf, k_buf, position_ids,
                             seq_len, num_q_heads, num_kv_heads, head_dim,
                             config_.mrope_section[0], config_.mrope_section[1],
                             config_.mrope_section[2], config_.rope_theta, stream);

    // 5. Write K/V to cache
    audio_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                      k_buf, v_buf,
                                      0, seq_len,  // start_pos = 0 for prefill
                                      num_kv_heads, head_dim, stream);

    // 6. Causal GQA prefill attention
    // Q: [seq_len, num_q_heads, head_dim]
    // K, V from cache: [seq_len, num_kv_heads, head_dim]
    audio_ops::invoke_causal_gqa_prefill(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        seq_len, num_q_heads, num_kv_heads, head_dim, stream);

    // 7. Output projection: [seq_len, q_dim] → [seq_len, h]
    cublas_linear_nobias(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream);

    // 8. Residual add
    audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);

    // === MLP (SwiGLU) ===

    // 9. RMSNorm
    audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w,
                               eps, seq_len, h, stream);

    // 10. Gate + Up projections: [seq_len, h] → [seq_len, ffn]
    int ffn = config_.decoder_intermediate_size;
    cublas_linear_nobias(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w, seq_len, h, ffn, stream);
    cublas_linear_nobias(cublas_handle_, up_buf, norm_buf, lw.up_proj_w, seq_len, h, ffn, stream);

    // 11. SwiGLU: out = silu(gate) * up
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, ffn, stream);

    // 12. Down projection: [seq_len, ffn] → [seq_len, h]
    cublas_linear_nobias(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, seq_len, ffn, h, stream);

    // 13. Residual add
    audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);

    cudaStreamSynchronize(stream);
}

// ============================================================================
// decoder_layer_forward_decode: single layer, T=1
// ============================================================================

void TextDecoder::decoder_layer_forward_decode(
    int layer_idx,
    __nv_bfloat16* hidden_states,     // [1, hidden_size]
    const int* position_ids,           // [3, 1]
    __nv_bfloat16* workspace_base,
    cudaStream_t stream)
{
    const auto& lw = layer_weights_[layer_idx];
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int num_q_heads = config_.decoder_num_attention_heads;
    int num_kv_heads = config_.decoder_num_kv_heads;
    int head_dim = config_.decoder_head_dim;
    float eps = config_.rms_norm_eps;

    // Workspace (T=1, small buffers)
    __nv_bfloat16* norm_buf  = workspace_base;
    __nv_bfloat16* q_buf     = norm_buf  + h;
    __nv_bfloat16* k_buf     = q_buf     + q_dim;
    __nv_bfloat16* v_buf     = k_buf     + kv_dim;
    __nv_bfloat16* attn_out  = v_buf     + kv_dim;
    __nv_bfloat16* gate_buf  = attn_out  + h;
    __nv_bfloat16* up_buf    = gate_buf  + config_.decoder_intermediate_size;

    // === Self-Attention ===
    audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.input_layernorm_w,
                               eps, 1, h, stream);

    cublas_linear_nobias(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, 1, h, q_dim, stream);
    cublas_linear_nobias(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, 1, h, kv_dim, stream);
    cublas_linear_nobias(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, 1, h, kv_dim, stream);

    audio_ops::invoke_per_head_rmsnorm(q_buf, q_buf, lw.q_norm_w,
                                        eps, 1, num_q_heads, head_dim, stream);
    audio_ops::invoke_per_head_rmsnorm(k_buf, k_buf, lw.k_norm_w,
                                        eps, 1, num_kv_heads, head_dim, stream);

    audio_ops::invoke_mrope(q_buf, k_buf, position_ids,
                             1, num_q_heads, num_kv_heads, head_dim,
                             config_.mrope_section[0], config_.mrope_section[1],
                             config_.mrope_section[2], config_.rope_theta, stream);

    // Write new K/V to cache
    audio_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                      k_buf, v_buf,
                                      cache_seq_len_, 1,
                                      num_kv_heads, head_dim, stream);

    // Decode attention: Q against full KV cache
    audio_ops::invoke_causal_gqa_decode(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        1,  // batch_size=1
        num_q_heads, num_kv_heads, head_dim,
        cache_seq_len_ + 1,  // current total seq len including this token
        stream);

    cublas_linear_nobias(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, 1, q_dim, h, stream);
    audio_ops::invoke_add_residual(hidden_states, norm_buf, h, stream);

    // === MLP ===
    audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w,
                               eps, 1, h, stream);

    int ffn = config_.decoder_intermediate_size;
    cublas_linear_nobias(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w, 1, h, ffn, stream);
    cublas_linear_nobias(cublas_handle_, up_buf, norm_buf, lw.up_proj_w, 1, h, ffn, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, ffn, stream);
    cublas_linear_nobias(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, 1, ffn, h, stream);
    audio_ops::invoke_add_residual(hidden_states, norm_buf, h, stream);

    cudaStreamSynchronize(stream);
}

// ============================================================================
// forward_prefill: process full sequence, populate KV cache
// ============================================================================

void TextDecoder::forward_prefill(
    const __nv_bfloat16* input_embeds,
    const int* position_ids,
    int seq_len,
    __nv_bfloat16* logits_out,
    cudaStream_t stream)
{
    if (!initialized_) {
        fprintf(stderr, "[ASR Decoder] ERROR: not initialized\n");
        return;
    }

    if (seq_len > max_seq_len_) {
        fprintf(stderr, "[ASR Decoder] ERROR: seq_len=%d > max_seq_len=%d\n", seq_len, max_seq_len_);
        return;
    }

    int h = config_.decoder_hidden_size;
    int num_layers = config_.decoder_layers;

    // Copy input_embeds to workspace as hidden_states (prefill modifies in-place)
    __nv_bfloat16* hidden_states = workspace_;
    cudaMemcpyAsync(hidden_states, input_embeds,
                    (size_t)seq_len * h * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);

    // Workspace for layer computation starts after hidden_states
    __nv_bfloat16* layer_ws = hidden_states + (size_t)max_seq_len_ * h;

    // Process all decoder layers
    for (int layer = 0; layer < num_layers; layer++) {
        decoder_layer_forward_prefill(layer, hidden_states, position_ids,
                                       seq_len, layer_ws, stream);
    }

    // Final RMSNorm (on last token only for efficiency)
    __nv_bfloat16* last_hidden = hidden_states + (size_t)(seq_len - 1) * h;
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, last_hidden, final_norm_w_,
                               config_.rms_norm_eps, 1, h, stream);

    // LM head: [1, h] → [1, vocab_size]
    cublas_linear_nobias(cublas_handle_, logits_out, norm_out, lm_head_w_,
                          1, h, config_.vocab_size, stream);

    cache_seq_len_ = seq_len;
    cudaStreamSynchronize(stream);
}

// ============================================================================
// forward_decode: single token step
// ============================================================================

void TextDecoder::forward_decode(
    int token_id,
    const int* position_ids,
    __nv_bfloat16* logits_out,
    cudaStream_t stream)
{
    if (!initialized_) {
        fprintf(stderr, "[ASR Decoder] ERROR: not initialized\n");
        return;
    }

    if (cache_seq_len_ >= max_seq_len_) {
        fprintf(stderr, "[ASR Decoder] ERROR: KV cache full (%d/%d)\n",
                cache_seq_len_, max_seq_len_);
        return;
    }

    int h = config_.decoder_hidden_size;

    // Embed the token
    __nv_bfloat16* hidden_states = workspace_;
    *token_id_gpu_ = token_id;
    audio_ops::invoke_embedding_lookup(hidden_states, token_id_gpu_,
                                        embed_tokens_w_, 1, h, stream);

    // Workspace for layer computation
    __nv_bfloat16* layer_ws = hidden_states + h;

    // Process all decoder layers
    for (int layer = 0; layer < config_.decoder_layers; layer++) {
        decoder_layer_forward_decode(layer, hidden_states, position_ids,
                                      layer_ws, stream);
    }

    // Final RMSNorm
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, hidden_states, final_norm_w_,
                               config_.rms_norm_eps, 1, h, stream);

    // LM head
    cublas_linear_nobias(cublas_handle_, logits_out, norm_out, lm_head_w_,
                          1, h, config_.vocab_size, stream);

    cache_seq_len_++;
    cudaStreamSynchronize(stream);
}

} // namespace asr
} // namespace qwen_thor
