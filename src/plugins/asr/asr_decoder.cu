// asr_decoder.cu — Qwen3-ASR Text Decoder 实现
//
// 28-layer GQA decoder with MRoPE, per-head Q/K RMSNorm, SwiGLU MLP
// 支持 prefill (T>1) 和 decode (T=1) 两条路径
// Decode: Fused RMSNorm+QKV merged GEMV, 散列 GEMV, 融合 GEMV+Add

#include "asr_decoder.h"
#include "audio_ops.h"
#include "engine/dense_gemm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// 环境变量 ASR_PROFILE_DECODE=1 时启用单步详细计时 (第 5 步 decode)
static bool s_profile_decode = (getenv("ASR_PROFILE_DECODE") && atoi(getenv("ASR_PROFILE_DECODE")));

namespace qwen_thor {
namespace asr {

// ============================================================================
// BF16 linear: out = input @ weight^T (no bias for decoder)
// Prefill 路径 (M>1) 使用 cuBLAS GEMM; Decode 路径 (M=1) 使用主引擎散列 GEMV
// ============================================================================

static void linear_nobias(
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
    for (auto p : merged_allocations_) cudaFree(p);
    if (workspace_) cudaFree(workspace_);
    if (token_id_gpu_) cudaFree(token_id_gpu_);
    if (cublas_workspace_) cudaFree(cublas_workspace_);
    if (attn_split_k_ws_) cudaFree(attn_split_k_ws_);
    if (prefill_attn_score_buf_) cudaFree(prefill_attn_score_buf_);
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

    // Token ID for decode step (device memory + cudaMemcpy, no managed memory)
    cudaMalloc(&token_id_gpu_, sizeof(int));

    // Pre-allocate cuBLAS workspace (prevents internal cudaMalloc during decode)
    size_t cublas_ws_size = 4 * 1024 * 1024;  // 4 MB
    cudaMalloc(&cublas_workspace_, cublas_ws_size);
    cublasSetWorkspace(cublas_handle_, cublas_workspace_, cublas_ws_size);

    // Pre-allocate split-K attention workspace
    // partition_size=128, max_parts = ceil(max_seq_len / 128)
    int num_q_heads = config_.decoder_num_attention_heads;
    int head_dim_v = config_.decoder_head_dim;
    attn_max_partitions_ = (max_seq_len_ + 127) / 128;
    size_t attn_ws_size = (size_t)num_q_heads * attn_max_partitions_ * head_dim_v * sizeof(float)  // partial_out
                        + (size_t)num_q_heads * attn_max_partitions_ * sizeof(float)                // partial_m
                        + (size_t)num_q_heads * attn_max_partitions_ * sizeof(float);               // partial_l
    cudaMalloc(&attn_split_k_ws_, attn_ws_size);
    cudaMemset(attn_split_k_ws_, 0, attn_ws_size);

    // Prefill attention workspace: [max_seq, max_seq] BF16 for cuBLAS-based attention
    size_t prefill_attn_size = (size_t)max_seq_len_ * max_seq_len_ * sizeof(__nv_bfloat16);
    cudaMalloc(&prefill_attn_score_buf_, prefill_attn_size);
    float prefill_attn_mb = prefill_attn_size / (1024.0f * 1024.0f);

    cache_seq_len_ = 0;
    initialized_ = true;

    float kv_mb = (float)num_layers * kv_per_layer * 2 * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float ws_mb = workspace_size_ * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float attn_ws_kb = attn_ws_size / 1024.0f;
    fprintf(stderr, "[ASR Decoder] initialized: %d layers, max_seq=%d, KV cache %.1f MB, workspace %.1f MB, attn_ws %.1f KB (split-K %d parts), prefill_attn %.1f MB\n",
            num_layers, max_seq_len_, kv_mb, ws_mb, attn_ws_kb, attn_max_partitions_, prefill_attn_mb);
}

void TextDecoder::reset_cache() {
    cache_seq_len_ = 0;
}

// ============================================================================
// GPU kernel: subtract 1.0 from BF16 array (plain → centered RMSNorm weight)
// ============================================================================

__global__ void subtract_one_bf16_kernel(__nv_bfloat16* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = __float2bfloat16(__bfloat162float(data[i]) - 1.0f);
    }
}

// ============================================================================
// prepare_optimized_weights: QKV merge + RMSNorm centered transform
// 在所有权重加载完成后调用一次
// ============================================================================

void TextDecoder::prepare_optimized_weights(cudaStream_t stream) {
    int h = config_.decoder_hidden_size;
    int q_dim = config_.decoder_q_dim();
    int kv_dim = config_.decoder_kv_dim();
    int qkv_dim = q_dim + 2 * kv_dim;  // 4096

    for (int layer = 0; layer < config_.decoder_layers; layer++) {
        auto& lw = layer_weights_[layer];

        // 1. Merge QKV weights: [q_dim, h] + [kv_dim, h] + [kv_dim, h] → [qkv_dim, h]
        __nv_bfloat16* merged;
        cudaMalloc(&merged, (size_t)qkv_dim * h * sizeof(__nv_bfloat16));
        merged_allocations_.push_back(merged);

        cudaMemcpyAsync(merged,
                        lw.q_proj_w, (size_t)q_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(merged + (size_t)q_dim * h,
                        lw.k_proj_w, (size_t)kv_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(merged + (size_t)(q_dim + kv_dim) * h,
                        lw.v_proj_w, (size_t)kv_dim * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);

        lw.qkv_proj_w = merged;
        // Update individual pointers into merged buffer (prefill compatibility)
        lw.q_proj_w = merged;
        lw.k_proj_w = merged + (size_t)q_dim * h;
        lw.v_proj_w = merged + (size_t)(q_dim + kv_dim) * h;

        // 2. Create centered copy of input_layernorm_w for fused RMSNorm+GEMV decode
        {
            __nv_bfloat16* centered;
            cudaMalloc(&centered, h * sizeof(__nv_bfloat16));
            merged_allocations_.push_back(centered);
            cudaMemcpyAsync(centered, lw.input_layernorm_w,
                            h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            int blocks = (h + 255) / 256;
            subtract_one_bf16_kernel<<<blocks, 256, 0, stream>>>(centered, h);
            lw.input_layernorm_w_centered = centered;
        }

        // 3. Merge gate+up weights: [ffn, h] + [ffn, h] → [2*ffn, h]
        int ffn = config_.decoder_intermediate_size;
        {
            __nv_bfloat16* gu_merged;
            cudaMalloc(&gu_merged, (size_t)2 * ffn * h * sizeof(__nv_bfloat16));
            merged_allocations_.push_back(gu_merged);
            cudaMemcpyAsync(gu_merged,
                            lw.gate_proj_w, (size_t)ffn * h * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(gu_merged + (size_t)ffn * h,
                            lw.up_proj_w, (size_t)ffn * h * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream);
            lw.gateup_proj_w = gu_merged;
            lw.gate_proj_w = gu_merged;
            lw.up_proj_w = gu_merged + (size_t)ffn * h;
        }

        // 4. Create centered copy of post_attention_layernorm_w for fused RMSNorm+gateup decode
        {
            __nv_bfloat16* centered;
            cudaMalloc(&centered, h * sizeof(__nv_bfloat16));
            merged_allocations_.push_back(centered);
            cudaMemcpyAsync(centered, lw.post_attention_layernorm_w,
                            h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
            int blocks = (h + 255) / 256;
            subtract_one_bf16_kernel<<<blocks, 256, 0, stream>>>(centered, h);
            lw.post_attention_layernorm_w_centered = centered;
        }
    }

    // 5. Create centered copy of final_norm_w for fused LM head decode
    {
        __nv_bfloat16* centered;
        cudaMalloc(&centered, h * sizeof(__nv_bfloat16));
        merged_allocations_.push_back(centered);
        cudaMemcpyAsync(centered, final_norm_w_,
                        h * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream);
        int blocks = (h + 255) / 256;
        subtract_one_bf16_kernel<<<blocks, 256, 0, stream>>>(centered, h);
        final_norm_w_centered_ = centered;
    }

    cudaStreamSynchronize(stream);

    int ffn = config_.decoder_intermediate_size;
    float qkv_mb = (float)config_.decoder_layers * qkv_dim * h * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    float gu_mb = (float)config_.decoder_layers * 2 * ffn * h * sizeof(__nv_bfloat16) / (1024.0f * 1024.0f);
    fprintf(stderr, "[ASR Decoder] Optimized: %d layers QKV [%d,%d] (%.0f MB) + GateUp [%d,%d] (%.0f MB), norms centered\n",
            config_.decoder_layers, qkv_dim, h, qkv_mb, 2*ffn, h, gu_mb);
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
    linear_nobias(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
    linear_nobias(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
    linear_nobias(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);

    // 3-4. Fused QK RMSNorm + mRoPE: 3 kernels → 1 (matches decode path)
    audio_ops::invoke_fused_qk_norm_rope(
        q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
        position_ids, eps, seq_len, num_q_heads, num_kv_heads, head_dim,
        config_.mrope_section[0], config_.mrope_section[1],
        config_.mrope_section[2], config_.rope_theta, stream);

    // 5. Write K/V to cache
    audio_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                      k_buf, v_buf,
                                      0, seq_len,  // start_pos = 0 for prefill
                                      num_kv_heads, head_dim, stream);

    // 6. Causal GQA prefill attention (cuBLAS GEMM: QK^T + softmax + SV)
    audio_ops::invoke_causal_gqa_prefill_cublas(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        prefill_attn_score_buf_,
        seq_len, num_q_heads, num_kv_heads, head_dim,
        cublas_handle_, stream);

    // 7. Output projection: [seq_len, q_dim] → [seq_len, h]
    linear_nobias(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream);

    // 8. Residual add
    audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);

    // === MLP (SwiGLU) ===

    // 9. RMSNorm
    audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w,
                               eps, seq_len, h, stream);

    // 10. Gate projection: [seq_len, h] → [seq_len, ffn]
    int ffn = config_.decoder_intermediate_size;
    linear_nobias(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w, seq_len, h, ffn, stream);

    // 11. Up projection: [seq_len, h] → [seq_len, ffn]
    linear_nobias(cublas_handle_, up_buf, norm_buf, lw.up_proj_w, seq_len, h, ffn, stream);

    // 12. SwiGLU: gate = silu(gate) * up
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, ffn, stream);

    // 13. Down projection: [seq_len, ffn] → [seq_len, h]
    linear_nobias(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, seq_len, ffn, h, stream);

    // 13. Residual add
    audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream);
}

// ============================================================================
// decoder_layer_forward_decode: single layer, T=1
// 优化:
//  - Fused RMSNorm + QKV merged GEMV: 1 launch (was 4: norm + Q + K + V)
//  - Gate+Up: invoke_dense_dual_gemv (1 launch, 共享输入 SMEM)
//  - o_proj + residual: invoke_dense_gemv_add (融合 GEMV + add)
//  - down_proj + residual: invoke_dense_gemv_add (融合 GEMV + add)
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
    int qkv_dim = q_dim + 2 * kv_dim;  // 4096
    int num_q_heads = config_.decoder_num_attention_heads;
    int num_kv_heads = config_.decoder_num_kv_heads;
    int head_dim = config_.decoder_head_dim;
    float eps = config_.rms_norm_eps;

    // Workspace layout (T=1): QKV output is contiguous for merged GEMV
    __nv_bfloat16* qkv_buf  = workspace_base;                          // [qkv_dim=4096]
    __nv_bfloat16* q_buf    = qkv_buf;                                 // [q_dim=2048]
    __nv_bfloat16* k_buf    = qkv_buf + q_dim;                        // [kv_dim=1024]
    __nv_bfloat16* v_buf    = qkv_buf + q_dim + kv_dim;               // [kv_dim=1024]
    __nv_bfloat16* attn_out = qkv_buf + qkv_dim;                      // [h=2048]
    __nv_bfloat16* gateup_buf = attn_out + h;                          // [2*intermediate=12288]
    __nv_bfloat16* gate_buf = gateup_buf;                              // [intermediate=6144]
    __nv_bfloat16* up_buf   = gateup_buf + config_.decoder_intermediate_size; // [6144]

    // === Self-Attention ===

    // Fused RMSNorm + merged QKV GEMV: hidden_states → SMEM RMSNorm(centered) → GEMV → qkv_buf
    // 1 launch instead of 4 (RMSNorm + Q + K + V)
    ops::invoke_dense_gemv_with_rmsnorm(
        hidden_states, lw.input_layernorm_w_centered, eps,
        lw.qkv_proj_w, qkv_buf, qkv_dim, h, stream);

    // Fused QK RMSNorm + mRoPE: 1 launch instead of 3 (Q_norm + K_norm + mRoPE)
    audio_ops::invoke_fused_qk_norm_rope(
        q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
        position_ids, eps, 1, num_q_heads, num_kv_heads, head_dim,
        config_.mrope_section[0], config_.mrope_section[1],
        config_.mrope_section[2], config_.rope_theta, stream);

    // Write new K/V to cache
    audio_ops::invoke_write_kv_cache(k_cache_[layer_idx], v_cache_[layer_idx],
                                      k_buf, v_buf,
                                      cache_seq_len_, 1,
                                      num_kv_heads, head_dim, stream);

    // Decode attention: Q against full KV cache (split-K for high parallelism)
    audio_ops::invoke_causal_gqa_decode(
        attn_out, q_buf,
        k_cache_[layer_idx], v_cache_[layer_idx],
        1,  // batch_size=1
        num_q_heads, num_kv_heads, head_dim,
        cache_seq_len_ + 1,  // current total seq len including this token
        stream,
        attn_split_k_ws_, attn_max_partitions_);

    // o_proj + residual add: 融合 GEMV+Add (省 1 kernel + 1 GMEM write/read)
    ops::invoke_dense_gemv_add(attn_out, lw.o_proj_w, hidden_states, hidden_states,
                               h, q_dim, stream);

    // === MLP ===
    int ffn = config_.decoder_intermediate_size;
    // Fused RMSNorm + merged GateUp GEMV: hidden_states → SMEM RMSNorm(centered) → GEMV → gateup_buf
    // 1 launch instead of 3 (RMSNorm + gate_GEMV + up_GEMV)
    ops::invoke_dense_gemv_with_rmsnorm(
        hidden_states, lw.post_attention_layernorm_w_centered, eps,
        lw.gateup_proj_w, gateup_buf, 2 * ffn, h, stream);
    audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, ffn, stream);

    // down_proj + residual add: 融合 GEMV+Add
    ops::invoke_dense_gemv_add(gate_buf, lw.down_proj_w, hidden_states, hidden_states,
                               h, ffn, stream);
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

    // Profiling: measure per-category time over all layers (triggered once)
    static bool s_prefill_profiled = false;
    if (s_profile_decode && !s_prefill_profiled && seq_len > 100) {
        s_prefill_profiled = true;
        cudaStreamSynchronize(stream);

        // Re-run all layers with per-operation timing
        // Reset hidden_states for second pass
        cudaMemcpyAsync(hidden_states, input_embeds,
                        (size_t)seq_len * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);

        cudaEvent_t ev_start, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);

        // Time categories: RMSNorm, QKV_proj, QKnorm+RoPE, write_KV, Attention, O_proj, add_res1, RMSNorm2, Gate+Up_proj, SwiGLU, Down_proj, add_res2
        enum PfOp { PF_NORM1=0, PF_QKV, PF_QKNORM_ROPE, PF_WRITE_KV, PF_ATTN,
                     PF_OPROJ, PF_ADD1, PF_NORM2, PF_GATEUP, PF_SWIGLU, PF_DOWN, PF_ADD2, PF_COUNT };
        const char* pf_names[] = {"RMSNorm1","QKV_proj","QKnorm+RoPE","write_KV","Attention",
                                    "O_proj","add_res1","RMSNorm2","Gate+Up_proj","SwiGLU","Down_proj","add_res2"};
        float pf_ms[PF_COUNT] = {};

        for (int layer = 0; layer < num_layers; layer++) {
            const auto& lw = layer_weights_[layer];
            int q_dim = config_.decoder_q_dim();
            int kv_dim = config_.decoder_kv_dim();
            int num_q_heads = config_.decoder_num_attention_heads;
            int num_kv_heads = config_.decoder_num_kv_heads;
            int head_dim = config_.decoder_head_dim;
            float eps = config_.rms_norm_eps;
            int ffn = config_.decoder_intermediate_size;

            __nv_bfloat16* norm_buf  = layer_ws;
            __nv_bfloat16* q_buf     = norm_buf + (size_t)seq_len * h;
            __nv_bfloat16* k_buf     = q_buf + (size_t)seq_len * q_dim;
            __nv_bfloat16* v_buf     = k_buf + (size_t)seq_len * kv_dim;
            __nv_bfloat16* attn_out  = v_buf + (size_t)seq_len * kv_dim;
            __nv_bfloat16* gate_buf  = attn_out + (size_t)seq_len * h;
            __nv_bfloat16* up_buf    = gate_buf + (size_t)seq_len * ffn;

            auto time_op = [&](PfOp op, auto fn) {
                cudaEventRecord(ev_start, stream);
                fn();
                cudaEventRecord(ev_end, stream);
                cudaEventSynchronize(ev_end);
                float ms = 0;
                cudaEventElapsedTime(&ms, ev_start, ev_end);
                pf_ms[op] += ms;
            };

            time_op(PF_NORM1, [&]{ audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.input_layernorm_w, eps, seq_len, h, stream); });
            time_op(PF_QKV, [&]{
                linear_nobias(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, seq_len, h, q_dim, stream);
                linear_nobias(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, seq_len, h, kv_dim, stream);
                linear_nobias(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, seq_len, h, kv_dim, stream);
            });
            time_op(PF_QKNORM_ROPE, [&]{
                audio_ops::invoke_fused_qk_norm_rope(
                    q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
                    position_ids, eps, seq_len, num_q_heads, num_kv_heads, head_dim,
                    config_.mrope_section[0], config_.mrope_section[1], config_.mrope_section[2], config_.rope_theta, stream);
            });
            time_op(PF_WRITE_KV, [&]{ audio_ops::invoke_write_kv_cache(k_cache_[layer], v_cache_[layer], k_buf, v_buf, 0, seq_len, num_kv_heads, head_dim, stream); });
            time_op(PF_ATTN, [&]{ audio_ops::invoke_causal_gqa_prefill_cublas(attn_out, q_buf, k_cache_[layer], v_cache_[layer], prefill_attn_score_buf_, seq_len, num_q_heads, num_kv_heads, head_dim, cublas_handle_, stream); });
            time_op(PF_OPROJ, [&]{ linear_nobias(cublas_handle_, norm_buf, attn_out, lw.o_proj_w, seq_len, q_dim, h, stream); });
            time_op(PF_ADD1, [&]{ audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream); });
            time_op(PF_NORM2, [&]{ audio_ops::invoke_rmsnorm(norm_buf, hidden_states, lw.post_attention_layernorm_w, eps, seq_len, h, stream); });
            time_op(PF_GATEUP, [&]{
                linear_nobias(cublas_handle_, gate_buf, norm_buf, lw.gate_proj_w, seq_len, h, ffn, stream);
                linear_nobias(cublas_handle_, up_buf, norm_buf, lw.up_proj_w, seq_len, h, ffn, stream);
            });
            time_op(PF_SWIGLU, [&]{ audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, seq_len, ffn, stream); });
            time_op(PF_DOWN, [&]{
                linear_nobias(cublas_handle_, norm_buf, gate_buf, lw.down_proj_w, seq_len, ffn, h, stream);
            });
            time_op(PF_ADD2, [&]{ audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * h, stream); });
        }

        float total = 0;
        for (int i = 0; i < PF_COUNT; i++) total += pf_ms[i];
        fprintf(stderr, "[ASR PREFILL PROFILE] seq_len=%d, %d layers:\n", seq_len, num_layers);
        for (int i = 0; i < PF_COUNT; i++)
            fprintf(stderr, "  %-16s %7.1f ms (%5.1f%%)\n", pf_names[i], pf_ms[i], 100.0f * pf_ms[i] / total);
        fprintf(stderr, "  TOTAL           %7.1f ms\n", total);

        // Compute GEMM GFLOPS
        int q_dim = config_.decoder_q_dim(), kv_dim = config_.decoder_kv_dim();
        float qkv_gflops = 2.0f * seq_len * h * (q_dim + 2*kv_dim) / 1e9f;
        float o_gflops = 2.0f * seq_len * q_dim * h / 1e9f;
        float gateup_gflops = 2.0f * seq_len * h * 2 * config_.decoder_intermediate_size / 1e9f;
        float down_gflops = 2.0f * seq_len * config_.decoder_intermediate_size * h / 1e9f;
        fprintf(stderr, "  GEMM TFLOPS: QKV %.2f  O %.2f  GateUp %.2f  Down %.2f\n",
                qkv_gflops * num_layers / pf_ms[PF_QKV], o_gflops * num_layers / pf_ms[PF_OPROJ],
                gateup_gflops * num_layers / pf_ms[PF_GATEUP], down_gflops * num_layers / pf_ms[PF_DOWN]);

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);

        // Reset KV cache and re-run correctly for actual inference (profiling 2nd pass corrupted caches)
        for (int layer = 0; layer < num_layers; layer++) {
            size_t kv_per_layer = (size_t)max_seq_len_ * config_.decoder_num_kv_heads * config_.decoder_head_dim;
            cudaMemsetAsync(k_cache_[layer], 0, kv_per_layer * sizeof(__nv_bfloat16), stream);
            cudaMemsetAsync(v_cache_[layer], 0, kv_per_layer * sizeof(__nv_bfloat16), stream);
        }
        cudaMemcpyAsync(hidden_states, input_embeds,
                        (size_t)seq_len * h * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        for (int layer = 0; layer < num_layers; layer++) {
            decoder_layer_forward_prefill(layer, hidden_states, position_ids,
                                           seq_len, layer_ws, stream);
        }
    }

    // Final RMSNorm (on last token only for efficiency)
    __nv_bfloat16* last_hidden = hidden_states + (size_t)(seq_len - 1) * h;
    __nv_bfloat16* norm_out = layer_ws;
    audio_ops::invoke_rmsnorm(norm_out, last_hidden, final_norm_w_,
                               config_.rms_norm_eps, 1, h, stream);

    // LM head: [1, h] → [1, vocab_size]
    linear_nobias(cublas_handle_, logits_out, norm_out, lm_head_w_,
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

    // Profile mode: detailed per-operation breakdown (5th decode step overall)
    static int s_decode_step_counter = 0;
    s_decode_step_counter++;
    bool do_profile = s_profile_decode && (s_decode_step_counter == 5);

    // Embed the token (cudaMemcpyAsync avoids managed memory ATS faults)
    __nv_bfloat16* hidden_states = workspace_;
    cudaMemcpyAsync(token_id_gpu_, &token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
    audio_ops::invoke_embedding_lookup(hidden_states, token_id_gpu_,
                                        embed_tokens_w_, 1, h, stream);

    // Workspace for layer computation
    __nv_bfloat16* layer_ws = hidden_states + h;

    if (!do_profile) {
        // Fast path: no profiling
        for (int layer = 0; layer < config_.decoder_layers; layer++) {
            decoder_layer_forward_decode(layer, hidden_states, position_ids,
                                          layer_ws, stream);
        }
        ops::invoke_dense_gemv_with_rmsnorm(
            hidden_states, final_norm_w_centered_, config_.rms_norm_eps,
            lm_head_w_, logits_out, config_.vocab_size, h, stream);
        cache_seq_len_++;
    } else {
        // Detailed per-operation profiling: inline decoder_layer_forward_decode
        // to measure each operation across all 28 layers
        enum OpIdx { OP_QKV=0, OP_QKNORM_ROPE, OP_WRITE_KV,
                     OP_ATTN, OP_OPROJ, OP_GATEUP, OP_SWIGLU, OP_DOWN, OP_COUNT };
        const char* op_names[] = {"Fused_RMSNorm+QKV", "Fused_QKnorm+RoPE",
                                   "write_KV", "GQA_decode", "o_proj+add",
                                   "Fused_RMSNorm+GateUp", "SwiGLU", "down+add"};
        float op_ms[OP_COUNT] = {};

        // Create 2 reusable events
        cudaEvent_t ev_a, ev_b;
        cudaEventCreate(&ev_a);
        cudaEventCreate(&ev_b);

        cudaEventRecord(ev_a, stream);  // start of layers

        int num_layers = config_.decoder_layers;
        for (int layer = 0; layer < num_layers; layer++) {
            const auto& lw = layer_weights_[layer];
            int q_dim = config_.decoder_q_dim();
            int kv_dim = config_.decoder_kv_dim();
            int qkv_dim = q_dim + 2 * kv_dim;
            int num_q_heads = config_.decoder_num_attention_heads;
            int num_kv_heads = config_.decoder_num_kv_heads;
            int head_dim = config_.decoder_head_dim;
            float eps = config_.rms_norm_eps;
            int ffn = config_.decoder_intermediate_size;

            __nv_bfloat16* qkv_buf = layer_ws;
            __nv_bfloat16* q_buf = qkv_buf;
            __nv_bfloat16* k_buf = qkv_buf + q_dim;
            __nv_bfloat16* v_buf = qkv_buf + q_dim + kv_dim;
            __nv_bfloat16* attn_out = qkv_buf + qkv_dim;
            __nv_bfloat16* gateup_buf = attn_out + h;
            __nv_bfloat16* gate_buf = gateup_buf;
            __nv_bfloat16* up_buf = gateup_buf + ffn;

            auto record_op = [&](int op) {
                cudaEventRecord(ev_b, stream);
                cudaEventSynchronize(ev_b);
                float ms;
                cudaEventElapsedTime(&ms, ev_a, ev_b);
                op_ms[op] += ms;
                cudaEventRecord(ev_a, stream);
            };

            // OP_QKV: Fused RMSNorm + merged QKV GEMV
            ops::invoke_dense_gemv_with_rmsnorm(
                hidden_states, lw.input_layernorm_w_centered, eps,
                lw.qkv_proj_w, qkv_buf, qkv_dim, h, stream);
            record_op(OP_QKV);

            // OP_QKNORM_ROPE: Fused QK RMSNorm + mRoPE
            audio_ops::invoke_fused_qk_norm_rope(
                q_buf, k_buf, lw.q_norm_w, lw.k_norm_w,
                position_ids, eps, 1, num_q_heads, num_kv_heads, head_dim,
                config_.mrope_section[0], config_.mrope_section[1],
                config_.mrope_section[2], config_.rope_theta, stream);
            record_op(OP_QKNORM_ROPE);

            // OP_WRITE_KV
            audio_ops::invoke_write_kv_cache(k_cache_[layer], v_cache_[layer],
                                              k_buf, v_buf,
                                              cache_seq_len_, 1,
                                              num_kv_heads, head_dim, stream);
            record_op(OP_WRITE_KV);

            // OP_ATTN
            audio_ops::invoke_causal_gqa_decode(
                attn_out, q_buf,
                k_cache_[layer], v_cache_[layer],
                1, num_q_heads, num_kv_heads, head_dim,
                cache_seq_len_ + 1, stream,
                attn_split_k_ws_, attn_max_partitions_);
            record_op(OP_ATTN);

            // OP_OPROJ
            ops::invoke_dense_gemv_add(attn_out, lw.o_proj_w, hidden_states, hidden_states,
                                       h, q_dim, stream);
            record_op(OP_OPROJ);

            // OP_GATEUP
            ops::invoke_dense_gemv_with_rmsnorm(
                hidden_states, lw.post_attention_layernorm_w_centered, eps,
                lw.gateup_proj_w, gateup_buf, 2 * ffn, h, stream);
            record_op(OP_GATEUP);

            // OP_SWIGLU
            audio_ops::invoke_swiglu(gate_buf, gate_buf, up_buf, 1, ffn, stream);
            record_op(OP_SWIGLU);

            // OP_DOWN
            ops::invoke_dense_gemv_add(gate_buf, lw.down_proj_w, hidden_states, hidden_states,
                                       h, ffn, stream);
            record_op(OP_DOWN);
        }

        // LM head
        cudaEvent_t ev_lm;
        cudaEventCreate(&ev_lm);
        cudaEventRecord(ev_a, stream);

        ops::invoke_dense_gemv_with_rmsnorm(
            hidden_states, final_norm_w_centered_, config_.rms_norm_eps,
            lm_head_w_, logits_out, config_.vocab_size, h, stream);

        cudaEventRecord(ev_lm, stream);
        cudaEventSynchronize(ev_lm);
        float ms_lmhead;
        cudaEventElapsedTime(&ms_lmhead, ev_a, ev_lm);

        cache_seq_len_++;

        // Print detailed breakdown
        float total_ops = 0;
        for (int i = 0; i < OP_COUNT; i++) total_ops += op_ms[i];
        float total_all = total_ops + ms_lmhead;

        fprintf(stderr, "[ASR PROFILE] step=%d seq_len=%d (per-op breakdown, %d layers)\n",
                cache_seq_len_ - 1, cache_seq_len_, config_.decoder_layers);
        for (int i = 0; i < OP_COUNT; i++) {
            fprintf(stderr, "  %-22s %6.2f ms (%4.1f%%) [%.3f ms/layer]\n",
                    op_names[i], op_ms[i], op_ms[i] / total_all * 100.0f,
                    op_ms[i] / config_.decoder_layers);
        }
        fprintf(stderr, "  %-22s %6.2f ms (%4.1f%%)\n", "LM_head", ms_lmhead,
                ms_lmhead / total_all * 100.0f);
        fprintf(stderr, "  %-22s %6.2f ms\n", "TOTAL", total_all);

        // Weight MB breakdown
        float qkv_mb = (config_.decoder_q_dim() + 2*config_.decoder_kv_dim()) * h * 2.0f / (1024*1024);
        float o_mb = config_.decoder_q_dim() * h * 2.0f / (1024*1024);
        float gu_mb = 2 * config_.decoder_intermediate_size * h * 2.0f / (1024*1024);
        float dn_mb = config_.decoder_intermediate_size * h * 2.0f / (1024*1024);
        float lm_mb = config_.vocab_size * h * 2.0f / (1024*1024);
        fprintf(stderr, "  GEMV BW: QKV %.0f GB/s  o_proj %.0f GB/s  GateUp %.0f GB/s  down %.0f GB/s  LM %.0f GB/s\n",
                qkv_mb * config_.decoder_layers / op_ms[OP_QKV],
                o_mb * config_.decoder_layers / op_ms[OP_OPROJ],
                gu_mb * config_.decoder_layers / op_ms[OP_GATEUP],
                dn_mb * config_.decoder_layers / op_ms[OP_DOWN],
                lm_mb / ms_lmhead);

        s_profile_decode = false;
        cudaEventDestroy(ev_a);
        cudaEventDestroy(ev_b);
        cudaEventDestroy(ev_lm);
    }
}

} // namespace asr
} // namespace qwen_thor
