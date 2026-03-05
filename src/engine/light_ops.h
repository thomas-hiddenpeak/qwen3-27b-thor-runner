#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace ops {

// RMSNorm 算子
void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    float eps, int num_tokens, int hidden_size, cudaStream_t stream = 0);

// RoPE (Rotary Positional Embedding) 算子
void invoke_rope(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids,
                 int num_tokens, int num_heads, int num_kv_heads, int head_dim,
                 float base = 10000.0f, cudaStream_t stream = 0);

// SwiGLU 激活函数
void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   int num_tokens, int intermediate_size, cudaStream_t stream = 0);

// Embedding Lookup 算子
void invoke_embedding_lookup(__nv_bfloat16* out, const int* tokens,
                              const __nv_bfloat16* embedding_table,
                              int num_tokens, int hidden_size, cudaStream_t stream = 0);

// 局部 RoPE：只旋转每个 head 前 rotary_dim 维，其余维度不变
void invoke_rope_partial(__nv_bfloat16* q, __nv_bfloat16* k,
                         const int* pos_ids,
                         int num_tokens,
                         int num_q_heads, int num_kv_heads,
                         int head_dim, int rotary_dim,
                         float base = 10000000.0f,
                         cudaStream_t stream = 0);

// Per-head RMSNorm（用于 q_norm / k_norm）
// centered=true → uses (1+w), centered=false → uses w
void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream = 0, bool centered = false);

// 写入 Paged KV Cache (支持 batched decode)
// batch_size=1 → prefill (所有 token 同一序列, 用 start_pos)
// batch_size>1 → batched decode (每 token 不同序列, 用 seq_positions)
void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            const int* block_tables,
                            int start_pos, int num_tokens,
                            int num_kv_heads, int head_dim,
                            int block_size, int max_num_blocks_per_seq,
                            cudaStream_t stream = 0, int batch_size = 1,
                            const int* seq_positions = nullptr);

// 因果短卷积 (causal conv1d)
// token_stride: 元素步长（= channels 独立布局; = in_qkv 交叉布局）
// batch_size>1 + conv_state_ptrs: batched decode, 每 token 用自己的 conv state
// conv_state_checkpoint: if non-null, save conv state after processing token[0]
// (for MTP speculative decode rollback on T=2 verify)
void invoke_causal_conv1d(__nv_bfloat16* x_io, __nv_bfloat16* conv_state,
                           const __nv_bfloat16* conv_w,
                           int num_tokens, int channels, int conv_k,
                           cudaStream_t stream = 0, int token_stride = 0,
                           int batch_size = 1, __nv_bfloat16** conv_state_ptrs = nullptr,
                           __nv_bfloat16* conv_state_checkpoint = nullptr);

// FP32 → BF16 转换（加载权重时使用）
void invoke_f32_to_bf16(const float* src, __nv_bfloat16* dst, size_t n,
                         cudaStream_t stream = 0);

// Gated DeltaNet 状态更新 (with fused alpha/sigmoid-beta computation)
// a_raw/beta_raw: raw projection outputs (BF16), alpha and sigmoid(beta) computed inline
// dt_bias: [nv] per-head bias for softplus, A_log: [nv] log decay rate (FP32)
// token_stride: q/k/v 中相邻 token 的步长（= nkh*kd 独立; = in_qkv 交叉）
// batch_size>1 + ssm_state_ptrs: batched decode, 每 token 用自己的 SSM state
// ssm_state_checkpoint: if non-null, save SSM state after processing token[0]
// (for MTP speculative decode rollback on T=2 verify)
void invoke_gated_delta_net(const __nv_bfloat16* q, const __nv_bfloat16* k,
                             const __nv_bfloat16* v,
                             const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias,
                             const float* A_log, const __nv_bfloat16* beta_raw,
                             __nv_bfloat16* ssm_state, __nv_bfloat16* y_out,
                             int num_tokens,
                             int nkh, int kd, int nv_per_kh, int vd,
                             cudaStream_t stream = 0, int token_stride = 0,
                             int batch_size = 1, __nv_bfloat16** ssm_state_ptrs = nullptr,
                             __nv_bfloat16* ssm_state_checkpoint = nullptr);

// element-wise: out[i] = a[i] + b[i]
void invoke_add(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                int n, cudaStream_t stream = 0);

// Fused Add + RMSNorm: residual += bias, then RMSNorm(residual) → norm_out
// Uses centered weight: out = x * rsqrt(var+eps) * (1+w)
void invoke_fused_add_rmsnorm(__nv_bfloat16* norm_out, __nv_bfloat16* residual,
                               const __nv_bfloat16* bias, const __nv_bfloat16* weight,
                               float eps, int num_tokens, int hidden_size,
                               cudaStream_t stream = 0);

// element-wise: out[i] = a[i] * sigmoid(b[i])
void invoke_sigmoid_mul(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                        int n, cudaStream_t stream = 0);

// Deinterleave Q+Gate from [T, num_heads, 2*hd] to Q [T, num_heads, hd] and Gate [T, num_heads*hd]
void invoke_deinterleave_qgate(__nv_bfloat16* q_out, __nv_bfloat16* gate_out,
                                const __nv_bfloat16* qg_in,
                                int num_tokens, int num_heads, int head_dim,
                                cudaStream_t stream = 0);

// Fused Deinterleave Q+Gate + Per-head RMSNorm on Q
// Input: qg_in [T, num_heads, 2*hd], Output: normalized Q [T, num_heads, hd] + Gate
// Uses centered weight: out = x * rsqrt(var+eps) * (1+w)
void invoke_fused_deinterleave_q_rmsnorm(
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out,
    const __nv_bfloat16* qg_in, const __nv_bfloat16* q_norm_weight,
    float eps, int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream = 0);

// Fused Deinterleave+Q_norm + K_norm + RoPE: 3 kernels → 1
// Grid: (T, num_q + num_kv), Block: min(head_dim, 256)
// Q blocks: deinterleave qg_in → q+gate, RMSNorm Q, RoPE first rotary_dim dims
// K blocks: RMSNorm K (centered), RoPE first rotary_dim dims
// 节省 2 launches/层 × 16 FullAttn 层 = 32 launches/step
void invoke_fused_qk_norm_rope(
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out,
    const __nv_bfloat16* qg_in, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const int* pos_ids,
    float eps, int num_tokens, int num_q, int num_kv,
    int head_dim, int rotary_dim, float rope_base,
    cudaStream_t stream = 0);

// GPU Argmax: 在 GPU 上计算 argmax，结果写到 result_idx（device 或 managed 内存）
void invoke_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                   cudaStream_t stream = 0);

// Batched GPU Argmax: 单次 launch 处理 batch_size 个序列的 argmax
// logits: [batch_size, n], result_idx: [batch_size]
void invoke_batched_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                           int batch_size, cudaStream_t stream = 0);

// Fused per-head RMSNorm (plain weight) + SiLU Gate
// y_ssm = rmsnorm(y_ssm, weight) * silu(z_out)
// For LinearAttn steps 7+8
void invoke_fused_norm_silu_gate(
    __nv_bfloat16* y_ssm,
    const __nv_bfloat16* z_out,
    const __nv_bfloat16* weight,
    float eps, int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream = 0);

// Deinterleave merged GEMM output [T, N_total] RowMajor into separate contiguous arrays
// Splits: [0, s1), [s1, s2), [s2, N_total)
// All dimensions must be multiples of 8 for vectorized access
void invoke_deinterleave_gemm_3way(
    __nv_bfloat16* out1, __nv_bfloat16* out2, __nv_bfloat16* out3,
    const __nv_bfloat16* merged, int num_tokens,
    int N_total, int split1, int split2,
    cudaStream_t stream = 0);

// Fused SwiGLU for merged gate+up layout
// Input: merged_gateup [T, 2*is] RowMajor (gate at cols [0, is), up at cols [is, 2*is))
// Output: out [T, is] = silu(gate) * up
void invoke_swiglu_merged(__nv_bfloat16* out, const __nv_bfloat16* merged_gateup,
                           int num_tokens, int intermediate_size, cudaStream_t stream = 0);

} // namespace ops
} // namespace qwen_thor
