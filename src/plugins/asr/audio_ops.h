// audio_ops.h — 音频独立 CUDA 算子库
//
// ASR/TTS 专用算子, 不复用 LLM light_ops.cu, 各自独立演进。
// 即使数学上相同 (RMSNorm, SwiGLU 等) 也独立实现, 原因:
//   1. 参数/行为差异大 (三种 Norm, 三种 RoPE, bias 有无)
//   2. 音频模型与 LLM 优化方向不同 (小 hidden, 短序列, MHA)
//   3. 避免改一处破两处
//
// 包含:
//   基础: RMSNorm(plain), LayerNorm(bias), SwiGLU, GELU
//         GQA attention, MHA bidirectional, MRoPE, 1D RoPE, sinusoidal PE
//         embedding lookup, add_residual
//   音频专用: (Phase 2 再加) SnakeBeta, CausalConv1d, CausalTransConv1d, etc.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace audio_ops {

// ============================================================================
// 归一化算子
// ============================================================================

// RMSNorm (plain weight): y = w * x * rsqrt(mean(x²) + eps)
// 注: ASR Decoder / TTS Talker / Predictor / Tokenizer 使用 plain weight
//     与 LLM 的 centered weight (1+w) 不同!
void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    float eps, int num_tokens, int hidden_size, cudaStream_t stream = 0);

// LayerNorm (标准, 含 bias): y = (x - mean) / sqrt(var + eps) * w + b
// ASR Encoder 全部使用 LayerNorm (不是 RMSNorm!)
void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size, cudaStream_t stream = 0);

// Per-head RMSNorm (plain weight): 用于 Q/K norm
// 每个 head 独立做 RMSNorm
void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream = 0);

// Fused per-head QK RMSNorm + MRoPE (3 kernels → 1)
// Combines: per_head_rmsnorm(Q) + per_head_rmsnorm(K) + mrope in single launch.
// Grid: (num_q_heads + num_kv_heads) × num_tokens blocks.
void invoke_fused_qk_norm_rope(
    __nv_bfloat16* q, __nv_bfloat16* k,
    const __nv_bfloat16* q_norm_w,     // [head_dim]
    const __nv_bfloat16* k_norm_w,     // [head_dim]
    const int* pos_ids,                // [3, num_tokens]
    float eps,
    int num_tokens,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    int s0, int s1, int s2,
    float theta,
    cudaStream_t stream = 0);

// ============================================================================
// 激活函数
// ============================================================================

// SwiGLU: out = silu(gate) * up (element-wise)
// ASR Decoder MLP, TTS Talker MLP
void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   int num_tokens, int intermediate_size, cudaStream_t stream = 0);

// GELU: out = x * 0.5 * (1 + erf(x / sqrt(2)))
// ASR Encoder FFN + projection
void invoke_gelu(__nv_bfloat16* out, const __nv_bfloat16* x,
                 int num_elements, cudaStream_t stream = 0);

// ============================================================================
// 位置编码
// ============================================================================

// Sinusoidal Positional Embedding (Whisper 风格)
// 预计算 [max_positions, d_model] 的 sin/cos 表
// 存入 BF16 buffer (GPU 端)
void compute_sinusoidal_pe(__nv_bfloat16* pe_out,
                           int max_positions, int d_model,
                           float max_timescale = 10000.0f,
                           cudaStream_t stream = 0);

// 加法位置编码: hidden += pe[pos_offset : pos_offset + seq_len]
void invoke_add_pe(__nv_bfloat16* hidden_states,
                   const __nv_bfloat16* pe_table,
                   int seq_len, int hidden_size,
                   int pos_offset = 0,
                   cudaStream_t stream = 0);

// Per-chunk PE: each chunk independently uses PE[0..chunk_len-1]
// For total_tokens = num_chunks * chunk_len, token t uses PE[t % chunk_len]
void invoke_add_pe_chunked(__nv_bfloat16* hidden_states,
                           const __nv_bfloat16* pe_table,
                           int total_tokens, int hidden_size,
                           int chunk_len,
                           cudaStream_t stream = 0);

// MRoPE (Multimodal Rotary Position Embedding)
// 半旋转 (d, d+D/2), interleaved section assignment
// sections=[s0, s1, s2] 决定每个 freq pair 使用哪个 position 维度:
//   d%3==0 或 d≥min(s1,s2)*3 → dim T, d%3==1 且 d<s1*3 → dim H, d%3==2 且 d<s2*3 → dim W
// 频率: 全局 inv_freq[d] = 1/theta^(2d/head_dim)
// pos_ids 布局: [3, num_tokens] (T, H, W)
// 用于 ASR Decoder / TTS Talker
void invoke_mrope(__nv_bfloat16* q, __nv_bfloat16* k,
                  const int* pos_ids,         // [3 * num_tokens]
                  int num_tokens,
                  int num_q_heads, int num_kv_heads,
                  int head_dim,
                  int s0, int s1, int s2,     // sections: e.g. 24, 20, 20
                  float theta = 1000000.0f,
                  cudaStream_t stream = 0);

// 标准 1D RoPE (半旋转)
// 用于 Code Predictor / Tokenizer Transformer
void invoke_rope_1d(__nv_bfloat16* q, __nv_bfloat16* k,
                    const int* pos_ids,
                    int num_tokens,
                    int num_q_heads, int num_kv_heads,
                    int head_dim,
                    float theta = 10000.0f,
                    cudaStream_t stream = 0);

// ============================================================================
// Attention
// ============================================================================

// Bidirectional Multi-Head Attention (ASR Encoder)
// 非因果, 全 token 间 attend, 支持 cu_seqlens 分段
// Q/K/V 已经投影好, 形状 [total_tokens, num_heads, head_dim]
// 输出 attn_out [total_tokens, num_heads, head_dim]
//
// cu_seqlens: 分段边界, [num_segments + 1], e.g. [0, 800, 1600, 1900]
//   segment 内双向 attend, segment 间不 attend
void invoke_bidirectional_mha(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    int total_tokens,
    int num_heads, int head_dim,
    const int* cu_seqlens,       // [num_segments + 1]
    int num_segments,
    cudaStream_t stream = 0);

// Causal GQA Decode Attention (单步 decode, T=1)
// 用于 ASR Decoder / TTS Talker decode 阶段
// q: [batch_size, num_q_heads, head_dim]
// k_cache, v_cache: [max_seq, num_kv_heads, head_dim]
// 输出: [batch_size, num_q_heads, head_dim]
void invoke_causal_gqa_decode(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int batch_size,
    int num_q_heads, int num_kv_heads, int head_dim,
    int current_seq_len,
    cudaStream_t stream = 0,
    float* attn_workspace = nullptr,
    int attn_max_partitions = 0);

// Causal GQA Prefill Attention (T > 1)
// 因果 mask, 单请求
// q/k/v: [seq_len, num_heads/num_kv_heads, head_dim]
void invoke_causal_gqa_prefill(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream = 0);

// ============================================================================
// 其他
// ============================================================================

// Embedding lookup: out[i] = table[ids[i]]
void invoke_embedding_lookup(__nv_bfloat16* out, const int* ids,
                              const __nv_bfloat16* table,
                              int num_tokens, int hidden_size,
                              cudaStream_t stream = 0);

// Residual add: out = a + b (in-place: a += b)
void invoke_add_residual(__nv_bfloat16* a, const __nv_bfloat16* b,
                         int num_elements, cudaStream_t stream = 0);

// Float16 clamp (ASR Encoder 防溢出): clamp(x, -65504, 65504)
void invoke_bf16_clamp(__nv_bfloat16* x, int num_elements,
                       float min_val, float max_val, cudaStream_t stream = 0);

// Write KV cache (simple contiguous, non-paged)
// ASR decoder 单请求, 不需要 paged KV cache
// k/v: [num_tokens, num_kv_heads, head_dim]
// k_cache/v_cache: [max_seq_len, num_kv_heads, head_dim]
void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            int start_pos, int num_tokens,
                            int num_kv_heads, int head_dim,
                            cudaStream_t stream = 0);

// GPU Argmax: 在 GPU 上计算 BF16 logits 的 argmax
// 结果写到 result_idx (device 或 managed memory)
void invoke_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                   cudaStream_t stream = 0);

// EOS 抑制: 将指定 token 的 logits 设为 -inf, 防止模型提前终止
// 用于长音频中语音停顿导致的 EOS 误判
void invoke_suppress_eos(__nv_bfloat16* logits, int eos_id1, int eos_id2,
                         cudaStream_t stream = 0);

// BF16 转换: float32 → BF16 (GPU kernel, 避免 CPU 逐元素转换 + H2D 拷贝)
void invoke_f32_to_bf16(__nv_bfloat16* out, const float* in, int n,
                        cudaStream_t stream = 0);

} // namespace audio_ops
} // namespace qwen_thor
