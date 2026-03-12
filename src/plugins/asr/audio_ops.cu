// audio_ops.cu — 音频独立 CUDA 算子库实现
//
// ASR/TTS 专用算子, 不复用 LLM light_ops.cu
// Phase 1: 基础算子 (RMSNorm, LayerNorm, SwiGLU, GELU, PE, RoPE, MHA, GQA, embedding, residual)

#include "audio_ops.h"
#include <cmath>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace audio_ops {

// ============================================================================
// Helper: BF16 ↔ float conversion
// ============================================================================

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// RMSNorm (plain weight): y = w * x * rsqrt(mean(x²) + eps)
// ============================================================================

__global__ void rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int hidden_size)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (size_t)token * hidden_size;
    __nv_bfloat16* o_row = out + (size_t)token * hidden_size;

    // Compute sum of squares using warp reduction
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    // Block reduction via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();

    if (threadIdx.x < blockDim.x / warpSize) {
        sum_sq = shared[threadIdx.x];
    } else {
        sum_sq = 0.0f;
    }
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) {
        s_rsqrt = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    float scale = s_rsqrt;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) * scale;
        float w = bf16_to_float(weight[i]);
        o_row[i] = float_to_bf16(w * v);  // plain weight, NOT (1+w)
    }
}

void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                    float eps, int num_tokens, int hidden_size, cudaStream_t stream) {
    int block = std::min(hidden_size, 1024);
    rmsnorm_kernel<<<num_tokens, block, 0, stream>>>(out, x, weight, eps, hidden_size);
}

// ============================================================================
// LayerNorm (with bias): y = (x - mean) / sqrt(var + eps) * w + b
// ============================================================================

__global__ void layernorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float eps, int hidden_size)
{
    int token = blockIdx.x;
    const __nv_bfloat16* x_row = x + (size_t)token * hidden_size;
    __nv_bfloat16* o_row = out + (size_t)token * hidden_size;

    // Two-pass: compute mean, then variance
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += bf16_to_float(x_row[i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) sum = shared[threadIdx.x];
    else sum = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / hidden_size;
    __syncthreads();

    float mean = s_mean;
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = bf16_to_float(x_row[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane == 0) shared[wid] = var_sum;
    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) var_sum = shared[threadIdx.x];
    else var_sum = 0.0f;
    if (wid == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) s_inv_std = rsqrtf(var_sum / hidden_size + eps);
    __syncthreads();

    float inv_std = s_inv_std;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (bf16_to_float(x_row[i]) - mean) * inv_std;
        float w = bf16_to_float(weight[i]);
        float b = bf16_to_float(bias[i]);
        o_row[i] = float_to_bf16(v * w + b);
    }
}

void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size, cudaStream_t stream) {
    int block = std::min(hidden_size, 1024);
    layernorm_kernel<<<num_tokens, block, 0, stream>>>(out, x, weight, bias, eps, hidden_size);
}

// ============================================================================
// Per-head RMSNorm (plain weight)
// ============================================================================

__global__ void per_head_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float eps, int num_tokens, int num_heads, int head_dim)
{
    // One block per (token, head)
    int token_head = blockIdx.x;
    int token = token_head / num_heads;
    int head = token_head % num_heads;

    const __nv_bfloat16* x_ptr = x + ((size_t)token * num_heads + head) * head_dim;
    __nv_bfloat16* o_ptr = out + ((size_t)token * num_heads + head) * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = bf16_to_float(x_ptr[i]);
        sum_sq += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);

    __shared__ float s_rsqrt;
    if (threadIdx.x == 0) s_rsqrt = rsqrtf(sum_sq / head_dim + eps);
    __syncthreads();

    float scale = s_rsqrt;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = bf16_to_float(x_ptr[i]) * scale;
        float w = bf16_to_float(weight[i]);  // weight is per-head, shape [head_dim]
        o_ptr[i] = float_to_bf16(w * v);
    }
}

void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                              const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream) {
    int total_heads = num_tokens * num_heads;
    int block = std::min(head_dim, 256);
    per_head_rmsnorm_kernel<<<total_heads, block, 0, stream>>>(
        out, x, weight, eps, num_tokens, num_heads, head_dim);
}

// ============================================================================
// SwiGLU: out = silu(gate) * up
// ============================================================================

__global__ void swiglu_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = bf16_to_float(gate[idx]);
    float u = bf16_to_float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = float_to_bf16(silu_g * u);
}

void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up,
                   int num_tokens, int intermediate_size, cudaStream_t stream) {
    int total = num_tokens * intermediate_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    swiglu_kernel<<<grid, block, 0, stream>>>(out, gate, up, total);
}

// ============================================================================
// GELU: out = x * 0.5 * (1 + erf(x / sqrt(2)))
// ============================================================================

__global__ void gelu_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float v = bf16_to_float(x[idx]);
    out[idx] = float_to_bf16(v * 0.5f * (1.0f + erff(v * 0.7071067811865476f)));
}

void invoke_gelu(__nv_bfloat16* out, const __nv_bfloat16* x,
                 int num_elements, cudaStream_t stream) {
    int block = 256;
    int grid = (num_elements + block - 1) / block;
    gelu_kernel<<<grid, block, 0, stream>>>(out, x, num_elements);
}

// ============================================================================
// Sinusoidal Positional Embedding
// ============================================================================

__global__ void sinusoidal_pe_kernel(
    __nv_bfloat16* __restrict__ pe_out,
    int max_positions, int d_model, float log_timescale_base)
{
    // pe_out: [max_positions, d_model]
    // First half = sin, second half = cos
    int pos = blockIdx.x;
    int half = d_model / 2;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float log_ts = -log_timescale_base * i / (half - 1);
        float inv_ts = expf(log_ts);
        float angle = pos * inv_ts;
        pe_out[(size_t)pos * d_model + i] = float_to_bf16(sinf(angle));
        pe_out[(size_t)pos * d_model + half + i] = float_to_bf16(cosf(angle));
    }
}

void compute_sinusoidal_pe(__nv_bfloat16* pe_out,
                           int max_positions, int d_model,
                           float max_timescale,
                           cudaStream_t stream) {
    float log_ts = logf(max_timescale);
    int block = std::min(d_model / 2, 256);
    sinusoidal_pe_kernel<<<max_positions, block, 0, stream>>>(
        pe_out, max_positions, d_model, log_ts);
}

// ============================================================================
// Add Positional Embedding
// ============================================================================

__global__ void add_pe_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ pe_table,
    int seq_len, int hidden_size, int pos_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden_size;
    if (idx >= total) return;

    int t = idx / hidden_size;
    int d = idx % hidden_size;
    float h = bf16_to_float(hidden[idx]);
    float p = bf16_to_float(pe_table[(t + pos_offset) * hidden_size + d]);
    hidden[idx] = float_to_bf16(h + p);
}

void invoke_add_pe(__nv_bfloat16* hidden_states,
                   const __nv_bfloat16* pe_table,
                   int seq_len, int hidden_size,
                   int pos_offset,
                   cudaStream_t stream) {
    int total = seq_len * hidden_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_pe_kernel<<<grid, block, 0, stream>>>(hidden_states, pe_table, seq_len, hidden_size, pos_offset);
}

// ============================================================================
// MRoPE (Multimodal Rotary Position Embedding) — half-rotation, interleaved sections
// ============================================================================
// 半旋转: pairs (d, d+D/2), 不是交错 (2i, 2i+1)
// 频率: 全局 1/theta^(2d/head_dim), 不是 section-local
// Section assignment: d%3==0 或 d≥cutoff → T, d%3==1 且 d<s1*3 → H, d%3==2 且 d<s2*3 → W

__global__ void mrope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ pos_ids,   // [3, num_tokens]
    int num_tokens,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    int s0, int s1, int s2,            // sections: sum = head_dim/2
    float theta)
{
    int token = blockIdx.x;
    int d = blockIdx.y;  // pair index: 0..head_dim/2-1

    if (d >= head_dim / 2) return;

    // Interleaved section assignment (matching Python apply_interleaved_mrope)
    int dim_idx = 0;  // default: temporal (T)
    if ((d % 3 == 1) && (d < s1 * 3)) dim_idx = 1;  // H
    if ((d % 3 == 2) && (d < s2 * 3)) dim_idx = 2;  // W

    int pos = pos_ids[dim_idx * num_tokens + token];

    // Global frequency (same inv_freq for all sections)
    float freq = 1.0f / powf(theta, (float)(d * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Half-rotation: pairs are (d, d + head_dim/2)
    int d_hi = d + head_dim / 2;

    // Apply to Q heads
    for (int h = threadIdx.x; h < num_q_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_q_heads + h) * head_dim;
        float x_lo = bf16_to_float(q[base + d]);
        float x_hi = bf16_to_float(q[base + d_hi]);
        q[base + d]    = float_to_bf16(x_lo * cos_a - x_hi * sin_a);
        q[base + d_hi] = float_to_bf16(x_hi * cos_a + x_lo * sin_a);
    }

    // Apply to KV heads
    for (int h = threadIdx.x; h < num_kv_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_kv_heads + h) * head_dim;
        float x_lo = bf16_to_float(k[base + d]);
        float x_hi = bf16_to_float(k[base + d_hi]);
        k[base + d]    = float_to_bf16(x_lo * cos_a - x_hi * sin_a);
        k[base + d_hi] = float_to_bf16(x_hi * cos_a + x_lo * sin_a);
    }
}

void invoke_mrope(__nv_bfloat16* q, __nv_bfloat16* k,
                  const int* pos_ids,
                  int num_tokens,
                  int num_q_heads, int num_kv_heads,
                  int head_dim,
                  int s0, int s1, int s2,
                  float theta,
                  cudaStream_t stream) {
    dim3 grid(num_tokens, head_dim / 2);
    int block = std::max(num_q_heads, num_kv_heads);
    block = std::min(block, 256);
    mrope_kernel<<<grid, block, 0, stream>>>(
        q, k, pos_ids, num_tokens,
        num_q_heads, num_kv_heads, head_dim,
        s0, s1, s2,
        theta);
}

// ============================================================================
// Standard 1D RoPE (half-rotation)
// ============================================================================

__global__ void rope_1d_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int* __restrict__ pos_ids,
    int num_tokens,
    int num_q_heads, int num_kv_heads,
    int head_dim, float theta)
{
    int token = blockIdx.x;
    int pair = blockIdx.y;  // dim pair index
    int half = head_dim / 2;
    if (pair >= half) return;

    int pos = pos_ids[token];
    float freq = 1.0f / powf(theta, (float)(pair * 2) / head_dim);
    float angle = pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Half-rotation: (d, d+half)
    for (int h = threadIdx.x; h < num_q_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_q_heads + h) * head_dim;
        float x0 = bf16_to_float(q[base + pair]);
        float x1 = bf16_to_float(q[base + pair + half]);
        q[base + pair] = float_to_bf16(x0 * cos_a - x1 * sin_a);
        q[base + pair + half] = float_to_bf16(x0 * sin_a + x1 * cos_a);
    }

    for (int h = threadIdx.x; h < num_kv_heads; h += blockDim.x) {
        size_t base = ((size_t)token * num_kv_heads + h) * head_dim;
        float x0 = bf16_to_float(k[base + pair]);
        float x1 = bf16_to_float(k[base + pair + half]);
        k[base + pair] = float_to_bf16(x0 * cos_a - x1 * sin_a);
        k[base + pair + half] = float_to_bf16(x0 * sin_a + x1 * cos_a);
    }
}

void invoke_rope_1d(__nv_bfloat16* q, __nv_bfloat16* k,
                    const int* pos_ids,
                    int num_tokens,
                    int num_q_heads, int num_kv_heads,
                    int head_dim,
                    float theta,
                    cudaStream_t stream) {
    dim3 grid(num_tokens, head_dim / 2);
    int block = std::max(num_q_heads, num_kv_heads);
    block = std::min(block, 256);
    rope_1d_kernel<<<grid, block, 0, stream>>>(
        q, k, pos_ids, num_tokens,
        num_q_heads, num_kv_heads, head_dim, theta);
}

// ============================================================================
// Bidirectional MHA (ASR Encoder)
// Naive implementation — correctness first, optimize later
// ============================================================================

__global__ void bidirectional_mha_kernel(
    __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int* __restrict__ cu_seqlens,
    int num_heads, int head_dim,
    float scale)
{
    // Each block: one (segment_token, head)
    // blockIdx.x iterates over segment tokens
    int seg = blockIdx.y;  // segment index
    int seg_start = cu_seqlens[seg];
    int seg_end = cu_seqlens[seg + 1];
    int seg_len = seg_end - seg_start;

    int local_token = blockIdx.x;
    if (local_token >= seg_len) return;
    int global_token = seg_start + local_token;
    int head = blockIdx.z;

    // Q for this token/head
    const __nv_bfloat16* q_ptr = q + ((size_t)global_token * num_heads + head) * head_dim;
    __nv_bfloat16* o_ptr = attn_out + ((size_t)global_token * num_heads + head) * head_dim;

    // Compute attention scores (softmax over segment)
    extern __shared__ float smem[];
    float* scores = smem;  // [seg_len]

    float max_score = -1e20f;
    for (int j = threadIdx.x; j < seg_len; j += blockDim.x) {
        const __nv_bfloat16* k_ptr = k + ((size_t)(seg_start + j) * num_heads + head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(q_ptr[d]) * bf16_to_float(k_ptr[d]);
        }
        dot *= scale;
        scores[j] = dot;
        max_score = fmaxf(max_score, dot);
    }
    // Reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    // Exp and sum
    float sum_exp = 0.0f;
    for (int j = threadIdx.x; j < seg_len; j += blockDim.x) {
        float e = expf(scores[j] - max_score);
        scores[j] = e;
        sum_exp += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // Weighted sum of V
    if (threadIdx.x == 0) {
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int j = 0; j < seg_len; j++) {
                const __nv_bfloat16* v_ptr = v + ((size_t)(seg_start + j) * num_heads + head) * head_dim;
                acc += scores[j] * inv_sum * bf16_to_float(v_ptr[d]);
            }
            o_ptr[d] = float_to_bf16(acc);
        }
    }
}

void invoke_bidirectional_mha(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    int total_tokens,
    int num_heads, int head_dim,
    const int* cu_seqlens,
    int num_segments,
    cudaStream_t stream) {

    // Find max segment length (needs CPU access to cu_seqlens)
    // For now use total_tokens as upper bound (safe but may over-allocate shared mem)
    // In practice segments are ≤ n_window_infer = 800
    int max_seg_len = 1024;  // Safe upper bound for shared memory

    float scale = 1.0f / sqrtf((float)head_dim);

    dim3 grid(max_seg_len, num_segments, num_heads);
    int block = 32;  // Single warp for simplicity
    size_t smem_size = max_seg_len * sizeof(float);

    bidirectional_mha_kernel<<<grid, block, smem_size, stream>>>(
        attn_out, q, k, v, cu_seqlens, num_heads, head_dim, scale);
}

// ============================================================================
// Causal GQA Decode Attention (T=1)
// Naive: correctness first
// ============================================================================

__global__ void causal_gqa_decode_kernel(
    __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale)
{
    // One block per Q head (batch_size=1 for now)
    int q_head = blockIdx.x;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = q + (size_t)q_head * head_dim;
    __nv_bfloat16* o_ptr = attn_out + (size_t)q_head * head_dim;

    // Compute scores over all past tokens
    extern __shared__ float smem[];
    float* scores = smem;

    float max_score = -1e20f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const __nv_bfloat16* k_ptr = k_cache + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(q_ptr[d]) * bf16_to_float(k_ptr[d]);
        }
        dot *= scale;
        scores[t] = dot;
        max_score = fmaxf(max_score, dot);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        sum_exp += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();

    if (threadIdx.x == 0) {
        float inv_sum = 1.0f / s_sum;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; t++) {
                const __nv_bfloat16* v_ptr = v_cache + ((size_t)t * num_kv_heads + kv_head) * head_dim;
                acc += scores[t] * inv_sum * bf16_to_float(v_ptr[d]);
            }
            o_ptr[d] = float_to_bf16(acc);
        }
    }
}

void invoke_causal_gqa_decode(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
    int batch_size,
    int num_q_heads, int num_kv_heads, int head_dim,
    int current_seq_len,
    cudaStream_t stream) {

    // batch_size=1 for ASR Phase 1
    int block = 32;
    size_t smem = current_seq_len * sizeof(float);
    causal_gqa_decode_kernel<<<num_q_heads, block, smem, stream>>>(
        attn_out, q, k_cache, v_cache,
        num_q_heads, num_kv_heads, head_dim,
        current_seq_len, 1.0f / sqrtf((float)head_dim));
}

// ============================================================================
// Causal GQA Prefill Attention (T > 1)
// Naive: correctness first
// ============================================================================

__global__ void causal_gqa_prefill_kernel(
    __nv_bfloat16* __restrict__ attn_out,
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float scale)
{
    int token = blockIdx.x;
    int q_head = blockIdx.y;
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const __nv_bfloat16* q_ptr = q + ((size_t)token * num_q_heads + q_head) * head_dim;
    __nv_bfloat16* o_ptr = attn_out + ((size_t)token * num_q_heads + q_head) * head_dim;

    // Causal: attend to positions [0, token]
    int attend_len = token + 1;

    float max_score = -1e20f;
    extern __shared__ float smem[];
    float* scores = smem;

    for (int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        const __nv_bfloat16* k_ptr = k + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += bf16_to_float(q_ptr[d]) * bf16_to_float(k_ptr[d]);
        }
        dot *= scale;
        scores[t] = dot;
        max_score = fmaxf(max_score, dot);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_score;
    __syncthreads();
    max_score = s_max;

    float sum_exp = 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x) {
        float e = expf(scores[t] - max_score);
        scores[t] = e;
        sum_exp += e;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum_exp;
    __syncthreads();

    if (threadIdx.x == 0) {
        float inv_sum = 1.0f / s_sum;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int t = 0; t < attend_len; t++) {
                const __nv_bfloat16* v_ptr = v + ((size_t)t * num_kv_heads + kv_head) * head_dim;
                acc += scores[t] * inv_sum * bf16_to_float(v_ptr[d]);
            }
            o_ptr[d] = float_to_bf16(acc);
        }
    }
}

void invoke_causal_gqa_prefill(
    __nv_bfloat16* attn_out,
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    int seq_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream) {

    dim3 grid(seq_len, num_q_heads);
    int block = 32;
    size_t smem = seq_len * sizeof(float);
    causal_gqa_prefill_kernel<<<grid, block, smem, stream>>>(
        attn_out, q, k, v, seq_len,
        num_q_heads, num_kv_heads, head_dim,
        1.0f / sqrtf((float)head_dim));
}

// ============================================================================
// Embedding Lookup
// ============================================================================

__global__ void embedding_lookup_kernel(
    __nv_bfloat16* __restrict__ out,
    const int* __restrict__ ids,
    const __nv_bfloat16* __restrict__ table,
    int num_tokens, int hidden_size)
{
    int token = blockIdx.x;
    if (token >= num_tokens) return;
    int id = ids[token];
    const __nv_bfloat16* row = table + (size_t)id * hidden_size;
    __nv_bfloat16* out_row = out + (size_t)token * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        out_row[i] = row[i];
    }
}

void invoke_embedding_lookup(__nv_bfloat16* out, const int* ids,
                              const __nv_bfloat16* table,
                              int num_tokens, int hidden_size,
                              cudaStream_t stream) {
    int block = std::min(hidden_size, 256);
    embedding_lookup_kernel<<<num_tokens, block, 0, stream>>>(out, ids, table, num_tokens, hidden_size);
}

// ============================================================================
// Residual Add: a += b
// ============================================================================

__global__ void add_residual_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float va = bf16_to_float(a[idx]);
    float vb = bf16_to_float(b[idx]);
    a[idx] = float_to_bf16(va + vb);
}

void invoke_add_residual(__nv_bfloat16* a, const __nv_bfloat16* b,
                         int num_elements, cudaStream_t stream) {
    int block = 256;
    int grid = (num_elements + block - 1) / block;
    add_residual_kernel<<<grid, block, 0, stream>>>(a, b, num_elements);
}

// ============================================================================
// BF16 Clamp (ASR Encoder FP16 overflow protection)
// ============================================================================

__global__ void bf16_clamp_kernel(
    __nv_bfloat16* __restrict__ x,
    int num_elements, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float v = bf16_to_float(x[idx]);
    v = fminf(fmaxf(v, min_val), max_val);
    x[idx] = float_to_bf16(v);
}

void invoke_bf16_clamp(__nv_bfloat16* x, int num_elements,
                       float min_val, float max_val, cudaStream_t stream) {
    int block = 256;
    int grid = (num_elements + block - 1) / block;
    bf16_clamp_kernel<<<grid, block, 0, stream>>>(x, num_elements, min_val, max_val);
}

// ============================================================================
// Write KV Cache (contiguous, non-paged)
// ============================================================================

__global__ void write_kv_cache_kernel(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    int start_pos, int num_tokens,
    int num_kv_heads, int head_dim)
{
    int kv_size = num_kv_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * kv_size;
    if (idx >= total) return;

    int t = idx / kv_size;
    int kv_idx = idx % kv_size;
    size_t cache_offset = ((size_t)(start_pos + t)) * kv_size + kv_idx;

    k_cache[cache_offset] = k[(size_t)t * kv_size + kv_idx];
    v_cache[cache_offset] = v[(size_t)t * kv_size + kv_idx];
}

void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            int start_pos, int num_tokens,
                            int num_kv_heads, int head_dim,
                            cudaStream_t stream) {
    int kv_size = num_kv_heads * head_dim;
    int total = num_tokens * kv_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    write_kv_cache_kernel<<<grid, block, 0, stream>>>(
        k_cache, v_cache, k, v,
        start_pos, num_tokens, num_kv_heads, head_dim);
}

} // namespace audio_ops
} // namespace qwen_thor
