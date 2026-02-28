// vision.cu — Qwen3.5 Vision Encoder CUDA Implementation
//
// 27-layer ViT + 2×2 Spatial Merger
// All GEMM via cuBLAS, custom kernels for LayerNorm/GELU/RoPE/Softmax
//
// Key differences from text model:
//   - Standard LayerNorm (with bias), not RMSNorm
//   - GELU(tanh) activation, not SiLU/SwiGLU
//   - 2D rotary position encoding (row + col)
//   - Bidirectional attention (no causal mask)
//   - head_dim=72, rotary_dim=36

#include "vision.h"
#include <cuda_bf16.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <cstring>

namespace qwen_thor {
namespace core {

// ============================================================================
// VisionConfig helpers
// ============================================================================
int VisionConfig::pos_grid_size() const {
    return (int)std::sqrt((double)num_position_embeddings);  // 48
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// ---------- LayerNorm with bias ----------
// Grid: (num_tokens), Block: min(hidden_size, 1024)
// Standard LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
__global__ void layernorm_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float eps, int hidden_size)
{
    int row = blockIdx.x;
    const __nv_bfloat16* x_row = x + row * hidden_size;
    __nv_bfloat16* o_row = out + row * hidden_size;

    // Pass 1: compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += __bfloat162float(x_row[i]);
    }
    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    // Block reduce via shared memory
    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / hidden_size;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(x_row[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane_id == 0) shared[warp_id] = var_sum;
    __syncthreads();
    if (warp_id == 0) {
        var_sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / hidden_size + eps);

    // Pass 3: normalize
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (__bfloat162float(x_row[i]) - mean) * inv_std;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        o_row[i] = __float2bfloat16(v * w + b);
    }
}

// ---------- Fused Add + LayerNorm with bias ----------
// hidden[row] += addend[row]; out[row] = LN(hidden[row])
// Saves 1 kernel launch + 1 global memory round-trip vs separate add + LN
// Grid: (num_tokens), Block: min(hidden_size, 1024)
// out may alias addend safely (addend fully read in pass 1, out written in pass 3)
__global__ void fused_add_layernorm_kernel(
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ addend,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float eps, int hidden_size)
{
    int row = blockIdx.x;
    __nv_bfloat16* h_row = hidden + row * hidden_size;
    const __nv_bfloat16* a_row = addend + row * hidden_size;
    __nv_bfloat16* o_row = out + row * hidden_size;

    // Pass 1: add + compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(h_row[i]) + __bfloat162float(a_row[i]);
        h_row[i] = __float2bfloat16(v);
        sum += v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / hidden_size;

    // Pass 2: compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = __bfloat162float(h_row[i]) - mean;
        var_sum += v * v;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    if (lane_id == 0) shared[warp_id] = var_sum;
    __syncthreads();
    if (warp_id == 0) {
        var_sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / hidden_size + eps);

    // Pass 3: normalize
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = (__bfloat162float(h_row[i]) - mean) * inv_std;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        o_row[i] = __float2bfloat16(v * w + b);
    }
}

// ---------- GELU with tanh approximation ----------
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
__global__ void gelu_tanh_kernel(__nv_bfloat16* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(x[idx]);
        const float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/pi)
        float inner = kSqrt2OverPi * (v + 0.044715f * v * v * v);
        float out = 0.5f * v * (1.0f + tanhf(inner));
        x[idx] = __float2bfloat16(out);
    }
}

// ---------- Standard GELU ----------
// y = 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(__nv_bfloat16* __restrict__ x, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(x[idx]);
        float out = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        x[idx] = __float2bfloat16(out);
    }
}

// ---------- Add bias ----------
// x[row * hidden + col] += bias[col]
__global__ void add_bias_kernel(__nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ bias,
                                 int num_tokens, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * hidden_size) {
        int col = idx % hidden_size;
        float v = __bfloat162float(x[idx]) + __bfloat162float(bias[col]);
        x[idx] = __float2bfloat16(v);
    }
}

// ---------- Add residual ----------
__global__ void add_kernel(__nv_bfloat16* __restrict__ out,
                            const __nv_bfloat16* __restrict__ a,
                            const __nv_bfloat16* __restrict__ b,
                            int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = __bfloat162float(a[idx]) + __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(v);
    }
}

// ---------- Bidirectional Softmax ----------
// Grid: (num_heads * seq_len), Block: min(seq_len, 1024)
// scores: [num_heads, seq_len, seq_len], applied per-row (dim=-1)
__global__ void softmax_kernel(float* __restrict__ scores, int seq_len)
{
    int row = blockIdx.x;  // which (head, query) row
    float* row_ptr = scores + row * seq_len;

    // Find max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    // Warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    __shared__ float shared_max[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane_id] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    __syncthreads();
    if (threadIdx.x == 0) shared_max[0] = max_val;
    __syncthreads();
    max_val = shared_max[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float v = expf(row_ptr[i] - max_val);
        row_ptr[i] = v;
        sum += v;
    }
    // Warp reduce sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float shared_sum[32];
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared_sum[0] = sum;
    __syncthreads();
    float inv_sum = 1.0f / shared_sum[0];

    // Normalize
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        row_ptr[i] *= inv_sum;
    }
}

// ---------- 2D Vision RoPE ----------
// Apply rotary position encoding to Q and K using precomputed cos/sin tables
// Q/K layout: [num_patches, num_heads, head_dim]
// rope_cos/sin: [num_patches, rotary_dim]
// Only first rotary_dim dimensions are rotated using half-rotation pattern
__global__ void vision_rope_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int num_patches, int num_heads, int head_dim, int rotary_dim)
{
    // Grid: (num_patches, num_heads), Block: rotary_dim/2
    int patch_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    if (patch_idx >= num_patches) return;

    int half_rot = rotary_dim / 2;  // 18
    int d = threadIdx.x;
    if (d >= half_rot) return;

    // cos/sin for this patch, dimension d
    float cos_val = rope_cos[patch_idx * rotary_dim + d];
    float sin_val = rope_sin[patch_idx * rotary_dim + d];
    // For half-rotation: also need cos/sin for dimensions [half_rot, rotary_dim)
    // These are the column frequencies - stored at offset half_rot
    float cos_val2 = rope_cos[patch_idx * rotary_dim + half_rot + d];
    float sin_val2 = rope_sin[patch_idx * rotary_dim + half_rot + d];

    // Q/K are in [H, N, D] layout (HND) after split_qkv_transpose
    int base_q = (head_idx * num_patches + patch_idx) * head_dim;
    int base_k = (head_idx * num_patches + patch_idx) * head_dim;

    // Apply half-rotation RoPE
    // Pair (d, d + rotary_dim) for first half (row frequencies, dims 0..17 paired with 36..53)
    {
        float q0 = __bfloat162float(q[base_q + d]);
        float q1 = __bfloat162float(q[base_q + d + rotary_dim]);
        q[base_q + d]              = __float2bfloat16(q0 * cos_val - q1 * sin_val);
        q[base_q + d + rotary_dim] = __float2bfloat16(q1 * cos_val + q0 * sin_val);

        float k0 = __bfloat162float(k[base_k + d]);
        float k1 = __bfloat162float(k[base_k + d + rotary_dim]);
        k[base_k + d]              = __float2bfloat16(k0 * cos_val - k1 * sin_val);
        k[base_k + d + rotary_dim] = __float2bfloat16(k1 * cos_val + k0 * sin_val);
    }
    // Pair (d + half_rot, d + half_rot + rotary_dim) for second half (col frequencies, dims 18..35 paired with 54..71)
    {
        float q0 = __bfloat162float(q[base_q + half_rot + d]);
        float q1 = __bfloat162float(q[base_q + half_rot + d + rotary_dim]);
        q[base_q + half_rot + d]              = __float2bfloat16(q0 * cos_val2 - q1 * sin_val2);
        q[base_q + half_rot + d + rotary_dim] = __float2bfloat16(q1 * cos_val2 + q0 * sin_val2);

        float k0 = __bfloat162float(k[base_k + half_rot + d]);
        float k1 = __bfloat162float(k[base_k + half_rot + d + rotary_dim]);
        k[base_k + half_rot + d]              = __float2bfloat16(k0 * cos_val2 - k1 * sin_val2);
        k[base_k + half_rot + d + rotary_dim] = __float2bfloat16(k1 * cos_val2 + k0 * sin_val2);
    }
}

// ---------- Compute 2D RoPE cos/sin tables ----------
// For each patch, compute cos/sin based on its (row, col) position
// position_hw: [num_patches, 2] ints — (row, col) for each patch
__global__ void compute_vision_rope_table_kernel(
    float* __restrict__ cos_out,
    float* __restrict__ sin_out,
    const int* __restrict__ position_hw,
    int num_patches, int rotary_dim, float theta)
{
    // Grid: (num_patches), Block: rotary_dim
    int patch_idx = blockIdx.x;
    int d = threadIdx.x;
    if (patch_idx >= num_patches || d >= rotary_dim) return;

    int half_rot = rotary_dim / 2;  // 18
    int row = position_hw[patch_idx * 2];
    int col = position_hw[patch_idx * 2 + 1];

    // Compute frequency for this dimension
    // First half_rot dims: row position
    // Second half_rot dims: col position
    int freq_idx = d % half_rot;
    float freq = 1.0f / powf(theta, (float)(freq_idx * 2) / (float)rotary_dim);

    float pos = (d < half_rot) ? (float)row : (float)col;
    float angle = pos * freq;

    cos_out[patch_idx * rotary_dim + d] = cosf(angle);
    sin_out[patch_idx * rotary_dim + d] = sinf(angle);
}

// ---------- Position Embedding Interpolation ----------
// Bilinear interpolation of learned [pos_grid×pos_grid, hidden_size] embeddings
// to arbitrary [grid_h, grid_w] grid, output in merge-friendly order
__global__ void pos_embed_interp_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ embed_table,
    int pos_grid_size, int grid_h, int grid_w,
    int num_patches, int hidden_size, int merge_size,
    int patches_per_frame)
{
    // Grid: (num_patches), Block: min(hidden_size, 1024)
    int patch_idx = blockIdx.x;
    if (patch_idx >= num_patches) return;

    // For video (grid_t>1): wrap patch index within each temporal frame
    // so all frames get the same spatial position embeddings
    int frame_patch_idx = patch_idx % patches_per_frame;

    // Decode merge-friendly index to (row, col) in the grid
    int merge_area = merge_size * merge_size;  // 4
    int block_idx = frame_patch_idx / merge_area;
    int intra_idx = patch_idx % merge_area;
    int intra_h = intra_idx / merge_size;
    int intra_w = intra_idx % merge_size;
    int blocks_w = grid_w / merge_size;
    int block_h = block_idx / blocks_w;
    int block_w = block_idx % blocks_w;
    int row = block_h * merge_size + intra_h;
    int col = block_w * merge_size + intra_w;

    // Map (row, col) to continuous coordinate in [0, pos_grid_size-1]
    float y = (grid_h > 1) ? (float)row * (float)(pos_grid_size - 1) / (float)(grid_h - 1) : 0.0f;
    float x_coord = (grid_w > 1) ? (float)col * (float)(pos_grid_size - 1) / (float)(grid_w - 1) : 0.0f;

    // Bilinear interpolation
    int y0 = (int)floorf(y);
    int y1 = min(y0 + 1, pos_grid_size - 1);
    int x0 = (int)floorf(x_coord);
    int x1 = min(x0 + 1, pos_grid_size - 1);
    float fy = y - (float)y0;
    float fx = x_coord - (float)x0;

    // 4 corner indices in the flat embedding table
    int idx00 = y0 * pos_grid_size + x0;
    int idx01 = y0 * pos_grid_size + x1;
    int idx10 = y1 * pos_grid_size + x0;
    int idx11 = y1 * pos_grid_size + x1;

    float w00 = (1.0f - fy) * (1.0f - fx);
    float w01 = (1.0f - fy) * fx;
    float w10 = fy * (1.0f - fx);
    float w11 = fy * fx;

    __nv_bfloat16* out_row = out + patch_idx * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = w00 * __bfloat162float(embed_table[idx00 * hidden_size + i])
                + w01 * __bfloat162float(embed_table[idx01 * hidden_size + i])
                + w10 * __bfloat162float(embed_table[idx10 * hidden_size + i])
                + w11 * __bfloat162float(embed_table[idx11 * hidden_size + i]);
        out_row[i] = __float2bfloat16(v);
    }
}

// ---------- Replace image token embeddings ----------
// Scans token_ids for image_token_id and replaces corresponding embeddings
__global__ void replace_image_tokens_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ vision_features,
    const int* __restrict__ token_ids,
    int total_tokens, int image_token_id, int hidden_size,
    int vision_offset)
{
    // Grid: (total_tokens), Block: min(hidden_size, 1024)
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    if (token_ids[token_idx] != image_token_id) return;

    // Count how many image tokens are before this one
    int vision_idx = 0;
    for (int i = 0; i < token_idx; i++) {
        if (token_ids[i] == image_token_id) vision_idx++;
    }
    vision_idx += vision_offset;

    // Replace embedding
    __nv_bfloat16* dst = hidden + token_idx * hidden_size;
    const __nv_bfloat16* src = vision_features + vision_idx * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// ---------- FP32 → BF16 ----------
__global__ void f32_to_bf16_kernel(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// ---------- Transpose for attention: [N, H, D] → [H, N, D] ----------
__global__ void transpose_nhd_to_hnd_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int N, int H, int D)
{
    // Grid: (N, H), Block: D
    int n = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (n >= N || h >= H || d >= D) return;

    // in[n, h, d] → out[h, n, d]
    out[(h * N + n) * D + d] = in[(n * H + h) * D + d];
}

// ---------- Fused Split QKV + Transpose: [N, 3*H*D] → Q/K/V each [H, N, D] ----------
// qkv layout: row n has [Q_h0_d0..d71, Q_h1_d0..d71, ..., K_h0_d0..d71, ..., V_h0_d0..d71, ...]
// Row stride = 3*H*D = 3456
__global__ void split_qkv_transpose_kernel(
    __nv_bfloat16* __restrict__ q_out,   // [H, N, D]
    __nv_bfloat16* __restrict__ k_out,   // [H, N, D]
    __nv_bfloat16* __restrict__ v_out,   // [H, N, D]
    const __nv_bfloat16* __restrict__ qkv,  // [N, 3*H*D]
    int N, int H, int D)
{
    // Grid: (N, H), Block: D
    int n = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (d >= D) return;

    int qkv_stride = 3 * H * D;  // 3456
    int src_base = n * qkv_stride + h * D + d;
    int dst_idx  = (h * N + n) * D + d;

    q_out[dst_idx] = qkv[src_base];              // Q: cols [0, H*D)
    k_out[dst_idx] = qkv[src_base + H * D];      // K: cols [H*D, 2*H*D)
    v_out[dst_idx] = qkv[src_base + 2 * H * D];  // V: cols [2*H*D, 3*H*D)
}

// ---------- Transpose back: [H, N, D] → [N, H, D] ----------
__global__ void transpose_hnd_to_nhd_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int N, int H, int D)
{
    int n = blockIdx.x;
    int h = blockIdx.y;
    int d = threadIdx.x;
    if (n >= N || h >= H || d >= D) return;

    // in[h, n, d] → out[n, h, d]
    out[(n * H + h) * D + d] = in[(h * N + n) * D + d];
}

// ---------- BF16 → FP32 for attention scores ----------
__global__ void bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ src,
                                    float* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

// ---------- FP32 → BF16 for attention output ----------
__global__ void f32_to_bf16_attn_kernel(const float* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// ---------- In-place FP32 → BF16 conversion ----------
// Each block handles a contiguous chunk. Reads FP32 value into register,
// then __syncthreads to ensure all reads in block complete, then writes BF16.
// Safe because BF16 (2 bytes) < FP32 (4 bytes): no cross-block overlap.
static __global__ void f32_to_bf16_inplace_kernel(
    void* __restrict__ buffer, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (idx < total_elements) {
        val = ((const float*)buffer)[idx];
    }
    __syncthreads();  // Ensure all FP32 reads in this block complete
    if (idx < total_elements) {
        ((__nv_bfloat16*)buffer)[idx] = __float2bfloat16(val);
    }
}

// ============================================================================
// FlashAttention for Vision (bidirectional, head_dim=72)
// ============================================================================
// Grid: (ceil(N / BLOCK_Q), num_heads)
// Block: BLOCK_Q threads (128), each thread handles one query row
// Shared memory: K_tile[TILE_KV][D] + V_tile[TILE_KV][D] in BF16
//
// Online softmax in registers: no scores buffer, no FP32→BF16 cast needed.
// Output directly in NHD layout [N, H, D] — eliminates transpose kernel.
//
// Q/K/V input: [H, N, D] (HND layout from split_qkv_transpose)
// Output:      [N, H, D] (NHD layout for subsequent GEMM)
//
// Register usage: q_reg[72] + acc[72] + misc ≈ 160 per thread
// Shared memory: 2 × 32 × 72 × 2 = 9216 bytes
// Occupancy: 2-3 blocks/SM with launch_bounds(128, 2)
#define VISION_FA_TILE_KV 32

__global__ void __launch_bounds__(128, 2)
vision_flash_attention_kernel(
    __nv_bfloat16* __restrict__ out,      // [N, H, D]  NHD layout
    const __nv_bfloat16* __restrict__ Q,  // [H, N, D]  HND layout
    const __nv_bfloat16* __restrict__ K,  // [H, N, D]
    const __nv_bfloat16* __restrict__ V,  // [H, N, D]
    int N, int H, int D, float scale)
{
    const int tid = threadIdx.x;
    const int qi = blockIdx.x * blockDim.x + tid;   // query index
    const int h = blockIdx.y;                         // head index

    // Shared memory for KV tile loading (broadcast reads — no bank conflicts)
    __shared__ __nv_bfloat16 K_smem[VISION_FA_TILE_KV][72];
    __shared__ __nv_bfloat16 V_smem[VISION_FA_TILE_KV][72];

    // Load query into registers (FP32 for accumulation precision)
    float q_reg[72];
    if (qi < N) {
        const __nv_bfloat16* q_ptr = Q + ((size_t)h * N + qi) * D;
        #pragma unroll
        for (int d = 0; d < 72; d++)
            q_reg[d] = __bfloat162float(q_ptr[d]);
    }

    // Online softmax state
    float m = -1e30f;   // running max
    float l = 0.0f;     // running sum of exp
    float acc[72];      // unnormalized accumulator: sum of exp(s_j - m) * v_j
    #pragma unroll
    for (int d = 0; d < 72; d++) acc[d] = 0.0f;

    // Iterate over KV tiles
    for (int kv_start = 0; kv_start < N; kv_start += VISION_FA_TILE_KV) {
        int actual_kv = min(VISION_FA_TILE_KV, N - kv_start);

        // Collaboratively load K/V tile into shared memory
        // 128 threads load 32×72 = 2304 BF16 elements → 18 per thread
        __syncthreads();
        {
            const int total_elems = VISION_FA_TILE_KV * 72;
            for (int i = tid; i < total_elems; i += blockDim.x) {
                int j = i / 72;   // row within tile
                int d = i % 72;   // dimension
                int gj = kv_start + j;
                if (gj < N) {
                    size_t src_idx = ((size_t)h * N + gj) * D + d;
                    K_smem[j][d] = K[src_idx];
                    V_smem[j][d] = V[src_idx];
                } else {
                    K_smem[j][d] = __float2bfloat16(0.0f);
                    V_smem[j][d] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        if (qi >= N) continue;

        // Process each key in this tile with online softmax
        for (int j = 0; j < actual_kv; j++) {
            // Compute attention score: q · k / sqrt(d)
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < 72; d++)
                s += q_reg[d] * __bfloat162float(K_smem[j][d]);
            s *= scale;

            // Online softmax update (rescale only when max changes)
            if (s > m) {
                float corr = expf(m - s);
                #pragma unroll
                for (int d = 0; d < 72; d++)
                    acc[d] *= corr;
                l *= corr;
                m = s;
            }

            float p = expf(s - m);
            #pragma unroll
            for (int d = 0; d < 72; d++)
                acc[d] += p * __bfloat162float(V_smem[j][d]);
            l += p;
        }
    }

    // Write output in NHD layout: out[qi, h, :] = acc / l
    if (qi < N && l > 0.0f) {
        float inv_l = 1.0f / l;
        __nv_bfloat16* out_ptr = out + ((size_t)qi * H + h) * D;
        #pragma unroll
        for (int d = 0; d < 72; d++)
            out_ptr[d] = __float2bfloat16(acc[d] * inv_l);
    }
}

// ============================================================================
// Fused bias + GELU(tanh) kernel
// ============================================================================
// x[i] = gelu_tanh(x[i] + bias[i % hidden_size])
// Replaces: invoke_add_bias + invoke_gelu_tanh (saves 1 kernel launch per block)
__global__ void add_bias_gelu_tanh_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int total, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = __bfloat162float(x[idx]) + __bfloat162float(bias[idx % hidden_size]);
        const float kSqrt2OverPi = 0.7978845608f;
        float inner = kSqrt2OverPi * (v + 0.044715f * v * v * v);
        x[idx] = __float2bfloat16(0.5f * v * (1.0f + tanhf(inner)));
    }
}

// ============================================================================
// Fused bias + residual add kernel
// ============================================================================
// hidden[i] += x[i] + bias[i % hidden_size]
// Replaces: invoke_add_bias on x + invoke_add(hidden, hidden, x)
__global__ void add_bias_residual_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int total, int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = __bfloat162float(x[idx]) + __bfloat162float(bias[idx % hidden_size]);
        hidden[idx] = __float2bfloat16(__bfloat162float(hidden[idx]) + val);
    }
}

// ============================================================================
// Fused add + bias + LayerNorm kernel
// ============================================================================
// hidden[i] += addend[i] + addend_bias[i % hidden_size]
// out = LayerNorm(hidden)
// Replaces: invoke_add_bias on addend + fused_add_layernorm
__global__ void fused_add_bias_layernorm_kernel(
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ addend,
    const __nv_bfloat16* __restrict__ addend_bias,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    float eps, int hidden_size)
{
    // Each block = one token
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;

    // Pass 1: hidden += addend + addend_bias, compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float h = __bfloat162float(hidden[base + i]);
        float a = __bfloat162float(addend[base + i]);
        float b = __bfloat162float(addend_bias[i]);
        float val = h + a + b;
        hidden[base + i] = __float2bfloat16(val);
        local_sum += val;
    }
    // Warp reduce
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    __shared__ float s_data[32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) s_data[wid] = local_sum;
    __syncthreads();
    float mean = 0.0f;
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s_data[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) s_data[0] = v / hidden_size;
    }
    __syncthreads();
    mean = s_data[0];

    // Pass 2: variance
    float local_var = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(hidden[base + i]) - mean;
        local_var += val * val;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    if (lane == 0) s_data[wid] = local_var;
    __syncthreads();
    float var_sum = 0.0f;
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? s_data[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (tid == 0) s_data[0] = v;
    }
    __syncthreads();
    float inv_std = rsqrtf(s_data[0] / hidden_size + eps);

    // Pass 3: normalize
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = (__bfloat162float(hidden[base + i]) - mean) * inv_std;
        out[base + i] = __float2bfloat16(val * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}


// ============================================================================
// Kernel Launch Wrappers
// ============================================================================
namespace vision_ops {

void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size,
                      cudaStream_t stream)
{
    int block = std::min(hidden_size, 1024);
    layernorm_kernel<<<num_tokens, block, 0, stream>>>(out, x, weight, bias, eps, hidden_size);
}

void invoke_fused_add_layernorm(__nv_bfloat16* out, __nv_bfloat16* hidden,
                                const __nv_bfloat16* addend,
                                const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                                float eps, int num_tokens, int hidden_size,
                                cudaStream_t stream)
{
    int block = std::min(hidden_size, 1024);
    fused_add_layernorm_kernel<<<num_tokens, block, 0, stream>>>(
        out, hidden, addend, weight, bias, eps, hidden_size);
}

void invoke_gelu_tanh(__nv_bfloat16* x, int n, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_tanh_kernel<<<grid, block, 0, stream>>>(x, n);
}

void invoke_gelu(__nv_bfloat16* x, int n, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_kernel<<<grid, block, 0, stream>>>(x, n);
}

void invoke_add_bias(__nv_bfloat16* x, const __nv_bfloat16* bias,
                     int num_tokens, int hidden_size, cudaStream_t stream)
{
    int total = num_tokens * hidden_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(x, bias, num_tokens, hidden_size);
}

void invoke_add(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                int n, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
}

void invoke_softmax(float* scores, int num_heads, int seq_len, cudaStream_t stream)
{
    int num_rows = num_heads * seq_len;
    int block = std::min(seq_len, 1024);
    softmax_kernel<<<num_rows, block, 0, stream>>>(scores, seq_len);
}

void invoke_f32_to_bf16_inplace(void* buffer, int total_elements, cudaStream_t stream)
{
    int block = 256;
    int grid = (total_elements + block - 1) / block;
    f32_to_bf16_inplace_kernel<<<grid, block, 0, stream>>>(buffer, total_elements);
}

void invoke_softmax_cast_bf16(float* scores_f32, __nv_bfloat16* scores_bf16_out,
                              int num_heads, int seq_len, cudaStream_t stream)
{
    // Step 1: FP32 softmax
    invoke_softmax(scores_f32, num_heads, seq_len, stream);
    // Step 2: In-place FP32 → BF16 conversion
    int total = num_heads * seq_len * seq_len;
    int block = 256;
    int grid = (total + block - 1) / block;
    f32_to_bf16_inplace_kernel<<<grid, block, 0, stream>>>((void*)scores_f32, total);
    // After this, scores_bf16_out (which aliases scores_f32) contains BF16 data
    // at the start of the buffer (first total*2 bytes)
}

void invoke_vision_rope(__nv_bfloat16* q, __nv_bfloat16* k,
                        const float* rope_cos, const float* rope_sin,
                        int num_patches, int num_heads, int head_dim, int rotary_dim,
                        cudaStream_t stream)
{
    dim3 grid(num_patches, num_heads);
    int block = rotary_dim / 2;  // 18
    vision_rope_kernel<<<grid, block, 0, stream>>>(
        q, k, rope_cos, rope_sin, num_patches, num_heads, head_dim, rotary_dim);
}

void invoke_compute_vision_rope_table(float* cos_out, float* sin_out,
                                       const int* positions_hw,
                                       int num_patches, int rotary_dim,
                                       float theta, int /*grid_h*/, int /*grid_w*/,
                                       cudaStream_t stream)
{
    compute_vision_rope_table_kernel<<<num_patches, rotary_dim, 0, stream>>>(
        cos_out, sin_out, positions_hw, num_patches, rotary_dim, theta);
}

void invoke_pos_embed_interp(__nv_bfloat16* out,
                              const __nv_bfloat16* embed_table,
                              int num_pos, int pos_grid_size,
                              int grid_h, int grid_w, int num_patches,
                              int hidden_size, int spatial_merge_size,
                              int patches_per_frame,
                              cudaStream_t stream)
{
    int block = std::min(hidden_size, 1024);
    pos_embed_interp_kernel<<<num_patches, block, 0, stream>>>(
        out, embed_table, pos_grid_size, grid_h, grid_w,
        num_patches, hidden_size, spatial_merge_size, patches_per_frame);
}

void invoke_replace_image_tokens(__nv_bfloat16* hidden,
                                  const __nv_bfloat16* vision_features,
                                  const int* token_ids, int total_tokens,
                                  int image_token_id, int hidden_size,
                                  int vision_offset,
                                  cudaStream_t stream)
{
    int block = std::min(hidden_size, 1024);
    replace_image_tokens_kernel<<<total_tokens, block, 0, stream>>>(
        hidden, vision_features, token_ids, total_tokens,
        image_token_id, hidden_size, vision_offset);
}

void invoke_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    f32_to_bf16_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

void invoke_vision_flash_attention(__nv_bfloat16* out,
                                    const __nv_bfloat16* Q,
                                    const __nv_bfloat16* K,
                                    const __nv_bfloat16* V,
                                    int N, int H, int D, float scale,
                                    cudaStream_t stream)
{
    const int BLOCK_Q = 128;
    dim3 grid((N + BLOCK_Q - 1) / BLOCK_Q, H);
    // Shared memory: K_tile[32][72] + V_tile[32][72] = 9216 bytes
    vision_flash_attention_kernel<<<grid, BLOCK_Q, 0, stream>>>(
        out, Q, K, V, N, H, D, scale);
}

void invoke_add_bias_gelu_tanh(__nv_bfloat16* x, const __nv_bfloat16* bias,
                                int num_tokens, int hidden_size,
                                cudaStream_t stream)
{
    int total = num_tokens * hidden_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_gelu_tanh_kernel<<<grid, block, 0, stream>>>(x, bias, total, hidden_size);
}

void invoke_add_bias_residual(__nv_bfloat16* hidden, const __nv_bfloat16* x,
                               const __nv_bfloat16* bias,
                               int num_tokens, int hidden_size,
                               cudaStream_t stream)
{
    int total = num_tokens * hidden_size;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_residual_kernel<<<grid, block, 0, stream>>>(hidden, x, bias, total, hidden_size);
}

void invoke_fused_add_bias_layernorm(__nv_bfloat16* out, __nv_bfloat16* hidden,
                                      const __nv_bfloat16* addend,
                                      const __nv_bfloat16* addend_bias,
                                      const __nv_bfloat16* weight,
                                      const __nv_bfloat16* bias,
                                      float eps, int num_tokens, int hidden_size,
                                      cudaStream_t stream)
{
    int block = std::min(hidden_size, 1024);
    fused_add_bias_layernorm_kernel<<<num_tokens, block, 0, stream>>>(
        out, hidden, addend, addend_bias, weight, bias, eps, hidden_size);
}

} // namespace vision_ops


// ============================================================================
// VisionEncoder Implementation
// ============================================================================

VisionEncoder::VisionEncoder(const VisionConfig& config)
    : config_(config)
{
    blocks_.resize(config_.depth);
}

VisionEncoder::~VisionEncoder() {
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

cublasHandle_t VisionEncoder::get_cublas() {
    if (!cublas_handle_) {
        cublasCreate(&cublas_handle_);
    }
    return cublas_handle_;
}

void VisionEncoder::set_patch_embed_weights(__nv_bfloat16* proj_w, __nv_bfloat16* proj_b) {
    patch_proj_w_ = proj_w;
    patch_proj_b_ = proj_b;
}

void VisionEncoder::set_pos_embed_weight(__nv_bfloat16* w) {
    pos_embed_w_ = w;
}

void VisionEncoder::set_block_weights(int block_idx,
                                       __nv_bfloat16* norm1_w, __nv_bfloat16* norm1_b,
                                       __nv_bfloat16* qkv_w,   __nv_bfloat16* qkv_b,
                                       __nv_bfloat16* proj_w,   __nv_bfloat16* proj_b,
                                       __nv_bfloat16* norm2_w, __nv_bfloat16* norm2_b,
                                       __nv_bfloat16* fc1_w,   __nv_bfloat16* fc1_b,
                                       __nv_bfloat16* fc2_w,   __nv_bfloat16* fc2_b) {
    auto& b = blocks_[block_idx];
    b.norm1_w = norm1_w; b.norm1_b = norm1_b;
    b.qkv_w = qkv_w;     b.qkv_b = qkv_b;
    b.proj_w = proj_w;   b.proj_b = proj_b;
    b.norm2_w = norm2_w; b.norm2_b = norm2_b;
    b.fc1_w = fc1_w;     b.fc1_b = fc1_b;
    b.fc2_w = fc2_w;     b.fc2_b = fc2_b;
}

void VisionEncoder::set_merger_weights(__nv_bfloat16* norm_w,  __nv_bfloat16* norm_b,
                                        __nv_bfloat16* fc1_w,   __nv_bfloat16* fc1_b,
                                        __nv_bfloat16* fc2_w,   __nv_bfloat16* fc2_b) {
    merger_norm_w_ = norm_w; merger_norm_b_ = norm_b;
    merger_fc1_w_ = fc1_w;   merger_fc1_b_ = fc1_b;
    merger_fc2_w_ = fc2_w;   merger_fc2_b_ = fc2_b;
}

// ============================================================================
// Workspace Layout (cuBLAS attention, L2-fit chunk, fused MLP kernels)
// ============================================================================
// For N patches (block_forward workspace):
//  [0]                   : norm_out        [N, 1152]           BF16
//  [norm_offset]         : QKV             [N, 3456]           BF16
//  [qkv_offset]          : Q transposed    [16, N, 72]         BF16
//  [q_trans_offset]      : K transposed    [16, N, 72]         BF16
//  [k_trans_offset]      : V transposed    [16, N, 72]         BF16
//  [v_trans_offset]      : attn scores     [16, chunk, N]      FP32  (≤ 256 MB)
//  [scores_offset]       : attn output t   [16, N, 72]         BF16
//  [attn_out_t_offset]   : attn output     [N, 1152]           BF16
//  [attn_out_offset]     : MLP intermediate [N, 4304]          BF16
// Forward-level:
//  [rope_cos_offset]     : RoPE cos        [N, 36]             FP32
//  [rope_sin_offset]     : RoPE sin        [N, 36]             FP32
//  [pos_hw_offset]       : positions_hw    [N, 2]              INT32
//  [pos_embed_offset]    : pos embeddings  [N, 1152]           BF16
//  [merger_offset]       : merger work     [N/4, 4608]         BF16
//  [output_offset]       : final output    [N/4, 5120]         BF16

// Compute attention chunk size — keep scores ≤ L2 cache (32 MB).
// L2=32 MB is the sweet spot: fits QK scores for softmax+PV pipeline in cache.
// Larger chunks cause L2 thrashing (verified: 256 MB → 47% regression).
static int compute_attn_chunk(int N, int num_heads) {
    constexpr size_t SCORES_BUDGET = 32ULL * 1024 * 1024;  // 32 MB = L2 cache size
    int attn_chunk = N;
    size_t scores_bytes = (size_t)num_heads * N * N * sizeof(float);
    if (scores_bytes > SCORES_BUDGET) {
        attn_chunk = (int)(SCORES_BUDGET / ((size_t)num_heads * N * sizeof(float)));
        attn_chunk = std::max(32, (attn_chunk / 32) * 32);
        attn_chunk = std::min(attn_chunk, N);
    }
    return attn_chunk;
}

size_t VisionEncoder::workspace_bytes(int N) const {
    int hs = config_.hidden_size;        // 1152
    int qkv_dim = 3 * hs;               // 3456
    int num_heads = config_.num_heads;   // 16
    int head_dim = config_.head_dim;     // 72
    int is = config_.intermediate_size;  // 4304
    int rot = config_.rotary_dim;        // 36
    int merge = config_.spatial_merge_size;  // 2
    int out_hs = config_.out_hidden_size;    // 5120
    int merger_hs = config_.merger_hidden();  // 4608
    int N_out = N / (merge * merge);
    int attn_chunk = compute_attn_chunk(N, num_heads);

    size_t bytes = 0;
    bytes += (size_t)N * config_.patch_input_dim() * 2;  // d_patches (BF16 input) 
    bytes += (size_t)N * hs * 2;              // hidden_states  BF16
    // NO residual buffer — hidden IS the residual
    bytes += (size_t)N * hs * 2;              // norm_out       BF16
    bytes += (size_t)N * qkv_dim * 2;         // QKV            BF16
    bytes += (size_t)num_heads * N * head_dim * 2; // Q transposed  BF16
    bytes += (size_t)num_heads * N * head_dim * 2; // K transposed  BF16
    bytes += (size_t)num_heads * N * head_dim * 2; // V transposed  BF16
    bytes += (size_t)num_heads * attn_chunk * N * 4; // attn scores FP32 (LARGE CHUNK)
    bytes += (size_t)num_heads * N * head_dim * 2; // attn output t BF16
    bytes += (size_t)N * hs * 2;              // attn output    BF16
    bytes += (size_t)N * is * 2;              // MLP intermediate BF16
    bytes += (size_t)N * rot * 4;             // RoPE cos       FP32
    bytes += (size_t)N * rot * 4;             // RoPE sin       FP32
    bytes += (size_t)N * 2 * 4;               // positions_hw   INT32
    bytes += (size_t)N * hs * 2;              // pos embeddings BF16
    bytes += (size_t)N_out * merger_hs * 2;   // merger work    BF16
    bytes += (size_t)N_out * out_hs * 2;      // final output   BF16
    bytes += 4096;                             // alignment padding

    return bytes;
}

// ============================================================================
// Image Preprocessing (CPU-side)
// ============================================================================

// Smart resize: find (h_bar, w_bar) aligned to factor, within [min_pixels, max_pixels]
static std::pair<int, int> smart_resize(int height, int width, int factor,
                                         int min_pixels, int max_pixels) {
    int h_bar = std::max(factor, (int)(std::round((double)height / factor) * factor));
    int w_bar = std::max(factor, (int)(std::round((double)width / factor) * factor));

    if ((long long)h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((double)height * width / max_pixels);
        h_bar = std::max(factor, (int)(std::floor((double)height / beta / factor) * factor));
        w_bar = std::max(factor, (int)(std::floor((double)width / beta / factor) * factor));
    } else if ((long long)h_bar * w_bar < min_pixels) {
        double beta = std::sqrt((double)min_pixels / ((double)height * width));
        h_bar = std::max(factor, (int)(std::ceil((double)height * beta / factor) * factor));
        w_bar = std::max(factor, (int)(std::ceil((double)width * beta / factor) * factor));
    }

    return {h_bar, w_bar};
}

// CPU float → BF16 conversion (truncation with round-to-nearest-even)
static inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    // Round to nearest even: add rounding bias based on bit 16 (LSB of result)
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    bits += rounding_bias;
    return (uint16_t)(bits >> 16);
}

// Bilinear resize on CPU (channel-first float output)
static std::vector<float> cpu_bilinear_resize(const uint8_t* rgb, int src_w, int src_h,
                                                int dst_w, int dst_h) {
    // Output: [3, dst_h, dst_w] normalized to [-1, 1]
    std::vector<float> out(3 * dst_h * dst_w);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < dst_h; y++) {
            float sy = (float)y * (src_h - 1) / std::max(dst_h - 1, 1);
            int y0 = (int)sy;
            int y1 = std::min(y0 + 1, src_h - 1);
            float fy = sy - y0;
            for (int x = 0; x < dst_w; x++) {
                float sx = (float)x * (src_w - 1) / std::max(dst_w - 1, 1);
                int x0 = (int)sx;
                int x1 = std::min(x0 + 1, src_w - 1);
                float fx = sx - x0;

                float v00 = rgb[(y0 * src_w + x0) * 3 + c];
                float v01 = rgb[(y0 * src_w + x1) * 3 + c];
                float v10 = rgb[(y1 * src_w + x0) * 3 + c];
                float v11 = rgb[(y1 * src_w + x1) * 3 + c];

                float val = (1 - fy) * ((1 - fx) * v00 + fx * v01)
                          + fy * ((1 - fx) * v10 + fx * v11);
                // Normalize: (val / 255 - 0.5) / 0.5 = val / 127.5 - 1.0
                out[c * dst_h * dst_w + y * dst_w + x] = val / 127.5f - 1.0f;
            }
        }
    }
    return out;
}

ProcessedImage VisionEncoder::preprocess_image(const ImageInput& image,
                                                const VisionConfig& config) {
    ProcessedImage result;

    int factor = config.factor();  // 32
    auto [h_bar, w_bar] = smart_resize(image.height, image.width, factor,
                                        config.min_pixels, config.max_pixels);

    // Resize and normalize to channel-first [3, h_bar, w_bar], range [-1, 1]
    auto resized = cpu_bilinear_resize(image.pixels.data(), image.width, image.height,
                                         w_bar, h_bar);

    // Grid dimensions
    result.grid_t = 1;
    result.grid_h = h_bar / config.patch_size;    // e.g., 256/16 = 16
    result.grid_w = w_bar / config.patch_size;

    int P = config.patch_size;      // 16
    int T = config.temporal_patch_size;  // 2
    int C = config.in_channels;     // 3
    int merge = config.spatial_merge_size;  // 2

    int total_patches = result.grid_t * result.grid_h * result.grid_w;
    int patch_dim = C * T * P * P;  // 1536
    result.pixel_values_bf16.resize(total_patches * patch_dim);

    // Extract patches in merge-friendly order and convert to BF16:
    // For block (bh, bw), intra (ih, iw):
    //   patch_idx = ((bh * blocks_w + bw) * merge + ih) * merge + iw
    //   pixel row range: [row * P, row * P + P)
    //   pixel col range: [col * P, col * P + P)
    //   where row = bh * merge + ih, col = bw * merge + iw
    int blocks_h = result.grid_h / merge;
    int blocks_w = result.grid_w / merge;

    int patch_idx = 0;
    for (int bh = 0; bh < blocks_h; bh++) {
        for (int bw = 0; bw < blocks_w; bw++) {
            for (int ih = 0; ih < merge; ih++) {
                for (int iw = 0; iw < merge; iw++) {
                    int row = bh * merge + ih;
                    int col = bw * merge + iw;

                    uint16_t* dst = result.pixel_values_bf16.data() + patch_idx * patch_dim;
                    int d = 0;

                    // For each channel, temporal frame, patch pixel → BF16
                    for (int c = 0; c < C; c++) {
                        for (int t = 0; t < T; t++) {
                            // For single image, both temporal frames are the same
                            for (int ph = 0; ph < P; ph++) {
                                for (int pw = 0; pw < P; pw++) {
                                    int py = row * P + ph;
                                    int px = col * P + pw;
                                    // resized is channel-first: [c, y, x]
                                    float val = resized[c * h_bar * w_bar + py * w_bar + px];
                                    dst[d++] = float_to_bf16(val);
                                }
                            }
                        }
                    }

                    patch_idx++;
                }
            }
        }
    }

    // Precompute RoPE positions on CPU (avoids CPU loop in GPU forward path)
    result.positions_hw.resize(total_patches * 2);
    int patches_per_frame = result.grid_h * result.grid_w;
    for (int i = 0; i < total_patches; i++) {
        int frame_i = i % patches_per_frame;
        int blk_idx = frame_i / (merge * merge);
        int intra_idx = frame_i % (merge * merge);
        int intra_h = intra_idx / merge;
        int intra_w = intra_idx % merge;
        int block_h = blk_idx / blocks_w;
        int block_w = blk_idx % blocks_w;
        result.positions_hw[i * 2]     = block_h * merge + intra_h;
        result.positions_hw[i * 2 + 1] = block_w * merge + intra_w;
    }

    return result;
}

// ============================================================================
// Video Preprocessing (CPU-side)
// ============================================================================
// Takes multi-frame video input, samples frames, smart-resizes with temporal budget,
// and extracts patches with grid_t > 1 for temporal-aware vision processing.
//
// Video smart_resize includes temporal dimension:
//   t_bar = ceil(num_frames / temporal_patch_size) * temporal_patch_size
//   Ensure t_bar * h_bar * w_bar within pixel budget

// Video-aware smart resize (includes temporal dimension)
static std::tuple<int, int, int> smart_resize_video(int num_frames, int height, int width,
                                                      int temporal_factor, int spatial_factor,
                                                      int min_pixels, int max_pixels) {
    int h_bar = std::max(spatial_factor, (int)(std::round((double)height / spatial_factor) * spatial_factor));
    int w_bar = std::max(spatial_factor, (int)(std::round((double)width / spatial_factor) * spatial_factor));
    int t_bar = (int)std::ceil((double)num_frames / temporal_factor) * temporal_factor;
    if (t_bar < temporal_factor) t_bar = temporal_factor;

    long long total = (long long)t_bar * h_bar * w_bar;
    if (total > max_pixels) {
        // Scale down spatial dimensions  
        double budget_per_frame = (double)max_pixels / t_bar;
        double beta = std::sqrt((double)height * width / budget_per_frame);
        h_bar = std::max(spatial_factor, (int)(std::floor((double)height / beta / spatial_factor) * spatial_factor));
        w_bar = std::max(spatial_factor, (int)(std::floor((double)width / beta / spatial_factor) * spatial_factor));
    } else if (total < min_pixels) {
        double budget_per_frame = (double)min_pixels / t_bar;
        double beta = std::sqrt(budget_per_frame / ((double)height * width));
        h_bar = (int)(std::ceil((double)height * beta / spatial_factor) * spatial_factor);
        w_bar = (int)(std::ceil((double)width * beta / spatial_factor) * spatial_factor);
    }

    return {t_bar, h_bar, w_bar};
}

ProcessedImage VisionEncoder::preprocess_video(const VideoInput& video,
                                                const VisionConfig& config) {
    ProcessedImage result;
    result.is_video = true;

    if (video.frames.empty()) return result;

    // --- 1. Frame sampling ---
    // If we have more frames than max_frames, uniformly subsample
    int total_frames = (int)video.frames.size();
    std::vector<int> selected_indices;

    int target_frames = total_frames;
    if (video.source_fps > 0 && video.target_fps > 0) {
        target_frames = (int)(total_frames / video.source_fps * video.target_fps);
    }
    target_frames = std::max(video.min_frames, std::min(target_frames, video.max_frames));
    target_frames = std::min(target_frames, total_frames);

    // Uniform sampling
    selected_indices.resize(target_frames);
    for (int i = 0; i < target_frames; i++) {
        selected_indices[i] = (int)std::round((double)i * (total_frames - 1) / std::max(1, target_frames - 1));
    }

    int num_selected = (int)selected_indices.size();

    // --- 2. Smart resize (with temporal budget) ---
    int factor = config.factor();  // 32
    // Use a generous budget for video total pixels
    int video_max_pixels = config.max_pixels;  // same budget as image
    // But per-frame budget is max_pixels / t_bar, so we allow more total for short videos
    auto [t_bar, h_bar, w_bar] = smart_resize_video(
        num_selected, video.height, video.width,
        config.temporal_patch_size, factor,
        config.min_pixels, video_max_pixels);

    // --- 3. Resize each selected frame ---
    // All frames resized to same (h_bar, w_bar)
    std::vector<std::vector<float>> resized_frames;
    resized_frames.reserve(num_selected);
    for (int idx : selected_indices) {
        auto resized = cpu_bilinear_resize(video.frames[idx].data(),
                                            video.width, video.height,
                                            w_bar, h_bar);
        resized_frames.push_back(std::move(resized));
    }

    // --- 4. Pad to t_bar frames (repeat last frame) ---
    while ((int)resized_frames.size() < t_bar) {
        resized_frames.push_back(resized_frames.back());
    }

    // --- 5. Grid dimensions ---
    result.grid_t = t_bar / config.temporal_patch_size;  // temporal groups
    result.grid_h = h_bar / config.patch_size;
    result.grid_w = w_bar / config.patch_size;

    int P = config.patch_size;
    int T = config.temporal_patch_size;  // 2
    int C = config.in_channels;          // 3
    int merge = config.spatial_merge_size; // 2

    int total_patches = result.grid_t * result.grid_h * result.grid_w;
    int patch_dim = C * T * P * P;  // 1536
    result.pixel_values_bf16.resize(total_patches * patch_dim);

    // --- 6. Compute timestamps ---
    result.timestamps.resize(result.grid_t);
    for (int gt = 0; gt < result.grid_t; gt++) {
        // Timestamp for this temporal group: average of the two frames' timestamps
        int frame0_idx = gt * T;
        int frame1_idx = frame0_idx + 1;
        // Map back to original video time
        float t0 = 0, t1 = 0;
        if (frame0_idx < num_selected && selected_indices[frame0_idx] < total_frames) {
            t0 = (video.source_fps > 0) ? selected_indices[frame0_idx] / video.source_fps : frame0_idx;
        }
        if (frame1_idx < num_selected && selected_indices[frame1_idx] < total_frames) {
            t1 = (video.source_fps > 0) ? selected_indices[frame1_idx] / video.source_fps : frame1_idx;
        } else {
            t1 = t0;
        }
        result.timestamps[gt] = (t0 + t1) / 2.0f;
    }

    // --- 7. Extract patches in merge-friendly order ---
    // For each temporal group (gt), extract spatial patches from frames [gt*T, gt*T+1]
    // Order: for each spatial block (bh, bw), for each intra (ih, iw), for each (c, t, ph, pw)
    int blocks_h = result.grid_h / merge;
    int blocks_w = result.grid_w / merge;

    int patch_idx = 0;
    for (int gt = 0; gt < result.grid_t; gt++) {
        for (int bh = 0; bh < blocks_h; bh++) {
            for (int bw = 0; bw < blocks_w; bw++) {
                for (int ih = 0; ih < merge; ih++) {
                    for (int iw = 0; iw < merge; iw++) {
                        int row = bh * merge + ih;
                        int col = bw * merge + iw;

                        uint16_t* dst = result.pixel_values_bf16.data() + patch_idx * patch_dim;
                        int d = 0;

                        for (int c = 0; c < C; c++) {
                            for (int t = 0; t < T; t++) {
                                int frame_idx = gt * T + t;
                                auto& frame = resized_frames[frame_idx];
                                for (int ph = 0; ph < P; ph++) {
                                    for (int pw = 0; pw < P; pw++) {
                                        int py = row * P + ph;
                                        int px = col * P + pw;
                                        dst[d++] = float_to_bf16(frame[c * h_bar * w_bar + py * w_bar + px]);
                                    }
                                }
                            }
                        }

                        patch_idx++;
                    }
                }
            }
        }
    }

    // Precompute RoPE positions on CPU (avoids CPU loop in GPU forward path)
    int patches_per_frame = result.grid_h * result.grid_w;
    result.positions_hw.resize(total_patches * 2);
    for (int i = 0; i < total_patches; i++) {
        int frame_i = i % patches_per_frame;
        int blk_idx = frame_i / (merge * merge);
        int intra_idx = frame_i % (merge * merge);
        int intra_h = intra_idx / merge;
        int intra_w = intra_idx % merge;
        int block_h = blk_idx / blocks_w;
        int block_w = blk_idx % blocks_w;
        result.positions_hw[i * 2]     = block_h * merge + intra_h;
        result.positions_hw[i * 2 + 1] = block_w * merge + intra_w;
    }

    return result;
}

// ============================================================================
// Compute Video Grid (lightweight, no pixel processing)
// ============================================================================
std::tuple<int, int, int> VisionEncoder::compute_video_grid(
    int num_frames, int height, int width, const VisionConfig& config) {
    auto [t_bar, h_bar, w_bar] = smart_resize_video(
        num_frames, height, width,
        config.temporal_patch_size, config.factor(),
        config.min_pixels, config.max_pixels);
    int grid_t = t_bar / config.temporal_patch_size;
    int grid_h = h_bar / config.patch_size;
    int grid_w = w_bar / config.patch_size;
    return {grid_t, grid_h, grid_w};
}

// ============================================================================
// GPU Forward Pass
// ============================================================================

void VisionEncoder::patch_embed_forward(__nv_bfloat16* out, const __nv_bfloat16* patches,
                                         int num_patches, cudaStream_t stream) {
    // GEMM: [N, 1536] × [1152, 1536]^T → [N, 1152]
    // Using cuBLAS: C = alpha * A * B^T + beta * C
    // A = patches [N, 1536], B = weight [1152, 1536], C = out [N, 1152]
    cublasHandle_t handle = get_cublas();
    cublasSetStream(handle, stream);

    int M = num_patches;       // N
    int N_dim = config_.hidden_size;  // 1152
    int K = config_.patch_input_dim(); // 1536

    // cuBLAS uses column-major, so we compute:
    // C^T = B * A^T  →  (N_dim, M) = (N_dim, K) * (K, M)
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_T,   // B^T → B (since B is [N_dim, K] row-major = [K, N_dim] col-major)
                 CUBLAS_OP_N,   // A stays as-is
                 N_dim, M, K,
                 &alpha,
                 patch_proj_w_, CUDA_R_16BF, K,   // B: [N_dim, K] row-major
                 patches, CUDA_R_16BF, K,          // A: [M, K] row-major → [K, M] col-major
                 &beta,
                 out, CUDA_R_16BF, N_dim,          // C: [M, N_dim] row-major → [N_dim, M] col-major
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Add bias
    vision_ops::invoke_add_bias(out, patch_proj_b_, num_patches, config_.hidden_size, stream);
}

void VisionEncoder::add_position_embedding(__nv_bfloat16* hidden, int grid_h, int grid_w,
                                            int num_patches, cudaStream_t stream) {
    // Interpolate learned position embeddings and add to hidden states
    // Use workspace for interpolated embeddings? No — compute inline with add
    // Actually, the interpolation kernel writes to a temp buffer, then we add.
    // For simplicity, use the pos_embed_interp kernel which writes interpolated values,
    // then add in-place.

    // Allocate temp on same workspace — but we're called before workspace is set up.
    // Instead, make the interp kernel add directly to hidden.

    // Actually, let's just use a simpler approach: compute interpolated embeddings
    // into a temp region, then add.
    // We'll use the pos_embed region of workspace for this.

    // For now, use a separate kernel that both interpolates and adds
    vision_ops::invoke_pos_embed_interp(hidden, pos_embed_w_,
                                         config_.num_position_embeddings,
                                         config_.pos_grid_size(),
                                         grid_h, grid_w, num_patches,
                                         config_.hidden_size,
                                         config_.spatial_merge_size,
                                         num_patches,  // patches_per_frame = num_patches for this unused method
                                         stream);
    // Note: invoke_pos_embed_interp currently WRITES (not adds).
    // We need to modify this to ADD to hidden states.
    // See the kernel — it writes to out. We need a fused version.
    // For now, we'll fix this by making the caller handle it properly.
}

void VisionEncoder::block_forward(int block_idx, __nv_bfloat16* hidden,
                                    __nv_bfloat16* workspace,
                                    const float* rope_cos, const float* rope_sin,
                                    int num_patches, cudaStream_t stream) {
    auto& b = blocks_[block_idx];
    int hs = config_.hidden_size;        // 1152
    int num_heads = config_.num_heads;   // 16
    int head_dim = config_.head_dim;     // 72
    int is = config_.intermediate_size;  // 4304
    int N = num_patches;

    // Large chunk: ≤ 256 MB scores budget (vs 32 MB before)
    // N=3888 → chunk=1024 → 4 chunks (vs 31 chunks at 32 MB limit)
    int attn_chunk = compute_attn_chunk(N, num_heads);

    // Workspace layout (cuBLAS attention with large chunk + fused kernels):
    __nv_bfloat16* norm_out   = workspace;                                     // [N, 1152]
    __nv_bfloat16* qkv        = norm_out + N * hs;                             // [N, 3456]
    __nv_bfloat16* q_trans    = qkv + N * 3 * hs;                              // [16, N, 72]
    __nv_bfloat16* k_trans    = q_trans + num_heads * N * head_dim;             // [16, N, 72]
    __nv_bfloat16* v_trans    = k_trans + num_heads * N * head_dim;             // [16, N, 72]
    float*         scores     = (float*)(v_trans + num_heads * N * head_dim);   // [16, chunk, N] FP32
    __nv_bfloat16* attn_out_t = (__nv_bfloat16*)((char*)scores
                                + (size_t)num_heads * attn_chunk * N * sizeof(float)); // [16, N, 72]
    __nv_bfloat16* attn_out   = attn_out_t + num_heads * N * head_dim;         // [N, 1152]
    __nv_bfloat16* mlp_buf    = attn_out + N * hs;                              // [N, 4304]

    cublasHandle_t handle = get_cublas();
    cublasSetStream(handle, stream);

    // ---- 1. LayerNorm 1 (hidden stays as residual) ----
    vision_ops::invoke_layernorm(norm_out, hidden, b.norm1_w, b.norm1_b,
                                  config_.layernorm_eps, N, hs, stream);

    // ---- 2. QKV Projection: [N, 1152] × [3456, 1152]^T → [N, 3456] ----
    {
        int M = N, N_dim = 3 * hs, K = hs;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     b.qkv_w, CUDA_R_16BF, K,
                     norm_out, CUDA_R_16BF, K,
                     &beta, qkv, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ---- 3. QKV add_bias ----
    vision_ops::invoke_add_bias(qkv, b.qkv_b, N, 3 * hs, stream);

    // ---- 4. Fused Split QKV + Transpose: [N, 3*H*D] → Q[H,N,D], K[H,N,D], V[H,N,D] ----
    {
        dim3 grid(N, num_heads);
        split_qkv_transpose_kernel<<<grid, head_dim, 0, stream>>>(
            q_trans, k_trans, v_trans, qkv, N, num_heads, head_dim);
    }

    // ---- 5. Apply 2D RoPE to Q and K in [H, N, D] layout ----
    vision_ops::invoke_vision_rope(q_trans, k_trans, rope_cos, rope_sin,
                                    N, num_heads, head_dim, config_.rotary_dim, stream);

    // ---- 6. Attention: Q×K^T → softmax → ×V (cuBLAS, large chunk) ----
    // With 256 MB budget: chunk=1024 for N=3888 → 4 chunks instead of 31
    {
        float scale = 1.0f / std::sqrt((float)head_dim);
        float beta_zero = 0.0f;
        float alpha_one = 1.0f;
        long long strideK = (long long)N * head_dim;
        long long strideV = (long long)N * head_dim;
        long long strideO = (long long)N * head_dim;

        for (int q_start = 0; q_start < N; q_start += attn_chunk) {
            int q_len = std::min(attn_chunk, N - q_start);

            __nv_bfloat16* q_chunk = q_trans + q_start * head_dim;
            long long strideQ = (long long)N * head_dim;
            long long strideS = (long long)q_len * N;

            // -- QK^T: scores[H, q_len, N] = Q_chunk × K^T × scale --
            cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, q_len, head_dim,
                &scale,
                k_trans, CUDA_R_16BF, head_dim, strideK,
                q_chunk, CUDA_R_16BF, head_dim, strideQ,
                &beta_zero,
                scores, CUDA_R_32F, N, strideS,
                num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

            // -- Softmax + FP32→BF16 cast --
            int num_rows = num_heads * q_len;
            int block_s = std::min(N, 1024);
            softmax_kernel<<<num_rows, block_s, 0, stream>>>(scores, N);

            int total_elements = num_heads * q_len * N;
            int grid_c = (total_elements + 255) / 256;
            f32_to_bf16_inplace_kernel<<<grid_c, 256, 0, stream>>>((void*)scores, total_elements);

            // -- PV: O_chunk[H, q_len, D] = P_chunk × V --
            __nv_bfloat16* scores_bf16 = (__nv_bfloat16*)scores;
            __nv_bfloat16* o_chunk = attn_out_t + q_start * head_dim;
            long long strideP = (long long)q_len * N;

            cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, q_len, N,
                &alpha_one,
                v_trans, CUDA_R_16BF, head_dim, strideV,
                scores_bf16, CUDA_R_16BF, N, strideP,
                &beta_zero,
                o_chunk, CUDA_R_16BF, head_dim, strideO,
                num_heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }

    // ---- 7. Transpose attention output: [H, N, D] → [N, H, D] ----
    {
        dim3 grid(N, num_heads);
        transpose_hnd_to_nhd_kernel<<<grid, head_dim, 0, stream>>>(
            attn_out, attn_out_t, N, num_heads, head_dim);
    }

    // ---- 8. Output projection: [N, 1152] × [1152, 1152]^T → [N, 1152] ----
    {
        int M = N, N_dim = hs, K = hs;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     b.proj_w, CUDA_R_16BF, K,
                     attn_out, CUDA_R_16BF, K,
                     &beta, norm_out, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ---- 9. Fused: hidden += (proj_out + proj_bias); norm_out = LN(hidden) ----
    vision_ops::invoke_fused_add_bias_layernorm(norm_out, hidden, norm_out,
                                                 b.proj_b, b.norm2_w, b.norm2_b,
                                                 config_.layernorm_eps, N, hs, stream);

    // ---- 10. FC1: [N, 1152] × [4304, 1152]^T → [N, 4304] ----
    {
        int M = N, N_dim = is, K = hs;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     b.fc1_w, CUDA_R_16BF, K,
                     norm_out, CUDA_R_16BF, K,
                     &beta, mlp_buf, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ---- 11. Fused bias + GELU(tanh) ----
    vision_ops::invoke_add_bias_gelu_tanh(mlp_buf, b.fc1_b, N, is, stream);

    // ---- 12. FC2: [N, 4304] × [1152, 4304]^T → [N, 1152] ----
    {
        int M = N, N_dim = hs, K = is;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     b.fc2_w, CUDA_R_16BF, K,
                     mlp_buf, CUDA_R_16BF, K,
                     &beta, norm_out, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }

    // ---- 13. Fused: hidden += (fc2_out + fc2_bias) ----
    vision_ops::invoke_add_bias_residual(hidden, norm_out, b.fc2_b, N, hs, stream);
}

void VisionEncoder::merger_forward(__nv_bfloat16* out, const __nv_bfloat16* hidden,
                                    __nv_bfloat16* workspace, int num_patches,
                                    cudaStream_t stream) {
    int hs = config_.hidden_size;         // 1152
    int merger_hs = config_.merger_hidden(); // 4608
    int out_hs = config_.out_hidden_size;    // 5120
    int merge = config_.spatial_merge_size;  // 2
    int N_out = num_patches / (merge * merge);

    __nv_bfloat16* norm_buf = workspace;                    // [N, 1152]
    __nv_bfloat16* merged   = norm_buf + num_patches * hs;  // [N/4, 4608]
    __nv_bfloat16* fc1_out  = merged + N_out * merger_hs;   // [N/4, 4608]

    cublasHandle_t handle = get_cublas();
    cublasSetStream(handle, stream);

    // ---- LayerNorm (pre-shuffle) ----
    // Apply LayerNorm to each patch independently (on 1152 dims)
    vision_ops::invoke_layernorm(norm_buf, hidden, merger_norm_w_, merger_norm_b_,
                                  config_.layernorm_eps, num_patches, hs, stream);

    // ---- Reshape: merge 4 consecutive patches ----
    // Since patches are already in merge-friendly order, view(-1, 4608) = just reinterpret
    // norm_buf [N, 1152] with N = N_out * 4 → merged [N_out, 4608]
    // This is a no-op reshape — the data is contiguous and already in the right order!
    merged = norm_buf;  // Just reinterpret

    // ---- FC1: [N/4, 4608] × [4608, 4608]^T → [N/4, 4608] ----
    {
        int M = N_out, N_dim = merger_hs, K = merger_hs;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     merger_fc1_w_, CUDA_R_16BF, K,
                     merged, CUDA_R_16BF, K,
                     &beta, fc1_out, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    vision_ops::invoke_add_bias(fc1_out, merger_fc1_b_, N_out, merger_hs, stream);

    // Standard GELU (not tanh approximation for merger)
    vision_ops::invoke_gelu(fc1_out, N_out * merger_hs, stream);

    // ---- FC2: [N/4, 4608] × [5120, 4608]^T → [N/4, 5120] ----
    {
        int M = N_out, N_dim = out_hs, K = merger_hs;
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N_dim, M, K, &alpha,
                     merger_fc2_w_, CUDA_R_16BF, K,
                     fc1_out, CUDA_R_16BF, K,
                     &beta, out, CUDA_R_16BF, N_dim,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    vision_ops::invoke_add_bias(out, merger_fc2_b_, N_out, out_hs, stream);
}

__nv_bfloat16* VisionEncoder::forward(const ProcessedImage& image,
                                       __nv_bfloat16* workspace,
                                       size_t workspace_bytes_avail,
                                       cudaStream_t stream) {
    int N = image.num_patches();
    int N_out = image.num_output_tokens();
    int hs = config_.hidden_size;   // 1152
    int out_hs = config_.out_hidden_size; // 5120

    size_t needed = workspace_bytes(N);
    if (workspace_bytes_avail < needed) {
        std::cerr << "[Vision] Workspace too small: need " << needed
                  << " bytes, have " << workspace_bytes_avail << std::endl;
        return nullptr;
    }

    // ---- Timing ----
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    // ---- cuBLAS warmup on first call ----
    // First cuBLAS call includes internal workspace allocation + algorithm selection.
    // Pre-warm with representative shapes to avoid ~100ms overhead in the timed section.
    if (!cublas_warmed_up_) {
        cublasHandle_t h = get_cublas();
        cublasSetStream(h, stream);
        // Use start of workspace as scratch (will be overwritten anyway)
        __nv_bfloat16* scratch = workspace;
        cudaMemsetAsync(scratch, 0, 4096, stream);
        float alpha = 1.0f, beta = 0.0f;
        // Warm up GemmEx (used for projections)
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                     128, 128, 128, &alpha,
                     scratch, CUDA_R_16BF, 128,
                     scratch, CUDA_R_16BF, 128,
                     &beta, scratch, CUDA_R_16BF, 128,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        // Warm up StridedBatchedEx (used for attention QK/PV)
        cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            128, 128, 72, &alpha,
            scratch, CUDA_R_16BF, 72, 0,
            scratch, CUDA_R_16BF, 72, 0,
            &beta, scratch, CUDA_R_32F, 128, 0,
            16, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaStreamSynchronize(stream);
        cublas_warmed_up_ = true;
    }

    cudaEventRecord(ev_start, stream);

    // ---- Workspace allocation ----
    __nv_bfloat16* d_patches = workspace;  // [N, 1536]
    __nv_bfloat16* hidden = d_patches + N * config_.patch_input_dim();  // [N, 1152]
    __nv_bfloat16* block_workspace = hidden + N * hs;

    // Block workspace (cuBLAS attention, L2-fit chunk, fused MLP kernels)
    int attn_chunk = compute_attn_chunk(N, config_.num_heads);
    size_t block_ws_used = (size_t)N * hs * 2              // norm_out [N, hs]
                         + (size_t)N * 3 * hs * 2          // QKV
                         + (size_t)config_.num_heads * N * config_.head_dim * 2 * 3  // Q/K/V transposed
                         + (size_t)config_.num_heads * attn_chunk * N * 4  // scores (FP32, large chunk)
                         + (size_t)config_.num_heads * N * config_.head_dim * 2  // attn_out_t
                         + (size_t)N * hs * 2              // attn_out
                         + (size_t)N * config_.intermediate_size * 2;  // MLP
    __nv_bfloat16* rope_workspace = block_workspace + block_ws_used / 2;
    float* rope_cos = (float*)rope_workspace;                      // [N, 36]
    float* rope_sin = rope_cos + N * config_.rotary_dim;           // [N, 36]
    int* positions_hw = (int*)(rope_sin + N * config_.rotary_dim); // [N, 2]
    __nv_bfloat16* pos_embed_buf = (__nv_bfloat16*)(positions_hw + N * 2); // [N, 1152]
    __nv_bfloat16* output = pos_embed_buf + N * hs;  // [N_out, 5120]

    // ---- Upload BF16 pixel values directly (converted on CPU during preprocessing) ----
    size_t bf16_size = N * config_.patch_input_dim() * sizeof(__nv_bfloat16);
    cudaMemcpyAsync(d_patches, image.pixel_values_bf16.data(),
                    bf16_size, cudaMemcpyHostToDevice, stream);

    // ---- Patch Embedding ----
    patch_embed_forward(hidden, d_patches, N, stream);

    // ---- Position Embeddings ----
    int patches_per_frame = image.grid_h * image.grid_w;
    vision_ops::invoke_pos_embed_interp(pos_embed_buf, pos_embed_w_,
                                         config_.num_position_embeddings,
                                         config_.pos_grid_size(),
                                         image.grid_h, image.grid_w, N,
                                         hs, config_.spatial_merge_size,
                                         patches_per_frame, stream);
    vision_ops::invoke_add(hidden, hidden, pos_embed_buf, N * hs, stream);

    // ---- Upload precomputed 2D RoPE positions (built during CPU preprocessing) ----
    cudaMemcpyAsync(positions_hw, image.positions_hw.data(), N * 2 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    vision_ops::invoke_compute_vision_rope_table(rope_cos, rope_sin, positions_hw,
                                                   N, config_.rotary_dim,
                                                   config_.rope_theta,
                                                   image.grid_h, image.grid_w, stream);

    // ---- 27 ViT Blocks (L2-fit chunk attention + fused MLP kernels, no memset) ----
    for (int i = 0; i < config_.depth; i++) {
        block_forward(i, hidden, block_workspace, rope_cos, rope_sin, N, stream);
    }

    // ---- Merger ----
    merger_forward(output, hidden, block_workspace, N, stream);

    // ---- Report timing ----
    cudaEventRecord(ev_end, stream);
    cudaEventSynchronize(ev_end);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_end);
    printf("[Vision] Forward: %.0f ms (%d patches, chunk=%d, %d output tokens)\n",
           elapsed_ms, N, attn_chunk, N_out);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    return output;
}

} // namespace core
} // namespace qwen_thor
