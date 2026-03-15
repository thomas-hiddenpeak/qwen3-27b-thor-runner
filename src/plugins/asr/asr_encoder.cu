// asr_encoder.cu — Qwen3-ASR Audio Encoder 实现
//
// Conv2D frontend + 24-layer bidirectional Transformer + post-projection
// Conv1 uses naive kernel (C_in=1), Conv2/Conv3 use im2col + cuBLAS GEMM (tensor core)

#include "asr_encoder.h"
#include "audio_ops.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace qwen_thor {
namespace asr {

// ============================================================================
// Constants
// ============================================================================

// Conv 处理批大小: 每次处理这么多 chunk, 平衡内存和效率
static constexpr int CONV_BATCH_SIZE = 8;

// 正弦位置编码最大长度
static constexpr int MAX_PE_LENGTH = 1500;

// 最大 attention window 分段数
static constexpr int MAX_SEGMENTS = 128;

// ============================================================================
// CUDA Helpers
// ============================================================================

__device__ __forceinline__ float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================================
// Conv2D kernel (k=3, stride=2, pad=1) with fused GELU
// ============================================================================
// Input:  [N, C_in, H, W]
// Weight: [C_out, C_in, 3, 3]
// Bias:   [C_out] or nullptr
// Output: [N, C_out, H_out, W_out]
//   H_out = (H-1)/2+1, W_out = (W-1)/2+1

__global__ void conv2d_gelu_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    bool apply_gelu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_o = idx % W_out;
    int h_o = (idx / W_out) % H_out;
    int c_o = (idx / (W_out * H_out)) % C_out;
    int n   = idx / (W_out * H_out * C_out);

    float sum = bias ? bf16_to_f32(bias[c_o]) : 0.0f;

    const int weight_plane = 3 * 3;   // kH * kW
    const int in_plane = H_in * W_in;

    for (int c_i = 0; c_i < C_in; c_i++) {
        const __nv_bfloat16* w_ptr = weight + ((size_t)c_o * C_in + c_i) * weight_plane;
        const __nv_bfloat16* in_ptr = input + ((size_t)n * C_in + c_i) * in_plane;

        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int h_i = h_o * 2 - 1 + kh;
            if (h_i < 0 || h_i >= H_in) continue;
            #pragma unroll
            for (int kw = 0; kw < 3; kw++) {
                int w_i = w_o * 2 - 1 + kw;
                if (w_i < 0 || w_i >= W_in) continue;
                sum += bf16_to_f32(in_ptr[h_i * W_in + w_i])
                     * bf16_to_f32(w_ptr[kh * 3 + kw]);
            }
        }
    }

    if (apply_gelu) {
        sum = sum * 0.5f * (1.0f + erff(sum * 0.7071067811865476f));
    }

    output[((size_t)n * C_out + c_o) * (H_out * W_out) + h_o * W_out + w_o] = f32_to_bf16(sum);
}

// ============================================================================
// Add bias kernel: x[i] += bias[i % dim]
// ============================================================================

__global__ void add_bias_kernel(
    __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    int num_tokens, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * dim) return;
    int d = idx % dim;
    x[idx] = f32_to_bf16(bf16_to_f32(x[idx]) + bf16_to_f32(bias[d]));
}

// ============================================================================
// im2col kernel for Conv2D k=3, stride=2, padding=1
// Input:  [N, C_in, H, W]  NCHW
// Output: [N*H_out*W_out, C_in*9]  row-major
// ============================================================================

__global__ void im2col_k3s2p1_kernel(
    __nv_bfloat16* __restrict__ col,
    const __nv_bfloat16* __restrict__ input,
    int N, int C_in, int H, int W, int H_out, int W_out)
{
    int M = N * H_out * W_out;
    int K = C_in * 9;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;

    int k = idx % K;
    int m = idx / K;

    int w_out = m % W_out;
    int h_out = (m / W_out) % H_out;
    int n = m / (H_out * W_out);

    int kw = k % 3;
    int kh = (k / 3) % 3;
    int c = k / 9;

    int h_in = h_out * 2 - 1 + kh;  // stride=2, pad=1
    int w_in = w_out * 2 - 1 + kw;

    __nv_bfloat16 val = __float2bfloat16(0.0f);
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((size_t)n * C_in + c) * H * W + (size_t)h_in * W + w_in];
    }
    col[idx] = val;
}

// ============================================================================
// NHWC → NCHW + bias + GELU (fused)
// src: [N*H*W, C] row-major (GEMM output)
// dst: [N, C, H, W] NCHW
// ============================================================================

__global__ void nhwc_to_nchw_bias_gelu_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ bias,
    int N, int C, int H, int W)
{
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // NCHW decomposition
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (C * H * W);

    // NHWC source: row = n*H*W + h*W + w, col = c
    int src_idx = ((n * H + h) * W + w) * C + c;
    float val = __bfloat162float(src[src_idx]) + __bfloat162float(bias[c]);

    // GELU (exact)
    val = val * 0.5f * (1.0f + erff(val * 0.7071067811865476f));

    dst[idx] = __float2bfloat16(val);
}

// ============================================================================
// Copy mel chunks from [128, total_frames] to padded [batch, 128, padded_len]
// ============================================================================

__global__ void copy_mel_chunks_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ chunk_starts,
    const int* __restrict__ chunk_lens,
    int mel_bins, int padded_len, int total_frames)
{
    int b = blockIdx.y;
    int mel_bin = blockIdx.x;
    int start = chunk_starts[b];
    int len = chunk_lens[b];

    for (int t = threadIdx.x; t < padded_len; t += blockDim.x) {
        float val = 0.0f;
        if (t < len) {
            val = bf16_to_f32(src[(size_t)mel_bin * total_frames + start + t]);
        }
        dst[((size_t)b * mel_bins + mel_bin) * padded_len + t] = f32_to_bf16(val);
    }
}

// ============================================================================
// Transpose [N, C, F, T] → [N, T, C*F]
// For conv output reshape before conv_out linear projection
// ============================================================================

__global__ void transpose_ncft_to_ntcf_kernel(
    __nv_bfloat16* __restrict__ dst,        // [N, T, C*F]
    const __nv_bfloat16* __restrict__ src,  // [N, C, F, T]
    int N, int C, int F, int T)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T * C * F;
    if (idx >= total) return;

    int f_ = idx % F;
    int c_ = (idx / F) % C;
    int t_ = (idx / (F * C)) % T;
    int n_ = idx / (F * C * T);

    // dst[n, t, c*F + f] = src[n, c, f, t]
    dst[(size_t)n_ * (T * C * F) + t_ * (C * F) + c_ * F + f_] =
        src[(size_t)n_ * (C * F * T) + c_ * (F * T) + f_ * T + t_];
}

// ============================================================================
// GELU kernel (in-place)
// ============================================================================

__global__ void gelu_inplace_kernel(__nv_bfloat16* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = bf16_to_f32(x[idx]);
    v = v * 0.5f * (1.0f + erff(v * 0.7071067811865476f));
    x[idx] = f32_to_bf16(v);
}

// ============================================================================
// BF16 linear projection: out = input @ weight^T + bias
// ============================================================================
// input:  [M, K] row-major BF16
// weight: [N, K] row-major BF16
// bias:   [N] BF16 or nullptr
// out:    [M, N] row-major BF16

static void bf16_linear(
    cublasHandle_t handle,
    __nv_bfloat16* out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
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
    if (bias) {
        int total = M * N;
        add_bias_kernel<<<(total + 255) / 256, 256, 0, stream>>>(out, bias, M, N);
    }
}

// ============================================================================
// AudioEncoder implementation
// ============================================================================

AudioEncoder::AudioEncoder(const ASRConfig& config)
    : config_(config), layer_weights_(config.encoder_layers) {}

AudioEncoder::~AudioEncoder() {
    if (workspace_) cudaFree(workspace_);
    if (pe_table_) cudaFree(pe_table_);
    if (im2col_buf_) cudaFree(im2col_buf_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
}

void AudioEncoder::set_conv_weights(
    __nv_bfloat16* conv2d1_w, __nv_bfloat16* conv2d1_b,
    __nv_bfloat16* conv2d2_w, __nv_bfloat16* conv2d2_b,
    __nv_bfloat16* conv2d3_w, __nv_bfloat16* conv2d3_b,
    __nv_bfloat16* conv_out_w)
{
    conv2d1_w_ = conv2d1_w; conv2d1_b_ = conv2d1_b;
    conv2d2_w_ = conv2d2_w; conv2d2_b_ = conv2d2_b;
    conv2d3_w_ = conv2d3_w; conv2d3_b_ = conv2d3_b;
    conv_out_w_ = conv_out_w;
}

void AudioEncoder::set_post_weights(
    __nv_bfloat16* ln_post_w, __nv_bfloat16* ln_post_b,
    __nv_bfloat16* proj1_w, __nv_bfloat16* proj1_b,
    __nv_bfloat16* proj2_w, __nv_bfloat16* proj2_b)
{
    ln_post_w_ = ln_post_w; ln_post_b_ = ln_post_b;
    proj1_w_ = proj1_w;     proj1_b_ = proj1_b;
    proj2_w_ = proj2_w;     proj2_b_ = proj2_b;
}

void AudioEncoder::set_layer_weights(int layer_idx, const EncoderLayerWeights& weights) {
    layer_weights_[layer_idx] = weights;
}

// ============================================================================
// initialize: allocate workspace + precompute sinusoidal PE
// ============================================================================

void AudioEncoder::initialize(cudaStream_t stream) {
    if (initialized_) return;

    // cuBLAS handle
    cublasCreate(&cublas_handle_);

    int d = config_.encoder_d_model;  // 1024
    int max_tokens = config_.max_source_positions;  // 1500
    int dhs = config_.downsample_hidden_size;  // 480
    int mel_bins = config_.num_mel_bins;  // 128
    int chunk_len = config_.n_window * 2;  // 100

    // Conv output spatial dims for full chunk
    int h1 = ASRConfig::conv_output_size(mel_bins);   // 64
    int w1 = ASRConfig::conv_output_size(chunk_len);   // 50
    int h2 = ASRConfig::conv_output_size(h1);          // 32
    int w2 = ASRConfig::conv_output_size(w1);          // 25
    int h3 = ASRConfig::conv_output_size(h2);          // 16
    int w3 = ASRConfig::conv_output_size(w2);          // 13

    // Conv buffer size (per chunk): max is after conv2d1
    size_t conv_buf_per_chunk = (size_t)dhs * h1 * w1;  // 480*64*50 = 1,536,000
    size_t conv_buf_size = CONV_BATCH_SIZE * conv_buf_per_chunk;

    // Workspace layout (all BF16 elements):
    //   [conv_buf_a]    conv_buf_size
    //   [conv_buf_b]    conv_buf_size
    //   [mel_padded]    CONV_BATCH_SIZE * mel_bins * chunk_len
    //   [reshape_buf]   CONV_BATCH_SIZE * w3 * dhs * h3  (= conv_buf for conv_out input)
    //   [hidden_states] max_tokens * d
    //   [norm_buf]      max_tokens * d
    //   [q_buf]         max_tokens * d
    //   [k_buf]         max_tokens * d
    //   [v_buf]         max_tokens * d
    //   [ffn_buf]       max_tokens * encoder_ffn_dim
    //   (padding for int arrays)
    //   [cu_seqlens]    (MAX_SEGMENTS+1) * sizeof(int) / sizeof(bf16)
    //   [chunk_meta]    (max_chunks * 2) * sizeof(int) / sizeof(bf16)

    size_t mel_padded_size = (size_t)CONV_BATCH_SIZE * mel_bins * chunk_len;
    size_t hidden_size = (size_t)max_tokens * d;
    size_t ffn_size = (size_t)max_tokens * config_.encoder_ffn_dim;

    // Integer arrays (stored as BF16 slots, rounded up)
    int max_chunks = (max_tokens + 12) / 13 + 1;  // ~116
    size_t int_slots = ((MAX_SEGMENTS + 1 + max_chunks * 2) * sizeof(int)
                        + sizeof(__nv_bfloat16) - 1) / sizeof(__nv_bfloat16);

    workspace_size_ = conv_buf_size * 2
                    + mel_padded_size
                    + hidden_size * 5   // hidden + norm + Q + K + V
                    + ffn_size
                    + int_slots
                    + 1024;  // alignment padding

    cudaMalloc(&workspace_, workspace_size_ * sizeof(__nv_bfloat16));
    cudaMemset(workspace_, 0, workspace_size_ * sizeof(__nv_bfloat16));

    // im2col buffer for Conv2D im2col+GEMM
    // Largest is Conv2: batch*h2*w2 * C_in*9 (im2col) + batch*h2*w2 * C_out (NHWC output)
    size_t im2col_size = (size_t)CONV_BATCH_SIZE * h2 * w2 * (dhs * 9 + dhs);
    cudaMalloc(&im2col_buf_, im2col_size * sizeof(__nv_bfloat16));

    // Sinusoidal PE: [max_source_positions, d_model]
    cudaMalloc(&pe_table_, (size_t)max_tokens * d * sizeof(__nv_bfloat16));
    audio_ops::compute_sinusoidal_pe(pe_table_, max_tokens, d, 10000.0f, stream);

    cudaStreamSynchronize(stream);
    initialized_ = true;

    fprintf(stderr, "[ASR Encoder] initialized: workspace %.1f MB, PE [%d, %d]\n",
            workspace_size_ * sizeof(__nv_bfloat16) / (1024.0 * 1024.0),
            max_tokens, d);
}

// ============================================================================
// conv2d_forward: single Conv2D layer (k=3, stride=2, pad=1)
// For C_in=1: naive per-element kernel (trivial compute)
// For C_in>=2: im2col + cuBLAS GEMM + NHWC→NCHW transpose (tensor core path)
// ============================================================================

void AudioEncoder::conv2d_forward(
    const __nv_bfloat16* input, int batch, int C_in, int H_in, int W_in,
    const __nv_bfloat16* weight, const __nv_bfloat16* bias, int C_out,
    __nv_bfloat16* output, cudaStream_t stream)
{
    int H_out = ASRConfig::conv_output_size(H_in);
    int W_out = ASRConfig::conv_output_size(W_in);
    int total = batch * C_out * H_out * W_out;
    if (total == 0) return;

    if (C_in <= 1) {
        // Naive kernel: 9 FMA per element, output-BW bound
        conv2d_gelu_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            input, weight, bias, output,
            batch, C_in, H_in, W_in,
            C_out, H_out, W_out, true);
        return;
    }

    // im2col + GEMM path for large C_in (e.g., 480)
    int M = batch * H_out * W_out;
    int K = C_in * 9;

    // 1. im2col: input[N, C_in, H, W] → im2col_buf_[M, K]
    int im2col_total = M * K;
    im2col_k3s2p1_kernel<<<(im2col_total + 255) / 256, 256, 0, stream>>>(
        im2col_buf_, input,
        batch, C_in, H_in, W_in, H_out, W_out);

    // 2. GEMM: im2col_buf_[M, K] × weight^T[K, C_out] → nhwc_out[M, C_out]
    //    nhwc_out is stored at im2col_buf_ + M*K (non-overlapping with im2col data)
    __nv_bfloat16* nhwc_out = im2col_buf_ + (size_t)M * K;

    cublasSetStream(cublas_handle_, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle_,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 C_out, M, K,
                 &alpha,
                 weight, CUDA_R_16BF, K,       // [C_out, K] row = [K, C_out] col
                 im2col_buf_, CUDA_R_16BF, K,   // [M, K] row = [K, M] col
                 &beta,
                 nhwc_out, CUDA_R_16BF, C_out,  // [M, C_out] row = [C_out, M] col
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // 3. NHWC→NCHW + bias + GELU
    nhwc_to_nchw_bias_gelu_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        output, nhwc_out, bias,
        batch, C_out, H_out, W_out);
}

// ============================================================================
// encoder_layer_forward: single Transformer encoder layer
// ============================================================================
//
// PRE-LN: LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual
// Attention is bidirectional with cu_seqlens segmentation

void AudioEncoder::encoder_layer_forward(
    int layer_idx,
    __nv_bfloat16* hidden_states,  // [seq_len, d_model], in-place
    int seq_len,
    const int* cu_seqlens,
    int num_segments,
    __nv_bfloat16* workspace_base,
    cudaStream_t stream)
{
    const auto& lw = layer_weights_[layer_idx];
    int d = config_.encoder_d_model;                 // 1024
    int num_heads = config_.encoder_attention_heads;  // 16
    int head_dim = config_.encoder_head_dim;          // 64
    int ffn_dim = config_.encoder_ffn_dim;            // 4096
    float eps = 1e-5f;  // LayerNorm epsilon (standard)

    // Workspace sub-buffers
    __nv_bfloat16* norm_buf = workspace_base;
    __nv_bfloat16* q_buf    = norm_buf + (size_t)seq_len * d;
    __nv_bfloat16* k_buf    = q_buf    + (size_t)seq_len * d;
    __nv_bfloat16* v_buf    = k_buf    + (size_t)seq_len * d;
    // FFN buffer (reuse norm_buf area + extra)
    __nv_bfloat16* ffn_buf  = v_buf    + (size_t)seq_len * d;

    // === Self-Attention ===

    // 1. LayerNorm
    audio_ops::invoke_layernorm(norm_buf, hidden_states,
                                 lw.self_attn_layer_norm_w, lw.self_attn_layer_norm_b,
                                 eps, seq_len, d, stream);

    // 2. Q/K/V projections
    bf16_linear(cublas_handle_, q_buf, norm_buf, lw.q_proj_w, lw.q_proj_b, seq_len, d, d, stream);
    bf16_linear(cublas_handle_, k_buf, norm_buf, lw.k_proj_w, lw.k_proj_b, seq_len, d, d, stream);
    bf16_linear(cublas_handle_, v_buf, norm_buf, lw.v_proj_w, lw.v_proj_b, seq_len, d, d, stream);

    // 3. Bidirectional MHA (Q/K/V already [seq_len, d_model] = [seq_len, 16, 64] reinterpreted)
    // norm_buf is consumed, reuse as attention output
    audio_ops::invoke_bidirectional_mha(
        norm_buf,   // attn_out: [seq_len, num_heads, head_dim]
        q_buf, k_buf, v_buf,
        seq_len, num_heads, head_dim,
        cu_seqlens, num_segments,
        stream);

    // 4. Output projection
    // Reshape attn_out [seq_len, 16*64] → [seq_len, 1024], project to [seq_len, 1024]
    // Reuse q_buf as output buffer
    bf16_linear(cublas_handle_, q_buf, norm_buf, lw.o_proj_w, lw.o_proj_b, seq_len, d, d, stream);

    // 5. Residual add
    audio_ops::invoke_add_residual(hidden_states, q_buf, seq_len * d, stream);

    // === FFN ===

    // 6. LayerNorm
    audio_ops::invoke_layernorm(norm_buf, hidden_states,
                                 lw.final_layer_norm_w, lw.final_layer_norm_b,
                                 eps, seq_len, d, stream);

    // 7. FC1 + GELU: [seq_len, 1024] → [seq_len, 4096]
    bf16_linear(cublas_handle_, ffn_buf, norm_buf, lw.fc1_w, lw.fc1_b, seq_len, d, ffn_dim, stream);
    audio_ops::invoke_gelu(ffn_buf, ffn_buf, seq_len * ffn_dim, stream);

    // 8. FC2: [seq_len, 4096] → [seq_len, 1024]
    bf16_linear(cublas_handle_, norm_buf, ffn_buf, lw.fc2_w, lw.fc2_b, seq_len, ffn_dim, d, stream);

    // 9. Residual add
    audio_ops::invoke_add_residual(hidden_states, norm_buf, seq_len * d, stream);

    // 10. BF16 clamp (prevent overflow, same as Python's fp16 clamp)
    audio_ops::invoke_bf16_clamp(hidden_states, seq_len * d,
                                  -65000.0f, 65000.0f, stream);
}

// ============================================================================
// forward: full encoder pipeline
// ============================================================================
// 输入: mel [128, mel_frames] on GPU
// 输出: encoder_out [out_seq_len, 2048] on GPU

void AudioEncoder::forward(
    const __nv_bfloat16* mel,
    int mel_frames,
    __nv_bfloat16* encoder_out,
    int& out_seq_len,
    cudaStream_t stream)
{
    if (!initialized_) {
        fprintf(stderr, "[ASR Encoder] ERROR: not initialized\n");
        return;
    }

    int d = config_.encoder_d_model;               // 1024
    int dhs = config_.downsample_hidden_size;       // 480
    int mel_bins = config_.num_mel_bins;             // 128
    int chunk_mel_len = config_.n_window * 2;        // 100
    int n_window_infer = config_.n_window_infer;     // 800
    int output_dim = config_.output_dim;             // 2048

    // --- Compute chunk parameters on CPU ---
    int chunk_num = (mel_frames + chunk_mel_len - 1) / chunk_mel_len;
    if (chunk_num == 0) {
        out_seq_len = 0;
        return;
    }

    // Chunk lengths and output offsets
    std::vector<int> chunk_starts(chunk_num);
    std::vector<int> chunk_lens(chunk_num);
    std::vector<int> chunk_output_lens(chunk_num);
    std::vector<int> chunk_output_offsets(chunk_num + 1, 0);

    for (int i = 0; i < chunk_num; i++) {
        chunk_starts[i] = i * chunk_mel_len;
        chunk_lens[i] = std::min(chunk_mel_len, mel_frames - i * chunk_mel_len);
        // Handle case where remainder is 0 (last chunk is full)
        if (chunk_lens[i] == 0) chunk_lens[i] = chunk_mel_len;
        chunk_output_lens[i] = ASRConfig::conv_output_size(
            ASRConfig::conv_output_size(
                ASRConfig::conv_output_size(chunk_lens[i])));
    }
    for (int i = 0; i < chunk_num; i++) {
        chunk_output_offsets[i + 1] = chunk_output_offsets[i] + chunk_output_lens[i];
    }

    int total_tokens = chunk_output_offsets[chunk_num];
    out_seq_len = total_tokens;

    if (total_tokens == 0 || total_tokens > config_.max_source_positions) {
        fprintf(stderr, "[ASR Encoder] ERROR: total_tokens=%d (max=%d)\n",
                total_tokens, config_.max_source_positions);
        out_seq_len = std::min(total_tokens, config_.max_source_positions);
        if (total_tokens == 0) return;
    }

    // --- Workspace layout ---
    int h1 = ASRConfig::conv_output_size(mel_bins);
    int w1 = ASRConfig::conv_output_size(chunk_mel_len);
    int h3 = ASRConfig::conv_output_size(ASRConfig::conv_output_size(h1));
    int w3 = ASRConfig::conv_output_size(ASRConfig::conv_output_size(w1));
    size_t conv_buf_per_chunk = (size_t)dhs * h1 * w1;
    size_t conv_buf_total = CONV_BATCH_SIZE * conv_buf_per_chunk;

    __nv_bfloat16* conv_buf_a = workspace_;
    __nv_bfloat16* conv_buf_b = conv_buf_a + conv_buf_total;
    __nv_bfloat16* mel_padded = conv_buf_b + conv_buf_total;
    size_t mel_padded_size = (size_t)CONV_BATCH_SIZE * mel_bins * chunk_mel_len;
    __nv_bfloat16* hidden_states = mel_padded + mel_padded_size;
    __nv_bfloat16* transformer_ws = hidden_states + (size_t)config_.max_source_positions * d;
    // transformer_ws: norm_buf, q, k, v, ffn — used during encoder layers

    // Integer workspace (at the end)
    __nv_bfloat16* int_ws = transformer_ws
        + (size_t)config_.max_source_positions * d * 4
        + (size_t)config_.max_source_positions * config_.encoder_ffn_dim;
    int* cu_seqlens_gpu = reinterpret_cast<int*>(int_ws);
    int* chunk_meta_gpu = cu_seqlens_gpu + MAX_SEGMENTS + 1;

    // --- Profiling: record start of conv phase ---
    cudaEvent_t ev_conv_start;
    cudaEventCreate(&ev_conv_start);
    cudaEventRecord(ev_conv_start, stream);

    // --- Phase 1: Conv2D pipeline (process chunks in batches) ---
    int max_conv_time = w3;  // tokens per full chunk after conv = 13

    for (int batch_start = 0; batch_start < chunk_num; batch_start += CONV_BATCH_SIZE) {
        int batch_count = std::min(CONV_BATCH_SIZE, chunk_num - batch_start);

        // Copy chunk metadata to GPU
        cudaMemcpyAsync(chunk_meta_gpu,
                        chunk_starts.data() + batch_start,
                        batch_count * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(chunk_meta_gpu + CONV_BATCH_SIZE,
                        chunk_lens.data() + batch_start,
                        batch_count * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Copy mel chunks to padded buffer
        cudaMemsetAsync(mel_padded, 0,
                        (size_t)batch_count * mel_bins * chunk_mel_len * sizeof(__nv_bfloat16),
                        stream);
        dim3 copy_grid(mel_bins, batch_count);
        copy_mel_chunks_kernel<<<copy_grid, 256, 0, stream>>>(
            mel_padded, mel,
            chunk_meta_gpu,
            chunk_meta_gpu + CONV_BATCH_SIZE,
            mel_bins, chunk_mel_len, mel_frames);

        // Conv2D pipeline: mel → conv2d1 → conv2d2 → conv2d3
        // Layer 1: [batch, 1, 128, 100] → [batch, 480, 64, 50]  (naive: C_in=1)
        int cur_C = 1, cur_H = mel_bins, cur_W = chunk_mel_len;
        conv2d_forward(mel_padded, batch_count, cur_C, cur_H, cur_W,
                       conv2d1_w_, conv2d1_b_, dhs, conv_buf_a, stream);
        cur_C = dhs;
        cur_H = ASRConfig::conv_output_size(cur_H);
        cur_W = ASRConfig::conv_output_size(cur_W);

        // Layer 2: [batch, 480, 64, 50] → [batch, 480, 32, 25]  (im2col+GEMM)
        conv2d_forward(conv_buf_a, batch_count, cur_C, cur_H, cur_W,
                       conv2d2_w_, conv2d2_b_, dhs, conv_buf_b, stream);
        cur_H = ASRConfig::conv_output_size(cur_H);
        cur_W = ASRConfig::conv_output_size(cur_W);

        // Layer 3: [batch, 480, 32, 25] → [batch, 480, 16, 13]  (im2col+GEMM)
        conv2d_forward(conv_buf_b, batch_count, cur_C, cur_H, cur_W,
                       conv2d3_w_, conv2d3_b_, dhs, conv_buf_a, stream);
        cur_H = ASRConfig::conv_output_size(cur_H);
        cur_W = ASRConfig::conv_output_size(cur_W);

        // Transpose: [batch, 480, h3, w3] → [batch, w3, 480*h3=7680]
        {
            int total_elems = batch_count * dhs * cur_H * cur_W;
            transpose_ncft_to_ntcf_kernel<<<(total_elems + 255) / 256, 256, 0, stream>>>(
                conv_buf_b,  // dst: [batch, w3, 480*h3]
                conv_buf_a,  // src: [batch, 480, h3, w3]
                batch_count, dhs, cur_H, cur_W);
        }

        // conv_out: Linear [batch*w3, 7680] → [batch*w3, 1024], no bias
        {
            int M_total = batch_count * cur_W;  // batch * 13
            int K_dim = config_.conv_out_features();  // 7680
            bf16_linear(cublas_handle_, conv_buf_a,
                               conv_buf_b, conv_out_w_, nullptr,
                               M_total, K_dim, d, stream);
        }

        // Add sinusoidal PE: each chunk independently uses PE[0..cur_W-1]
        // Python: padded_embed = padded_embed + positional_embedding
        //   where positional_embedding is [1, T_padded, d_model] broadcast across chunks
        audio_ops::invoke_add_pe_chunked(conv_buf_a, pe_table_,
                                         batch_count * cur_W, d, cur_W, stream);

        // Extract valid tokens to hidden_states buffer
        for (int bi = 0; bi < batch_count; bi++) {
            int chunk_idx = batch_start + bi;
            int valid = chunk_output_lens[chunk_idx];
            int offset = chunk_output_offsets[chunk_idx];
            if (valid > 0) {
                cudaMemcpyAsync(
                    hidden_states + (size_t)offset * d,
                    conv_buf_a + (size_t)bi * cur_W * d,
                    (size_t)valid * d * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    // --- Phase 2: Construct cu_seqlens for attention windowing ---
    // window_aftercnn = max_conv_time * (n_window_infer / chunk_mel_len)
    int window_aftercnn = max_conv_time * (n_window_infer / chunk_mel_len);
    if (window_aftercnn <= 0) window_aftercnn = total_tokens;  // fallback

    std::vector<int> cu_seqlens_host;
    cu_seqlens_host.push_back(0);
    int remaining = total_tokens;
    int pos = 0;
    while (remaining > 0) {
        int seg = std::min(window_aftercnn, remaining);
        pos += seg;
        cu_seqlens_host.push_back(pos);
        remaining -= seg;
    }
    int num_segments = (int)cu_seqlens_host.size() - 1;

    cudaMemcpyAsync(cu_seqlens_gpu, cu_seqlens_host.data(),
                    cu_seqlens_host.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // --- Profiling: CUDA events for phase timing ---
    cudaEvent_t ev_conv_end, ev_layers_end, ev_post_end;
    cudaEventCreate(&ev_conv_end);
    cudaEventCreate(&ev_layers_end);
    cudaEventCreate(&ev_post_end);
    cudaEventRecord(ev_conv_end, stream);

    // --- Phase 3: 24 Transformer encoder layers ---
    for (int layer = 0; layer < config_.encoder_layers; layer++) {
        encoder_layer_forward(layer, hidden_states, total_tokens,
                              cu_seqlens_gpu, num_segments,
                              transformer_ws, stream);
    }
    cudaEventRecord(ev_layers_end, stream);

    // --- Phase 4: Post-processing ---
    // ln_post
    __nv_bfloat16* post_buf = transformer_ws;  // reuse norm_buf
    audio_ops::invoke_layernorm(post_buf, hidden_states,
                                 ln_post_w_, ln_post_b_,
                                 1e-5f, total_tokens, d, stream);

    // proj1 + GELU
    __nv_bfloat16* proj_buf = post_buf + (size_t)total_tokens * d;
    bf16_linear(cublas_handle_, proj_buf, post_buf, proj1_w_, proj1_b_,
                       total_tokens, d, d, stream);
    gelu_inplace_kernel<<<((size_t)total_tokens * d + 255) / 256, 256, 0, stream>>>(
        proj_buf, total_tokens * d);

    // proj2: [total_tokens, 1024] → [total_tokens, 2048]
    bf16_linear(cublas_handle_, encoder_out, proj_buf, proj2_w_, proj2_b_,
                       total_tokens, d, output_dim, stream);
    cudaEventRecord(ev_post_end, stream);

    cudaStreamSynchronize(stream);

    // Report phase timing
    float conv_ms, layers_ms, post_ms;
    cudaEventElapsedTime(&conv_ms, ev_conv_start, ev_conv_end);
    cudaEventElapsedTime(&layers_ms, ev_conv_end, ev_layers_end);
    cudaEventElapsedTime(&post_ms, ev_layers_end, ev_post_end);
    fprintf(stderr, "[Encoder Profile] tokens=%d segs=%d | Conv: %.1fms | Layers: %.1fms | Post: %.1fms | Total: %.1fms\n",
            total_tokens, num_segments, conv_ms, layers_ms, post_ms, conv_ms + layers_ms + post_ms);
    cudaEventDestroy(ev_conv_start);
    cudaEventDestroy(ev_conv_end);
    cudaEventDestroy(ev_layers_end);
    cudaEventDestroy(ev_post_end);
}

} // namespace asr
} // namespace qwen_thor
