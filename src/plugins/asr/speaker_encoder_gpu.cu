// speaker_encoder_gpu.cu — CAM++ GPU Speaker Encoder (buffer-reuse version)
//
// GPU 加速版本的 CAM++ 说话人编码器
// 所有运算在 GPU 上执行，使用 cuBLAS SGEMM + 自定义 CUDA kernels
// 关键优化: 预分配固定 scratch buffers 并在每层复用，避免内存爆炸

#include "speaker_encoder_gpu.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace qwen_thor {
namespace asr {

// ============================================================================
// CUDA Kernels
// ============================================================================

// BN + ReLU fused: y[c,s] = max(0, gamma[c] * (x[c,s] - mean[c]) / sqrt(var[c]+eps) + beta[c])
__global__ void bn_relu_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                const float* __restrict__ mean,
                                const float* __restrict__ var,
                                int C, int spatial, bool do_relu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * spatial;
    if (idx >= total) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;
    float val = g * (input[idx] - mean[c]) * inv_std + b;
    output[idx] = do_relu ? fmaxf(val, 0.0f) : val;
}

__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = fmaxf(data[idx], 0.0f);
}

// Conv2d: [Cin, H, W] → [Cout, H', W']
__global__ void conv2d_kernel(const float* __restrict__ input,
                               const float* __restrict__ weight,
                               float* __restrict__ output,
                               int Cin, int H, int W,
                               int Cout, int H_out, int W_out,
                               int k, int stride_h, int stride_w,
                               int pad_h, int pad_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * H_out * W_out;
    if (idx >= total) return;

    int co = idx / (H_out * W_out);
    int rem = idx % (H_out * W_out);
    int ho = rem / W_out;
    int wo = rem % W_out;

    float sum = 0;
    for (int ci = 0; ci < Cin; ++ci)
        for (int kh = 0; kh < k; ++kh)
            for (int kw = 0; kw < k; ++kw) {
                int hi = ho * stride_h - pad_h + kh;
                int wi = wo * stride_w - pad_w + kw;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W)
                    sum += weight[co * Cin * k * k + ci * k * k + kh * k + kw]
                         * input[ci * H * W + hi * W + wi];
            }
    output[idx] = sum;
}

// Conv1d with dilation
__global__ void conv1d_kernel(const float* __restrict__ input,
                               const float* __restrict__ weight,
                               const float* __restrict__ bias,
                               float* __restrict__ output,
                               int Cin, int T, int Cout, int T_out,
                               int k, int stride, int pad, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * T_out;
    if (idx >= total) return;

    int co = idx / T_out;
    int to = idx % T_out;

    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < Cin; ++ci)
        for (int ki = 0; ki < k; ++ki) {
            int ti = to * stride - pad + ki * dilation;
            if (ti >= 0 && ti < T)
                sum += weight[co * Cin * k + ci * k + ki] * input[ci * T + ti];
        }
    output[idx] = sum;
}

__global__ void add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

// Segment pooling: each timestep gets the average of its segment
__global__ void seg_pool_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int C, int T, int seg_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx >= total) return;
    int c = idx / T;
    int t = idx % T;
    int seg_start = (t / seg_len) * seg_len;
    int seg_end = min(seg_start + seg_len, T);
    float sum = 0;
    for (int i = seg_start; i < seg_end; ++i) sum += input[c * T + i];
    output[idx] = sum / (seg_end - seg_start);
}

// Context = global_mean + seg_pool
__global__ void context_kernel(const float* __restrict__ input,
                                const float* __restrict__ seg_pool,
                                float* __restrict__ context,
                                int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * T;
    if (idx >= total) return;
    int c = idx / T;
    float global_mean = 0;
    for (int t = 0; t < T; ++t) global_mean += input[c * T + t];
    global_mean /= T;
    context[idx] = global_mean + seg_pool[idx];
}

// Sigmoid multiply: output[i] *= sigmoid(gate[i])
__global__ void sigmoid_mul_kernel(float* __restrict__ output,
                                    const float* __restrict__ gate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] *= 1.0f / (1.0f + expf(-gate[idx]));
}

// StatsPool: [C, T] → [2*C] (mean + std per channel)
__global__ void stats_pool_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int C, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0;
    for (int t = 0; t < T; ++t) sum += input[c * T + t];
    float mean = sum / T;
    float var_sum = 0;
    for (int t = 0; t < T; ++t) {
        float diff = input[c * T + t] - mean;
        var_sum += diff * diff;
    }
    output[c] = mean;
    output[C + c] = sqrtf(var_sum / max(1, T - 1) + 1e-2f);
}

// BN without affine
__global__ void bn_no_affine_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ mean,
                                     const float* __restrict__ var, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    output[c] = (input[c] - mean[c]) * rsqrtf(var[c] + 1e-5f);
}

__global__ void l2_normalize_kernel(float* data, int C) {
    float norm = 0;
    for (int i = 0; i < C; ++i) norm += data[i] * data[i];
    norm = rsqrtf(norm + 1e-12f);
    for (int i = 0; i < C; ++i) data[i] *= norm;
}

// Bias + ReLU: data[c*T+t] = max(0, data[c*T+t] + bias[c])
__global__ void add_bias_relu_kernel(float* data, const float* __restrict__ bias,
                                      int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    data[idx] = fmaxf(data[idx] + (bias ? bias[c] : 0.0f), 0.0f);
}

// Bias only
__global__ void add_bias_kernel(float* data, const float* __restrict__ bias,
                                 int C, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    int c = idx / T;
    data[idx] += bias ? bias[c] : 0.0f;
}

// Copy rows: dst[offset*T ... (offset+C)*T-1] = src[0 ... C*T-1]
__global__ void copy_rows_kernel(float* dst, const float* src,
                                  int C, int T, int dst_offset_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * T) return;
    dst[dst_offset_channels * T + idx] = src[idx];
}

static constexpr int BLOCK = 256;
static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// ScratchPool method implementations
// ============================================================================
bool ScratchPool::alloc(int max_T, int max_spatial) {
    size_t a_sz = max_spatial;
    size_t b_sz = max_spatial;
    // c/d are used as FCM ResBlock scratch (up to 32*40*T = 1280*T elements)
    // AND DenseTDNN scratch (128*T), so take the max
    size_t fcm_scratch = (size_t)32 * 40 * max_T;
    size_t c_sz = std::max((size_t)128 * max_T, fcm_scratch);
    size_t d_sz = std::max((size_t)128 * max_T, fcm_scratch);
    size_t e_sz = 128 * max_T;
    size_t f_sz = 64  * max_T;
    size_t cat_sz = 1024 * max_T;

    total_bytes = (a_sz + b_sz + c_sz + d_sz + e_sz + f_sz + 2 * cat_sz) * sizeof(float);

    float* base = nullptr;
    if (cudaMalloc(&base, total_bytes) != cudaSuccess) return false;
    cudaMemset(base, 0, total_bytes);  // Zero-initialize to prevent stale data artifacts

    size_t off = 0;
    a = base + off; off += a_sz;
    b = base + off; off += b_sz;
    c = base + off; off += c_sz;
    d = base + off; off += d_sz;
    e = base + off; off += e_sz;
    f = base + off; off += f_sz;
    concat[0] = base + off; off += cat_sz;
    concat[1] = base + off; off += cat_sz;
    which_concat = 0;
    return true;
}

void ScratchPool::free() {
    if (a) { cudaFree(a); a = nullptr; }
}

// ============================================================================
// GpuSpeakerEncoder Implementation
// ============================================================================

GpuSpeakerEncoder::GpuSpeakerEncoder() = default;

GpuSpeakerEncoder::~GpuSpeakerEncoder() {
    scratch_.free();
    if (stream_) cudaStreamDestroy(stream_);
    if (cublas_) cublasDestroy(cublas_);
    for (auto& kv : gpu_tensors_) {
        if (kv.second) cudaFree(kv.second);
    }
    if (workspace_) cudaFree(workspace_);
}

bool GpuSpeakerEncoder::load(const std::string& safetensors_path) {
    auto cpu_tensors = load_safetensors(safetensors_path);
    if (cpu_tensors.empty()) return false;

    if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[SpeakerGPU] Failed to create cuBLAS handle\n");
        return false;
    }

    size_t total_bytes = 0;
    for (auto& kv : cpu_tensors) {
        float* d_ptr = nullptr;
        size_t bytes = kv.second.size() * sizeof(float);
        if (cudaMalloc(&d_ptr, bytes) != cudaSuccess) {
            fprintf(stderr, "[SpeakerGPU] cudaMalloc failed for %s (%zu bytes)\n",
                    kv.first.c_str(), bytes);
            return false;
        }
        cudaMemcpy(d_ptr, kv.second.data(), bytes, cudaMemcpyHostToDevice);
        gpu_tensors_[kv.first] = d_ptr;
        tensor_sizes_[kv.first] = (int)kv.second.size();
        total_bytes += bytes;
    }

    loaded_ = true;

    // Create persistent stream + pre-allocate scratch for typical VAD segments (≤10s → ~1000 frames)
    cudaStreamCreate(&stream_);
    ensure_scratch(1000);

    fprintf(stderr, "[SpeakerGPU] CAM++ loaded to GPU: %zu tensors, %.1f MB, scratch %.1f MB\n",
            gpu_tensors_.size(), total_bytes / (1024.0f * 1024.0f),
            scratch_.total_bytes / (1024.0f * 1024.0f));
    return true;
}

// Ensure scratch pool is large enough for T frames (auto-grow, never shrink)
bool GpuSpeakerEncoder::ensure_scratch(int T) {
    if (T <= scratch_max_T_) return true;  // already large enough
    scratch_.free();
    int T2 = (T + 2*2 - 1*(5-1) - 1) / 2 + 1;
    int max_fcm_spatial = 32 * 80 * T;
    int max_block_spatial = 1024 * T2;
    int max_spatial = std::max(max_fcm_spatial, max_block_spatial);
    if (!scratch_.alloc(std::max(T, T2), max_spatial)) {
        fprintf(stderr, "[SpeakerGPU] scratch realloc failed for T=%d\n", T);
        scratch_max_T_ = 0;
        return false;
    }
    scratch_max_T_ = T;
    return true;
}

const float* GpuSpeakerEncoder::get_gpu(const std::string& name) const {
    auto it = gpu_tensors_.find(name);
    if (it == gpu_tensors_.end()) return nullptr;
    return it->second;
}

// ============================================================================
// GPU ResBlock (FCM) — member function, accesses get_gpu() directly
// ============================================================================
void GpuSpeakerEncoder::gpu_res_block(const float* d_input, float* d_output,
                           int C, int H, int W,
                           const std::string& prefix, int stride,
                           float* scratch_a, float* scratch_b,
                           cudaStream_t stream) {
    int pad = 1;
    int H2 = (H + 2*pad - 3) / stride + 1;
    int conv_size = C * H2 * W;

    // conv1 → BN → ReLU → scratch_a
    conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        d_input, get_gpu(prefix + ".conv1.weight"), scratch_a,
        C, H, W, C, H2, W, 3, stride, 1, 1, 1);
    bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_a, scratch_a,
        get_gpu(prefix + ".bn1.weight"), get_gpu(prefix + ".bn1.bias"),
        get_gpu(prefix + ".bn1.running_mean"), get_gpu(prefix + ".bn1.running_var"),
        C, H2 * W, true);

    // conv2 → BN → scratch_b
    conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_a, get_gpu(prefix + ".conv2.weight"), scratch_b,
        C, H2, W, C, H2, W, 3, 1, 1, 1, 1);
    bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
        scratch_b, scratch_b,
        get_gpu(prefix + ".bn2.weight"), get_gpu(prefix + ".bn2.bias"),
        get_gpu(prefix + ".bn2.running_mean"), get_gpu(prefix + ".bn2.running_var"),
        C, H2 * W, false);

    // Shortcut
    if (stride != 1) {
        // d_output = shortcut(d_input)
        conv2d_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
            d_input, get_gpu(prefix + ".shortcut.0.weight"), d_output,
            C, H, W, C, H2, W, 1, stride, 1, 0, 0);
        bn_relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(
            d_output, d_output,
            get_gpu(prefix + ".shortcut.1.weight"), get_gpu(prefix + ".shortcut.1.bias"),
            get_gpu(prefix + ".shortcut.1.running_mean"), get_gpu(prefix + ".shortcut.1.running_var"),
            C, H2 * W, false);
        // output = scratch_b + shortcut
        add_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, scratch_b, conv_size);
    } else {
        // output = d_input + scratch_b
        cudaMemcpyAsync(d_output, d_input, conv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        add_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, scratch_b, conv_size);
    }
    relu_kernel<<<div_ceil(conv_size, BLOCK), BLOCK, 0, stream>>>(d_output, conv_size);
}

// ============================================================================
// GPU extract() — main entry point
// ============================================================================
std::vector<float> GpuSpeakerEncoder::extract(const float* mel_80xT, int T) {
    if (!loaded_ || T < 10) return {};

    // Ensure persistent scratch is large enough (auto-grows if needed)
    if (!ensure_scratch(T)) return {};

    // Use persistent stream_
    cublasSetStream(cublas_, stream_);

    // Compute dimensions
    int T2 = (T + 2*2 - 1*(5-1) - 1) / 2 + 1;  // after TDNN stride=2

    // Reset concat buffer index for this call
    scratch_.which_concat = 0;

    // Reference persistent scratch
    ScratchPool& sp = scratch_;


    // 1. CMN (Cepstral Mean Normalization): subtract per-bin mean across time
    //    This matches FunASR CAMPPlus extract_feature: feature - feature.mean(dim=0)
    std::vector<float> mel_cmn(80 * T);
    float bin_mean[80] = {};
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < 80; ++f)
            bin_mean[f] += mel_80xT[t * 80 + f];
    float inv_T = 1.0f / T;
    for (int f = 0; f < 80; ++f)
        bin_mean[f] *= inv_T;
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < 80; ++f)
            mel_cmn[t * 80 + f] = mel_80xT[t * 80 + f] - bin_mean[f];

    // 2. Transpose mel [T, 80] → [80, T] and upload
    std::vector<float> transposed(80 * T);
    for (int t = 0; t < T; ++t)
        for (int f = 0; f < 80; ++f)
            transposed[f * T + t] = mel_cmn[t * 80 + f];

    // Use sp.a as input buffer
    float* d_x = sp.a;
    cudaMemcpyAsync(d_x, transposed.data(), 80 * T * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // ======================== FCM ========================
    // conv1: [1, 80, T] → [32, 80, T]
    int H = 80;
    int conv1_size = 32 * H * T;
    float* d_fcm = sp.b;  // FCM output goes to b
    conv2d_kernel<<<div_ceil(conv1_size, BLOCK), BLOCK, 0, stream_>>>(
        d_x, get_gpu("head.conv1.weight"), d_fcm,
        1, H, T, 32, H, T, 3, 1, 1, 1, 1);
    bn_relu_kernel<<<div_ceil(conv1_size, BLOCK), BLOCK, 0, stream_>>>(
        d_fcm, d_fcm,
        get_gpu("head.bn1.weight"), get_gpu("head.bn1.bias"),
        get_gpu("head.bn1.running_mean"), get_gpu("head.bn1.running_var"),
        32, H * T, true);

    // layer1[0]: stride=2 → H: 80→40
    // Input: d_fcm (sp.b), output: d_x (sp.a), scratch: sp.c, sp.d
    gpu_res_block(d_fcm, d_x, 32, H, T, "head.layer1.0", 2, sp.c, sp.d, stream_);
    H = (H + 2 - 3) / 2 + 1;  // = 40

    // layer1[1]: stride=1
    gpu_res_block(d_x, d_fcm, 32, H, T, "head.layer1.1", 1, sp.c, sp.d, stream_);

    // layer2[0]: stride=2 → H: 40→20
    gpu_res_block(d_fcm, d_x, 32, H, T, "head.layer2.0", 2, sp.c, sp.d, stream_);
    H = (H + 2 - 3) / 2 + 1;  // = 20

    // layer2[1]: stride=1
    gpu_res_block(d_x, d_fcm, 32, H, T, "head.layer2.1", 1, sp.c, sp.d, stream_);

    // conv2: stride_h=2 → H: 20→10
    int H2 = (H + 2 - 3) / 2 + 1;  // = 10
    int conv2_size = 32 * H2 * T;
    conv2d_kernel<<<div_ceil(conv2_size, BLOCK), BLOCK, 0, stream_>>>(
        d_fcm, get_gpu("head.conv2.weight"), d_x,
        32, H, T, 32, H2, T, 3, 2, 1, 1, 1);
    bn_relu_kernel<<<div_ceil(conv2_size, BLOCK), BLOCK, 0, stream_>>>(
        d_x, d_x,
        get_gpu("head.bn2.weight"), get_gpu("head.bn2.bias"),
        get_gpu("head.bn2.running_mean"), get_gpu("head.bn2.running_var"),
        32, H2 * T, true);
    H = H2;
    int feat_dim = 32 * H;  // = 320
    // d_x now contains [320, T] (same memory layout)

    // ======================== TDNN ========================
    // Conv1d(320→128, k=5, s=2, p=2)
    int tdnn_size = 128 * T2;
    float* d_tdnn = sp.b;
    conv1d_kernel<<<div_ceil(tdnn_size, BLOCK), BLOCK, 0, stream_>>>(
        d_x, get_gpu("xvector.tdnn.linear.weight"), get_gpu("xvector.tdnn.linear.bias"),
        d_tdnn, feat_dim, T, 128, T2, 5, 2, 2, 1);
    bn_relu_kernel<<<div_ceil(tdnn_size, BLOCK), BLOCK, 0, stream_>>>(
        d_tdnn, d_tdnn,
        get_gpu("xvector.tdnn.nonlinear.batchnorm.weight"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.bias"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.tdnn.nonlinear.batchnorm.running_var"),
        128, T2, true);

    // Copy TDNN output into concat buffer 0
    cudaMemcpyAsync(sp.cur_concat(), d_tdnn, 128 * T2 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // ======================== DenseTDNN Blocks ========================
    int cur_dim = 128;

    // Block 1: 12 layers, dilation=1
    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block1", 12, 1, stream_);
    cur_dim += 12 * 32;  // = 512

    // Transit 1: BN → ReLU → Conv1d(512→256)
    gpu_transit(sp, cur_dim, T2, "xvector.transit1", cur_dim / 2, stream_);
    cur_dim /= 2;  // = 256

    // Block 2: 24 layers, dilation=2
    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block2", 24, 2, stream_);
    cur_dim += 24 * 32;  // = 1024

    // Transit 2
    gpu_transit(sp, cur_dim, T2, "xvector.transit2", cur_dim / 2, stream_);
    cur_dim /= 2;  // = 512

    // Block 3: 16 layers, dilation=2
    gpu_cam_dense_block(sp, cur_dim, T2, "xvector.block3", 16, 2, stream_);
    cur_dim += 16 * 32;  // = 1024

    // Transit 3
    gpu_transit(sp, cur_dim, T2, "xvector.transit3", cur_dim / 2, stream_);
    cur_dim /= 2;  // = 512

    // ======================== Out nonlinear: BN + ReLU ========================
    int embed_channels = cur_dim;  // 512
    float* d_final = sp.cur_concat();
    int out_size = embed_channels * T2;
    bn_relu_kernel<<<div_ceil(out_size, BLOCK), BLOCK, 0, stream_>>>(
        d_final, d_final,
        get_gpu("xvector.out_nonlinear.batchnorm.weight"),
        get_gpu("xvector.out_nonlinear.batchnorm.bias"),
        get_gpu("xvector.out_nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.out_nonlinear.batchnorm.running_var"),
        embed_channels, T2, true);

    // ======================== StatsPool ========================
    float* d_pooled = sp.a;  // reuse scratch.a
    stats_pool_kernel<<<div_ceil(embed_channels, BLOCK), BLOCK, 0, stream_>>>(
        d_final, d_pooled, embed_channels, T2);

    // ======================== Dense: GEMV(1024→192) ========================
    const int emb_size = 192;
    float* d_emb = sp.b;  // reuse scratch.b
    {
        float alpha = 1.0f, beta = 0.0f;
        // Weight: [192, 1024] row-major = [1024, 192] col-major
        // x: [1024, 1], y: [192, 1]
        cublasSgemv(cublas_, CUBLAS_OP_T, embed_channels * 2, emb_size,
                    &alpha, get_gpu("xvector.dense.linear.weight"),
                    embed_channels * 2, d_pooled, 1, &beta, d_emb, 1);
    }

    // BN (no affine)
    bn_no_affine_kernel<<<1, BLOCK, 0, stream_>>>(
        d_emb, d_emb,
        get_gpu("xvector.dense.nonlinear.batchnorm.running_mean"),
        get_gpu("xvector.dense.nonlinear.batchnorm.running_var"),
        emb_size);

    // L2 normalize
    l2_normalize_kernel<<<1, 1, 0, stream_>>>(d_emb, emb_size);

    // Copy result back
    cudaStreamSynchronize(stream_);
    std::vector<float> result(emb_size);
    cudaMemcpy(result.data(), d_emb, emb_size * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

// ============================================================================
// GPU CAM Dense TDNN Block — uses scratch pool with buffer reuse
// ============================================================================
void GpuSpeakerEncoder::gpu_cam_dense_block(ScratchPool& sp, int in_dim, int T,
                                              const std::string& prefix,
                                              int num_layers, int dilation,
                                              cudaStream_t stream) {
    const int growth = 32;
    const int bn_ch = 128;
    int k = 3;
    int pad = (k - 1) / 2 * dilation;


    int cur_dim = in_dim;
    // sp.cur_concat() has the current concatenation [cur_dim, T]

    for (int l = 0; l < num_layers; ++l) {
        std::string lp = prefix + ".tdnnd" + std::to_string(l + 1);

        // nonlinear1: BN(cur_dim) + ReLU → sp.a (scratch)
        bn_relu_kernel<<<div_ceil(cur_dim * T, BLOCK), BLOCK, 0, stream>>>(
            sp.cur_concat(), sp.a,
            get_gpu(lp + ".nonlinear1.batchnorm.weight"),
            get_gpu(lp + ".nonlinear1.batchnorm.bias"),
            get_gpu(lp + ".nonlinear1.batchnorm.running_mean"),
            get_gpu(lp + ".nonlinear1.batchnorm.running_var"),
            cur_dim, T, true);

        // linear1: Conv1d(cur_dim→128, k=1) = GEMM → sp.b
        // Row-major: W[128, cur_dim] × X[cur_dim, T] → H[128, T]
        // cuBLAS col-major: X^T[T, cur_dim] × W^T[cur_dim, 128] → H^T[T, 128]
        {
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                        T, bn_ch, cur_dim,
                        &alpha, sp.a, T,
                        get_gpu(lp + ".linear1.weight"), cur_dim,
                        &beta, sp.b, T);
        }

        // nonlinear2: BN(128) + ReLU (in-place on sp.b)
        bn_relu_kernel<<<div_ceil(bn_ch * T, BLOCK), BLOCK, 0, stream>>>(
            sp.b, sp.b,
            get_gpu(lp + ".nonlinear2.batchnorm.weight"),
            get_gpu(lp + ".nonlinear2.batchnorm.bias"),
            get_gpu(lp + ".nonlinear2.batchnorm.running_mean"),
            get_gpu(lp + ".nonlinear2.batchnorm.running_var"),
            bn_ch, T, true);

        // CAM layer → sp.c (output is [growth, T] = [32, T])
        gpu_cam_layer(sp, bn_ch, growth, T, lp + ".cam_layer", k, dilation, pad, stream);
        // After this, sp.c has [32, T] output

        // Append sp.c [32, T] to concat buffer
        // Copy current concat [cur_dim, T] to next_concat
        cudaMemcpyAsync(sp.next_concat(), sp.cur_concat(),
                        cur_dim * T * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // Append growth channels
        copy_rows_kernel<<<div_ceil(growth * T, BLOCK), BLOCK, 0, stream>>>(
            sp.next_concat(), sp.c, growth, T, cur_dim);
        cudaStreamSynchronize(stream);  // Ensure concat copy+append complete before next layer reads
        sp.swap_concat();
        cur_dim += growth;
    }
}

// ============================================================================
// GPU CAM Layer — input: sp.b [bn_ch, T], output: sp.c [out_ch, T]
// ============================================================================
void GpuSpeakerEncoder::gpu_cam_layer(ScratchPool& sp, int bn_ch, int out_ch,
                                        int T, const std::string& prefix,
                                        int k, int dilation, int padding,
                                        cudaStream_t stream) {

    // linear_local: Conv1d(128→32, k=3, dilation) → sp.c
    int local_size = out_ch * T;
    conv1d_kernel<<<div_ceil(local_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, get_gpu(prefix + ".linear_local.weight"),
        get_gpu(prefix + ".linear_local.bias"),
        sp.c, bn_ch, T, out_ch, T, k, 1, padding, dilation);

    // Segment pooling → sp.d
    int ctx_size = bn_ch * T;
    seg_pool_kernel<<<div_ceil(ctx_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, sp.d, bn_ch, T, 100);

    // Context = global_mean + seg_pool → sp.e
    context_kernel<<<div_ceil(ctx_size, BLOCK), BLOCK, 0, stream>>>(
        sp.b, sp.d, sp.e, bn_ch, T);

    // linear1: Conv1d(128→64, k=1) = GEMM → sp.d (reuse)
    int mid = bn_ch / 2;  // 64
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    T, mid, bn_ch,
                    &alpha, sp.e, T,
                    get_gpu(prefix + ".linear1.weight"), bn_ch,
                    &beta, sp.d, T);
    }
    // bias + ReLU
    add_bias_relu_kernel<<<div_ceil(mid * T, BLOCK), BLOCK, 0, stream>>>(
        sp.d, get_gpu(prefix + ".linear1.bias"), mid, T);

    // linear2: Conv1d(64→32, k=1) = GEMM → sp.e (reuse)
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                    T, out_ch, mid,
                    &alpha, sp.d, T,
                    get_gpu(prefix + ".linear2.weight"), mid,
                    &beta, sp.e, T);
    }
    add_bias_kernel<<<div_ceil(out_ch * T, BLOCK), BLOCK, 0, stream>>>(
        sp.e, get_gpu(prefix + ".linear2.bias"), out_ch, T);

    // sigmoid(gate) * local_out → sp.c stays as output
    sigmoid_mul_kernel<<<div_ceil(local_size, BLOCK), BLOCK, 0, stream>>>(
        sp.c, sp.e, local_size);
}

// ============================================================================
// GPU Transit Layer
// ============================================================================
void GpuSpeakerEncoder::gpu_transit(ScratchPool& sp, int in_dim, int T,
                                      const std::string& prefix, int out_dim,
                                      cudaStream_t stream) {
    // BN + ReLU on concat → sp.a
    bn_relu_kernel<<<div_ceil(in_dim * T, BLOCK), BLOCK, 0, stream>>>(
        sp.cur_concat(), sp.a,
        get_gpu(prefix + ".nonlinear.batchnorm.weight"),
        get_gpu(prefix + ".nonlinear.batchnorm.bias"),
        get_gpu(prefix + ".nonlinear.batchnorm.running_mean"),
        get_gpu(prefix + ".nonlinear.batchnorm.running_var"),
        in_dim, T, true);

    // Conv1d(in→out, k=1) = GEMM → next_concat
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                T, out_dim, in_dim,
                &alpha, sp.a, T,
                get_gpu(prefix + ".linear.weight"), in_dim,
                &beta, sp.next_concat(), T);
    sp.swap_concat();
}

// ============================================================================
// Safetensors loader
// ============================================================================
GpuSpeakerEncoder::TensorMap GpuSpeakerEncoder::load_safetensors(const std::string& path) {
    TensorMap result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return result;

    uint64_t header_size = 0;
    ifs.read(reinterpret_cast<char*>(&header_size), 8);
    if (header_size > 1000000) return result;

    std::string header(header_size, '\0');
    ifs.read(&header[0], header_size);

    size_t data_offset_base = 8 + header_size;
    size_t pos = 0;

    while (pos < header.size()) {
        size_t key_start = header.find('"', pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        std::string key = header.substr(key_start + 1, key_end - key_start - 1);
        pos = key_end + 1;

        if (key == "__metadata__") {
            size_t brace = header.find('{', pos);
            if (brace != std::string::npos) {
                int depth = 1; pos = brace + 1;
                while (pos < header.size() && depth > 0) {
                    if (header[pos] == '{') depth++;
                    else if (header[pos] == '}') depth--;
                    pos++;
                }
            }
            continue;
        }

        size_t offsets_pos = header.find("data_offsets", pos);
        if (offsets_pos == std::string::npos) break;
        size_t bracket = header.find('[', offsets_pos);
        size_t comma = header.find(',', bracket);
        size_t end_bracket = header.find(']', comma);

        uint64_t start = std::stoull(header.substr(bracket + 1, comma - bracket - 1));
        uint64_t end = std::stoull(header.substr(comma + 1, end_bracket - comma - 1));

        size_t num_bytes = end - start;
        size_t num_floats = num_bytes / sizeof(float);

        std::vector<float> data(num_floats);
        ifs.seekg(data_offset_base + start);
        ifs.read(reinterpret_cast<char*>(data.data()), num_bytes);

        result[key] = std::move(data);
        pos = end_bracket + 1;
    }
    return result;
}

} // namespace asr
} // namespace qwen_thor
