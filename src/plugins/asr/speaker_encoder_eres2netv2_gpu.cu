// speaker_encoder_eres2netv2_gpu.cu — ERes2NetV2 GPU Speaker Encoder
//
// GPU 加速版本: cuBLAS SGEMM (im2col for conv2d) + CUDA kernels
// 1331 chunks: CPU 17min → GPU ~3s (estimated)

#include "speaker_encoder_eres2netv2_gpu.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace qwen_thor {
namespace asr {

static constexpr int BLOCK = 256;
static inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// CUDA Kernels
// ============================================================================

// CMN + Transpose: input [T, 80] → output [80, T] with per-freq mean subtraction
__global__ void eres2_cmn_transpose_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int T, int F) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * F;
    if (idx >= total) return;
    int t = idx / F;
    int f = idx % F;

    // Compute mean for this frequency bin
    float sum = 0;
    for (int tt = 0; tt < T; ++tt) sum += input[tt * F + f];
    float mean = sum / T;

    // Transpose + CMN: output[f, t] = input[t, f] - mean[f]
    output[f * T + t] = input[t * F + f] - mean;
}

// BN + HardTanh(0, 20): y = clamp(gamma * (x - mean) / sqrt(var+eps) + beta, 0, 20)
__global__ void bn_hardtanh_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ var,
                                    int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    float val = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
    output[idx] = fminf(fmaxf(val, 0.0f), 20.0f);
}

// BN + ReLU
__global__ void eres2_bn_relu_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ var,
                                      int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    float val = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
    output[idx] = fmaxf(val, 0.0f);
}

// BN only (no activation)
__global__ void eres2_bn_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 const float* __restrict__ mean,
                                 const float* __restrict__ var,
                                 int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    output[idx] = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
}

// BN + SiLU: val = BN(x), out = val * sigmoid(val)
__global__ void bn_silu_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                const float* __restrict__ mean,
                                const float* __restrict__ var,
                                int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    float inv_std = rsqrtf(var[c] + 1e-5f);
    float val = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
    output[idx] = val / (1.0f + expf(-val));
}

// Residual add + HardTanh(0, 20): out = clamp(a + b, 0, 20)
__global__ void add_hardtanh_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ output,
                                     int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = fminf(fmaxf(a[idx] + b[idx], 0.0f), 20.0f);
}

// Residual add (in-place to a): a[i] += b[i]
__global__ void eres2_add_kernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

// AFF attention: out = x * att + y * (2 - att) where att = 1 + tanh(h)
__global__ void aff_combine_kernel(const float* __restrict__ x,
                                    const float* __restrict__ y,
                                    const float* __restrict__ h,
                                    float* __restrict__ output,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float att = 1.0f + tanhf(h[idx]);
    output[idx] = x[idx] * att + y[idx] * (2.0f - att);
}

// Conv2d bias add: data[c * spatial + i] += bias[c]
__global__ void conv2d_bias_kernel(float* data, const float* __restrict__ bias,
                                    int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    int c = idx / spatial;
    data[idx] += bias[c];
}

// im2col GPU kernel: [Cin, H, W] → [Cin*k*k, H_out*W_out]
__global__ void im2col_kernel(const float* __restrict__ input,
                               float* __restrict__ col,
                               int Cin, int H, int W,
                               int k, int stride, int pad,
                               int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int K = Cin * k * k;
    int N = H_out * W_out;
    int total = K * N;
    if (idx >= total) return;

    int col_row = idx / N;
    int col_col = idx % N;

    int c = col_row / (k * k);
    int rem = col_row % (k * k);
    int kh = rem / k;
    int kw = rem % k;

    int ho = col_col / W_out;
    int wo = col_col % W_out;

    int hi = ho * stride - pad + kh;
    int wi = wo * stride - pad + kw;

    col[idx] = (hi >= 0 && hi < H && wi >= 0 && wi < W)
                   ? input[c * H * W + hi * W + wi] : 0.0f;
}

// TSTP pool: [C, H, W] → [C*H*2] (mean + std with Bessel's correction)
// Each thread handles one (c, h) pair
__global__ void tstp_pool_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int CH = C * H;
    if (idx >= CH) return;

    int c = idx / H;
    int h = idx % H;
    const float* row = input + c * H * W + h * W;

    float sum = 0;
    for (int w = 0; w < W; ++w) sum += row[w];
    float mean_val = sum / W;

    float var_sum = 0;
    for (int w = 0; w < W; ++w) {
        float d = row[w] - mean_val;
        var_sum += d * d;
    }
    float var_val = (W > 1) ? var_sum / (W - 1) : 0.0f;
    float std_val = sqrtf(var_val + 1e-8f);

    output[idx] = mean_val;
    output[CH + idx] = std_val;
}

// L2 normalize (single thread)
__global__ void eres2_l2_normalize_kernel(float* data, int C) {
    float norm = 0;
    for (int i = 0; i < C; ++i) norm += data[i] * data[i];
    norm = rsqrtf(norm + 1e-12f);
    for (int i = 0; i < C; ++i) data[i] *= norm;
}

// Concatenate two tensors along channel dim: [C1, S] + [C2, S] → [C1+C2, S]
__global__ void concat_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ output,
                               int C1, int C2, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (C1 + C2) * spatial;
    if (idx >= total) return;
    int c = idx / spatial;
    int s = idx % spatial;
    output[idx] = (c < C1) ? a[c * spatial + s] : b[(c - C1) * spatial + s];
}

// Copy channel slice: dst[0:C*S] = src[offset*S : (offset+C)*S]
__global__ void slice_channels_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int C, int spatial, int src_offset_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    dst[idx] = src[src_offset_channels * spatial + idx];
}

// ============================================================================
// ERes2Scratch — pre-allocated workspace
// ============================================================================
bool ERes2Scratch::alloc(int max_T) {
    free();

    // Compute max buffer sizes for ERes2NetV2 with max_T frames
    // Layer1 is the biggest: [128, 80, T] = 128*80*T
    int H0 = 80;
    size_t max_layer = (size_t)128 * H0 * max_T;          // layer1 output
    size_t max_spatial = std::max(max_layer, (size_t)1024 * 10 * ((max_T + 7) / 8));  // layer4

    // im2col: biggest at layer1 convs: [26*9, 80*T]
    size_t max_im2col = (size_t)26 * 9 * H0 * max_T;
    // Also consider conv1: [1*9, 80*T] = small
    // layer2 convs: [52*9, 40*T/2] — smaller
    max_im2col = std::max(max_im2col, (size_t)208 * 9 * 10 * ((max_T + 7) / 8));

    // Each main buffer: max of all intermediates
    // Note: layer4 output [1024, H4, W4] is the largest active tensor per layer
    // We need buffers for current layer IO + shortcut, not all layers simultaneously
    size_t buf_size = std::max(max_layer, max_spatial);
    // AFF fusion needs [2048, H4, W4] for concat, but H4 = ceil(80/8) = 10, W4 = ceil(T/8)
    buf_size = std::max(buf_size, (size_t)2048 * 10 * ((max_T + 7) / 8));

    total_bytes = 0;
    for (int i = 0; i < 8; i++) {
        if (cudaMalloc(&buf[i], buf_size * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "[ERes2V2GPU] buf[%d] alloc failed (%.1f MB)\n",
                    i, buf_size * 4.0f / (1024 * 1024));
            free();
            return false;
        }
        total_bytes += buf_size * sizeof(float);
    }

    if (cudaMalloc(&im2col, max_im2col * sizeof(float)) != cudaSuccess) {
        free(); return false;
    }
    total_bytes += max_im2col * sizeof(float);

    // scale_cat: [width*SCALE, H, W] max = [52, 80, T]
    size_t cat_size = (size_t)52 * H0 * max_T;
    if (cudaMalloc(&scale_cat, cat_size * sizeof(float)) != cudaSuccess) {
        free(); return false;
    }
    total_bytes += cat_size * sizeof(float);

    // aff_tmp: for AFF intermediate
    if (cudaMalloc(&aff_tmp, buf_size * sizeof(float)) != cudaSuccess) {
        free(); return false;
    }
    total_bytes += buf_size * sizeof(float);

    // out3: dedicated buffer for layer3 output backup [512, H3, W3]
    // H3 = ceil(80/4) = 20, W3 = ceil(T/4). Max size = 512 * 20 * ceil(max_T/4)
    size_t out3_size = (size_t)512 * 20 * ((max_T + 3) / 4);
    out3_size = std::max(out3_size, (size_t)1);  // at least 1
    if (cudaMalloc(&out3, out3_size * sizeof(float)) != cudaSuccess) {
        free(); return false;
    }
    total_bytes += out3_size * sizeof(float);

    return true;
}

void ERes2Scratch::free() {
    for (int i = 0; i < 8; i++) {
        if (buf[i]) { cudaFree(buf[i]); buf[i] = nullptr; }
    }
    if (im2col) { cudaFree(im2col); im2col = nullptr; }
    if (scale_cat) { cudaFree(scale_cat); scale_cat = nullptr; }
    if (aff_tmp) { cudaFree(aff_tmp); aff_tmp = nullptr; }
    if (out3) { cudaFree(out3); out3 = nullptr; }
    total_bytes = 0;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================
GpuERes2NetV2Encoder::GpuERes2NetV2Encoder() {
    cublasCreate(&cublas_);
    cudaStreamCreate(&stream_);
}

GpuERes2NetV2Encoder::~GpuERes2NetV2Encoder() {
    for (auto& [name, ptr] : gpu_tensors_) {
        if (ptr) cudaFree(ptr);
    }
    scratch_.free();
    for (int i = 0; i < BATCH_CONCURRENCY; i++) {
        batch_.scratch[i].free();
        if (batch_.streams[i]) cudaStreamDestroy(batch_.streams[i]);
        if (batch_.cublas[i]) cublasDestroy(batch_.cublas[i]);
    }
    if (batch_.d_emb_buf) cudaFree(batch_.d_emb_buf);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
}

// ============================================================================
// Weight loading
// ============================================================================
bool GpuERes2NetV2Encoder::load(const std::string& safetensors_path) {
    auto cpu_tensors = load_safetensors(safetensors_path);
    if (cpu_tensors.empty()) return false;

    // Verify ERes2NetV2 model
    if (cpu_tensors.find("layer3_ds.weight") == cpu_tensors.end() ||
        cpu_tensors.find("seg_1.weight") == cpu_tensors.end()) {
        fprintf(stderr, "[ERes2V2GPU] ERROR: not an ERes2NetV2 model\n");
        return false;
    }

    // Upload all tensors to GPU
    size_t total_bytes = 0;
    for (auto& [name, data] : cpu_tensors) {
        float* d_ptr = nullptr;
        size_t bytes = data.size() * sizeof(float);
        if (cudaMalloc(&d_ptr, bytes) != cudaSuccess) {
            fprintf(stderr, "[ERes2V2GPU] cudaMalloc failed for %s (%.1f KB)\n",
                    name.c_str(), bytes / 1024.0f);
            return false;
        }
        cudaMemcpy(d_ptr, data.data(), bytes, cudaMemcpyHostToDevice);
        gpu_tensors_[name] = d_ptr;
        tensor_sizes_[name] = (int)data.size();
        total_bytes += bytes;
    }

    loaded_ = true;
    fprintf(stderr, "[ERes2V2GPU] Loaded %zu tensors (%.1f MB) to GPU\n",
            gpu_tensors_.size(), total_bytes / (1024.0f * 1024.0f));
    return true;
}

const float* GpuERes2NetV2Encoder::get_gpu(const std::string& name) const {
    auto it = gpu_tensors_.find(name);
    if (it == gpu_tensors_.end()) {
        fprintf(stderr, "[ERes2V2GPU] WARNING: tensor '%s' not found\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

// ============================================================================
// Ensure scratch buffers
// ============================================================================
bool GpuERes2NetV2Encoder::ensure_scratch(int T) {
    if (scratch_max_T_ >= T) return true;
    int new_max = std::max(T, 400);  // default for up to 4s chunks
    if (!scratch_.alloc(new_max)) return false;
    scratch_max_T_ = new_max;
    fprintf(stderr, "[ERes2V2GPU] Scratch allocated for T=%d (%.1f MB)\n",
            new_max, scratch_.total_bytes / (1024.0f * 1024.0f));
    return true;
}

// ============================================================================
// GPU Conv2d: im2col + cuBLAS SGEMM
// ============================================================================
void GpuERes2NetV2Encoder::gpu_conv2d(
        const float* input, float* output, float* im2col_buf,
        int Cin, int H, int W, int Cout, int k, int stride, int pad,
        const std::string& weight_name,
        cublasHandle_t cublas, cudaStream_t stream) {
    int H_out = (H + 2 * pad - k) / stride + 1;
    int W_out = (W + 2 * pad - k) / stride + 1;
    int N = H_out * W_out;  // spatial output
    int K = Cin * k * k;

    const float* weight = get_gpu(weight_name);

    if (k == 1 && stride == 1 && pad == 0) {
        // 1×1 conv: direct GEMM, no im2col
        // RowMajor: W[Cout, Cin] × X[Cin, N] → Y[Cout, N]
        // cuBLAS ColMajor: X^T[N, Cin] × W^T[Cin, Cout] → Y^T[N, Cout]
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, Cout, Cin,
                    &alpha, input, N,
                    weight, Cin,
                    &beta, output, N);
    } else {
        // General conv: im2col + GEMM
        int total_im2col = K * N;
        im2col_kernel<<<div_ceil(total_im2col, BLOCK), BLOCK, 0, stream>>>(
            input, im2col_buf, Cin, H, W, k, stride, pad, H_out, W_out);

        // W[Cout, K] × col[K, N] → Y[Cout, N]
        // cuBLAS: col^T[N, K] × W^T[K, Cout] → Y^T[N, Cout]
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, Cout, K,
                    &alpha, im2col_buf, N,
                    weight, K,
                    &beta, output, N);
    }
}

// ============================================================================
// GPU AFF (Attention Feature Fusion)
// ============================================================================
void GpuERes2NetV2Encoder::gpu_aff(
        const float* x, const float* y, float* output,
        ERes2Scratch& sp,
        int C, int H, int W, const std::string& prefix,
        cublasHandle_t cublas, cudaStream_t stream) {
    int spatial = H * W;
    int inter_C = C / 4;  // r=4

    // Concatenate x, y → buf[6] [2C, H, W]
    float* d_concat = sp.buf[6];
    concat_kernel<<<div_ceil((2 * C) * spatial, BLOCK), BLOCK, 0, stream>>>(
        x, y, d_concat, C, C, spatial);

    // local_att.0: Conv2d(2C→inter_C, k=1, bias)
    float* d_h = sp.aff_tmp;
    gpu_conv2d(d_concat, d_h, sp.im2col,
               2 * C, H, W, inter_C, 1, 1, 0,
               prefix + ".local_att.0.weight", cublas, stream);
    conv2d_bias_kernel<<<div_ceil(inter_C * spatial, BLOCK), BLOCK, 0, stream>>>(
        d_h, get_gpu(prefix + ".local_att.0.bias"), inter_C, spatial);

    // local_att.1: BN + SiLU
    float* d_h2 = sp.buf[7];
    bn_silu_kernel<<<div_ceil(inter_C * spatial, BLOCK), BLOCK, 0, stream>>>(
        d_h, d_h2,
        get_gpu(prefix + ".local_att.1.weight"),
        get_gpu(prefix + ".local_att.1.bias"),
        get_gpu(prefix + ".local_att.1.running_mean"),
        get_gpu(prefix + ".local_att.1.running_var"),
        inter_C, spatial);

    // local_att.3: Conv2d(inter_C→C, k=1, bias)
    gpu_conv2d(d_h2, d_h, sp.im2col,
               inter_C, H, W, C, 1, 1, 0,
               prefix + ".local_att.3.weight", cublas, stream);
    conv2d_bias_kernel<<<div_ceil(C * spatial, BLOCK), BLOCK, 0, stream>>>(
        d_h, get_gpu(prefix + ".local_att.3.bias"), C, spatial);

    // local_att.4: BN (no activation)
    eres2_bn_kernel<<<div_ceil(C * spatial, BLOCK), BLOCK, 0, stream>>>(
        d_h, d_h2,
        get_gpu(prefix + ".local_att.4.weight"),
        get_gpu(prefix + ".local_att.4.bias"),
        get_gpu(prefix + ".local_att.4.running_mean"),
        get_gpu(prefix + ".local_att.4.running_var"),
        C, spatial);

    // Combine: out = x * (1+tanh(h)) + y * (1-tanh(h))
    aff_combine_kernel<<<div_ceil(C * spatial, BLOCK), BLOCK, 0, stream>>>(
        x, y, d_h2, output, C * spatial);
}

// ============================================================================
// GPU BasicBlock — Res2Net block with optional AFF
// ============================================================================
void GpuERes2NetV2Encoder::gpu_basic_block(
        const float* input, float* output,
        ERes2Scratch& sp,
        int in_planes, int planes, int H, int W,
        int stride, const std::string& prefix,
        bool use_aff,
        int& H_out, int& W_out,
        cublasHandle_t cublas, cudaStream_t stream) {

    int width = (int)std::floor(planes * (BASE_WIDTH / 64.0));
    int width_x_scale = width * SCALE;
    int out_planes = planes * EXPANSION;

    H_out = (H - 1) / stride + 1;
    W_out = (W - 1) / stride + 1;
    int spatial_out = H_out * W_out;

    // conv1: in_planes → width*scale, k=1, s=stride
    float* d_conv1_out = sp.buf[2];
    gpu_conv2d(input, d_conv1_out, sp.im2col,
               in_planes, H, W, width_x_scale, 1, stride, 0,
               prefix + ".conv1.weight", cublas, stream);

    // BN1 + HardTanh
    float* d_bn1_out = sp.buf[3];
    bn_hardtanh_kernel<<<div_ceil(width_x_scale * spatial_out, BLOCK), BLOCK, 0, stream>>>(
        d_conv1_out, d_bn1_out,
        get_gpu(prefix + ".bn1.weight"),
        get_gpu(prefix + ".bn1.bias"),
        get_gpu(prefix + ".bn1.running_mean"),
        get_gpu(prefix + ".bn1.running_var"),
        width_x_scale, spatial_out);

    // Process scales (Res2Net): for each scale chunk, conv3x3 + BN + HardTanh
    // scale_cat will accumulate [width*SCALE, H_out, W_out]
    float* d_sp_state = sp.buf[4];  // running state for cumulative processing
    float* d_conv_tmp = sp.buf[5];  // temp for conv output

    for (int s = 0; s < SCALE; ++s) {
        // Extract spx[s] from d_bn1_out: channels [s*width .. (s+1)*width)
        float* d_spx = d_conv_tmp;  // reuse
        slice_channels_kernel<<<div_ceil(width * spatial_out, BLOCK), BLOCK, 0, stream>>>(
            d_bn1_out, d_spx, width, spatial_out, s * width);

        if (s == 0) {
            // sp = spx (just copy)
            cudaMemcpyAsync(d_sp_state, d_spx, width * spatial_out * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        } else if (use_aff) {
            // sp = fuse_models[s-1](sp, spx) via AFF
            // AFF internally uses buf[6], buf[7], aff_tmp.
            // Output directly to d_sp_state (in-place safe: aff_combine reads x[i] then writes out[i])
            gpu_aff(d_sp_state, d_spx, d_sp_state,
                    sp, width, H_out, W_out,
                    prefix + ".fuse_models." + std::to_string(s - 1),
                    cublas, stream);
        } else {
            // sp += spx
            eres2_add_kernel<<<div_ceil(width * spatial_out, BLOCK), BLOCK, 0, stream>>>(
                d_sp_state, d_spx, width * spatial_out);
        }

        // convs[s](sp) → d_conv_tmp
        gpu_conv2d(d_sp_state, d_conv_tmp, sp.im2col,
                   width, H_out, W_out, width, 3, 1, 1,
                   prefix + ".convs." + std::to_string(s) + ".weight",
                   cublas, stream);

        // bns[s] + HardTanh → d_sp_state (also copy to scale_cat at correct offset)
        bn_hardtanh_kernel<<<div_ceil(width * spatial_out, BLOCK), BLOCK, 0, stream>>>(
            d_conv_tmp, d_sp_state,
            get_gpu(prefix + ".bns." + std::to_string(s) + ".weight"),
            get_gpu(prefix + ".bns." + std::to_string(s) + ".bias"),
            get_gpu(prefix + ".bns." + std::to_string(s) + ".running_mean"),
            get_gpu(prefix + ".bns." + std::to_string(s) + ".running_var"),
            width, spatial_out);

        // Copy this scale's output to the concatenation area at offset s*width
        cudaMemcpyAsync(sp.scale_cat + s * width * spatial_out,
                        d_sp_state,
                        width * spatial_out * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // conv3: width*scale → out_planes, k=1
    float* d_conv3_out = sp.buf[2];
    gpu_conv2d(sp.scale_cat, d_conv3_out, sp.im2col,
               width_x_scale, H_out, W_out, out_planes, 1, 1, 0,
               prefix + ".conv3.weight", cublas, stream);

    // BN3 (no activation yet)
    float* d_bn3_out = sp.buf[3];
    eres2_bn_kernel<<<div_ceil(out_planes * spatial_out, BLOCK), BLOCK, 0, stream>>>(
        d_conv3_out, d_bn3_out,
        get_gpu(prefix + ".bn3.weight"),
        get_gpu(prefix + ".bn3.bias"),
        get_gpu(prefix + ".bn3.running_mean"),
        get_gpu(prefix + ".bn3.running_var"),
        out_planes, spatial_out);

    // Shortcut
    float* d_shortcut = sp.buf[4];
    if (stride != 1 || in_planes != out_planes) {
        // shortcut.0: Conv2d + shortcut.1: BN
        gpu_conv2d(input, d_shortcut, sp.im2col,
                   in_planes, H, W, out_planes, 1, stride, 0,
                   prefix + ".shortcut.0.weight", cublas, stream);
        float* d_shortcut_bn = sp.buf[5];
        eres2_bn_kernel<<<div_ceil(out_planes * spatial_out, BLOCK), BLOCK, 0, stream>>>(
            d_shortcut, d_shortcut_bn,
            get_gpu(prefix + ".shortcut.1.weight"),
            get_gpu(prefix + ".shortcut.1.bias"),
            get_gpu(prefix + ".shortcut.1.running_mean"),
            get_gpu(prefix + ".shortcut.1.running_var"),
            out_planes, spatial_out);
        d_shortcut = d_shortcut_bn;
    } else {
        // Identity shortcut: just point to input
        d_shortcut = const_cast<float*>(input);
    }

    // Residual add + HardTanh → output
    add_hardtanh_kernel<<<div_ceil(out_planes * spatial_out, BLOCK), BLOCK, 0, stream>>>(
        d_bn3_out, d_shortcut, output, out_planes * spatial_out);
}

// ============================================================================
// forward_one — core forward pass for one chunk
// ============================================================================
void GpuERes2NetV2Encoder::forward_one(
        const float* d_mel, int T, ERes2Scratch& sp,
        cudaStream_t stream, cublasHandle_t cublas,
        float* d_emb_out) {

    cublasSetStream(cublas, stream);

    // 0. CMN + Transpose: [T, 80] → [80, T]  (treat as [C=1, H=80, W=T])
    float* d_input = sp.buf[0];
    eres2_cmn_transpose_kernel<<<div_ceil(T * 80, BLOCK), BLOCK, 0, stream>>>(
        d_mel, d_input, T, 80);

    // 1. conv1(1→64, k=3, s=1, p=1) → BN1 → ReLU
    int H = 80, W = T;
    float* d_conv1 = sp.buf[1];
    gpu_conv2d(d_input, d_conv1, sp.im2col,
               1, H, W, 64, 3, 1, 1,
               "conv1.weight", cublas, stream);

    float* d_bn1 = sp.buf[0];
    eres2_bn_relu_kernel<<<div_ceil(64 * H * W, BLOCK), BLOCK, 0, stream>>>(
        d_conv1, d_bn1,
        get_gpu("bn1.weight"), get_gpu("bn1.bias"),
        get_gpu("bn1.running_mean"), get_gpu("bn1.running_var"),
        64, H * W);

    // Layer buffers: alternate between buf[0] and buf[1] for layer input/output
    float* d_layer_in = d_bn1;   // buf[0]
    float* d_layer_out = sp.buf[1];
    int in_planes = 64;

    // 2. layer1: 3× BasicBlock(64→128, s=1)
    for (int i = 0; i < 3; ++i) {
        int s = (i == 0) ? 1 : 1;
        int H_out, W_out;
        gpu_basic_block(d_layer_in, d_layer_out, sp,
                        in_planes, 64, H, W, s,
                        "layer1." + std::to_string(i),
                        false, H_out, W_out, cublas, stream);
        H = H_out; W = W_out;
        in_planes = 64 * EXPANSION;  // 128
        std::swap(d_layer_in, d_layer_out);
    }

    // 3. layer2: 4× BasicBlock(128→256, s=2)
    for (int i = 0; i < 4; ++i) {
        int s = (i == 0) ? 2 : 1;
        int H_out, W_out;
        gpu_basic_block(d_layer_in, d_layer_out, sp,
                        in_planes, 128, H, W, s,
                        "layer2." + std::to_string(i),
                        false, H_out, W_out, cublas, stream);
        H = H_out; W = W_out;
        in_planes = 128 * EXPANSION;  // 256
        std::swap(d_layer_in, d_layer_out);
    }

    // d_layer_in has current output after swaps

    // 4. layer3: 6× BasicBlockAFF(256→512, s=2)
    for (int i = 0; i < 6; ++i) {
        int s = (i == 0) ? 2 : 1;
        int H_out, W_out;
        gpu_basic_block(d_layer_in, d_layer_out, sp,
                        in_planes, 256, H, W, s,
                        "layer3." + std::to_string(i),
                        true, H_out, W_out, cublas, stream);
        H = H_out; W = W_out;
        in_planes = 256 * EXPANSION;  // 512
        std::swap(d_layer_in, d_layer_out);
    }

    // Save layer3 output for layer3_ds + fuse34
    int H3 = H, W3 = W;
    // d_layer_in is layer3 output. Copy to dedicated out3 buffer (safe from basic_block internals)
    float* d_out3 = sp.out3;
    cudaMemcpyAsync(d_out3, d_layer_in, (size_t)in_planes * H3 * W3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // 5. layer4: 3× BasicBlockAFF(512→1024, s=2)
    for (int i = 0; i < 3; ++i) {
        int s = (i == 0) ? 2 : 1;
        int H_out, W_out;
        gpu_basic_block(d_layer_in, d_layer_out, sp,
                        in_planes, 512, H, W, s,
                        "layer4." + std::to_string(i),
                        true, H_out, W_out, cublas, stream);
        H = H_out; W = W_out;
        in_planes = 512 * EXPANSION;  // 1024
        std::swap(d_layer_in, d_layer_out);
    }
    // d_layer_in = layer4 output [1024, H4, W4]
    int H4 = H, W4 = W;

    // 6. layer3_ds: Conv2d(512→1024, k=3, s=2, p=1) on layer3 output
    float* d_out3_ds = sp.buf[2];
    gpu_conv2d(d_out3, d_out3_ds, sp.im2col,
               512, H3, W3, 1024, 3, 2, 1,
               "layer3_ds.weight", cublas, stream);

    // 7. fuse34: AFF fusion of layer4_out and out3_ds
    float* d_fused = sp.buf[3];
    gpu_aff(d_layer_in, d_out3_ds, d_fused, sp,
            1024, H4, W4, "fuse34",
            cublas, stream);

    // 8. TSTP pool: [1024, H4, W4] → [1024*H4*2]
    int CH = 1024 * H4;
    float* d_stats = sp.buf[4];
    tstp_pool_kernel<<<div_ceil(CH, BLOCK), BLOCK, 0, stream>>>(
        d_fused, d_stats, 1024, H4, W4);

    // 9. seg_1: Linear(20480→192) — weight [192, 20480], bias [192]
    int feat_dim = CH * 2;  // 1024 * H4 * 2 = e.g. 20480
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas, CUBLAS_OP_T, feat_dim, 192,
                    &alpha, get_gpu("seg_1.weight"),
                    feat_dim, d_stats, 1,
                    &beta, d_emb_out, 1);
    }
    // Add bias
    const float* d_seg_bias = get_gpu("seg_1.bias");
    if (d_seg_bias) {
        eres2_add_kernel<<<div_ceil(192, BLOCK), BLOCK, 0, stream>>>(
            d_emb_out, d_seg_bias, 192);
    }

    // 10. L2 normalize
    eres2_l2_normalize_kernel<<<1, 1, 0, stream>>>(d_emb_out, 192);
}

// ============================================================================
// extract_gpu — single chunk, GPU d_mel input
// ============================================================================
std::vector<float> GpuERes2NetV2Encoder::extract_gpu(const float* d_mel, int T) {
    if (!loaded_ || T < 10) return {};
    if (!ensure_scratch(T)) return {};

    float* d_emb = nullptr;
    cudaMalloc(&d_emb, 192 * sizeof(float));

    cublasSetStream(cublas_, stream_);
    forward_one(d_mel, T, scratch_, stream_, cublas_, d_emb);
    cudaStreamSynchronize(stream_);

    std::vector<float> emb(192);
    cudaMemcpy(emb.data(), d_emb, 192 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_emb);

    // Validate
    for (float v : emb)
        if (std::isnan(v) || std::isinf(v)) return {};

    return emb;
}

// ============================================================================
// Batch resources
// ============================================================================
bool GpuERes2NetV2Encoder::ensure_batch(int max_T) {
    if (batch_.initialized && max_T <= batch_.max_T) return true;

    int new_max = std::max(max_T, 400);

    for (int i = 0; i < BATCH_CONCURRENCY; i++) {
        if (!batch_.initialized) {
            cudaStreamCreate(&batch_.streams[i]);
            cublasCreate(&batch_.cublas[i]);
        }
        batch_.scratch[i].free();
        if (!batch_.scratch[i].alloc(new_max)) {
            fprintf(stderr, "[ERes2V2GPU] batch scratch[%d] alloc failed for T=%d\n", i, new_max);
            return false;
        }
    }

    if (!batch_.d_emb_buf) {
        cudaMalloc(&batch_.d_emb_buf, BATCH_CONCURRENCY * 192 * sizeof(float));
    }

    batch_.max_T = new_max;
    batch_.initialized = true;
    fprintf(stderr, "[ERes2V2GPU] Batch resources: %d streams, scratch %.1f MB each\n",
            BATCH_CONCURRENCY,
            batch_.scratch[0].total_bytes / (1024.0f * 1024.0f));
    return true;
}

// ============================================================================
// extract_batch_gpu — multi-stream batch embedding extraction
// ============================================================================
std::vector<std::vector<float>> GpuERes2NetV2Encoder::extract_batch_gpu(
        const std::vector<BatchChunk>& chunks) {
    int n = (int)chunks.size();
    std::vector<std::vector<float>> results(n);
    if (!loaded_ || n == 0) return results;

    int max_T = 0;
    for (auto& c : chunks) max_T = std::max(max_T, c.T);
    if (!ensure_batch(max_T)) return results;

    const int emb_size = 192;

    for (int base = 0; base < n; base += BATCH_CONCURRENCY) {
        int batch_size = std::min(BATCH_CONCURRENCY, n - base);

        for (int i = 0; i < batch_size; i++) {
            auto& c = chunks[base + i];
            if (c.T < 10) continue;

            cublasSetStream(batch_.cublas[i], batch_.streams[i]);
            forward_one(c.d_mel, c.T,
                       batch_.scratch[i], batch_.streams[i],
                       batch_.cublas[i],
                       batch_.d_emb_buf + i * emb_size);
        }

        for (int i = 0; i < batch_size; i++)
            cudaStreamSynchronize(batch_.streams[i]);

        std::vector<float> host_embs(batch_size * emb_size);
        cudaMemcpy(host_embs.data(), batch_.d_emb_buf,
                   batch_size * emb_size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; i++) {
            auto& c = chunks[base + i];
            if (c.T < 10) continue;

            float* emb = host_embs.data() + i * emb_size;
            bool valid = true;
            for (int j = 0; j < emb_size; j++) {
                if (std::isnan(emb[j]) || std::isinf(emb[j])) { valid = false; break; }
            }
            if (valid)
                results[base + i].assign(emb, emb + emb_size);
        }
    }

    return results;
}

// ============================================================================
// Safetensors loader (same as CAM++ GPU version)
// ============================================================================
GpuERes2NetV2Encoder::TensorMap GpuERes2NetV2Encoder::load_safetensors(const std::string& path) {
    TensorMap result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return result;

    uint64_t header_size = 0;
    ifs.read(reinterpret_cast<char*>(&header_size), 8);
    if (header_size > 10000000) return result;

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
