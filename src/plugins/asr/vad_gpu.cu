// vad_gpu.cu — GPU-Accelerated FSMN-VAD Implementation
//
// 核心思路: 所有帧一次性 batch 推理
//   1. Fbank: cuFFT batched R2C + mel filterbank SGEMM
//   2. LFR: GPU kernel 拼接连续 5 帧
//   3. CMVN: GPU kernel
//   4. FSMN: 线性层 → SGEMM, 因果卷积 → 自定义 kernel
//   5. State machine: CPU 顺序扫描 speech_prob
//
// 权重总量: ~0.4M 参数 ≈ 1.6 MB FP32 → GPU 内存开销极小

#include "vad_gpu.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <cstring>

namespace {
// 简易 safetensors 解析 (CPU, F32 only) — 从 vad_engine.h 提取
using TensorMap = std::unordered_map<std::string, std::vector<float>>;
TensorMap load_safetensors_simple(const std::string& path) {
    TensorMap result;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return result;
    uint64_t header_size = 0;
    ifs.read(reinterpret_cast<char*>(&header_size), 8);
    if (header_size > 100000) return result;
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
        if (bracket == std::string::npos) break;
        size_t comma = header.find(',', bracket);
        if (comma == std::string::npos) break;
        size_t end_bracket = header.find(']', comma);
        if (end_bracket == std::string::npos) break;
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
} // anonymous namespace

namespace qwen_thor {
namespace asr {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Hamming window + frame extraction + zero-pad
__global__ void vad_frame_window_kernel(const float* __restrict__ pcm,
                                         const float* __restrict__ window,
                                         float* __restrict__ frames,
                                         int num_frames, int n_fft, int hop, int fft_size) {
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    if (frame_idx >= num_frames || sample_idx >= fft_size) return;

    float* out = frames + frame_idx * fft_size;
    int pcm_offset = frame_idx * hop;

    if (sample_idx < n_fft) {
        out[sample_idx] = pcm[pcm_offset + sample_idx] * window[sample_idx];
    } else {
        out[sample_idx] = 0.0f;
    }
}

// Power spectrum → mel filterbank → log
__global__ void vad_power_spectrum_kernel(const cufftComplex* __restrict__ fft_out,
                                           float* __restrict__ power,
                                           int num_frames, int n_freqs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames * n_freqs) return;
    float re = fft_out[idx].x;
    float im = fft_out[idx].y;
    power[idx] = re * re + im * im;
}

// Log inplace
__global__ void vad_log_kernel(float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = logf(fmaxf(data[idx], 1e-10f));
}

// LFR: 拼接连续 lfr_m 帧 Fbank → [num_lfr, input_dim]
// input: [num_fbank, n_mels], output: [num_lfr, lfr_m * n_mels]
__global__ void lfr_kernel(const float* __restrict__ fbank,
                            float* __restrict__ lfr_out,
                            int num_fbank, int n_mels, int lfr_m, int num_lfr) {
    int lfr_idx = blockIdx.x;
    int feat_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int input_dim = lfr_m * n_mels;
    if (lfr_idx >= num_lfr || feat_idx >= input_dim) return;

    int sub_frame = feat_idx / n_mels;  // which of the lfr_m sub-frames
    int mel_idx = feat_idx % n_mels;
    int fbank_frame = lfr_idx + sub_frame;
    if (fbank_frame >= num_fbank) {
        lfr_out[lfr_idx * input_dim + feat_idx] = 0.0f;
    } else {
        lfr_out[lfr_idx * input_dim + feat_idx] = fbank[fbank_frame * n_mels + mel_idx];
    }
}

// CMVN: x[i] = (x[i] - mean[i]) * invstd[i]
__global__ void cmvn_kernel(float* __restrict__ data,
                             const float* __restrict__ mean,
                             const float* __restrict__ invstd,
                             int num_frames, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames * dim) return;
    int d = idx % dim;
    data[idx] = (data[idx] - mean[d]) * invstd[d];
}

// Add bias + ReLU fused: y[i] = max(0, x[i] + bias[i % dim])
__global__ void bias_relu_kernel(float* __restrict__ data,
                                  const float* __restrict__ bias,
                                  int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % dim;
    float val = data[idx] + bias[d];
    data[idx] = fmaxf(val, 0.0f);
}

// Add bias (no relu): y[i] = x[i] + bias[i % dim]
__global__ void bias_kernel(float* __restrict__ data,
                             const float* __restrict__ bias,
                             int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] += bias[idx % dim];
}

// Causal conv1d for FSMN memory block
// input: [L, proj_dim], output: [L, proj_dim]
// weight: [proj_dim, lorder] (tap 0 = current frame, tap k = k frames ago)
// out[t, d] = Σ_{k=0}^{lorder-1} weight[d, k] * input[t-k, d] (where input[<0] = 0)
__global__ void fsmn_causal_conv_kernel(const float* __restrict__ input,
                                         const float* __restrict__ weight,
                                         float* __restrict__ output,
                                         int L, int proj_dim, int lorder) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= L || d >= proj_dim) return;

    float sum = 0.0f;
    for (int k = 0; k < lorder; ++k) {
        int src_t = t - k;
        if (src_t >= 0) {
            sum += weight[d * lorder + k] * input[src_t * proj_dim + d];
        }
    }
    output[t * proj_dim + d] = sum;
}

// Residual add + ReLU: a[i] = max(0, a[i] + b[i])
__global__ void add_relu_kernel(float* __restrict__ a, const float* __restrict__ b,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = fmaxf(a[idx] + b[idx], 0.0f);
}

// Residual add: a[i] += b[i]
__global__ void add_kernel_vad(float* __restrict__ a, const float* __restrict__ b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] += b[idx];
}

// Softmax → extract speech probability (1 - class[0])
// input: [L, output_dim], output: [L] speech probabilities
__global__ void softmax_speech_prob_kernel(const float* __restrict__ logits,
                                            float* __restrict__ probs,
                                            int L, int output_dim, int sil_class) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;

    const float* row = logits + t * output_dim;
    float max_val = row[0];
    for (int i = 1; i < output_dim; ++i)
        max_val = fmaxf(max_val, row[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < output_dim; ++i)
        sum_exp += expf(row[i] - max_val);

    float sil_prob = expf(row[sil_class] - max_val) / sum_exp;
    probs[t] = 1.0f - sil_prob;
}

// ============================================================================
// GpuVadEngine implementation
// ============================================================================

GpuVadEngine::GpuVadEngine() = default;

GpuVadEngine::~GpuVadEngine() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);

    cudaFree(d_in1_w_); cudaFree(d_in1_b_);
    cudaFree(d_in2_w_); cudaFree(d_in2_b_);
    for (int l = 0; l < 4; ++l) {
        cudaFree(fsmn_weights_[l].d_linear_w);
        cudaFree(fsmn_weights_[l].d_fsmn_w);
        cudaFree(fsmn_weights_[l].d_affine_w);
        cudaFree(fsmn_weights_[l].d_affine_b);
    }
    cudaFree(d_out1_w_); cudaFree(d_out1_b_);
    cudaFree(d_out2_w_); cudaFree(d_out2_b_);
    cudaFree(d_cmvn_mean_); cudaFree(d_cmvn_invstd_);
    cudaFree(d_window_); cudaFree(d_mel_fb_);
    cudaFree(d_pcm_); cudaFree(d_frames_); cudaFree(d_fft_out_);
    cudaFree(d_fbank_); cudaFree(d_lfr_);
    cudaFree(d_h_); cudaFree(d_tmp_); cudaFree(d_probs_);
}

// Helper: allocate GPU tensor and copy from CPU
static float* alloc_and_copy(const std::vector<float>& data) {
    if (data.empty()) return nullptr;
    float* d = nullptr;
    cudaMalloc(&d, data.size() * sizeof(float));
    cudaMemcpy(d, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}

bool GpuVadEngine::load(const std::string& model_dir) {
    config_.model_path = model_dir + "/fsmn_vad.safetensors";
    config_.cmvn_path  = model_dir + "/cmvn.safetensors";

    // Load weights
    auto tensors = load_safetensors_simple(config_.model_path);
    if (tensors.empty()) { fprintf(stderr, "[GpuVAD] Failed to load weights\n"); return false; }

    auto get = [&](const std::string& name) -> const std::vector<float>* {
        auto it = tensors.find(name);
        return it != tensors.end() ? &it->second : nullptr;
    };

    // in_linear1/2
    auto* w = get("encoder.in_linear1.linear.weight");
    auto* b = get("encoder.in_linear1.linear.bias");
    if (!w || !b) return false;
    d_in1_w_ = alloc_and_copy(*w);
    d_in1_b_ = alloc_and_copy(*b);

    w = get("encoder.in_linear2.linear.weight");
    b = get("encoder.in_linear2.linear.bias");
    if (!w || !b) return false;
    d_in2_w_ = alloc_and_copy(*w);
    d_in2_b_ = alloc_and_copy(*b);

    // FSMN layers
    for (int l = 0; l < config_.fsmn_layers; ++l) {
        std::string prefix = "encoder.fsmn." + std::to_string(l) + ".";
        w = get(prefix + "linear.linear.weight");
        if (!w) return false;
        fsmn_weights_[l].d_linear_w = alloc_and_copy(*w);

        w = get(prefix + "fsmn_block.conv_left.weight");
        if (!w) return false;
        fsmn_weights_[l].d_fsmn_w = alloc_and_copy(*w);

        w = get(prefix + "affine.linear.weight");
        b = get(prefix + "affine.linear.bias");
        if (!w || !b) return false;
        fsmn_weights_[l].d_affine_w = alloc_and_copy(*w);
        fsmn_weights_[l].d_affine_b = alloc_and_copy(*b);
    }

    // out_linear1/2
    w = get("encoder.out_linear1.linear.weight");
    b = get("encoder.out_linear1.linear.bias");
    if (!w || !b) return false;
    d_out1_w_ = alloc_and_copy(*w);
    d_out1_b_ = alloc_and_copy(*b);

    w = get("encoder.out_linear2.linear.weight");
    b = get("encoder.out_linear2.linear.bias");
    if (!w || !b) return false;
    d_out2_w_ = alloc_and_copy(*w);
    d_out2_b_ = alloc_and_copy(*b);

    // CMVN
    auto cmvn = load_safetensors_simple(config_.cmvn_path);
    if (cmvn.find("cmvn_mean") == cmvn.end() || cmvn.find("cmvn_invstd") == cmvn.end()) {
        fprintf(stderr, "[GpuVAD] Failed to load CMVN\n");
        return false;
    }
    d_cmvn_mean_ = alloc_and_copy(cmvn["cmvn_mean"]);
    d_cmvn_invstd_ = alloc_and_copy(cmvn["cmvn_invstd"]);

    // CUDA resources
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    // Fbank constants
    build_fbank_constants();

    loaded_ = true;
    fprintf(stderr, "[GpuVAD] FSMN-VAD loaded to GPU (~%.1f KB weights)\n",
            (float)(tensors.size() * 4) / 1024.0f);
    return true;
}

void GpuVadEngine::build_fbank_constants() {
    int n_fft = config_.window_samples();  // 400
    fft_size_ = 1;
    while (fft_size_ < n_fft) fft_size_ <<= 1;  // 512
    int n_freqs = fft_size_ / 2 + 1;

    // Hamming window
    std::vector<float> win(n_fft);
    for (int i = 0; i < n_fft; ++i)
        win[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (n_fft - 1));
    cudaMalloc(&d_window_, n_fft * sizeof(float));
    cudaMemcpy(d_window_, win.data(), n_fft * sizeof(float), cudaMemcpyHostToDevice);

    // Mel filterbank (HTK)
    std::vector<float> fb(config_.n_mels * n_freqs, 0.0f);
    auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel((float)config_.sample_rate / 2.0f);
    std::vector<float> mel_points(config_.n_mels + 2);
    for (int i = 0; i < config_.n_mels + 2; ++i)
        mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (config_.n_mels + 1));

    float freq_step = (float)config_.sample_rate / fft_size_;
    for (int m = 0; m < config_.n_mels; ++m) {
        for (int k = 0; k < n_freqs; ++k) {
            float freq = k * freq_step;
            if (freq >= mel_points[m] && freq <= mel_points[m + 1])
                fb[m * n_freqs + k] = (freq - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
            else if (freq > mel_points[m + 1] && freq <= mel_points[m + 2])
                fb[m * n_freqs + k] = (mel_points[m + 2] - freq) / (mel_points[m + 2] - mel_points[m + 1]);
        }
    }
    cudaMalloc(&d_mel_fb_, config_.n_mels * n_freqs * sizeof(float));
    cudaMemcpy(d_mel_fb_, fb.data(), config_.n_mels * n_freqs * sizeof(float), cudaMemcpyHostToDevice);
}

bool GpuVadEngine::ensure_scratch(int num_fbank_frames, int num_lfr_frames) {
    int n_fft = config_.window_samples();
    int n_freqs = fft_size_ / 2 + 1;
    int input_dim = config_.input_dim;      // 400
    int linear_dim = config_.linear_dim;    // 250
    int proj_dim = config_.proj_dim;        // 128
    int output_dim = config_.output_dim;    // 248

    if (num_fbank_frames > scratch_max_fbank_) {
        cudaFree(d_pcm_); d_pcm_ = nullptr;
        cudaFree(d_frames_); d_frames_ = nullptr;
        cudaFree(d_fft_out_); d_fft_out_ = nullptr;
        cudaFree(d_fbank_); d_fbank_ = nullptr;

        scratch_max_fbank_ = num_fbank_frames + num_fbank_frames / 4;
        // PCM: generous allocation (fbank_frames * hop + n_fft)
        size_t max_pcm = (size_t)scratch_max_fbank_ * config_.frame_samples() + n_fft;
        cudaMalloc(&d_pcm_, max_pcm * sizeof(float));
        cudaMalloc(&d_frames_, (size_t)scratch_max_fbank_ * fft_size_ * sizeof(float));
        cudaMalloc(&d_fft_out_, (size_t)scratch_max_fbank_ * n_freqs * sizeof(cufftComplex));
        // fbank: [n_mels, num_frames] for SGEMM output
        cudaMalloc(&d_fbank_, (size_t)config_.n_mels * scratch_max_fbank_ * sizeof(float));
    }

    // cuFFT plan (exact batch size)
    if (num_fbank_frames != fft_plan_batch_) {
        if (fft_plan_) cufftDestroy(fft_plan_);
        cufftPlan1d(&fft_plan_, fft_size_, CUFFT_R2C, num_fbank_frames);
        cufftSetStream(fft_plan_, stream_);
        fft_plan_batch_ = num_fbank_frames;
    }

    if (num_lfr_frames > scratch_max_lfr_) {
        cudaFree(d_lfr_); d_lfr_ = nullptr;
        cudaFree(d_h_); d_h_ = nullptr;
        cudaFree(d_tmp_); d_tmp_ = nullptr;
        cudaFree(d_probs_); d_probs_ = nullptr;

        scratch_max_lfr_ = num_lfr_frames + num_lfr_frames / 4;
        cudaMalloc(&d_lfr_, (size_t)scratch_max_lfr_ * input_dim * sizeof(float));
        cudaMalloc(&d_h_, (size_t)scratch_max_lfr_ * std::max({linear_dim, proj_dim, output_dim, input_dim}) * sizeof(float));
        cudaMalloc(&d_tmp_, (size_t)scratch_max_lfr_ * std::max({linear_dim, proj_dim, output_dim, input_dim}) * sizeof(float));
        cudaMalloc(&d_probs_, (size_t)scratch_max_lfr_ * sizeof(float));
    }

    return true;
}

std::vector<float> GpuVadEngine::forward_batch(const float* pcm, int num_samples,
                                                 int& num_lfr_frames_out) {
    int n_fft = config_.window_samples();   // 400
    int hop = config_.frame_samples();      // 160
    int n_freqs = fft_size_ / 2 + 1;
    int n_mels = config_.n_mels;
    int lfr_m = config_.lfr_m;             // 5
    int input_dim = config_.input_dim;      // 400
    int linear_dim = config_.linear_dim;    // 250
    int proj_dim = config_.proj_dim;        // 128
    int output_dim = config_.output_dim;    // 248

    int num_fbank_frames = (num_samples - n_fft) / hop + 1;
    if (num_fbank_frames < lfr_m) return {};

    int num_lfr = num_fbank_frames - lfr_m + 1;  // valid LFR frames
    num_lfr_frames_out = num_lfr;

    ensure_scratch(num_fbank_frames, num_lfr);

    const int BLOCK = 256;

    // ================================================================
    // Step 1: Upload PCM → GPU
    // ================================================================
    cudaMemcpyAsync(d_pcm_, pcm, num_samples * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // ================================================================
    // Step 2: Fbank extraction (cuFFT approach)
    // ================================================================
    // 2a. Frame extraction + Hamming window
    vad_frame_window_kernel<<<num_fbank_frames, fft_size_, 0, stream_>>>(
        d_pcm_, d_window_, d_frames_, num_fbank_frames, n_fft, hop, fft_size_);

    // 2b. Batched FFT
    cufftExecR2C(fft_plan_, d_frames_, d_fft_out_);

    // 2c. Power spectrum
    int total_freq = num_fbank_frames * n_freqs;
    vad_power_spectrum_kernel<<<(total_freq + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_fft_out_, (float*)d_fft_out_, num_fbank_frames, n_freqs);
    // Reuse d_fft_out_ memory for power (cufftComplex is 8 bytes, float is 4, so fits)
    float* d_power = (float*)d_fft_out_;

    // 2d. Mel filterbank: d_fbank_[n_mels, T] = mel_fb[n_mels, n_freqs] × power[n_freqs, T]
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    n_mels, num_fbank_frames, n_freqs,
                    &alpha,
                    d_mel_fb_, n_freqs,      // [n_freqs, n_mels] col-major
                    d_power, n_freqs,        // [n_freqs, T] col-major
                    &beta,
                    d_fbank_, n_mels);       // [n_mels, T] col-major = [T, n_mels] row-major
    }

    // 2e. Log
    int total_fbank = num_fbank_frames * n_mels;
    vad_log_kernel<<<(total_fbank + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_fbank_, total_fbank);

    // ================================================================
    // Step 3: LFR concatenation
    // ================================================================
    // d_fbank_ is [T_fb, n_mels] row-major. LFR: concatenate lfr_m consecutive frames
    // d_lfr_[t, :] = concat(d_fbank_[t], d_fbank_[t+1], ..., d_fbank_[t+lfr_m-1])
    {
        dim3 grid(num_lfr, (input_dim + BLOCK - 1) / BLOCK);
        lfr_kernel<<<grid, BLOCK, 0, stream_>>>(
            d_fbank_, d_lfr_, num_fbank_frames, n_mels, lfr_m, num_lfr);
    }

    // ================================================================
    // Step 4: CMVN normalization
    // ================================================================
    {
        int total = num_lfr * input_dim;
        cmvn_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
            d_lfr_, d_cmvn_mean_, d_cmvn_invstd_, num_lfr, input_dim);
    }

    // ================================================================
    // Step 5: FSMN forward (batch mode)
    // ================================================================
    // in_linear1: [L, 400] × [400, 140]^T → [L, 140] + bias + ReLU
    // cuBLAS SGEMM: C[L, 140] = A[L, 400] × B[400, 140]^T
    //   col-major: C[140, L] = B^T[140, 400] × A^T[400, L]
    //   But weight is stored row-major [out=140, in=400], which is col-major [400, 140]
    //   So transA = CUBLAS_OP_T for weight, transB = CUBLAS_OP_N for input
    //   Actually: Y = X * W^T → Y^T = W * X^T
    //   X: [L, 400] row = [400, L] col, ldb=400
    //   W: [140, 400] row = [400, 140] col, lda=400
    //   Y: [L, 140] row = [140, L] col, ldc=140
    //   cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, 140, L, 400, ...)
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    config_.input_affine_dim, num_lfr, input_dim,      // m=140, n=L, k=400
                    &alpha,
                    d_in1_w_, input_dim,                                // [400, 140] col
                    d_lfr_, input_dim,                                  // [400, L] col
                    &beta,
                    d_h_, config_.input_affine_dim);                    // [140, L] col
        int total = num_lfr * config_.input_affine_dim;
        bias_relu_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
            d_h_, d_in1_b_, total, config_.input_affine_dim);
    }

    // in_linear2: [L, 140] → [L, 250] + bias + ReLU
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    linear_dim, num_lfr, config_.input_affine_dim,
                    &alpha,
                    d_in2_w_, config_.input_affine_dim,
                    d_h_, config_.input_affine_dim,
                    &beta,
                    d_tmp_, linear_dim);
        int total = num_lfr * linear_dim;
        bias_relu_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
            d_tmp_, d_in2_b_, total, linear_dim);
    }

    // d_tmp_ now contains h: [L, 250]

    // 4 FSMN blocks
    for (int l = 0; l < config_.fsmn_layers; ++l) {
        auto& fw = fsmn_weights_[l];

        // linear: [L, 250] → [L, 128] (no bias, no activation)
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    proj_dim, num_lfr, linear_dim,
                    &alpha,
                    fw.d_linear_w, linear_dim,
                    d_tmp_, linear_dim,
                    &beta,
                    d_h_, proj_dim);

        // FSMN causal conv: [L, 128] → [L, 128]
        // d_h_ = projection, need conv output in d_lfr_ (reuse as temp)
        {
            dim3 grid(num_lfr, (proj_dim + BLOCK - 1) / BLOCK);
            fsmn_causal_conv_kernel<<<grid, BLOCK, 0, stream_>>>(
                d_h_, fw.d_fsmn_w, d_lfr_, num_lfr, proj_dim, config_.lorder);
        }

        // p = h + conv_out (skip connection)
        {
            int total = num_lfr * proj_dim;
            add_kernel_vad<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
                d_h_, d_lfr_, total);
        }

        // affine: [L, 128] → [L, 250] + bias + ReLU, accumulated to d_tmp_ (residual)
        // h_new = ReLU(p × affine_w^T + affine_b)
        // Then h = h + h_new (residual)

        // Compute h_new into d_lfr_ (reuse)
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    linear_dim, num_lfr, proj_dim,
                    &alpha,
                    fw.d_affine_w, proj_dim,
                    d_h_, proj_dim,
                    &beta,
                    d_lfr_, linear_dim);
        {
            int total = num_lfr * linear_dim;
            bias_relu_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
                d_lfr_, fw.d_affine_b, total, linear_dim);
        }

        // Residual: d_tmp_ += d_lfr_
        {
            int total = num_lfr * linear_dim;
            add_kernel_vad<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
                d_tmp_, d_lfr_, total);
        }
    }

    // d_tmp_ now contains h: [L, 250] after all FSMN blocks

    // out_linear1: [L, 250] → [L, 140] + bias + ReLU
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    config_.output_affine_dim, num_lfr, linear_dim,
                    &alpha,
                    d_out1_w_, linear_dim,
                    d_tmp_, linear_dim,
                    &beta,
                    d_h_, config_.output_affine_dim);
        int total = num_lfr * config_.output_affine_dim;
        bias_relu_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
            d_h_, d_out1_b_, total, config_.output_affine_dim);
    }

    // out_linear2: [L, 140] → [L, 248] (no activation)
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    output_dim, num_lfr, config_.output_affine_dim,
                    &alpha,
                    d_out2_w_, config_.output_affine_dim,
                    d_h_, config_.output_affine_dim,
                    &beta,
                    d_tmp_, output_dim);
        int total = num_lfr * output_dim;
        bias_kernel<<<(total + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
            d_tmp_, d_out2_b_, total, output_dim);
    }

    // Softmax → speech probability
    softmax_speech_prob_kernel<<<(num_lfr + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_tmp_, d_probs_, num_lfr, output_dim, config_.sil_pdf_ids[0]);

    // Copy probabilities back to CPU
    cudaStreamSynchronize(stream_);
    std::vector<float> probs(num_lfr);
    cudaMemcpy(probs.data(), d_probs_, num_lfr * sizeof(float), cudaMemcpyDeviceToHost);

    return probs;
}

std::vector<GpuVadSegment> GpuVadEngine::run_state_machine(const std::vector<float>& probs,
                                                             int max_silence_ms,
                                                             int max_segment_ms) {
    std::vector<GpuVadSegment> segments;
    int frame_shift = config_.frame_shift_ms;
    float thres = config_.speech_noise_thres;
    int window_frames = config_.window_size_ms / frame_shift;

    enum class State { IDLE, SPEECH };
    State state = State::IDLE;
    int speech_start_ms = -1;
    int L = (int)probs.size();

    // count_speech: how many frames in [from, to) have speech_prob >= thres
    auto count_speech = [&](int from, int to) {
        int count = 0;
        for (int i = std::max(0, from); i < std::min(to, L); ++i)
            if (probs[i] >= thres) ++count;
        return count;
    };

    for (int t = 0; t < L; ++t) {
        int cur_ms = t * frame_shift;
        bool is_speech = probs[t] >= thres;

        switch (state) {
        case State::IDLE:
            if (is_speech) {
                int speech_count = count_speech(t - window_frames, t + 1);
                int thres_frames = config_.sil_to_speech_time_thres / frame_shift;
                if (speech_count >= thres_frames) {
                    state = State::SPEECH;
                    speech_start_ms = std::max(0, cur_ms - config_.lookback_time_start_point);
                }
            }
            break;

        case State::SPEECH:
            if (!is_speech) {
                // Count trailing silence
                int silence_count = 0;
                for (int i = t; i >= 0; --i) {
                    if (probs[i] < thres) ++silence_count;
                    else break;
                }
                int silence_ms = silence_count * frame_shift;
                if (silence_ms >= max_silence_ms) {
                    int end_ms = cur_ms - silence_ms + config_.lookahead_time_end_point;
                    GpuVadSegment seg;
                    seg.start_ms = speech_start_ms;
                    seg.end_ms = end_ms;
                    segments.push_back(seg);
                    state = State::IDLE;
                    speech_start_ms = -1;
                }
            }
            // Max duration limit
            if (state == State::SPEECH && cur_ms - speech_start_ms >= max_segment_ms) {
                GpuVadSegment seg;
                seg.start_ms = speech_start_ms;
                seg.end_ms = cur_ms;
                segments.push_back(seg);
                state = State::IDLE;
                speech_start_ms = -1;
            }
            break;
        }
    }

    // Final segment
    if (state == State::SPEECH) {
        GpuVadSegment seg;
        seg.start_ms = speech_start_ms;
        seg.end_ms = L * frame_shift;
        segments.push_back(seg);
    }

    return segments;
}

std::vector<GpuVadSegment> GpuVadEngine::detect_all(const float* pcm, int num_samples,
                                                       int max_silence_ms, int max_segment_ms) {
    if (!loaded_) return {};

    int num_lfr_frames = 0;
    auto probs = forward_batch(pcm, num_samples, num_lfr_frames);
    if (probs.empty()) return {};

    return run_state_machine(probs, max_silence_ms, max_segment_ms);
}

} // namespace asr
} // namespace qwen_thor
