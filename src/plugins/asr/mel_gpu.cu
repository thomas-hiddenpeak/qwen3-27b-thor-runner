// mel_gpu.cu — GPU Mel Spectrogram (cuFFT + CUDA kernels)
//
// CPU O(N²) DFT → GPU cuFFT O(N log N) + CUDA mel filterbank。
// asrTest2.mp3 (60min) baseline: Phase 3 compute_mel_80 在 CPU 上约 60-100s
// GPU 版本全部帧 batch 一次 cuFFT + SGEMM = <100ms 预期。

#include "mel_gpu.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace qwen_thor {
namespace asr {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Pre-emphasis + scale: out[i] = scale * (pcm[i] - coeff * pcm[i-1])
__global__ void preemphasis_kernel(const float* __restrict__ pcm,
                                    float* __restrict__ out,
                                    int num_samples, float coeff, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_samples) return;
    float prev = (i > 0) ? pcm[i - 1] : 0.0f;
    float val = pcm[i] - coeff * prev;
    // 首样本: 匹配 CPU compute_mel_80 的 scaled[0] *= (1-coeff) 行为
    if (i == 0) val = pcm[0] * (1.0f - coeff);
    out[i] = scale * val;
}

// Frame extraction + windowing + zero-pad
// 从预处理后 PCM 提取帧, 乘窗函数, 写入 [num_frames, fft_size] (zero-padded)
__global__ void frame_window_kernel(const float* __restrict__ pcm,
                                     const float* __restrict__ window,
                                     float* __restrict__ frames,
                                     int num_frames, int n_fft, int hop, int fft_size) {
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    if (frame_idx >= num_frames) return;

    float* out = frames + frame_idx * fft_size;
    int pcm_offset = frame_idx * hop;

    if (sample_idx < n_fft) {
        out[sample_idx] = pcm[pcm_offset + sample_idx] * window[sample_idx];
    }
    // Zero-pad: fft_size > n_fft 的部分
    if (sample_idx >= n_fft && sample_idx < fft_size) {
        out[sample_idx] = 0.0f;
    }
}

// 大帧数版本: 使用 2D grid 处理帧数 > max blocks
__global__ void frame_window_kernel_2d(const float* __restrict__ pcm,
                                        const float* __restrict__ window,
                                        float* __restrict__ frames,
                                        int num_frames, int n_fft, int hop, int fft_size) {
    int frame_idx = blockIdx.x;
    int sample_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (frame_idx >= num_frames || sample_idx >= fft_size) return;

    float* out = frames + frame_idx * fft_size;
    int pcm_offset = frame_idx * hop;

    if (sample_idx < n_fft) {
        out[sample_idx] = pcm[pcm_offset + sample_idx] * window[sample_idx];
    } else {
        out[sample_idx] = 0.0f;
    }
}

// 复数功率谱: power[k] = re² + im²
__global__ void power_spectrum_kernel(const cufftComplex* __restrict__ fft_out,
                                       float* __restrict__ power,
                                       int num_frames, int n_freqs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * n_freqs;
    if (idx >= total) return;
    float re = fft_out[idx].x;
    float im = fft_out[idx].y;
    power[idx] = re * re + im * im;
}

// Log mel: 对 mel filterbank 输出取 log (in-place)
__global__ void log_mel_kernel(float* __restrict__ mel, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    mel[idx] = logf(fmaxf(mel[idx], 1.175494e-38f));
}

// Transpose: [n_mels, T] → [T, n_mels]
__global__ void transpose_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[idx];
}

// ============================================================================
// GpuMelExtractor
// ============================================================================

GpuMelExtractor::GpuMelExtractor() = default;

GpuMelExtractor::~GpuMelExtractor() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(d_window_);
    cudaFree(d_mel_fb_);
    cudaFree(d_pcm_);
    cudaFree(d_frames_);
    cudaFree(d_fft_out_);
    cudaFree(d_power_);
    cudaFree(d_mel_);
    cudaFree(d_mel_col_);
}

bool GpuMelExtractor::init(const GpuMelConfig& config) {
    cfg_ = config;

    // FFT size: pad to next power of 2 if configured (Kaldi default)
    if (cfg_.pad_to_power_of_two) {
        fft_size_ = 1;
        while (fft_size_ < cfg_.n_fft) fft_size_ <<= 1;
    } else {
        fft_size_ = cfg_.n_fft;
    }
    n_freqs_ = fft_size_ / 2 + 1;

    // Create CUDA stream + cuBLAS
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    // Build constant buffers
    build_window();
    build_mel_filterbank();

    initialized_ = true;
    return true;
}

void GpuMelExtractor::build_window() {
    std::vector<float> win(cfg_.n_fft);
    if (cfg_.window == GpuMelConfig::WindowType::POVEY) {
        // Povey = symmetric Hann^0.85 (Kaldi default for speaker verification)
        for (int i = 0; i < cfg_.n_fft; ++i)
            win[i] = std::pow(0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (cfg_.n_fft - 1)), 0.85f);
    } else if (cfg_.window == GpuMelConfig::WindowType::HANN) {
        for (int i = 0; i < cfg_.n_fft; ++i)
            win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / cfg_.n_fft));
    } else {
        for (int i = 0; i < cfg_.n_fft; ++i)
            win[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (cfg_.n_fft - 1));
    }
    cudaMalloc(&d_window_, cfg_.n_fft * sizeof(float));
    cudaMemcpy(d_window_, win.data(), cfg_.n_fft * sizeof(float), cudaMemcpyHostToDevice);
}

void GpuMelExtractor::build_mel_filterbank() {
    // HTK mel filterbank (matches compute_mel_80 exactly)
    std::vector<float> fb(cfg_.n_mels * n_freqs_, 0.0f);
    auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    float min_mel = hz_to_mel(cfg_.low_freq);
    float max_mel = hz_to_mel((float)cfg_.sample_rate / 2.0f);
    std::vector<float> mel_points(cfg_.n_mels + 2);
    for (int i = 0; i < cfg_.n_mels + 2; ++i)
        mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (cfg_.n_mels + 1));

    for (int m = 0; m < cfg_.n_mels; ++m) {
        // Use fft_size_ (padded) for frequency bin resolution, not n_fft
        float left   = mel_points[m] * fft_size_ / cfg_.sample_rate;
        float center = mel_points[m + 1] * fft_size_ / cfg_.sample_rate;
        float right  = mel_points[m + 2] * fft_size_ / cfg_.sample_rate;
        for (int k = 0; k < n_freqs_; ++k) {
            float fk = (float)k;
            if (fk >= left && fk <= center)
                fb[m * n_freqs_ + k] = (fk - left) / (center - left);
            else if (fk > center && fk <= right)
                fb[m * n_freqs_ + k] = (right - fk) / (right - center);
        }
    }

    cudaMalloc(&d_mel_fb_, cfg_.n_mels * n_freqs_ * sizeof(float));
    cudaMemcpy(d_mel_fb_, fb.data(), cfg_.n_mels * n_freqs_ * sizeof(float), cudaMemcpyHostToDevice);
}

bool GpuMelExtractor::ensure_buffers(int num_samples, int num_frames) {
    bool need_realloc = false;

    if (num_samples > buf_max_samples_) {
        cudaFree(d_pcm_);
        buf_max_samples_ = num_samples + num_samples / 4;  // 25% headroom
        cudaMalloc(&d_pcm_, buf_max_samples_ * sizeof(float));
        need_realloc = true;
    }

    if (num_frames > buf_max_frames_) {
        cudaFree(d_frames_);
        cudaFree(d_fft_out_);
        cudaFree(d_power_);
        cudaFree(d_mel_);
        cudaFree(d_mel_col_);

        buf_max_frames_ = num_frames + num_frames / 4;  // 25% headroom
        cudaMalloc(&d_frames_, (size_t)buf_max_frames_ * fft_size_ * sizeof(float));
        cudaMalloc(&d_fft_out_, (size_t)buf_max_frames_ * n_freqs_ * sizeof(cufftComplex));
        cudaMalloc(&d_power_, (size_t)buf_max_frames_ * n_freqs_ * sizeof(float));
        cudaMalloc(&d_mel_col_, (size_t)cfg_.n_mels * buf_max_frames_ * sizeof(float));
        cudaMalloc(&d_mel_, (size_t)buf_max_frames_ * cfg_.n_mels * sizeof(float));

        // Recreate cuFFT plan for new batch size
        if (fft_plan_) cufftDestroy(fft_plan_);
        cufftPlan1d(&fft_plan_, fft_size_, CUFFT_R2C, buf_max_frames_);
        cufftSetStream(fft_plan_, stream_);
        max_frames_ = buf_max_frames_;
        need_realloc = true;
    }

    // If num_frames changed but still fits, need new plan only if batch size is smaller
    // cuFFT plan with max frames works for fewer frames too (just uses subset of output)
    // Actually, cuFFT batch plan = fixed batch size. Need to recreate for exact batch.
    if (num_frames != max_frames_ || need_realloc) {
        if (fft_plan_) cufftDestroy(fft_plan_);
        cufftPlan1d(&fft_plan_, fft_size_, CUFFT_R2C, num_frames);
        cufftSetStream(fft_plan_, stream_);
        max_frames_ = num_frames;
    }

    return true;
}

GpuMelExtractor::GpuMelResult GpuMelExtractor::compute_gpu(const float* pcm, int num_samples) {
    GpuMelResult result;
    if (!initialized_ || num_samples < cfg_.n_fft) return result;

    int num_frames = (num_samples - cfg_.n_fft) / cfg_.hop + 1;
    if (num_frames <= 0) return result;

    ensure_buffers(num_samples, num_frames);

    const int BLOCK = 256;

    // 1. Upload PCM to GPU
    cudaMemcpyAsync(d_pcm_, pcm, num_samples * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // 2. Pre-emphasis + scale on GPU
    float scale = cfg_.scale_32768 ? 32768.0f : 1.0f;
    int grid_samples = (num_samples + BLOCK - 1) / BLOCK;
    preemphasis_kernel<<<grid_samples, BLOCK, 0, stream_>>>(
        d_pcm_, d_pcm_, num_samples, cfg_.pre_emphasis, scale);

    // 3. Frame extraction + windowing + zero-pad
    if (fft_size_ <= 1024) {
        // Simple version: one block per frame, threads handle samples
        frame_window_kernel<<<num_frames, fft_size_, 0, stream_>>>(
            d_pcm_, d_window_, d_frames_, num_frames, cfg_.n_fft, cfg_.hop, fft_size_);
    } else {
        // For larger FFT sizes, use 2D grid
        dim3 grid2d(num_frames, (fft_size_ + BLOCK - 1) / BLOCK);
        frame_window_kernel_2d<<<grid2d, BLOCK, 0, stream_>>>(
            d_pcm_, d_window_, d_frames_, num_frames, cfg_.n_fft, cfg_.hop, fft_size_);
    }

    // 4. Batched cuFFT R2C
    cufftExecR2C(fft_plan_, d_frames_, d_fft_out_);

    // 5. Power spectrum
    int total_freq = num_frames * n_freqs_;
    int grid_freq = (total_freq + BLOCK - 1) / BLOCK;
    power_spectrum_kernel<<<grid_freq, BLOCK, 0, stream_>>>(
        d_fft_out_, d_power_, num_frames, n_freqs_);

    // 6. Mel filterbank: mel_col[n_mels, T] = mel_fb[n_mels, n_freqs] × power[n_freqs, T]
    //    power 布局: [T, n_freqs] row-major = [n_freqs, T] col-major (for cuBLAS)
    //    mel_fb 布局: [n_mels, n_freqs] row-major
    //    结果: [n_mels, T] col-major → which is [T, n_mels] row-major transposed
    {
        float alpha = 1.0f, beta = 0.0f;
        // cuBLAS column-major:
        // C[n_mels, T] = A[n_mels, n_freqs] × B[n_freqs, T]
        // A = mel_fb, stored row-major [n_mels, n_freqs] = col-major [n_freqs, n_mels]^T
        // B = power, stored row-major [T, n_freqs] = col-major [n_freqs, T]
        // C = mel_col, col-major [n_mels, T]
        //
        // cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        // We want C = mel_fb × power^T? No.
        //
        // power is [T, n_freqs] row-major → in col-major it's [n_freqs, T] matrix with ldb=n_freqs
        // mel_fb is [n_mels, n_freqs] row-major → in col-major it's [n_freqs, n_mels] with lda=n_freqs
        // We want mel_col [n_mels, T] = mel_fb × power^T
        //   i.e. C[m, t] = Σ_k mel_fb[m, k] × power[t, k]
        //
        // In col-major: C[n_mels, T]  = A^T [n_mels, n_freqs] × B [n_freqs, T]
        //   A = mel_fb as col-major [n_freqs, n_mels], lda=n_freqs
        //   B = power as col-major [n_freqs, T], ldb=n_freqs
        //   C = mel_col, ldc=n_mels
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    cfg_.n_mels, num_frames, n_freqs_,
                    &alpha,
                    d_mel_fb_, n_freqs_,     // A: [n_freqs, n_mels] col-major
                    d_power_, n_freqs_,      // B: [n_freqs, T] col-major
                    &beta,
                    d_mel_col_, cfg_.n_mels); // C: [n_mels, T] col-major
    }

    // 7. Log mel
    int total_mel = num_frames * cfg_.n_mels;
    int grid_mel = (total_mel + BLOCK - 1) / BLOCK;
    log_mel_kernel<<<grid_mel, BLOCK, 0, stream_>>>(d_mel_col_, total_mel);

    // 8. Transpose [n_mels, T] col-major → [T, n_mels] row-major
    //    Actually d_mel_col_ is [n_mels, T] col-major = [T, n_mels] row-major (they're the same!)
    //    So d_mel_col_ is already [T * n_mels] with layout mel_col_[t * n_mels + m] for cuBLAS output

    // No wait. cuBLAS outputs in column-major: C[i,j] = C[i + j*ldc]
    // C is [n_mels × T] with ldc=n_mels
    // C[m, t] = d_mel_col_[m + t * n_mels]
    // This IS [T, n_mels] row-major (where row=t, col=m)!
    // So d_mel_col_ is already in [T, n_mels] row-major format. Perfect.

    result.d_mel = d_mel_col_;
    result.num_frames = num_frames;
    return result;
}

int GpuMelExtractor::compute(const float* pcm, int num_samples,
                              std::vector<float>& mel_out) {
    auto result = compute_gpu(pcm, num_samples);
    if (result.num_frames <= 0) {
        mel_out.clear();
        return 0;
    }

    // Sync and copy back
    cudaStreamSynchronize(stream_);
    int total = result.num_frames * cfg_.n_mels;
    mel_out.resize(total);
    cudaMemcpy(mel_out.data(), result.d_mel, total * sizeof(float), cudaMemcpyDeviceToHost);
    return result.num_frames;
}

} // namespace asr
} // namespace qwen_thor

// ============================================================================
// GpuWhisperMel — GPU Whisper mel spectrogram (128 channels)
// ============================================================================
namespace qwen_thor {
namespace asr {

// Whisper frame extraction: read from reflect-padded PCM, apply Hann window
__global__ void whisper_frame_kernel(const float* __restrict__ pcm,
                                      const float* __restrict__ window,
                                      float* __restrict__ frames,
                                      int num_frames, int n_fft, int hop) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_frames * n_fft;
    if (idx >= total) return;
    int t = idx / n_fft;
    int s = idx % n_fft;
    frames[idx] = pcm[t * hop + s] * window[s];
}

// Whisper log10 + normalization: log10(max(x, 1e-10)) → (x - max) / max(max, -8) + 4
// Phase 1: compute max across all elements (single block reduction)
__global__ void whisper_log10_kernel(float* __restrict__ mel, int total,
                                      float* __restrict__ d_max) {
    extern __shared__ float smem[];

    float local_max = -1e20f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        float v = log10f(fmaxf(mel[i], 1e-10f));
        mel[i] = v;
        local_max = fmaxf(local_max, v);
    }

    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) d_max[0] = smem[0];
}

// Phase 2: normalize with found max
__global__ void whisper_normalize_kernel(float* __restrict__ mel, int total,
                                          const float* __restrict__ d_max) {
    float max_val = d_max[0];
    float floor_val = max_val - 8.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        float v = fmaxf(mel[i], floor_val);
        mel[i] = (v + 4.0f) / 4.0f;
    }
}

GpuWhisperMel::GpuWhisperMel() = default;

GpuWhisperMel::~GpuWhisperMel() {
    if (fft_plan_) cufftDestroy(fft_plan_);
    if (cublas_) cublasDestroy(cublas_);
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(d_window_);
    cudaFree(d_mel_fb_);
    cudaFree(d_pcm_);
    cudaFree(d_frames_);
    cudaFree(d_fft_);
    cudaFree(d_power_);
    cudaFree(d_mel_out_);
}

bool GpuWhisperMel::init(const float* mel_fb_128x201) {
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_);
    cublasSetStream(cublas_, stream_);

    // Hann window (periodic: divide by N, not N-1)
    std::vector<float> win(N_FFT);
    for (int i = 0; i < N_FFT; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / N_FFT));
    cudaMalloc(&d_window_, N_FFT * sizeof(float));
    cudaMemcpy(d_window_, win.data(), N_FFT * sizeof(float), cudaMemcpyHostToDevice);

    // Mel filterbank: upload caller's precomputed [128, 201]
    cudaMalloc(&d_mel_fb_, N_MELS * N_FREQS * sizeof(float));
    cudaMemcpy(d_mel_fb_, mel_fb_128x201, N_MELS * N_FREQS * sizeof(float), cudaMemcpyHostToDevice);

    initialized_ = true;
    fprintf(stderr, "[GpuWhisperMel] Initialized: %d mels, FFT %d, hop %d\n", N_MELS, N_FFT, HOP);
    return true;
}

bool GpuWhisperMel::ensure_buffers(int padded_samples, int num_frames) {
    if (padded_samples > buf_max_samples_) {
        cudaFree(d_pcm_);
        buf_max_samples_ = padded_samples + padded_samples / 4;
        cudaMalloc(&d_pcm_, buf_max_samples_ * sizeof(float));
    }
    if (num_frames > buf_max_frames_) {
        cudaFree(d_frames_);
        cudaFree(d_fft_);
        cudaFree(d_power_);
        cudaFree(d_mel_out_);
        buf_max_frames_ = num_frames + num_frames / 4;
        cudaMalloc(&d_frames_, (size_t)buf_max_frames_ * N_FFT * sizeof(float));
        cudaMalloc(&d_fft_, (size_t)buf_max_frames_ * N_FREQS * sizeof(cufftComplex));
        cudaMalloc(&d_power_, (size_t)buf_max_frames_ * N_FREQS * sizeof(float));
        // +1 extra float for d_max scratch
        cudaMalloc(&d_mel_out_, ((size_t)N_MELS * buf_max_frames_ + 1) * sizeof(float));
    }
    if (num_frames != cur_plan_frames_) {
        if (fft_plan_) cufftDestroy(fft_plan_);
        cufftPlan1d(&fft_plan_, N_FFT, CUFFT_R2C, num_frames);
        cufftSetStream(fft_plan_, stream_);
        cur_plan_frames_ = num_frames;
    }
    return true;
}

GpuWhisperMel::Result GpuWhisperMel::compute(const float* pcm, int num_samples) {
    Result result{nullptr, 0};
    if (!initialized_ || num_samples < N_FFT) return result;

    // 1. Reflect-pad on CPU (add N_FFT/2 = 200 on each side)
    int pad = N_FFT / 2;  // 200
    int padded_len = num_samples + 2 * pad;
    std::vector<float> padded(padded_len);
    for (int i = 0; i < padded_len; i++) {
        int src = i - pad;
        if (src < 0) src = -src;
        else if (src >= num_samples) src = 2 * num_samples - src - 2;
        padded[i] = (src >= 0 && src < num_samples) ? pcm[src] : 0.0f;
    }

    // Compute frames (before dropping last)
    int num_frames = (padded_len - N_FFT) / HOP + 1;
    // Whisper drops last frame
    if (num_frames > 1) num_frames--;
    if (num_frames <= 0) return result;

    ensure_buffers(padded_len, num_frames);

    const int BLOCK = 256;

    // 2. Upload padded PCM
    cudaMemcpyAsync(d_pcm_, padded.data(), padded_len * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    // 3. Frame extraction + windowing
    int total_frame_elems = num_frames * N_FFT;
    int grid = (total_frame_elems + BLOCK - 1) / BLOCK;
    whisper_frame_kernel<<<grid, BLOCK, 0, stream_>>>(
        d_pcm_, d_window_, d_frames_, num_frames, N_FFT, HOP);

    // 4. Batched cuFFT R2C (n=400, batch=num_frames)
    cufftExecR2C(fft_plan_, d_frames_, d_fft_);

    // 5. Power spectrum
    int total_freq = num_frames * N_FREQS;
    power_spectrum_kernel<<<(total_freq + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_fft_, d_power_, num_frames, N_FREQS);

    // 6. Mel filterbank: mel_out[N_MELS, T] = mel_fb[N_MELS, N_FREQS] × power_T[N_FREQS, T]
    //    power is [T, N_FREQS] row-major = [N_FREQS, T] col-major
    //    mel_fb is [N_MELS, N_FREQS] row-major = [N_FREQS, N_MELS] col-major
    //    mel_out should be [N_MELS, T] row-major = [T, N_MELS] col-major
    //    But we want [N_MELS, T] row-major. cuBLAS gives col-major.
    //    C_colmaj[m, t] = data[m + t*N_MELS] → this is [T, N_MELS] row-major
    //    We need data[m*T + t] = [N_MELS, T] row-major
    //    Solution: swap A and B, get C = [T, N_MELS] col-major = [N_MELS, T] row-major
    {
        float alpha = 1.0f, beta = 0.0f;
        // C[T, N_MELS] col-major = B^T[T, N_FREQS] × A[N_FREQS, N_MELS]
        // = power[T, N_FREQS]row × mel_fb^T[N_FREQS, N_MELS]
        // In cuBLAS: C[m=T, n=N_MELS, k=N_FREQS] = A(N_FREQS, T) × B(N_FREQS, N_MELS)^T ? No...
        //
        // Simpler: compute result as [N_MELS, T] col-major (standard), then transpose
        // C[N_MELS, T] colmaj = mel_fb^T [N_MELS, N_FREQS] × power [N_FREQS, T]
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    N_MELS, num_frames, N_FREQS,
                    &alpha,
                    d_mel_fb_, N_FREQS,    // [N_FREQS, N_MELS] col-major (= [N_MELS, N_FREQS] row-major)
                    d_power_, N_FREQS,     // [N_FREQS, T] col-major (= [T, N_FREQS] row-major)
                    &beta,
                    d_mel_out_, N_MELS);   // [N_MELS, T] col-major = [T, N_MELS] row-major
    }

    // Now d_mel_out_ is [T, N_MELS] row-major (column-major [N_MELS, T])
    // Transpose to [N_MELS, T] row-major
    // Use power buffer as temp (large enough: T*N_FREQS ≥ T*N_MELS for N_FREQS≥N_MELS)
    // Actually N_FREQS=201, N_MELS=128, so power buffer is [T, 201] ≥ [128, T]? Only if T≤201*T/128.
    // Just allocate properly: d_mel_out_ has N_MELS*T+1 floats. But I need a separate transpose target.
    // Use d_power_ as temp since it's T*201 and we need 128*T for the transposed result.
    // 128*T ≤ 201*T always. So d_power_ is big enough.
    transpose_kernel<<<(N_MELS * num_frames + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_mel_out_, d_power_, num_frames, N_MELS);
    // d_power_ now has [N_MELS, T] row-major
    // Copy back to d_mel_out_
    cudaMemcpyAsync(d_mel_out_, d_power_, (size_t)N_MELS * num_frames * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_);

    // 7. Log10 + Whisper normalization (2-phase: find max, then normalize)
    int total_mel = N_MELS * num_frames;
    float* d_max_scratch = d_mel_out_ + (size_t)N_MELS * buf_max_frames_;  // 1 float at end
    whisper_log10_kernel<<<1, 1024, 1024 * sizeof(float), stream_>>>(
        d_mel_out_, total_mel, d_max_scratch);
    whisper_normalize_kernel<<<(total_mel + BLOCK - 1) / BLOCK, BLOCK, 0, stream_>>>(
        d_mel_out_, total_mel, d_max_scratch);

    result.d_mel = d_mel_out_;
    result.num_frames = num_frames;
    return result;
}

} // namespace asr
} // namespace qwen_thor
