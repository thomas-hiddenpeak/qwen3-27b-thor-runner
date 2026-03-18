// mel_gpu.h — GPU Mel Spectrogram (cuFFT + CUDA kernels)
//
// 替代 CPU 上的 O(N²) DFT 实现, 使用 cuFFT batched R2C FFT。
// 用于 CAM++ speaker diarization (compute_mel_80) 和可选的 VAD Fbank。
//
// 性能:
//   CPU DFT: 201 × 400 = 80,400 MACs/frame, O(N²)
//   GPU cuFFT: O(N log N), batch 全部帧一次完成
//   Mel filterbank: 矩阵乘法 → cuBLAS SGEMM
//   预期: 50-200× 加速 (单段), 更多加速来自 batch 多段

#pragma once

#include <vector>
#include <cufft.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace qwen_thor {
namespace asr {

// GPU Mel Spectrogram 配置
struct GpuMelConfig {
    int n_fft       = 400;    // Window size (25ms @ 16kHz)
    int hop         = 160;    // Hop size (10ms @ 16kHz)
    int n_mels      = 80;
    int sample_rate = 16000;
    float pre_emphasis = 0.97f;   // Pre-emphasis coefficient (0 = disabled)
    bool scale_32768   = true;    // FunASR convention: float [-1,1] → PCM16 range
    float low_freq     = 20.0f;   // Kaldi default low frequency for mel filterbank
    bool pad_to_power_of_two = true;  // Kaldi default: zero-pad to power of 2 for FFT
    enum class WindowType { HANN, HAMMING, POVEY } window = WindowType::POVEY;
};

class GpuMelExtractor {
public:
    GpuMelExtractor();
    ~GpuMelExtractor();

    // 初始化: 创建 cuFFT plan, mel filterbank, window 等
    bool init(const GpuMelConfig& config = GpuMelConfig{});
    bool is_initialized() const { return initialized_; }

    // 从 CPU PCM 提取 mel spectrogram, 结果写回 CPU
    // mel_out: [T, n_mels] row-major (T = num_frames)
    // 返回 num_frames
    int compute(const float* pcm, int num_samples,
                std::vector<float>& mel_out);

    // 从 CPU PCM 提取 mel spectrogram, 结果保留在 GPU
    // 返回 {d_mel 指针, num_frames}; d_mel 布局 [T, n_mels] row-major
    // 调用者不 own d_mel (内部 buffer, 下次 compute 会覆盖)
    struct GpuMelResult {
        float* d_mel = nullptr;   // GPU 指针, [T, n_mels]
        int num_frames = 0;
    };
    GpuMelResult compute_gpu(const float* pcm, int num_samples);

    const GpuMelConfig& config() const { return cfg_; }

private:
    GpuMelConfig cfg_;
    bool initialized_ = false;

    // FFT
    int fft_size_ = 0;       // n_fft padded to power of 2 (400 → 512)
    int n_freqs_ = 0;        // fft_size/2 + 1 = 257
    cufftHandle fft_plan_ = 0;
    int max_frames_ = 0;     // 当前 cuFFT plan 的最大帧数

    // cuBLAS
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // GPU 常量 buffers (预分配, 不变)
    float* d_window_ = nullptr;       // [n_fft]
    float* d_mel_fb_ = nullptr;       // [n_mels, n_freqs] mel filterbank

    // GPU 工作 buffers (按需增长)
    float* d_pcm_ = nullptr;          // [max_samples]
    float* d_frames_ = nullptr;       // [max_frames, fft_size] (windowed, zero-padded)
    cufftComplex* d_fft_out_ = nullptr;  // [max_frames, n_freqs]
    float* d_power_ = nullptr;        // [max_frames, n_freqs]
    float* d_mel_ = nullptr;          // [max_frames, n_mels] (transposed output)
    float* d_mel_col_ = nullptr;      // [n_mels, max_frames] (column-major for GEMM)
    int buf_max_samples_ = 0;
    int buf_max_frames_ = 0;

    bool ensure_buffers(int num_samples, int num_frames);
    void build_mel_filterbank();
    void build_window();
};

} // namespace asr
} // namespace qwen_thor
