// vad_gpu.h — GPU-Accelerated FSMN-VAD
//
// CPU FSMN-VAD 的 GPU 加速版本。
// 将所有计算密集的操作迁移到 GPU:
//   - Fbank 提取: cuFFT batched R2C (全部帧一次完成)
//   - LFR 拼接 + CMVN: GPU kernel
//   - FSMN forward: 所有线性层 batch 化为 cuBLAS SGEMM, 因果卷积为 GPU kernel
// 状态机判决保留在 CPU (简单逻辑, 无需 GPU)。
//
// 原 CPU 版本: 3615s 音频 → 361K 帧逐帧处理, ~80-100s
// GPU 版本: batch GEMM + cuFFT = 预期 <1s

#pragma once

#include "vad_config.h"
#include <vector>
#include <string>
#include <cublas_v2.h>
#include <cufft.h>
#include <cuda_runtime.h>

namespace qwen_thor {
namespace asr {

// VAD 检测到的语音段
struct GpuVadSegment {
    int start_ms = 0;
    int end_ms   = 0;
    // 不拷贝 PCM: 调用者根据 start/end 从原始 PCM 切片
};

class GpuVadEngine {
public:
    GpuVadEngine();
    ~GpuVadEngine();

    // 加载模型权重到 GPU + CMVN
    bool load(const std::string& model_dir);
    bool is_loaded() const { return loaded_; }

    // 整段检测 (非流式): 全部帧一次性 GPU 推理
    // 返回语音段列表 (不含 PCM 数据, 仅时间戳)
    std::vector<GpuVadSegment> detect_all(const float* pcm, int num_samples,
                                           int max_silence_ms = 300,
                                           int max_segment_ms = 8000);

    VadConfig& mutable_config() { return config_; }
    const VadConfig& config() const { return config_; }

private:
    VadConfig config_;
    bool loaded_ = false;

    // cuBLAS + stream
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // cuFFT for Fbank
    cufftHandle fft_plan_ = 0;
    int fft_size_ = 0;
    int fft_plan_batch_ = 0;  // batch size of current plan

    // GPU weights (预分配, 推理时不 malloc)
    // in_linear1: [input_dim=400, 140] + bias [140]
    float* d_in1_w_ = nullptr;
    float* d_in1_b_ = nullptr;
    // in_linear2: [140, 250] + bias [250]
    float* d_in2_w_ = nullptr;
    float* d_in2_b_ = nullptr;
    // FSMN layers × 4
    struct GpuFsmnWeights {
        float* d_linear_w = nullptr;   // [linear_dim, proj_dim] = [250, 128]
        float* d_fsmn_w = nullptr;     // [proj_dim, lorder] = [128, 20]
        float* d_affine_w = nullptr;   // [proj_dim, linear_dim] = [128, 250]
        float* d_affine_b = nullptr;   // [linear_dim] = [250]
    };
    GpuFsmnWeights fsmn_weights_[4];
    // out_linear1: [250, 140] + bias [140]
    float* d_out1_w_ = nullptr;
    float* d_out1_b_ = nullptr;
    // out_linear2: [140, 248] + bias [248]
    float* d_out2_w_ = nullptr;
    float* d_out2_b_ = nullptr;
    // CMVN
    float* d_cmvn_mean_ = nullptr;    // [input_dim]
    float* d_cmvn_invstd_ = nullptr;  // [input_dim]
    // Fbank
    float* d_window_ = nullptr;       // [n_fft=400]
    float* d_mel_fb_ = nullptr;       // [n_mels=80, n_freqs]

    // GPU scratch buffers (按需增长)
    float* d_pcm_ = nullptr;
    float* d_frames_ = nullptr;
    cufftComplex* d_fft_out_ = nullptr;
    float* d_fbank_ = nullptr;         // [num_fbank_frames, n_mels]
    float* d_lfr_ = nullptr;           // [num_lfr_frames, input_dim]
    float* d_h_ = nullptr;             // [num_lfr_frames, linear_dim]
    float* d_tmp_ = nullptr;           // [num_lfr_frames, max(linear_dim, proj_dim, output_dim)]
    float* d_probs_ = nullptr;         // [num_lfr_frames] speech probabilities

    int scratch_max_fbank_ = 0;
    int scratch_max_lfr_ = 0;

    bool ensure_scratch(int num_fbank_frames, int num_lfr_frames);
    void build_fbank_constants();

    // Forward: 全部帧 batch GPU 推理, 返回 speech probabilities
    std::vector<float> forward_batch(const float* pcm, int num_samples,
                                      int& num_lfr_frames);

    // State machine: 从 speech probabilities → segments
    std::vector<GpuVadSegment> run_state_machine(const std::vector<float>& probs,
                                                  int max_silence_ms,
                                                  int max_segment_ms);
};

} // namespace asr
} // namespace qwen_thor
