// asr_engine.h — Qwen3-ASR 推理引擎
//
// 顶层 ASR 引擎:
//   1. 加载权重 (SafetensorsLoader, cudaMalloc)
//   2. 编排 AudioEncoder + TextDecoder
//   3. 提供 transcribe() 接口: PCM → 文字
//
// 用法:
//   ASREngine engine;
//   engine.load_model("/path/to/Qwen3-ASR-1.7B");
//   std::string text = engine.transcribe(pcm_samples, num_samples, sample_rate);

#pragma once

#include "asr_config.h"
#include "asr_encoder.h"
#include "asr_decoder.h"
#include "engine/tokenizer.h"
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace asr {

class GpuWhisperMel;  // forward declaration (mel_gpu.h)

class ASREngine {
public:
    ASREngine();
    ~ASREngine();

    // 加载模型: model_dir 需含 config.json + safetensors + tokenizer
    void load_model(const std::string& model_dir);

    // 转录: PCM 浮点 → 文本
    // samples: 单声道 float32, sample_rate: 采样率 (自动重采样到 16kHz)
    // suppress_early_eos: 根据音频时长抑制过早 EOS, 防止停顿导致截断
    std::string transcribe(const float* samples, int num_samples,
                           int sample_rate = 16000,
                           float temperature = 0.0f,
                           int max_new_tokens = 448,
                           bool suppress_early_eos = false);

    // 转录: WAV 文件路径 → 文本
    std::string transcribe_file(const std::string& wav_path,
                                float temperature = 0.0f,
                                int max_new_tokens = 448,
                                bool suppress_early_eos = false);

    // 批量转录: 多个 PCM 段同时解码 (GEMV→GEMM, B× 吞吐)
    // 返回与 chunks 等长的文本 vector
    struct AudioChunk { const float* samples; int num_samples; };
    std::vector<std::string> transcribe_batch(
        const std::vector<AudioChunk>& chunks,
        int sample_rate = 16000,
        bool suppress_early_eos = false);

    bool is_loaded() const { return loaded_; }
    const ASRConfig& config() const { return config_; }

    // 设置 repetition penalty (> 1.0 抑制重复, 1.0 = 无效果)
    void set_repetition_penalty(float p) { repetition_penalty_ = p; }
    float get_repetition_penalty() const { return repetition_penalty_; }

private:
    ASRConfig config_;
    std::unique_ptr<AudioEncoder> encoder_;
    std::unique_ptr<TextDecoder> decoder_;

    // BPE tokenizer (复用现有 Tokenizer)
    Tokenizer tokenizer_;
    std::string model_dir_;

    // GPU buffers
    __nv_bfloat16* mel_gpu_ = nullptr;       // [128, max_mel_frames]
    __nv_bfloat16* encoder_out_ = nullptr;    // [max_tokens, 2048]
    __nv_bfloat16* input_embeds_ = nullptr;   // [max_prompt_len, 2048]
    __nv_bfloat16* logits_ = nullptr;         // [vocab_size]
    int* position_ids_ = nullptr;             // [3, max_seq_len]
    int* token_id_gpu_ = nullptr;             // [1]
    int* prompt_tokens_gpu_ = nullptr;        // [max_prompt_len], pre-allocated
    float* mel_staging_gpu_ = nullptr;        // [128, max_mel_frames] F32 staging for GPU conversion

    // Repetition penalty
    float repetition_penalty_ = 1.0f;
    int* rep_tokens_gpu_ = nullptr;            // [max_new_tokens] output token history for penalty

    // Embed weight pointer (shared with decoder, not owned separately)
    __nv_bfloat16* embed_tokens_w_ = nullptr;

    // 权重所有权
    std::vector<void*> device_weights_;

    // 容量
    int max_mel_frames_ = 0;
    int max_prompt_len_ = 0;

    // 缓存的 mel filterbank 和 Hann window (CPU fallback)
    std::vector<float> cached_mel_fb_;     // [n_mels * n_freqs]
    std::vector<float> cached_hann_window_; // [n_fft]
    int cached_n_fft_ = 0;
    int cached_n_mels_ = 0;
    int cached_sample_rate_ = 0;

    // GPU Whisper mel (cuFFT-accelerated)
    std::unique_ptr<GpuWhisperMel> gpu_whisper_mel_;

    // Batch decode buffers (allocated on first batch transcribe)
    __nv_bfloat16* batch_logits_ = nullptr;    // [max_batch, vocab_size]
    int* batch_token_ids_ = nullptr;           // [max_batch] on GPU
    int* batch_position_ids_ = nullptr;        // [3, max_batch] on GPU
    int* batch_result_ids_ = nullptr;          // [max_batch] on GPU (argmax results)
    int max_batch_allocated_ = 0;

    cudaStream_t stream_ = 0;
    bool loaded_ = false;

    // 内部方法
    void load_weights(const std::string& model_dir);
    void build_prompt(int encoder_out_len, std::vector<int>& token_ids,
                      const std::string& language = "Chinese");
    void init_mel_cache();  // 预计算 mel filterbank + Hann window
};

} // namespace asr
} // namespace qwen_thor
