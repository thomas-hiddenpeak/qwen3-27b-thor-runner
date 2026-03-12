// audio_utils.h — 音频预处理工具
//
// CPU 端实现:
//   - WAV 文件读取 (16-bit PCM, 支持多声道→单声道)
//   - Mel spectrogram 计算 (Whisper 兼容: n_fft=400, hop=160, 128 mel bins)
//   - PCM 输出 (16-bit, 24kHz)
//
// 30s 音频 Mel 计算 <1ms (纯 CPU, 不需要 GPU)

#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace qwen_thor {
namespace audio {

// ============================================================================
// WAV 文件读取
// ============================================================================

struct AudioData {
    std::vector<float> samples;   // 归一化到 [-1, 1]
    int sample_rate = 0;
    int channels = 0;
};

// 读取 WAV 文件, 输出单声道浮点 PCM
// 支持 16-bit PCM WAV, 自动混音为单声道
bool load_wav(const std::string& path, AudioData& out);

// 从内存读取 WAV (用于 HTTP API 接收的音频数据)
bool load_wav_from_memory(const uint8_t* data, size_t size, AudioData& out);

// 简单重采样 (线性插值, 无需高质量)
// 用于 non-16kHz 输入重采样到 16kHz
void resample(const std::vector<float>& in, int in_sr,
              std::vector<float>& out, int out_sr);

// ============================================================================
// Mel Spectrogram (Whisper 兼容)
// ============================================================================

struct MelConfig {
    int n_fft       = 400;
    int hop_length  = 160;
    int n_mels      = 128;
    int sample_rate = 16000;
};

// 计算 Log-Mel spectrogram
// 输入: PCM samples (16kHz, float, [-1, 1])
// 输出: mel_out [n_mels, num_frames], row-major
//       num_frames = (num_samples - n_fft) / hop_length + 1 (如不足 n_fft 则 padding)
//
// Whisper 风格归一化:
//   1. STFT: Hann window(n_fft), hop(hop_length)
//   2. Power spectrum: |FFT|²
//   3. Mel filterbank (n_mels bins, Slaney 归一化)
//   4. Log-mel: log10(max(val, 1e-10))
//   5. Normalize: (mel - max_val) / max(max_val, -8.0) + 4.0, clamp [0, ∞)
void compute_mel(const float* samples, int num_samples,
                 const MelConfig& config,
                 std::vector<float>& mel_out,
                 int& num_frames);

// ============================================================================
// PCM 输出 (用于 TTS)
// ============================================================================

// 写 WAV 文件 (单声道, 16-bit PCM)
bool write_wav(const std::string& path, const int16_t* samples, int num_samples, int sample_rate);

// float PCM [-1,1] → int16 PCM
void float_to_int16(const float* in, int16_t* out, int num_samples);

} // namespace audio
} // namespace qwen_thor
