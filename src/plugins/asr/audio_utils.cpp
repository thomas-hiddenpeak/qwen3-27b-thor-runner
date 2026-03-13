// audio_utils.cpp — 音频预处理工具实现
//
// Mel spectrogram 参考 whisper.cpp 实现, Whisper 兼容

#include "audio_utils.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen_thor {
namespace audio {

// ============================================================================
// WAV 读取
// ============================================================================

// Minimal WAV parser (RIFF/WAVE PCM16 only)
struct WavHeader {
    char riff[4];
    uint32_t file_size;
    char wave[4];
};

bool load_wav(const std::string& path, AudioData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[audio] Cannot open: " << path << std::endl;
        return false;
    }

    // Read entire file
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(file_size);
    f.read(reinterpret_cast<char*>(buf.data()), file_size);

    return load_wav_from_memory(buf.data(), buf.size(), out);
}

bool load_wav_from_memory(const uint8_t* data, size_t size, AudioData& out) {
    if (size < 44) return false;

    // Verify RIFF header
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        std::cerr << "[audio] Not a WAV file" << std::endl;
        return false;
    }

    // Find "fmt " chunk
    size_t pos = 12;
    int channels = 0, sample_rate = 0, bits_per_sample = 0;
    int audio_format = 0;
    const uint8_t* pcm_data = nullptr;
    size_t pcm_size = 0;

    while (pos + 8 <= size) {
        uint32_t chunk_size;
        std::memcpy(&chunk_size, data + pos + 4, 4);

        if (std::memcmp(data + pos, "fmt ", 4) == 0) {
            if (pos + 8 + 16 > size) return false;
            std::memcpy(&audio_format, data + pos + 8, 2);
            std::memcpy(&channels, data + pos + 10, 2);
            std::memcpy(&sample_rate, data + pos + 12, 4);
            std::memcpy(&bits_per_sample, data + pos + 22, 2);
        } else if (std::memcmp(data + pos, "data", 4) == 0) {
            pcm_data = data + pos + 8;
            pcm_size = chunk_size;
        }

        pos += 8 + chunk_size;
        if (chunk_size % 2 != 0) pos++;  // Word-align
    }

    if (audio_format != 1 || bits_per_sample != 16 || !pcm_data) {
        std::cerr << "[audio] Unsupported WAV format (need PCM16)" << std::endl;
        return false;
    }

    int num_samples = pcm_size / (channels * 2);
    out.sample_rate = sample_rate;
    out.channels = channels;
    out.samples.resize(num_samples);

    const int16_t* src = reinterpret_cast<const int16_t*>(pcm_data);
    for (int i = 0; i < num_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++) {
            sum += src[i * channels + c] / 32768.0f;
        }
        out.samples[i] = sum / channels;
    }

    return true;
}

void resample(const std::vector<float>& in, int in_sr,
              std::vector<float>& out, int out_sr) {
    if (in_sr == out_sr) {
        out = in;
        return;
    }

    double ratio = (double)out_sr / in_sr;
    int out_len = (int)(in.size() * ratio);
    out.resize(out_len);

    for (int i = 0; i < out_len; i++) {
        double src_idx = i / ratio;
        int idx0 = (int)src_idx;
        int idx1 = std::min(idx0 + 1, (int)in.size() - 1);
        double frac = src_idx - idx0;
        out[i] = (float)((1.0 - frac) * in[idx0] + frac * in[idx1]);
    }
}

// ============================================================================
// Mel Filterbank 构建 (Slaney 归一化, 与 Whisper/librosa 兼容)
// ============================================================================

namespace {

// Hz → Mel (HTK 公式)
inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

// Mel → Hz
inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Real-valued DFT (slow reference, sufficient for audio preprocessing)
void rdft(const float* x, int n, float* real_out, float* imag_out) {
    int half = n / 2 + 1;
    for (int k = 0; k < half; k++) {
        float re = 0, im = 0;
        for (int t = 0; t < n; t++) {
            float angle = 2.0f * (float)M_PI * k * t / n;
            re += x[t] * std::cos(angle);
            im -= x[t] * std::sin(angle);
        }
        real_out[k] = re;
        imag_out[k] = im;
    }
}

} // anonymous namespace

// ============================================================================
// 公开的 filterbank/window 构建 (供外部缓存)
// ============================================================================

std::vector<float> build_mel_filterbank(int n_mels, int n_fft, int sample_rate) {
    int n_freqs = n_fft / 2 + 1;
    std::vector<float> fb(n_mels * n_freqs, 0.0f);

    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel((float)sample_rate / 2.0f);

    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_mels + 1));
    }

    std::vector<float> fft_bins(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        fft_bins[i] = mel_points[i] * n_fft / sample_rate;
    }

    for (int m = 0; m < n_mels; m++) {
        float left = fft_bins[m];
        float center = fft_bins[m + 1];
        float right = fft_bins[m + 2];

        for (int k = 0; k < n_freqs; k++) {
            float fk = (float)k;
            if (fk >= left && fk <= center) {
                fb[m * n_freqs + k] = (fk - left) / (center - left);
            } else if (fk > center && fk <= right) {
                fb[m * n_freqs + k] = (right - fk) / (right - center);
            }
        }

        float enorm = 2.0f / (mel_points[m + 2] - mel_points[m]);
        for (int k = 0; k < n_freqs; k++) {
            fb[m * n_freqs + k] *= enorm;
        }
    }

    return fb;
}

std::vector<float> build_hann_window(int size) {
    std::vector<float> w(size);
    for (int i = 0; i < size; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / size));
    }
    return w;
}

// ============================================================================
// Mel Spectrogram 计算
// ============================================================================

void compute_mel(const float* samples, int num_samples,
                 const MelConfig& config,
                 std::vector<float>& mel_out,
                 int& num_frames) {
    auto mel_fb = build_mel_filterbank(config.n_mels, config.n_fft, config.sample_rate);
    auto window = build_hann_window(config.n_fft);
    compute_mel_cached(samples, num_samples, config, mel_fb, window, mel_out, num_frames);
}

void compute_mel_cached(const float* samples, int num_samples,
                        const MelConfig& config,
                        const std::vector<float>& mel_fb,
                        const std::vector<float>& window,
                        std::vector<float>& mel_out,
                        int& num_frames) {
    int n_fft = config.n_fft;
    int hop = config.hop_length;
    int n_mels = config.n_mels;
    int n_freqs = n_fft / 2 + 1;

    // Zero-pad input to ensure at least one frame
    int padded_len = std::max(num_samples, n_fft);
    std::vector<float> padded(padded_len, 0.0f);
    std::memcpy(padded.data(), samples, num_samples * sizeof(float));

    num_frames = (padded_len - n_fft) / hop + 1;

    // Compute STFT → power spectrum → mel
    std::vector<float> mel_spec(n_mels * num_frames, 0.0f);
    std::vector<float> frame(n_fft);
    std::vector<float> fft_real(n_freqs), fft_imag(n_freqs);

    for (int t = 0; t < num_frames; t++) {
        // Window input
        for (int i = 0; i < n_fft; i++) {
            frame[i] = padded[t * hop + i] * window[i];
        }

        // DFT
        rdft(frame.data(), n_fft, fft_real.data(), fft_imag.data());

        // Power spectrum and mel filterbank
        for (int m = 0; m < n_mels; m++) {
            float sum = 0;
            for (int k = 0; k < n_freqs; k++) {
                float power = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
                sum += mel_fb[m * n_freqs + k] * power;
            }
            mel_spec[m * num_frames + t] = sum;
        }
    }

    // Log-mel + Whisper normalization (matching WhisperFeatureExtractor)
    float max_val = -1e20f;
    for (auto& v : mel_spec) {
        v = std::log10(std::max(v, 1e-10f));
        max_val = std::max(max_val, v);
    }

    // Clamp floor to max_val - 8 (80 dB dynamic range), then normalize
    float floor_val = max_val - 8.0f;
    for (auto& v : mel_spec) {
        v = std::max(v, floor_val);
        v = (v + 4.0f) / 4.0f;
    }

    mel_out = std::move(mel_spec);
}

// ============================================================================
// PCM 输出
// ============================================================================

bool write_wav(const std::string& path, const int16_t* samples, int num_samples, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    int data_size = num_samples * 2;
    int file_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&file_size), 4);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    int fmt_size = 16;
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    int16_t audio_format = 1;
    int16_t n_channels = 1;
    int byte_rate = sample_rate * 2;
    int16_t block_align = 2;
    int16_t bits = 16;
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&n_channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits), 2);

    // data chunk
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    f.write(reinterpret_cast<const char*>(samples), data_size);

    return true;
}

void float_to_int16(const float* in, int16_t* out, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float v = std::max(-1.0f, std::min(1.0f, in[i]));
        out[i] = static_cast<int16_t>(v * 32767.0f);
    }
}

} // namespace audio
} // namespace qwen_thor
