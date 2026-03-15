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
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <spawn.h>
#include <fcntl.h>
#include <cerrno>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" { extern char** environ; }

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

bool load_audio_from_memory(const uint8_t* data, size_t size, AudioData& out,
                            const std::string& filename_hint) {
    // 先尝试 WAV 解析
    if (load_wav_from_memory(data, size, out)) {
        return true;
    }

    // WAV 解析失败, 通过 ffmpeg 转码为 PCM16 mono 16kHz
    // 使用临时文件 (M4A/MP4 等容器需要 seek, 管道不支持)
    fprintf(stderr, "[audio] Not WAV, trying ffmpeg transcode (%zu bytes)...\n", size);

    // 提取扩展名 (ffmpeg 对 m4a/mp4 等容器需要扩展名辅助探测)
    std::string suffix;
    if (!filename_hint.empty()) {
        auto dot = filename_hint.rfind('.');
        if (dot != std::string::npos) suffix = filename_hint.substr(dot);
    }

    // 写入临时文件 (带扩展名)
    std::string tmp_path;
    int tmp_fd;
    if (!suffix.empty()) {
        std::string tmpl = "/tmp/qwen_audio_XXXXXX" + suffix;
        std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
        tmpl_buf.push_back('\0');
        tmp_fd = mkstemps(tmpl_buf.data(), (int)suffix.size());
        if (tmp_fd >= 0) tmp_path = tmpl_buf.data();
    } else {
        char tmpl[] = "/tmp/qwen_audio_XXXXXX";
        tmp_fd = mkstemp(tmpl);
        if (tmp_fd >= 0) tmp_path = tmpl;
    }
    if (tmp_fd < 0) {
        std::cerr << "[audio] mkstemp() failed" << std::endl;
        return false;
    }

    size_t written = 0;
    while (written < size) {
        ssize_t n = write(tmp_fd, data + written, size - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        written += n;
    }
    fsync(tmp_fd);
    close(tmp_fd);

    if (written != size) {
        std::cerr << "[audio] failed to write temp file" << std::endl;
        unlink(tmp_path.c_str());
        return false;
    }

    fprintf(stderr, "[audio] temp file: %s (written=%zu/%zu bytes)\n", tmp_path.c_str(), written, size);

    // 验证临时文件大小
    {
        struct stat st;
        if (stat(tmp_path.c_str(), &st) == 0) {
            fprintf(stderr, "[audio] temp file on disk: %ld bytes\n", (long)st.st_size);
        }
    }

    // ffmpeg 转码: 使用 posix_spawn (避免 fork 后 CUDA/ATS 环境问题)
    std::string out_path = tmp_path + ".pcm";

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, STDIN_FILENO);

    // 关闭所有继承的 FD (3-1023), 防止 CUDA/ATS FD 干扰 ffmpeg
    for (int fd = 3; fd < 1024; fd++) {
        posix_spawn_file_actions_addclose(&actions, fd);
    }

    // stderr 不重定向, 让 ffmpeg 错误输出到服务器 stderr
    std::string ffmpeg_log = tmp_path + ".ffmpeg.log";

    const char* argv[] = {
        "ffmpeg", "-nostdin", "-y", "-v", "warning",
        "-i", tmp_path.c_str(),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "16000",
        out_path.c_str(), nullptr
    };

    pid_t pid;
    int spawn_ret = posix_spawn(&pid, "/usr/bin/ffmpeg", &actions, nullptr,
                                 const_cast<char**>(argv), environ);
    posix_spawn_file_actions_destroy(&actions);

    if (spawn_ret != 0) {
        fprintf(stderr, "[audio] posix_spawn failed: %s\n", strerror(spawn_ret));
        unlink(tmp_path.c_str());
        return false;
    }

    // 等待 ffmpeg 完成
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

    int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    if (exit_code != 0) {
        fprintf(stderr, "[audio] ffmpeg exited with code %d\n", exit_code);
    }

    // 清理输入临时文件
    unlink(tmp_path.c_str());

    // 读取输出 PCM 文件
    std::ifstream pcm_file(out_path, std::ios::binary | std::ios::ate);
    if (!pcm_file.is_open() || pcm_file.tellg() <= 0) {
        std::cerr << "[audio] ffmpeg produced no output (exit_code=" << exit_code << ")" << std::endl;
        unlink(out_path.c_str());
        return false;
    }

    size_t pcm_size = pcm_file.tellg();
    pcm_file.seekg(0);

    std::vector<uint8_t> pcm_buf(pcm_size);
    pcm_file.read(reinterpret_cast<char*>(pcm_buf.data()), pcm_size);
    pcm_file.close();
    unlink(out_path.c_str());

    // 解析 raw PCM16 mono 16kHz
    int num_samples = pcm_buf.size() / 2;
    out.sample_rate = 16000;
    out.channels = 1;
    out.samples.resize(num_samples);

    const int16_t* src = reinterpret_cast<const int16_t*>(pcm_buf.data());
    for (int i = 0; i < num_samples; i++) {
        out.samples[i] = src[i] / 32768.0f;
    }

    fprintf(stderr, "[audio] ffmpeg transcode OK: %d samples @ %d Hz (%.2fs)\n",
            num_samples, out.sample_rate, (float)num_samples / out.sample_rate);
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

// Radix-2 in-place FFT (Cooley-Tukey, decimation-in-time)
// Input/output: re[n], im[n] where n must be power of 2
static void fft_radix2(float* re, float* im, int n) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    // Butterfly stages
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len;
        float wre = std::cos(ang), wim = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            float cur_re = 1.0f, cur_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float tre = re[i + j + len/2] * cur_re - im[i + j + len/2] * cur_im;
                float tim = re[i + j + len/2] * cur_im + im[i + j + len/2] * cur_re;
                re[i + j + len/2] = re[i + j] - tre;
                im[i + j + len/2] = im[i + j] - tim;
                re[i + j] += tre;
                im[i + j] += tim;
                float new_re = cur_re * wre - cur_im * wim;
                cur_im = cur_re * wim + cur_im * wre;
                cur_re = new_re;
            }
        }
    }
}

// Bluestein FFT: exact N-point DFT for arbitrary N using O(N log N) radix-2 FFT.
// Works by converting the DFT into a convolution via chirp-z transform:
//   X[k] = w[k] * IFFT(FFT(a) * FFT(b))  where
//   a[j] = x[j] * w[j],  w[j] = exp(-iπj²/N),  b is the conjugate chirp
// The convolution is zero-padded to M ≥ 2N-1 (power of 2) for radix-2 FFT.
void rdft(const float* x, int n, float* real_out, float* imag_out) {
    int half = n / 2 + 1;

    // Cached Bluestein tables (recomputed only when n changes)
    thread_local int cached_n = 0;
    thread_local int cached_M = 0;
    thread_local std::vector<float> chirp_re, chirp_im;     // w[k] = exp(-iπk²/N), k=0..N-1
    thread_local std::vector<float> B_re, B_im;             // FFT of conjugate chirp, length M

    if (cached_n != n) {
        // Compute chirp factors: w[k] = exp(-iπk²/N)
        chirp_re.resize(n);
        chirp_im.resize(n);
        for (int k = 0; k < n; k++) {
            // Use fmod to maintain precision for large k²
            double angle = -M_PI * (double)((long long)k * k % (2 * (long long)n)) / (double)n;
            chirp_re[k] = (float)std::cos(angle);
            chirp_im[k] = (float)std::sin(angle);
        }

        // Pad size: M ≥ 2N-1, power of 2
        int M = 1;
        while (M < 2 * n - 1) M <<= 1;
        cached_M = M;

        // Build conjugate chirp sequence b[k]: b[k]=conj(w[k]) for k=0..N-1, b[M-k]=conj(w[k]) for k=1..N-1
        B_re.assign(M, 0.0f);
        B_im.assign(M, 0.0f);
        B_re[0] = chirp_re[0];
        B_im[0] = -chirp_im[0];
        for (int k = 1; k < n; k++) {
            B_re[k] = chirp_re[k];
            B_im[k] = -chirp_im[k];
            B_re[M - k] = chirp_re[k];
            B_im[M - k] = -chirp_im[k];
        }
        // Precompute FFT of B
        fft_radix2(B_re.data(), B_im.data(), M);

        cached_n = n;
    }

    int M = cached_M;

    // Build a[j] = x[j] * w[j], zero-padded to M
    thread_local std::vector<float> A_re, A_im;
    A_re.assign(M, 0.0f);
    A_im.assign(M, 0.0f);
    for (int j = 0; j < n; j++) {
        A_re[j] = x[j] * chirp_re[j];
        A_im[j] = x[j] * chirp_im[j];
    }

    // FFT(A)
    fft_radix2(A_re.data(), A_im.data(), M);

    // Pointwise multiply: C = FFT(A) * FFT(B)
    for (int i = 0; i < M; i++) {
        float re = A_re[i] * B_re[i] - A_im[i] * B_im[i];
        float im = A_re[i] * B_im[i] + A_im[i] * B_re[i];
        A_re[i] = re;
        A_im[i] = im;
    }

    // IFFT: conjugate, FFT, conjugate, scale by 1/M
    for (int i = 0; i < M; i++) A_im[i] = -A_im[i];
    fft_radix2(A_re.data(), A_im.data(), M);
    float inv_M = 1.0f / M;
    for (int i = 0; i < M; i++) {
        A_re[i] *= inv_M;
        A_im[i] *= -inv_M;
    }

    // X[k] = w[k] * IFFT_result[k]
    for (int k = 0; k < half; k++) {
        real_out[k] = chirp_re[k] * A_re[k] - chirp_im[k] * A_im[k];
        imag_out[k] = chirp_re[k] * A_im[k] + chirp_im[k] * A_re[k];
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

    // Center-padded STFT (matching WhisperFeatureExtractor):
    // Reflect-pad input by n_fft/2 on each side so each frame is centered
    int pad_len = n_fft / 2;  // 200 for n_fft=400
    int padded_len = num_samples + 2 * pad_len;
    padded_len = std::max(padded_len, n_fft);
    std::vector<float> padded(padded_len, 0.0f);

    // Reflect padding: pad[i] = samples[|i - pad_len| reflected]
    for (int i = 0; i < padded_len; i++) {
        int src_idx = i - pad_len;
        if (src_idx < 0) {
            src_idx = -src_idx;  // reflect at start
        } else if (src_idx >= num_samples) {
            src_idx = 2 * num_samples - src_idx - 2;  // reflect at end
        }
        if (src_idx >= 0 && src_idx < num_samples) {
            padded[i] = samples[src_idx];
        }
    }

    num_frames = (padded_len - n_fft) / hop + 1;

    // WhisperFeatureExtractor drops the last frame: log_spec = log_spec[:, :-1]
    if (num_frames > 1) num_frames--;

    // Compute STFT → power spectrum → mel
    std::vector<float> mel_spec(n_mels * num_frames, 0.0f);
    std::vector<float> frame(n_fft);
    std::vector<float> fft_real(n_freqs), fft_imag(n_freqs);
    std::vector<float> power(n_freqs);

    // Precompute non-zero ranges for each mel bin (sparse filterbank)
    std::vector<int> mel_start(n_mels, n_freqs);
    std::vector<int> mel_end(n_mels, 0);
    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_freqs; k++) {
            if (mel_fb[m * n_freqs + k] != 0.0f) {
                if (k < mel_start[m]) mel_start[m] = k;
                if (k + 1 > mel_end[m]) mel_end[m] = k + 1;
            }
        }
    }

    for (int t = 0; t < num_frames; t++) {
        // Window input
        for (int i = 0; i < n_fft; i++) {
            frame[i] = padded[t * hop + i] * window[i];
        }

        // FFT
        rdft(frame.data(), n_fft, fft_real.data(), fft_imag.data());

        // Power spectrum (computed once, not 128 times)
        for (int k = 0; k < n_freqs; k++) {
            power[k] = fft_real[k] * fft_real[k] + fft_imag[k] * fft_imag[k];
        }

        // Mel filterbank (sparse: only iterate non-zero range per mel bin)
        for (int m = 0; m < n_mels; m++) {
            float sum = 0;
            for (int k = mel_start[m]; k < mel_end[m]; k++) {
                sum += mel_fb[m * n_freqs + k] * power[k];
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
