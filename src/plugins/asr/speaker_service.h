// speaker_service.h — 说话人识别服务
//
// 封装 Mel 提取 + CAM++ embedding + SpeakerManager 匹配, 供 serve 层和 pipeline 层调用。
// 线程安全: 内部持有 mutex 保护 speaker_encoder_ 和 speaker_manager_。

#pragma once

#include "speaker_manager.h"
#include "speaker_encoder_gpu.h"
#include "mel_gpu.h"
#include "audio_utils.h"

#include <vector>
#include <mutex>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen_thor {
namespace asr {

class SpeakerService {
public:
    SpeakerService() = default;

    // 设置组件引用 (不拥有所有权, serve 层负责生命周期)
    void init(GpuSpeakerEncoder* encoder, GpuMelExtractor* gpu_mel) {
        speaker_encoder_ = encoder;
        gpu_mel_ = gpu_mel;
    }

    bool is_available() const { return speaker_encoder_ != nullptr; }

    // 从 PCM 音频识别说话人
    SpeakerManager::MatchResult identify(
            const float* samples, int num_samples, int sample_rate,
            bool auto_register = false, float threshold = 0.65f) {
        SpeakerManager::MatchResult result;
        result.speaker_id = -1;
        result.name = "Unknown";

        if (!speaker_encoder_) return result;

        std::vector<float> mel;
        int num_frames = 0;
        compute_mel_80(samples, num_samples, sample_rate, mel, num_frames);
        if (num_frames < 10) return result;

        std::lock_guard<std::mutex> lock(mutex_);
        auto embedding = speaker_encoder_->extract(mel.data(), num_frames);
        if (embedding.empty()) return result;

        result = speaker_manager_.identify(embedding, threshold, auto_register);
        return result;
    }

    // 提取 speaker embedding (不匹配, 仅返回 192-dim 向量)
    std::vector<float> extract_embedding(const float* mel, int num_frames) {
        if (!speaker_encoder_) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        return speaker_encoder_->extract(mel, num_frames);
    }

    // 带锁提取: PCM → Mel → embedding
    std::vector<float> extract_embedding_from_pcm(
            const float* samples, int num_samples, int sample_rate) {
        std::vector<float> mel;
        int num_frames = 0;
        compute_mel_80(samples, num_samples, sample_rate, mel, num_frames);
        if (num_frames < 10) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        return speaker_encoder_->extract(mel.data(), num_frames);
    }

    // GPU Mel 提取 (for batch pipeline)
    int compute_gpu_mel(const float* pcm, int num_samples, std::vector<float>& mel) {
        if (gpu_mel_ && gpu_mel_->is_initialized())
            return gpu_mel_->compute(pcm, num_samples, mel);
        return 0;
    }

    bool has_gpu_mel() const { return gpu_mel_ && gpu_mel_->is_initialized(); }

    // 批量 GPU 提取 embeddings (for V4 pipeline spectral clustering)
    std::vector<std::vector<float>> extract_batch_gpu(
            const std::vector<GpuSpeakerEncoder::BatchChunk>& chunks) {
        if (!speaker_encoder_) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        return speaker_encoder_->extract_batch_gpu(chunks);
    }

    // 访问 SpeakerManager (注册/列表/删除)
    SpeakerManager& manager() { return speaker_manager_; }
    const SpeakerManager& manager() const { return speaker_manager_; }
    std::mutex& mutex() { return mutex_; }

    // 80-dim Mel 特征提取 (CPU, Kaldi-compatible for CAM++)
    static void compute_mel_80(const float* samples, int num_samples, int sample_rate,
                               std::vector<float>& mel_out, int& num_frames) {
        const int n_fft = 400;
        const int fft_size = 512;
        const int hop = 160;
        const int n_mels = 80;
        const int n_freqs = fft_size / 2 + 1;
        const float low_freq = 20.0f;
        const int target_sr = 16000;

        std::vector<float> resampled_buf;
        const float* pcm = samples;
        int pcm_len = num_samples;
        if (sample_rate != target_sr) {
            std::vector<float> input_vec(samples, samples + num_samples);
            audio::resample(input_vec, sample_rate, resampled_buf, target_sr);
            pcm = resampled_buf.data();
            pcm_len = (int)resampled_buf.size();
        }

        std::vector<float> scaled(pcm_len);
        for (int i = 0; i < pcm_len; i++)
            scaled[i] = pcm[i] * 32768.0f;

        for (int i = pcm_len - 1; i > 0; i--)
            scaled[i] -= 0.97f * scaled[i - 1];
        scaled[0] *= (1.0f - 0.97f);

        num_frames = (pcm_len - n_fft) / hop + 1;
        if (num_frames <= 0) { mel_out.clear(); return; }

        thread_local std::vector<float> mel_fb;
        thread_local std::vector<float> povey_win;
        thread_local bool fb_built = false;
        if (!fb_built) {
            mel_fb.resize((size_t)n_mels * n_freqs, 0.0f);
            auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
            auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

            float min_mel = hz_to_mel(low_freq);
            float max_mel = hz_to_mel((float)target_sr / 2.0f);
            std::vector<float> mel_points(n_mels + 2);
            for (int i = 0; i < n_mels + 2; i++)
                mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_mels + 1));

            for (int m = 0; m < n_mels; m++) {
                float left = mel_points[m] * fft_size / target_sr;
                float center = mel_points[m + 1] * fft_size / target_sr;
                float right = mel_points[m + 2] * fft_size / target_sr;
                for (int k = 0; k < n_freqs; k++) {
                    float fk = (float)k;
                    if (fk >= left && fk <= center)
                        mel_fb[m * n_freqs + k] = (fk - left) / (center - left);
                    else if (fk > center && fk <= right)
                        mel_fb[m * n_freqs + k] = (right - fk) / (right - center);
                }
            }

            povey_win.resize(n_fft);
            for (int i = 0; i < n_fft; i++)
                povey_win[i] = std::pow(0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (n_fft - 1)), 0.85f);

            fb_built = true;
        }

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

        std::vector<float> mel_spec(n_mels * num_frames, 0.0f);
        std::vector<float> frame(fft_size, 0.0f);

        for (int t = 0; t < num_frames; t++) {
            for (int i = 0; i < n_fft; i++)
                frame[i] = scaled[t * hop + i] * povey_win[i];
            for (int i = n_fft; i < fft_size; i++)
                frame[i] = 0.0f;

            thread_local std::vector<float> tw_re, tw_im;
            thread_local bool tw_built = false;
            if (!tw_built) {
                tw_re.resize(n_freqs * fft_size);
                tw_im.resize(n_freqs * fft_size);
                for (int k = 0; k < n_freqs; k++) {
                    for (int n = 0; n < fft_size; n++) {
                        double angle = -2.0 * M_PI * k * n / fft_size;
                        tw_re[k * fft_size + n] = (float)std::cos(angle);
                        tw_im[k * fft_size + n] = (float)std::sin(angle);
                    }
                }
                tw_built = true;
            }

            std::vector<float> power(n_freqs);
            for (int k = 0; k < n_freqs; k++) {
                float re = 0, im = 0;
                const float* tw_r = &tw_re[k * fft_size];
                const float* tw_i = &tw_im[k * fft_size];
                for (int n = 0; n < fft_size; n++) {
                    re += frame[n] * tw_r[n];
                    im += frame[n] * tw_i[n];
                }
                power[k] = re * re + im * im;
            }

            for (int m = 0; m < n_mels; m++) {
                float sum = 0;
                for (int k = mel_start[m]; k < mel_end[m]; k++)
                    sum += mel_fb[m * n_freqs + k] * power[k];
                mel_spec[m * num_frames + t] = std::log(std::max(sum, 1.175494e-38f));
            }
        }

        mel_out.resize(n_mels * num_frames);
        for (int t = 0; t < num_frames; t++)
            for (int f = 0; f < n_mels; f++)
                mel_out[t * n_mels + f] = mel_spec[f * num_frames + t];
    }

private:
    GpuSpeakerEncoder* speaker_encoder_ = nullptr;
    GpuMelExtractor* gpu_mel_ = nullptr;
    SpeakerManager speaker_manager_;
    std::mutex mutex_;
};

} // namespace asr
} // namespace qwen_thor
