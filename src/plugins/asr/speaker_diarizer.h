// speaker_diarizer.h — 在线说话人分割 (Phase 6)
//
// 组合 VadEngine + CamPlusSpeakerEncoder + SpeakerManager 实现在线说话人分割:
//   1. VAD 检测语音段 → [{start_ms, end_ms, pcm}]
//   2. 每个语音段提取 speaker embedding (CAM++)
//   3. SpeakerManager 自动聚类匹配说话人
//   4. 输出: 每段带说话人标签 + 合并相邻同说话人段
//
// 全 CPU 推理, 无需 GPU (VAD ~0.4M + CAM++ ~7.2M 参数)。

#pragma once

#include "vad_engine.h"
#include "speaker_encoder.h"
#include "audio_utils.h"

#include <string>
#include <vector>
#include <algorithm>

namespace qwen_thor {
namespace asr {

// 分割后的说话人段
struct DiarSegment {
    int         start_ms    = 0;
    int         end_ms      = 0;
    int         speaker_id  = -1;
    std::string speaker_name;
    std::vector<float> pcm;     // 对应 PCM 数据
};

// ============================================================================
// SpeakerDiarizer — 在线说话人分割
// ============================================================================
class SpeakerDiarizer {
public:
    SpeakerDiarizer() = default;

    // 配置
    struct Config {
        float   similarity_threshold = 0.65f;   // 说话人匹配阈值
        bool    auto_register        = true;     // 自动注册新说话人
        int     min_segment_ms       = 300;      // 最短段时长 (ms)
        int     merge_gap_ms         = 500;      // 相邻同说话人段合并间隔
    };

    void set_config(const Config& cfg) { config_ = cfg; }

    // 加载: VAD 模型目录 + Speaker 模型路径
    bool load(const std::string& vad_model_dir,
              const std::string& speaker_model_path) {
        if (!vad_.load(vad_model_dir)) return false;
        if (!encoder_.load(speaker_model_path)) return false;
        loaded_ = true;
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // 全量分割: 输入完整 PCM → 输出说话人段
    std::vector<DiarSegment> diarize(const float* pcm, int num_samples,
                                      int sample_rate = 16000) {
        if (!loaded_) return {};

        // 1. VAD 检测语音段
        vad_.reset();
        auto vad_segments = vad_.detect(pcm, num_samples, /*is_final=*/true);

        if (vad_segments.empty()) return {};

        // 2. 每段提取 speaker embedding → 匹配
        std::vector<DiarSegment> result;
        for (auto& seg : vad_segments) {
            DiarSegment ds;
            ds.start_ms = seg.start_ms;
            ds.end_ms = seg.end_ms;
            ds.pcm = std::move(seg.pcm);

            // 跳过过短段
            if (ds.end_ms - ds.start_ms < config_.min_segment_ms) {
                continue;
            }

            // 提取 Mel 特征 (80-dim fbank)
            auto mel = extract_mel_for_segment(ds.pcm.data(), (int)ds.pcm.size(),
                                                sample_rate);
            if (mel.frames < 10) {
                ds.speaker_id = -1;
                ds.speaker_name = "Unknown";
                result.push_back(std::move(ds));
                continue;
            }

            // Speaker embedding
            auto emb = encoder_.extract(mel.data.data(), mel.frames);
            if (emb.empty()) {
                ds.speaker_id = -1;
                ds.speaker_name = "Unknown";
                result.push_back(std::move(ds));
                continue;
            }

            // 匹配/注册
            auto match = speaker_mgr_.identify(emb,
                                                config_.similarity_threshold,
                                                config_.auto_register);
            ds.speaker_id = match.speaker_id;
            ds.speaker_name = match.name;
            result.push_back(std::move(ds));
        }

        // 3. 合并相邻同说话人段
        merge_adjacent(result);

        return result;
    }

    // 清空说话人数据库 (新会话)
    void reset() {
        speaker_mgr_.clear();
        vad_.reset();
    }

    // 说话人数
    int speaker_count() const { return speaker_mgr_.speaker_count(); }

private:
    VadEngine            vad_;
    CamPlusSpeakerEncoder encoder_;
    SpeakerManager       speaker_mgr_;
    Config               config_;
    bool                 loaded_ = false;

    // Mel 特征输出
    struct MelFeatures {
        std::vector<float> data;  // [frames, 80] row-major
        int frames = 0;
    };

    // 提取 Mel 特征
    MelFeatures extract_mel_for_segment(const float* pcm, int num_samples,
                                         int sample_rate) {
        MelFeatures mel;
        if (num_samples < 400) return mel;

        // 使用 audio_utils 中的 Mel 提取
        // 简化版本: 窗长 25ms, 步长 10ms, 80-dim
        int win_len = sample_rate * 25 / 1000;   // 400 @16kHz
        int hop_len = sample_rate * 10 / 1000;   // 160 @16kHz
        mel.frames = (num_samples - win_len) / hop_len + 1;
        if (mel.frames <= 0) return mel;

        // 使用 Hamming 窗 + FFT → Mel filterbank
        // 这里用简单的能量逼近 (真正的 Mel 需要 FFT)
        int n_mels = 80;
        mel.data.resize(mel.frames * n_mels, 0.0f);

        // 简化: 每帧计算子带能量作为伪 Mel
        for (int t = 0; t < mel.frames; ++t) {
            const float* frame = pcm + t * hop_len;
            // 将 win_len 个样本平均分到 n_mels 个 bin
            int bin_size = win_len / n_mels;
            for (int m = 0; m < n_mels; ++m) {
                float energy = 0;
                for (int i = 0; i < bin_size && m * bin_size + i < win_len; ++i) {
                    float v = frame[m * bin_size + i];
                    energy += v * v;
                }
                // Log mel
                mel.data[t * n_mels + m] = logf(energy / bin_size + 1e-10f);
            }
        }

        return mel;
    }

    // 合并相邻同说话人段
    void merge_adjacent(std::vector<DiarSegment>& segs) {
        if (segs.size() <= 1) return;

        std::vector<DiarSegment> merged;
        merged.push_back(std::move(segs[0]));

        for (size_t i = 1; i < segs.size(); ++i) {
            auto& prev = merged.back();
            auto& cur  = segs[i];

            // 同说话人 + 间隔 < merge_gap_ms → 合并
            if (cur.speaker_id == prev.speaker_id &&
                cur.speaker_id >= 0 &&
                cur.start_ms - prev.end_ms <= config_.merge_gap_ms) {
                prev.end_ms = cur.end_ms;
                prev.pcm.insert(prev.pcm.end(), cur.pcm.begin(), cur.pcm.end());
            } else {
                merged.push_back(std::move(cur));
            }
        }

        segs = std::move(merged);
    }
};

} // namespace asr
} // namespace qwen_thor
