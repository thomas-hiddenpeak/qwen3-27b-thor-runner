// aligner_engine.h — Qwen3-ForcedAligner 字级时间戳 (Phase 5)
//
// 基于 Qwen3-ForcedAligner-0.6B 的非自回归强制对齐:
//   1. Audio → Encoder → audio features
//   2. 构建 prompt: <audio> + text + <timestamp> tokens
//   3. Single forward → logits at <timestamp> positions
//   4. argmax → time class → timestamp_ms (class_id × 80ms)
//
// 架构: 与 ASR-1.7B 相同但 hidden=1024 (半尺寸)。
// 复用 ASREngine 的 Encoder/Decoder, 参数化为 1024 维度。
//
// 注意: 这是一个独立模块, 需要额外 ~1.8 GB 内存。
// 按需加载/卸载以控制内存占用。

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace qwen_thor {
namespace asr {

// 对齐结果: 每个词的时间戳
struct AlignedWord {
    std::string word;
    int   start_ms    = -1;
    int   end_ms      = -1;
    float confidence  = 0;
};

// ============================================================================
// AlignerEngine — Qwen3-ForcedAligner 时间戳提取
// ============================================================================
class AlignerEngine {
public:
    AlignerEngine() = default;
    ~AlignerEngine() = default;

    // 占位: 加载模型 (需要 GPU, 复用 ASR encoder/decoder 实现)
    // model_dir: /path/to/Qwen3-ForcedAligner-0.6B/
    bool load_model(const std::string& model_dir) {
        model_dir_ = model_dir;
        // 验证文件存在
        std::string config_path = model_dir + "/config.json";
        std::string model_path = model_dir + "/model.safetensors";

        std::ifstream cf(config_path);
        std::ifstream mf(model_path);
        if (!cf.is_open() || !mf.is_open()) {
            return false;
        }

        // 解析关键参数
        classify_num_ = 5000;
        timestamp_segment_time_ = 80;  // ms per class
        hidden_size_ = 1024;
        encoder_layers_ = 24;
        decoder_layers_ = 28;
        loaded_ = true;

        // 注意: 完整 GPU 加载需要复用 ASR Encoder/Decoder 框架
        // 当前为结构占位, forward 使用 stub
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // 对齐: 给定 PCM 音频 + ASR 文本 → 每词时间戳
    // words: ASR 分词结果 (UTF-8 字/词)
    std::vector<AlignedWord> align(const float* pcm, int num_samples,
                                    int sample_rate,
                                    const std::vector<std::string>& words) {
        if (!loaded_ || words.empty()) return {};

        float duration_s = (float)num_samples / sample_rate;
        int duration_ms = (int)(duration_s * 1000);

        // 均匀分布时间戳 (占位实现, 等待 GPU forward 完善)
        // 真正的实现需要:
        //   1. Mel → Encoder → audio_features
        //   2. 构建 <audio> + word + <ts> + word + <ts> + ... 序列
        //   3. Decoder forward → logits at <ts> positions
        //   4. argmax → class_id × 80ms
        return align_uniform(words, duration_ms);
    }

    // 从 ASR 文本提取词列表 (中文按字, 英文按空格)
    static std::vector<std::string> tokenize_for_align(const std::string& text) {
        std::vector<std::string> words;
        int i = 0;
        while (i < (int)text.size()) {
            unsigned char c = (unsigned char)text[i];
            if (c < 0x80) {
                // ASCII: 累积连续字母/数字为一个词
                if (c == ' ' || c == '\t' || c == '\n') {
                    ++i;
                    continue;
                }
                std::string word;
                while (i < (int)text.size() && (unsigned char)text[i] < 0x80 &&
                       text[i] != ' ' && text[i] != '\t' && text[i] != '\n') {
                    word += text[i++];
                }
                if (!word.empty()) words.push_back(word);
            } else {
                // UTF-8 多字节: 中文按字拆分
                int len = 1;
                if (c >= 0xF0) len = 4;
                else if (c >= 0xE0) len = 3;
                else if (c >= 0xC0) len = 2;
                words.push_back(text.substr(i, len));
                i += len;
            }
        }
        return words;
    }

    // 确保时间戳单调递增 (LIS 后处理)
    static void fix_timestamps(std::vector<AlignedWord>& words) {
        if (words.size() <= 1) return;

        // Longest Increasing Subsequence on start_ms
        int n = (int)words.size();
        std::vector<int> starts(n);
        for (int i = 0; i < n; ++i) starts[i] = words[i].start_ms;

        // Find LIS
        std::vector<int> dp, parent(n, -1), pos;
        for (int i = 0; i < n; ++i) {
            auto it = std::lower_bound(dp.begin(), dp.end(), starts[i]);
            int idx = (int)(it - dp.begin());
            if (it == dp.end()) {
                dp.push_back(starts[i]);
                pos.push_back(i);
            } else {
                *it = starts[i];
                pos[idx] = i;
            }
            parent[i] = idx > 0 ? pos[idx - 1] : -1;
        }

        // Reconstruct LIS
        std::vector<bool> in_lis(n, false);
        int cur = pos.back();
        while (cur >= 0) {
            in_lis[cur] = true;
            cur = parent[cur];
        }

        // Interpolate non-LIS elements
        int prev_ms = 0;
        int prev_idx = -1;
        for (int i = 0; i < n; ++i) {
            if (in_lis[i]) {
                // Fix gaps between prev_idx and i
                if (prev_idx >= 0 && i - prev_idx > 1) {
                    int gap = words[i].start_ms - prev_ms;
                    int steps = i - prev_idx;
                    for (int j = prev_idx + 1; j < i; ++j) {
                        words[j].start_ms = prev_ms + gap * (j - prev_idx) / steps;
                    }
                }
                prev_ms = words[i].start_ms;
                prev_idx = i;
            }
        }
        // Fix trailing
        if (prev_idx >= 0 && prev_idx < n - 1) {
            for (int j = prev_idx + 1; j < n; ++j) {
                words[j].start_ms = prev_ms + 80 * (j - prev_idx);
            }
        }

        // Ensure end_ms = next start_ms
        for (int i = 0; i < n - 1; ++i) {
            words[i].end_ms = words[i + 1].start_ms;
        }
    }

private:
    std::string model_dir_;
    bool loaded_ = false;

    int classify_num_ = 5000;
    int timestamp_segment_time_ = 80;   // ms per class
    int hidden_size_ = 1024;
    int encoder_layers_ = 24;
    int decoder_layers_ = 28;

    // 均匀分布占位对齐
    std::vector<AlignedWord> align_uniform(const std::vector<std::string>& words,
                                            int duration_ms) {
        std::vector<AlignedWord> result;
        int n = (int)words.size();
        for (int i = 0; i < n; ++i) {
            AlignedWord aw;
            aw.word = words[i];
            aw.start_ms = duration_ms * i / n;
            aw.end_ms = duration_ms * (i + 1) / n;
            aw.confidence = 0.5f;  // Low confidence for uniform alignment
            result.push_back(aw);
        }
        return result;
    }

    // Token IDs for prompt construction
    static constexpr int AUDIO_START_TOKEN = 151669;
    static constexpr int AUDIO_END_TOKEN   = 151670;
    static constexpr int AUDIO_PAD_TOKEN   = 151676;
    static constexpr int TIMESTAMP_TOKEN   = 151705;
};

} // namespace asr
} // namespace qwen_thor
