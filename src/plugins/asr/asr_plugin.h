// asr_plugin.h — ASR (Automatic Speech Recognition) 插件
//
// 提供语音识别功能, 将音频文件转录为文本。
// 默认实现基于子进程调用外部 ASR 工具 (如 whisper.cpp, sherpa-onnx, faster-whisper 等)。
//
// API 端点 (OpenAI 兼容):
//   POST /v1/audio/transcriptions  — 音频转文本
//
// 配置:
//   asr_enabled=true
//   asr_executable=/path/to/whisper-cli
//   asr_model=/path/to/whisper-model
//   asr_language=auto

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>

// Forward declarations
namespace qwen_thor { namespace asr { class ASREngine; } }

namespace qwen_thor {
namespace plugins {

// ============================================================================
// ASR 配置
// ============================================================================
struct AsrConfig {
    bool        enabled     = false;
    std::string mode        = "subprocess"; // "native" 或 "subprocess"
    std::string executable;           // ASR 可执行文件路径 (subprocess only)
    std::string model_path;           // ASR 模型目录 (native) 或模型文件 (subprocess)
    std::string language    = "auto"; // 默认语言 ("auto", "zh", "en", "ja", ...)
    int         threads     = 4;      // CPU 线程数
    std::string extra_args;           // 额外 CLI 参数 (直接追加到命令行)
    std::string tmp_dir     = "tmp";  // 临时文件目录
    std::string speaker_model;        // CAM++ 说话人编码器 safetensors 路径 (可选)
    float       repetition_penalty = 1.0f; // ASR 解码 repetition penalty (> 1.0 抑制重复)
    int         chunk_max_duration_ms = 30000; // ASR 分段最大时长 (ms), 短段减少错误累积

    static AsrConfig from_file(const std::string& path);
    void print() const;
};

// ============================================================================
// ASR 转录结果
// ============================================================================
struct AsrResult {
    // ─── 基础 ───
    std::string text;              // 转录文本 (无标点)
    std::string language;          // 检测到的语言
    float       duration_s = 0;    // 音频时长 (秒)
    int         error_code = 0;    // 0 = 成功
    std::string error_message;     // 错误信息

    // ─── Phase 4: 标点恢复 ───
    std::string text_with_punc;    // 带标点文本

    // ─── Phase 5: 时间戳 ───
    struct WordInfo {
        std::string word;
        int   start_ms = -1;
        int   end_ms   = -1;
        float confidence = 0;
        int   speaker_id = -1;    // Phase 6
    };
    std::vector<WordInfo> words;

    // ─── Phase 6: 说话人分割 ───
    struct SpeakerSegment {
        int         start_ms = 0;
        int         end_ms   = 0;
        int         speaker_id = -1;
        std::string speaker_name;
        std::string text;
    };
    std::vector<SpeakerSegment> segments;

    // ─── Phase 3: 关键词识别 ───
    struct KeywordHit {
        std::string keyword;
        std::string action;
        int   char_offset = 0;     // UTF-8 字符偏移
        float confidence = 0;
    };
    std::vector<KeywordHit> keyword_hits;

    // ─── Phase 7: 情感 (预留) ───
    std::string emotion;           // "neutral"/"happy"/"sad"/...
    float emotion_confidence = 0;
};

// ============================================================================
// ASR 插件接口
// ============================================================================
class AsrPlugin {
public:
    virtual ~AsrPlugin() = default;

    // 转录音频文件
    // audio_path: 音频文件路径 (wav/mp3/m4a/ogg/webm/flac)
    // language: 语言代码, "auto" 自动检测
    virtual AsrResult transcribe(const std::string& audio_path,
                                 const std::string& language = "auto") = 0;

    // 转录内存中的音频数据 (跳过临时文件 I/O)
    // 默认实现: 写临时文件 → 调用 transcribe(), 子类可覆盖
    // suppress_early_eos: 抑制停顿导致的过早 EOS, 适用于长音频场景
    virtual AsrResult transcribe_memory(const uint8_t* data, size_t size,
                                        const std::string& language = "auto",
                                        const std::string& filename_hint = "",
                                        bool suppress_early_eos = false);

    // 转录原始 PCM float 样本 (流式 ASR 使用)
    // samples: float 数组 [-1, 1], sample_rate: 采样率
    virtual AsrResult transcribe_pcm(const float* samples, int num_samples,
                                     int sample_rate = 16000,
                                     const std::string& language = "auto",
                                     bool suppress_early_eos = false);

    // 检查插件是否可用 (可执行文件存在, 模型存在等)
    virtual bool is_available() const = 0;

    // 获取插件名称
    virtual std::string name() const = 0;
};

// ============================================================================
// 子进程 ASR 实现 — 调用外部 ASR 可执行文件
// ============================================================================
class SubprocessAsrPlugin : public AsrPlugin {
public:
    explicit SubprocessAsrPlugin(const AsrConfig& config);

    AsrResult transcribe(const std::string& audio_path,
                         const std::string& language = "auto") override;
    bool is_available() const override;
    std::string name() const override { return "subprocess-asr"; }

private:
    AsrConfig config_;
};

// ============================================================================
// 原生 ASR 实现 — 使用内置 Qwen3-ASR 引擎
// ============================================================================
class NativeAsrPlugin : public AsrPlugin {
public:
    explicit NativeAsrPlugin(const AsrConfig& config);
    ~NativeAsrPlugin() override;

    AsrResult transcribe(const std::string& audio_path,
                         const std::string& language = "auto") override;
    AsrResult transcribe_memory(const uint8_t* data, size_t size,
                                const std::string& language = "auto",
                                const std::string& filename_hint = "",
                                bool suppress_early_eos = false) override;
    AsrResult transcribe_pcm(const float* samples, int num_samples,
                             int sample_rate = 16000,
                             const std::string& language = "auto",
                             bool suppress_early_eos = false) override;

    // 批量转录: 多段 PCM → batch decode (GEMV→GEMM)
    struct PcmChunk { const float* samples; int num_samples; };
    std::vector<AsrResult> transcribe_batch_pcm(
        const std::vector<PcmChunk>& chunks,
        int sample_rate = 16000,
        const std::string& language = "auto",
        bool suppress_early_eos = false);

    bool is_available() const override;
    std::string name() const override { return "native-asr"; }
    int chunk_max_duration_ms() const { return config_.chunk_max_duration_ms; }

private:
    AsrConfig config_;
    std::unique_ptr<asr::ASREngine> engine_;
    std::mutex mutex_;
};

// 工厂: 根据配置创建 ASR 插件
std::unique_ptr<AsrPlugin> create_asr_plugin(const AsrConfig& config);

} // namespace plugins
} // namespace qwen_thor
