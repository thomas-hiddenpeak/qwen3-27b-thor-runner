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

    static AsrConfig from_file(const std::string& path);
    void print() const;
};

// ============================================================================
// ASR 转录结果
// ============================================================================
struct AsrResult {
    std::string text;              // 转录文本
    std::string language;          // 检测到的语言
    float       duration_s = 0;    // 音频时长 (秒)
    int         error_code = 0;    // 0 = 成功
    std::string error_message;     // 错误信息
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
    virtual AsrResult transcribe_memory(const uint8_t* data, size_t size,
                                        const std::string& language = "auto");

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
                                const std::string& language = "auto") override;
    bool is_available() const override;
    std::string name() const override { return "native-asr"; }

private:
    AsrConfig config_;
    std::unique_ptr<asr::ASREngine> engine_;
    std::mutex mutex_;
};

// 工厂: 根据配置创建 ASR 插件
std::unique_ptr<AsrPlugin> create_asr_plugin(const AsrConfig& config);

} // namespace plugins
} // namespace qwen_thor
