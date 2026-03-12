// tts_plugin.h — TTS (Text-to-Speech) 插件
//
// 提供语音合成功能, 将文本转换为音频。
// 默认实现基于子进程调用外部 TTS 工具 (如 piper, sherpa-onnx, espeak-ng 等)。
//
// API 端点 (OpenAI 兼容):
//   POST /v1/audio/speech  — 文本转语音
//
// 配置:
//   tts_enabled=true
//   tts_executable=/path/to/piper
//   tts_model=/path/to/tts-model
//   tts_voice=default

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace qwen_thor {
namespace plugins {

// ============================================================================
// TTS 配置
// ============================================================================
struct TtsConfig {
    bool        enabled      = false;
    std::string executable;             // TTS 可执行文件路径
    std::string model_path;             // TTS 模型路径
    std::string voice        = "default"; // 默认语音
    float       speed        = 1.0f;    // 语速 (0.5-2.0)
    std::string format       = "wav";   // 输出格式 (wav/mp3/opus/pcm)
    std::string extra_args;             // 额外 CLI 参数
    std::string tmp_dir      = "tmp";   // 临时文件目录

    static TtsConfig from_file(const std::string& path);
    void print() const;
};

// ============================================================================
// TTS 合成结果
// ============================================================================
struct TtsResult {
    std::vector<uint8_t> audio_data;   // 音频二进制数据
    std::string          format;        // 音频格式 (wav/mp3/opus)
    float                duration_s = 0; // 音频时长 (秒)
    int                  error_code = 0;
    std::string          error_message;
};

// ============================================================================
// TTS 插件接口
// ============================================================================
class TtsPlugin {
public:
    virtual ~TtsPlugin() = default;

    // 合成语音
    // text: 要合成的文本
    // voice: 语音名称 (空字符串使用默认)
    // speed: 语速倍率
    // format: 输出格式 (wav/mp3/opus/pcm)
    virtual TtsResult synthesize(const std::string& text,
                                 const std::string& voice = "",
                                 float speed = 1.0f,
                                 const std::string& format = "wav") = 0;

    // 检查插件是否可用
    virtual bool is_available() const = 0;

    // 获取插件名称
    virtual std::string name() const = 0;
};

// ============================================================================
// 子进程 TTS 实现 — 调用外部 TTS 可执行文件
// ============================================================================
class SubprocessTtsPlugin : public TtsPlugin {
public:
    explicit SubprocessTtsPlugin(const TtsConfig& config);

    TtsResult synthesize(const std::string& text,
                         const std::string& voice = "",
                         float speed = 1.0f,
                         const std::string& format = "wav") override;
    bool is_available() const override;
    std::string name() const override { return "subprocess-tts"; }

private:
    TtsConfig config_;
};

// 工厂: 根据配置创建 TTS 插件
std::unique_ptr<TtsPlugin> create_tts_plugin(const TtsConfig& config);

} // namespace plugins
} // namespace qwen_thor
