// tts_plugin.h — TTS (Text-to-Speech) 插件
//
// 提供语音合成功能, 将文本转换为音频。
// 两种实现:
//   1. NativeTtsPlugin — 使用内置 Qwen3-TTS 引擎 (GPU 推理)
//   2. SubprocessTtsPlugin — 调用外部 TTS 可执行文件
//
// API 端点 (OpenAI 兼容):
//   POST /v1/audio/speech  — 文本转语音

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <mutex>

// Forward declarations
namespace qwen_thor { namespace tts { class TTSEngine; } }

namespace qwen_thor {
namespace plugins {

// ============================================================================
// TTS 配置
// ============================================================================
struct TtsConfig {
    bool        enabled      = false;
    std::string mode         = "native";    // "native" 或 "subprocess"
    std::string model_path;                 // TTS 模型目录 (native) 或模型文件 (subprocess)
    std::string executable;                 // TTS 可执行文件路径 (subprocess only)
    std::string voice        = "serena";    // 默认语音
    std::string language     = "auto";      // 默认语言
    float       speed        = 1.0f;        // 语速 (0.5-2.0)
    std::string format       = "wav";       // 输出格式 (wav/pcm)
    std::string extra_args;                 // 额外 CLI 参数 (subprocess)
    std::string tmp_dir      = "tmp";       // 临时文件目录
    int         max_new_tokens = 4096;      // 最大生成 token 数

    static TtsConfig from_file(const std::string& path);
    void print() const;
};

// ============================================================================
// TTS 合成结果
// ============================================================================
struct TtsResult {
    std::vector<uint8_t> audio_data;   // 音频二进制数据
    std::string          format;        // 音频格式 (wav/pcm)
    float                duration_s = 0; // 音频时长 (秒)
    int                  sample_rate = 24000;
    int                  error_code = 0;
    std::string          error_message;
};

// ============================================================================
// TTS 插件接口
// ============================================================================
class TtsPlugin {
public:
    virtual ~TtsPlugin() = default;

    virtual TtsResult synthesize(const std::string& text,
                                 const std::string& voice = "",
                                 float speed = 1.0f,
                                 const std::string& format = "wav") = 0;

    // Continue synthesis without resetting talker state (voice consistency)
    virtual TtsResult synthesize_continue(const std::string& text,
                                          const std::string& format = "pcm") = 0;

    virtual bool is_available() const = 0;
    virtual std::string name() const = 0;

    // Set TTS sampling parameters (temperature, top_k, top_p, repetition_penalty)
    virtual void set_sampling(float temperature, int top_k, float top_p, float rep_penalty) {}
};

// ============================================================================
// 原生 TTS 实现 — 使用内置 Qwen3-TTS 引擎
// ============================================================================
class NativeTtsPlugin : public TtsPlugin {
public:
    explicit NativeTtsPlugin(const TtsConfig& config);
    ~NativeTtsPlugin() override;

    TtsResult synthesize(const std::string& text,
                         const std::string& voice = "",
                         float speed = 1.0f,
                         const std::string& format = "wav") override;
    TtsResult synthesize_continue(const std::string& text,
                                  const std::string& format = "pcm") override;
    bool is_available() const override;
    std::string name() const override { return "native-qwen3-tts"; }
    void set_sampling(float temperature, int top_k, float top_p, float rep_penalty) override;

private:
    TtsConfig config_;
    std::unique_ptr<tts::TTSEngine> engine_;
    std::mutex mutex_;  // TTS engine is not thread-safe
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
    TtsResult synthesize_continue(const std::string& text,
                                  const std::string& format = "pcm") override;
    bool is_available() const override;
    std::string name() const override { return "subprocess-tts"; }

private:
    TtsConfig config_;
};

// 工厂: 根据配置创建 TTS 插件
std::unique_ptr<TtsPlugin> create_tts_plugin(const TtsConfig& config);

} // namespace plugins
} // namespace qwen_thor
