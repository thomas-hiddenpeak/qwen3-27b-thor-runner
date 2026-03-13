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
#include <functional>

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
    std::string instruct;                   // VoiceDesign 模式的音色描述指令
    float       speed        = 1.0f;        // 语速 (0.5-2.0)
    std::string format       = "wav";       // 输出格式 (wav/pcm)
    std::string extra_args;                 // 额外 CLI 参数 (subprocess)
    std::string tmp_dir      = "tmp";       // 临时文件目录
    int         max_new_tokens = 4096;      // 最大生成 token 数
    // TTS 采样参数 (官方默认)
    float       tts_temperature   = 0.9f;
    int         tts_top_k         = 50;
    float       tts_top_p         = 1.0f;
    float       tts_rep_penalty   = 1.05f;

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
                                 const std::string& format = "wav",
                                 const std::string& instruct = "",
                                 const std::string& language = "") = 0;

    // Continue synthesis without resetting talker state (voice consistency)
    virtual TtsResult synthesize_continue(const std::string& text,
                                          const std::string& format = "pcm") = 0;

    // Streaming synthesis: calls pcm_callback with PCM chunks as they become available
    // Returns total PCM samples, 0 on failure. callback returns true to continue, false to abort.
    using PcmCallback = std::function<bool(const float* data, int num_samples)>;
    virtual int synthesize_streaming(const std::string& text,
                                     const std::string& voice,
                                     const std::string& instruct,
                                     PcmCallback callback,
                                     int chunk_frames = 24,
                                     const std::string& language = "") { return 0; }

    // Continue streaming: inject new text, preserve KV cache, decode in chunks
    virtual int continue_streaming(const std::string& text,
                                   PcmCallback callback,
                                   int chunk_frames = 24) { return 0; }

    virtual bool is_available() const = 0;
    virtual std::string name() const = 0;

    // Set TTS sampling parameters (temperature, top_k, top_p, repetition_penalty)
    virtual void set_sampling(float temperature, int top_k, float top_p, float rep_penalty) {}

    // Model info: type, available voices, sample rate
    struct ModelInfo {
        std::string model_type;                    // custom_voice, voice_design, voice_clone
        std::string default_instruct;              // VoiceDesign 默认音色描述
        std::vector<std::string> available_voices; // speaker names
        std::vector<std::string> available_languages; // language/dialect names
        std::unordered_map<std::string, std::string> speaker_dialects; // speaker → dialect (empty = standard)
        int sample_rate = 24000;
    };
    virtual ModelInfo model_info() const { return {}; }
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
                         const std::string& format = "wav",
                         const std::string& instruct = "",
                         const std::string& language = "") override;
    TtsResult synthesize_continue(const std::string& text,
                                  const std::string& format = "pcm") override;
    int synthesize_streaming(const std::string& text,
                              const std::string& voice,
                              const std::string& instruct,
                              PcmCallback callback,
                              int chunk_frames = 24,
                              const std::string& language = "") override;
    int continue_streaming(const std::string& text,
                           PcmCallback callback,
                           int chunk_frames = 24) override;
    bool is_available() const override;
    std::string name() const override { return "native-qwen3-tts"; }
    void set_sampling(float temperature, int top_k, float top_p, float rep_penalty) override;
    ModelInfo model_info() const override;

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
                         const std::string& format = "wav",
                         const std::string& instruct = "",
                         const std::string& language = "") override;
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
