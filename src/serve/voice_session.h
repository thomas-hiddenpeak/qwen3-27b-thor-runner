// voice_session.h — 统一 WebSocket 语音会话
//
// 合并 /v1/voice 和 /v1/realtime 的共享逻辑:
//   - VAD (RMS 能量 / FSMN 神经网络)
//   - 流式 ASR (partial + final)
//   - 说话人识别 + target_speaker 路由
//   - LLM + TTS 生产者-消费者管线
//
// 两种协议模式通过 ProtocolMode 区分消息名称和行为差异。

#pragma once

#include "../plugins/asr/vad_engine.h"

#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdint>

namespace qwen_thor {
namespace serve {

class ServeApp;  // forward

// ============================================================================
// VoiceSession — 统一 WebSocket 语音会话
// ============================================================================

class VoiceSession {
public:
    // 协议模式
    enum class ProtocolMode {
        VOICE,      // /v1/voice — 显式 stream.start/stop, 手动 barge-in
        REALTIME    // /v1/realtime — 始终 on, 自动 barge-in
    };

    // VAD 配置
    struct VadConfig {
        float energy_threshold = 0.01f;
        int silence_ms = 800;
        int min_speech_ms = 500;
        int max_duration_s = 30;
        float min_avg_energy = 0.008f;
    };

    // 目标说话人路由
    struct SpeakerRouting {
        bool enabled = false;
        std::string target_speaker;
        // other_mode: 非目标说话人的处理方式
        //   RESPOND_ALL — 忽略路由, 所有人都回复 (默认)
        //   PREFILL     — 注入上下文 "[Speaker说]: text", 不触发 LLM 生成
        //   IGNORE      — 完全忽略
        enum OtherMode { RESPOND_ALL, PREFILL, IGNORE } other_mode = RESPOND_ALL;
    };

    VoiceSession(ServeApp* app, ProtocolMode mode, int client_fd);
    ~VoiceSession();

    // 运行会话主循环 (阻塞直到连接关闭)
    void run();

private:
    // ---- 协议适配: 消息名映射 ----
    struct MsgNames {
        // Voice:                              Realtime:
        const char* session_created;        // session.created        | session.created
        const char* speech_started;         // stream.vad             | input.speech_started
        const char* speech_stopped;         // stream.vad             | input.speech_stopped
        const char* asr_partial;            // asr.partial            | input.transcription.partial
        const char* asr_result;             // asr                    | input.transcription
        const char* llm_start;              // llm.start              | response.started
        const char* llm_delta;              // llm.delta              | response.delta
        const char* llm_done;              // llm.done               | response.done
        const char* tts_start;              // tts.stream_start       | audio.started
        const char* tts_done;               // tts.done               | audio.done
        const char* error;                  // error                  | error
    };
    static MsgNames voice_msgs();
    static MsgNames realtime_msgs();

    // ---- WS I/O (线程安全) ----
    bool ws_send_text(const std::string& text);
    bool ws_send_binary(const uint8_t* data, size_t len);

    // ---- 事件处理 ----
    void handle_voice_binary(const uint8_t* data, size_t len);
    void handle_voice_text(const std::string& msg);

    // Voice 模式专用事件
    void on_config_event(const std::string& msg);
    void on_stream_start(const std::string& msg);
    void on_stream_stop();
    void on_audio_event(const std::string& msg);

    // ---- VAD ----
    void reset_vad_state();

    // ---- 工作线程管线 ----
    void start_voice_worker(std::vector<int16_t> audio, int sr, bool do_llm);
    void start_text_worker(const std::string& text);
    void run_voice_pipeline(std::vector<int16_t> audio, int sr, bool do_llm);
    void run_llm_tts(const std::string& user_text, bool is_voice_protocol);

    // ---- 说话人路由 ----
    enum class SpeakerAction { RESPOND, PREFILL, IGNORE };
    SpeakerAction evaluate_speaker(const std::string& speaker_name, float similarity);

    // ---- 录音 (Voice 模式专用) ----
    std::string save_recording_wav();

    // ---- 成员 ----
    ServeApp* app_;
    ProtocolMode mode_;
    int client_fd_;
    MsgNames msgs_;

    // 会话配置
    std::string voice_ = "serena";
    std::string tts_instruct_;
    std::string tts_language_;
    bool tts_enabled_ = true;
    bool asr_to_llm_ = true;
    int sample_rate_ = 16000;
    VadConfig vad_config_;
    SpeakerRouting speaker_routing_;

    // 连接/生成控制
    std::atomic<bool> conn_alive_{true};
    std::atomic<bool> generating_{false};
    std::atomic<bool> interrupted_{false};
    std::mutex send_mutex_;

    // 对话历史
    std::vector<std::pair<std::string, std::string>> chat_history_;

    // 音频缓冲 + VAD
    bool streaming_audio_ = false;  // Voice 模式: 需 stream.start 激活
    std::vector<int16_t> pcm_buffer_;
    int silence_samples_ = 0;
    bool speech_detected_ = false;
    double total_energy_sum_ = 0;
    int total_speech_samples_ = 0;

    // 流式 ASR
    static constexpr float STREAMING_ASR_CHUNK_S = 2.0f;
    float streaming_asr_next_s_ = STREAMING_ASR_CHUNK_S;

    // Voice 模式: audio.level 节流
    size_t gen_audio_sample_count_ = 0;

    // Voice 模式: 服务端录音
    std::vector<int16_t> recording_buffer_;
    int recording_sample_rate_ = 16000;
    std::chrono::steady_clock::time_point recording_start_time_;

    // FSMN VAD (per-session 副本, 当 ServeApp.vad_engine_ 已加载时启用)
    bool use_fsmn_vad_ = false;
    asr::VadEngine fsmn_vad_;

    // 工作线程
    std::thread worker_thread_;
};

} // namespace serve
} // namespace qwen_thor
