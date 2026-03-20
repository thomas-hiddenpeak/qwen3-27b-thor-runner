// voice_session.cpp — 统一 WebSocket 语音会话实现
//
// 合并 handle_websocket_voice 和 handle_websocket_realtime 的共享逻辑。

#include "voice_session.h"
#include "serve.h"
#include "ws_utils.h"
#include "../plugins/asr/audio_utils.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <queue>
#include <condition_variable>
#include <poll.h>
#include <sys/socket.h>

namespace qwen_thor {
namespace serve {

using namespace ws;

// ============================================================================
// 消息名映射
// ============================================================================

VoiceSession::MsgNames VoiceSession::voice_msgs() {
    return {
        "session.created",      // session_created
        "stream.vad",           // speech_started (VAD 触发)
        "stream.vad",           // speech_stopped (Voice 不区分 start/stop)
        "asr.partial",          // asr_partial
        "asr",                  // asr_result
        "llm.start",            // llm_start
        "llm.delta",            // llm_delta
        "llm.done",             // llm_done
        "tts.stream_start",     // tts_start
        "tts.done",             // tts_done
        "error",                // error
    };
}

VoiceSession::MsgNames VoiceSession::realtime_msgs() {
    return {
        "session.created",      // session_created
        "input.speech_started", // speech_started
        "input.speech_stopped", // speech_stopped
        "input.transcription.partial", // asr_partial
        "input.transcription",  // asr_result
        "response.started",     // llm_start
        "response.delta",       // llm_delta
        "response.done",        // llm_done
        "audio.started",        // tts_start
        "audio.done",           // tts_done
        "error",                // error
    };
}

// ============================================================================
// 构造 / 析构
// ============================================================================

VoiceSession::VoiceSession(ServeApp* app, ProtocolMode mode, int client_fd)
    : app_(app), mode_(mode), client_fd_(client_fd)
{
    msgs_ = (mode == ProtocolMode::VOICE) ? voice_msgs() : realtime_msgs();

    // 从 TTS plugin 初始化默认 instruct
    if (app_->tts_plugin_) {
        auto info = app_->tts_plugin_->model_info();
        tts_instruct_ = info.default_instruct;
    }

    // 协议差异: VAD 参数
    if (mode == ProtocolMode::VOICE) {
        vad_config_ = {0.01f, 800, 500, 30, 0.008f};
    } else {
        vad_config_ = {0.01f, 600, 300, 30, 0.008f};
    }
}

VoiceSession::~VoiceSession() {
    conn_alive_ = false;
    interrupted_ = true;
    if (worker_thread_.joinable()) worker_thread_.join();
}

// ============================================================================
// WS I/O (线程安全)
// ============================================================================

bool VoiceSession::ws_send_text(const std::string& text) {
    if (!conn_alive_) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (!ws::send_text(client_fd_, text)) {
        conn_alive_ = false;
        interrupted_ = true;
        return false;
    }
    return true;
}

bool VoiceSession::ws_send_binary(const uint8_t* data, size_t len) {
    if (!conn_alive_) return false;
    std::lock_guard<std::mutex> lock(send_mutex_);
    if (!ws::send_binary(client_fd_, data, len)) {
        conn_alive_ = false;
        interrupted_ = true;
        return false;
    }
    return true;
}

// ============================================================================
// VAD 状态重置
// ============================================================================

void VoiceSession::reset_vad_state() {
    pcm_buffer_.clear();
    pcm_buffer_.reserve(sample_rate_ * 10);
    silence_samples_ = 0;
    speech_detected_ = false;
    total_energy_sum_ = 0;
    total_speech_samples_ = 0;
    streaming_asr_next_s_ = STREAMING_ASR_CHUNK_S;
}

// ============================================================================
// 说话人路由评估
// ============================================================================

VoiceSession::SpeakerAction VoiceSession::evaluate_speaker(
    const std::string& speaker_name, float similarity) {
    if (!speaker_routing_.enabled || speaker_routing_.target_speaker.empty())
        return SpeakerAction::RESPOND;

    if (speaker_name == speaker_routing_.target_speaker)
        return SpeakerAction::RESPOND;

    switch (speaker_routing_.other_mode) {
        case SpeakerRouting::PREFILL: return SpeakerAction::PREFILL;
        case SpeakerRouting::IGNORE:  return SpeakerAction::IGNORE;
        default:                       return SpeakerAction::RESPOND;
    }
}

// ============================================================================
// 录音保存 (Voice 模式专用)
// ============================================================================

std::string VoiceSession::save_recording_wav() {
    if (recording_buffer_.empty()) return "";

    std::string dir = "tmp/recordings";
    std::filesystem::create_directories(dir);

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    char fname[64];
    snprintf(fname, sizeof(fname), "recording_%04d%02d%02d_%02d%02d%02d.wav",
             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
    std::string path = dir + "/" + fname;

    int sr = recording_sample_rate_;
    int data_bytes = (int)recording_buffer_.size() * 2;
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[WS] Failed to create recording file: %s\n", path.c_str());
        return "";
    }

    auto write16 = [&](uint16_t v) { f.write((char*)&v, 2); };
    auto write32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    f.write("RIFF", 4);
    write32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    write32(16);
    write16(1);          // PCM
    write16(1);          // mono
    write32(sr);
    write32(sr * 2);     // byte rate
    write16(2);
    write16(16);         // 16-bit
    f.write("data", 4);
    write32(data_bytes);
    f.write((const char*)recording_buffer_.data(), data_bytes);
    f.close();

    float dur_s = (float)recording_buffer_.size() / sr;
    float size_mb = data_bytes / 1048576.0f;
    fprintf(stderr, "[WS] Recording saved: %s (%.1fs, %.1f MB, %d Hz)\n",
            path.c_str(), dur_s, size_mb, sr);
    return path;
}

// ============================================================================
// 工作线程: 启动 voice / text pipeline
// ============================================================================

void VoiceSession::start_voice_worker(std::vector<int16_t> audio, int sr, bool do_llm) {
    if (generating_ || !conn_alive_) return;
    if (worker_thread_.joinable()) worker_thread_.join();
    generating_ = true;
    interrupted_ = false;
    worker_thread_ = std::thread([this, audio = std::move(audio), sr, do_llm]() {
        try {
            run_voice_pipeline(std::move(const_cast<std::vector<int16_t>&>(audio)), sr, do_llm);
        } catch (const std::exception& e) {
            fprintf(stderr, "[WS] EXCEPTION in voice worker: %s\n", e.what());
            ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                         "\",\"message\":\"Internal error\"}");
            generating_ = false;
        } catch (...) {
            fprintf(stderr, "[WS] UNKNOWN EXCEPTION in voice worker\n");
            ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                         "\",\"message\":\"Internal error\"}");
            generating_ = false;
        }
    });
}

void VoiceSession::start_text_worker(const std::string& text) {
    if (generating_ || !conn_alive_) return;
    if (worker_thread_.joinable()) worker_thread_.join();
    generating_ = true;
    interrupted_ = false;
    bool is_voice = (mode_ == ProtocolMode::VOICE);
    worker_thread_ = std::thread([this, text, is_voice]() {
        try {
            run_llm_tts(text, is_voice);
        } catch (const std::exception& e) {
            fprintf(stderr, "[WS] EXCEPTION in text worker: %s\n", e.what());
            ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                         "\",\"message\":\"Internal error\"}");
            generating_ = false;
        } catch (...) {
            fprintf(stderr, "[WS] UNKNOWN EXCEPTION in text worker\n");
            ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                         "\",\"message\":\"Internal error\"}");
            generating_ = false;
        }
    });
}

// ============================================================================
// Voice pipeline: PCM → ASR → speaker ID → LLM + TTS
// ============================================================================

void VoiceSession::run_voice_pipeline(std::vector<int16_t> audio, int sr, bool do_llm) {
    if (interrupted_) { generating_ = false; return; }

    // int16 → float
    std::vector<float> float_pcm(audio.size());
    for (size_t i = 0; i < audio.size(); i++)
        float_pcm[i] = audio[i] / 32768.0f;

    // --- ASR ---
    std::string asr_text;
    if (app_->asr_plugin_ && app_->asr_plugin_->is_available()) {
        if (mode_ == ProtocolMode::VOICE)
            ws_send_text("{\"type\":\"status\",\"stage\":\"asr\"}");

        auto result = app_->asr_plugin_->transcribe_pcm(
            float_pcm.data(), (int)float_pcm.size(), sr, "auto", true);

        if (result.error_code == 0 && !result.text.empty()) {
            asr_text = result.text;
        }
    }

    // ASR 结果过滤
    if (mode_ == ProtocolMode::VOICE) {
        // Voice 模式: 需要 ≥2 个有效字符
        if (!asr_text.empty()) {
            int char_count = 0;
            for (size_t i = 0; i < asr_text.size(); ) {
                unsigned char c = asr_text[i];
                int len = 1;
                if (c >= 0xC0) len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
                if (c > 0x20 && c != '.' && c != ',' && c != '!' && c != '?')
                    char_count++;
                i += len;
            }
            if (char_count < 2) {
                fprintf(stderr, "[WS] ASR filtered (char_count=%d): \"%s\"\n",
                        char_count, asr_text.c_str());
                asr_text.clear();
            }
        }
    } else {
        // Realtime 模式: 仅去除空白
        while (!asr_text.empty() && (asr_text.front() == ' ' || asr_text.front() == '\n'))
            asr_text.erase(asr_text.begin());
        while (!asr_text.empty() && (asr_text.back() == ' ' || asr_text.back() == '\n'))
            asr_text.pop_back();
    }

    if (asr_text.empty()) {
        if (mode_ == ProtocolMode::VOICE) {
            fprintf(stderr, "[WS] No valid ASR result, silently resetting\n");
        } else {
            ws_send_text("{\"type\":\"" + std::string(msgs_.asr_result) + "\",\"text\":\"\"}");
        }
        generating_ = false;
        return;
    }

    // --- 说话人识别 ---
    std::string speaker_json;
    std::string speaker_name;
    float speaker_sim = 0;
    if (app_->speaker_encoder_ && app_->speaker_manager_.speaker_count() > 0) {
        auto spk = app_->identify_speaker(float_pcm.data(), (int)float_pcm.size(), sr);
        if (spk.speaker_id >= 0 && spk.similarity >= 0.65f) {
            speaker_name = spk.name;
            speaker_sim = spk.similarity;
            speaker_json = ",\"speaker\":\"" + ws::json_escape(spk.name) +
                           "\",\"speaker_id\":" + std::to_string(spk.speaker_id) +
                           ",\"speaker_similarity\":" + std::to_string(spk.similarity);
            fprintf(stderr, "[WS] Speaker identified: %s (sim=%.3f)\n",
                    spk.name.c_str(), spk.similarity);
        }
    }

    // --- target_speaker 路由 (P0) ---
    SpeakerAction action = evaluate_speaker(speaker_name, speaker_sim);
    if (action == SpeakerAction::IGNORE) {
        fprintf(stderr, "[WS] Speaker '%s' ignored by routing\n", speaker_name.c_str());
        ws_send_text("{\"type\":\"" + std::string(msgs_.asr_result) +
                     "\",\"text\":\"" + ws::json_escape(asr_text) + "\"" +
                     speaker_json + ",\"routed\":\"ignore\"}");
        generating_ = false;
        return;
    }
    if (action == SpeakerAction::PREFILL) {
        fprintf(stderr, "[WS] Speaker '%s' → prefill context\n", speaker_name.c_str());
        // 注入上下文, 不触发 LLM 生成
        std::string ctx = "[" + speaker_name + "说]: " + asr_text;
        chat_history_.push_back({"system", ctx});
        ws_send_text("{\"type\":\"" + std::string(msgs_.asr_result) +
                     "\",\"text\":\"" + ws::json_escape(asr_text) + "\"" +
                     speaker_json + ",\"routed\":\"prefill\"}");
        generating_ = false;
        return;
    }

    // --- 发送 ASR 结果 ---
    ws_send_text("{\"type\":\"" + std::string(msgs_.asr_result) +
                 "\",\"text\":\"" + ws::json_escape(asr_text) + "\"" +
                 speaker_json + "}");
    if (interrupted_) { generating_ = false; return; }

    // ASR→LLM 开关
    if (!do_llm) {
        ws_send_text("{\"type\":\"asr.done\"}");
        generating_ = false;
        return;
    }

    // --- LLM + TTS ---
    bool is_voice = (mode_ == ProtocolMode::VOICE);
    run_llm_tts(asr_text, is_voice);
}

// ============================================================================
// LLM + TTS 生产者-消费者管线
// ============================================================================

void VoiceSession::run_llm_tts(const std::string& user_text, bool is_voice_protocol) {
    const auto& tok = app_->backend_.tokenizer();
    if (!tok.is_loaded()) {
        ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                     "\",\"message\":\"Tokenizer not loaded\"}");
        generating_ = false;
        return;
    }

    chat_history_.push_back({"user", user_text});

    // 保留最近 N 轮
    const size_t max_messages = (size_t)app_->config_.voice_max_turns * 2;
    while (chat_history_.size() > max_messages)
        chat_history_.erase(chat_history_.begin());

    // 构建 messages
    std::vector<std::pair<std::string, std::string>> messages;
    const std::string& voice_prompt = app_->config_.voice_system_prompt.empty()
        ? std::string(ws::default_voice_system_prompt()) : app_->config_.voice_system_prompt;
    messages.push_back({"system", voice_prompt});
    for (auto& [role, content] : chat_history_)
        messages.push_back({role, content});

    auto prompt_tokens = tok.apply_chat_template(messages, true, false);
    int prompt_count = (int)prompt_tokens.size();

    InferRequest infer_req;
    infer_req.request_id     = app_->next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens = app_->config_.voice_max_output_tokens;
    infer_req.temperature    = 0.7f;
    infer_req.top_p          = 0.8f;
    infer_req.top_k          = 20;
    infer_req.presence_penalty = 1.5f;
    infer_req.frequency_penalty = 0.5f;
    infer_req.stream         = true;

    app_->register_request(infer_req.request_id);

    if (!app_->backend_.submit(infer_req)) {
        app_->unregister_request(infer_req.request_id);
        ws_send_text("{\"type\":\"" + std::string(msgs_.error) +
                     "\",\"message\":\"Request queue full\"}");
        generating_ = false;
        return;
    }

    bool do_tts = tts_enabled_ && app_->tts_plugin_ && app_->tts_plugin_->is_available();

    ws_send_text("{\"type\":\"" + std::string(msgs_.llm_start) + "\"}");

    // ---- TTS 生产者-消费者 ----
    std::queue<std::pair<std::string, std::string>> tts_queue;
    std::mutex tts_mutex;
    std::condition_variable tts_cv;
    bool tts_done_flag = false;
    std::atomic<int> tts_segment_idx{0};

    if (do_tts) {
        if (is_voice_protocol) {
            ws_send_text("{\"type\":\"" + std::string(msgs_.tts_start) +
                         "\",\"sample_rate\":24000,\"format\":\"pcm16\"}");
        } else {
            ws_send_text("{\"type\":\"" + std::string(msgs_.tts_start) + "\"}");
        }
    }

    auto* tts_raw = do_tts ? app_->tts_plugin_.get() : nullptr;
    constexpr size_t AUDIO_CHUNK_SAMPLES = 4800;  // 200ms @ 24kHz (Realtime 模式分块)

    std::thread tts_thread;
    if (do_tts) {
        std::string instruct_copy = tts_instruct_;
        std::string voice_copy = voice_;
        std::string lang_copy = tts_language_;
        tts_thread = std::thread([&, tts_raw, instruct_copy, voice_copy, lang_copy,
                                  is_voice_protocol]() {
            while (true) {
                std::string sentence;
                std::string sent_instruct;
                {
                    std::unique_lock<std::mutex> lock(tts_mutex);
                    tts_cv.wait(lock, [&]{ return !tts_queue.empty() || tts_done_flag; });
                    if (tts_queue.empty() && tts_done_flag) break;
                    if (tts_queue.empty()) continue;
                    auto& front = tts_queue.front();
                    sentence = std::move(front.first);
                    sent_instruct = std::move(front.second);
                    tts_queue.pop();
                }

                if (interrupted_) break;

                // 合并 voice design base instruct 和 per-sentence emotion
                std::string use_instruct;
                if (!sent_instruct.empty() && !instruct_copy.empty()) {
                    use_instruct = instruct_copy + "，" + sent_instruct;
                } else if (!sent_instruct.empty()) {
                    use_instruct = sent_instruct;
                } else {
                    use_instruct = instruct_copy;
                }

                fprintf(stderr, "[TTS] Synthesize #%d [%s]: %.60s...\n",
                        tts_segment_idx.load() + 1,
                        use_instruct.empty() ? "default" : use_instruct.c_str(),
                        sentence.c_str());

                tts_raw->synthesize_streaming(sentence, voice_copy, use_instruct,
                    [&, is_voice_protocol](const float* data, int num_samples) -> bool {
                        if (interrupted_) return false;
                        std::vector<int16_t> pcm16(num_samples);
                        for (int i = 0; i < num_samples; i++) {
                            float v = std::max(-1.0f, std::min(1.0f, data[i]));
                            pcm16[i] = (int16_t)(v * 32767.0f);
                        }
                        if (is_voice_protocol) {
                            // Voice: 直接发送全部
                            return ws_send_binary(
                                reinterpret_cast<const uint8_t*>(pcm16.data()),
                                pcm16.size() * sizeof(int16_t));
                        } else {
                            // Realtime: 分块发送 (200ms chunks)
                            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(pcm16.data());
                            size_t remaining = pcm16.size() * sizeof(int16_t);
                            const size_t chunk_bytes = AUDIO_CHUNK_SAMPLES * 2;
                            while (remaining > 0 && !interrupted_) {
                                size_t send_size = std::min(remaining, chunk_bytes);
                                ws_send_binary(ptr, send_size);
                                ptr += send_size;
                                remaining -= send_size;
                            }
                            return !interrupted_;
                        }
                    }, 8, is_voice_protocol ? lang_copy : "");
                tts_segment_idx++;
            }
        });
    }

    // push_tts: 推送句子到 TTS 队列
    std::string last_emotion;
    auto push_tts = [&](const std::string& sentence) {
        if (!do_tts || sentence.empty()) return;

        std::string clean = sentence;
        // Realtime 模式: strip markdown
        if (!is_voice_protocol) {
            std::string stripped;
            stripped.reserve(clean.size());
            for (size_t i = 0; i < clean.size(); i++) {
                char c = clean[i];
                if (c == '*' || c == '#' || c == '`' || c == '$') continue;
                if (c >= '0' && c <= '9' && i + 1 < clean.size() && clean[i+1] == '.') {
                    i++;
                    if (i + 1 < clean.size() && clean[i+1] == ' ') i++;
                    continue;
                }
                stripped += c;
            }
            clean = std::move(stripped);
            // Trim
            while (!clean.empty() && (clean.front() == ' ' || clean.front() == '\n'))
                clean.erase(clean.begin());
            while (!clean.empty() && (clean.back() == ' ' || clean.back() == '\n'))
                clean.pop_back();
            if (clean.size() < 6) return;
        }

        auto [text_part, emotion] = ws::extract_tts_instruct(clean);
        if (!is_voice_protocol && text_part.size() < 6) return;
        if (text_part.empty()) return;

        if (!emotion.empty()) last_emotion = emotion;

        std::string formatted_instruct;
        if (!last_emotion.empty())
            formatted_instruct = "用" + last_emotion + "的语气说";

        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_queue.push({std::move(text_part), std::move(formatted_instruct)});
        }
        tts_cv.notify_one();
    };

    // ---- LLM 流式生成 ----
    std::string full_response;
    std::string pending_sentence;

    int comp_toks = app_->poll_tokens(infer_req.request_id,
        [&](const std::string& piece) {
            if (interrupted_) return;
            full_response += piece;
            pending_sentence += piece;

            ws_send_text("{\"type\":\"" + std::string(
                is_voice_protocol ? msgs_.llm_delta : msgs_.llm_delta) +
                "\",\"delta\":\"" + ws::json_escape(piece) + "\"}");

            if (!do_tts) return;

            // 句尾检测
            size_t pos = pending_sentence.size();
            if (pos == 0) return;
            size_t last_start = pos - 1;
            while (last_start > 0 && (pending_sentence[last_start] & 0xC0) == 0x80)
                last_start--;
            std::string last_ch = pending_sentence.substr(last_start);

            bool is_sent_end = ws::is_sentence_end_punct(last_ch) &&
                               pending_sentence.size() >= 15;
            bool is_clause = (pending_sentence.size() > 100 &&
                             (last_ch == "，" || last_ch == "," ||
                              last_ch == "；" || last_ch == ";" ||
                              last_ch == "：" || last_ch == ":"));

            if (is_sent_end || is_clause) {
                std::string sentence = pending_sentence;
                while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
                    sentence.pop_back();
                if (!sentence.empty()) {
                    fprintf(stderr, "[LLM] Sentence split (%zu bytes): %.60s...\n",
                            sentence.size(), sentence.c_str());
                    push_tts(sentence);
                }
                pending_sentence.clear();
            }
        },
        app_->config_.timeout_s,
        false, {}, {}, {}, nullptr, &interrupted_, nullptr
    );

    // 推送剩余文本
    if (!interrupted_ && !pending_sentence.empty()) {
        std::string sentence = pending_sentence;
        while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
            sentence.pop_back();
        if (!sentence.empty()) {
            fprintf(stderr, "[LLM] Final fragment (%zu bytes): %.60s...\n",
                    sentence.size(), sentence.c_str());
            push_tts(sentence);
        }
    }

    // 发送 LLM 完成
    if (is_voice_protocol) {
        if (!interrupted_) {
            ws_send_text("{\"type\":\"" + std::string(msgs_.llm_done) +
                         "\",\"text\":\"" + ws::json_escape(full_response) +
                         "\",\"prompt_tokens\":" + std::to_string(prompt_count) +
                         ",\"completion_tokens\":" + std::to_string(comp_toks) + "}");
        }
    } else {
        bool was_interrupted = interrupted_.load();
        ws_send_text("{\"type\":\"" + std::string(msgs_.llm_done) +
                     "\",\"text\":\"" + ws::json_escape(full_response) +
                     "\",\"interrupted\":" + (was_interrupted ? "true" : "false") + "}");
    }

    chat_history_.push_back({"assistant", full_response});

    // 等待 TTS 完成
    if (do_tts && tts_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_done_flag = true;
            tts_cv.notify_one();
        }
        tts_thread.join();

        if (is_voice_protocol) {
            if (!interrupted_) {
                ws_send_text("{\"type\":\"" + std::string(msgs_.tts_done) +
                             "\",\"segments\":" + std::to_string(tts_segment_idx.load()) + "}");
            }
        } else {
            ws_send_text("{\"type\":\"" + std::string(msgs_.tts_done) + "\"}");
        }
    }

    generating_ = false;
}

// ============================================================================
// Voice 模式: config 事件
// ============================================================================

void VoiceSession::on_config_event(const std::string& msg) {
    std::string v = ws::json_get_string(msg, "voice");
    if (!v.empty()) voice_ = v;

    if (msg.find("\"tts\"") != std::string::npos)
        tts_enabled_ = ws::json_get_bool(msg, "tts", true);

    if (msg.find("\"tts_instruct\"") != std::string::npos) {
        std::string inst = ws::json_get_string(msg, "tts_instruct");
        if (inst.empty() && app_->tts_plugin_) {
            tts_instruct_ = app_->tts_plugin_->model_info().default_instruct;
        } else {
            tts_instruct_ = inst;
        }
    }

    if (msg.find("\"system_prompt\"") != std::string::npos) {
        std::string sp = ws::json_get_string(msg, "system_prompt");
        if (sp.empty()) {
            app_->config_.voice_system_prompt = app_->config_.voice_system_prompt_default;
        } else {
            app_->config_.voice_system_prompt = sp;
        }
    }

    std::string lang = ws::json_get_string(msg, "tts_language");
    if (msg.find("\"tts_language\"") != std::string::npos) tts_language_ = lang;

    if (app_->tts_plugin_ && msg.find("\"tts_temperature\"") != std::string::npos) {
        float tts_temp = (float)ws::json_get_number(msg, "tts_temperature", 0.9);
        int tts_topk = ws::json_get_int(msg, "tts_top_k", 50);
        float tts_topp = (float)ws::json_get_number(msg, "tts_top_p", 1.0);
        float tts_rep = (float)ws::json_get_number(msg, "tts_rep_penalty", 1.05);
        app_->tts_plugin_->set_sampling(tts_temp, tts_topk, tts_topp, tts_rep);
    }

    if (msg.find("\"voice_max_turns\"") != std::string::npos) {
        int vmt = ws::json_get_int(msg, "voice_max_turns", app_->config_.voice_max_turns);
        if (vmt >= 1 && vmt <= 100) app_->config_.voice_max_turns = vmt;
    }
    if (msg.find("\"voice_max_output_tokens\"") != std::string::npos) {
        int vmot = ws::json_get_int(msg, "voice_max_output_tokens",
                                     app_->config_.voice_max_output_tokens);
        if (vmot >= 10 && vmot <= 4096) app_->config_.voice_max_output_tokens = vmot;
    }

    if (msg.find("\"asr_to_llm\"") != std::string::npos)
        asr_to_llm_ = ws::json_get_bool(msg, "asr_to_llm", true);

    // target_speaker (P0)
    if (msg.find("\"target_speaker\"") != std::string::npos) {
        std::string ts = ws::json_get_string(msg, "target_speaker");
        if (ts.empty()) {
            speaker_routing_.enabled = false;
            speaker_routing_.target_speaker.clear();
        } else {
            speaker_routing_.enabled = true;
            speaker_routing_.target_speaker = ts;
        }
    }
    if (msg.find("\"other_speaker_mode\"") != std::string::npos) {
        std::string osm = ws::json_get_string(msg, "other_speaker_mode");
        if (osm == "prefill") speaker_routing_.other_mode = SpeakerRouting::PREFILL;
        else if (osm == "ignore") speaker_routing_.other_mode = SpeakerRouting::IGNORE;
        else speaker_routing_.other_mode = SpeakerRouting::RESPOND_ALL;
    }

    fprintf(stderr, "[WS] config: voice=%s tts=%s lang=%s turns=%d tokens=%d "
            "asr_to_llm=%d target_speaker=%s fd=%d\n",
            voice_.c_str(), tts_enabled_ ? "on" : "off",
            tts_language_.empty() ? "auto" : tts_language_.c_str(),
            app_->config_.voice_max_turns, app_->config_.voice_max_output_tokens,
            (int)asr_to_llm_,
            speaker_routing_.target_speaker.empty() ? "(none)" :
                speaker_routing_.target_speaker.c_str(),
            client_fd_);

    // 返回当前配置
    if (mode_ == ProtocolMode::VOICE) {
        const std::string& sp = app_->config_.voice_system_prompt.empty()
            ? std::string(ws::default_voice_system_prompt()) : app_->config_.voice_system_prompt;
        ws_send_text("{\"type\":\"config.updated\",\"system_prompt\":\"" +
            ws::json_escape(sp) + "\""
            ",\"voice_max_turns\":" + std::to_string(app_->config_.voice_max_turns) +
            ",\"voice_max_output_tokens\":" + std::to_string(app_->config_.voice_max_output_tokens) + "}");
    }
}

// ============================================================================
// Voice 模式: stream.start
// ============================================================================

void VoiceSession::on_stream_start(const std::string& msg) {
    streaming_audio_ = true;
    reset_vad_state();
    sample_rate_ = ws::json_get_int(msg, "sample_rate", 16000);
    if (sample_rate_ < 8000) sample_rate_ = 8000;
    if (sample_rate_ > 48000) sample_rate_ = 48000;

    // 开始录音
    recording_buffer_.clear();
    recording_buffer_.reserve(sample_rate_ * 60);
    recording_sample_rate_ = sample_rate_;
    recording_start_time_ = std::chrono::steady_clock::now();
    ws_send_text("{\"type\":\"stream.started\"}");
    fprintf(stderr, "[WS] Audio stream started, rate=%d fd=%d\n", sample_rate_, client_fd_);
}

// ============================================================================
// Voice 模式: stream.stop
// ============================================================================

void VoiceSession::on_stream_stop() {
    if (!streaming_audio_) {
        ws_send_text("{\"type\":\"stream.stopped\"}");
        return;
    }
    streaming_audio_ = false;

    float audio_dur = (float)pcm_buffer_.size() / sample_rate_;
    if (pcm_buffer_.size() >= (size_t)(sample_rate_ * vad_config_.min_speech_ms / 1000)) {
        float avg_rms = std::sqrt((float)(total_energy_sum_ /
            std::max((size_t)1, pcm_buffer_.size())));
        if (avg_rms < vad_config_.min_avg_energy) {
            fprintf(stderr, "[WS] Stream stopped: rejected (avg_rms=%.4f too quiet)\n", avg_rms);
            ws_send_text("{\"type\":\"error\",\"message\":\"未检测到语音\"}");
            reset_vad_state();
            ws_send_text("{\"type\":\"stream.stopped\"}");
            goto save_rec;
        }

        fprintf(stderr, "[WS] Stream stopped manually: %.1fs audio, avg_rms=%.4f\n",
                audio_dur, avg_rms);

        auto audio_copy = std::move(pcm_buffer_);
        int sr = sample_rate_;
        reset_vad_state();
        start_voice_worker(std::move(audio_copy), sr, asr_to_llm_);
    } else {
        fprintf(stderr, "[WS] Stream stopped, too short (%.1fs)\n", audio_dur);
        ws_send_text("{\"type\":\"error\",\"message\":\"录音太短\"}");
        reset_vad_state();
    }

    ws_send_text("{\"type\":\"stream.stopped\"}");

save_rec:
    {
        std::string rec_path = save_recording_wav();
        if (!rec_path.empty()) {
            ws_send_text("{\"type\":\"recording.saved\",\"path\":\"" +
                         ws::json_escape(rec_path) + "\"}");
        }
        recording_buffer_.clear();
    }
}

// ============================================================================
// Voice 模式: base64 audio event
// ============================================================================

void VoiceSession::on_audio_event(const std::string& msg) {
    if (generating_) return;
    std::string audio_b64 = ws::json_get_string(msg, "data");
    if (audio_b64.empty()) return;

    if (!app_->asr_plugin_ || !app_->asr_plugin_->is_available()) {
        ws_send_text("{\"type\":\"error\",\"message\":\"ASR not available\"}");
        return;
    }

    if (worker_thread_.joinable()) worker_thread_.join();
    generating_ = true;
    interrupted_ = false;
    worker_thread_ = std::thread([this, audio_b64 = std::move(audio_b64)]() {
        try {
            auto audio_bytes = ws::base64_decode(audio_b64);
            if (audio_bytes.empty()) {
                ws_send_text("{\"type\":\"error\",\"message\":\"Invalid audio data\"}");
                generating_ = false;
                return;
            }

            ws_send_text("{\"type\":\"status\",\"stage\":\"asr\"}");
            auto result = app_->asr_plugin_->transcribe_memory(
                audio_bytes.data(), audio_bytes.size(), "auto");

            if (result.error_code != 0 || result.text.empty()) {
                ws_send_text("{\"type\":\"error\",\"message\":\"ASR failed: " +
                             ws::json_escape(result.error_message) + "\"}");
                generating_ = false;
                return;
            }

            ws_send_text("{\"type\":\"asr\",\"text\":\"" +
                         ws::json_escape(result.text) + "\"}");
            if (interrupted_) { generating_ = false; return; }

            run_llm_tts(result.text, true);
        } catch (const std::exception& e) {
            fprintf(stderr, "[WS] EXCEPTION in audio worker: %s\n", e.what());
            ws_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating_ = false;
        } catch (...) {
            fprintf(stderr, "[WS] UNKNOWN EXCEPTION in audio worker\n");
            ws_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating_ = false;
        }
    });
}

// ============================================================================
// Binary 帧处理 (统一 Voice/Realtime)
// ============================================================================

void VoiceSession::handle_voice_binary(const uint8_t* data, size_t len) {
    size_t num_samples = len / 2;
    if (num_samples == 0) return;
    const int16_t* samples = reinterpret_cast<const int16_t*>(data);

    // Voice 模式: 需要 streaming_audio_ 开启
    if (mode_ == ProtocolMode::VOICE && !streaming_audio_) return;

    // 录音 (Voice 模式)
    if (mode_ == ProtocolMode::VOICE) {
        recording_buffer_.insert(recording_buffer_.end(), samples, samples + num_samples);
    }

    // 计算帧能量
    double energy_sum = 0;
    for (size_t i = 0; i < num_samples; i++) {
        float s = samples[i] / 32768.0f;
        energy_sum += s * s;
    }
    float rms = std::sqrt((float)(energy_sum / num_samples));

    // --- generating 期间 ---
    if (generating_) {
        if (mode_ == ProtocolMode::VOICE) {
            // 发 audio.level (节流: 每 100ms)
            size_t prev_count = gen_audio_sample_count_;
            gen_audio_sample_count_ += num_samples;
            if (gen_audio_sample_count_ / 1600 > prev_count / 1600) {
                char level_buf[64];
                snprintf(level_buf, sizeof(level_buf),
                         "{\"type\":\"audio.level\",\"rms\":%.4f}", rms);
                ws_send_text(level_buf);
            }
        } else {
            // Realtime: 用户说话 → 自动打断
            if (rms > vad_config_.energy_threshold * 3) {
                interrupted_ = true;
                if (worker_thread_.joinable()) worker_thread_.join();
                generating_ = false;
                interrupted_ = false;
                ws_send_text("{\"type\":\"response.done\",\"text\":\"\",\"interrupted\":true}");
                ws_send_text("{\"type\":\"audio.done\"}");
            }
        }
        return;
    }

    // Voice 模式: generating 结束后重置计数
    if (mode_ == ProtocolMode::VOICE)
        gen_audio_sample_count_ = 0;

    // --- VAD 处理 ---
    if (mode_ == ProtocolMode::REALTIME) {
        // Realtime: 检测到语音时清空缓冲重新开始
        if (rms > vad_config_.energy_threshold) {
            if (!speech_detected_) {
                speech_detected_ = true;
                pcm_buffer_.clear();
                total_energy_sum_ = 0;
                total_speech_samples_ = 0;
                silence_samples_ = 0;
                streaming_asr_next_s_ = STREAMING_ASR_CHUNK_S;
                ws_send_text("{\"type\":\"input.speech_started\"}");
            }
            silence_samples_ = 0;
        } else if (speech_detected_) {
            silence_samples_ += (int)num_samples;
        }

        if (!speech_detected_) return;  // 未检测到语音, 丢弃

        pcm_buffer_.insert(pcm_buffer_.end(), samples, samples + num_samples);
        total_energy_sum_ += energy_sum;
        total_speech_samples_ += (int)num_samples;
    } else {
        // Voice 模式: 持续累积
        size_t prev_size = pcm_buffer_.size();
        pcm_buffer_.insert(pcm_buffer_.end(), samples, samples + num_samples);

        if (rms > vad_config_.energy_threshold) {
            speech_detected_ = true;
            silence_samples_ = 0;
            total_speech_samples_ += (int)num_samples;
        } else {
            silence_samples_ += (int)num_samples;
        }
        total_energy_sum_ += energy_sum;

        // audio.level (节流)
        if (pcm_buffer_.size() / 1600 > prev_size / 1600) {
            char level_buf[64];
            snprintf(level_buf, sizeof(level_buf),
                     "{\"type\":\"audio.level\",\"rms\":%.4f}", rms);
            ws_send_text(level_buf);
        }
    }

    // --- 流式 ASR ---
    float total_s = (float)pcm_buffer_.size() / sample_rate_;
    if (speech_detected_ && total_s >= streaming_asr_next_s_
        && app_->asr_plugin_ && app_->asr_plugin_->is_available()) {
        std::vector<float> float_pcm(pcm_buffer_.size());
        for (size_t i = 0; i < pcm_buffer_.size(); i++)
            float_pcm[i] = pcm_buffer_[i] / 32768.0f;

        auto partial = app_->asr_plugin_->transcribe_pcm(
            float_pcm.data(), (int)float_pcm.size(), sample_rate_, "auto", true);

        if (partial.error_code == 0 && !partial.text.empty()) {
            fprintf(stderr, "[WS] Streaming ASR (%.1fs): \"%s\"\n",
                    total_s, partial.text.substr(0, 80).c_str());
            ws_send_text("{\"type\":\"" + std::string(msgs_.asr_partial) +
                         "\",\"text\":\"" + ws::json_escape(partial.text) + "\"}");
        }
        streaming_asr_next_s_ = total_s + STREAMING_ASR_CHUNK_S;
    }

    // --- VAD 触发检测 ---
    float silence_ms = (float)silence_samples_ * 1000.0f / sample_rate_;
    bool vad_triggered = speech_detected_ &&
                         silence_ms >= vad_config_.silence_ms &&
                         total_s >= (vad_config_.min_speech_ms / 1000.0f);
    bool timeout = total_s >= vad_config_.max_duration_s;

    if (!vad_triggered && !timeout) return;

    // VAD 或超时触发
    if (mode_ == ProtocolMode::VOICE) {
        ws_send_text("{\"type\":\"stream.vad\"}");
    } else {
        speech_detected_ = false;
        ws_send_text("{\"type\":\"input.speech_stopped\"}");
    }

    // 检查平均能量
    float avg_rms = (pcm_buffer_.size() > 0)
        ? std::sqrt((float)(total_energy_sum_ / pcm_buffer_.size())) : 0.0f;

    if (avg_rms < vad_config_.min_avg_energy) {
        fprintf(stderr, "[WS] Rejected audio: avg_rms=%.4f (too quiet)\n", avg_rms);
        reset_vad_state();
        return;
    }

    if (generating_) {
        // Voice 模式: 上一轮还在生成, 丢弃
        fprintf(stderr, "[WS] VAD during generation, dropping segment (%.1fs)\n",
                (float)pcm_buffer_.size() / sample_rate_);
        reset_vad_state();
        return;
    }

    // 去掉尾部静音 (Voice 模式)
    if (mode_ == ProtocolMode::VOICE) {
        int trim_samples = std::min(silence_samples_, (int)pcm_buffer_.size());
        if (trim_samples > sample_rate_ / 10)
            pcm_buffer_.resize(pcm_buffer_.size() - trim_samples + sample_rate_ / 10);
    }

    float audio_dur = (float)pcm_buffer_.size() / sample_rate_;
    float speech_ratio = (float)total_speech_samples_ / std::max(1, (int)pcm_buffer_.size());
    fprintf(stderr, "[WS] VAD: %.1fs audio, avg_rms=%.4f speech=%.0f%%\n",
            audio_dur, avg_rms, speech_ratio * 100);

    auto audio_copy = std::move(pcm_buffer_);
    int sr = sample_rate_;
    reset_vad_state();
    start_voice_worker(std::move(audio_copy), sr, asr_to_llm_);
}

// ============================================================================
// Text 帧处理 (统一 Voice/Realtime)
// ============================================================================

void VoiceSession::handle_voice_text(const std::string& msg) {
    std::string event_type;

    if (mode_ == ProtocolMode::VOICE) {
        event_type = ws::json_get_string(msg, "type");
    } else {
        // Realtime 用全局 json_get_string 代替本地对象
        event_type = ws::json_get_string(msg, "type");
    }

    // ---- Voice 模式专用事件 ----
    if (mode_ == ProtocolMode::VOICE) {
        if (event_type == "config") {
            on_config_event(msg);
        } else if (event_type == "chat") {
            if (!generating_) {
                std::string text = ws::json_get_string(msg, "text");
                if (!text.empty()) {
                    fprintf(stderr, "[WS] chat request: voice=%s tts=%s fd=%d\n",
                            voice_.c_str(), tts_enabled_ ? "on" : "off", client_fd_);
                    start_text_worker(text);
                }
            }
        } else if (event_type == "stream.start") {
            on_stream_start(msg);
        } else if (event_type == "stream.stop") {
            on_stream_stop();
        } else if (event_type == "audio") {
            on_audio_event(msg);
        } else if (event_type == "interrupt" || event_type == "tts.stop") {
            if (generating_) {
                interrupted_ = true;
                fprintf(stderr, "[WS] Client interrupt fd=%d\n", client_fd_);
            }
        } else if (event_type == "clear") {
            if (!generating_) {
                chat_history_.clear();
                ws_send_text("{\"type\":\"history.cleared\"}");
            }
        }
        return;
    }

    // ---- Realtime 模式事件 ----
    if (event_type == "session.update") {
        // 简化版 config
        std::string v = ws::json_get_string(msg, "voice");
        if (!v.empty()) voice_ = v;
        std::string inst = ws::json_get_string(msg, "tts_instruct");
        if (!inst.empty()) tts_instruct_ = inst;
        int sr = ws::json_get_int(msg, "sample_rate");
        if (sr > 0) sample_rate_ = sr;
        if (msg.find("\"asr_to_llm\"") != std::string::npos)
            asr_to_llm_ = ws::json_get_bool(msg, "asr_to_llm", true);
        // target_speaker (P0)
        if (msg.find("\"target_speaker\"") != std::string::npos) {
            std::string ts = ws::json_get_string(msg, "target_speaker");
            if (ts.empty()) {
                speaker_routing_.enabled = false;
                speaker_routing_.target_speaker.clear();
            } else {
                speaker_routing_.enabled = true;
                speaker_routing_.target_speaker = ts;
            }
        }
        if (msg.find("\"other_speaker_mode\"") != std::string::npos) {
            std::string osm = ws::json_get_string(msg, "other_speaker_mode");
            if (osm == "prefill") speaker_routing_.other_mode = SpeakerRouting::PREFILL;
            else if (osm == "ignore") speaker_routing_.other_mode = SpeakerRouting::IGNORE;
            else speaker_routing_.other_mode = SpeakerRouting::RESPOND_ALL;
        }
        fprintf(stderr, "[RT] Config: voice=%s instruct=%s sample_rate=%d asr_to_llm=%d "
                "target_speaker=%s\n",
                voice_.c_str(),
                tts_instruct_.empty() ? "(default)" : tts_instruct_.c_str(),
                sample_rate_, (int)asr_to_llm_,
                speaker_routing_.target_speaker.empty() ? "(none)" :
                    speaker_routing_.target_speaker.c_str());
    }
    else if (event_type == "text") {
        std::string text = ws::json_get_string(msg, "text");
        if (!text.empty() && !generating_) {
            start_text_worker(text);
        }
    }
    else if (event_type == "interrupt") {
        if (generating_) {
            interrupted_ = true;
        }
    }
}

// ============================================================================
// 主循环
// ============================================================================

void VoiceSession::run() {
    const char* log_prefix = (mode_ == ProtocolMode::VOICE) ? "[WS]" : "[RT]";

    // Voice 模式: 设置 socket 超时
    if (mode_ == ProtocolMode::VOICE) {
        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(client_fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        setsockopt(client_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    ws_send_text("{\"type\":\"session.created\"}");

    auto last_activity = std::chrono::steady_clock::now();
    constexpr int WS_PING_INTERVAL_S = 15;

    while (app_->running_ && conn_alive_) {
        struct pollfd pfd;
        pfd.fd = client_fd_;
        pfd.events = POLLIN;
        int ret = ::poll(&pfd, 1, 100);

        if (ret < 0) break;
        if (ret == 0) {
            // Voice 模式: 心跳 ping
            if (mode_ == ProtocolMode::VOICE) {
                auto now = std::chrono::steady_clock::now();
                float idle_s = std::chrono::duration<float>(now - last_activity).count();
                if (idle_s >= WS_PING_INTERVAL_S) {
                    std::lock_guard<std::mutex> lock(send_mutex_);
                    if (!ws::send_frame(client_fd_, ws::OP_PING, nullptr, 0)) {
                        conn_alive_ = false;
                        break;
                    }
                    last_activity = now;
                }
            }
            continue;
        }
        if (!(pfd.revents & POLLIN)) break;

        last_activity = std::chrono::steady_clock::now();

        uint8_t opcode;
        std::vector<uint8_t> payload;
        if (!ws::recv_frame(client_fd_, opcode, payload)) {
            conn_alive_ = false;
            interrupted_ = true;
            break;
        }

        if (opcode == ws::OP_CLOSE) {
            std::lock_guard<std::mutex> lock(send_mutex_);
            ws::send_frame(client_fd_, ws::OP_CLOSE, nullptr, 0);
            conn_alive_ = false;
            interrupted_ = true;
            break;
        }
        if (opcode == ws::OP_PING) {
            std::lock_guard<std::mutex> lock(send_mutex_);
            ws::send_frame(client_fd_, ws::OP_PONG, payload.data(), payload.size());
            continue;
        }

        if (opcode == ws::OP_BINARY) {
            handle_voice_binary(payload.data(), payload.size());
            continue;
        }

        if (opcode == ws::OP_TEXT && !payload.empty()) {
            std::string msg(payload.begin(), payload.end());
            handle_voice_text(msg);
        }
    }

    // 清理
    conn_alive_ = false;
    interrupted_ = true;

    // Voice: 连接断开时保存录音
    if (mode_ == ProtocolMode::VOICE && !recording_buffer_.empty()) {
        std::string rec_path = save_recording_wav();
        if (!rec_path.empty()) {
            fprintf(stderr, "[WS] Recording saved on disconnect: %s\n", rec_path.c_str());
        }
        recording_buffer_.clear();
    }

    if (worker_thread_.joinable()) worker_thread_.join();

    fprintf(stderr, "%s Session ended fd=%d\n", log_prefix, client_fd_);
}

} // namespace serve
} // namespace qwen_thor
