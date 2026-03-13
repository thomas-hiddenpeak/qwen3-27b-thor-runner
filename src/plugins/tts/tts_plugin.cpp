// tts_plugin.cpp — TTS 插件实现

#include "tts_plugin.h"
#include "tts_engine.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <sys/wait.h>
#include <unistd.h>

namespace qwen_thor {
namespace plugins {

// ============================================================================
// TtsConfig
// ============================================================================

TtsConfig TtsConfig::from_file(const std::string& path) {
    TtsConfig config;
    std::ifstream f(path);
    if (!f.is_open()) return config;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto key = line.substr(0, eq);
        auto val = line.substr(eq + 1);
        while (!val.empty() && (val.back() == '\r' || val.back() == ' ')) val.pop_back();
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!key.empty() && key.front() == ' ') key.erase(key.begin());

        if (key == "tts_enabled")         config.enabled    = (val == "true" || val == "1");
        else if (key == "tts_mode")       config.mode       = val;
        else if (key == "tts_executable") config.executable = val;
        else if (key == "tts_model")      config.model_path = val;
        else if (key == "tts_voice")      config.voice      = val;
        else if (key == "tts_language")   config.language    = val;
        else if (key == "tts_instruct")   config.instruct   = val;
        else if (key == "tts_speed")      config.speed      = std::stof(val);
        else if (key == "tts_format")     config.format     = val;
        else if (key == "tts_extra_args") config.extra_args = val;
        else if (key == "tts_tmp_dir")    config.tmp_dir    = val;
        else if (key == "tts_max_tokens") config.max_new_tokens = std::stoi(val);
        else if (key == "tts_temperature")   config.tts_temperature = std::stof(val);
        else if (key == "tts_top_k")         config.tts_top_k = std::stoi(val);
        else if (key == "tts_top_p")         config.tts_top_p = std::stof(val);
        else if (key == "tts_rep_penalty")   config.tts_rep_penalty = std::stof(val);
    }
    return config;
}

void TtsConfig::print() const {
    fprintf(stderr, "[TTS Config]\n");
    fprintf(stderr, "  enabled:    %s\n", enabled ? "true" : "false");
    fprintf(stderr, "  mode:       %s\n", mode.c_str());
    fprintf(stderr, "  model:      %s\n", model_path.c_str());
    fprintf(stderr, "  voice:      %s\n", voice.c_str());
    fprintf(stderr, "  language:   %s\n", language.c_str());
    fprintf(stderr, "  speed:      %.1f\n", speed);
    fprintf(stderr, "  format:     %s\n", format.c_str());
    if (!instruct.empty())
        fprintf(stderr, "  instruct:   %s\n", instruct.c_str());
    if (!extra_args.empty())
        fprintf(stderr, "  extra_args: %s\n", extra_args.c_str());
    fprintf(stderr, "  sampling:   temp=%.2f top_k=%d top_p=%.2f rep=%.2f\n",
            tts_temperature, tts_top_k, tts_top_p, tts_rep_penalty);
}

// ============================================================================
// SubprocessTtsPlugin
// ============================================================================

SubprocessTtsPlugin::SubprocessTtsPlugin(const TtsConfig& config)
    : config_(config) {
    std::filesystem::create_directories(config_.tmp_dir);
}

bool SubprocessTtsPlugin::is_available() const {
    if (config_.executable.empty()) return false;
    return std::filesystem::exists(config_.executable);
}

// MIME type 映射
static std::string format_to_content_type(const std::string& fmt) {
    if (fmt == "wav")  return "audio/wav";
    if (fmt == "mp3")  return "audio/mpeg";
    if (fmt == "opus") return "audio/opus";
    if (fmt == "ogg")  return "audio/ogg";
    if (fmt == "flac") return "audio/flac";
    if (fmt == "pcm")  return "audio/pcm";
    if (fmt == "aac")  return "audio/aac";
    return "application/octet-stream";
}

TtsResult SubprocessTtsPlugin::synthesize(const std::string& text,
                                           const std::string& voice,
                                           float speed,
                                           const std::string& format,
                                           const std::string& instruct,
                                           const std::string& language) {
    TtsResult result;
    result.format = format.empty() ? config_.format : format;

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "TTS executable not found: " + config_.executable;
        return result;
    }

    if (text.empty()) {
        result.error_code = 2;
        result.error_message = "Empty text input";
        return result;
    }

    // 生成唯一临时文件名
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string output_path = config_.tmp_dir + "/tts_output_" +
                              std::to_string(now) + "_" +
                              std::to_string(getpid()) + "." + result.format;

    // 将文本写到临时文件 (避免 shell 注入)
    std::string text_path = config_.tmp_dir + "/tts_input_" +
                            std::to_string(now) + "_" +
                            std::to_string(getpid()) + ".txt";
    {
        std::ofstream tf(text_path);
        if (!tf.is_open()) {
            result.error_code = 3;
            result.error_message = "Failed to create temp text file";
            return result;
        }
        tf << text;
    }

    // 构建命令行
    // 通用格式: executable [--model model] [--voice voice] [--output_file out] < input.txt
    std::ostringstream cmd;
    cmd << "cat '" << text_path << "' | "
        << "'" << config_.executable << "'";

    if (!config_.model_path.empty()) {
        cmd << " --model '" << config_.model_path << "'";
    }

    std::string use_voice = voice.empty() ? config_.voice : voice;
    if (!use_voice.empty() && use_voice != "default") {
        cmd << " --voice " << use_voice;
    }

    float use_speed = (speed > 0) ? speed : config_.speed;
    if (use_speed != 1.0f) {
        cmd << " --speed " << use_speed;
    }

    if (!config_.extra_args.empty()) {
        cmd << " " << config_.extra_args;
    }

    cmd << " --output_file '" << output_path << "'"
        << " 2>/dev/null";

    std::string cmd_str = cmd.str();
    fprintf(stderr, "[TTS] Running: %s\n", cmd_str.c_str());

    auto t0 = std::chrono::steady_clock::now();

    int ret = system(cmd_str.c_str());
    int exit_code = WEXITSTATUS(ret);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();

    // 清理输入文本文件
    std::filesystem::remove(text_path);

    if (exit_code != 0) {
        result.error_code = 4;
        result.error_message = "TTS process exited with code " + std::to_string(exit_code);
        std::filesystem::remove(output_path);
        return result;
    }

    // 读取输出音频
    if (!std::filesystem::exists(output_path)) {
        result.error_code = 5;
        result.error_message = "TTS output file not produced";
        return result;
    }

    {
        std::ifstream af(output_path, std::ios::binary | std::ios::ate);
        auto size = af.tellg();
        if (size <= 0) {
            result.error_code = 6;
            result.error_message = "TTS output file is empty";
            std::filesystem::remove(output_path);
            return result;
        }
        result.audio_data.resize(static_cast<size_t>(size));
        af.seekg(0, std::ios::beg);
        af.read(reinterpret_cast<char*>(result.audio_data.data()), size);
    }

    std::filesystem::remove(output_path);

    result.duration_s = elapsed_s;
    fprintf(stderr, "[TTS] Synthesis completed in %.2fs, %zu bytes (%s)\n",
            elapsed_s, result.audio_data.size(), result.format.c_str());

    return result;
}

TtsResult SubprocessTtsPlugin::synthesize_continue(const std::string& text,
                                                    const std::string& format) {
    // Subprocess mode cannot maintain state — fall back to regular synthesis
    return synthesize(text, config_.voice, config_.speed, format);
}

// ============================================================================
// NativeTtsPlugin — 使用内置 Qwen3-TTS 引擎
// ============================================================================

NativeTtsPlugin::NativeTtsPlugin(const TtsConfig& config)
    : config_(config) {
    engine_ = std::make_unique<tts::TTSEngine>();
    fprintf(stderr, "[TTS Native] Loading model from %s...\n", config.model_path.c_str());
    engine_->load_model(config.model_path);
    fprintf(stderr, "[TTS Native] Model loaded, sample_rate=%d, model_type=%s\n",
            engine_->sample_rate(), engine_->config().tts_model_type.c_str());
    // Apply initial sampling parameters from config
    engine_->set_sampling(config.tts_temperature, config.tts_top_k,
                          config.tts_top_p, config.tts_rep_penalty);
}

NativeTtsPlugin::~NativeTtsPlugin() = default;

bool NativeTtsPlugin::is_available() const {
    return engine_ && engine_->is_loaded();
}

void NativeTtsPlugin::set_sampling(float temperature, int top_k, float top_p, float rep_penalty) {
    if (engine_) {
        std::lock_guard<std::mutex> lock(mutex_);
        engine_->set_sampling(temperature, top_k, top_p, rep_penalty);
        fprintf(stderr, "[TTS Native] Sampling updated: temp=%.2f top_k=%d top_p=%.2f rep=%.2f\n",
                temperature, top_k, top_p, rep_penalty);
    }
}

// Build WAV header in memory
static std::vector<uint8_t> build_wav_header_and_data(
    const std::vector<float>& pcm, int sample_rate)
{
    int num_samples = (int)pcm.size();
    int bytes_per_sample = 2;
    int data_size = num_samples * bytes_per_sample;
    int file_size = 36 + data_size;

    std::vector<uint8_t> wav(44 + data_size);
    auto w32 = [&](size_t off, uint32_t v) { memcpy(&wav[off], &v, 4); };
    auto w16 = [&](size_t off, uint16_t v) { memcpy(&wav[off], &v, 2); };

    // RIFF header
    memcpy(&wav[0], "RIFF", 4);
    w32(4, file_size);
    memcpy(&wav[8], "WAVE", 4);

    // fmt chunk
    memcpy(&wav[12], "fmt ", 4);
    w32(16, 16);                    // chunk size
    w16(20, 1);                     // PCM format
    w16(22, 1);                     // mono
    w32(24, sample_rate);
    w32(28, sample_rate * bytes_per_sample);  // byte rate
    w16(32, bytes_per_sample);      // block align
    w16(34, 16);                    // bits per sample

    // data chunk
    memcpy(&wav[36], "data", 4);
    w32(40, data_size);

    // Convert float → int16
    int16_t* samples = reinterpret_cast<int16_t*>(&wav[44]);
    for (int i = 0; i < num_samples; i++) {
        float v = std::max(-1.0f, std::min(1.0f, pcm[i]));
        samples[i] = (int16_t)(v * 32767.0f);
    }

    return wav;
}

TtsResult NativeTtsPlugin::synthesize(const std::string& text,
                                       const std::string& voice,
                                       float speed,
                                       const std::string& format,
                                       const std::string& instruct,
                                       const std::string& language) {
    TtsResult result;
    result.format = (format.empty() || format == "wav") ? "wav" : "pcm";

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "TTS engine not loaded";
        return result;
    }

    if (text.empty()) {
        result.error_code = 2;
        result.error_message = "Empty text input";
        return result;
    }

    std::string use_voice = voice.empty() ? config_.voice : voice;
    std::string use_instruct = instruct.empty() ? config_.instruct : instruct;
    std::string use_lang = language.empty() ? config_.language : language;

    // Serialize access — TTS engine is not thread-safe (single GPU stream)
    std::lock_guard<std::mutex> lock(mutex_);

    auto t0 = std::chrono::steady_clock::now();

    auto pcm = engine_->synthesize_to_pcm(text, use_voice, use_lang, use_instruct,
                                           config_.max_new_tokens);

    if (pcm.empty()) {
        result.error_code = 3;
        result.error_message = "TTS synthesis produced no audio";
        return result;
    }

    int sr = engine_->sample_rate();
    result.sample_rate = sr;
    result.duration_s = (float)pcm.size() / sr;

    if (result.format == "pcm") {
        // Raw PCM16 LE, mono
        result.audio_data.resize(pcm.size() * 2);
        int16_t* out = reinterpret_cast<int16_t*>(result.audio_data.data());
        for (size_t i = 0; i < pcm.size(); i++) {
            float v = std::max(-1.0f, std::min(1.0f, pcm[i]));
            out[i] = (int16_t)(v * 32767.0f);
        }
    } else {
        // WAV format
        result.audio_data = build_wav_header_and_data(pcm, sr);
    }

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();
    fprintf(stderr, "[TTS Native] Synthesized %.1fs audio in %.1fs (%.1fx realtime), %zu bytes\n",
            result.duration_s, elapsed_s,
            result.duration_s / std::max(elapsed_s, 0.001f),
            result.audio_data.size());

    return result;
}

TtsResult NativeTtsPlugin::synthesize_continue(const std::string& text,
                                                const std::string& format) {
    TtsResult result;
    result.format = (format.empty() || format == "wav") ? "wav" : "pcm";

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "TTS engine not loaded";
        return result;
    }

    if (text.empty()) {
        result.error_code = 2;
        result.error_message = "Empty continuation text";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto t0 = std::chrono::steady_clock::now();

    auto pcm = engine_->continue_to_pcm(text, config_.max_new_tokens);

    if (pcm.empty()) {
        result.error_code = 3;
        result.error_message = "TTS continuation produced no audio";
        return result;
    }

    int sr = engine_->sample_rate();
    result.sample_rate = sr;
    result.duration_s = (float)pcm.size() / sr;

    if (result.format == "pcm") {
        result.audio_data.resize(pcm.size() * 2);
        int16_t* out = reinterpret_cast<int16_t*>(result.audio_data.data());
        for (size_t i = 0; i < pcm.size(); i++) {
            float v = std::max(-1.0f, std::min(1.0f, pcm[i]));
            out[i] = (int16_t)(v * 32767.0f);
        }
    } else {
        result.audio_data = build_wav_header_and_data(pcm, sr);
    }

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();
    fprintf(stderr, "[TTS Native] Continue %.1fs audio in %.1fs (%.1fx realtime)\n",
            result.duration_s, elapsed_s,
            result.duration_s / std::max(elapsed_s, 0.001f));

    return result;
}

int NativeTtsPlugin::synthesize_streaming(const std::string& text,
                                           const std::string& voice,
                                           const std::string& instruct,
                                           PcmCallback callback,
                                           int chunk_frames,
                                           const std::string& language) {
    if (!is_available() || !callback) return 0;

    std::lock_guard<std::mutex> lock(mutex_);

    std::string speaker = voice;
    std::string use_lang = language.empty() ? config_.language : language;
    // Use instruct from call parameter, fall back to config default
    std::string use_instruct = instruct.empty() ? config_.instruct : instruct;

    return engine_->synthesize_streaming(text, speaker, use_lang, use_instruct,
                                          config_.max_new_tokens,
                                          chunk_frames, callback);
}

int NativeTtsPlugin::continue_streaming(const std::string& text,
                                         PcmCallback callback,
                                         int chunk_frames) {
    if (!is_available() || !callback) return 0;

    std::lock_guard<std::mutex> lock(mutex_);

    return engine_->continue_streaming(text,
                                        config_.max_new_tokens,
                                        chunk_frames, callback);
}

// ============================================================================
// NativeTtsPlugin::model_info — 返回模型类型和可用音色
// ============================================================================

TtsPlugin::ModelInfo NativeTtsPlugin::model_info() const {
    ModelInfo info;
    if (!engine_ || !engine_->is_loaded()) return info;
    const auto& cfg = engine_->config();
    info.model_type = cfg.tts_model_type;
    info.sample_rate = cfg.tokenizer_decoder.output_sample_rate;
    for (const auto& [name, id] : cfg.talker.spk_id) {
        info.available_voices.push_back(name);
    }
    for (const auto& [name, id] : cfg.talker.codec_language_id) {
        info.available_languages.push_back(name);
    }
    // Sort for stable ordering
    std::sort(info.available_voices.begin(), info.available_voices.end());
    std::sort(info.available_languages.begin(), info.available_languages.end());
    info.speaker_dialects = cfg.talker.spk_is_dialect;
    return info;
}

// ============================================================================
// 工厂
// ============================================================================

std::unique_ptr<TtsPlugin> create_tts_plugin(const TtsConfig& config) {
    if (!config.enabled) return nullptr;
    if (config.mode == "native" && !config.model_path.empty()) {
        return std::make_unique<NativeTtsPlugin>(config);
    }
    return std::make_unique<SubprocessTtsPlugin>(config);
}

} // namespace plugins
} // namespace qwen_thor
