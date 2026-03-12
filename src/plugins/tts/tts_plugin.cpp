// tts_plugin.cpp — TTS 插件实现

#include "tts_plugin.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
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

        if (key == "tts_enabled")     config.enabled    = (val == "true" || val == "1");
        else if (key == "tts_executable") config.executable = val;
        else if (key == "tts_model")      config.model_path = val;
        else if (key == "tts_voice")      config.voice      = val;
        else if (key == "tts_speed")      config.speed      = std::stof(val);
        else if (key == "tts_format")     config.format     = val;
        else if (key == "tts_extra_args") config.extra_args = val;
        else if (key == "tts_tmp_dir")    config.tmp_dir    = val;
    }
    return config;
}

void TtsConfig::print() const {
    fprintf(stderr, "[TTS Config]\n");
    fprintf(stderr, "  enabled:    %s\n", enabled ? "true" : "false");
    fprintf(stderr, "  executable: %s\n", executable.c_str());
    fprintf(stderr, "  model:      %s\n", model_path.c_str());
    fprintf(stderr, "  voice:      %s\n", voice.c_str());
    fprintf(stderr, "  speed:      %.1f\n", speed);
    fprintf(stderr, "  format:     %s\n", format.c_str());
    if (!extra_args.empty())
        fprintf(stderr, "  extra_args: %s\n", extra_args.c_str());
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
                                           const std::string& format) {
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

// ============================================================================
// 工厂
// ============================================================================

std::unique_ptr<TtsPlugin> create_tts_plugin(const TtsConfig& config) {
    if (!config.enabled) return nullptr;
    return std::make_unique<SubprocessTtsPlugin>(config);
}

} // namespace plugins
} // namespace qwen_thor
