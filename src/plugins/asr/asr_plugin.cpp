// asr_plugin.cpp — ASR 插件实现

#include "asr_plugin.h"
#include "asr_engine.h"
#include "audio_utils.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <array>
#include <chrono>
#include <sys/wait.h>
#include <unistd.h>

namespace qwen_thor {
namespace plugins {

// ============================================================================
// AsrConfig
// ============================================================================

AsrConfig AsrConfig::from_file(const std::string& path) {
    AsrConfig config;
    std::ifstream f(path);
    if (!f.is_open()) return config;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto key = line.substr(0, eq);
        auto val = line.substr(eq + 1);
        // trim
        while (!val.empty() && (val.back() == '\r' || val.back() == ' ')) val.pop_back();
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!key.empty() && key.front() == ' ') key.erase(key.begin());

        if (key == "asr_enabled")     config.enabled    = (val == "true" || val == "1");
        else if (key == "asr_mode")       config.mode       = val;
        else if (key == "asr_executable") config.executable = val;
        else if (key == "asr_model")      config.model_path = val;
        else if (key == "asr_language")   config.language   = val;
        else if (key == "asr_threads")    config.threads    = std::stoi(val);
        else if (key == "asr_extra_args") config.extra_args = val;
        else if (key == "asr_tmp_dir")    config.tmp_dir    = val;
    }
    return config;
}

void AsrConfig::print() const {
    fprintf(stderr, "[ASR Config]\n");
    fprintf(stderr, "  enabled:    %s\n", enabled ? "true" : "false");
    fprintf(stderr, "  mode:       %s\n", mode.c_str());
    fprintf(stderr, "  executable: %s\n", executable.c_str());
    fprintf(stderr, "  model:      %s\n", model_path.c_str());
    fprintf(stderr, "  language:   %s\n", language.c_str());
    fprintf(stderr, "  threads:    %d\n", threads);
    if (!extra_args.empty())
        fprintf(stderr, "  extra_args: %s\n", extra_args.c_str());
}

// ============================================================================
// SubprocessAsrPlugin
// ============================================================================

SubprocessAsrPlugin::SubprocessAsrPlugin(const AsrConfig& config)
    : config_(config) {
    // 确保临时目录存在
    std::filesystem::create_directories(config_.tmp_dir);
}

bool SubprocessAsrPlugin::is_available() const {
    if (config_.executable.empty()) return false;
    return std::filesystem::exists(config_.executable);
}

AsrResult SubprocessAsrPlugin::transcribe(const std::string& audio_path,
                                           const std::string& language) {
    AsrResult result;

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "ASR executable not found: " + config_.executable;
        return result;
    }

    if (!std::filesystem::exists(audio_path)) {
        result.error_code = 2;
        result.error_message = "Audio file not found: " + audio_path;
        return result;
    }

    // 构建输出文件路径
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string output_path = config_.tmp_dir + "/asr_output_" +
                              std::to_string(now) + "_" +
                              std::to_string(getpid()) + ".txt";

    // 构建命令行
    // 支持通用格式: executable [--model model] [--language lang] [--threads n] [--output-txt out] input
    // 用户可通过 extra_args 自定义参数格式
    std::ostringstream cmd;
    cmd << "'" << config_.executable << "'";

    if (!config_.model_path.empty()) {
        cmd << " --model '" << config_.model_path << "'";
    }

    std::string lang = (language == "auto" || language.empty()) ? config_.language : language;
    if (lang != "auto" && !lang.empty()) {
        cmd << " --language " << lang;
    }

    cmd << " --threads " << config_.threads;

    if (!config_.extra_args.empty()) {
        cmd << " " << config_.extra_args;
    }

    cmd << " --output-txt"
        << " --output-file '" << output_path << "'"
        << " '" << audio_path << "'"
        << " 2>/dev/null";

    std::string cmd_str = cmd.str();
    fprintf(stderr, "[ASR] Running: %s\n", cmd_str.c_str());

    auto t0 = std::chrono::steady_clock::now();

    // 执行子进程
    int ret = system(cmd_str.c_str());
    int exit_code = WEXITSTATUS(ret);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();

    if (exit_code != 0) {
        // 尝试用 popen 获取 stderr
        result.error_code = 3;
        result.error_message = "ASR process exited with code " + std::to_string(exit_code);
        // 清理
        std::filesystem::remove(output_path);
        return result;
    }

    // 检查输出文件 — whisper.cpp 可能追加 .txt 后缀
    std::string actual_output = output_path;
    if (!std::filesystem::exists(actual_output)) {
        actual_output = output_path + ".txt";
    }

    // 读取转录结果
    if (std::filesystem::exists(actual_output)) {
        std::ifstream ofs(actual_output);
        std::ostringstream ss;
        ss << ofs.rdbuf();
        result.text = ss.str();

        // trim 首尾空白
        while (!result.text.empty() && (result.text.back() == '\n' || result.text.back() == '\r' || result.text.back() == ' '))
            result.text.pop_back();
        while (!result.text.empty() && (result.text.front() == '\n' || result.text.front() == '\r' || result.text.front() == ' '))
            result.text.erase(result.text.begin());

        std::filesystem::remove(actual_output);
    } else {
        // 没有输出文件, 尝试用 popen 捕获 stdout
        std::string popen_cmd = cmd_str.substr(0, cmd_str.rfind("2>/dev/null")) + "2>/dev/null";
        FILE* pipe = popen(popen_cmd.c_str(), "r");
        if (pipe) {
            std::array<char, 4096> buf;
            while (fgets(buf.data(), buf.size(), pipe)) {
                result.text += buf.data();
            }
            pclose(pipe);
            // trim
            while (!result.text.empty() && (result.text.back() == '\n' || result.text.back() == '\r'))
                result.text.pop_back();
        } else {
            result.error_code = 4;
            result.error_message = "No ASR output produced";
            return result;
        }
    }

    result.duration_s = elapsed_s;
    result.language = lang;
    fprintf(stderr, "[ASR] Transcription completed in %.2fs: \"%s\"\n",
            elapsed_s, result.text.substr(0, 100).c_str());

    return result;
}

// ============================================================================
// NativeAsrPlugin — 使用内置 Qwen3-ASR 引擎
// ============================================================================

NativeAsrPlugin::NativeAsrPlugin(const AsrConfig& config)
    : config_(config) {
    engine_ = std::make_unique<asr::ASREngine>();
    fprintf(stderr, "[ASR Native] Loading model from %s...\n", config.model_path.c_str());
    engine_->load_model(config.model_path);
    fprintf(stderr, "[ASR Native] Model loaded\n");
}

NativeAsrPlugin::~NativeAsrPlugin() = default;

bool NativeAsrPlugin::is_available() const {
    return engine_ && engine_->is_loaded();
}

AsrResult NativeAsrPlugin::transcribe(const std::string& audio_path,
                                       const std::string& language) {
    AsrResult result;

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "ASR engine not loaded";
        return result;
    }

    if (!std::filesystem::exists(audio_path)) {
        result.error_code = 2;
        result.error_message = "Audio file not found: " + audio_path;
        return result;
    }

    // Serialize access — ASR engine is not thread-safe (single GPU stream)
    std::lock_guard<std::mutex> lock(mutex_);

    auto t0 = std::chrono::steady_clock::now();

    std::string text = engine_->transcribe_file(audio_path);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();

    if (text.empty()) {
        result.error_code = 3;
        result.error_message = "ASR transcription produced no text";
        return result;
    }

    // Engine already does token-level extraction (only decodes tokens after <asr_text> marker).
    // Just trim whitespace.
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) text.erase(text.begin());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n')) text.pop_back();

    result.text = text;
    result.language = language;
    result.duration_s = elapsed_s;

    fprintf(stderr, "[ASR Native] Transcribed in %.2fs: \"%s\"\n",
            elapsed_s, text.substr(0, 100).c_str());

    return result;
}

// ============================================================================
// AsrPlugin::transcribe_memory — 默认实现 (写临时文件)
// ============================================================================

AsrResult AsrPlugin::transcribe_memory(const uint8_t* data, size_t size,
                                        const std::string& language,
                                        const std::string& filename_hint,
                                        bool /*suppress_early_eos*/) {
    // 默认: 写临时文件, 调用 transcribe(path)
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string tmp_path = "tmp/asr_mem_" + std::to_string(now) + "_" +
                           std::to_string(getpid()) + ".wav";
    {
        std::ofstream af(tmp_path, std::ios::binary);
        if (!af.is_open()) {
            AsrResult r;
            r.error_code = 10;
            r.error_message = "Failed to create temp file for transcribe_memory";
            return r;
        }
        af.write(reinterpret_cast<const char*>(data), size);
    }
    auto result = transcribe(tmp_path, language);
    std::filesystem::remove(tmp_path);
    return result;
}

// ============================================================================
// AsrPlugin::transcribe_pcm — 默认实现 (不支持)
// ============================================================================

AsrResult AsrPlugin::transcribe_pcm(const float* /*samples*/, int /*num_samples*/,
                                     int /*sample_rate*/, const std::string& /*language*/,
                                     bool /*suppress_early_eos*/) {
    AsrResult r;
    r.error_code = 99;
    r.error_message = "transcribe_pcm not supported by this plugin";
    return r;
}

// ============================================================================
// NativeAsrPlugin::transcribe_memory — 零临时文件, 直接内存解析
// ============================================================================

AsrResult NativeAsrPlugin::transcribe_memory(const uint8_t* data, size_t size,
                                              const std::string& language,
                                              const std::string& filename_hint,
                                              bool suppress_early_eos) {
    AsrResult result;

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "ASR engine not loaded";
        return result;
    }

    // 在内存中解析音频 — 支持 WAV/MP3/M4A/OGG/FLAC 等
    audio::AudioData wav;
    if (!audio::load_audio_from_memory(data, size, wav, filename_hint)) {
        result.error_code = 2;
        result.error_message = "Failed to parse audio data (supported: WAV/MP3/M4A/OGG/FLAC)";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto t0 = std::chrono::steady_clock::now();

    std::string text = engine_->transcribe(wav.samples.data(), (int)wav.samples.size(),
                                            wav.sample_rate, 0.0f, 448,
                                            suppress_early_eos);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();

    if (text.empty()) {
        result.error_code = 3;
        result.error_message = "ASR transcription produced no text";
        return result;
    }

    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) text.erase(text.begin());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n')) text.pop_back();

    result.text = text;
    result.language = language;
    result.duration_s = elapsed_s;

    fprintf(stderr, "[ASR Native] Transcribed (memory) in %.2fs: \"%s\"\n",
            elapsed_s, text.substr(0, 100).c_str());

    return result;
}

// ============================================================================
// NativeAsrPlugin::transcribe_pcm — 原始 float 样本直接转录
// ============================================================================

AsrResult NativeAsrPlugin::transcribe_pcm(const float* samples, int num_samples,
                                           int sample_rate, const std::string& language,
                                           bool suppress_early_eos) {
    AsrResult result;

    if (!is_available()) {
        result.error_code = 1;
        result.error_message = "ASR engine not loaded";
        return result;
    }

    if (!samples || num_samples <= 0) {
        result.error_code = 2;
        result.error_message = "Empty audio data";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto t0 = std::chrono::steady_clock::now();

    std::string text = engine_->transcribe(samples, num_samples, sample_rate,
                                            0.0f, 448, suppress_early_eos);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed_s = std::chrono::duration<float>(t1 - t0).count();

    if (text.empty()) {
        result.error_code = 3;
        result.error_message = "ASR transcription produced no text";
        return result;
    }

    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) text.erase(text.begin());
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n')) text.pop_back();

    result.text = text;
    result.language = language;
    result.duration_s = elapsed_s;

    fprintf(stderr, "[ASR Native] Transcribed (PCM stream, %.1fs audio) in %.2fs: \"%s\"\n",
            (float)num_samples / sample_rate, elapsed_s, text.substr(0, 100).c_str());

    return result;
}

// ============================================================================
// 工厂
// ============================================================================

std::unique_ptr<AsrPlugin> create_asr_plugin(const AsrConfig& config) {
    if (!config.enabled) return nullptr;
    if (config.mode == "native" && !config.model_path.empty()) {
        return std::make_unique<NativeAsrPlugin>(config);
    }
    return std::make_unique<SubprocessAsrPlugin>(config);
}

} // namespace plugins
} // namespace qwen_thor
