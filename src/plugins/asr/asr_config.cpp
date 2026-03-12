// asr_config.cpp — Qwen3-ASR config.json 解析

#include "asr_config.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

// 最小 JSON 解析: 只提取需要的字段, 不引入第三方库
// 利用项目中已有的简单 JSON 解析模式 (参考 model.cpp 的 config 解析)

namespace qwen_thor {
namespace asr {

namespace {

// Simple JSON value extraction helpers
// These are intentionally minimal — ASR config is small and well-structured

std::string read_file_contents(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Find a JSON key and return its integer value
bool json_get_int(const std::string& json, const std::string& key, int& out) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    // Handle negative numbers
    bool negative = false;
    if (pos < json.size() && json[pos] == '-') { negative = true; pos++; }
    if (pos >= json.size() || !std::isdigit(json[pos])) return false;
    long val = 0;
    while (pos < json.size() && std::isdigit(json[pos])) {
        val = val * 10 + (json[pos] - '0');
        pos++;
    }
    out = static_cast<int>(negative ? -val : val);
    return true;
}

// Find a JSON key and return its float value
bool json_get_float(const std::string& json, const std::string& key, float& out) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    size_t end = pos;
    while (end < json.size() && (std::isdigit(json[end]) || json[end] == '.' || json[end] == 'e' || json[end] == 'E' || json[end] == '-' || json[end] == '+')) end++;
    if (end == pos) return false;
    try {
        out = std::stof(json.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

// Find a JSON key and return its boolean value
bool json_get_bool(const std::string& json, const std::string& key, bool& out) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    if (json.compare(pos, 4, "true") == 0) { out = true; return true; }
    if (json.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

// Extract a JSON sub-object by key name
std::string json_get_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('{', pos + search.size());
    if (pos == std::string::npos) return "";
    // Find matching closing brace
    int depth = 1;
    size_t start = pos;
    pos++;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') depth++;
        else if (json[pos] == '}') depth--;
        pos++;
    }
    return json.substr(start, pos - start);
}

// Extract mrope_section array [a, b, c]
bool json_get_int_array3(const std::string& json, const std::string& key, std::array<int, 3>& out) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return false;
    pos = json.find('[', pos + search.size());
    if (pos == std::string::npos) return false;
    pos++; // skip '['
    for (int i = 0; i < 3; i++) {
        while (pos < json.size() && !std::isdigit(json[pos]) && json[pos] != '-') pos++;
        if (pos >= json.size()) return false;
        int val = 0;
        bool neg = false;
        if (json[pos] == '-') { neg = true; pos++; }
        while (pos < json.size() && std::isdigit(json[pos])) {
            val = val * 10 + (json[pos] - '0');
            pos++;
        }
        out[i] = neg ? -val : val;
    }
    return true;
}

} // anonymous namespace

bool ASRConfig::load_from_json(const std::string& config_path) {
    std::string content = read_file_contents(config_path);
    if (content.empty()) {
        std::cerr << "[ASR] Failed to read config: " << config_path << std::endl;
        return false;
    }

    // Navigate: thinker_config -> audio_config / text_config
    std::string thinker = json_get_object(content, "thinker_config");
    if (thinker.empty()) {
        std::cerr << "[ASR] Missing thinker_config in " << config_path << std::endl;
        return false;
    }

    std::string audio = json_get_object(thinker, "audio_config");
    std::string text = json_get_object(thinker, "text_config");

    if (audio.empty() || text.empty()) {
        std::cerr << "[ASR] Missing audio_config or text_config" << std::endl;
        return false;
    }

    // Audio Encoder config
    json_get_int(audio, "num_mel_bins", num_mel_bins);
    json_get_int(audio, "encoder_layers", encoder_layers);
    json_get_int(audio, "d_model", encoder_d_model);
    json_get_int(audio, "encoder_attention_heads", encoder_attention_heads);
    json_get_int(audio, "encoder_ffn_dim", encoder_ffn_dim);
    json_get_int(audio, "downsample_hidden_size", downsample_hidden_size);
    json_get_int(audio, "max_source_positions", max_source_positions);
    json_get_int(audio, "n_window", n_window);
    json_get_int(audio, "n_window_infer", n_window_infer);
    json_get_int(audio, "conv_chunksize", conv_chunksize);
    json_get_int(audio, "output_dim", output_dim);

    encoder_head_dim = encoder_d_model / encoder_attention_heads;

    // Text Decoder config
    json_get_int(text, "num_hidden_layers", decoder_layers);
    json_get_int(text, "hidden_size", decoder_hidden_size);
    json_get_int(text, "num_attention_heads", decoder_num_attention_heads);
    json_get_int(text, "num_key_value_heads", decoder_num_kv_heads);
    json_get_int(text, "head_dim", decoder_head_dim);
    json_get_int(text, "intermediate_size", decoder_intermediate_size);
    json_get_int(text, "vocab_size", vocab_size);
    json_get_float(text, "rms_norm_eps", rms_norm_eps);

    // rope_theta can be int or float in JSON
    float rt = 0;
    if (json_get_float(text, "rope_theta", rt)) rope_theta = rt;

    json_get_bool(text, "tie_word_embeddings", tie_word_embeddings);

    // MRoPE (nested in rope_scaling)
    std::string rope_scaling = json_get_object(text, "rope_scaling");
    if (!rope_scaling.empty()) {
        json_get_bool(rope_scaling, "mrope_interleaved", mrope_interleaved);
        json_get_int_array3(rope_scaling, "mrope_section", mrope_section);
    }

    std::cerr << "[ASR] Config loaded: encoder=" << encoder_layers << "L×" << encoder_d_model
              << " decoder=" << decoder_layers << "L×" << decoder_hidden_size
              << " vocab=" << vocab_size
              << " mrope=[" << mrope_section[0] << "," << mrope_section[1] << "," << mrope_section[2] << "]"
              << std::endl;

    return true;
}

} // namespace asr
} // namespace qwen_thor
