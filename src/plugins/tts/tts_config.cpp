#include "tts_config.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

// Minimal JSON parsing (reuse the pattern from asr_config.cpp)
namespace {

std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Simple JSON helpers — extract values from a JSON string
// These are intentionally minimal for config parsing only.

std::string json_find_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find('{', pos);
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') { depth--; if (depth == 0) return json.substr(start, i - start + 1); }
    }
    return "";
}

int json_get_int(const std::string& json, const std::string& key, int def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    // Handle null
    if (pos + 3 < json.size() && json.substr(pos, 4) == "null") return def;
    return atoi(json.c_str() + pos);
}

float json_get_float(const std::string& json, const std::string& key, float def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos + 3 < json.size() && json.substr(pos, 4) == "null") return def;
    return strtof(json.c_str() + pos, nullptr);
}

bool json_get_bool(const std::string& json, const std::string& key, bool def) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos + 3 < json.size() && json.substr(pos, 4) == "true") return true;
    if (pos + 4 < json.size() && json.substr(pos, 5) == "false") return false;
    return def;
}

std::string json_get_string(const std::string& json, const std::string& key, const std::string& def = "") {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos);
    if (pos == std::string::npos) return def;
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return def;
    pos++; // skip opening quote
    auto end = json.find('"', pos);
    if (end == std::string::npos) return def;
    return json.substr(pos, end - pos);
}

// Parse a flat { "key": int_value, ... } object into a map
std::unordered_map<std::string, int> json_parse_string_int_map(const std::string& json, const std::string& key) {
    std::unordered_map<std::string, int> result;
    std::string obj = json_find_object(json, key);
    if (obj.empty()) return result;

    size_t pos = 0;
    while (true) {
        pos = obj.find('"', pos);
        if (pos == std::string::npos) break;
        pos++; // skip opening quote
        auto end = obj.find('"', pos);
        if (end == std::string::npos) break;
        std::string k = obj.substr(pos, end - pos);
        pos = end + 1;
        auto colon = obj.find(':', pos);
        if (colon == std::string::npos) break;
        pos = colon + 1;
        while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t')) pos++;
        int val = atoi(obj.c_str() + pos);
        result[k] = val;
        // Skip past the number
        while (pos < obj.size() && (obj[pos] == '-' || (obj[pos] >= '0' && obj[pos] <= '9'))) pos++;
    }
    return result;
}

// Parse mrope_section from rope_scaling object
void json_parse_mrope_sections(const std::string& json, int sections[3]) {
    std::string rope_scaling = json_find_object(json, "rope_scaling");
    if (rope_scaling.empty()) return;

    std::string search = "\"mrope_section\"";
    auto pos = rope_scaling.find(search);
    if (pos == std::string::npos) return;
    pos = rope_scaling.find('[', pos);
    if (pos == std::string::npos) return;
    pos++; // skip '['
    for (int i = 0; i < 3; i++) {
        while (pos < rope_scaling.size() && (rope_scaling[pos] == ' ' || rope_scaling[pos] == ',')) pos++;
        sections[i] = atoi(rope_scaling.c_str() + pos);
        while (pos < rope_scaling.size() && rope_scaling[pos] != ',' && rope_scaling[pos] != ']') pos++;
    }
}

} // anonymous namespace

namespace qwen_thor {
namespace tts {

bool TTSConfig::load_from_json(const std::string& config_path) {
    std::string json = read_file_to_string(config_path);
    if (json.empty()) {
        fprintf(stderr, "[TTS] Error: cannot read %s\n", config_path.c_str());
        return false;
    }

    // Top-level fields
    model_type = json_get_string(json, "model_type", model_type);
    tts_model_type = json_get_string(json, "tts_model_type", tts_model_type);
    im_start_token_id = json_get_int(json, "im_start_token_id", im_start_token_id);
    im_end_token_id = json_get_int(json, "im_end_token_id", im_end_token_id);
    tts_pad_token_id = json_get_int(json, "tts_pad_token_id", tts_pad_token_id);
    tts_bos_token_id = json_get_int(json, "tts_bos_token_id", tts_bos_token_id);
    tts_eos_token_id = json_get_int(json, "tts_eos_token_id", tts_eos_token_id);
    assistant_token_id = json_get_int(json, "assistant_token_id", assistant_token_id);

    // Talker config
    std::string tc = json_find_object(json, "talker_config");
    if (!tc.empty()) {
        // Extract and remove code_predictor_config first to avoid
        // nested key collisions (e.g., "hidden_size" in both objects)
        std::string cp = json_find_object(tc, "code_predictor_config");

        // Remove code_predictor_config from tc for clean talker parsing
        std::string tc_clean = tc;
        if (!cp.empty()) {
            auto cp_key_pos = tc_clean.find("\"code_predictor_config\"");
            if (cp_key_pos != std::string::npos) {
                auto cp_obj_end = tc_clean.find(cp, cp_key_pos);
                if (cp_obj_end != std::string::npos) {
                    size_t erase_end = cp_obj_end + cp.size();
                    // Also erase trailing comma if any
                    while (erase_end < tc_clean.size() &&
                           (tc_clean[erase_end] == ',' || tc_clean[erase_end] == ' ' ||
                            tc_clean[erase_end] == '\n' || tc_clean[erase_end] == '\r'))
                        erase_end++;
                    tc_clean.erase(cp_key_pos, erase_end - cp_key_pos);
                }
            }
        }

        talker.hidden_size = json_get_int(tc_clean, "hidden_size", talker.hidden_size);
        talker.num_hidden_layers = json_get_int(tc_clean, "num_hidden_layers", talker.num_hidden_layers);
        talker.num_attention_heads = json_get_int(tc_clean, "num_attention_heads", talker.num_attention_heads);
        talker.num_kv_heads = json_get_int(tc_clean, "num_key_value_heads", talker.num_kv_heads);
        talker.head_dim = json_get_int(tc_clean, "head_dim", talker.head_dim);
        talker.intermediate_size = json_get_int(tc_clean, "intermediate_size", talker.intermediate_size);
        talker.vocab_size = json_get_int(tc_clean, "vocab_size", talker.vocab_size);
        talker.text_vocab_size = json_get_int(tc_clean, "text_vocab_size", talker.text_vocab_size);
        talker.text_hidden_size = json_get_int(tc_clean, "text_hidden_size", talker.text_hidden_size);
        talker.rms_norm_eps = json_get_float(tc_clean, "rms_norm_eps", talker.rms_norm_eps);
        talker.rope_theta = json_get_float(tc_clean, "rope_theta", talker.rope_theta);
        talker.max_position_embeddings = json_get_int(tc_clean, "max_position_embeddings", talker.max_position_embeddings);
        talker.num_code_groups = json_get_int(tc_clean, "num_code_groups", talker.num_code_groups);
        talker.attention_bias = json_get_bool(tc_clean, "attention_bias", talker.attention_bias);

        // Token IDs
        talker.codec_pad_id = json_get_int(tc_clean, "codec_pad_id", talker.codec_pad_id);
        talker.codec_bos_id = json_get_int(tc_clean, "codec_bos_id", talker.codec_bos_id);
        talker.codec_eos_token_id = json_get_int(tc_clean, "codec_eos_token_id", talker.codec_eos_token_id);
        talker.codec_think_id = json_get_int(tc_clean, "codec_think_id", talker.codec_think_id);
        talker.codec_nothink_id = json_get_int(tc_clean, "codec_nothink_id", talker.codec_nothink_id);
        talker.codec_think_bos_id = json_get_int(tc_clean, "codec_think_bos_id", talker.codec_think_bos_id);
        talker.codec_think_eos_id = json_get_int(tc_clean, "codec_think_eos_id", talker.codec_think_eos_id);

        // MRoPE sections
        json_parse_mrope_sections(tc_clean, talker.mrope_sections);

        // Speaker IDs
        talker.spk_id = json_parse_string_int_map(tc_clean, "spk_id");

        // Language IDs
        talker.codec_language_id = json_parse_string_int_map(tc_clean, "codec_language_id");

        // Code Predictor config (nested inside talker_config, already extracted above)
        if (!cp.empty()) {
            code_predictor.hidden_size = json_get_int(cp, "hidden_size", code_predictor.hidden_size);
            code_predictor.num_hidden_layers = json_get_int(cp, "num_hidden_layers", code_predictor.num_hidden_layers);
            code_predictor.num_attention_heads = json_get_int(cp, "num_attention_heads", code_predictor.num_attention_heads);
            code_predictor.num_kv_heads = json_get_int(cp, "num_key_value_heads", code_predictor.num_kv_heads);
            code_predictor.head_dim = json_get_int(cp, "head_dim", code_predictor.head_dim);
            code_predictor.intermediate_size = json_get_int(cp, "intermediate_size", code_predictor.intermediate_size);
            code_predictor.vocab_size = json_get_int(cp, "vocab_size", code_predictor.vocab_size);
            code_predictor.rms_norm_eps = json_get_float(cp, "rms_norm_eps", code_predictor.rms_norm_eps);
            code_predictor.rope_theta = json_get_float(cp, "rope_theta", code_predictor.rope_theta);
            code_predictor.max_position_embeddings = json_get_int(cp, "max_position_embeddings", code_predictor.max_position_embeddings);
            code_predictor.attention_bias = json_get_bool(cp, "attention_bias", code_predictor.attention_bias);
        }
    }

    fprintf(stderr, "[TTS] Config loaded:\n");
    fprintf(stderr, "  model_type=%s, tts_model_type=%s\n", model_type.c_str(), tts_model_type.c_str());
    fprintf(stderr, "  talker: %dL, hidden=%d, %dQ/%dKV, head_dim=%d, ffn=%d, vocab=%d\n",
            talker.num_hidden_layers, talker.hidden_size,
            talker.num_attention_heads, talker.num_kv_heads,
            talker.head_dim, talker.intermediate_size, talker.vocab_size);
    fprintf(stderr, "  code_predictor: %dL, hidden=%d, vocab=%d, num_code_groups=%d\n",
            code_predictor.num_hidden_layers, code_predictor.hidden_size,
            code_predictor.vocab_size, talker.num_code_groups);
    fprintf(stderr, "  speakers: %zu, languages: %zu\n",
            talker.spk_id.size(), talker.codec_language_id.size());
    return true;
}

bool TTSConfig::load_tokenizer_config(const std::string& config_path) {
    std::string json = read_file_to_string(config_path);
    if (json.empty()) {
        fprintf(stderr, "[TTS] Error: cannot read tokenizer config %s\n", config_path.c_str());
        return false;
    }

    // Decoder config
    std::string dc = json_find_object(json, "decoder_config");
    if (!dc.empty()) {
        tokenizer_decoder.num_quantizers = json_get_int(dc, "num_quantizers", tokenizer_decoder.num_quantizers);
        tokenizer_decoder.num_semantic_quantizers = json_get_int(dc, "num_semantic_quantizers", tokenizer_decoder.num_semantic_quantizers);
        tokenizer_decoder.codebook_size = json_get_int(dc, "codebook_size", tokenizer_decoder.codebook_size);
        tokenizer_decoder.semantic_codebook_size = json_get_int(dc, "semantic_codebook_size", tokenizer_decoder.semantic_codebook_size);
        tokenizer_decoder.codebook_dim = json_get_int(dc, "codebook_dim", tokenizer_decoder.codebook_dim);
        tokenizer_decoder.vq_hidden_dim = json_get_int(dc, "vector_quantization_hidden_dimension", tokenizer_decoder.vq_hidden_dim);
        tokenizer_decoder.latent_dim = json_get_int(dc, "latent_dim", tokenizer_decoder.latent_dim);
        tokenizer_decoder.hidden_size = json_get_int(dc, "hidden_size", tokenizer_decoder.hidden_size);
        tokenizer_decoder.num_hidden_layers = json_get_int(dc, "num_hidden_layers", tokenizer_decoder.num_hidden_layers);
        tokenizer_decoder.num_attention_heads = json_get_int(dc, "num_attention_heads", tokenizer_decoder.num_attention_heads);
        tokenizer_decoder.num_kv_heads = json_get_int(dc, "num_key_value_heads", tokenizer_decoder.num_kv_heads);
        tokenizer_decoder.head_dim = json_get_int(dc, "head_dim", tokenizer_decoder.head_dim);
        tokenizer_decoder.intermediate_size = json_get_int(dc, "intermediate_size", tokenizer_decoder.intermediate_size);
        tokenizer_decoder.rms_norm_eps = json_get_float(dc, "rms_norm_eps", tokenizer_decoder.rms_norm_eps);
        tokenizer_decoder.rope_theta = json_get_float(dc, "rope_theta", tokenizer_decoder.rope_theta);
        tokenizer_decoder.sliding_window = json_get_int(dc, "sliding_window", tokenizer_decoder.sliding_window);
        tokenizer_decoder.layer_scale_init = json_get_float(dc, "layer_scale_initial_scale", tokenizer_decoder.layer_scale_init);
        tokenizer_decoder.decoder_dim = json_get_int(dc, "decoder_dim", tokenizer_decoder.decoder_dim);

        // Parse upsample_rates array
        std::string ur_search = "\"upsample_rates\"";
        auto pos = dc.find(ur_search);
        if (pos != std::string::npos) {
            pos = dc.find('[', pos);
            if (pos != std::string::npos) {
                pos++;
                for (int i = 0; i < 4; i++) {
                    while (pos < dc.size() && (dc[pos] == ' ' || dc[pos] == ',')) pos++;
                    tokenizer_decoder.upsample_rates[i] = atoi(dc.c_str() + pos);
                    while (pos < dc.size() && dc[pos] != ',' && dc[pos] != ']') pos++;
                }
            }
        }

        // Parse upsampling_ratios array
        std::string us_search = "\"upsampling_ratios\"";
        pos = dc.find(us_search);
        if (pos != std::string::npos) {
            pos = dc.find('[', pos);
            if (pos != std::string::npos) {
                pos++;
                for (int i = 0; i < 2; i++) {
                    while (pos < dc.size() && (dc[pos] == ' ' || dc[pos] == ',')) pos++;
                    tokenizer_decoder.upsampling_ratios[i] = atoi(dc.c_str() + pos);
                    while (pos < dc.size() && dc[pos] != ',' && dc[pos] != ']') pos++;
                }
            }
        }
    }

    tokenizer_decoder.output_sample_rate = json_get_int(json, "output_sample_rate", tokenizer_decoder.output_sample_rate);
    tokenizer_decoder.decode_upsample_rate = json_get_int(json, "decode_upsample_rate", tokenizer_decoder.decode_upsample_rate);

    fprintf(stderr, "[TTS] Tokenizer decoder config loaded:\n");
    fprintf(stderr, "  quantizers=%d, codebook=%dx%d, latent=%d\n",
            tokenizer_decoder.num_quantizers, tokenizer_decoder.codebook_size,
            tokenizer_decoder.vq_hidden_dim, tokenizer_decoder.latent_dim);
    fprintf(stderr, "  pre_transformer: %dL, hidden=%d, %d heads, sliding_window=%d\n",
            tokenizer_decoder.num_hidden_layers, tokenizer_decoder.hidden_size,
            tokenizer_decoder.num_attention_heads, tokenizer_decoder.sliding_window);
    fprintf(stderr, "  BigVGAN: dim=%d, upsample=[%d,%d,%d,%d], pre_upsample=[%d,%d]\n",
            tokenizer_decoder.decoder_dim,
            tokenizer_decoder.upsample_rates[0], tokenizer_decoder.upsample_rates[1],
            tokenizer_decoder.upsample_rates[2], tokenizer_decoder.upsample_rates[3],
            tokenizer_decoder.upsampling_ratios[0], tokenizer_decoder.upsampling_ratios[1]);
    return true;
}

void TTSConfig::load_generation_config(const std::string& gen_config_path) {
    std::string json = read_file_to_string(gen_config_path);
    if (json.empty()) return;

    temperature = json_get_float(json, "temperature", temperature);
    top_k = json_get_int(json, "top_k", top_k);
    top_p = json_get_float(json, "top_p", top_p);
    repetition_penalty = json_get_float(json, "repetition_penalty", repetition_penalty);
}

} // namespace tts
} // namespace qwen_thor
