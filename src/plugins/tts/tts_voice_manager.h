// tts_voice_manager.h — Voice Clone x-vector management
//
// Saves/loads speaker embeddings (x-vectors) to/from JSON files.
// Each registered voice is stored as: {voice_dir}/{name}.json
// containing the enc_dim-dimensional float vector.

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <mutex>
#include <cstdio>

namespace qwen_thor {
namespace tts {

class VoiceManager {
public:
    VoiceManager() = default;

    void set_directory(const std::string& dir) {
        voice_dir_ = dir;
        std::filesystem::create_directories(dir);
        load_all();
    }

    // Register a new voice with pre-computed embedding
    bool register_voice(const std::string& name, const std::vector<float>& embedding) {
        if (name.empty() || embedding.empty()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        voices_[name] = embedding;
        return save_voice(name, embedding);
    }

    // Delete a registered voice
    bool delete_voice(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = voices_.find(name);
        if (it == voices_.end()) return false;
        voices_.erase(it);
        std::string path = voice_dir_ + "/" + name + ".json";
        return std::filesystem::remove(path);
    }

    // Get embedding for a registered voice
    const std::vector<float>* get_embedding(const std::string& name) const {
        auto it = voices_.find(name);
        return it != voices_.end() ? &it->second : nullptr;
    }

    bool has_voice(const std::string& name) const {
        return voices_.count(name) > 0;
    }

    // List all registered voice names
    std::vector<std::string> list_voices() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : voices_) names.push_back(name);
        std::sort(names.begin(), names.end());
        return names;
    }

private:
    std::string voice_dir_;
    std::unordered_map<std::string, std::vector<float>> voices_;
    std::mutex mutex_;

    bool save_voice(const std::string& name, const std::vector<float>& embedding) {
        std::string path = voice_dir_ + "/" + name + ".json";
        std::ofstream f(path);
        if (!f.is_open()) {
            fprintf(stderr, "[VoiceManager] ERROR: cannot write %s\n", path.c_str());
            return false;
        }
        f << "{\"name\":\"" << name << "\",\"dim\":" << embedding.size() << ",\"embedding\":[";
        for (size_t i = 0; i < embedding.size(); i++) {
            if (i > 0) f << ",";
            f << embedding[i];
        }
        f << "]}\n";
        fprintf(stderr, "[VoiceManager] Saved voice '%s' (%zu-dim) to %s\n",
                name.c_str(), embedding.size(), path.c_str());
        return true;
    }

    void load_all() {
        if (!std::filesystem::exists(voice_dir_)) return;
        for (const auto& entry : std::filesystem::directory_iterator(voice_dir_)) {
            if (entry.path().extension() != ".json") continue;
            std::string name = entry.path().stem().string();
            auto embedding = load_voice_file(entry.path().string());
            if (!embedding.empty()) {
                voices_[name] = std::move(embedding);
                fprintf(stderr, "[VoiceManager] Loaded voice '%s' (%zu-dim)\n",
                        name.c_str(), voices_[name].size());
            }
        }
        fprintf(stderr, "[VoiceManager] %zu voices loaded from %s\n",
                voices_.size(), voice_dir_.c_str());
    }

    static std::vector<float> load_voice_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return {};

        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        // Simple JSON parsing: find "embedding":[...] array
        auto pos = content.find("\"embedding\"");
        if (pos == std::string::npos) return {};
        pos = content.find('[', pos);
        if (pos == std::string::npos) return {};
        auto end = content.find(']', pos);
        if (end == std::string::npos) return {};

        std::string arr = content.substr(pos + 1, end - pos - 1);
        std::vector<float> result;
        std::istringstream ss(arr);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try { result.push_back(std::stof(token)); }
            catch (...) { break; }
        }
        return result;
    }
};

} // namespace tts
} // namespace qwen_thor
