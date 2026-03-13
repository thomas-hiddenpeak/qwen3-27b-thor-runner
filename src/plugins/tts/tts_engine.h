// tts_engine.h — Qwen3-TTS Inference Engine
//
// Top-level orchestrator:
//   1. Load config + safetensors + tokenizer
//   2. Orchestrate Talker + (future) Speech Tokenizer Decoder
//   3. Provide synthesize() interface: text → codec tokens (Phase 1)
//
// Usage:
//   TTSEngine engine;
//   engine.load_model("/path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice");
//   auto codes = engine.synthesize("Hello world", "serena");

#pragma once

#include "tts_config.h"
#include "tts_talker.h"
#include "tts_tokenizer_decoder.h"
#include "engine/tokenizer.h"
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace tts {

class TTSEngine {
public:
    TTSEngine();
    ~TTSEngine();

    // Load TTS model from directory
    // model_dir must contain: config.json, model.safetensors, generation_config.json
    // Also expects speech_tokenizer/ subdirectory for Phase 2
    void load_model(const std::string& model_dir);

    // Synthesize: text → codec tokens (Phase 1)
    // Returns 2D vector [num_steps][num_code_groups], each entry is a codec token ID
    // speaker: speaker name (e.g., "serena"), must exist in config
    // language: language hint (e.g., "zh", "en"), or "auto"
    std::vector<std::vector<int>> synthesize(
        const std::string& text,
        const std::string& speaker = "serena",
        const std::string& language = "auto",
        int max_new_tokens = 4096);

    // Synthesize: text → WAV file (end-to-end)
    // Returns true on success, false on failure
    bool synthesize_to_wav(
        const std::string& text,
        const std::string& output_path,
        const std::string& speaker = "serena",
        const std::string& language = "auto",
        const std::string& instruct = "",
        int max_new_tokens = 4096);

    bool is_loaded() const { return loaded_; }
    const TTSConfig& config() const { return config_; }

    // Sampling parameter overrides
    void set_sampling(float temperature, int top_k, float top_p, float rep_penalty);
    void set_sub_sampling(float temperature, int top_k, float top_p);

private:
    TTSConfig config_;
    std::unique_ptr<Talker> talker_;
    std::unique_ptr<SpeechTokenizerDecoder> st_decoder_;
    Tokenizer tokenizer_;
    std::string model_dir_;

    // Weight ownership: all cudaMalloc'd pointers tracked here for cleanup
    std::vector<void*> device_weights_;

    cudaStream_t stream_ = 0;
    bool loaded_ = false;

    // Internal methods
    void load_weights(const std::string& model_dir);

    // Build the full chat-template text token sequence for TTS
    // Input: user text
    // Output: token IDs including <|im_start|>assistant\n...text...<|im_end|>\n<|im_start|>assistant\n
    std::vector<int> build_text_tokens(const std::string& text);

    // Build instruct text tokens for VoiceDesign mode
    // <|im_start|>user\n{instruct}<|im_end|>\n<|im_start|>assistant\n...text...<|im_end|>\n<|im_start|>assistant\n
    std::vector<int> build_instruct_text_tokens(const std::string& text,
                                                 const std::string& instruct);
};

} // namespace tts
} // namespace qwen_thor
