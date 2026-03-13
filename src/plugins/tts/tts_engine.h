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
#include "tts_speaker_encoder.h"
#include "tts_voice_manager.h"
#include "engine/tokenizer.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
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

    // Synthesize: text → raw PCM float samples (24kHz, mono)
    // Returns empty vector on failure
    std::vector<float> synthesize_to_pcm(
        const std::string& text,
        const std::string& speaker = "serena",
        const std::string& language = "auto",
        const std::string& instruct = "",
        int max_new_tokens = 4096);

    // Streaming synthesis: generate and decode audio in chunks
    // Calls pcm_callback with PCM chunks as they become available
    // Returns total number of PCM samples generated, 0 on failure
    // pcm_callback receives (data, num_samples) and returns true to continue, false to abort
    using PcmCallback = std::function<bool(const float* data, int num_samples)>;
    int synthesize_streaming(
        const std::string& text,
        const std::string& speaker = "serena",
        const std::string& language = "auto",
        const std::string& instruct = "",
        int max_new_tokens = 4096,
        int chunk_frames = 24,
        PcmCallback pcm_callback = nullptr);

    // Continue synthesis: inject new text and continue decoding without resetting
    // Preserves talker KV cache for voice consistency across segments
    // Returns PCM float samples for the continuation segment
    std::vector<float> continue_to_pcm(
        const std::string& text,
        int max_new_tokens = 4096);

    // Continue synthesis with streaming: inject new text and decode in chunks
    // Preserves talker KV cache for voice consistency across segments
    int continue_streaming(
        const std::string& text,
        int max_new_tokens = 4096,
        int chunk_frames = 24,
        PcmCallback pcm_callback = nullptr);

    int sample_rate() const { return loaded_ ? config_.tokenizer_decoder.output_sample_rate : 24000; }
    bool is_loaded() const { return loaded_; }
    const TTSConfig& config() const { return config_; }

    // Sampling parameter overrides
    void set_sampling(float temperature, int top_k, float top_p, float rep_penalty);
    void set_sub_sampling(float temperature, int top_k, float top_p);

    // ===== Voice Clone (Base model only) =====

    // Extract speaker embedding from raw PCM audio
    // Returns enc_dim-dimensional x-vector, empty on failure
    std::vector<float> extract_speaker_embedding(
        const float* audio, int num_samples, int sample_rate);

    // Register a named voice from audio
    bool register_voice(const std::string& name,
                        const float* audio, int num_samples, int sample_rate);

    // Register a named voice from pre-computed embedding
    bool register_voice_embedding(const std::string& name,
                                   const std::vector<float>& embedding);

    // Delete a registered voice
    bool delete_voice(const std::string& name);

    // List registered voice names
    std::vector<std::string> list_clone_voices() const;

    // Check if a clone voice exists
    bool has_clone_voice(const std::string& name) const;

    // Synthesize with voice clone (using registered voice name)
    std::vector<float> synthesize_voice_clone(
        const std::string& text,
        const std::string& voice_name,
        const std::string& language = "auto",
        int max_new_tokens = 4096);

    // Synthesize with voice clone (using raw embedding)
    std::vector<float> synthesize_voice_clone_embedding(
        const std::string& text,
        const std::vector<float>& speaker_embedding,
        const std::string& language = "auto",
        int max_new_tokens = 4096);

    bool has_speaker_encoder() const { return speaker_encoder_ && speaker_encoder_->is_loaded(); }

private:
    TTSConfig config_;
    std::unique_ptr<Talker> talker_;
    std::unique_ptr<SpeechTokenizerDecoder> st_decoder_;
    std::unique_ptr<SpeakerEncoder> speaker_encoder_;
    VoiceManager voice_manager_;
    Tokenizer tokenizer_;
    std::string model_dir_;

    // Weight ownership: all cudaMalloc'd pointers tracked here for cleanup
    std::vector<void*> device_weights_;

    cudaStream_t stream_ = 0;
    bool loaded_ = false;

    // Persisted decoder left context across streaming calls (for continue_streaming)
    std::vector<std::vector<int>> code_history_;

    // Internal methods
    void load_weights(const std::string& model_dir);
    void load_speaker_encoder_weights(const std::string& model_dir);

    // Build the full chat-template text token sequence for TTS
    // Input: user text
    // Output: token IDs including <|im_start|>assistant\n...text...<|im_end|>\n<|im_start|>assistant\n
    std::vector<int> build_text_tokens(const std::string& text);

    // Build instruct-only tokens: <|im_start|>user\n{instruct}<|im_end|>\n
    // These are embedded as pure text-track in prefill (no codec counterpart)
    std::vector<int> build_instruct_tokens(const std::string& instruct);
};

} // namespace tts
} // namespace qwen_thor
