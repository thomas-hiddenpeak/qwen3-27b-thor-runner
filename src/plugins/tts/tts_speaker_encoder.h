// tts_speaker_encoder.h — ECAPA-TDNN Speaker Encoder (CPU inference)
//
// Extracts speaker embedding (x-vector) from reference audio.
// Architecture: mel → Conv1d → 3×SERes2Net → MFA → ASP → FC → enc_dim
//
// Used by voice clone mode (Base model only).
// CPU-only: speaker encoding runs once per voice registration, not per generation.

#pragma once

#include "tts_config.h"
#include <vector>
#include <string>
#include <cstdint>

namespace qwen_thor {
namespace tts {

// ============================================================
// Conv1d weight storage (always kernel_size=k, padding="same", reflect)
// ============================================================
struct Conv1dWeights {
    std::vector<float> weight;  // [out_ch, in_ch, kernel_size]
    std::vector<float> bias;    // [out_ch] (optional, empty if no bias)
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 1;
    int dilation = 1;
};

// ============================================================
// Speaker Encoder (ECAPA-TDNN)
// ============================================================
class SpeakerEncoder {
public:
    SpeakerEncoder() = default;

    // Load weights from safetensors data
    // weight_data: map of "speaker_encoder.xxx" → (bf16_ptr, num_elements)
    bool load_weights(
        const std::vector<std::pair<std::string, std::pair<const uint16_t*, size_t>>>& weights);

    // Extract speaker embedding from raw PCM audio
    // audio: float32 samples, sample_rate: must be 24kHz (or will be resampled)
    // Returns: enc_dim-dimensional speaker embedding
    std::vector<float> extract(const float* audio, int num_samples, int sample_rate);

    bool is_loaded() const { return loaded_; }
    int enc_dim() const { return config_.enc_dim; }

    void set_config(const SpeakerEncoderConfig& cfg) { config_ = cfg; }

private:
    SpeakerEncoderConfig config_;
    bool loaded_ = false;

    // ===== Weights =====
    // blocks[0]: initial TDNN (mel_dim→enc_channels[0], k=5, d=1)
    Conv1dWeights block0_conv_;

    // blocks[1-3]: SE-Res2Net blocks
    struct SERes2NetWeights {
        Conv1dWeights tdnn1;   // Conv1d(in, out, k=1, d=1)
        std::vector<Conv1dWeights> res2net_blocks;  // scale-1 Conv1d(sub_ch, sub_ch, k, d)
        Conv1dWeights tdnn2;   // Conv1d(out, out, k=1, d=1)
        Conv1dWeights se_conv1;  // Conv1d(out, se_ch, k=1)
        Conv1dWeights se_conv2;  // Conv1d(se_ch, out, k=1)
    };
    std::vector<SERes2NetWeights> se_res2net_blocks_;  // 3 blocks

    // MFA: TimeDelayNet(1536→1536, k=1, d=1)
    Conv1dWeights mfa_conv_;

    // ASP (Attentive Statistical Pooling)
    Conv1dWeights asp_tdnn_conv_;  // Conv1d(4608→128, k=1)
    Conv1dWeights asp_conv_;       // Conv1d(128→1536, k=1)

    // FC: Conv1d(3072→enc_dim, k=1)
    Conv1dWeights fc_conv_;

    // ===== Mel Spectrogram =====
    std::vector<float> mel_filterbank_;  // [n_mels, n_fft/2+1]
    std::vector<float> hann_window_;     // [n_fft]

    // ===== Internal =====
    // Conv1d forward with reflect padding
    void conv1d_forward(const std::vector<float>& input, int in_ch, int T,
                        const Conv1dWeights& w,
                        std::vector<float>& output);

    // Compute TTS-style mel spectrogram
    // Output: [T_mel, n_mels], row-major
    void compute_mel(const float* audio, int num_samples,
                     std::vector<float>& mel_out, int& num_frames);

    // Basic forward operations
    void relu_inplace(std::vector<float>& x);
    void sigmoid_inplace(std::vector<float>& x);
};

} // namespace tts
} // namespace qwen_thor
