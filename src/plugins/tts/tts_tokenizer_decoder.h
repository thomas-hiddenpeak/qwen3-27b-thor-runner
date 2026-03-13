// tts_tokenizer_decoder.h — Speech Tokenizer Decoder (Codes → PCM Audio)
//
// Architecture (Qwen3-TTS-Tokenizer-12Hz V2 Decoder):
//   codes [16, T] → RVQ dequant → pre_conv → pre_transformer → upsample → BigVGAN → PCM
//
// Pipeline:
//   1. RVQ Dequantization: 16 codebooks → sum → output_proj → [512, T]
//   2. Pre-conv: CausalConv1d(512→1024, k=3) → [1024, T]
//   3. Pre-transformer: 8L sliding-window causal transformer (512h, 16 heads, SwiGLU)
//   4. Upsample: 2× ConvTranspose+ConvNeXt → [1024, 4T]
//   5. BigVGAN: 4 stages (SnakeBeta+TransConv+ResBlocks) → [1, 1920T]
//   6. Clamp [-1, 1] → 24kHz PCM
//
// All weights F32. Loaded from speech_tokenizer/model.safetensors.

#pragma once

#include "tts_config.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace qwen_thor {
namespace tts {

// Per-layer weights for pre-transformer (8 layers)
struct TokenizerTransformerLayerWeights {
    float* input_layernorm_w = nullptr;     // [hidden=512]
    float* q_proj_w = nullptr;              // [1024, 512] (16 heads × 64)
    float* k_proj_w = nullptr;              // [1024, 512]
    float* v_proj_w = nullptr;              // [1024, 512]
    float* o_proj_w = nullptr;              // [512, 1024]
    float* attn_layer_scale = nullptr;      // [512]
    float* post_attention_layernorm_w = nullptr; // [512]
    float* gate_proj_w = nullptr;           // [1024, 512]
    float* up_proj_w = nullptr;             // [1024, 512]
    float* down_proj_w = nullptr;           // [512, 1024]
    float* mlp_layer_scale = nullptr;       // [512]
};

// BigVGAN ResBlock weights
struct ResBlockWeights {
    float* act1_alpha = nullptr;            // [dim]
    float* act1_beta = nullptr;
    float* conv1_w = nullptr;               // [dim, dim, k] (k=7 with dilation)
    float* conv1_b = nullptr;               // [dim]
    float* act2_alpha = nullptr;
    float* act2_beta = nullptr;
    float* conv2_w = nullptr;               // [dim, dim, 1]
    float* conv2_b = nullptr;               // [dim]
};

// BigVGAN decoder stage weights
struct DecoderStageWeights {
    float* snake_alpha = nullptr;           // [in_dim]
    float* snake_beta = nullptr;
    float* transconv_w = nullptr;           // [in_dim, out_dim, kernel]
    float* transconv_b = nullptr;           // [out_dim]
    ResBlockWeights res_blocks[3];          // dilations 1, 3, 9
};

// ConvNeXt block weights
struct ConvNeXtWeights {
    float* dwconv_w = nullptr;              // [dim, 1, 7] groups=dim
    float* dwconv_b = nullptr;              // [dim]
    float* norm_w = nullptr;                // [dim] LayerNorm
    float* norm_b = nullptr;                // [dim]
    float* pwconv1_w = nullptr;             // [4*dim, dim]
    float* pwconv1_b = nullptr;             // [4*dim]
    float* pwconv2_w = nullptr;             // [dim, 4*dim]
    float* pwconv2_b = nullptr;             // [dim]
    float* gamma = nullptr;                 // [dim]
};

// Upsample stage weights
struct UpsampleStageWeights {
    float* transconv_w = nullptr;           // [dim, dim, factor]
    float* transconv_b = nullptr;           // [dim]
    ConvNeXtWeights convnext;
};

class SpeechTokenizerDecoder {
public:
    SpeechTokenizerDecoder();
    ~SpeechTokenizerDecoder();

    // Load weights from speech_tokenizer/model.safetensors
    bool load_weights(const std::string& tokenizer_dir,
                      const TokenizerDecoderConfig& config);

    // Initialize (allocate workspace, cuBLAS handle)
    void initialize(cudaStream_t stream = 0);

    // Decode: codes [num_groups=16, T] → PCM samples [-1, 1]
    // Returns PCM samples at 24kHz
    std::vector<float> decode(const int* codes_cpu, int num_groups, int num_frames,
                              cudaStream_t stream = 0);

    // Write PCM to WAV file (24kHz, 16-bit, mono)
    static bool write_wav(const std::string& path, const std::vector<float>& pcm,
                          int sample_rate = 24000);

    bool is_loaded() const { return loaded_; }

private:
    TokenizerDecoderConfig config_;
    cublasHandle_t cublas_ = nullptr;
    bool loaded_ = false;

    // ===== Weights =====

    // RVQ codebooks (pre-computed: embed_sum / cluster_usage)
    float* semantic_codebook_ = nullptr;    // [2048, 256]
    float* acoustic_codebooks_ = nullptr;   // [15 * 2048, 256] contiguous
    float* semantic_output_proj_w_ = nullptr; // [512, 256]
    float* acoustic_output_proj_w_ = nullptr; // [512, 256]

    // Pre-conv
    float* pre_conv_w_ = nullptr;           // [1024, 512, 3]
    float* pre_conv_b_ = nullptr;           // [1024]

    // Pre-transformer
    float* pt_input_proj_w_ = nullptr;      // [512, 1024]
    float* pt_input_proj_b_ = nullptr;      // [512]
    float* pt_output_proj_w_ = nullptr;     // [1024, 512]
    float* pt_output_proj_b_ = nullptr;     // [1024]
    float* pt_norm_w_ = nullptr;            // [512]
    TokenizerTransformerLayerWeights pt_layers_[8];

    // Upsample (2 stages)
    UpsampleStageWeights upsample_[2];

    // BigVGAN decoder
    float* initial_conv_w_ = nullptr;       // [1536, 1024, 7]
    float* initial_conv_b_ = nullptr;       // [1536]
    DecoderStageWeights decoder_stages_[4]; // 4 stages
    float* final_snake_alpha_ = nullptr;    // [96]
    float* final_snake_beta_ = nullptr;
    float* final_conv_w_ = nullptr;         // [1, 96, 7]
    float* final_conv_b_ = nullptr;         // [1]

    // ===== Workspace =====
    float* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    // Weight ownership tracking
    std::vector<void*> device_ptrs_;

    // ===== Internal methods =====

    // RVQ dequantization: codes [16, T] → latent [512, T]
    void rvq_dequant(const int* d_codes, int T, float* d_output, cudaStream_t s);

    // Pre-conv: CausalConv1d(512→1024, k=3)
    void run_pre_conv(float* input, int T, float* output, cudaStream_t s);

    // Pre-transformer forward (8L)
    void run_pre_transformer(float* input, int T, float* output, cudaStream_t s);

    // Single transformer layer
    void transformer_layer_forward(const TokenizerTransformerLayerWeights& w,
                                   float* hidden, int T, float* workspace, cudaStream_t s);

    // Upsample stages
    void run_upsample(float* input, int T_in, float* output, int& T_out, cudaStream_t s);

    // BigVGAN decoder
    void run_bigvgan(float* input, int T_in, float* output, int& T_out, cudaStream_t s);

    // Chunked decode to handle long sequences
    std::vector<float> chunked_decode(const int* d_codes, int T, cudaStream_t s);
};

} // namespace tts
} // namespace qwen_thor
