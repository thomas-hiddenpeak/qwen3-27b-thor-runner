#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace qwen_thor {
namespace tts {

// ============================================================
// Speaker Encoder Configuration (ECAPA-TDNN, Base model only)
// ============================================================
struct SpeakerEncoderConfig {
    int mel_dim              = 128;
    int enc_dim              = 2048;  // output speaker embedding dimension
    std::vector<int> enc_channels     = {512, 512, 512, 512, 1536};
    std::vector<int> enc_kernel_sizes = {5, 3, 3, 3, 1};
    std::vector<int> enc_dilations    = {1, 2, 3, 4, 1};
    int enc_attention_channels = 128;
    int enc_res2net_scale    = 8;
    int enc_se_channels      = 128;
    int sample_rate          = 24000;
};

// ============================================================
// Talker Configuration (28-layer GQA transformer)
// ============================================================
struct TalkerConfig {
    int hidden_size        = 2048;
    int num_hidden_layers  = 28;
    int num_attention_heads = 16;   // Q heads
    int num_kv_heads       = 8;     // KV heads (GQA)
    int head_dim           = 128;
    int intermediate_size  = 6144;  // SwiGLU FFN
    int vocab_size         = 3072;  // codec vocab (2048 codebook + 1024 special)
    int text_vocab_size    = 151936;
    int text_hidden_size   = 2048;
    float rms_norm_eps     = 1e-6f;
    float rope_theta       = 1000000.0f;
    int max_position_embeddings = 32768;
    int num_code_groups    = 16;
    bool attention_bias    = false;

    // MRoPE sections (interleaved=True)
    int mrope_sections[3]  = {24, 20, 20};  // sum = 64 (head_dim/2)

    // Token IDs — codec track
    int codec_pad_id       = 2148;
    int codec_bos_id       = 2149;
    int codec_eos_token_id = 2150;
    int codec_think_id     = 2154;
    int codec_nothink_id   = 2155;
    int codec_think_bos_id = 2156;
    int codec_think_eos_id = 2157;

    // Speaker IDs (codec embedding indices)
    std::unordered_map<std::string, int> spk_id;

    // Language IDs (codec embedding indices)
    std::unordered_map<std::string, int> codec_language_id;

    // Speaker → dialect mapping (empty string = standard speaker)
    std::unordered_map<std::string, std::string> spk_is_dialect;
};

// ============================================================
// Code Predictor Configuration (5-layer GQA transformer)
// ============================================================
struct CodePredictorConfig {
    int hidden_size        = 1024;
    int num_hidden_layers  = 5;
    int num_attention_heads = 16;
    int num_kv_heads       = 8;
    int head_dim           = 128;
    int intermediate_size  = 3072;
    int vocab_size         = 2048;  // codebook size
    float rms_norm_eps     = 1e-6f;
    float rope_theta       = 1000000.0f;
    int max_position_embeddings = 65536;
    bool attention_bias    = false;
};

// ============================================================
// Speech Tokenizer Decoder Configuration
// ============================================================
struct TokenizerDecoderConfig {
    // RVQ
    int num_quantizers     = 16;
    int num_semantic_quantizers = 1;
    int codebook_size      = 2048;
    int semantic_codebook_size = 4096;  // unused for decode (same 2048)
    int codebook_dim       = 512;       // output dim after project_out
    int vq_hidden_dim      = 256;       // internal codebook dimension

    // Pre-conv
    int latent_dim         = 1024;      // pre_conv output

    // Pre-transformer
    int hidden_size        = 512;
    int num_hidden_layers  = 8;
    int num_attention_heads = 16;       // MHA (not GQA)
    int num_kv_heads       = 16;
    int head_dim           = 64;
    int intermediate_size  = 1024;
    float rms_norm_eps     = 1e-5f;
    float rope_theta       = 10000.0f;
    int sliding_window     = 72;
    float layer_scale_init = 0.01f;

    // BigVGAN
    int decoder_dim        = 1536;
    int upsample_rates[4]  = {8, 5, 4, 3};    // BigVGAN strides
    int upsampling_ratios[2] = {2, 2};         // pre-upsample strides

    // Audio
    int output_sample_rate = 24000;
    int decode_upsample_rate = 1920;  // total upsample factor

    // Chunked decode
    int chunk_size = 300;
    int left_context_size = 25;
};

// ============================================================
// Top-level TTS Configuration
// ============================================================
struct TTSConfig {
    TalkerConfig talker;
    CodePredictorConfig code_predictor;
    TokenizerDecoderConfig tokenizer_decoder;
    SpeakerEncoderConfig speaker_encoder;  // ECAPA-TDNN (Base model only)

    // Text token IDs
    int im_start_token_id  = 151644;
    int im_end_token_id    = 151645;
    int tts_pad_token_id   = 151671;
    int tts_bos_token_id   = 151672;
    int tts_eos_token_id   = 151673;
    int assistant_token_id = 77091;

    // Model info
    std::string model_type     = "qwen3_tts";
    std::string tts_model_type = "custom_voice";

    // Sampling defaults
    float temperature       = 0.9f;
    int   top_k             = 50;
    float top_p             = 1.0f;
    float repetition_penalty = 1.05f;
    float sub_temperature   = 0.9f;
    int   sub_top_k         = 50;
    float sub_top_p         = 1.0f;

    // Load from config.json (talker + code predictor)
    bool load_from_json(const std::string& config_path);

    // Load from speech_tokenizer/config.json
    bool load_tokenizer_config(const std::string& config_path);

    // Load from generation_config.json
    void load_generation_config(const std::string& gen_config_path);
};

} // namespace tts
} // namespace qwen_thor
