// asr_config.h — Qwen3-ASR 模型配置
//
// 解析 Qwen3-ASR config.json, 提供 Encoder + Decoder 的全部参数。
// Audio Encoder: Whisper 风格 Conv2D + 24-layer bidirectional Transformer
// Text Decoder:  Qwen3 28-layer GQA + MRoPE + SwiGLU

#pragma once

#include <string>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace qwen_thor {
namespace asr {

struct ASRConfig {
    // ========== Audio Encoder (Whisper 风格) ==========
    int num_mel_bins            = 128;
    int encoder_layers          = 24;
    int encoder_d_model         = 1024;   // d_model
    int encoder_attention_heads = 16;
    int encoder_head_dim        = 64;     // = d_model / attention_heads
    int encoder_ffn_dim         = 4096;
    int downsample_hidden_size  = 480;
    int max_source_positions    = 1500;   // sinusoidal PE 最大长度
    int n_window                = 50;     // training window
    int n_window_infer          = 800;    // inference attention window
    int conv_chunksize          = 500;    // conv chunking for memory
    int output_dim              = 2048;   // proj2 output → decoder hidden

    // Conv2D 下采样: 3 层 stride=2, padding=1 → 总 8× 下采样
    // mel_bins=128 → (128+1)/2=64 → (64+1)/2=32 → (32+1)/2=16 = freq_after_conv
    int freq_after_conv() const {
        return (((num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2;  // = 16
    }
    int conv_out_features() const {
        return downsample_hidden_size * freq_after_conv();  // 480 * 16 = 7680
    }

    // Conv2D stride-2 output dimension: (x-1)/2+1 for x>0, else 0
    // 注意: Python 用 floor division (//), C++ 整除对负数截断方向不同,
    //       当 x=0 时 (x-1)/2 在 C++ 为 0 而 Python 为 -1, 需特殊处理
    static int conv_output_size(int x) {
        return x <= 0 ? 0 : (x - 1) / 2 + 1;
    }

    // _get_feat_extract_output_lengths: 计算 CNN 后序列长度
    // 来自 Python: input_lengths_leave = input_lengths % 100
    //              feat_lengths = (input_lengths_leave - 1) // 2 + 1
    //              output = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
    //                       + (input_lengths // 100) * 13
    static int get_output_length(int mel_frames) {
        int leave = mel_frames % 100;
        int t = conv_output_size(conv_output_size(conv_output_size(leave)));
        return t + (mel_frames / 100) * 13;
    }

    // ========== Text Decoder (Qwen3) ==========
    int decoder_layers          = 28;
    int decoder_hidden_size     = 2048;
    int decoder_num_attention_heads = 16; // Q heads
    int decoder_num_kv_heads    = 8;      // K/V heads (GQA)
    int decoder_head_dim        = 128;
    int decoder_intermediate_size = 6144; // SwiGLU
    int vocab_size              = 151936;
    float rms_norm_eps          = 1e-6f;
    float rope_theta            = 1000000.0f;  // 1e6
    bool tie_word_embeddings    = true;

    // MRoPE
    bool mrope_interleaved      = true;
    std::array<int, 3> mrope_section = {24, 20, 20};

    // ========== Token IDs ==========
    static constexpr int AUDIO_START_TOKEN  = 151669;
    static constexpr int AUDIO_END_TOKEN    = 151670;
    static constexpr int AUDIO_PAD_TOKEN    = 151676;  // <|audio_pad|>

    // EOS tokens
    static constexpr int EOS_TOKEN_1        = 151643;  // <|endoftext|>
    static constexpr int EOS_TOKEN_2        = 151645;  // <|im_end|>

    // ========== 派生尺寸 ==========
    int decoder_q_dim() const { return decoder_num_attention_heads * decoder_head_dim; }  // 2048
    int decoder_kv_dim() const { return decoder_num_kv_heads * decoder_head_dim; }        // 1024

    // ========== 从 config.json 加载 ==========
    // 解析 model_dir/config.json, 填充所有字段
    // config.json 结构: { "thinker_config": { "audio_config": {...}, "text_config": {...} } }
    bool load_from_json(const std::string& config_path);

    // 从模型目录加载 (自动追加 /config.json)
    bool load_from_model_dir(const std::string& model_dir) {
        std::string path = model_dir;
        if (path.back() != '/') path += '/';
        return load_from_json(path + "config.json");
    }
};

} // namespace asr
} // namespace qwen_thor
