// vad_config.h — VAD 配置 (Phase 1)
//
// FSMN-VAD 模型参数 + 判决状态机配置

#pragma once

#include <string>

namespace qwen_thor {
namespace asr {

struct VadConfig {
    // ========== 模型路径 ==========
    std::string model_path;    // fsmn_vad.safetensors
    std::string cmvn_path;     // cmvn.safetensors (CMVN 归一化参数)

    // ========== 前端参数 ==========
    int sample_rate        = 16000;
    int n_mels             = 80;
    int frame_length_ms    = 25;
    int frame_shift_ms     = 10;
    int lfr_m              = 5;    // LFR (Low Frame Rate): 拼接 5 帧
    int lfr_n              = 1;    // LFR stride

    // ========== FSMN 模型参数 ==========
    int input_dim          = 400;  // 80 * 5 (LFR)
    int input_affine_dim   = 140;
    int fsmn_layers        = 4;
    int linear_dim         = 250;
    int proj_dim           = 128;
    int lorder             = 20;   // left context
    int rorder             = 0;    // right context (0 = causal)
    int output_affine_dim  = 140;
    int output_dim         = 248;  // 分类数 (含 silence/speech/noise 等)

    // ========== 判决状态机参数 ==========
    int   window_size_ms             = 200;
    int   sil_to_speech_time_thres   = 150;    // ms
    int   speech_to_sil_time_thres   = 150;    // ms
    int   max_end_silence_time       = 800;    // ms
    int   max_single_segment_time    = 60000;  // ms
    float speech_noise_thres         = 0.6f;   // 语音/噪声判决门限
    int   lookback_time_start_point  = 200;    // ms
    int   lookahead_time_end_point   = 100;    // ms
    int   max_start_silence_time     = 3000;   // ms
    int   silence_pdf_num            = 1;      // sil_pdf_ids 数量
    int   sil_pdf_ids[1]             = {0};    // 静音类别 ID

    // ========== 派生 ==========
    int frame_samples() const { return sample_rate * frame_shift_ms / 1000; }  // 160
    int window_samples() const { return sample_rate * frame_length_ms / 1000; } // 400
};

} // namespace asr
} // namespace qwen_thor
