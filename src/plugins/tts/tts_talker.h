// tts_talker.h — Qwen3-TTS Talker (28-layer GQA transformer + CodePredictor)
//
// Architecture:
//   - 28-layer Qwen3 decoder with GQA (16Q/8KV, head_dim=128)
//   - Dual-track embedding: text_projection(text_embedding) + codec_embedding
//   - MRoPE (interleaved sections [24,20,20]) — degenerates to 1D for TTS
//   - SwiGLU MLP
//   - codec_head: Linear(hidden→vocab) for group-0 logit prediction
//   - CodePredictor: 5-layer transformer generating groups 1-15
//
// Weight prefixes:
//   talker.model.layers.{i}.*
//   talker.model.text_embedding.weight
//   talker.model.codec_embedding.weight
//   talker.model.norm.weight
//   talker.text_projection.linear_fc1.weight/bias
//   talker.text_projection.linear_fc2.weight/bias
//   talker.codec_head.weight
//   talker.code_predictor.*

#pragma once

#include "tts_config.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

namespace qwen_thor {
namespace tts {

// ============================================================
// Per-layer weights for Talker (28 layers, GQA, no attn bias)
// ============================================================
struct TalkerLayerWeights {
    __nv_bfloat16* input_layernorm_w = nullptr;          // [hidden_size]
    __nv_bfloat16* q_proj_w = nullptr;                   // [q_dim, hidden_size]
    __nv_bfloat16* k_proj_w = nullptr;                   // [kv_dim, hidden_size]
    __nv_bfloat16* v_proj_w = nullptr;                   // [kv_dim, hidden_size]
    __nv_bfloat16* o_proj_w = nullptr;                   // [hidden_size, q_dim]
    __nv_bfloat16* q_norm_w = nullptr;                   // [head_dim]
    __nv_bfloat16* k_norm_w = nullptr;                   // [head_dim]
    __nv_bfloat16* post_attention_layernorm_w = nullptr;  // [hidden_size]
    __nv_bfloat16* gate_proj_w = nullptr;                // [intermediate, hidden]
    __nv_bfloat16* up_proj_w = nullptr;                  // [intermediate, hidden]
    __nv_bfloat16* down_proj_w = nullptr;                // [hidden, intermediate]
};

// ============================================================
// Per-layer weights for CodePredictor (5 layers)
// ============================================================
struct CodePredictorLayerWeights {
    __nv_bfloat16* input_layernorm_w = nullptr;          // [hidden_size=1024]
    __nv_bfloat16* q_proj_w = nullptr;                   // [q_dim=2048, hidden=1024]
    __nv_bfloat16* k_proj_w = nullptr;                   // [kv_dim=1024, hidden=1024]
    __nv_bfloat16* v_proj_w = nullptr;                   // [kv_dim=1024, hidden=1024]
    __nv_bfloat16* o_proj_w = nullptr;                   // [hidden=1024, q_dim=2048]
    __nv_bfloat16* q_norm_w = nullptr;                   // [head_dim=128]
    __nv_bfloat16* k_norm_w = nullptr;                   // [head_dim=128]
    __nv_bfloat16* post_attention_layernorm_w = nullptr;  // [hidden_size=1024]
    __nv_bfloat16* gate_proj_w = nullptr;                // [intermediate=3072, hidden=1024]
    __nv_bfloat16* up_proj_w = nullptr;                  // [intermediate=3072, hidden=1024]
    __nv_bfloat16* down_proj_w = nullptr;                // [hidden=1024, intermediate=3072]
};

class Talker {
public:
    Talker(const TTSConfig& config, int max_seq_len = 4096);
    ~Talker();

    // ===== Weight Binding =====

    // Talker shared weights
    void set_text_embedding(__nv_bfloat16* w);            // [text_vocab_size, text_hidden_size]
    void set_codec_embedding(__nv_bfloat16* w);           // [vocab_size, hidden_size]
    void set_text_projection(__nv_bfloat16* fc1_w,        // [text_hidden_size, text_hidden_size]
                             __nv_bfloat16* fc1_b,        // [text_hidden_size]
                             __nv_bfloat16* fc2_w,        // [hidden_size, text_hidden_size]
                             __nv_bfloat16* fc2_b);       // [hidden_size]
    void set_final_norm(__nv_bfloat16* w);                // [hidden_size]
    void set_codec_head(__nv_bfloat16* w);                // [vocab_size, hidden_size]
    void set_talker_layer_weights(int layer_idx, const TalkerLayerWeights& w);

    // CodePredictor weights
    void set_code_predictor_projection(__nv_bfloat16* w,  // [cp_hidden, hidden_size]
                                       __nv_bfloat16* b); // [cp_hidden]
    void set_code_predictor_final_norm(__nv_bfloat16* w); // [cp_hidden]
    void set_code_predictor_layer_weights(int layer_idx, const CodePredictorLayerWeights& w);
    void set_code_predictor_lm_head(int group_idx,        // 0-14 (for groups 1-15)
                                    __nv_bfloat16* w);    // [cp_vocab, cp_hidden]
    void set_code_predictor_codec_embedding(int group_idx, // 0-14
                                            __nv_bfloat16* w); // [cp_vocab, hidden_size]

    // ===== Initialization =====
    void initialize(cudaStream_t stream = 0);
    void reset();   // Reset all KV caches for new generation

    // ===== Generation =====

    // Build prefill embeddings for CustomVoice mode
    // text_ids: tokenized text including <|im_start|>assistant\n...text...<|im_end|>...
    // text_len: number of text tokens
    // speaker: speaker name (looked up in config), empty = no speaker
    // language: language name (looked up in config), "auto" = no language
    // Returns prefill embedding length
    int build_prefill(const int* text_ids_cpu, int text_len,
                      const std::string& speaker,
                      const std::string& language,
                      cudaStream_t stream = 0);

    // Run prefill through Talker transformer
    // After this call, KV cache is populated and logits are produced
    void forward_prefill(cudaStream_t stream = 0);

    // Run one decode step: sample group_0, run CodePredictor for groups 1-15,
    // combine embeddings, inject trailing text, forward transformer
    // Returns the 16-group codec for this step, or -1 if EOS
    // codec_out: [num_code_groups] (host), codec_out[0] = group_0 sampled token
    int forward_decode_step(int* codec_out, cudaStream_t stream = 0);

    // Get current generation length
    int generation_step() const { return generation_step_; }

    // Get max generation length
    int max_new_tokens() const { return max_new_tokens_; }
    void set_max_new_tokens(int n) { max_new_tokens_ = n; }

    // Sampling parameters
    void set_sampling(float temperature, int top_k, float top_p, float rep_penalty);
    void set_sub_sampling(float temperature, int top_k, float top_p);

private:
    TTSConfig config_;
    int max_seq_len_;
    int max_new_tokens_ = 4096;
    cublasHandle_t cublas_handle_ = nullptr;

    // ===== Talker Weights =====
    __nv_bfloat16* text_embedding_w_ = nullptr;
    __nv_bfloat16* codec_embedding_w_ = nullptr;
    __nv_bfloat16* text_proj_fc1_w_ = nullptr;
    __nv_bfloat16* text_proj_fc1_b_ = nullptr;
    __nv_bfloat16* text_proj_fc2_w_ = nullptr;
    __nv_bfloat16* text_proj_fc2_b_ = nullptr;
    __nv_bfloat16* final_norm_w_ = nullptr;
    __nv_bfloat16* codec_head_w_ = nullptr;
    std::vector<TalkerLayerWeights> talker_layer_weights_;

    // ===== CodePredictor Weights =====
    __nv_bfloat16* cp_projection_w_ = nullptr;     // small_to_mtp_projection
    __nv_bfloat16* cp_projection_b_ = nullptr;
    __nv_bfloat16* cp_final_norm_w_ = nullptr;
    std::vector<CodePredictorLayerWeights> cp_layer_weights_;
    std::vector<__nv_bfloat16*> cp_lm_heads_;      // [num_code_groups-1]
    std::vector<__nv_bfloat16*> cp_codec_embeddings_; // [num_code_groups-1]

    // ===== Talker KV Cache =====
    std::vector<__nv_bfloat16*> talker_k_cache_;    // [num_layers], each [max_seq_len, kv_heads, head_dim]
    std::vector<__nv_bfloat16*> talker_v_cache_;
    int talker_cache_len_ = 0;

    // ===== CodePredictor KV Cache =====
    // Max length per talker step: 2 (prefill) + 15 (max decode) = 17
    std::vector<__nv_bfloat16*> cp_k_cache_;
    std::vector<__nv_bfloat16*> cp_v_cache_;
    int cp_cache_len_ = 0;

    // ===== Generation State =====
    int generation_step_ = 0;
    int prefill_len_ = 0;

    // Trailing text hidden states (pre-computed text embeddings for streaming injection)
    __nv_bfloat16* trailing_text_hidden_ = nullptr;  // [max_text_len, hidden_size]
    int trailing_text_len_ = 0;

    // TTS special embeddings (pre-computed at build_prefill time)
    __nv_bfloat16* tts_pad_embed_ = nullptr;          // [hidden_size]

    // Past hidden state (last hidden from talker, fed to CodePredictor)
    __nv_bfloat16* past_hidden_ = nullptr;             // [1, hidden_size]

    // ===== Workspace =====
    __nv_bfloat16* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    // Prefill embeddings (built by build_prefill)
    __nv_bfloat16* prefill_embeds_ = nullptr;          // [prefill_len, hidden_size]
    int* position_ids_ = nullptr;                      // [3, max_seq_len]

    // Logits buffer
    __nv_bfloat16* logits_ = nullptr;                  // [talker.vocab_size]
    __nv_bfloat16* cp_logits_ = nullptr;               // [code_predictor.vocab_size]

    // Token IDs (managed memory for GPU→CPU)
    int* sampled_token_ = nullptr;                     // [1]

    // Repetition penalty tracking
    std::vector<int> generated_tokens_;                 // history for rep_penalty

    // ===== Pre-allocated Decode Buffers (avoid per-step cudaMalloc) =====
    __nv_bfloat16* decode_all_embeds_ = nullptr;       // [num_code_groups, hidden_size]
    __nv_bfloat16* decode_input_embeds_ = nullptr;     // [hidden_size]
    int* decode_token_gpu_ = nullptr;                  // [1] for embedding lookup
    int* decode_pos_gpu_ = nullptr;                    // [3] for position IDs
    int* rep_ids_gpu_ = nullptr;                       // [max_new_tokens] for repetition penalty
    // CodePredictor pre-allocated buffers
    __nv_bfloat16* cp_input_buf_ = nullptr;            // [2, talker_hidden_size]
    __nv_bfloat16* cp_hidden_buf_ = nullptr;           // [2, cp_hidden_size]
    __nv_bfloat16* cp_embed_buf_ = nullptr;            // [talker_hidden_size]
    __nv_bfloat16* cp_decode_hidden_ = nullptr;        // [cp_hidden_size]
    int* cp_pos_gpu_ = nullptr;                        // [3 * max_cp_seq_len]

    // Sampling parameters
    float temperature_ = 0.9f;
    int top_k_ = 50;
    float top_p_ = 1.0f;
    float rep_penalty_ = 1.05f;
    float sub_temperature_ = 0.9f;
    int sub_top_k_ = 50;
    float sub_top_p_ = 1.0f;

    bool initialized_ = false;

    // ===== Internal Helpers =====
    void talker_layer_forward_prefill(int layer_idx, __nv_bfloat16* hidden,
                                     const int* pos_ids, int seq_len,
                                     __nv_bfloat16* ws, cudaStream_t stream);
    void talker_layer_forward_decode(int layer_idx, __nv_bfloat16* hidden,
                                     const int* pos_ids,
                                     __nv_bfloat16* ws, cudaStream_t stream);

    void cp_layer_forward_prefill(int layer_idx, __nv_bfloat16* hidden,
                                  int seq_len, __nv_bfloat16* ws, cudaStream_t stream);
    void cp_layer_forward_decode(int layer_idx, __nv_bfloat16* hidden,
                                  __nv_bfloat16* ws, cudaStream_t stream);

    // Run CodePredictor to generate 15 codec groups
    void run_code_predictor(const __nv_bfloat16* past_hidden,
                            int group_0_id,
                            int* codec_out,  // [15], host output
                            cudaStream_t stream);

    // Text projection: Linear(SiLU(Linear(x)))
    void text_projection_forward(__nv_bfloat16* output,
                                 const __nv_bfloat16* input,
                                 int num_tokens,
                                 __nv_bfloat16* ws,
                                 cudaStream_t stream);

    // cuBLAS GEMM helper
    void gemm_bf16(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                   int M, int N, int K, cudaStream_t stream);
    // GEMM + bias
    void gemm_bf16_bias(__nv_bfloat16* C, const __nv_bfloat16* A, const __nv_bfloat16* B,
                        const __nv_bfloat16* bias, int M, int N, int K, cudaStream_t stream);
};

} // namespace tts
} // namespace qwen_thor
