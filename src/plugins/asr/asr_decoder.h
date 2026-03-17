// asr_decoder.h — Qwen3-ASR Text Decoder
//
// 28-layer Qwen3 decoder:
//   - RMSNorm (plain weight, eps=1e-6)
//   - GQA: 16 Q heads, 8 KV heads, head_dim=128
//   - Per-head Q/K RMSNorm
//   - MRoPE (interleaved section, half-rotation)
//   - SwiGLU MLP (no bias on any projection)
//   - Contiguous KV cache (non-paged, ASR 单请求)
//
// 权重前缀: thinker.model.layers.{i}.*
//           thinker.model.embed_tokens.weight
//           thinker.model.norm.weight
//           thinker.lm_head.weight (= embed_tokens, tied)

#pragma once

#include "asr_config.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>

namespace qwen_thor {
namespace asr {

// 每层 Decoder 的权重指针 (所有 projection 无 bias)
struct DecoderLayerWeights {
    // Pre-attention norm
    __nv_bfloat16* input_layernorm_w = nullptr;       // [hidden_size=2048]

    // Self-attention (GQA)
    __nv_bfloat16* q_proj_w = nullptr;                // [q_dim=2048, hidden_size=2048]
    __nv_bfloat16* k_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* v_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* o_proj_w = nullptr;                // [hidden_size=2048, q_dim=2048]
    __nv_bfloat16* q_norm_w = nullptr;                // [head_dim=128] per-head RMSNorm
    __nv_bfloat16* k_norm_w = nullptr;                // [head_dim=128]

    // Post-attention norm
    __nv_bfloat16* post_attention_layernorm_w = nullptr; // [hidden_size=2048]

    // MLP (SwiGLU, no bias)
    __nv_bfloat16* gate_proj_w = nullptr;             // [intermediate=6144, hidden=2048]
    __nv_bfloat16* up_proj_w = nullptr;               // [intermediate=6144, hidden=2048]
    __nv_bfloat16* down_proj_w = nullptr;             // [hidden=2048, intermediate=6144]
};

class TextDecoder {
public:
    TextDecoder(const ASRConfig& config, int max_seq_len = 512);
    ~TextDecoder();

    // 绑定共享权重
    void set_embed_weights(__nv_bfloat16* embed_tokens_w,  // [vocab_size, hidden_size]
                           __nv_bfloat16* lm_head_w,       // [vocab_size, hidden_size] (同 embed_tokens)
                           __nv_bfloat16* final_norm_w);   // [hidden_size]

    // 绑定第 layer_idx 层权重
    void set_layer_weights(int layer_idx, const DecoderLayerWeights& weights);

    // 初始化 KV cache 和 workspace
    void initialize(cudaStream_t stream = 0);

    // 重置 KV cache (新请求时调用)
    void reset_cache();

    // Prefill: 输入 embeddings (已替换音频), 输出最后一个 token 的 logits
    // input_embeds: [seq_len, hidden_size] (GPU BF16)
    // position_ids: [3, seq_len] (GPU int, MRoPE 3D 位置)
    // logits_out: [vocab_size] (GPU BF16, 仅最后一个 token)
    void forward_prefill(const __nv_bfloat16* input_embeds,
                         const int* position_ids,
                         int seq_len,
                         __nv_bfloat16* logits_out,
                         cudaStream_t stream = 0);

    // Decode: 单 token 步进
    // token_id: 当前 token
    // position_ids: [3] (GPU int, 当前位置的 3D position)
    // logits_out: [vocab_size] (GPU BF16)
    void forward_decode(int token_id,
                        const int* position_ids,
                        __nv_bfloat16* logits_out,
                        cudaStream_t stream = 0);

    int current_seq_len() const { return cache_seq_len_; }

private:
    ASRConfig config_;
    int max_seq_len_;
    cublasHandle_t cublas_handle_ = nullptr;

    // Shared weights
    __nv_bfloat16* embed_tokens_w_ = nullptr;
    __nv_bfloat16* lm_head_w_ = nullptr;
    __nv_bfloat16* final_norm_w_ = nullptr;

    // Layer weights
    std::vector<DecoderLayerWeights> layer_weights_;

    // KV Cache: per-layer, contiguous [max_seq_len, num_kv_heads, head_dim]
    std::vector<__nv_bfloat16*> k_cache_;  // [num_layers]
    std::vector<__nv_bfloat16*> v_cache_;  // [num_layers]
    int cache_seq_len_ = 0;

    // Workspace
    __nv_bfloat16* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    // Token ID for decode step (device memory, write via cudaMemcpy)
    int* token_id_gpu_ = nullptr;

    // cuBLAS pre-allocated workspace (prevents internal cudaMalloc)
    void* cublas_workspace_ = nullptr;

    bool initialized_ = false;

    // 单层 decoder forward (prefill 路径)
    void decoder_layer_forward_prefill(
        int layer_idx,
        __nv_bfloat16* hidden_states,    // [seq_len, hidden_size], in-place
        const int* position_ids,          // [3, seq_len]
        int seq_len,
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);

    // 单层 decoder forward (decode 路径, T=1)
    void decoder_layer_forward_decode(
        int layer_idx,
        __nv_bfloat16* hidden_states,    // [1, hidden_size], in-place
        const int* position_ids,          // [3, 1]
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);
};

} // namespace asr
} // namespace qwen_thor
