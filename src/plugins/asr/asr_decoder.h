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
    __nv_bfloat16* input_layernorm_w = nullptr;       // [hidden_size=2048] plain weight
    __nv_bfloat16* input_layernorm_w_centered = nullptr; // [hidden_size] (w-1) for fused RMSNorm+GEMV

    // Self-attention (GQA)
    __nv_bfloat16* q_proj_w = nullptr;                // [q_dim=2048, hidden_size=2048]
    __nv_bfloat16* k_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* v_proj_w = nullptr;                // [kv_dim=1024, hidden_size=2048]
    __nv_bfloat16* qkv_proj_w = nullptr;              // [q_dim+2*kv_dim=4096, hidden_size=2048] merged
    __nv_bfloat16* o_proj_w = nullptr;                // [hidden_size=2048, q_dim=2048]
    __nv_bfloat16* q_norm_w = nullptr;                // [head_dim=128] per-head RMSNorm
    __nv_bfloat16* k_norm_w = nullptr;                // [head_dim=128]

    // Post-attention norm
    __nv_bfloat16* post_attention_layernorm_w = nullptr; // [hidden_size=2048]
    __nv_bfloat16* post_attention_layernorm_w_centered = nullptr; // [hidden_size] (w-1)

    // MLP (SwiGLU, no bias)
    __nv_bfloat16* gate_proj_w = nullptr;             // [intermediate=6144, hidden=2048]
    __nv_bfloat16* up_proj_w = nullptr;               // [intermediate=6144, hidden=2048]
    __nv_bfloat16* gateup_proj_w = nullptr;           // [2*intermediate=12288, hidden=2048] merged
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

    // 准备优化权重 (QKV merge + RMSNorm centered transform)
    // 必须在 set_layer_weights + set_embed_weights 之后、首次推理之前调用
    void prepare_optimized_weights(cudaStream_t stream = 0);

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

    // ========================================================================
    // Batch decode: B sequences decode simultaneously using cuBLAS GEMM
    // ========================================================================

    // Allocate batch KV cache and workspace (call once, idempotent)
    void initialize_batch(int max_batch_size, cudaStream_t stream = 0);

    // Reset batch state for a new set of B sequences
    void reset_batch(int batch_size);

    // Prefill single item in batch (redirects KV writes to batch_k/v_cache_[idx])
    // Call for each item 0..batch_size-1 before batch decode
    void forward_prefill_batch_item(int batch_idx,
                                     const __nv_bfloat16* input_embeds,
                                     const int* position_ids,
                                     int seq_len,
                                     __nv_bfloat16* logits_out,
                                     cudaStream_t stream = 0);

    // Batch decode: process active_batch_size tokens simultaneously via GEMM
    // token_ids: [active_batch_size] on GPU
    // position_ids: [3, active_batch_size] on GPU
    // logits_out: [active_batch_size, vocab_size] on GPU
    void forward_decode_batch(const int* token_ids,
                              const int* position_ids,
                              int active_batch_size,
                              __nv_bfloat16* logits_out,
                              cudaStream_t stream = 0);

    // Batch accessors
    int batch_seq_len(int idx) const { return batch_seq_lens_[idx]; }
    void set_batch_seq_len(int idx, int len) { batch_seq_lens_[idx] = len; }
    void increment_batch_seq_lens(const std::vector<bool>& finished);
    bool batch_initialized() const { return batch_initialized_; }

private:
    ASRConfig config_;
    int max_seq_len_;
    cublasHandle_t cublas_handle_ = nullptr;

    // Shared weights
    __nv_bfloat16* embed_tokens_w_ = nullptr;
    __nv_bfloat16* lm_head_w_ = nullptr;
    __nv_bfloat16* final_norm_w_ = nullptr;
    __nv_bfloat16* final_norm_w_centered_ = nullptr;  // (w-1) for fused decode LM head

    // Layer weights
    std::vector<DecoderLayerWeights> layer_weights_;

    // Merged/optimized weight allocations (freed in destructor)
    std::vector<void*> merged_allocations_;

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

    // Split-K attention workspace (float buffers for partial results)
    float* attn_split_k_ws_ = nullptr;
    int attn_max_partitions_ = 0;

    // Prefill attention workspace: [max_seq, max_seq] BF16 for cuBLAS attention
    __nv_bfloat16* prefill_attn_score_buf_ = nullptr;

    bool initialized_ = false;

    // 单层 decoder forward (prefill 路径)
    void decoder_layer_forward_prefill(
        int layer_idx,
        __nv_bfloat16* hidden_states,    // [seq_len, hidden_size], in-place
        const int* position_ids,          // [3, seq_len]
        int seq_len,
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);

    // 单層 decoder forward (decode 路径, T=1)
    void decoder_layer_forward_decode(
        int layer_idx,
        __nv_bfloat16* hidden_states,    // [1, hidden_size], in-place
        const int* position_ids,          // [3, 1]
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);

    // ========================================================================
    // Batch decode private members
    // ========================================================================
    int max_batch_size_ = 0;
    int cur_batch_size_ = 0;
    std::vector<int> batch_seq_lens_;    // [cur_batch_size_] per-seq cache lengths

    // Batch KV caches: [num_layers], each [max_batch, max_seq, kv_heads, head_dim]
    std::vector<__nv_bfloat16*> batch_k_cache_;
    std::vector<__nv_bfloat16*> batch_v_cache_;

    // Batch workspace
    __nv_bfloat16* batch_workspace_ = nullptr;
    size_t batch_workspace_size_ = 0;

    bool batch_initialized_ = false;

    // Per-layer batch decode
    void decoder_layer_forward_decode_batch(
        int layer_idx,
        __nv_bfloat16* hidden_states,    // [B, hidden_size], in-place
        const int* position_ids,          // [3, B]
        int B,
        __nv_bfloat16* workspace_base,
        cudaStream_t stream);
};

} // namespace asr
} // namespace qwen_thor
