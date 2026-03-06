#pragma once

#include "tensor.h"
#include "paged_attention.h"
#include "streaming_attention.h"
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace core {

// ============================================================================
// Qwen3.5-27B 模型配置
// 架构: 64层混合 (48层 Linear Attention Gated DeltaNet + 16层 Full Attention GQA)
//   layer_idx % 4 == 3 → full_attention  (layers 3, 7, 11, ..., 63)
//   layer_idx % 4 != 3 → linear_attention (所有其他层)
// ============================================================================
struct Qwen35Config {
    // -- 基础 --
    int hidden_size          = 5120;
    int intermediate_size    = 17408;
    int num_hidden_layers    = 64;
    int vocab_size           = 248320;
    float rms_norm_eps       = 1e-6f;
    std::string model_dir    = "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";

    // -- Full Attention (GQA) --
    int num_attention_heads  = 24;     // Q heads
    int num_key_value_heads  = 4;      // K/V heads (GQA)
    int head_dim             = 256;
    float rope_theta         = 10000000.0f;   // 1e7
    // partial_rotary_factor=0.25: 只旋转 head_dim*0.25 = 64 维
    int rope_rotary_dim      = 64;

    // -- Linear Attention (Gated DeltaNet) --
    int linear_num_key_heads   = 16;
    int linear_key_head_dim    = 128;
    int linear_num_value_heads = 48;
    int linear_value_head_dim  = 128;
    int linear_conv_kernel_dim = 4;

    // -- 层类型判断 --
    // layers 3, 7, 11, ..., 63 是 full_attention
    bool is_full_attention(int layer_idx) const {
        return (layer_idx % 4) == 3;
    }
    int num_full_attn_layers() const {
        int count = 0;
        for (int i = 0; i < num_hidden_layers; i++)
            if (is_full_attention(i)) count++;
        return count;
    }

    // -- EOS Token IDs --
    static constexpr int EOS_TOKEN_IM_END  = 248046;  // <|im_end|>
    static constexpr int EOS_TOKEN_ENDOFTEXT = 248044; // <|endoftext|>
    static bool is_eos(int token_id) {
        return token_id == EOS_TOKEN_IM_END || token_id == EOS_TOKEN_ENDOFTEXT;
    }

    // -- MTP 常量 --
    static constexpr int MAX_MTP_DRAFTS = 8;  // 最大 draft 数 (T_max = MAX_MTP_DRAFTS + 1)

    // -- 派生尺寸 --
    int q_dim()        const { return num_attention_heads * head_dim; }         // 24*256=6144
    int q_proj_dim()   const { return num_attention_heads * head_dim * 2; }     // 24*256*2=12288 (Q+Gate)
    int kv_dim()       const { return num_key_value_heads * head_dim; }         // 4*256=1024
    int lin_qk_dim()   const { return linear_num_key_heads * linear_key_head_dim; } // 16*128=2048
    int lin_v_dim()    const { return linear_num_value_heads * linear_value_head_dim; } // 48*128=6144
    // 每个 key-head 对应的 value 维度: (48/16)*128 = 384
    int lin_v_per_kh() const {
        return (linear_num_value_heads / linear_num_key_heads) * linear_value_head_dim;
    }

    // FullAttn 层 T=1 workspace 元素数 (bf16)
    // norm_out[hs] + qg_proj[qp_dim] + k[kv_dim] + v[kv_dim] + attn_out[q_dim]
    // + o_proj_out[hs] + post_norm_out[hs] + gate_buf[is] + up_out[is] + swiglu_out[is] + down_out[hs]
    int full_attn_workspace_elems_t1() const {
        return hidden_size + q_proj_dim() + kv_dim() * 2 + q_dim()
             + hidden_size * 3 + intermediate_size * 3;
    }
};


// ============================================================================
// Full Attention Layer - GQA + RoPE(partial) + Paged KV Cache + Dense MLP
// ============================================================================
class Qwen35FullAttnLayer {
public:
    Qwen35FullAttnLayer(const Qwen35Config& config, int layer_idx);
    ~Qwen35FullAttnLayer() = default;

    void set_weights(
        __nv_bfloat16* q_proj_w,          // [q_dim, hs]
        __nv_bfloat16* k_proj_w,          // [kv_dim, hs]
        __nv_bfloat16* v_proj_w,          // [kv_dim, hs]
        __nv_bfloat16* o_proj_w,          // [hs, q_dim]
        __nv_bfloat16* q_norm_w,          // [head_dim]  - per-head RMSNorm before RoPE
        __nv_bfloat16* k_norm_w,          // [head_dim]
        __nv_bfloat16* gate_proj_w,       // [is, hs]
        __nv_bfloat16* up_proj_w,         // [is, hs]
        __nv_bfloat16* down_proj_w,       // [hs, is]
        __nv_bfloat16* input_norm_w,      // [hs]
        __nv_bfloat16* post_attn_norm_w   // [hs]
    );

    // workspace: >= num_tokens*(hs + q_dim + kv_dim*2 + q_dim + hs*3 + is*3) * sizeof(half)
    // 对 batched decode: block_tables [batch_size, max_blocks_per_seq], context_lens [batch_size]
    void forward(
        __nv_bfloat16* hidden_states,         // [T, hs], in-place
        const int* pos_ids,          // [T]
        const ops::KVCacheManager& kv_manager,
        const int* block_tables,     // prefill: [max_blocks]; batched: [batch_size, max_blocks]
        const int* context_lens,     // prefill: [1]; batched: [batch_size]
        int max_num_blocks_per_seq,
        int max_context_len,
        int num_tokens,
        int full_attn_idx,           // index among full_attn layers (0..15) for per-layer KV cache
        __nv_bfloat16* workspace,
        cudaStream_t stream = 0,
        int batch_size = 1,          // >1 for batched decode
        bool force_paged_attn = false, // chunked prefill: force paged attention for non-first chunks
        ops::StreamingAttnCtx* streaming_ctx = nullptr  // SSD 流式注意力上下文
    );

    // 设置合并后的 QKV 权重 (T=1 GEMV 优化: 3 launches → 1)
    // merged 布局: [q_proj_dim + kv_dim*2, hs], 同时更新 q/k/v_proj_w_ 指向子区域
    void set_merged_qkv(__nv_bfloat16* merged) {
        qkv_merged_w_ = merged;
        q_proj_w_ = merged;
        k_proj_w_ = merged + (size_t)config_.q_proj_dim() * config_.hidden_size;
        v_proj_w_ = merged + (size_t)(config_.q_proj_dim() + config_.kv_dim()) * config_.hidden_size;
    }

    // 设置合并后的 Gate+Up 权重 (T>1 GEMM 优化: 2 launches → 1)
    // merged 布局: [2*is, hs], 同时更新 gate/up_proj_w_ 指向子区域
    void set_merged_gate_up(__nv_bfloat16* merged) {
        gate_up_merged_w_ = merged;
        gate_proj_w_ = merged;
        up_proj_w_ = merged + (size_t)config_.intermediate_size * config_.hidden_size;
    }

private:
    Qwen35Config config_;
    int layer_idx_;

    __nv_bfloat16* qkv_merged_w_ = nullptr;  // [q_proj_dim + kv_dim*2, hs] 合并权重
    __nv_bfloat16* gate_up_merged_w_ = nullptr; // [2*is, hs] 合并权重
    __nv_bfloat16* q_proj_w_   = nullptr;
    __nv_bfloat16* k_proj_w_   = nullptr;
    __nv_bfloat16* v_proj_w_   = nullptr;
    __nv_bfloat16* o_proj_w_   = nullptr;
    __nv_bfloat16* q_norm_w_   = nullptr;
    __nv_bfloat16* k_norm_w_   = nullptr;
    __nv_bfloat16* gate_proj_w_   = nullptr;
    __nv_bfloat16* up_proj_w_     = nullptr;
    __nv_bfloat16* down_proj_w_   = nullptr;
    __nv_bfloat16* input_layernorm_w_          = nullptr;
    __nv_bfloat16* post_attention_layernorm_w_ = nullptr;
};

// ============================================================================
// Linear Attention Layer - Gated DeltaNet SSM + Dense MLP
// SSM 状态由 engine 管理，forward 时通过指针传入并 in-place 更新
// ============================================================================
class Qwen35LinearAttnLayer {
public:
    Qwen35LinearAttnLayer(const Qwen35Config& config, int layer_idx);
    ~Qwen35LinearAttnLayer() = default;

    void set_weights(
        __nv_bfloat16* in_proj_qkv_w,   // [in_qkv, hs]
        __nv_bfloat16* in_proj_z_w,     // [lin_v, hs]
        __nv_bfloat16* in_proj_a_w,     // [nv=48, hs]  → per-value-head alpha gate
        __nv_bfloat16* in_proj_b_w,     // [nv=48, hs]  → per-value-head beta gate
        __nv_bfloat16* out_proj_w,      // [hs, lin_v]
        __nv_bfloat16* conv1d_w,        // [in_qkv, 1, conv_k]
        float* A_log,          // [nv=48] F32  ← 注意 F32
        __nv_bfloat16* dt_bias,         // [nv=48]
        __nv_bfloat16* attn_norm_w,     // [vd=128]
        __nv_bfloat16* gate_proj_w, __nv_bfloat16* up_proj_w, __nv_bfloat16* down_proj_w,
        __nv_bfloat16* input_norm_w, __nv_bfloat16* post_attn_norm_w);

    // 设置超级合并权重: QKV+ZAB → 单次 GEMV (T=1 时 2 launches → 1)
    // merged 布局: [in_qkv + lin_v + nv*2, hs] = [16480, 5120]
    // 同时更新 qkv/z/a/b 指向子区域 (T>1 GEMM 仍用各子指针)
    void set_merged_all_proj(__nv_bfloat16* merged) {
        all_proj_merged_w_ = merged;
        int in_qkv = config_.lin_qk_dim() * 2 + config_.lin_v_dim();  // 10240
        int lin_v  = config_.lin_v_dim();    // 6144
        int nv     = config_.linear_num_value_heads;  // 48
        int hs     = config_.hidden_size;    // 5120
        in_proj_qkv_w_ = merged;
        in_proj_z_w_   = merged + (size_t)in_qkv * hs;
        in_proj_a_w_   = merged + (size_t)(in_qkv + lin_v) * hs;
        in_proj_b_w_   = merged + (size_t)(in_qkv + lin_v + nv) * hs;
    }

    // ssm_state:  [nkh, kd, v_per_kh]  bf16, device, in-place updated
    // conv_state: [nkh*2, kd, conv_k-1] fp16,    device, in-place updated
    // batched decode: ssm_state_ptrs/conv_state_ptrs 是 device 上的指针数组 [batch_size]
    // ssm_state_checkpoint/conv_state_checkpoint: MTP T>1 verify rollback checkpoints
    //   if non-null, save state after processing first num_checkpoints tokens
    //   Layout: [num_checkpoints * state_size_per_layer] contiguous
    void forward(
        __nv_bfloat16* hidden_states,    // [T, hs], in-place
        __nv_bfloat16* ssm_state,       // [nkh, kd, v_per_kh] bf16 (batch_size==1)
        __nv_bfloat16*  conv_state,      // [nkh*2, kd, conv_kernel_dim-1] fp16 (batch_size==1)
        int num_tokens,
        __nv_bfloat16* workspace,
        cudaStream_t stream = 0,
        int batch_size = 1,
        __nv_bfloat16** ssm_state_ptrs = nullptr,      // [batch_size] device ptr array
        __nv_bfloat16** conv_state_ptrs = nullptr,  // [batch_size] device ptr array
        __nv_bfloat16* ssm_state_checkpoint = nullptr,
        __nv_bfloat16* conv_state_checkpoint = nullptr,
        int num_checkpoints = 1
    );

private:
    Qwen35Config config_;
    int layer_idx_;

    __nv_bfloat16* in_proj_qkv_w_ = nullptr;
    __nv_bfloat16* in_proj_z_w_   = nullptr;
    __nv_bfloat16* in_proj_a_w_   = nullptr;
    __nv_bfloat16* in_proj_b_w_   = nullptr;
    __nv_bfloat16* all_proj_merged_w_ = nullptr;  // [in_qkv+lin_v+nv*2, hs] 超级合并权重
    __nv_bfloat16* gate_up_merged_w_ = nullptr;   // [2*is, hs] 合并权重
    __nv_bfloat16* out_proj_w_    = nullptr;
    __nv_bfloat16* conv1d_w_      = nullptr;
    float* A_log_f32_    = nullptr;  // F32 pointer (A_log is stored as float32)
    __nv_bfloat16* dt_bias_       = nullptr;
    __nv_bfloat16* attn_norm_w_   = nullptr;
    __nv_bfloat16* gate_proj_w_   = nullptr;
    __nv_bfloat16* up_proj_w_     = nullptr;
    __nv_bfloat16* down_proj_w_   = nullptr;
    __nv_bfloat16* input_layernorm_w_          = nullptr;
    __nv_bfloat16* post_attention_layernorm_w_ = nullptr;

public:
    // 设置合并后的 Gate+Up 权重 (T>1 GEMM 优化: 2 launches → 1)
    void set_merged_gate_up(__nv_bfloat16* merged) {
        gate_up_merged_w_ = merged;
        gate_proj_w_ = merged;
        up_proj_w_ = merged + (size_t)config_.intermediate_size * config_.hidden_size;
    }
};

// ============================================================================
// Unified Layer Wrapper - 根据 layer_idx 分配到正确实现
// ============================================================================
class Qwen35Layer {
public:
    Qwen35Layer(const Qwen35Config& config, int layer_idx);
    ~Qwen35Layer() = default;
    Qwen35Layer(Qwen35Layer&&) noexcept = default;
    Qwen35Layer& operator=(Qwen35Layer&&) noexcept = default;

    bool is_full_attention() const { return config_.is_full_attention(layer_idx_); }
    Qwen35FullAttnLayer*   get_full_attn()   { return full_attn_.get();   }
    Qwen35LinearAttnLayer* get_linear_attn() { return linear_attn_.get(); }
    int get_layer_idx() const { return layer_idx_; }

private:
    Qwen35Config config_;
    int layer_idx_;
    std::unique_ptr<Qwen35FullAttnLayer>   full_attn_;
    std::unique_ptr<Qwen35LinearAttnLayer> linear_attn_;
};

} // namespace core
} // namespace qwen_thor
