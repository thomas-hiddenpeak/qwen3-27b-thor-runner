#pragma once

#include "tensor.h"
#include "paged_attention.h"
#include "streaming_attention.h"
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace core {

// ============================================================================
// Qwen3.5 模型配置 (支持 4B / 9B / 27B 等多规模变体)
// 架构: N层混合 (Linear Attention Gated DeltaNet + Full Attention GQA)
//   layer_idx % full_attention_interval == (full_attention_interval-1) → full_attention
//   其余 → linear_attention
// ============================================================================
struct Qwen35Config {
    // -- 基础 --
    int hidden_size          = 5120;
    int intermediate_size    = 17408;
    int num_hidden_layers    = 64;
    int vocab_size           = 248320;
    float rms_norm_eps       = 1e-6f;
    std::string model_dir    = "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";
    bool tie_word_embeddings = false;  // 4B: true (lm_head 共享 embed_tokens)

    // -- Full Attention (GQA) --
    int num_attention_heads  = 24;     // Q heads
    int num_key_value_heads  = 4;      // K/V heads (GQA)
    int head_dim             = 256;
    float rope_theta         = 10000000.0f;   // 1e7
    // partial_rotary_factor=0.25: 只旋转 head_dim*0.25 = 64 维
    int rope_rotary_dim      = 64;
    int full_attention_interval = 4;   // 每 N 层一个 full_attention

    // -- Linear Attention (Gated DeltaNet) --
    int linear_num_key_heads   = 16;
    int linear_key_head_dim    = 128;
    int linear_num_value_heads = 48;
    int linear_value_head_dim  = 128;
    int linear_conv_kernel_dim = 4;

    // -- MoE (Mixture of Experts) --
    bool is_moe = false;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;

    // -- 层类型判断 --
    bool is_full_attention(int layer_idx) const {
        return (layer_idx % full_attention_interval) == (full_attention_interval - 1);
    }
    int num_full_attn_layers() const {
        int count = 0;
        for (int i = 0; i < num_hidden_layers; i++)
            if (is_full_attention(i)) count++;
        return count;
    }

    // -- 从模型目录的 config.json 加载配置 --
    bool load_from_model_dir(const std::string& dir) {
        std::string path = dir + "/config.json";
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        std::string json = ss.str();

        // 找到 "text_config" 块
        auto tc_pos = json.find("\"text_config\"");
        std::string section = (tc_pos != std::string::npos) ? json.substr(tc_pos) : json;

        auto get_int = [&](const std::string& key) -> int {
            std::string pat = "\"" + key + "\"";
            auto p = section.find(pat);
            if (p == std::string::npos) return -1;
            p = section.find(':', p + pat.size());
            if (p == std::string::npos) return -1;
            p++;
            while (p < section.size() && (section[p] == ' ' || section[p] == '\t')) p++;
            return std::atoi(section.c_str() + p);
        };
        auto get_float = [&](const std::string& key) -> float {
            std::string pat = "\"" + key + "\"";
            auto p = section.find(pat);
            if (p == std::string::npos) return -1.0f;
            p = section.find(':', p + pat.size());
            if (p == std::string::npos) return -1.0f;
            p++;
            while (p < section.size() && (section[p] == ' ' || section[p] == '\t')) p++;
            return std::atof(section.c_str() + p);
        };
        auto get_bool = [&](const std::string& key, const std::string& src) -> int {
            std::string pat = "\"" + key + "\"";
            auto p = src.find(pat);
            if (p == std::string::npos) return -1;
            p = src.find(':', p + pat.size());
            if (p == std::string::npos) return -1;
            p++;
            while (p < src.size() && (src[p] == ' ' || src[p] == '\t')) p++;
            if (src.substr(p, 4) == "true") return 1;
            if (src.substr(p, 5) == "false") return 0;
            return -1;
        };

        int v;
        float fv;
        if ((v = get_int("hidden_size"))          > 0) hidden_size = v;
        if ((v = get_int("intermediate_size"))    > 0) intermediate_size = v;
        if ((v = get_int("num_hidden_layers"))     > 0) num_hidden_layers = v;
        if ((v = get_int("vocab_size"))            > 0) vocab_size = v;
        if ((v = get_int("num_attention_heads"))   > 0) num_attention_heads = v;
        if ((v = get_int("num_key_value_heads"))   > 0) num_key_value_heads = v;
        if ((v = get_int("head_dim"))              > 0) head_dim = v;
        if ((v = get_int("full_attention_interval")) > 0) full_attention_interval = v;
        if ((v = get_int("linear_num_key_heads"))  > 0) linear_num_key_heads = v;
        if ((v = get_int("linear_key_head_dim"))   > 0) linear_key_head_dim = v;
        if ((v = get_int("linear_num_value_heads")) > 0) linear_num_value_heads = v;
        if ((v = get_int("linear_value_head_dim")) > 0) linear_value_head_dim = v;
        if ((v = get_int("linear_conv_kernel_dim")) > 0) linear_conv_kernel_dim = v;
        if ((fv = get_float("rms_norm_eps"))        > 0) rms_norm_eps = fv;

        // MoE fields
        if ((v = get_int("num_experts")) > 0) { num_experts = v; is_moe = true; }
        if ((v = get_int("num_experts_per_tok")) > 0) num_experts_per_tok = v;
        if ((v = get_int("moe_intermediate_size")) > 0) moe_intermediate_size = v;
        if ((v = get_int("shared_expert_intermediate_size")) > 0) shared_expert_intermediate_size = v;
        // MoE models have no dense MLP; use shared_expert_intermediate_size for workspace compat
        if (is_moe && moe_intermediate_size > 0) intermediate_size = shared_expert_intermediate_size;

        // partial_rotary_factor → rope_rotary_dim
        float prf = get_float("partial_rotary_factor");
        if (prf > 0) rope_rotary_dim = (int)(head_dim * prf);

        // rope_theta (在 rope_parameters 子块中)
        auto rp_pos = section.find("\"rope_parameters\"");
        if (rp_pos != std::string::npos) {
            std::string rp_section = section.substr(rp_pos);
            std::string pat = "\"rope_theta\"";
            auto p = rp_section.find(pat);
            if (p != std::string::npos) {
                p = rp_section.find(':', p + pat.size());
                if (p != std::string::npos) {
                    p++;
                    while (p < rp_section.size() && (rp_section[p] == ' ' || rp_section[p] == '\t')) p++;
                    rope_theta = std::atof(rp_section.c_str() + p);
                }
            }
        }

        // tie_word_embeddings (在顶层 JSON 中)
        v = get_bool("tie_word_embeddings", json);
        if (v >= 0) tie_word_embeddings = (v == 1);

        model_dir = dir;

        fprintf(stderr, "[Config] Loaded from %s:\n"
                "  hidden_size=%d, intermediate_size=%d, num_layers=%d\n"
                "  num_attn_heads=%d, num_kv_heads=%d, head_dim=%d\n"
                "  linear: nkh=%d, kd=%d, nv=%d, vd=%d\n"
                "  full_attn_interval=%d, tie_word_embeddings=%s\n",
                path.c_str(),
                hidden_size, intermediate_size, num_hidden_layers,
                num_attention_heads, num_key_value_heads, head_dim,
                linear_num_key_heads, linear_key_head_dim,
                linear_num_value_heads, linear_value_head_dim,
                full_attention_interval, tie_word_embeddings ? "true" : "false");
        if (is_moe) {
            fprintf(stderr, "  MoE: %d experts, top-%d, moe_is=%d, shared_is=%d\n",
                    num_experts, num_experts_per_tok, moe_intermediate_size,
                    shared_expert_intermediate_size);
        }
        return true;
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

    // gate_buf 同时用于 dense MLP gate 输出 [is] 和 attn_gate [q_dim]
    // MoE 模型: is=512 远小于 q_dim=4096, 必须取 max 防止 attn_gate 溢出到 up_out
    // (T>1 时 chunked/prefill attention 用 up_out 当 workspace, 会覆盖溢出的 attn_gate)
    int gate_buf_size() const { return std::max(intermediate_size, q_dim()); }

    // FullAttn 层 T=1 workspace 元素数 (bf16)
    // norm_out[hs] + qg_proj[qp_dim] + k[kv_dim] + v[kv_dim] + attn_out[q_dim]
    // + o_proj_out[hs] + post_norm_out[hs] + gate_buf[gate_buf_size] + up_out[is] + swiglu_out[is] + down_out[hs]
    int full_attn_workspace_elems_t1() const {
        return hidden_size + q_proj_dim() + kv_dim() * 2 + q_dim()
             + hidden_size * 3 + gate_buf_size() + intermediate_size * 2;
    }

    // MoE 额外 workspace 元素数 (bf16 words, 加到 ws_per_tok 上)
    // 用于替代 dense MLP 的 3*is+hs 部分, 差额即 moe_extra
    int moe_workspace_extra_t1() const {
        if (!is_moe) return 0;
        // MoE workspace after post_norm_out (per T=1):
        //   router_logits[E] + indices[topk*2] + weights[topk*2]
        //   + moe_acc[hs] + shared_gate/up/swiglu[3*shared_is] + gate_scalar[1]
        //   + (batched) expert_gu_all[topk*2*moe_is] + expert_swiglu_all[topk*moe_is]
        //   + expert_down_all[topk*hs]
        // Dense MLP (already counted): gate_buf_size+2*is + hs
        // Extra = MoE_total - dense_total
        int moe_total = num_experts + num_experts_per_tok * 4
                      + hidden_size + 3 * shared_expert_intermediate_size + 1
                      + num_experts_per_tok * 2 * moe_intermediate_size
                      + num_experts_per_tok * moe_intermediate_size
                      + num_experts_per_tok * hidden_size;
        int dense_total = gate_buf_size() + intermediate_size * 2 + hidden_size;
        return moe_total - dense_total;
    }
};


// ============================================================================
// ============================================================================
// NVFP4 量化权重: FP4 E2M1 packed + F8_E4M3 per-group scale + F32 global scale
// ============================================================================
struct QuantizedWeight {
    uint8_t* packed = nullptr;   // [N, K/2] FP4 packed as U8 (2 values per byte)
    uint8_t* scale = nullptr;    // [N, K/group_size] F8_E4M3 per-group scale
    float global_scale = 1.0f;   // per-projection F32 weight_global_scale
    float input_scale = 1.0f;    // per-projection F32 input_global_scale
    int N = 0, K = 0;            // logical weight shape [N, K]
    bool valid() const { return packed != nullptr; }
};

// MoE 权重集合: 路由器 + packed experts + 共享专家 + 共享专家门控
// 支持 BF16 和 NVFP4 两种精度 — 推理时根据 is_fp4() 选择路径
// ============================================================================
struct MoEWeights {
    // BF16 权重 (dense model / BF16 MoE)
    __nv_bfloat16* router_w = nullptr;              // [num_experts, hidden_size]
    __nv_bfloat16* experts_gate_up_w = nullptr;     // [num_experts, 2*moe_is, hs] packed contiguous
    __nv_bfloat16* experts_down_w = nullptr;        // [num_experts, hs, moe_is] packed contiguous
    __nv_bfloat16* shared_gate_w = nullptr;          // [shared_is, hs]
    __nv_bfloat16* shared_up_w = nullptr;            // [shared_is, hs]
    __nv_bfloat16* shared_down_w = nullptr;          // [hs, shared_is]
    __nv_bfloat16* shared_expert_gate_w = nullptr;   // [1, hs]

    // NVFP4 量化权重 (NVFP4 MoE)
    // Expert weights: packed contiguous across all experts
    //   gate_up: packed[E * 2*moe_is, K/2], scale[E * 2*moe_is, K/16]
    //   down:    packed[E * hs, K_down/2], scale[E * hs, K_down/16]
    //   其中每个 expert 的 N 维度连续排列, stride = N_per_expert * K/2 (packed) or N * K/16 (scale)
    uint8_t* fp4_experts_gate_up_packed = nullptr;
    uint8_t* fp4_experts_gate_up_scale = nullptr;
    float*   fp4_experts_gate_up_inv_gs = nullptr;  // [E] per-expert 1/global_scale
    int      fp4_experts_gate_up_N = 0;  // = 2 * moe_is (per expert)
    int      fp4_experts_gate_up_K = 0;  // = hs
    uint8_t* fp4_experts_down_packed = nullptr;
    uint8_t* fp4_experts_down_scale = nullptr;
    float*   fp4_experts_down_inv_gs = nullptr;      // [E] per-expert 1/global_scale
    int      fp4_experts_down_N = 0;     // = hs (per expert)
    int      fp4_experts_down_K = 0;     // = moe_is

    // Shared expert FP4
    QuantizedWeight shared_gate_qw;
    QuantizedWeight shared_up_qw;
    QuantizedWeight shared_down_qw;

    bool valid() const { return router_w != nullptr; }
    bool is_fp4() const { return fp4_experts_gate_up_packed != nullptr; }
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
    // tokens_per_seq: batch prefill 模式 — batch_size > 1 && tokens_per_seq > 1
    //   每请求 tokens_per_seq 个 token, 共 batch_size 个请求
    //   num_tokens = batch_size * tokens_per_seq
    //   GEMMs 用 B*T 批量处理; attention/KV write 逐请求串行
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
        ops::StreamingAttnCtx* streaming_ctx = nullptr,  // SSD 流式注意力上下文
        int tokens_per_seq = 0       // >1 for batch prefill (batch_size requests × tokens_per_seq tokens)
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

    // NVFP4 量化权重 (仅在 quantized_==true 时使用)
    bool quantized_ = false;
    // Self-Attention 投影 (q/k/v/o)
    QuantizedWeight q_qw_, k_qw_, v_qw_, o_qw_;
    // 合并后的 FP4 QKV 投影 (同层 Q/K/V global_scale 相同 → 可直接 concat)
    QuantizedWeight qkv_qw_merged_;  // [qp_dim+kv_dim*2, K]
    // MLP 投影 (gate/up/down)
    QuantizedWeight gate_qw_, up_qw_, down_qw_;
    // 合并后的 FP4 Gate+Up 投影 (同层 gate/up global_scale 相同)
    QuantizedWeight gate_up_qw_merged_;  // [2*is, K]

public:
    bool is_quantized() const { return quantized_; }
    QuantizedWeight& get_q_qw() { return q_qw_; }
    QuantizedWeight& get_k_qw() { return k_qw_; }
    QuantizedWeight& get_v_qw() { return v_qw_; }
    QuantizedWeight& get_gate_qw() { return gate_qw_; }
    QuantizedWeight& get_up_qw() { return up_qw_; }

    void set_quantized_attn(
        const QuantizedWeight& q, const QuantizedWeight& k,
        const QuantizedWeight& v, const QuantizedWeight& o) {
        quantized_ = true;
        q_qw_ = q; k_qw_ = k; v_qw_ = v; o_qw_ = o;
    }

    void set_quantized_mlp(
        const QuantizedWeight& gate, const QuantizedWeight& up,
        const QuantizedWeight& down) {
        gate_qw_ = gate; up_qw_ = up; down_qw_ = down;
    }

    // FP4 QKV 合并: Q[qp,K] + K[kv,K] + V[kv,K] → [qp+kv*2, K]
    void set_merged_fp4_qkv(const QuantizedWeight& merged) {
        qkv_qw_merged_ = merged;
    }

    // FP4 Gate+Up 合并: gate[is,K] + up[is,K] → [2*is, K]
    void set_merged_fp4_gate_up(const QuantizedWeight& merged) {
        gate_up_qw_merged_ = merged;
    }

    // MoE weights (replaces dense MLP when active)
    MoEWeights moe_;
    void set_moe_weights(const MoEWeights& moe) { moe_ = moe; }
    bool is_moe() const { return moe_.valid(); }
    const MoEWeights& get_moe() const { return moe_; }
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
    // tokens_per_seq: batch prefill 模式 — batch_size > 1 && tokens_per_seq > 1
    //   每请求 tokens_per_seq 个 token, GEMMs 批量, conv1d/SSM 逐请求
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
        int num_checkpoints = 1,
        int tokens_per_seq = 0       // >1 for batch prefill (batch_size requests × tokens_per_seq tokens)
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

    // NVFP4 量化权重 — MLP 始终被量化; 部分模型 Linear Attn 投影也量化
    bool quantized_ = false;
    QuantizedWeight gate_qw_, up_qw_, down_qw_;
    // LinearAttn 投影 FP4 (Kbenkhaled 模型)
    QuantizedWeight in_proj_qkv_qw_, in_proj_z_qw_, out_proj_qw_;
    // 合并后的 FP4 QKV+Z 投影 [in_qkv+lin_v, K]
    QuantizedWeight qkv_z_qw_merged_;

public:
    bool is_quantized() const { return quantized_; }
    QuantizedWeight& get_gate_qw() { return gate_qw_; }
    QuantizedWeight& get_up_qw() { return up_qw_; }
    QuantizedWeight& get_qkv_qw() { return in_proj_qkv_qw_; }
    QuantizedWeight& get_z_qw() { return in_proj_z_qw_; }

    void set_quantized_mlp(
        const QuantizedWeight& gate, const QuantizedWeight& up,
        const QuantizedWeight& down) {
        quantized_ = true;
        gate_qw_ = gate; up_qw_ = up; down_qw_ = down;
    }

    void set_quantized_attn(
        const QuantizedWeight& qkv, const QuantizedWeight& z,
        const QuantizedWeight& out) {
        quantized_ = true;
        in_proj_qkv_qw_ = qkv; in_proj_z_qw_ = z; out_proj_qw_ = out;
    }

    // 设置合并后的 Gate+Up 权重 (T>1 GEMM 优化: 2 launches → 1)
    void set_merged_gate_up(__nv_bfloat16* merged) {
        gate_up_merged_w_ = merged;
        gate_proj_w_ = merged;
        up_proj_w_ = merged + (size_t)config_.intermediate_size * config_.hidden_size;
    }

    // FP4 Gate+Up 合并 (merge packed+scale, 共享 global_scale)
    QuantizedWeight gate_up_qw_merged_;
    void set_merged_fp4_gate_up(const QuantizedWeight& merged) {
        gate_up_qw_merged_ = merged;
    }

    // FP4 QKV+Z 合并: in_proj_qkv[in_qkv,K] + in_proj_z[lin_v,K] → [in_qkv+lin_v, K]
    void set_merged_fp4_qkv_z(const QuantizedWeight& merged) {
        qkv_z_qw_merged_ = merged;
    }

    // MoE weights (replaces dense MLP when active)
    MoEWeights moe_;
    void set_moe_weights(const MoEWeights& moe) { moe_ = moe; }
    bool is_moe() const { return moe_.valid(); }
    const MoEWeights& get_moe() const { return moe_; }
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
    bool is_quantized() const {
        if (full_attn_) return full_attn_->is_quantized();
        if (linear_attn_) return linear_attn_->is_quantized();
        return false;
    }
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
