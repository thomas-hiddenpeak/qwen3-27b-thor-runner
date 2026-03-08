#include "layer.h"
#include "light_ops.h"
#include "dense_gemm.h"
#include "dense_gemm_fp4.h"
#include "streaming_attention.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>

// Set to 1 to enable per-step debug prints
#define LAYER_DEBUG 0
#if LAYER_DEBUG
#define DBG_PRINTF(...) do { printf(__VA_ARGS__); fflush(stdout); } while(0)
#else
#define DBG_PRINTF(...) ((void)0)
#endif

namespace qwen_thor {
namespace core {

// ============================================================================
// Helper: Dense MLP (gate_proj + up_proj + silu + down_proj + residual)
// 两种层类型共用
// ============================================================================
static void run_mlp(
    __nv_bfloat16* hidden_states,      // [T, hs] in-place
    __nv_bfloat16* post_norm_out,      // [T, hs] workspace
    __nv_bfloat16* gate_out,           // [T, is] workspace (contiguous with up_out)
    __nv_bfloat16* up_out,             // [T, is] workspace
    __nv_bfloat16* swiglu_out,         // [T, is] workspace
    __nv_bfloat16* down_out,           // [T, hs] workspace
    const __nv_bfloat16* gate_proj_w,  // [is, hs]
    const __nv_bfloat16* up_proj_w,    // [is, hs]
    const __nv_bfloat16* down_proj_w,  // [hs, is]
    float rms_norm_eps,
    int num_tokens, int hs, int is,
    cudaStream_t stream,
    const __nv_bfloat16* post_attn_norm_w = nullptr,
    const __nv_bfloat16* residual_in = nullptr,
    const QuantizedWeight* gate_qw = nullptr,
    const QuantizedWeight* up_qw = nullptr,
    const QuantizedWeight* down_qw = nullptr,
    const QuantizedWeight* gate_up_qw_merged = nullptr)
{
    // 可选: fused residual + RMSNorm (单 kernel 替代 add + rmsnorm)
    if (residual_in && post_attn_norm_w) {
        ops::invoke_fused_add_rmsnorm(post_norm_out, hidden_states, residual_in,
                                       post_attn_norm_w, rms_norm_eps,
                                       num_tokens, hs, stream);
    } else {
        if (residual_in) {
            ops::invoke_add(hidden_states, hidden_states, residual_in, num_tokens * hs, stream);
        }
        if (post_attn_norm_w) {
            ops::invoke_rmsnorm(post_norm_out, hidden_states, post_attn_norm_w,
                                rms_norm_eps, num_tokens, hs, stream);
        }
    }

    // Gate + Up projections: [T, hs] × [hs, is] → [T, is] (各一次)
    bool use_fp4 = (gate_qw && gate_qw->valid());
    bool fp4_merged = (gate_up_qw_merged && gate_up_qw_merged->valid());
    if (num_tokens == 1) {
        if (fp4_merged) {
            // Merged FP4 gate+up: single GEMV → [2*is], then fused SwiGLU
            ops::invoke_fp4_gemv(post_norm_out, *gate_up_qw_merged, gate_out, stream);
            ops::invoke_swiglu_merged(swiglu_out, gate_out, 1, is, stream);
        } else if (use_fp4) {
            ops::invoke_fp4_dual_gemv(post_norm_out, *gate_qw, *up_qw,
                                      gate_out, up_out, stream);
            ops::invoke_swiglu(swiglu_out, gate_out, up_out, 1, is, stream);
        } else {
            // Decode 路径: dual GEMV — 1 kernel 同时完成 gate + up, A 只加载一次
            ops::invoke_dense_dual_gemv(post_norm_out, gate_proj_w, up_proj_w,
                                         gate_out, up_out, is, hs, stream);
            ops::invoke_swiglu(swiglu_out, gate_out, up_out, 1, is, stream);
        }
    } else if (fp4_merged) {
        // T>1 merged FP4 gate+up: single GEMM → [T, 2*is], then fused SwiGLU
        ops::invoke_fp4_gemm(post_norm_out, *gate_up_qw_merged, gate_out, num_tokens, stream);
        ops::invoke_swiglu_merged(swiglu_out, gate_out, num_tokens, is, stream);
    } else if (use_fp4) {
        // T>1 non-merged FP4: separate gate + up GEMM
        ops::invoke_fp4_gemm(post_norm_out, *gate_qw, gate_out, num_tokens, stream);
        ops::invoke_fp4_gemm(post_norm_out, *up_qw,   up_out,   num_tokens, stream);
        ops::invoke_swiglu(swiglu_out, gate_out, up_out, num_tokens, is, stream);
    } else if (gate_proj_w + (size_t)is * hs == up_proj_w) {
        // T>1 + merged gate_up weight: single GEMM → [T, 2*is], then fused SwiGLU
        // gate_out has room for [T, 2*is] since gate_out → up_out are contiguous
        ops::invoke_dense_gemm(post_norm_out, gate_proj_w, gate_out, num_tokens, 2 * is, hs, stream);
        ops::invoke_swiglu_merged(swiglu_out, gate_out, num_tokens, is, stream);
    } else {
        ops::invoke_dense_gemm(post_norm_out, gate_proj_w, gate_out, num_tokens, is, hs, stream);
        ops::invoke_dense_gemm(post_norm_out, up_proj_w,   up_out,  num_tokens, is, hs, stream);
        ops::invoke_swiglu(swiglu_out, gate_out, up_out, num_tokens, is, stream);
    }

    // Down projection + residual add
    if (use_fp4) {
        if (num_tokens == 1) {
            ops::invoke_fp4_gemv_add(swiglu_out, *down_qw, hidden_states, hidden_states, stream);
        } else {
            ops::invoke_fp4_gemm_add(swiglu_out, *down_qw, hidden_states, hidden_states, num_tokens, stream);
        }
    } else if (num_tokens == 1) {
        ops::invoke_dense_gemv_add(swiglu_out, down_proj_w, hidden_states, hidden_states, hs, is, stream);
    } else {
        ops::invoke_dense_gemm_add(swiglu_out, down_proj_w, hidden_states, hidden_states, num_tokens, hs, is, stream);
    }
}

// ============================================================================
// Helper: MoE MLP (Router + TopK Experts + Shared Expert + Gating)
// 两种层类型共用 — 替代 run_mlp 当 is_moe() 为 true 时
//
// Workspace layout after post_norm_out[T*hs]:
//   router_logits  [T * E]          bf16
//   expert_indices [T * top_k]      int32
//   expert_weights [T * top_k]      float32
//   moe_acc        [T * hs]         bf16  -- 路由专家 + 共享专家的累加器
//   shared_gate_out [T * shared_is] bf16
//   shared_up_out   [T * shared_is] bf16
//   shared_swiglu   [T * shared_is] bf16
//   gate_scalar     [T]             bf16
//   expert_gu       [2 * moe_is]    bf16  -- 复用 buffer (每次单 token)
//   expert_swiglu   [moe_is]        bf16
//   expert_out      [hs]            bf16
// ============================================================================
static void run_moe_mlp(
    __nv_bfloat16* hidden_states,      // [T, hs] in-place residual
    __nv_bfloat16* post_norm_out,      // [T, hs] workspace
    __nv_bfloat16* moe_workspace,      // workspace starting after post_norm_out[T*hs]
    const MoEWeights& moe,
    const Qwen35Config& config,
    float rms_norm_eps,
    int num_tokens,
    cudaStream_t stream,
    const __nv_bfloat16* post_attn_norm_w = nullptr,
    const __nv_bfloat16* residual_in = nullptr)
{
    // MoE MLP 内部细分计时 (QWEN_LAYER_TIMING=1)
    static bool moe_timing = !!getenv("QWEN_LAYER_TIMING");
    static int moe_timing_count = 0;
    static float moe_total_ms = 0, moe_expert_ms = 0, moe_shared_ms = 0, moe_router_ms = 0;
    cudaEvent_t moe_ev0, moe_ev1, moe_ev2, moe_ev3, moe_ev4;
    if (moe_timing && num_tokens == 1) {
        cudaEventCreate(&moe_ev0); cudaEventCreate(&moe_ev1);
        cudaEventCreate(&moe_ev2); cudaEventCreate(&moe_ev3);
        cudaEventCreate(&moe_ev4);
    }
    const int hs        = config.hidden_size;
    const int E         = config.num_experts;
    const int top_k     = config.num_experts_per_tok;
    const int moe_is    = config.moe_intermediate_size;
    const int shared_is = config.shared_expert_intermediate_size;

    // 1. Fused residual + post-attention RMSNorm
    if (residual_in && post_attn_norm_w) {
        ops::invoke_fused_add_rmsnorm(post_norm_out, hidden_states, residual_in,
                                       post_attn_norm_w, rms_norm_eps,
                                       num_tokens, hs, stream);
    } else {
        if (residual_in)
            ops::invoke_add(hidden_states, hidden_states, residual_in, num_tokens * hs, stream);
        if (post_attn_norm_w)
            ops::invoke_rmsnorm(post_norm_out, hidden_states, post_attn_norm_w,
                                rms_norm_eps, num_tokens, hs, stream);
    }

    // 2. Workspace pointer layout (per-token buffers)
    __nv_bfloat16* router_logits = moe_workspace;
    int*   expert_indices  = (int*)(router_logits + num_tokens * E);
    float* expert_weights  = (float*)(expert_indices + num_tokens * top_k);
    __nv_bfloat16* moe_acc = (__nv_bfloat16*)(expert_weights + num_tokens * top_k);
    __nv_bfloat16* shared_gate_out = moe_acc + num_tokens * hs;
    __nv_bfloat16* shared_up_out   = shared_gate_out + num_tokens * shared_is;
    __nv_bfloat16* shared_swiglu_buf = shared_up_out + num_tokens * shared_is;
    __nv_bfloat16* gate_scalar     = shared_swiglu_buf + num_tokens * shared_is;
    // Fixed scratch (不随 T 缩放), aligned to 8 bf16 for vectorized stores
    __nv_bfloat16* expert_scratch  = gate_scalar + ((num_tokens + 7) & ~7);

    // 3. Router: [T, hs] × [hs, E] → [T, E] + Top-K selection
    if (moe_timing && num_tokens == 1) cudaEventRecord(moe_ev0, stream);
    if (num_tokens == 1) {
        // Fused router GEMV + top-K: multi-block scattered GEMV + last-block top-K
        // Reuse router_logits as float logits workspace (E×4 bytes, overlaps into expert_indices
        // region which is only written AFTER top-K reads all logits — safe)
        float* fused_logits = reinterpret_cast<float*>(router_logits);
        // Atomic counter carved from gate_scalar region (unused in T=1 fused path)
        int* block_counter = reinterpret_cast<int*>(gate_scalar);
        cudaMemsetAsync(block_counter, 0, sizeof(int), stream);
        ops::invoke_moe_router_gemv_topk(post_norm_out, moe.router_w,
                                          fused_logits, expert_indices, expert_weights,
                                          block_counter, E, hs, top_k, stream);
    } else {
        ops::invoke_dense_gemm(post_norm_out, moe.router_w, router_logits, num_tokens, E, hs, stream);
        ops::invoke_moe_router_topk(router_logits, expert_indices, expert_weights,
                                     num_tokens, E, top_k, stream);
    }

    const size_t gu_stride = (size_t)2 * moe_is * hs;   // bf16 elements per expert gate_up
    const size_t dn_stride = (size_t)hs * moe_is;       // bf16 elements per expert down

    if (num_tokens == 1) {
        // ============================================================
        // T=1 快速路径: Grouped Expert GEMV (GPU-resident, 无 D2H/sync)
        // 4 launches 替代 32 launches + 1 cudaStreamSynchronize
        // ============================================================
        if (moe_timing) cudaEventRecord(moe_ev1, stream);
        __nv_bfloat16* expert_gu_all      = expert_scratch;                              // [top_k, 2*moe_is]
        __nv_bfloat16* expert_swiglu_all  = expert_gu_all + top_k * 2 * moe_is;         // [top_k, moe_is]
        __nv_bfloat16* expert_down_all    = expert_swiglu_all + top_k * moe_is;          // [top_k, hs]

        // 5a. Grouped gate_up GEMV: all top_k experts, shared input
        ops::invoke_grouped_expert_gemv(
            post_norm_out, moe.experts_gate_up_w, expert_gu_all,
            expert_indices, 2 * moe_is, hs, gu_stride,
            top_k, /*shared_input=*/true, stream);

        // 5b. Batched SwiGLU: top_k experts in parallel
        ops::invoke_swiglu_merged(expert_swiglu_all, expert_gu_all, top_k, moe_is, stream);

        // 5c. Grouped down GEMV: per-expert inputs
        ops::invoke_grouped_expert_gemv(
            expert_swiglu_all, moe.experts_down_w, expert_down_all,
            expert_indices, hs, moe_is, dn_stride,
            top_k, /*shared_input=*/false, stream);

        // 5d. Weighted reduce: moe_acc = sum_k(weights[k] * expert_down_all[k])
        ops::invoke_weighted_expert_reduce(moe_acc, expert_down_all, expert_weights,
                                            hs, top_k, stream);
    } else {
        // ============================================================
        // T>1 路径: 逐 token 顺序 GEMV (保持兼容)
        // ============================================================
        __nv_bfloat16* expert_gu     = expert_scratch;
        __nv_bfloat16* expert_swiglu = expert_gu + 2 * moe_is;
        __nv_bfloat16* expert_out    = expert_swiglu + moe_is;

        // D2H copy for CPU-side expert dispatch
        std::vector<int>   h_indices(num_tokens * top_k);
        std::vector<float> h_weights(num_tokens * top_k);
        cudaMemcpyAsync(h_indices.data(), expert_indices,
                         num_tokens * top_k * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_weights.data(), expert_weights,
                         num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaMemsetAsync(moe_acc, 0, num_tokens * hs * sizeof(__nv_bfloat16), stream);

        for (int t = 0; t < num_tokens; ++t) {
            const __nv_bfloat16* token_in = post_norm_out + t * hs;
            __nv_bfloat16* token_acc = moe_acc + t * hs;
            for (int k = 0; k < top_k; ++k) {
                int eidx = h_indices[t * top_k + k];
                float w  = h_weights[t * top_k + k];
                const __nv_bfloat16* egu_w = moe.experts_gate_up_w + eidx * gu_stride;
                ops::invoke_dense_gemv(token_in, egu_w, expert_gu, 2 * moe_is, hs, stream);
                ops::invoke_swiglu_merged(expert_swiglu, expert_gu, 1, moe_is, stream);
                const __nv_bfloat16* edn_w = moe.experts_down_w + eidx * dn_stride;
                ops::invoke_dense_gemv(expert_swiglu, edn_w, expert_out, hs, moe_is, stream);
                ops::invoke_scale_add(token_acc, expert_out, w, hs, stream);
            }
        }
    }

    // 8. Shared expert MLP: gate + up + SwiGLU
    if (moe_timing && num_tokens == 1) cudaEventRecord(moe_ev2, stream);
    if (num_tokens == 1) {
        ops::invoke_dense_dual_gemv(post_norm_out, moe.shared_gate_w, moe.shared_up_w,
                                     shared_gate_out, shared_up_out, shared_is, hs, stream);
    } else {
        ops::invoke_dense_gemm(post_norm_out, moe.shared_gate_w, shared_gate_out,
                                num_tokens, shared_is, hs, stream);
        ops::invoke_dense_gemm(post_norm_out, moe.shared_up_w, shared_up_out,
                                num_tokens, shared_is, hs, stream);
    }
    ops::invoke_swiglu(shared_swiglu_buf, shared_gate_out, shared_up_out,
                        num_tokens, shared_is, stream);

    // 9-10. Shared expert down + gate scalar + gated add
    __nv_bfloat16* shared_down_buf = (num_tokens == 1)
        ? (expert_scratch + top_k * (2 * moe_is + moe_is))
        : (expert_scratch + 2 * moe_is + moe_is);
    if (num_tokens == 1) {
        // T=1: down GEMV, then fused gate_scalar dot + sigmoid_gated_add (2→1 launch)
        ops::invoke_dense_gemv(shared_swiglu_buf, moe.shared_down_w,
                                shared_down_buf, hs, shared_is, stream);
        ops::invoke_sigmoid_gated_add_with_dot(
            moe_acc, shared_down_buf, post_norm_out, moe.shared_expert_gate_w,
            hs, hs, stream);
    } else {
        // T>1: separate gate_scalar GEMM + per-token down GEMV + sigmoid_gated_add
        ops::invoke_dense_gemm(post_norm_out, moe.shared_expert_gate_w,
                                gate_scalar, num_tokens, 1, hs, stream);
        for (int t = 0; t < num_tokens; ++t) {
            ops::invoke_dense_gemv(shared_swiglu_buf + t * shared_is, moe.shared_down_w,
                                    shared_down_buf, hs, shared_is, stream);
            ops::invoke_sigmoid_gated_add(moe_acc + t * hs, shared_down_buf,
                                           gate_scalar + t, hs, stream);
        }
    }

    // 11. Final residual: hidden_states += moe_acc
    ops::invoke_add(hidden_states, hidden_states, moe_acc, num_tokens * hs, stream);

    // MoE timing reporting
    if (moe_timing && num_tokens == 1) {
        cudaEventRecord(moe_ev3, stream);
        cudaEventSynchronize(moe_ev3);
        float ms_router = 0, ms_expert = 0, ms_shared = 0;
        cudaEventElapsedTime(&ms_router, moe_ev0, moe_ev1);
        cudaEventElapsedTime(&ms_expert, moe_ev1, moe_ev2);
        cudaEventElapsedTime(&ms_shared, moe_ev2, moe_ev3);
        moe_router_ms += ms_router;
        moe_expert_ms += ms_expert;
        moe_shared_ms += ms_shared;
        moe_total_ms += ms_router + ms_expert + ms_shared;
        moe_timing_count++;
        // Print every 400 calls (= 10 full decode steps × 40 layers)
        if (moe_timing_count % 400 == 0) {
            int steps = moe_timing_count / 400;
            fprintf(stderr, "[MoETiming] %d steps: router=%.2fms expert=%.2fms shared=%.2fms total=%.2fms (per-layer: r=%.3f e=%.3f s=%.3f t=%.3f)\n",
                    steps * 10, moe_router_ms / moe_timing_count, moe_expert_ms / moe_timing_count,
                    moe_shared_ms / moe_timing_count, moe_total_ms / moe_timing_count,
                    moe_router_ms / moe_timing_count, moe_expert_ms / moe_timing_count,
                    moe_shared_ms / moe_timing_count, moe_total_ms / moe_timing_count);
            fflush(stderr);
        }
        cudaEventDestroy(moe_ev0); cudaEventDestroy(moe_ev1);
        cudaEventDestroy(moe_ev2); cudaEventDestroy(moe_ev3);
        cudaEventDestroy(moe_ev4);
    }
}

// ============================================================================
// Qwen35FullAttnLayer
// ============================================================================
Qwen35FullAttnLayer::Qwen35FullAttnLayer(const Qwen35Config& config, int layer_idx)
    : config_(config), layer_idx_(layer_idx) {}

void Qwen35FullAttnLayer::set_weights(
    __nv_bfloat16* q_proj_w, __nv_bfloat16* k_proj_w, __nv_bfloat16* v_proj_w, __nv_bfloat16* o_proj_w,
    __nv_bfloat16* q_norm_w, __nv_bfloat16* k_norm_w,
    __nv_bfloat16* gate_proj_w, __nv_bfloat16* up_proj_w, __nv_bfloat16* down_proj_w,
    __nv_bfloat16* input_norm_w, __nv_bfloat16* post_attn_norm_w)
{
    q_proj_w_  = q_proj_w;  k_proj_w_  = k_proj_w;
    v_proj_w_  = v_proj_w;  o_proj_w_  = o_proj_w;
    q_norm_w_  = q_norm_w;  k_norm_w_  = k_norm_w;
    gate_proj_w_ = gate_proj_w; up_proj_w_ = up_proj_w; down_proj_w_ = down_proj_w;
    input_layernorm_w_          = input_norm_w;
    post_attention_layernorm_w_ = post_attn_norm_w;
}

// Helper kernel: compute KV write start positions
// Batched decode: each sequence writes 1 token at position context_lens[i] - 1
__global__ void compute_write_positions_kernel(int* positions, const int* context_lens, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) positions[i] = context_lens[i] - 1;
}
// Single-sequence prefill/decode: write T tokens starting at context_lens[0] - T
// For decode (T=1): same as context_lens[0] - 1
// For prefill (T>1): correctly places chunk at [context_len-T, context_len-1]
__global__ void compute_write_start_kernel(int* positions, const int* context_lens, int num_tokens) {
    if (threadIdx.x == 0) positions[0] = context_lens[0] - num_tokens;
}

void Qwen35FullAttnLayer::forward(
    __nv_bfloat16* hidden_states,
    const int* pos_ids,
    const ops::KVCacheManager& kv_manager,
    const int* block_tables,
    const int* context_lens,
    int max_num_blocks_per_seq,
    int max_context_len,
    int num_tokens,
    int full_attn_idx,
    __nv_bfloat16* workspace,
    cudaStream_t stream,
    int batch_size,
    bool force_paged_attn,
    ops::StreamingAttnCtx* streaming_ctx)
{
    if (!q_proj_w_ && !quantized_) return;

    const int hs       = config_.hidden_size;
    const int is       = config_.intermediate_size;
    const int q_dim    = config_.q_dim();          // 6144
    const int qp_dim   = config_.q_proj_dim();     // 12288 = 2*q_dim (Q+Gate)
    const int kv_dim   = config_.kv_dim();
    const int num_q    = config_.num_attention_heads;
    const int num_kv   = config_.num_key_value_heads;
    const int hd       = config_.head_dim;
    const int rot_d    = config_.rope_rotary_dim;

    // gate_buf 同时存放 attn_gate [q_dim] 和 dense MLP gate [is]
    // MoE 模型 is << q_dim, 用 gate_buf_size = max(is, q_dim) 防止溢出到 up_out
    const int gate_buf_sz = config_.gate_buf_size();

    __nv_bfloat16* norm_out      = workspace;
    __nv_bfloat16* qg_proj       = norm_out   + num_tokens * hs;     // [T, 2*q_dim] = Q + Gate
    __nv_bfloat16* k             = qg_proj    + num_tokens * qp_dim;
    __nv_bfloat16* v             = k          + num_tokens * kv_dim;
    __nv_bfloat16* attn_out      = v          + num_tokens * kv_dim;
    __nv_bfloat16* o_proj_out    = attn_out   + num_tokens * q_dim;
    __nv_bfloat16* post_norm_out = o_proj_out + num_tokens * hs;
    __nv_bfloat16* gate_buf      = post_norm_out + num_tokens * hs;  // attn_gate + dense MLP gate
    __nv_bfloat16* up_out        = gate_buf   + num_tokens * gate_buf_sz;
    __nv_bfloat16* swiglu_out    = up_out     + num_tokens * is;
    __nv_bfloat16* down_out      = swiglu_out + num_tokens * is;

    // 1+2. Input RMSNorm + QKV Projection
    //   T=1: Fused RMSNorm in SMEM + merged GEMV (saves 1 launch, no norm_out GMEM I/O)
    //   T>1: Separate RMSNorm + GEMM
    //   NVFP4: Separate RMSNorm + individual FP4 GEMV/GEMM (no merge)
    if (quantized_) {
        // NVFP4 path: separate RMSNorm, then FP4 QKV projection
        ops::invoke_rmsnorm(norm_out, hidden_states, input_layernorm_w_,
                            config_.rms_norm_eps, num_tokens, hs, stream);
        if (qkv_qw_merged_.valid()) {
            // Merged FP4 QKV: [qp_dim+kv_dim*2, hs] single GEMV/GEMM
            if (num_tokens == 1) {
                // T=1: qg_proj→k→v contiguous in workspace, single GEMV fills all
                ops::invoke_fp4_gemv(norm_out, qkv_qw_merged_, qg_proj, stream);
            } else {
                // T>1: GEMM into temp (gate_buf), then deinterleave to qg_proj/k/v
                int merged_N = qp_dim + kv_dim * 2;
                ops::invoke_fp4_gemm(norm_out, qkv_qw_merged_, gate_buf, num_tokens, stream);
                ops::invoke_deinterleave_gemm_3way(qg_proj, k, v, gate_buf,
                    num_tokens, merged_N, qp_dim, qp_dim + kv_dim, stream);
            }
        } else {
            // Non-merged FP4: separate Q/K/V
            if (num_tokens == 1) {
                ops::invoke_fp4_gemv(norm_out, q_qw_, qg_proj, stream);
                ops::invoke_fp4_gemv(norm_out, k_qw_, k,       stream);
                ops::invoke_fp4_gemv(norm_out, v_qw_, v,       stream);
            } else {
                ops::invoke_fp4_gemm(norm_out, q_qw_, qg_proj, num_tokens, stream);
                ops::invoke_fp4_gemm(norm_out, k_qw_, k,       num_tokens, stream);
                ops::invoke_fp4_gemm(norm_out, v_qw_, v,       num_tokens, stream);
            }
        }
    } else if (num_tokens == 1 && qkv_merged_w_) {
        ops::invoke_dense_gemv_with_rmsnorm(
            hidden_states, input_layernorm_w_, config_.rms_norm_eps,
            qkv_merged_w_, qg_proj, qp_dim + kv_dim * 2, hs, stream);
    } else {
        ops::invoke_rmsnorm(norm_out, hidden_states, input_layernorm_w_,
                            config_.rms_norm_eps, num_tokens, hs, stream);
        if (num_tokens == 1) {
            if (qkv_merged_w_) {
                ops::invoke_dense_gemv(norm_out, qkv_merged_w_, qg_proj, qp_dim + kv_dim * 2, hs, stream);
            } else {
                ops::invoke_dense_gemv(norm_out, q_proj_w_, qg_proj, qp_dim, hs, stream);
                ops::invoke_dense_gemv(norm_out, k_proj_w_, k,       kv_dim, hs, stream);
                ops::invoke_dense_gemv(norm_out, v_proj_w_, v,       kv_dim, hs, stream);
            }
        } else {
            if (qkv_merged_w_) {
                // T>1 + merged QKV: single GEMM + deinterleave (3 calls → 1+1)
                // Use gate_buf as temp buffer (not used until MLP, size >= T * merged_N)
                int merged_N = qp_dim + kv_dim * 2;
                ops::invoke_dense_gemm(norm_out, qkv_merged_w_, gate_buf, num_tokens, merged_N, hs, stream);
                ops::invoke_deinterleave_gemm_3way(qg_proj, k, v, gate_buf,
                    num_tokens, merged_N, qp_dim, qp_dim + kv_dim, stream);
            } else {
                ops::invoke_dense_gemm(norm_out, q_proj_w_, qg_proj, num_tokens, qp_dim, hs, stream);
                ops::invoke_dense_gemm(norm_out, k_proj_w_, k,       num_tokens, kv_dim, hs, stream);
                ops::invoke_dense_gemm(norm_out, v_proj_w_, v,       num_tokens, kv_dim, hs, stream);
            }
        }
    }

    // 2b+3+4. Fused: deinterleave Q+Gate, Q/K per-head RMSNorm, partial RoPE
    // 3 kernels → 1: saves 2 launches/layer × 16 FullAttn = 32 launches/step
    __nv_bfloat16* q = qg_proj;
    __nv_bfloat16* attn_gate = gate_buf;
    if (q_norm_w_ && k_norm_w_) {
        ops::invoke_fused_qk_norm_rope(q, attn_gate, qg_proj, k,
                                        q_norm_w_, k_norm_w_, pos_ids,
                                        config_.rms_norm_eps, num_tokens, num_q, num_kv,
                                        hd, rot_d, config_.rope_theta, stream);
    } else {
        // Fallback: separate kernels
        if (q_norm_w_) {
            ops::invoke_fused_deinterleave_q_rmsnorm(q, attn_gate, qg_proj, q_norm_w_,
                                                      config_.rms_norm_eps, num_tokens, num_q, hd, stream);
        } else {
            ops::invoke_deinterleave_qgate(q, attn_gate, qg_proj, num_tokens, num_q, hd, stream);
        }
        if (k_norm_w_)
            ops::invoke_per_head_rmsnorm(k, k, k_norm_w_,
                                         config_.rms_norm_eps, num_tokens, num_kv, hd, stream, /*centered=*/true);
        ops::invoke_rope_partial(q, k, pos_ids, num_tokens, num_q, num_kv,
                                  hd, rot_d, config_.rope_theta, stream);
    }

    // 5. Write K/V into Paged Cache (per-layer)
    __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_manager.get_layer_k_cache(full_attn_idx));
    __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_manager.get_layer_v_cache(full_attn_idx));

    if (batch_size > 1) {
        // Batched decode: compute per-sequence write positions from context_lens
        // Place seq_positions int array at end of workspace (after down_out)
        int* seq_positions = reinterpret_cast<int*>(down_out + num_tokens * hs);
        compute_write_positions_kernel<<<1, batch_size, 0, stream>>>(
            seq_positions, context_lens, batch_size);
        ops::invoke_write_kv_cache(k_cache, v_cache, k, v,
                                    block_tables, 0 /*unused*/, num_tokens,
                                    num_kv, hd,
                                    kv_manager.get_block_size(), max_num_blocks_per_seq,
                                    stream, batch_size, seq_positions);
    } else {
        // Single sequence: write start = context_lens[0] - num_tokens
        // For decode (T=1): start = context_len - 1 (the new token's position)
        // For prefill (T>1): start = context_len - T (chunk's first position)
        int* seq_positions = reinterpret_cast<int*>(down_out + num_tokens * hs);
        compute_write_start_kernel<<<1, 1, 0, stream>>>(
            seq_positions, context_lens, num_tokens);
        ops::invoke_write_kv_cache(k_cache, v_cache, k, v,
                                    block_tables, 0 /*unused*/, num_tokens,
                                    num_kv, hd,
                                    kv_manager.get_block_size(), max_num_blocks_per_seq,
                                    stream, 1, seq_positions);
    }

    // 6. Attention
    float sm_scale = 1.0f / sqrtf((float)hd);

    if (streaming_ctx && streaming_ctx->total_ssd_blocks > 0) {
        // =========================================================
        // Streaming Paged Attention: GPU + SSD 两阶段
        // =========================================================
        __nv_bfloat16* k_cache_ptr = const_cast<__nv_bfloat16*>(kv_manager.get_layer_k_cache(full_attn_idx));
        __nv_bfloat16* v_cache_ptr = const_cast<__nv_bfloat16*>(kv_manager.get_layer_v_cache(full_attn_idx));
        int bs = kv_manager.get_block_size();

        // Phase 1: GPU-resident blocks — partial attention with causal masking
        // d_gpu_context_lens 已被 engine 设置为: gpu_tokens + current_chunk_tokens
        ops::invoke_paged_attention_partial(
            streaming_ctx->partial_out, streaming_ctx->partial_m, streaming_ctx->partial_l,
            q, k_cache_ptr, v_cache_ptr,
            streaming_ctx->d_gpu_block_tables, streaming_ctx->d_gpu_context_lens,
            streaming_ctx->gpu_num_blocks, streaming_ctx->gpu_num_blocks * bs,
            num_tokens, num_q, num_kv, hd, bs, sm_scale, stream,
            batch_size,
            0 /* forced_context_len = 0: 使用 causal masking */);

        // Phase 2: SSD blocks — 分批加载到 staging, partial attention, merge
        int loaded = 0;
        while (loaded < streaming_ctx->total_ssd_blocks) {
            int to_load = streaming_ctx->total_ssd_blocks - loaded;
            if (to_load > streaming_ctx->staging_capacity) to_load = streaming_ctx->staging_capacity;

            // 回调: engine 负责从 SSD 读取 blocks 到 staging_k/staging_v
            cudaStreamSynchronize(stream);  // 确保 GPU pass 完成, staging buffer 可写
            int actually_loaded = streaming_ctx->load_ssd_batch(full_attn_idx, loaded, to_load);
            if (actually_loaded <= 0) break;

            int ssd_batch_tokens = actually_loaded * bs;
            // 设置 staging context lens (device)
            cudaMemcpyAsync(streaming_ctx->d_staging_context_lens, &ssd_batch_tokens,
                            sizeof(int), cudaMemcpyHostToDevice, stream);

            // SSD batch partial attention: forced_context_len = ssd_batch_tokens
            // 所有 Q token 看到全部这批 SSD tokens, 无 causal masking
            ops::invoke_paged_attention_partial(
                streaming_ctx->partial_out2, streaming_ctx->partial_m2, streaming_ctx->partial_l2,
                q, streaming_ctx->staging_k, streaming_ctx->staging_v,
                streaming_ctx->d_staging_block_tables, streaming_ctx->d_staging_context_lens,
                actually_loaded, ssd_batch_tokens,
                num_tokens, num_q, num_kv, hd, bs, sm_scale, stream,
                1,
                ssd_batch_tokens /* forced_context_len: 无 causal masking */);

            // Merge: partial_out/m/l += partial_out2/m2/l2
            ops::invoke_merge_attention(
                streaming_ctx->partial_out, streaming_ctx->partial_m, streaming_ctx->partial_l,
                streaming_ctx->partial_out2, streaming_ctx->partial_m2, streaming_ctx->partial_l2,
                num_tokens, num_q, hd, stream);

            loaded += actually_loaded;
        }

        // Phase 3: Finalize: attn_out = acc / l
        ops::invoke_finalize_attention(
            attn_out, streaming_ctx->partial_out, streaming_ctx->partial_l,
            num_tokens, num_q, hd, stream);

    } else if (num_tokens >= 256 && batch_size <= 1 && !force_paged_attn) {
        // Prefill T≥256: GEMM-based causal self-attention with CUTLASS
        // score_workspace 复用 up_out 缓冲区 (MLP 阶段才需要)
        __nv_bfloat16* score_workspace = up_out;
        ops::invoke_prefill_attention(
            attn_out, q, k, v,
            num_tokens, num_q, num_kv, hd,
            sm_scale, score_workspace, stream);
    } else if (force_paged_attn && num_tokens > 1 && batch_size <= 1) {
        // Chunked prefill (chunk 1+): tiled GEMM attention with paged KV cache
        // Uses flash-attention-style tiling to avoid O(T_q × context) per-token traversal
        // max_context_len == context_lens[0] for batch_size=1
        __nv_bfloat16* chunked_workspace = up_out;
        ops::invoke_chunked_prefill_paged_attention(
            attn_out, q,
            kv_manager.get_layer_k_cache(full_attn_idx),
            kv_manager.get_layer_v_cache(full_attn_idx),
            block_tables, max_context_len, num_tokens,
            num_q, num_kv, hd,
            kv_manager.get_block_size(), max_num_blocks_per_seq,
            sm_scale, chunked_workspace, stream);
    } else {
        // Decode (T=1) or batched decode: use paged attention from KV cache
        ops::invoke_paged_attention(
            attn_out, q,
            kv_manager.get_layer_k_cache(full_attn_idx),
            kv_manager.get_layer_v_cache(full_attn_idx),
            block_tables, context_lens,
            max_num_blocks_per_seq, max_context_len,
            num_tokens, num_q, num_kv, hd,
            kv_manager.get_block_size(), sm_scale, stream,
            batch_size);
    }

    // 6b. Apply attention output gate: attn_out *= sigmoid(attn_gate)
    ops::invoke_sigmoid_mul(attn_out, attn_out, attn_gate, num_tokens * q_dim, stream);

    // 7. O Projection

    if (quantized_) {
        if (num_tokens == 1) {
            ops::invoke_fp4_gemv(attn_out, o_qw_, o_proj_out, stream);
        } else {
            ops::invoke_fp4_gemm(attn_out, o_qw_, o_proj_out, num_tokens, stream);
        }
    } else if (num_tokens == 1) {
        ops::invoke_dense_gemv(attn_out, o_proj_w_, o_proj_out, hs, q_dim, stream);
    } else {
        ops::invoke_dense_gemm(attn_out, o_proj_w_, o_proj_out, num_tokens, hs, q_dim, stream);
    }

    // 8+9. Residual + MLP
    if (is_moe()) {
        run_moe_mlp(hidden_states, post_norm_out, post_norm_out + num_tokens * hs,
                    moe_, config_, config_.rms_norm_eps, num_tokens, stream,
                    post_attention_layernorm_w_, o_proj_out);
    } else {
        run_mlp(hidden_states, post_norm_out, gate_buf, up_out, swiglu_out, down_out,
                gate_proj_w_, up_proj_w_, down_proj_w_,
                config_.rms_norm_eps, num_tokens, hs, is, stream,
                post_attention_layernorm_w_, o_proj_out,
                quantized_ ? &gate_qw_ : nullptr,
                quantized_ ? &up_qw_ : nullptr,
                quantized_ ? &down_qw_ : nullptr,
                quantized_ ? &gate_up_qw_merged_ : nullptr);
    }
}

// ============================================================================
// Qwen35LinearAttnLayer
// ============================================================================
Qwen35LinearAttnLayer::Qwen35LinearAttnLayer(const Qwen35Config& config, int layer_idx)
    : config_(config), layer_idx_(layer_idx) {}

void Qwen35LinearAttnLayer::set_weights(
    __nv_bfloat16* in_proj_qkv_w, __nv_bfloat16* in_proj_z_w,
    __nv_bfloat16* in_proj_a_w, __nv_bfloat16* in_proj_b_w,
    __nv_bfloat16* out_proj_w, __nv_bfloat16* conv1d_w,
    float* A_log,        // F32 pointer
    __nv_bfloat16* dt_bias,
    __nv_bfloat16* attn_norm_w,
    __nv_bfloat16* gate_proj_w, __nv_bfloat16* up_proj_w, __nv_bfloat16* down_proj_w,
    __nv_bfloat16* input_norm_w, __nv_bfloat16* post_attn_norm_w)
{
    in_proj_qkv_w_ = in_proj_qkv_w;  in_proj_z_w_  = in_proj_z_w;
    in_proj_a_w_   = in_proj_a_w;    in_proj_b_w_  = in_proj_b_w;
    out_proj_w_    = out_proj_w;      conv1d_w_     = conv1d_w;
    A_log_f32_     = A_log;           dt_bias_      = dt_bias;
    attn_norm_w_   = attn_norm_w;
    gate_proj_w_   = gate_proj_w;     up_proj_w_    = up_proj_w;
    down_proj_w_   = down_proj_w;
    input_layernorm_w_          = input_norm_w;
    post_attention_layernorm_w_ = post_attn_norm_w;
}

// Tiny device lambda as file-scope __global__ function
__global__ void silu_gate_kernel(__nv_bfloat16* y, const __nv_bfloat16* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float zv = __bfloat162float(z[idx]);
        float zg = zv / (1.f + expf(-zv));
        y[idx]   = __float2bfloat16(__bfloat162float(y[idx]) * zg);
    }
}

void Qwen35LinearAttnLayer::forward(
    __nv_bfloat16* hidden_states,
    __nv_bfloat16* ssm_state,
    __nv_bfloat16*  conv_state,
    int num_tokens,
    __nv_bfloat16* workspace,
    cudaStream_t stream,
    int batch_size,
    __nv_bfloat16** ssm_state_ptrs,
    __nv_bfloat16** conv_state_ptrs,
    __nv_bfloat16* ssm_state_checkpoint,
    __nv_bfloat16* conv_state_checkpoint,
    int num_checkpoints)
{
    if (!in_proj_qkv_w_ && !in_proj_qkv_qw_.valid()) return;

    const bool attn_fp4 = in_proj_qkv_qw_.valid();
    const int hs     = config_.hidden_size;
    const int is     = config_.intermediate_size;
    const int nkh    = config_.linear_num_key_heads;     // 16
    const int nv     = config_.linear_num_value_heads;   // 48
    const int kd     = config_.linear_key_head_dim;      // 128
    const int vd     = config_.linear_value_head_dim;    // 128
    const int lin_v  = config_.lin_v_dim();              // 6144 = nv*vd
    const int vpk    = config_.lin_v_per_kh();           // 384 = (nv/nkh)*vd
    const int qk_dim = config_.lin_qk_dim();             // 2048 = nkh*kd
    const int conv_k = config_.linear_conv_kernel_dim;
    const int in_qkv = qk_dim * 2 + lin_v;              // 10240
    const int nv_per_kh = nv / nkh;                     // 3

    // Workspace layout (per-token 尺寸最大化)
    __nv_bfloat16*  norm_out      = workspace;
    __nv_bfloat16*  qkv_out       = norm_out      + num_tokens * hs;
    __nv_bfloat16*  z_out         = qkv_out       + num_tokens * in_qkv;
    __nv_bfloat16*  a_out_f16     = z_out         + num_tokens * lin_v;  // [T, nv=48]
    __nv_bfloat16*  beta_out      = a_out_f16     + num_tokens * nv;     // [T, nv=48]
    __nv_bfloat16*  y_ssm         = beta_out      + num_tokens * nv;     // [T, lin_v=6144]
    __nv_bfloat16*  out_proj_buf  = y_ssm         + num_tokens * lin_v;
    __nv_bfloat16*  post_norm_out = out_proj_buf  + num_tokens * hs;
    __nv_bfloat16*  gate_out      = post_norm_out + num_tokens * hs;
    __nv_bfloat16*  up_out        = gate_out      + num_tokens * is;
    __nv_bfloat16*  swiglu_out    = up_out        + num_tokens * is;
    __nv_bfloat16*  down_out      = swiglu_out    + num_tokens * is;

    // 1+2. Input RMSNorm + All projections (QKV + ZAB)
    //   T=1 + BF16 super-merged: Fused RMSNorm in SMEM + GEMV (saves 1 launch)
    //   FP4 attn (Kbenkhaled): separate RMSNorm + FP4 GEMV/GEMM for QKV/Z + BF16 for A/B
    //   Otherwise: separate RMSNorm + BF16 GEMV/GEMM
    if (num_tokens == 1 && all_proj_merged_w_) {
        // Fused: RMSNorm in SMEM + super-merged GEMV (N=16480), no norm_out GMEM I/O
        ops::invoke_dense_gemv_with_rmsnorm(
            hidden_states, input_layernorm_w_, config_.rms_norm_eps,
            all_proj_merged_w_, qkv_out, in_qkv + lin_v + nv * 2, hs, stream);
    } else if (attn_fp4) {
        // FP4 attn path: separate RMSNorm + FP4 GEMV/GEMM for QKV/Z, BF16 for A/B
        ops::invoke_rmsnorm(norm_out, hidden_states, input_layernorm_w_,
                            config_.rms_norm_eps, num_tokens, hs, stream);
        if (num_tokens == 1) {
            if (qkv_z_qw_merged_.valid()) {
                // Merged FP4 QKV+Z: single GEMV for both (saves 1 launch)
                ops::invoke_fp4_gemv(norm_out, qkv_z_qw_merged_, qkv_out, stream);
            } else {
                ops::invoke_fp4_gemv(norm_out, in_proj_qkv_qw_, qkv_out, stream);
                ops::invoke_fp4_gemv(norm_out, in_proj_z_qw_,   z_out,   stream);
            }
            ops::invoke_dense_gemv(norm_out, in_proj_a_w_, a_out_f16, nv, hs, stream);
            ops::invoke_dense_gemv(norm_out, in_proj_b_w_, beta_out,  nv, hs, stream);
        } else {
            if (qkv_z_qw_merged_.valid()) {
                // Merged FP4 QKV+Z: single GEMM, then split
                int qkv_z_N = in_qkv + lin_v;  // 16384
                ops::invoke_fp4_gemm(norm_out, qkv_z_qw_merged_, gate_out, num_tokens, stream);
                ops::invoke_deinterleave_gemm_3way(qkv_out, z_out, z_out, gate_out,
                    num_tokens, qkv_z_N, in_qkv, qkv_z_N, stream);
            } else {
                ops::invoke_fp4_gemm(norm_out, in_proj_qkv_qw_, qkv_out, num_tokens, stream);
                ops::invoke_fp4_gemm(norm_out, in_proj_z_qw_,   z_out,   num_tokens, stream);
            }
            ops::invoke_dense_gemm(norm_out, in_proj_a_w_, a_out_f16, num_tokens, nv, hs, stream);
            ops::invoke_dense_gemm(norm_out, in_proj_b_w_, beta_out,  num_tokens, nv, hs, stream);
        }
    } else {
        ops::invoke_rmsnorm(norm_out, hidden_states, input_layernorm_w_,
                            config_.rms_norm_eps, num_tokens, hs, stream);
        if (num_tokens == 1) {
            if (all_proj_merged_w_) {
                ops::invoke_dense_gemv(norm_out, all_proj_merged_w_, qkv_out,
                                       in_qkv + lin_v + nv * 2, hs, stream);
            } else {
                ops::invoke_dense_gemv(norm_out, in_proj_qkv_w_, qkv_out, in_qkv, hs, stream);
                ops::invoke_dense_gemv(norm_out, in_proj_z_w_,   z_out,     lin_v,  hs, stream);
                ops::invoke_dense_gemv(norm_out, in_proj_a_w_,   a_out_f16, nv,     hs, stream);
                ops::invoke_dense_gemv(norm_out, in_proj_b_w_,   beta_out,  nv,     hs, stream);
            }
        } else {
            if (all_proj_merged_w_) {
                // T>1 + super-merged: merge QKV+Z into 1 GEMM (saves 1 cuBLAS call)
                // in_proj_qkv_w_ and in_proj_z_w_ are contiguous sub-pointers of all_proj_merged
                int qkv_z_N = in_qkv + lin_v;  // 10240 + 6144 = 16384
                // Use gate_out as temp (not used until MLP, large enough)
                ops::invoke_dense_gemm(norm_out, in_proj_qkv_w_, gate_out, num_tokens, qkv_z_N, hs, stream);
                // 2-way split: QKV[T, in_qkv] + Z[T, lin_v]
                ops::invoke_deinterleave_gemm_3way(qkv_out, z_out, z_out, gate_out,
                    num_tokens, qkv_z_N, in_qkv, qkv_z_N, stream);
                // A and B are tiny (N=48 each), keep separate
                ops::invoke_dense_gemm(norm_out, in_proj_a_w_, a_out_f16, num_tokens, nv, hs, stream);
                ops::invoke_dense_gemm(norm_out, in_proj_b_w_, beta_out, num_tokens, nv, hs, stream);
            } else {
                ops::invoke_dense_gemm(norm_out, in_proj_qkv_w_, qkv_out, num_tokens, in_qkv, hs, stream);
                ops::invoke_dense_gemm(norm_out, in_proj_z_w_,   z_out,     num_tokens, lin_v,  hs, stream);
                ops::invoke_dense_gemm(norm_out, in_proj_a_w_,   a_out_f16, num_tokens, nv,     hs, stream);
                ops::invoke_dense_gemm(norm_out, in_proj_b_w_,   beta_out,  num_tokens, nv,     hs, stream);
            }
        }
    }

    __nv_bfloat16* q_buf = qkv_out;
    __nv_bfloat16* k_buf = qkv_out + qk_dim;
    __nv_bfloat16* v_buf = qkv_out + qk_dim * 2;

    // 3. Short Conv1d on Q+K+V (all in_qkv channels) with SiLU activation
    if (conv1d_w_ && (conv_state || conv_state_ptrs)) {
        ops::invoke_causal_conv1d(qkv_out, conv_state,
                                   conv1d_w_, num_tokens, in_qkv, conv_k, stream,
                                   0 /* token_stride */, batch_size, conv_state_ptrs,
                                   conv_state_checkpoint, num_checkpoints);
    }

    // 4+5+6. Gated DeltaNet recurrence (alpha/sigmoid-beta computed inline)
    ops::invoke_gated_delta_net(
        q_buf, k_buf, v_buf,
        a_out_f16, dt_bias_, A_log_f32_, beta_out,
        ssm_state, y_ssm,
        num_tokens, nkh, kd, nv_per_kh, vd, stream, in_qkv,
        batch_size, ssm_state_ptrs,
        ssm_state_checkpoint, num_checkpoints);

    // 7+8. Fused per-head RMSNorm + SiLU gate: y_ssm = rmsnorm(y_ssm) * silu(z_out)
    if (attn_norm_w_) {
        ops::invoke_fused_norm_silu_gate(y_ssm, z_out, attn_norm_w_,
                                         config_.rms_norm_eps, num_tokens, nv, vd, stream);
    } else {
        int total = num_tokens * lin_v;
        silu_gate_kernel<<<(total+255)/256, 256, 0, stream>>>(y_ssm, z_out, total);
    }

    // 9. Output Projection
    if (out_proj_qw_.valid()) {
        if (num_tokens == 1) {
            ops::invoke_fp4_gemv(y_ssm, out_proj_qw_, out_proj_buf, stream);
        } else {
            ops::invoke_fp4_gemm(y_ssm, out_proj_qw_, out_proj_buf, num_tokens, stream);
        }
    } else if (num_tokens == 1) {
        ops::invoke_dense_gemv(y_ssm, out_proj_w_, out_proj_buf, hs, lin_v, stream);
    } else {
        ops::invoke_dense_gemm(y_ssm, out_proj_w_, out_proj_buf, num_tokens, hs, lin_v, stream);
    }

    // 10+11. Residual + MLP
    if (is_moe()) {
        run_moe_mlp(hidden_states, post_norm_out, post_norm_out + num_tokens * hs,
                    moe_, config_, config_.rms_norm_eps, num_tokens, stream,
                    post_attention_layernorm_w_, out_proj_buf);
    } else {
        run_mlp(hidden_states, post_norm_out, gate_out, up_out, swiglu_out, down_out,
                gate_proj_w_, up_proj_w_, down_proj_w_,
                config_.rms_norm_eps, num_tokens, hs, is, stream,
                post_attention_layernorm_w_, out_proj_buf,
                quantized_ ? &gate_qw_ : nullptr,
                quantized_ ? &up_qw_ : nullptr,
                quantized_ ? &down_qw_ : nullptr,
                quantized_ ? &gate_up_qw_merged_ : nullptr);
    }
}

// ============================================================================
// Qwen35Layer
// ============================================================================
Qwen35Layer::Qwen35Layer(const Qwen35Config& config, int layer_idx)
    : config_(config), layer_idx_(layer_idx)
{
    if (config.is_full_attention(layer_idx)) {
        full_attn_   = std::make_unique<Qwen35FullAttnLayer>(config, layer_idx);
    } else {
        linear_attn_ = std::make_unique<Qwen35LinearAttnLayer>(config, layer_idx);
    }
}

} // namespace core
} // namespace qwen_thor
