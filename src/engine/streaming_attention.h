// Streaming Paged Attention — 处理 GPU + SSD 混合 KV blocks
//
// 当 KV cache budget 不足以容纳全部 context 时 (如 4 GB + 200K context):
//
//   Phase 1: 在 GPU-resident blocks 上运行 paged attention → 得到 partial result
//            (per-head 的 m, l, acc: online softmax 中间状态)
//
//   Phase 2: 从 SSD 分批加载 cold blocks 到 staging cache →
//            在 staging blocks 上运行 paged attention → 得到 partial result →
//            与 Phase 1 结果合并 (online softmax merge)
//
// 关键数学: Online Softmax Merge
//   给定两个 partial results:
//     pass1: (m1, l1, acc1) — max, sum_exp, weighted_sum
//     pass2: (m2, l2, acc2)
//   合并:
//     m = max(m1, m2)
//     l = l1 * exp(m1 - m) + l2 * exp(m2 - m)
//     acc = acc1 * exp(m1 - m) + acc2 * exp(m2 - m)
//     out = acc / l
//
// 实现:   paged_attention_partial_kernel  → 输出 (out, m, l) per head per token
//         merge_attention_kernel          → 合并两个 partial results
//         finalize_attention_kernel       → 最终归一化
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <functional>

namespace qwen_thor {
namespace ops {

// ---------------------------------------------------------------------------
// StreamingAttnCtx: 传递给 FullAttnLayer::forward 的流式注意力上下文
//
// 当非 null 且 total_ssd_blocks > 0 时, layer 使用两阶段注意力:
//   1. GPU 阶段: partial attention on GPU-resident blocks
//   2. SSD 阶段: 分批从 SSD 加载 blocks 到 staging, partial attention + merge
//   3. Finalize: out = acc / l
// ---------------------------------------------------------------------------
struct StreamingAttnCtx {
    // SSD block 信息
    int total_ssd_blocks = 0;         // SSD 上的 block 数量
    int ssd_tokens = 0;               // SSD 上的 token 数 (= total_ssd_blocks * block_size)
    int staging_capacity = 0;         // staging buffer 能容纳的 max blocks

    // GPU-resident block table (紧凑, 不含 SSD blocks)
    int* d_gpu_block_tables = nullptr; // device [gpu_num_blocks]
    int* d_gpu_context_lens = nullptr; // device [1] — GPU tokens + current chunk tokens
    int gpu_num_blocks = 0;

    // Staging KV cache (SSD blocks 加载到这里)
    __nv_bfloat16* staging_k = nullptr;   // device [staging_capacity * block_elems]
    __nv_bfloat16* staging_v = nullptr;   // device [staging_capacity * block_elems]
    int* d_staging_block_tables = nullptr; // device [staging_capacity] = {0,1,2,...}
    int* d_staging_context_lens = nullptr; // device [1]

    // Partial attention 状态 buffers (GPU pass)
    __nv_bfloat16* partial_out = nullptr;  // [num_tokens, num_heads, head_dim]
    float* partial_m = nullptr;            // [num_tokens, num_heads]
    float* partial_l = nullptr;            // [num_tokens, num_heads]

    // Partial attention 状态 buffers (SSD pass, 用于每批次)
    __nv_bfloat16* partial_out2 = nullptr;
    float* partial_m2 = nullptr;
    float* partial_l2 = nullptr;

    // 回调: 加载 SSD blocks 到 staging (由 engine 设置)
    // 参数: (full_attn_layer_idx, batch_start_in_ssd_indices, batch_count)
    // 返回: 实际加载的 block 数
    std::function<int(int, int, int)> load_ssd_batch;
};

// --- Partial paged attention: 输出 weighted sum + softmax state ---
// d_out:    [num_tokens, num_heads, head_dim]  — weighted accumulator (未归一化)
// d_m:      [num_tokens, num_heads]            — per-head max score
// d_l:      [num_tokens, num_heads]            — per-head sum of exp
//
// forced_context_len: 当 > 0 时, 所有 token 使用此值作为 context_len
//                     (SSD pass 需要: SSD tokens 对所有 Q token 完全可见, 无 causal masking)
void invoke_paged_attention_partial(
    __nv_bfloat16* d_out,     // weighted acc (不除以 l)
    float* d_m,               // max scores
    float* d_l,               // sum exp
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* block_tables,
    const int* context_lens,
    int max_num_blocks_per_seq,
    int max_context_len,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float sm_scale,
    cudaStream_t stream,
    int batch_size = 1,
    int forced_context_len = 0
);

// --- Merge two partial attention results using online softmax ---
// in-place 更新 d_out1/d_m1/d_l1, 用 d_out2/d_m2/d_l2 合并
void invoke_merge_attention(
    __nv_bfloat16* d_out1,
    float* d_m1,
    float* d_l1,
    const __nv_bfloat16* d_out2,
    const float* d_m2,
    const float* d_l2,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

// --- Finalize: out = acc / l ---
void invoke_finalize_attention(
    __nv_bfloat16* d_final_out,
    const __nv_bfloat16* d_acc,
    const float* d_l,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

} // namespace ops
} // namespace qwen_thor
