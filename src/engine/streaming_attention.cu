// Streaming Paged Attention — CUDA kernels
//
// 三个 kernel:
//   1. paged_attention_partial_kernel: 与标准 paged attention 相同的 online softmax,
//      但输出 (acc, m, l) 而非 acc/l, 以便后续合并
//   2. merge_attention_kernel: 合并两个 partial results
//   3. finalize_attention_kernel: 最终归一化 out = acc / l

#include "streaming_attention.h"
#include <cuda_bf16.h>
#include <float.h>

namespace qwen_thor {
namespace ops {

#define SA_WARP_SIZE 32

// ==========================================================================
// Kernel 1: Partial Paged Attention
//   与 paged_attention_kernel 完全相同的逻辑, 但输出不归一化的 acc, m, l
//   Grid: (num_tokens, num_heads), Block: (head_dim)
// ==========================================================================
__global__ void paged_attention_partial_kernel(
    __nv_bfloat16*       out,        // [num_tokens, num_heads, head_dim] — weighted acc
    float*               out_m,      // [num_tokens, num_heads] — max score
    float*               out_l,      // [num_tokens, num_heads] — sum exp
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int*  block_tables,
    const int*  context_lens,
    int  max_num_blocks_per_seq,
    int  num_heads,
    int  num_kv_heads,
    float sm_scale,
    int  head_dim,
    int  block_size,
    int  batch_size,
    int  forced_context_len)       // >0: 全部 token 使用此值 (SSD pass 无 causal masking)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;

    int seq_idx;
    int context_len;
    if (forced_context_len > 0) {
        // SSD pass: 所有 token 看到全部 SSD blocks, 无 causal masking
        seq_idx = 0;
        context_len = forced_context_len;
    } else if (batch_size == 1) {
        seq_idx = 0;
        int total_context = context_lens[0];
        int num_tokens_grid = gridDim.x;
        int start_pos = total_context - num_tokens_grid;
        context_len = start_pos + token_idx + 1;
    } else {
        seq_idx = token_idx;
        context_len = context_lens[seq_idx];
    }

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int num_warps   = blockDim.x / SA_WARP_SIZE;

    extern __shared__ float s_smem[];
    float* s_q        = s_smem;
    float* s_qk_parts = s_smem + head_dim;

    // Load Q
    int q_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
    s_q[tid] = __bfloat162float(q[q_offset]) * sm_scale;
    __syncthreads();

    // Online softmax state
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc = 0.0f;

    // Iterate KV blocks
    int num_blocks = (context_len + block_size - 1) / block_size;
    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block_number = seq_block_table[block_idx];
        int start_token_in_block  = block_idx * block_size;
        int tokens_in_block = min(block_size, context_len - start_token_in_block);

        for (int i = 0; i < tokens_in_block; ++i) {
            int kv_offset = physical_block_number * (block_size * num_kv_heads * head_dim)
                          + i                     * (num_kv_heads * head_dim)
                          + kv_head_idx           * head_dim
                          + tid;

            float k_val = __bfloat162float(k_cache[kv_offset]);
            float qk    = s_q[tid] * k_val;

            // Warp reduction
            for (int offset = SA_WARP_SIZE / 2; offset > 0; offset /= 2)
                qk += __shfl_down_sync(0xffffffff, qk, offset);

            int wid  = tid / SA_WARP_SIZE;
            int lane = tid % SA_WARP_SIZE;
            if (lane == 0) s_qk_parts[wid] = qk;
            __syncthreads();

            float final_qk = 0.0f;
            if (tid < num_warps) {
                final_qk = s_qk_parts[tid];
                unsigned mask = (1u << num_warps) - 1u;
                for (int offset = num_warps / 2; offset > 0; offset /= 2)
                    final_qk += __shfl_down_sync(mask, final_qk, offset);
                if (tid == 0) s_qk_parts[0] = final_qk;
            }
            __syncthreads();
            final_qk = s_qk_parts[0];

            float m_i_new  = fmaxf(m_i, final_qk);
            float exp_qk   = expf(final_qk - m_i_new);
            float exp_diff = expf(m_i       - m_i_new);

            l_i = l_i * exp_diff + exp_qk;

            float v_val = __bfloat162float(v_cache[kv_offset]);
            acc = acc * exp_diff + exp_qk * v_val;

            m_i = m_i_new;
        }
    }

    // Output: weighted acc (NOT divided by l)
    int out_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
    out[out_offset] = __float2bfloat16(acc);

    // Output m, l (only tid==0 writes, but all threads have same m_i, l_i)
    if (tid == 0) {
        int ml_offset = token_idx * num_heads + head_idx;
        out_m[ml_offset] = m_i;
        out_l[ml_offset] = l_i;
    }
}

// ==========================================================================
// Kernel 2: Merge two partial attention results
//   合并公式 (online softmax merge):
//     m = max(m1, m2)
//     new_l1 = l1 * exp(m1 - m)
//     new_l2 = l2 * exp(m2 - m)
//     l = new_l1 + new_l2
//     acc[d] = acc1[d] * exp(m1 - m) + acc2[d] * exp(m2 - m)
//
//   Grid: (num_tokens, num_heads), Block: (head_dim)
//   In-place 更新 out1, m1, l1
// ==========================================================================
__global__ void merge_attention_kernel(
    __nv_bfloat16*       out1,   // [num_tokens, num_heads, head_dim] — in-place update
    float*               m1,     // [num_tokens, num_heads]
    float*               l1,     // [num_tokens, num_heads]
    const __nv_bfloat16* out2,
    const float*         m2,
    const float*         l2,
    int num_heads,
    int head_dim)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int d         = threadIdx.x;  // 0..head_dim-1

    int ml_offset = token_idx * num_heads + head_idx;
    float m1_val = m1[ml_offset];
    float m2_val = m2[ml_offset];
    float l1_val = l1[ml_offset];
    float l2_val = l2[ml_offset];

    // 如果 pass2 没有有效数据 (l2==0), 不合并
    if (l2_val <= 0.0f) return;
    // 如果 pass1 没有有效数据, 直接用 pass2
    if (l1_val <= 0.0f) {
        int off = token_idx * (num_heads * head_dim) + head_idx * head_dim + d;
        out1[off] = out2[off];
        if (d == 0) {
            m1[ml_offset] = m2_val;
            l1[ml_offset] = l2_val;
        }
        return;
    }

    float m_new = fmaxf(m1_val, m2_val);
    float scale1 = expf(m1_val - m_new);
    float scale2 = expf(m2_val - m_new);

    int off = token_idx * (num_heads * head_dim) + head_idx * head_dim + d;
    float a1 = __bfloat162float(out1[off]);
    float a2 = __bfloat162float(out2[off]);
    out1[off] = __float2bfloat16(a1 * scale1 + a2 * scale2);

    if (d == 0) {
        m1[ml_offset] = m_new;
        l1[ml_offset] = l1_val * scale1 + l2_val * scale2;
    }
}

// ==========================================================================
// Kernel 3: Finalize — out = acc / l
//   Grid: (num_tokens, num_heads), Block: (head_dim)
// ==========================================================================
__global__ void finalize_attention_kernel(
    __nv_bfloat16*       final_out,  // [num_tokens, num_heads, head_dim]
    const __nv_bfloat16* acc,
    const float*         l,
    int num_heads,
    int head_dim)
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int d         = threadIdx.x;

    int off = token_idx * (num_heads * head_dim) + head_idx * head_dim + d;
    int ml_offset = token_idx * num_heads + head_idx;
    float l_val = l[ml_offset];
    float a = __bfloat162float(acc[off]);
    final_out[off] = __float2bfloat16(l_val > 0.0f ? a / l_val : 0.0f);
}

// ==========================================================================
// Host-side launch functions
// ==========================================================================

void invoke_paged_attention_partial(
    __nv_bfloat16* d_out,
    float* d_m,
    float* d_l,
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
    int batch_size,
    int forced_context_len)
{
    if (batch_size <= 0) batch_size = 1;

    dim3 blocks(num_tokens, num_heads);
    dim3 threads(head_dim);
    int num_warps = head_dim / SA_WARP_SIZE;
    size_t smem_bytes = (size_t)(head_dim + num_warps) * sizeof(float);

    paged_attention_partial_kernel<<<blocks, threads, smem_bytes, stream>>>(
        d_out, d_m, d_l,
        q, k_cache, v_cache,
        block_tables, context_lens, max_num_blocks_per_seq,
        num_heads, num_kv_heads, sm_scale,
        head_dim, block_size, batch_size, forced_context_len
    );
}

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
    cudaStream_t stream)
{
    dim3 blocks(num_tokens, num_heads);
    dim3 threads(head_dim);
    merge_attention_kernel<<<blocks, threads, 0, stream>>>(
        d_out1, d_m1, d_l1,
        d_out2, d_m2, d_l2,
        num_heads, head_dim);
}

void invoke_finalize_attention(
    __nv_bfloat16* d_final_out,
    const __nv_bfloat16* d_acc,
    const float* d_l,
    int num_tokens,
    int num_heads,
    int head_dim,
    cudaStream_t stream)
{
    dim3 blocks(num_tokens, num_heads);
    dim3 threads(head_dim);
    finalize_attention_kernel<<<blocks, threads, 0, stream>>>(
        d_final_out, d_acc, d_l,
        num_heads, head_dim);
}

} // namespace ops
} // namespace qwen_thor
