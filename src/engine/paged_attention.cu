// paged_attention.cu — 支持 batched decode 的 Paged Attention kernel + Prefill GEMM Attention
//
// Paged Attention: 每个 CUDA block 负责一个 (token_idx, head_idx) 对, 用于 decode
// Prefill Attention: CUTLASS GEMM + causal softmax, 用于 prefill (T ≥ 256)

#include "paged_attention.h"
#include "dense_gemm.h"
#include <cuda_bf16.h>
#include <float.h>
#include <stdio.h>

// Safe cudaMalloc with error checking
#define SAFE_CUDA_REALLOC(ptr, sz_var, need, stream) do { \
    if ((need) > (sz_var)) { \
        if (ptr) { cudaStreamSynchronize(stream); cudaFree(ptr); ptr = nullptr; } \
        cudaError_t _err = cudaMalloc(&(ptr), (need)); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "[CUDA] cudaMalloc failed (%zu bytes): %s\n", \
                    (size_t)(need), cudaGetErrorString(_err)); \
            fflush(stderr); \
            ptr = nullptr; sz_var = 0; \
        } else { \
            sz_var = (need); \
        } \
    } \
} while(0)

namespace qwen_thor {
namespace ops {

#define WARP_SIZE 32

__global__ void paged_attention_kernel(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int*  block_tables,         // [batch_size, max_num_blocks_per_seq]
    const int*  context_lens,         // [batch_size]
    int  max_num_blocks_per_seq,
    int  num_heads,
    int  num_kv_heads,
    float sm_scale,
    int  head_dim,
    int  block_size,
    int  batch_size)                  // batch_size==1 → prefill, ==num_tokens → batched decode
{
    int token_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    int tid       = threadIdx.x;   // 0 ~ head_dim-1

    // 确定此 token 属于哪个序列
    int seq_idx;
    int context_len;
    if (batch_size == 1) {
        // Prefill 模式: 所有 token 属于序列 0, causal masking
        seq_idx = 0;
        // causal: token i 能看到 [0, start_pos + token_idx] 共 start_pos+token_idx+1 个 token
        // start_pos 通过 context_lens[0] - num_tokens 计算
        // 但这里 context_lens[0] 就是完整序列长度, num_tokens = gridDim.x
        int total_context = context_lens[0];
        int num_tokens = gridDim.x;
        int start_pos = total_context - num_tokens;
        context_len = start_pos + token_idx + 1;
    } else {
        // Batched decode: token i 来自序列 i
        seq_idx = token_idx;
        context_len = context_lens[seq_idx];
    }

    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    int num_warps   = blockDim.x / WARP_SIZE;

    extern __shared__ float s_smem[];
    float* s_q        = s_smem;
    float* s_qk_parts = s_smem + head_dim;

    // ---- 1. Q → shared memory (乘 sm_scale) ----
    int q_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
    s_q[tid] = __bfloat162float(q[q_offset]) * sm_scale;
    __syncthreads();

    // ---- Online softmax 状态 ----
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc = 0.0f;

    // ---- 2. 遍历该序列的所有 KV block ----
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

            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                qk += __shfl_down_sync(0xffffffff, qk, offset);

            int wid  = tid / WARP_SIZE;
            int lane = tid % WARP_SIZE;
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

    // ---- 3. 写出结果 ----
    int out_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
    out[out_offset] = __float2bfloat16(l_i > 0.0f ? acc / l_i : 0.0f);
}

// --------------------------------------------------------------------------
// ==========================================================================
// Split-K Paged Attention for Decode (vLLM PagedAttention V2 style)
//
// 问题: 原始 kernel 每 (token, head) 1 个 block (共 24 blocks),
//       串行遍历所有 KV → 17% SM 占用率, 无法隐藏内存延迟.
//       32K context: 48ms/layer, 理论带宽下限 0.57ms.
//
// 方案: 将 KV range 按 partition_size 切分, 每 (token, head, partition)
//       一个 block. 每个 partition 独立做 online softmax, 输出 partial
//       (acc_f32, m_f32, l_f32). 然后 merge kernel 合并所有 partitions.
//
// Grid: (num_tokens, num_heads, num_partitions)
// Block: head_dim threads (256 for Qwen3.5)
// Shared: (head_dim + num_warps) × sizeof(float) = 1056 bytes
// ==========================================================================

__global__ void paged_attention_split_k_kernel(
    float* __restrict__  partial_out,   // [num_tokens, num_heads, max_parts, head_dim]
    float* __restrict__  partial_m,     // [num_tokens, num_heads, max_parts]
    float* __restrict__  partial_l,     // [num_tokens, num_heads, max_parts]
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int  max_num_blocks_per_seq,
    int  num_heads,
    int  num_kv_heads,
    float sm_scale,
    int  head_dim,
    int  block_size,
    int  batch_size,
    int  num_partitions,
    int  partition_size)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int part_idx  = blockIdx.z;
    const int tid       = threadIdx.x;  // 0..head_dim-1

    // 确定序列 & 上下文长度
    const int seq_idx = (batch_size > 1) ? token_idx : 0;
    const int context_len = context_lens[seq_idx];
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // 本 partition 的 KV range
    const int part_start = part_idx * partition_size;
    int part_end = part_start + partition_size;
    if (part_end > context_len) part_end = context_len;

    const int part_off = ((token_idx * num_heads + head_idx) * num_partitions + part_idx);

    // 空 partition → 写 sentinel 值
    if (part_start >= context_len) {
        partial_out[part_off * head_dim + tid] = 0.0f;
        if (tid == 0) {
            partial_m[part_off] = -FLT_MAX;
            partial_l[part_off] = 0.0f;
        }
        return;
    }

    const int num_warps = blockDim.x / WARP_SIZE;
    extern __shared__ float s_smem[];
    float* s_q        = s_smem;
    float* s_qk_parts = s_smem + head_dim;

    // 加载 Q (乘 sm_scale) 到 shared memory
    const int q_off = token_idx * (num_heads * head_dim) + head_idx * head_dim + tid;
    s_q[tid] = __bfloat162float(q[q_off]) * sm_scale;
    __syncthreads();

    // Online softmax 状态 (FP32)
    float m_i = -FLT_MAX;
    float l_i = 0.0f;
    float acc = 0.0f;

    const int* seq_bt = block_tables + seq_idx * max_num_blocks_per_seq;

    // 遍历本 partition 的 KV tokens
    for (int pos = part_start; pos < part_end; ++pos) {
        int pb   = seq_bt[pos / block_size];
        int slot = pos % block_size;
        int kv_off = pb * (block_size * num_kv_heads * head_dim)
                   + slot * (num_kv_heads * head_dim)
                   + kv_head_idx * head_dim + tid;

        // QK dot product
        float qk = s_q[tid] * __bfloat162float(k_cache[kv_off]);

        // Warp reduce
        for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
            qk += __shfl_down_sync(0xffffffff, qk, o);

        int wid  = tid / WARP_SIZE;
        int lane = tid % WARP_SIZE;
        if (lane == 0) s_qk_parts[wid] = qk;
        __syncthreads();

        float score = 0.0f;
        if (tid < num_warps) {
            score = s_qk_parts[tid];
            unsigned mask = (1u << num_warps) - 1u;
            for (int o = num_warps / 2; o > 0; o >>= 1)
                score += __shfl_down_sync(mask, score, o);
            if (tid == 0) s_qk_parts[0] = score;
        }
        __syncthreads();
        score = s_qk_parts[0];

        // Online softmax + V accumulation
        float v_val = __bfloat162float(v_cache[kv_off]);
        float m_new = fmaxf(m_i, score);
        float exp_diff = expf(m_i - m_new);
        float exp_s   = expf(score - m_new);
        acc = acc * exp_diff + exp_s * v_val;
        l_i = l_i * exp_diff + exp_s;
        m_i = m_new;
    }

    // 写 partial results (FP32)
    partial_out[part_off * head_dim + tid] = acc;
    if (tid == 0) {
        partial_m[part_off] = m_i;
        partial_l[part_off] = l_i;
    }
}

// ==========================================================================
// Split-K Merge Kernel
// 合并 num_partitions 个 partial results, 用 online softmax
// Grid: (num_tokens, num_heads), Block: head_dim threads
// ==========================================================================
__global__ void paged_attention_merge_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    int num_heads,
    int head_dim,
    int num_partitions)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int d         = threadIdx.x;

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    const int base = (token_idx * num_heads + head_idx) * num_partitions;

    for (int p = 0; p < num_partitions; p++) {
        float m_p = partial_m[base + p];
        float l_p = partial_l[base + p];
        if (l_p <= 0.0f) continue;  // 空 partition

        float o_p = partial_out[(base + p) * head_dim + d];
        float m_new = fmaxf(m, m_p);
        float a = expf(m - m_new);
        float b = expf(m_p - m_new);
        acc = acc * a + o_p * b;
        l   = l * a + l_p * b;
        m   = m_new;
    }

    int off = token_idx * (num_heads * head_dim) + head_idx * head_dim + d;
    out[off] = __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
}

// --------------------------------------------------------------------------
void invoke_paged_attention(
    __nv_bfloat16*       out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int*  block_tables,       // [batch_size, max_num_blocks_per_seq]
    const int*  context_lens,       // [batch_size]
    int  max_num_blocks_per_seq,
    int  max_context_len,
    int  num_tokens,
    int  num_heads,
    int  num_kv_heads,
    int  head_dim,
    int  block_size,
    float sm_scale,
    cudaStream_t stream,
    int  batch_size)
{
    if (batch_size <= 0) batch_size = 1;

    // Split-K 条件: decode 模式 + 上下文足够长
    // decode: num_tokens==1 (单序列) 或 batch_size>1 (batched decode)
    const int SPLIT_K_THRESHOLD = 512;
    bool use_split_k = (max_context_len >= SPLIT_K_THRESHOLD) &&
                       (num_tokens == 1 || batch_size > 1);

    if (use_split_k) {
        // 计算 partition 参数
        int partition_size = 256;
        int num_partitions = (max_context_len + partition_size - 1) / partition_size;
        if (num_partitions > 64) {
            num_partitions = 64;
            partition_size = (max_context_len + num_partitions - 1) / num_partitions;
        }

        // Lazy 静态分配 partial 缓冲区
        // 大小: num_tokens × num_heads × num_partitions × (head_dim + 2) × sizeof(float)
        static float* s_partial_out = nullptr;
        static float* s_partial_m   = nullptr;
        static float* s_partial_l   = nullptr;
        static size_t s_out_cap = 0, s_ml_cap = 0;

        size_t out_need = (size_t)num_tokens * num_heads * num_partitions * head_dim * sizeof(float);
        size_t ml_need  = (size_t)num_tokens * num_heads * num_partitions * sizeof(float);

        SAFE_CUDA_REALLOC(s_partial_out, s_out_cap, out_need, stream);
        if (ml_need > s_ml_cap) {
            if (s_partial_m || s_partial_l) { cudaStreamSynchronize(stream); }
            if (s_partial_m) { cudaFree(s_partial_m); s_partial_m = nullptr; }
            if (s_partial_l) { cudaFree(s_partial_l); s_partial_l = nullptr; }
            cudaError_t _e1 = cudaMalloc(&s_partial_m, ml_need);
            cudaError_t _e2 = cudaMalloc(&s_partial_l, ml_need);
            if (_e1 != cudaSuccess || _e2 != cudaSuccess) {
                fprintf(stderr, "[CUDA] split-K cudaMalloc failed: %s / %s\n",
                        cudaGetErrorString(_e1), cudaGetErrorString(_e2));
                fflush(stderr);
                s_ml_cap = 0;
            } else {
                s_ml_cap = ml_need;
            }
        }
        if (!s_partial_out || !s_partial_m || !s_partial_l) return; // OOM guard

        // Phase 1: Split-K partial attention
        dim3 grid(num_tokens, num_heads, num_partitions);
        int num_warps  = head_dim / WARP_SIZE;
        size_t smem    = (size_t)(head_dim + num_warps) * sizeof(float);

        paged_attention_split_k_kernel<<<grid, head_dim, smem, stream>>>(
            s_partial_out, s_partial_m, s_partial_l,
            q, k_cache, v_cache,
            block_tables, context_lens, max_num_blocks_per_seq,
            num_heads, num_kv_heads, sm_scale,
            head_dim, block_size, batch_size,
            num_partitions, partition_size);

        // Phase 2: Merge partitions → final BF16 output
        dim3 merge_grid(num_tokens, num_heads);
        paged_attention_merge_kernel<<<merge_grid, head_dim, 0, stream>>>(
            out, s_partial_out, s_partial_m, s_partial_l,
            num_heads, head_dim, num_partitions);
    } else {
        // 原始 kernel: 短上下文或 prefill 模式
        dim3 blocks(num_tokens, num_heads);
        dim3 threads(head_dim);

        int    num_warps   = head_dim / WARP_SIZE;
        size_t smem_bytes  = (size_t)(head_dim + num_warps) * sizeof(float);

        paged_attention_kernel<<<blocks, threads, smem_bytes, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, context_lens, max_num_blocks_per_seq,
            num_heads, num_kv_heads, sm_scale,
            head_dim, block_size, batch_size
        );
    }
}

// =============================================================================
// Prefill Attention: CUTLASS GEMM + data layout kernels + causal softmax
// 替代 cuBLAS: 使用已编译的 CUTLASS kernel (无 JIT 开销)
// =============================================================================

// Gather Q heads for one GQA group into interleaved [hpg*T, hd] contiguous layout
// Row 6t+b corresponds to Q head (group_start_head + b) at token t
// Applies sm_scale to Q during extraction
__global__ void gather_q_group_kernel(
    __nv_bfloat16* __restrict__ dst,     // [hpg*T, hd] contiguous
    const __nv_bfloat16* __restrict__ src, // [T, q_dim]
    int T, int hd, int q_dim, int group_start_head, int hpg, float sm_scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hpg * T * hd;
    if (idx >= total) return;
    int d  = idx % hd;
    int tb = idx / hd;
    int t  = tb / hpg;
    int b  = tb % hpg;
    float val = __bfloat162float(src[t * q_dim + (group_start_head + b) * hd + d]);
    dst[idx] = __float2bfloat16(val * sm_scale);
}

// Extract one KV head from interleaved [T, kv_dim] to contiguous [T, hd]
__global__ void extract_kv_head_kernel(
    __nv_bfloat16* __restrict__ dst,       // [T, hd] contiguous
    const __nv_bfloat16* __restrict__ src,  // [T, kv_dim]
    int T, int hd, int kv_dim, int head_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * hd) return;
    int t = idx / hd;
    int d = idx % hd;
    dst[idx] = src[t * kv_dim + head_idx * hd + d];
}

// Transpose [T, hd] row-major → [hd, T] row-major (standard, smem-tiled for bank-conflict avoidance)
__global__ void transpose_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,       // [hd, T] row-major
    const __nv_bfloat16* __restrict__ src,  // [T, hd] row-major
    int T, int hd)
{
    __shared__ __nv_bfloat16 tile[32][33]; // +1 for bank-conflict avoidance
    int bx = blockIdx.x * 32; // along T
    int by = blockIdx.y * 32; // along hd
    int tx = threadIdx.x;     // within tile
    int ty = threadIdx.y;

    // Load src[bx+ty, by+tx] into tile[ty][tx]
    if ((bx + ty) < T && (by + tx) < hd)
        tile[ty][tx] = src[(bx + ty) * hd + (by + tx)];
    __syncthreads();

    // Write tile[tx][ty] to dst[by+ty, bx+tx] = dst transposed
    if ((by + ty) < hd && (bx + tx) < T)
        dst[(by + ty) * T + (bx + tx)] = tile[tx][ty];
}

// Scatter [hpg*T, hd] → [T, q_dim] (reverse of gather_q_group)
__global__ void scatter_out_group_kernel(
    __nv_bfloat16* __restrict__ dst,       // [T, q_dim]
    const __nv_bfloat16* __restrict__ src,  // [hpg*T, hd] contiguous
    int T, int hd, int q_dim, int group_start_head, int hpg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = hpg * T * hd;
    if (idx >= total) return;
    int d  = idx % hd;
    int tb = idx / hd;
    int t  = tb / hpg;
    int b  = tb % hpg;
    dst[t * q_dim + (group_start_head + b) * hd + d] = src[idx];
}

// Causal softmax for interleaved Q layout:
// Score[hpg*T, T] where row r corresponds to token t = r / hpg
// valid_len = t + 1 (causal: query at position t attends to keys 0..t)
__global__ void causal_softmax_interleaved_kernel(
    __nv_bfloat16* __restrict__ scores,   // [hpg*T, T_padded] row-major
    int T,         // actual sequence length
    int T_padded,  // padded T (cols in score matrix)
    int hpg)       // heads per group
{
    const int row = blockIdx.x;  // 0..(hpg*T-1)
    const int tid = threadIdx.x;
    const int t = row / hpg;     // actual query token index
    const int valid_len = t + 1; // causal: can attend to 0..t

    __nv_bfloat16* row_data = scores + (long long)row * T_padded;

    extern __shared__ float smem[];

    // Pass 1: find max
    float thread_max = -FLT_MAX;
    for (int i = tid; i < valid_len; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __bfloat162float(row_data[i]));
    }
    smem[tid] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // Pass 2: sum of exp
    float thread_sum = 0.0f;
    for (int i = tid; i < valid_len; i += blockDim.x) {
        thread_sum += expf(__bfloat162float(row_data[i]) - max_val);
    }
    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = (smem[0] > 0.0f) ? 1.0f / smem[0] : 0.0f;

    // Pass 3: normalize valid positions
    for (int i = tid; i < valid_len; i += blockDim.x) {
        float v = expf(__bfloat162float(row_data[i]) - max_val) * inv_sum;
        row_data[i] = __float2bfloat16(v);
    }
    // Zero invalid positions (including padding)
    for (int i = valid_len + tid; i < T_padded; i += blockDim.x) {
        row_data[i] = __nv_bfloat16(0);
    }
}

// --------------------------------------------------------------------------
// ===========================================================================
// Fused Prefill Attention Kernel (FlashAttention-style online softmax)
//
// Grid: (num_heads, T) — one block per (head, query_position)
// Block: head_dim threads — one thread per output dimension
//
// 直接从 strided Q/K/V 布局读取, 无需 extract/transpose/gather/scatter.
// 使用 online softmax 避免材料化 [T,T] score 矩阵.
// GQA: kv_head = head / (num_heads / num_kv_heads)
// 
// 对比旧实现 (28 kernel launches per layer): 理论 ~40× 提速
// ===========================================================================

// Warp-level reduce sum (full warp, 32 threads)
__device__ __forceinline__ float warpReduceSum_attn(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduce sum for exactly 256 threads (8 warps)
// Returns result in all threads (broadcast)
__device__ __forceinline__ float blockReduceSum_attn(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warpReduceSum_attn(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    // Warp 0 reduces 8 warp-level sums (all 32 threads participate to avoid shfl deadlock)
    if (warp == 0) {
        val = (lane < 8) ? smem[lane] : 0.0f;
        val = warpReduceSum_attn(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

__global__ void __launch_bounds__(256, 1)
fused_prefill_attention_kernel(
    __nv_bfloat16* __restrict__ out,       // [T, q_dim]
    const __nv_bfloat16* __restrict__ q,   // [T, q_dim], q_dim = num_heads * hd
    const __nv_bfloat16* __restrict__ k,   // [T, kv_dim], kv_dim = num_kv_heads * hd
    const __nv_bfloat16* __restrict__ v,   // [T, kv_dim]
    int T, int q_dim, int kv_dim, int hpg, int hd,
    float sm_scale)
{
    const int head = blockIdx.x;    // 0..num_heads-1
    const int t    = blockIdx.y;    // query position 0..T-1
    const int d    = threadIdx.x;   // output dimension 0..hd-1
    const int kv_head = head / hpg;

    // Shared memory for block reduce (8 warps)
    __shared__ float reduce_smem[8];

    // Load q[t, head, d] once
    float q_d = __bfloat162float(q[t * q_dim + head * hd + d]) * sm_scale;

    // Online softmax state
    float max_score = -FLT_MAX;
    float sum_exp = 0.0f;
    float o_acc = 0.0f;  // weighted value accumulator

    // Causal loop: attend to positions 0..t
    for (int s = 0; s <= t; s++) {
        // 1. Compute dot(q[t,head], k[s,kv_head]) — each thread contributes q_d * k_d
        float k_d = __bfloat162float(k[s * kv_dim + kv_head * hd + d]);
        float partial = q_d * k_d;

        // 2. Block-level reduction to get full score
        float score = blockReduceSum_attn(partial, reduce_smem);

        // 3. Read v[s, kv_head, d]
        float v_d = __bfloat162float(v[s * kv_dim + kv_head * hd + d]);

        // 4. Online softmax + value accumulation
        if (score > max_score) {
            float rescale = expf(max_score - score);
            o_acc = o_acc * rescale + v_d;
            sum_exp = sum_exp * rescale + 1.0f;
            max_score = score;
        } else {
            float w = expf(score - max_score);
            o_acc += w * v_d;
            sum_exp += w;
        }
    }

    // 5. Write normalized output
    out[t * q_dim + head * hd + d] = __float2bfloat16(o_acc / sum_exp);
}

// --------------------------------------------------------------------------
void invoke_prefill_attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int T,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    __nv_bfloat16* workspace,     // workspace for temp buffers
    cudaStream_t stream)
{
    const int q_dim  = num_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int hpg    = num_heads / num_kv_heads;  // heads per group = 6

    __nv_bfloat16* k_all     = workspace;
    __nv_bfloat16* v_all_t   = k_all     + num_kv_heads * T * head_dim;
    __nv_bfloat16* q_grp     = v_all_t   + num_kv_heads * head_dim * T;
    __nv_bfloat16* score_buf = q_grp     + hpg * T * head_dim;
    __nv_bfloat16* out_grp   = score_buf + (long long)hpg * T * T;

    const int block_sz = 256;
    const int kv_elems = T * head_dim;
    const int kv_grids = (kv_elems + block_sz - 1) / block_sz;

    // 1. Pre-extract K/V heads
    for (int g = 0; g < num_kv_heads; g++) {
        extract_kv_head_kernel<<<kv_grids, block_sz, 0, stream>>>(
            k_all + g * T * head_dim, k, T, head_dim, kv_dim, g);
        extract_kv_head_kernel<<<kv_grids, block_sz, 0, stream>>>(
            out_grp, v, T, head_dim, kv_dim, g);
        dim3 tp_grid((T + 31) / 32, (head_dim + 31) / 32);
        dim3 tp_block(32, 32);
        transpose_bf16_kernel<<<tp_grid, tp_block, 0, stream>>>(
            v_all_t + g * head_dim * T, out_grp, T, head_dim);
    }

    int sm_threads = 256;
    while (sm_threads > T && sm_threads > 32) sm_threads >>= 1;

    // 2. Per-KV-group loop
    for (int g = 0; g < num_kv_heads; g++) {
        int q_elems = hpg * T * head_dim;
        int q_grids_n = (q_elems + block_sz - 1) / block_sz;

        // Gather Q
        gather_q_group_kernel<<<q_grids_n, block_sz, 0, stream>>>(
            q_grp, q, T, head_dim, q_dim, g * hpg, hpg, sm_scale);

        // Score GEMM
        invoke_dense_gemm(q_grp, k_all + g * T * head_dim, score_buf,
                         hpg * T, T, head_dim, stream);

        // Softmax
        causal_softmax_interleaved_kernel<<<hpg * T, sm_threads,
            sm_threads * sizeof(float), stream>>>(
            score_buf, T, T, hpg);

        // Output GEMM
        invoke_dense_gemm(score_buf, v_all_t + g * head_dim * T, out_grp,
                         hpg * T, head_dim, T, stream);

        // Scatter
        scatter_out_group_kernel<<<q_grids_n, block_sz, 0, stream>>>(
            out, out_grp, T, head_dim, q_dim, g * hpg, hpg);
    }
}

// ============================================================================
// Chunked Prefill Paged Attention: Flash-Attention-style Tiled GEMM
//
// For chunked prefill chunks 1+ where force_paged_attn=true and T_q > 1.
// Instead of per-token paged attention (O(T_q × context) sequential),
// uses tiled GEMM: K/V gathered from paged cache per tile, score GEMM +
// online softmax merge across tiles.
//
// Expected speedup: ~1000× for 8K context (69s → <100ms)
// ============================================================================

static constexpr int CHUNKED_TILE_SZ = 256;

// Gather K or V for positions [tile_start, tile_start+actual_sz) from paged cache
// into contiguous [actual_sz, head_dim] for one kv_head
__global__ void gather_kv_paged_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ kv_cache,
    const int* __restrict__ block_table,
    int tile_start, int actual_sz,
    int kv_head, int num_kv_heads, int head_dim, int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= actual_sz * head_dim) return;
    int pos_in_tile = idx / head_dim;
    int d = idx % head_dim;
    int abs_pos = tile_start + pos_in_tile;
    int logical_block = abs_pos / block_size;
    int slot = abs_pos % block_size;
    int pb = block_table[logical_block];
    // Layout: [physical_block, slot, num_kv_heads, head_dim]
    dst[idx] = kv_cache[pb * (block_size * num_kv_heads * head_dim)
                       + slot * (num_kv_heads * head_dim)
                       + kv_head * head_dim + d];
}

// Initialize online softmax state: acc=0, m=-inf, l=0
__global__ void init_online_softmax_kernel(float* acc, float* m, float* l, int M, int hd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * hd) acc[idx] = 0.0f;
    if (idx < M) { m[idx] = -FLT_MAX; l[idx] = 0.0f; }
}

// Tiled causal softmax: produce P = exp(S - rowmax), write m_tile and l_tile.
// score_buf [M, actual_sz] in row-major, M = hpg × T_q.
// Causal: query at row i → abs position start_pos + (i / hpg),
//         can attend to key positions ≤ that position.
__global__ void tiled_causal_softmax_kernel(
    __nv_bfloat16* __restrict__ score_buf,  // [M, actual_sz] in-place → P
    float* __restrict__ m_tile,              // [M]
    float* __restrict__ l_tile,              // [M]
    int M, int actual_sz,
    int T_q, int hpg,
    int start_pos, int tile_start)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    int q_idx = row / hpg;
    int q_abs_pos = start_pos + q_idx;
    // Causal: can attend to positions [tile_start, min(tile_start+actual_sz, q_abs_pos+1))
    int valid_end = actual_sz;
    int causal_end = q_abs_pos - tile_start + 1;
    if (causal_end < valid_end) valid_end = causal_end;

    __nv_bfloat16* row_data = score_buf + (long long)row * actual_sz;
    extern __shared__ float smem[];

    if (valid_end <= 0) {
        if (tid == 0) { m_tile[row] = -FLT_MAX; l_tile[row] = 0.0f; }
        for (int j = tid; j < actual_sz; j += blockDim.x)
            row_data[j] = __float2bfloat16(0.0f);
        return;
    }

    // Pass 1: rowmax
    float thread_max = -FLT_MAX;
    for (int j = tid; j < valid_end; j += blockDim.x)
        thread_max = fmaxf(thread_max, __bfloat162float(row_data[j]));
    smem[tid] = thread_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];

    // Pass 2: exp + sum
    float thread_sum = 0.0f;
    for (int j = tid; j < valid_end; j += blockDim.x) {
        float p = expf(__bfloat162float(row_data[j]) - max_val);
        row_data[j] = __float2bfloat16(p);
        thread_sum += p;
    }
    // Zero invalid positions
    for (int j = valid_end + tid; j < actual_sz; j += blockDim.x)
        row_data[j] = __float2bfloat16(0.0f);
    smem[tid] = thread_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) { m_tile[row] = max_val; l_tile[row] = smem[0]; }
}

// Merge tile output into running online softmax state.
// acc[i,d] = acc[i,d] * α + O_tile[i,d] * β
// where α = exp(m_old - m_new), β = exp(m_tile - m_new), m_new = max(m_old, m_tile)
// Grid: M blocks, head_dim threads per block (one row per block)
__global__ void merge_attention_tile_kernel(
    float* __restrict__ acc,
    float* __restrict__ m,
    float* __restrict__ l,
    const __nv_bfloat16* __restrict__ O_tile,
    const float* __restrict__ m_tile,
    const float* __restrict__ l_tile,
    int M, int hd)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int d = threadIdx.x;
    if (d >= hd) return;

    float m_old = m[row];
    float m_t = m_tile[row];
    float m_new = fmaxf(m_old, m_t);
    float alpha = expf(m_old - m_new);
    float beta  = expf(m_t - m_new);

    int off = row * hd + d;
    acc[off] = acc[off] * alpha + __bfloat162float(O_tile[off]) * beta;

    if (d == 0) {
        l[row] = l[row] * alpha + l_tile[row] * beta;
        m[row] = m_new;
    }
}

// Finalize: out[i,d] = acc[i,d] / l[i]
// Grid: M blocks, head_dim threads per block
__global__ void finalize_chunked_softmax_kernel(
    __nv_bfloat16* __restrict__ out,
    const float* __restrict__ acc,
    const float* __restrict__ l,
    int M, int hd)
{
    int row = blockIdx.x;
    if (row >= M) return;
    int d = threadIdx.x;
    if (d >= hd) return;
    float inv_l = (l[row] > 0.0f) ? 1.0f / l[row] : 0.0f;
    out[row * hd + d] = __float2bfloat16(acc[row * hd + d] * inv_l);
}

// --------------------------------------------------------------------------
void invoke_chunked_prefill_paged_attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* block_tables,
    int context_len,
    int T_q,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    float sm_scale,
    __nv_bfloat16* workspace,
    cudaStream_t stream)
{
    const int hpg    = num_heads / num_kv_heads;
    const int q_dim  = num_heads * head_dim;
    const int M      = hpg * T_q;
    const int start_pos = context_len - T_q;
    const int TILE   = CHUNKED_TILE_SZ;

    // Workspace layout:
    //   q_grp      [M, hd]          BF16
    //   score_buf  [M, TILE]        BF16
    //   k_tile     [TILE, hd]       BF16
    //   v_tile     [TILE, hd]       BF16
    //   v_tile_t   [hd, TILE]       BF16
    //   O_tile_buf [M, hd]          BF16 (reused as out_grp after loop)
    //   acc        [M, hd]          FP32
    //   m_buf      [M]              FP32
    //   l_buf      [M]              FP32
    //   m_tile_f   [M]              FP32
    //   l_tile_f   [M]              FP32
    __nv_bfloat16* q_grp      = workspace;
    __nv_bfloat16* score_buf  = q_grp     + (long long)M * head_dim;
    __nv_bfloat16* k_tile     = score_buf + (long long)M * TILE;
    __nv_bfloat16* v_tile     = k_tile    + TILE * head_dim;
    __nv_bfloat16* v_tile_t   = v_tile    + TILE * head_dim;
    __nv_bfloat16* O_tile_buf = v_tile_t  + head_dim * TILE;

    float* acc      = (float*)(O_tile_buf + (long long)M * head_dim);
    float* m_buf    = acc     + (long long)M * head_dim;
    float* l_buf    = m_buf   + M;
    float* m_tile_f = l_buf   + M;
    float* l_tile_f = m_tile_f + M;

    const int blk = 256;

    int sm_threads = 256;
    while (sm_threads > TILE && sm_threads > 32) sm_threads >>= 1;

    // Per KV group
    for (int g = 0; g < num_kv_heads; g++) {
        // Gather Q for this group (interleaved [hpg*T_q, hd])
        int q_elems = M * head_dim;
        gather_q_group_kernel<<<(q_elems + blk - 1) / blk, blk, 0, stream>>>(
            q_grp, q, T_q, head_dim, q_dim, g * hpg, hpg, sm_scale);

        // Init online softmax state
        int init_elems = M * head_dim;
        init_online_softmax_kernel<<<(init_elems + blk - 1) / blk, blk, 0, stream>>>(
            acc, m_buf, l_buf, M, head_dim);

        // Tile loop over KV positions
        for (int tile_start = 0; tile_start < context_len; tile_start += TILE) {
            int actual_sz = context_len - tile_start;
            if (actual_sz > TILE) actual_sz = TILE;
            int kv_elems = actual_sz * head_dim;
            int kv_grids = (kv_elems + blk - 1) / blk;

            // Gather K tile from paged cache
            gather_kv_paged_kernel<<<kv_grids, blk, 0, stream>>>(
                k_tile, k_cache, block_tables, tile_start, actual_sz,
                g, num_kv_heads, head_dim, block_size);

            // Score GEMM: q_grp[M, hd] × k_tile^T → score_buf[M, actual_sz]
            // k_tile is [actual_sz, hd] row-major = [hd, actual_sz] col-major
            // GEMM: C[M, actual_sz] = A[M, hd] × B[hd, actual_sz]
            invoke_dense_gemm(q_grp, k_tile, score_buf, M, actual_sz, head_dim, stream);

            // Tiled causal softmax → P in-place, m_tile, l_tile
            tiled_causal_softmax_kernel<<<M, sm_threads,
                sm_threads * sizeof(float), stream>>>(
                score_buf, m_tile_f, l_tile_f,
                M, actual_sz, T_q, hpg, start_pos, tile_start);

            // Gather V tile from paged cache
            gather_kv_paged_kernel<<<kv_grids, blk, 0, stream>>>(
                v_tile, v_cache, block_tables, tile_start, actual_sz,
                g, num_kv_heads, head_dim, block_size);

            // Transpose V: [actual_sz, hd] → [hd, actual_sz]
            dim3 tp_grid((actual_sz + 31) / 32, (head_dim + 31) / 32);
            dim3 tp_block(32, 32);
            transpose_bf16_kernel<<<tp_grid, tp_block, 0, stream>>>(
                v_tile_t, v_tile, actual_sz, head_dim);

            // Output GEMM: P[M, actual_sz] × V_tile_T → O_tile[M, hd]
            // V_tile_T is [hd, actual_sz] row-major = [actual_sz, hd] col-major
            // GEMM: C[M, hd] = A[M, actual_sz] × B[actual_sz, hd]
            invoke_dense_gemm(score_buf, v_tile_t, O_tile_buf, M, head_dim, actual_sz, stream);

            // Merge into online softmax accumulator
            merge_attention_tile_kernel<<<M, head_dim, 0, stream>>>(
                acc, m_buf, l_buf, O_tile_buf, m_tile_f, l_tile_f, M, head_dim);
        }

        // Finalize: out_grp = acc / l (reuse O_tile_buf as out_grp)
        __nv_bfloat16* out_grp = O_tile_buf;
        finalize_chunked_softmax_kernel<<<M, head_dim, 0, stream>>>(
            out_grp, acc, l_buf, M, head_dim);

        // Scatter back to [T_q, q_dim]
        int q_grids_n = (q_elems + blk - 1) / blk;
        scatter_out_group_kernel<<<q_grids_n, blk, 0, stream>>>(
            out, out_grp, T_q, head_dim, q_dim, g * hpg, hpg);
    }
}

} // namespace ops
} // namespace qwen_thor
