// KV Cache Offload — CUDA Kernels 实现
// Paged KV Cache ↔ 扁平 buffer 的高效 scatter/gather
//
// 内存带宽分析 (T=1024 tokens):
//   总数据量: 16 layers × 2(K/V) × 1024 × 4 heads × 256 dim × 2 bytes = 64 MB
//   Jetson Thor 实测带宽: ~230 GB/s
//   理论耗时: 64 MB / 230 GB/s = 0.28 ms
//   单次 kernel launch: ~5 μs overhead
//   远优于 2048 次 cudaMemcpy (每次 ~10 μs overhead = 20 ms)

#include "cache_kernels.h"
#include <algorithm>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// Extract kernel: Paged KV → flat buffer
// 每个线程处理一个 BF16 元素
// Grid:  ceil(total_elements / 256)
// Block: 256 threads
// ---------------------------------------------------------------------------
__global__ void extract_kv_from_pages_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_blocks_per_layer,
    int num_layers)
{
    // dst 的扁平索引: [layer, kv, token, head, dim]
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    if (tid >= total) return;

    // 解码多维索引
    int hd = num_kv_heads * head_dim;
    int d   = tid % head_dim;
    int h   = (tid / head_dim) % num_kv_heads;
    int t   = (tid / hd) % num_tokens;
    int kv  = (tid / (hd * num_tokens)) % 2;
    int l   = tid / (hd * num_tokens * 2);

    // 计算 paged cache 中的偏移
    int block_id = block_table[t / block_size];
    int slot     = t % block_size;

    // 每层的 cache 偏移: layer * num_blocks_per_layer * block_size * hd
    size_t cache_offset = ((size_t)(l * num_blocks_per_layer + block_id) * block_size + slot)
                          * num_kv_heads * head_dim + h * head_dim + d;

    const __nv_bfloat16* cache = (kv == 0) ? k_cache : v_cache;
    dst[tid] = cache[cache_offset];
}

// ---------------------------------------------------------------------------
// Inject kernel: flat buffer → Paged KV
// 与 extract 对称
// ---------------------------------------------------------------------------
__global__ void inject_kv_to_pages_kernel(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ block_table,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_blocks_per_layer,
    int num_layers)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    if (tid >= total) return;

    int hd = num_kv_heads * head_dim;
    int d   = tid % head_dim;
    int h   = (tid / head_dim) % num_kv_heads;
    int t   = (tid / hd) % num_tokens;
    int kv  = (tid / (hd * num_tokens)) % 2;
    int l   = tid / (hd * num_tokens * 2);

    int block_id = block_table[t / block_size];
    int slot     = t % block_size;

    size_t cache_offset = ((size_t)(l * num_blocks_per_layer + block_id) * block_size + slot)
                          * num_kv_heads * head_dim + h * head_dim + d;

    __nv_bfloat16* cache = (kv == 0) ? k_cache : v_cache;
    cache[cache_offset] = src[tid];
}

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------
void invoke_extract_kv_from_pages(
    __nv_bfloat16* dst,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* block_table,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_blocks_per_layer,
    int num_layers,
    cudaStream_t stream)
{
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    extract_kv_from_pages_kernel<<<blocks, threads, 0, stream>>>(
        dst, k_cache, v_cache, block_table,
        num_tokens, num_kv_heads, head_dim, block_size,
        num_blocks_per_layer, num_layers);
}

void invoke_inject_kv_to_pages(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* src,
    const int* block_table,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int num_blocks_per_layer,
    int num_layers,
    cudaStream_t stream)
{
    int total = num_layers * 2 * num_tokens * num_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    inject_kv_to_pages_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache, src, block_table,
        num_tokens, num_kv_heads, head_dim, block_size,
        num_blocks_per_layer, num_layers);
}

} // namespace cache
} // namespace qwen_thor
