// KV Cache Offload — CUDA Kernels
// KV cache ↔ flat buffer 的 scatter/gather 操作
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// extract_kv_from_pages: 从 Paged KV Cache 提取 KV 数据到扁平连续 buffer
//
// 功能: 遍历 block_table 中的所有 token, 从 paged k/v_cache 中提取数据
//       写入平坦连续布局:
//         dst[layer, kv, token, head, dim] — row-major, BF16
//
// 参数:
//   dst:        输出 buffer [num_layers * 2 * num_tokens * num_kv_heads * head_dim] BF16
//   k_cache:    paged K cache (所有层) [total_blocks, block_size, num_kv_heads, head_dim]
//   v_cache:    paged V cache (同上)
//   block_table: [max_blocks_per_seq] 物理 block ID
//   num_tokens:  要提取的 token 数
//   num_kv_heads, head_dim, block_size: KV cache 参数
//   num_blocks_per_layer: KV pool 中每层的 block 数
//   num_layers:  full attention 层数 (16)
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
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// inject_kv_to_pages: 从扁平 buffer 写回 Paged KV Cache
//
// 与 extract 相反: 读取 src 中的连续 KV 数据, 写入 paged k/v_cache
//
// 参数:
//   k_cache, v_cache: paged cache (non-const, 可写)
//   src:        输入 buffer [num_layers * 2 * num_tokens * num_kv_heads * head_dim] BF16
//   block_table: [max_blocks_per_seq]
//   其他参数同上
// ---------------------------------------------------------------------------
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
    cudaStream_t stream);

} // namespace cache
} // namespace qwen_thor
