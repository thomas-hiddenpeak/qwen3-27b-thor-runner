#pragma once

#include "tensor.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace ops {

// Paged Attention 算子 (针对 SM110 优化)
void invoke_paged_attention(
    __nv_bfloat16* out,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    const int* block_tables,      // [batch_size, max_num_blocks_per_seq]
    const int* context_lens,      // [batch_size]
    int max_num_blocks_per_seq,
    int max_context_len,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    float sm_scale,
    cudaStream_t stream = 0,
    int batch_size = 1            // 1=prefill (single seq), >1=batched decode
);

// -----------------------------------------------------------------------------
// KVCacheManager: 负责管理 PagedAttention 的显存池
// -----------------------------------------------------------------------------
class KVCacheManager {
public:
    // 构造函数
    // num_blocks: 显存池中每层的 Block 数量
    // block_size: 每个 Block 包含的 Token 数量 (通常为 16 或 32)
    // num_heads: KV 头的数量 (Qwen3.5 使用 GQA，所以这里是 KV_heads)
    // head_dim: 每个头的维度
    // dtype: 数据类型 (如 FP16)
    // num_layers: full attention 层数 (每层独立的 KV cache)
    KVCacheManager(int num_blocks, int block_size, int num_heads, int head_dim, core::DataType dtype, std::shared_ptr<core::Allocator> allocator, int num_layers = 1);
    ~KVCacheManager();

    // 为一个新的请求分配初始的 Blocks
    // 返回分配的 Block 索引列表
    std::vector<int> allocate_blocks(int num_blocks_needed);

    // 释放一个请求占用的所有 Blocks
    void free_blocks(const std::vector<int>& block_indices);

    // 获取底层的物理 Tensor (用于传递给 CUDA Kernel)
    // 形状通常为: [num_layers * num_blocks, block_size, num_heads, head_dim]
    const core::Tensor& get_k_cache() const { return *k_cache_; }
    const core::Tensor& get_v_cache() const { return *v_cache_; }

    // 获取指定层的 KV cache 起始指针
    const __nv_bfloat16* get_layer_k_cache(int layer_idx) const;
    const __nv_bfloat16* get_layer_v_cache(int layer_idx) const;

    // 非 const 版本 (CacheEngine inject 时需要写入)
    __nv_bfloat16* get_layer_k_cache_mut(int layer_idx);
    __nv_bfloat16* get_layer_v_cache_mut(int layer_idx);

    int get_block_size() const { return block_size_; }
    int get_num_layers() const { return num_layers_; }
    int get_num_blocks_per_layer() const { return num_blocks_; }
    int num_free_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return (int)free_blocks_.size();
    }

private:
    int num_blocks_;
    int block_size_;
    int num_heads_;
    int head_dim_;
    int num_layers_;
    core::DataType dtype_;

    std::unique_ptr<core::Tensor> k_cache_;
    std::unique_ptr<core::Tensor> v_cache_;

    // 空闲 Block 列表
    std::vector<int> free_blocks_;
    mutable std::mutex mutex_;
};

// -----------------------------------------------------------------------------
// Chunked Prefill Paged Attention: Flash-Attention-style tiled GEMM
// For chunked prefill chunks 1+ where force_paged_attn=true and T_q > 1.
// Gathers K/V from paged cache per tile, uses GEMM score + online softmax merge.
// workspace: temp buffer (from up_out, ~60 MB for typical sizes)
// -----------------------------------------------------------------------------
void invoke_chunked_prefill_paged_attention(
    __nv_bfloat16* out,           // [T_q, q_dim]
    const __nv_bfloat16* q,       // [T_q, q_dim]
    const __nv_bfloat16* k_cache, // paged K cache for one layer
    const __nv_bfloat16* v_cache, // paged V cache for one layer
    const int* block_tables,      // [1, max_blocks_per_seq]
    int context_len,              // total context length (including current chunk)
    int T_q,                      // number of query tokens
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq,
    float sm_scale,
    __nv_bfloat16* workspace,     // temp workspace
    cudaStream_t stream = 0
);

// -----------------------------------------------------------------------------
// Prefill Attention: GEMM-based causal self-attention (替代 O(T²) paged attention)
// 使用 cuBLAS batched GEMM + causal softmax, 仅在 pure prefill (start_pos=0) 使用
// Q: [T, num_heads * head_dim], K/V: [T, num_kv_heads * head_dim], all row-major
// score_workspace: 至少 (num_heads/num_kv_heads) * T * T 个 BF16 元素
// -----------------------------------------------------------------------------
void invoke_prefill_attention(
    __nv_bfloat16* out,                // [T, num_heads * head_dim]
    const __nv_bfloat16* q,            // [T, num_heads * head_dim]
    const __nv_bfloat16* k,            // [T, num_kv_heads * head_dim]
    const __nv_bfloat16* v,            // [T, num_kv_heads * head_dim]
    int T,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    __nv_bfloat16* score_workspace,    // [heads_per_group * T * T]
    cudaStream_t stream = 0
);

// -----------------------------------------------------------------------------
// PagedAttention 算子接口
// -----------------------------------------------------------------------------
class PagedAttention {
public:
    PagedAttention() = default;
    virtual ~PagedAttention() = default;

    // 执行 PagedAttention 计算 (Decode 阶段)
    // 输入:
    //   query: [num_seqs, num_heads, head_dim] (当前步的 Query)
    //   k_cache: 物理 K Cache Tensor (来自 KVCacheManager)
    //   v_cache: 物理 V Cache Tensor (来自 KVCacheManager)
    //   block_tables: [num_seqs, max_num_blocks_per_seq] (每个序列的逻辑到物理 Block 映射)
    //   context_lens: [num_seqs] (每个序列当前的上下文长度)
    // 输出:
    //   output: [num_seqs, num_heads, head_dim]
    virtual void forward_decode(
        const core::Tensor& query,
        const core::Tensor& k_cache,
        const core::Tensor& v_cache,
        const core::Tensor& block_tables,
        const core::Tensor& context_lens,
        core::Tensor& output,
        void* stream = nullptr
    ) = 0;
};

} // namespace ops
} // namespace qwen_thor
