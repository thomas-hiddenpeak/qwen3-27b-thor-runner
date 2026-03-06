// cache_manager.h — 统一缓存管理器
//
// 整合 KVCacheManager / BlockTracker / BlockSSDStore / CacheEngine / KVSwapper
// 为 InferenceEngine 提供单一缓存操作接口。
//
// 设计原则 (参考 vLLM KVCacheCoordinator + LMCache HybridBackend + SGLang HybridPool):
//   1. Engine 只通过 CacheManager 操作缓存 — 不直接接触内部组件
//   2. GPU Block Pool + SSD Block Store 作为内部 tier 实现
//   3. SSM/Conv Pool 统一管理，与 KV 生命周期绑定
//   4. RequestCacheState 提供统一的 context_len / block 位置查询
//   5. streaming attention 上下文组装由 CacheManager 负责
#pragma once

#include "paged_attention.h"
#include "cache_config.h"
#include "cache_engine.h"
#include "kv_swapper.h"
#include "block_tracker.h"
#include "streaming_attention.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace core {
    struct Qwen35Config;
    struct RequestContext;
}  // forward declarations
namespace cache {

// SSM/Conv 状态池最大槽位数
static constexpr int MAX_CACHE_SSM_SLOTS = 8;

// -----------------------------------------------------------------------
// RequestCacheState: 单请求的缓存状态 (替代 RequestContext 中散布的多个字段)
// -----------------------------------------------------------------------
struct RequestCacheState {
    // Block 位置追踪 (统一来源)
    BlockTracker block_tracker;      // 完整的 logical → GPU/SSD 映射

    // 物化 block table: physical block IDs, SSD 已驱逐的为 -1
    // 与 block_tracker 保持同步, 用于高频 GPU 上传
    std::vector<int> block_table;

    // 快速查询字段 (由 CacheManager 更新)
    int context_len = 0;             // 总 token 数 (GPU + SSD)
    int ssm_slot = -1;               // SSM/Conv pool slot (-1 = 未分配)
    bool is_swapped = false;         // 整请求换出到 SSD

    // SSM/Conv 状态指针 (指向池化 buffer, 非 owned)
    std::vector<__nv_bfloat16*> ssm_states;   // [num_linear_layers]
    std::vector<__nv_bfloat16*> conv_states;  // [num_linear_layers]

    // ---- 便捷访问 ----

    // 获取只含 GPU-resident blocks 的 block table (去掉 -1)
    std::vector<int> gpu_block_table() const {
        return block_tracker.get_gpu_block_table();
    }

    // SSD 上的 logical block indices
    std::vector<int> ssd_logical_indices() const {
        return block_tracker.get_ssd_logical_indices();
    }

    bool has_ssd_blocks() const {
        return block_tracker.has_ssd_blocks();
    }

    int num_gpu_blocks() const { return block_tracker.num_gpu_blocks(); }
    int num_ssd_blocks() const { return block_tracker.num_ssd_blocks(); }
    int num_total_blocks() const { return block_tracker.num_blocks(); }
};

// -----------------------------------------------------------------------
// CacheManager: 统一缓存管理入口
// -----------------------------------------------------------------------
class CacheManager {
public:
    // Config 从 CacheConfig + model params 构建
    CacheManager(const core::Qwen35Config& model_config,
                 const CacheConfig& cache_config,
                 const ModelCacheParams& model_cache_params,
                 const CapacityReport& capacity,
                 cudaStream_t stream,
                 bool verbose = true);
    ~CacheManager();

    // =============== 请求生命周期 ===============

    // 为新请求分配 SSM/Conv slot + 初始 KV blocks
    // 返回 false 如果分配失败
    bool allocate_request(uint64_t req_id, int initial_tokens,
                          RequestCacheState& state);

    // 扩展 KV blocks (分配新 blocks 以容纳更多 tokens)
    // 返回 true 成功
    bool extend_blocks(uint64_t req_id, int total_tokens_needed,
                       RequestCacheState& state);

    // 释放请求的所有资源 (GPU blocks, SSD 文件, SSM slot)
    void free_request(uint64_t req_id, RequestCacheState& state);

    // =============== GPU → SSD Eviction (Block 级) ===============

    // 驱逐请求的最旧 blocks 到 SSD，释放 GPU blocks
    // 返回实际释放的 block 数
    int evict_blocks_to_ssd(uint64_t req_id, int num_blocks_to_free,
                            RequestCacheState& state);

    // =============== 请求级 Swap ===============

    // 将请求的全部 KV + SSM/Conv 换出到 SSD
    bool swap_out_request(uint64_t req_id, RequestCacheState& state);

    // 从 SSD 换入请求
    bool swap_in_request(uint64_t req_id, RequestCacheState& state);

    // 查询换出状态
    int get_swapped_context_len(uint64_t req_id) const;

    // =============== Prefix Cache ===============

    // 查询 prefix cache 命中的 token 数
    int lookup_prefix(const int* tokens, int num_tokens);

    // 从 prefix cache 恢复 KV + SSM/Conv
    // 返回恢复的 token 数 (0 = miss)
    int restore_prefix(const int* tokens, int num_tokens,
                       RequestCacheState& state,
                       __nv_bfloat16* workspace,
                       int* d_block_table_buf);

    // 存储 prefix cache
    void store_prefix(const int* tokens, int num_tokens,
                      const RequestCacheState& state,
                      __nv_bfloat16* workspace);

    // 记录统计
    void record_prefix_stats(int prompt_tokens, int restored, int computed);

    // =============== Streaming Attention 支持 ===============

    // 构建 StreamingAttnCtx 用于有 SSD blocks 的 forward
    // gpu_ctx_len: GPU tokens + current chunk tokens (用于 causal masking)
    // 返回构建好的上下文
    ops::StreamingAttnCtx build_streaming_ctx(
        uint64_t req_id,
        const RequestCacheState& state,
        int gpu_ctx_len);

    // 从 SSD 加载 blocks 到 staging (streaming forward callback)
    int load_ssd_blocks_for_layer(uint64_t req_id, int layer_idx,
                                  const std::vector<int>& ssd_indices,
                                  int batch_start, int batch_count);

    // =============== SSM/Conv Pool ===============

    // 分配 SSM/Conv slot (已在 allocate_request 中调用, 外部一般不需要)
    int allocate_ssm_slot();
    void free_ssm_slot(int slot);

    // 获取 SSM/Conv 状态指针
    __nv_bfloat16* get_ssm_state(int slot, int layer) const;
    __nv_bfloat16* get_conv_state(int slot, int layer) const;

    // 池化的所有 SSM/Conv 指针 (用于 forward 需要)
    const std::vector<std::vector<__nv_bfloat16*>>& pooled_ssm_states() const { return pooled_ssm_states_; }
    const std::vector<std::vector<__nv_bfloat16*>>& pooled_conv_states() const { return pooled_conv_states_; }

    // =============== 查询 ===============

    int num_free_gpu_blocks() const;
    int total_gpu_blocks() const { return total_gpu_blocks_; }
    int gpu_max_tokens() const { return gpu_max_tokens_; }
    int block_size() const { return block_size_; }

    // KV cache 物理指针 (供 model forward 使用)
    ops::KVCacheManager& kv_manager() { return *kv_manager_; }
    const ops::KVCacheManager& kv_manager() const { return *kv_manager_; }

    // Prefix cache 引擎 (统计接口)
    CacheEngine* prefix_cache() { return cache_engine_.get(); }

    // Swap 统计
    void print_swap_stats() const;
    void drain_swapper();

    // Staging buffer 容量
    int staging_num_blocks() const { return staging_num_blocks_; }

    // =============== Phase 1 直接访问 (engine.cpp 迁移用) ===============

    bool has_ssd_store() const { return block_ssd_store_ != nullptr; }
    bool has_swapper() const { return kv_swapper_ != nullptr; }
    int num_free_ssm_slots() const { return (int)free_ssm_slots_.size(); }

    // 直接组件访问 (后续 Phase 2 替换为高层 API)
    BlockSSDStore* ssd_store() { return block_ssd_store_.get(); }
    KVSwapper* swapper() { return kv_swapper_.get(); }

    // Streaming attention buffer 访问
    __nv_bfloat16* staging_k() { return d_staging_k_; }
    __nv_bfloat16* staging_v() { return d_staging_v_; }
    __nv_bfloat16* partial_out() { return d_partial_out_; }
    __nv_bfloat16* partial_out2() { return d_partial_out2_; }
    float* partial_m() { return d_partial_m_; }
    float* partial_l() { return d_partial_l_; }
    float* partial_m2() { return d_partial_m2_; }
    float* partial_l2() { return d_partial_l2_; }
    int* ssd_block_tables() { return d_ssd_block_tables_; }
    int* ssd_context_lens() { return d_ssd_context_lens_; }

    // =============== Phase 2: RequestContext 桥接 API ===============
    // Engine 使用 RequestContext (engine.h), CacheManager 内部用 RequestCacheState.
    // 下列方法直接操作 RequestContext 的字段, 避免 engine.cpp 手动管理.

    // 释放请求所有缓存资源 (GPU blocks + SSD + swap + SSM slot)
    // 不处理: MTP blocks, images, prompt/generated tokens (由 engine 管理)
    void free_request(uint64_t req_id, core::RequestContext* ctx);

    // 换出请求 (KV + SSM/Conv → SSD, 释放 GPU blocks + SSM slot)
    bool swap_out_request(uint64_t req_id, core::RequestContext* ctx);

    // 换入请求 (SSD → GPU, 分配 SSM slot + KV blocks)
    bool swap_in_request(uint64_t req_id, core::RequestContext* ctx);

    // Prefix cache: 是否启用
    bool is_prefix_cache_enabled() const;

    // Prefix cache: 恢复 — 返回恢复的 token 数 (0 = miss)
    int restore_prefix(const int* tokens, int num_tokens,
                       core::RequestContext* ctx,
                       __nv_bfloat16* workspace, int* d_block_table_buf);

    // Prefix cache: 存储
    void store_prefix(const int* tokens, int num_tokens,
                      const core::RequestContext* ctx,
                      __nv_bfloat16* workspace);

    // Streaming attention context (从 RequestContext 构建)
    ops::StreamingAttnCtx build_streaming_ctx(
        uint64_t req_id,
        const core::RequestContext* ctx,
        int gpu_ctx_len);

private:
    // ---- 内部组件 ----
    std::unique_ptr<ops::KVCacheManager> kv_manager_;   // GPU KV block pool
    std::unique_ptr<CacheEngine>         cache_engine_;  // Prefix cache
    std::unique_ptr<KVSwapper>           kv_swapper_;    // 请求级 swap
    std::unique_ptr<BlockSSDStore>       block_ssd_store_; // Block 级 SSD I/O

    // ---- SSM/Conv Pool ----
    __nv_bfloat16* ssm_pool_base_ = nullptr;
    __nv_bfloat16* conv_pool_base_ = nullptr;
    std::vector<std::vector<__nv_bfloat16*>> pooled_ssm_states_;  // [slot][layer]
    std::vector<std::vector<__nv_bfloat16*>> pooled_conv_states_; // [slot][layer]
    std::vector<int> free_ssm_slots_;  // LIFO stack

    // ---- Streaming Attention Buffers ----
    __nv_bfloat16* d_staging_k_ = nullptr;
    __nv_bfloat16* d_staging_v_ = nullptr;
    __nv_bfloat16* d_partial_out_ = nullptr;
    float* d_partial_m_ = nullptr;
    float* d_partial_l_ = nullptr;
    __nv_bfloat16* d_partial_out2_ = nullptr;
    float* d_partial_m2_ = nullptr;
    float* d_partial_l2_ = nullptr;
    int* d_ssd_block_tables_ = nullptr;
    int* d_ssd_context_lens_ = nullptr;
    int staging_num_blocks_ = 0;
    void* ssd_io_staging_ = nullptr;      // Host staging (4K aligned)
    size_t ssd_io_staging_size_ = 0;

    // ---- Model Params ----
    int num_layers_ = 0;             // full_attn layers (16)
    int num_linear_layers_ = 0;      // linear_attn layers (48)
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int block_size_ = 16;
    int total_gpu_blocks_ = 0;
    int gpu_max_tokens_ = 0;
    int max_chunk_size_ = 256;       // SMMU 硬约束
    size_t ssm_size_per_layer_ = 0;
    size_t conv_size_per_layer_ = 0;

    // ---- CUDA ----
    cudaStream_t stream_;
    bool verbose_ = true;
};

} // namespace cache
} // namespace qwen_thor
