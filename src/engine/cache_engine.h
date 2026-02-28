// KV Cache Offload — CacheEngine (SSD Prefix Caching)
//
// 简化架构 (Thor 统一内存 — 仅 SSD):
//   CacheEngine → DiskBackend (SSD) → Paged KV Cache (统一内存)
//
// 数据流:
//   Store:  KV (统一内存) → [CUDA kernel: extract] → host buffer → fwrite → SSD
//   Retrieve: SSD → fread → host buffer → [CUDA kernel: inject] → KV (统一内存)
//
// 注: 在 Thor 上 "host buffer" 和 "device buffer" 是同一物理内存,
//     cudaMemcpy D2H / H2D 退化为 memcpy, 但 CUDA kernel scatter/gather
//     仍然需要处理 paged 布局 ↔ flat 布局的转换.
#pragma once

#include "cache_config.h"
#include "cache_key.h"
#include "cache_entry.h"
#include "cache_monitor.h"
#include "storage_backend.h"
#include "paged_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <memory>
#include <vector>
#include <mutex>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// LookupResult: 前缀匹配的结果
// ---------------------------------------------------------------------------
struct LookupResult {
    int matched_chunks  = 0;
    int matched_tokens  = 0;
    bool has_ssm_state  = false;
};

// ---------------------------------------------------------------------------
// CacheStats: 缓存统计信息 (兼容旧接口)
// ---------------------------------------------------------------------------
struct CacheStats {
    uint64_t total_lookups      = 0;
    uint64_t cache_hits         = 0;
    uint64_t cache_misses       = 0;
    uint64_t total_tokens_saved = 0;
    uint64_t store_count        = 0;
    uint64_t store_bytes        = 0;
    uint64_t retrieve_count     = 0;
    uint64_t retrieve_bytes     = 0;
};

// ---------------------------------------------------------------------------
// CacheEngine: SSD Prefix Caching 主引擎
// ---------------------------------------------------------------------------
class CacheEngine {
public:
    CacheEngine(const CacheConfig& config, const ModelCacheParams& params);
    ~CacheEngine();

    // 是否启用
    bool is_enabled() const { return config_.enabled; }

    // ---- 查询: 检查已缓存的前缀 token 数 ----
    LookupResult lookup_prefix(const int* tokens, int num_tokens);

    // ---- 存储: prefill 完成后缓存 KV + SSM/Conv ----
    void store_prefix(
        const int* tokens, int num_tokens,
        const ops::KVCacheManager& kv_manager,
        const int* d_block_table,
        int num_blocks,
        float** ssm_states,
        __nv_bfloat16** conv_states,
        __nv_bfloat16* workspace,
        cudaStream_t stream);

    // ---- 恢复: 从 SSD 恢复 KV + SSM/Conv 到新分配的 paged blocks ----
    // 返回恢复的 token 数 (0 = miss)
    int retrieve_prefix(
        const int* tokens, int num_tokens,
        ops::KVCacheManager& kv_manager,
        std::vector<int>& out_block_table,
        float** ssm_states,
        __nv_bfloat16** conv_states,
        __nv_bfloat16* workspace,
        cudaStream_t stream);

    // ---- 统计 (旧接口, 保持兼容) ----
    CacheStats get_stats() const;
    void print_stats() const;
    void reset_stats();

    // ---- LMCache 监控 ----
    CacheMonitor& monitor() { return monitor_; }
    const CacheMonitor& monitor() const { return monitor_; }

    // 记录一次完整请求 (由 Engine 在 prefill 完成后调用)
    void record_request(int prompt_tokens, int restored_tokens, int computed_tokens);

    const CacheConfig& config() const { return config_; }
    const ModelCacheParams& model_params() const { return params_; }

private:
    void copy_state_to_host(CacheEntry& entry,
                            float** ssm_states, __nv_bfloat16** conv_states,
                            cudaStream_t stream);
    void copy_state_from_host(const CacheEntry& entry,
                              float** ssm_states, __nv_bfloat16** conv_states,
                              cudaStream_t stream);

    // 更新容量指标快照
    void update_capacity_snapshot();

    CacheConfig config_;
    ModelCacheParams params_;
    TokenHasher hasher_;

    std::unique_ptr<StorageBackend> backend_;   // SSD (DiskBackend)

    CacheStats stats_;
    mutable std::mutex stats_mutex_;

    CacheMonitor monitor_;
};

} // namespace cache
} // namespace qwen_thor
