// KV Cache Offload — CacheEngine 实现 (SSD-only for Jetson AGX Thor)
//
// Thor 统一内存: CPU/GPU 共享 128 GB LPDDR5X
//   → "offload 到 CPU" 无意义 (本来就在同一块物理内存)
//   → 唯一有意义的 offload 目标: SSD
//
// 数据流:
//   Store:  paged KV (device) → extract kernel → flat buffer → fwrite → SSD
//   Retrieve: SSD → fread → flat buffer → inject kernel → paged KV (device)
//
// SSM/Conv 状态 (~147 MB) 使用 cudaMemcpy D2H/H2D, 在 Thor 上退化为 memcpy

#include "cache_engine.h"
#include "disk_backend.h"
#include "cache_kernels.h"
#include <iostream>
#include <chrono>
#include <cstring>

namespace qwen_thor {
namespace cache {

// 简洁计时工具
struct ScopedTimer {
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
CacheEngine::CacheEngine(const CacheConfig& config, const ModelCacheParams& params)
    : config_(config), params_(params), hasher_(config.chunk_size) {

    if (!config_.enabled) {
        std::cerr << "[CacheEngine] Prefix caching DISABLED" << std::endl;
        return;
    }

    // 创建 SSD 后端
    backend_ = std::make_unique<DiskBackend>(config_.cache_dir, config_.max_cache_bytes);

    // 设置驱逐回调 → 更新 Monitor
    backend_->set_eviction_callback([this](uint64_t evicted_bytes) {
        monitor_.record_eviction(evicted_bytes);
    });

    std::cerr << "[CacheEngine] Prefix caching ENABLED (SSD-only)" << std::endl;
    std::cerr << "  Cache Dir:     " << config_.cache_dir << std::endl;
    std::cerr << "  Max SSD:       " << (config_.max_cache_bytes / (1024ULL*1024*1024)) << " GB" << std::endl;
    std::cerr << "  Chunk Size:    " << config_.chunk_size << " tokens" << std::endl;
    std::cerr << "  SSM Caching:   " << (config_.cache_ssm_state ? "ON" : "OFF") << std::endl;
    std::cerr << "  KV/token:      " << params_.kv_bytes_per_token() << " bytes" << std::endl;
    std::cerr << "  SSM total:     " << (params_.ssm_bytes_total() / (1024*1024)) << " MB" << std::endl;
    std::cerr << "  Conv total:    " << (params_.conv_bytes_total() / 1024) << " KB" << std::endl;
}

CacheEngine::~CacheEngine() = default;

// ---------------------------------------------------------------------------
// lookup_prefix: 查询已缓存的前缀长度
// ---------------------------------------------------------------------------
LookupResult CacheEngine::lookup_prefix(const int* tokens, int num_tokens) {
    LookupResult result;
    if (!config_.enabled || !backend_) return result;

    {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.total_lookups++;
    }

    // 将 token 序列切成 chunk, 计算前缀哈希链
    auto keys = hasher_.compute_keys(tokens, num_tokens);
    if (keys.empty()) return result;

    // SSD 上匹配连续前缀 chunk 数
    int matched = backend_->prefix_match(keys);

    if (matched > 0) {
        result.matched_chunks = matched;
        result.matched_tokens = matched * config_.chunk_size;
        // 最后一个匹配的 chunk 是否包含 SSM 状态? (仅查内存索引, 不读 SSD)
        auto* disk = static_cast<DiskBackend*>(backend_.get());
        result.has_ssm_state = disk->has_ssm_state(keys[matched - 1]);

        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.cache_hits++;
    } else {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.cache_misses++;
    }

    return result;
}

// ---------------------------------------------------------------------------
// store_prefix: Prefill 完成后, 将 KV + SSM/Conv 缓存到 SSD
//
// 策略: 按 chunk 粒度存储, 每个 chunk 一个 CacheEntry
//   - 前 N-1 个 chunk: 只存 KV data (SSM 是中间状态, 恢复代价小)
//   - 最后一个 chunk: 存 KV + SSM + Conv (完整恢复点)
//
// 性能分析 (T=1024, chunk=256):
//   4 个 chunk, 每个 chunk KV = 256 * 64KB = 16 MB
//   最后一个还包含 SSM 144 MB + Conv 2.88 MB ≈ 163 MB
//   SSD 写入 ~227 MB, 以 2 GB/s 写入速度 → ~114 ms
//   相比 prefill 1024 token (~200ms), 异步写入不影响延迟
// ---------------------------------------------------------------------------
void CacheEngine::store_prefix(
    const int* tokens, int num_tokens,
    const ops::KVCacheManager& kv_manager,
    const int* d_block_table,
    int num_blocks,
    __nv_bfloat16** ssm_states,
    __nv_bfloat16** conv_states,
    __nv_bfloat16* workspace,
    cudaStream_t stream)
{
    if (!config_.enabled || !backend_) return;
    if (num_tokens < config_.chunk_size) return;  // 太短不缓存

    auto keys = hasher_.compute_keys(tokens, num_tokens);
    int num_chunks = num_tokens / config_.chunk_size;  // 只处理完整 chunk

    // 检查哪些 chunk 已经缓存
    int already_cached = backend_->prefix_match(keys);
    if (already_cached >= num_chunks) return;  // 全部已缓存

    // 从第 already_cached 个 chunk 开始存储
    ScopedTimer store_timer;
    uint64_t total_store_bytes = 0;
    int stored_chunks = 0;

    for (int ci = already_cached; ci < num_chunks; ci++) {
        int chunk_start = ci * config_.chunk_size;
        int chunk_end   = chunk_start + config_.chunk_size;

        auto entry = std::make_shared<CacheEntry>();
        entry->num_tokens = config_.chunk_size;
        entry->tokens.assign(tokens + chunk_start, tokens + chunk_end);

        // ---- 提取该 chunk 的 KV 数据 ----
        size_t kv_bytes = (size_t)config_.chunk_size * params_.kv_bytes_per_token();
        entry->kv_data.resize(kv_bytes);

        // 使用 CUDA kernel: paged KV → flat buffer
        // block_table 中相关的 block 范围: [chunk_start/block_size, chunk_end/block_size)
        int block_start = chunk_start / params_.block_size;
        int block_count = config_.chunk_size / params_.block_size;

        // 临时 block_table 只包含该 chunk 的 block
        // 注: d_block_table 是整个序列的, 我们需要偏移
        invoke_extract_kv_from_pages(
            reinterpret_cast<__nv_bfloat16*>(workspace),
            kv_manager.get_layer_k_cache(0),  // 整个 K cache pool
            kv_manager.get_layer_v_cache(0),   // 整个 V cache pool
            d_block_table + block_start,        // 偏移到该 chunk 的 block
            config_.chunk_size,
            params_.num_kv_heads,
            params_.head_dim,
            params_.block_size,
            kv_manager.get_num_blocks_per_layer(),
            params_.num_full_attn_layers,
            stream);

        cudaStreamSynchronize(stream);

        // 从 workspace (cudaMalloc, GPU-only) 复制到 entry (host)
        cudaMemcpy(entry->kv_data.data(), workspace, kv_bytes, cudaMemcpyDeviceToHost);

        // ---- SSM/Conv 状态 (仅最后一个 chunk) ----
        bool is_last = (ci == num_chunks - 1);
        if (is_last && config_.cache_ssm_state && ssm_states && conv_states) {
            copy_state_to_host(*entry, ssm_states, conv_states, stream);
        }

        // ---- 存入 SSD ----
        if (backend_->put(keys[ci], entry)) {
            std::lock_guard<std::mutex> lk(stats_mutex_);
            stats_.store_count++;
            stats_.store_bytes += entry->total_bytes();
            total_store_bytes += entry->total_bytes();
            stored_chunks++;
        }
    }

    // 更新 Monitor 指标
    if (stored_chunks > 0) {
        double elapsed = store_timer.elapsed_ms();
        monitor_.record_store(total_store_bytes, elapsed,
                              stored_chunks,
                              stored_chunks * config_.chunk_size);
        update_capacity_snapshot();
    }
}

// ---------------------------------------------------------------------------
// retrieve_prefix: 从 SSD 恢复 KV + SSM/Conv 到新 paged blocks
//
// 返回恢复的 token 数 (0 = miss, 调用方应 fallback to full prefill)
//
// 工作流程:
//   1. lookup_prefix 确认命中
//   2. 为匹配的 token 分配 KV blocks (block_table)
//   3. 逐 chunk 从 SSD 读取 → inject kernel → 写入 paged KV
//   4. 恢复最后一个 chunk 的 SSM/Conv 状态
//   5. 调用方从恢复的位置继续 prefill 剩余 token
// ---------------------------------------------------------------------------
int CacheEngine::retrieve_prefix(
    const int* tokens, int num_tokens,
    ops::KVCacheManager& kv_manager,
    std::vector<int>& out_block_table,
    __nv_bfloat16** ssm_states,
    __nv_bfloat16** conv_states,
    __nv_bfloat16* workspace,
    int* d_block_table_buf,
    cudaStream_t stream)
{
    if (!config_.enabled || !backend_) return 0;

    auto keys = hasher_.compute_keys(tokens, num_tokens);
    int matched = backend_->prefix_match(keys);
    if (matched <= 0) return 0;

    // P0 安全检查: 如果启用了 SSM 缓存, 最后匹配 chunk 必须包含 SSM 状态
    // 否则 48 层 DeltaNet 将从零 SSM 状态开始, 产生错误输出
    if (config_.cache_ssm_state && ssm_states) {
        auto* disk = static_cast<DiskBackend*>(backend_.get());
        if (!disk->has_ssm_state(keys[matched - 1])) {
            std::cerr << "[CacheEngine] Last matched chunk lacks SSM state, "
                      << "falling back to full prefill" << std::endl;
            return 0;
        }
    }

    ScopedTimer retrieve_timer;

    int restore_tokens = matched * config_.chunk_size;
    int restore_blocks = restore_tokens / params_.block_size;

    // 1. 分配 KV blocks (可能因 GPU 容量不足而失败, 特别是 SSD 模式下)
    std::vector<int> blocks;
    try {
        blocks = kv_manager.allocate_blocks(restore_blocks);
    } catch (const std::runtime_error& e) {
        // GPU KV 容量不足以恢复全部 prefix — 跳过 cache 恢复, 走正常 prefill
        std::cerr << "[CacheEngine] Cannot allocate " << restore_blocks
                  << " blocks for prefix restore (free=" << kv_manager.num_free_blocks()
                  << "), falling back to full prefill" << std::endl;
        return 0;
    }
    if ((int)blocks.size() < restore_blocks) {
        std::cerr << "[CacheEngine] Not enough KV blocks for restore: need "
                  << restore_blocks << ", got " << blocks.size() << std::endl;
        kv_manager.free_blocks(blocks);
        return 0;
    }
    out_block_table = blocks;

    // 2. 上传 block_table 到 device (使用调用方预分配的 d_block_table_buf)
    cudaMemcpyAsync(d_block_table_buf, blocks.data(), restore_blocks * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // 3. 逐 chunk 恢复 KV
    for (int ci = 0; ci < matched; ci++) {
        auto entry = backend_->get(keys[ci]);
        if (!entry) {
            std::cerr << "[CacheEngine] Chunk " << ci << " disappeared from SSD!" << std::endl;
            kv_manager.free_blocks(blocks);
            out_block_table.clear();
            return 0;
        }

        int chunk_start = ci * config_.chunk_size;
        int block_start = chunk_start / params_.block_size;
        size_t kv_bytes = entry->kv_data.size();

        // 复制 KV 数据到 workspace (cudaMalloc, GPU-only)
        cudaMemcpy(workspace, entry->kv_data.data(), kv_bytes, cudaMemcpyHostToDevice);

        // inject kernel: flat buffer → paged KV
        invoke_inject_kv_to_pages(
            kv_manager.get_layer_k_cache_mut(0),
            kv_manager.get_layer_v_cache_mut(0),
            reinterpret_cast<const __nv_bfloat16*>(workspace),
            d_block_table_buf + block_start,
            config_.chunk_size,
            params_.num_kv_heads,
            params_.head_dim,
            params_.block_size,
            kv_manager.get_num_blocks_per_layer(),
            params_.num_full_attn_layers,
            stream);

        // 最后一个 chunk: 恢复 SSM/Conv 状态
        if (ci == matched - 1 && entry->has_ssm_state() && ssm_states && conv_states) {
            copy_state_from_host(*entry, ssm_states, conv_states, stream);
        }
    }

    cudaStreamSynchronize(stream);

    {
        std::lock_guard<std::mutex> lk(stats_mutex_);
        stats_.retrieve_count++;
        stats_.retrieve_bytes += (size_t)restore_tokens * params_.kv_bytes_per_token();
        stats_.total_tokens_saved += restore_tokens;
    }

    // 更新 Monitor 指标
    {
        uint64_t bytes = (size_t)restore_tokens * params_.kv_bytes_per_token();
        double elapsed = retrieve_timer.elapsed_ms();
        monitor_.record_retrieve(bytes, elapsed);
        update_capacity_snapshot();
    }

    return restore_tokens;
}

// ---------------------------------------------------------------------------
// SSM/Conv 状态拷贝
// ---------------------------------------------------------------------------
void CacheEngine::copy_state_to_host(CacheEntry& entry,
                                     __nv_bfloat16** ssm_states,
                                     __nv_bfloat16** conv_states,
                                     cudaStream_t stream) {
    int num_lin = params_.num_linear_attn_layers;
    size_t ssm_per = params_.ssm_bytes_per_layer();
    size_t conv_per = params_.conv_bytes_per_layer();

    entry.ssm_data.resize(num_lin * ssm_per);
    entry.conv_data.resize(num_lin * conv_per);

    for (int i = 0; i < num_lin; i++) {
        cudaMemcpyAsync(entry.ssm_data.data() + i * ssm_per,
                        ssm_states[i], ssm_per,
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(entry.conv_data.data() + i * conv_per,
                        conv_states[i], conv_per,
                        cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
}

void CacheEngine::copy_state_from_host(const CacheEntry& entry,
                                       __nv_bfloat16** ssm_states,
                                       __nv_bfloat16** conv_states,
                                       cudaStream_t stream) {
    int num_lin = params_.num_linear_attn_layers;
    size_t ssm_per = params_.ssm_bytes_per_layer();
    size_t conv_per = params_.conv_bytes_per_layer();

    if (entry.ssm_data.size() < num_lin * ssm_per) return;
    if (entry.conv_data.size() < num_lin * conv_per) return;

    for (int i = 0; i < num_lin; i++) {
        cudaMemcpyAsync(ssm_states[i],
                        entry.ssm_data.data() + i * ssm_per, ssm_per,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(conv_states[i],
                        entry.conv_data.data() + i * conv_per, conv_per,
                        cudaMemcpyHostToDevice, stream);
    }
    cudaStreamSynchronize(stream);
}

// ---------------------------------------------------------------------------
// record_request: 由 Engine 在 prefill 完成后调用, 记录请求级指标
// ---------------------------------------------------------------------------
void CacheEngine::record_request(int prompt_tokens, int restored_tokens, int computed_tokens) {
    monitor_.record_request(prompt_tokens, restored_tokens, computed_tokens);
}

// ---------------------------------------------------------------------------
// update_capacity_snapshot: 更新 Monitor 中的容量指标
// ---------------------------------------------------------------------------
void CacheEngine::update_capacity_snapshot() {
    if (!backend_) return;
    monitor_.update_capacity(
        backend_->current_size_bytes(),
        backend_->max_size_bytes(),
        backend_->num_entries(),
        0, 0  // KV block 使用情况由 Engine 层级更新
    );
}

// ---------------------------------------------------------------------------
// 统计
// ---------------------------------------------------------------------------
CacheStats CacheEngine::get_stats() const {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    return stats_;
}

void CacheEngine::print_stats() const {
    // 旧格式 (兼容)
    auto s = get_stats();
    std::cerr << "=== KV Cache Stats ===" << std::endl;
    std::cerr << "  Lookups:       " << s.total_lookups << std::endl;
    std::cerr << "  Hits:          " << s.cache_hits
              << " (" << (s.total_lookups > 0 ? 100.0 * s.cache_hits / s.total_lookups : 0) << "%)" << std::endl;
    std::cerr << "  Misses:        " << s.cache_misses << std::endl;
    std::cerr << "  Tokens Saved:  " << s.total_tokens_saved << std::endl;
    std::cerr << "  Store Count:   " << s.store_count << std::endl;
    std::cerr << "  Store Bytes:   " << (s.store_bytes / (1024*1024)) << " MB" << std::endl;
    std::cerr << "  Retrieve:      " << s.retrieve_count << std::endl;
    std::cerr << "  Retrieve Bytes:" << (s.retrieve_bytes / (1024*1024)) << " MB" << std::endl;
    if (backend_) {
        std::cerr << "  SSD Used:      " << (backend_->current_size_bytes() / (1024*1024)) << " MB"
                  << " / " << (backend_->max_size_bytes() / (1024ULL*1024*1024)) << " GB" << std::endl;
        std::cerr << "  SSD Entries:   " << backend_->num_entries() << std::endl;
    }
    // LMCache-style 全面报告
    monitor_.print_report();
}

void CacheEngine::reset_stats() {
    std::lock_guard<std::mutex> lk(stats_mutex_);
    stats_ = {};
}

} // namespace cache
} // namespace qwen_thor
