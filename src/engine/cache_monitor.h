// KV Cache Offload — LMCache 监控指标
//
// 对标 LMCache 的 monitoring dashboard, 提供全面的缓存性能指标:
//
// === 前缀缓存加速 (Prefix Caching Acceleration) ===
//   - cache.hit_rate          命中率 (%)
//   - cache.token_save_ratio  Token 节省比 (恢复/总计)
//   - cache.ttft_saved_ms     TTFT 累计节省 (ms)
//   - cache.prefill_skipped   跳过的 prefill token 计数
//
// === 上下文无限扩展 (Context Extension) ===
//   - cache.effective_context_tokens  有效上下文 (内存 + SSD)
//   - cache.kv_memory_utilization     KV 内存利用率 (%)
//   - cache.ssd_utilization           SSD 利用率 (%)
//   - cache.offloaded_tokens          已 offload 到 SSD 的 token 总数
//
// === SSD I/O 性能 ===
//   - cache.ssd.write_bandwidth_gbps  SSD 写入带宽 (GB/s)
//   - cache.ssd.read_bandwidth_gbps   SSD 读取带宽 (GB/s)
//   - cache.store_latency_ms          Store 操作平均延迟
//   - cache.retrieve_latency_ms       Retrieve 操作平均延迟
//
// === 驱逐与容量 ===
//   - cache.eviction_count    驱逐次数
//   - cache.entries           当前条目数
//   - cache.ssd_used_mb       SSD 已用 (MB)
//   - cache.ssd_capacity_gb   SSD 配额 (GB)
//
#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// LatencyTracker: 延迟统计 (min/max/avg/p50/p99)
// ---------------------------------------------------------------------------
class LatencyTracker {
public:
    void record(double ms) {
        std::lock_guard<std::mutex> lk(mu_);
        samples_.push_back(ms);
        total_ms_ += ms;
        count_++;
        if (ms < min_ms_) min_ms_ = ms;
        if (ms > max_ms_) max_ms_ = ms;
    }

    double avg_ms() const { return count_ > 0 ? total_ms_ / count_ : 0; }
    double min_ms() const { return count_ > 0 ? min_ms_ : 0; }
    double max_ms() const { return max_ms_; }
    uint64_t count() const { return count_; }
    double total_ms() const { return total_ms_; }

    double percentile(double p) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (samples_.empty()) return 0;
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        int idx = (int)(sorted.size() * p);
        if (idx >= (int)sorted.size()) idx = (int)sorted.size() - 1;
        return sorted[idx];
    }
    double p50() const { return percentile(0.50); }
    double p95() const { return percentile(0.95); }
    double p99() const { return percentile(0.99); }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        samples_.clear();
        total_ms_ = 0; count_ = 0;
        min_ms_ = 1e18; max_ms_ = 0;
    }

private:
    mutable std::mutex mu_;
    std::vector<double> samples_;
    double total_ms_ = 0;
    uint64_t count_ = 0;
    double min_ms_ = 1e18;
    double max_ms_ = 0;
};

// ---------------------------------------------------------------------------
// CacheMetrics: LMCache 风格的全面监控指标
// ---------------------------------------------------------------------------
struct CacheMetrics {
    // ---- 前缀缓存加速 ----
    uint64_t total_requests        = 0;  // 总请求数
    uint64_t cache_hit_requests    = 0;  // 命中的请求数 (至少 1 chunk 命中)
    uint64_t full_hit_requests     = 0;  // 完全命中 (所有 chunk 命中, 跳过 prefill)
    uint64_t partial_hit_requests  = 0;  // 部分命中 (部分 chunk 命中)
    uint64_t cache_miss_requests   = 0;  // 完全未命中

    uint64_t total_prompt_tokens   = 0;  // 总 prompt token 数
    uint64_t tokens_restored       = 0;  // 从 SSD 恢复的 token 数 (节省的 prefill)
    uint64_t tokens_computed       = 0;  // 实际 prefill 计算的 token 数

    double   ttft_saved_ms         = 0;  // 估算的 TTFT 累计节省 (ms)
    double   compute_flops_saved   = 0;  // 避免的计算量 (GFLOPS)

    // ---- 上下文扩展 ----
    uint64_t max_context_seen      = 0;  // 处理过的最大上下文长度
    uint64_t offloaded_chunks      = 0;  // 已 offload 到 SSD 的 chunk 总数
    uint64_t offloaded_tokens      = 0;  // 对应的 token 总数
    uint64_t kv_blocks_in_use      = 0;  // 当前 KV 内存中的 block 数 (snapshot)
    uint64_t kv_blocks_total       = 0;  // KV block 池总数

    // ---- SSD I/O ----
    uint64_t ssd_write_bytes       = 0;  // SSD 累计写入字节
    uint64_t ssd_read_bytes        = 0;  // SSD 累计读取字节
    double   ssd_write_time_ms     = 0;  // SSD 累计写入耗时
    double   ssd_read_time_ms      = 0;  // SSD 累计读取耗时

    uint64_t store_ops             = 0;  // Store 操作次数
    uint64_t retrieve_ops          = 0;  // Retrieve 操作次数

    // ---- 驱逐 ----
    uint64_t eviction_count        = 0;  // 驱逐次数
    uint64_t eviction_bytes        = 0;  // 驱逐数据量

    // ---- 容量 ----
    uint64_t ssd_used_bytes        = 0;  // SSD 当前使用
    uint64_t ssd_capacity_bytes    = 0;  // SSD 配额
    uint64_t num_entries           = 0;  // 当前缓存条目数

    // ---- 派生指标 ----
    double hit_rate() const {
        return total_requests > 0 ? 100.0 * cache_hit_requests / total_requests : 0;
    }
    double full_hit_rate() const {
        return total_requests > 0 ? 100.0 * full_hit_requests / total_requests : 0;
    }
    double token_save_ratio() const {
        return total_prompt_tokens > 0 ? 100.0 * tokens_restored / total_prompt_tokens : 0;
    }
    double ssd_utilization() const {
        return ssd_capacity_bytes > 0 ? 100.0 * ssd_used_bytes / ssd_capacity_bytes : 0;
    }
    double kv_memory_utilization() const {
        return kv_blocks_total > 0 ? 100.0 * kv_blocks_in_use / kv_blocks_total : 0;
    }
    double ssd_write_bandwidth_gbps() const {
        return ssd_write_time_ms > 0 ? ssd_write_bytes / ssd_write_time_ms / 1e6 : 0;
    }
    double ssd_read_bandwidth_gbps() const {
        return ssd_read_time_ms > 0 ? ssd_read_bytes / ssd_read_time_ms / 1e6 : 0;
    }
    uint64_t effective_context_tokens() const {
        return (kv_blocks_in_use * 16) + offloaded_tokens;
    }
};

// ---------------------------------------------------------------------------
// CacheMonitor: LMCache 监控引擎
//
// 使用方式:
//   1. CacheEngine 内部调用 record_* 方法更新指标
//   2. 外部通过 snapshot() 获取 CacheMetrics 快照
//   3. print_report() 输出人类可读报告
//   4. to_json() 输出 JSON 格式 (for dashboards)
// ---------------------------------------------------------------------------
class CacheMonitor {
public:
    CacheMonitor() { start_time_ = std::chrono::steady_clock::now(); }

    // ---- Record events (called by CacheEngine) ----

    void record_request(int prompt_tokens, int restored_tokens, int computed_tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_.total_requests++;
        metrics_.total_prompt_tokens += prompt_tokens;
        metrics_.tokens_restored += restored_tokens;
        metrics_.tokens_computed += computed_tokens;

        if (restored_tokens >= prompt_tokens) {
            metrics_.full_hit_requests++;
            metrics_.cache_hit_requests++;
        } else if (restored_tokens > 0) {
            metrics_.partial_hit_requests++;
            metrics_.cache_hit_requests++;
        } else {
            metrics_.cache_miss_requests++;
        }

        // 估算 TTFT 节省: ~0.2 ms/token (Qwen3.5-27B prefill on Thor)
        metrics_.ttft_saved_ms += restored_tokens * 0.2;

        // 估算 FLOPS 节省: ~2 * 2 * num_params_per_layer * tokens
        // 简化: ~100 GFLOPS per token (粗估)
        metrics_.compute_flops_saved += restored_tokens * 0.1;  // TFLOPS

        if ((uint64_t)prompt_tokens > metrics_.max_context_seen) {
            metrics_.max_context_seen = prompt_tokens;
        }
    }

    void record_store(uint64_t bytes, double elapsed_ms, int chunks, int tokens) {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_.store_ops++;
        metrics_.ssd_write_bytes += bytes;
        metrics_.ssd_write_time_ms += elapsed_ms;
        metrics_.offloaded_chunks += chunks;
        metrics_.offloaded_tokens += tokens;
        store_latency_.record(elapsed_ms);
    }

    void record_retrieve(uint64_t bytes, double elapsed_ms) {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_.retrieve_ops++;
        metrics_.ssd_read_bytes += bytes;
        metrics_.ssd_read_time_ms += elapsed_ms;
        retrieve_latency_.record(elapsed_ms);
    }

    void record_eviction(uint64_t evicted_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_.eviction_count++;
        metrics_.eviction_bytes += evicted_bytes;
    }

    void update_capacity(uint64_t ssd_used, uint64_t ssd_capacity,
                         uint64_t num_entries,
                         uint64_t kv_blocks_used, uint64_t kv_blocks_total) {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_.ssd_used_bytes = ssd_used;
        metrics_.ssd_capacity_bytes = ssd_capacity;
        metrics_.num_entries = num_entries;
        metrics_.kv_blocks_in_use = kv_blocks_used;
        metrics_.kv_blocks_total = kv_blocks_total;
    }

    // ---- Query ----

    CacheMetrics snapshot() const {
        std::lock_guard<std::mutex> lk(mu_);
        return metrics_;
    }

    double uptime_seconds() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

    // ---- Reports ----

    void print_report() const {
        auto m = snapshot();
        double uptime = uptime_seconds();

        std::cerr << "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║             LMCache Monitor Report                     ║\n"
            "╠══════════════════════════════════════════════════════════╣\n";

        // 前缀缓存加速
        fprintf(stderr, "║ %-30s                          ║\n", "── Prefix Cache Acceleration ──");
        fprintf(stderr, "║   Hit Rate:           %6.1f%%  (%lu/%lu requests)       \n",
               m.hit_rate(), m.cache_hit_requests, m.total_requests);
        fprintf(stderr, "║   Full Hit Rate:      %6.1f%%  (%lu skipped prefill)    \n",
               m.full_hit_rate(), m.full_hit_requests);
        fprintf(stderr, "║   Token Save Ratio:   %6.1f%%  (%lu/%lu tokens)        \n",
               m.token_save_ratio(), m.tokens_restored, m.total_prompt_tokens);
        fprintf(stderr, "║   TTFT Saved:       %8.1f ms cumulative              \n", m.ttft_saved_ms);
        fprintf(stderr, "║   Compute Saved:    %8.2f TFLOPS                     \n", m.compute_flops_saved);

        // 上下文扩展
        fprintf(stderr, "║ %-30s                          ║\n", "── Context Extension ──");
        fprintf(stderr, "║   Effective Context:  %6lu tokens (mem + SSD)          \n",
               m.effective_context_tokens());
        fprintf(stderr, "║   Max Context Seen:   %6lu tokens                     \n", m.max_context_seen);
        fprintf(stderr, "║   Offloaded to SSD:   %6lu chunks (%lu tokens)         \n",
               m.offloaded_chunks, m.offloaded_tokens);
        fprintf(stderr, "║   KV Mem Utilization: %6.1f%%                          \n",
               m.kv_memory_utilization());

        // SSD I/O
        fprintf(stderr, "║ %-30s                          ║\n", "── SSD I/O Performance ──");
        fprintf(stderr, "║   Write BW:           %5.2f GB/s  (%.0f MB total)      \n",
               m.ssd_write_bandwidth_gbps(), m.ssd_write_bytes / 1e6);
        fprintf(stderr, "║   Read BW:            %5.2f GB/s  (%.0f MB total)      \n",
               m.ssd_read_bandwidth_gbps(), m.ssd_read_bytes / 1e6);
        fprintf(stderr, "║   Store Latency:      %5.1f ms avg (%lu ops)           \n",
               store_latency_.avg_ms(), m.store_ops);
        fprintf(stderr, "║   Retrieve Latency:   %5.1f ms avg (%lu ops)           \n",
               retrieve_latency_.avg_ms(), m.retrieve_ops);

        // 驱逐与容量
        fprintf(stderr, "║ %-30s                          ║\n", "── Eviction & Capacity ──");
        fprintf(stderr, "║   SSD Utilization:    %6.1f%%  (%.0f/%.0f MB)          \n",
               m.ssd_utilization(), m.ssd_used_bytes / 1e6, m.ssd_capacity_bytes / 1e6);
        fprintf(stderr, "║   Entries:            %6lu                             \n", m.num_entries);
        fprintf(stderr, "║   Evictions:          %6lu  (%.0f MB freed)            \n",
               m.eviction_count, m.eviction_bytes / 1e6);

        fprintf(stderr, "║   Uptime:            %6.1f seconds                     \n", uptime);
        std::cerr <<
            "╚══════════════════════════════════════════════════════════╝\n";
    }

    // JSON 导出 (适合外部监控系统 / Grafana / Prometheus exporter)
    std::string to_json() const {
        auto m = snapshot();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "{\n"
            << "  \"prefix_cache\": {\n"
            << "    \"hit_rate\": " << m.hit_rate() << ",\n"
            << "    \"full_hit_rate\": " << m.full_hit_rate() << ",\n"
            << "    \"token_save_ratio\": " << m.token_save_ratio() << ",\n"
            << "    \"total_requests\": " << m.total_requests << ",\n"
            << "    \"cache_hits\": " << m.cache_hit_requests << ",\n"
            << "    \"full_hits\": " << m.full_hit_requests << ",\n"
            << "    \"partial_hits\": " << m.partial_hit_requests << ",\n"
            << "    \"misses\": " << m.cache_miss_requests << ",\n"
            << "    \"total_prompt_tokens\": " << m.total_prompt_tokens << ",\n"
            << "    \"tokens_restored\": " << m.tokens_restored << ",\n"
            << "    \"tokens_computed\": " << m.tokens_computed << ",\n"
            << "    \"ttft_saved_ms\": " << m.ttft_saved_ms << ",\n"
            << "    \"compute_saved_tflops\": " << m.compute_flops_saved << "\n"
            << "  },\n"
            << "  \"context_extension\": {\n"
            << "    \"effective_context_tokens\": " << m.effective_context_tokens() << ",\n"
            << "    \"max_context_seen\": " << m.max_context_seen << ",\n"
            << "    \"offloaded_chunks\": " << m.offloaded_chunks << ",\n"
            << "    \"offloaded_tokens\": " << m.offloaded_tokens << ",\n"
            << "    \"kv_memory_utilization\": " << m.kv_memory_utilization() << ",\n"
            << "    \"kv_blocks_in_use\": " << m.kv_blocks_in_use << ",\n"
            << "    \"kv_blocks_total\": " << m.kv_blocks_total << "\n"
            << "  },\n"
            << "  \"ssd_io\": {\n"
            << "    \"write_bandwidth_gbps\": " << m.ssd_write_bandwidth_gbps() << ",\n"
            << "    \"read_bandwidth_gbps\": " << m.ssd_read_bandwidth_gbps() << ",\n"
            << "    \"write_bytes\": " << m.ssd_write_bytes << ",\n"
            << "    \"read_bytes\": " << m.ssd_read_bytes << ",\n"
            << "    \"store_ops\": " << m.store_ops << ",\n"
            << "    \"retrieve_ops\": " << m.retrieve_ops << ",\n"
            << "    \"store_latency_avg_ms\": " << store_latency_.avg_ms() << ",\n"
            << "    \"store_latency_p99_ms\": " << store_latency_.p99() << ",\n"
            << "    \"retrieve_latency_avg_ms\": " << retrieve_latency_.avg_ms() << ",\n"
            << "    \"retrieve_latency_p99_ms\": " << retrieve_latency_.p99() << "\n"
            << "  },\n"
            << "  \"eviction\": {\n"
            << "    \"count\": " << m.eviction_count << ",\n"
            << "    \"bytes\": " << m.eviction_bytes << "\n"
            << "  },\n"
            << "  \"capacity\": {\n"
            << "    \"ssd_used_bytes\": " << m.ssd_used_bytes << ",\n"
            << "    \"ssd_capacity_bytes\": " << m.ssd_capacity_bytes << ",\n"
            << "    \"ssd_utilization\": " << m.ssd_utilization() << ",\n"
            << "    \"num_entries\": " << m.num_entries << "\n"
            << "  },\n"
            << "  \"uptime_seconds\": " << std::setprecision(1) << uptime_seconds() << "\n"
            << "}";
        return oss.str();
    }

    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        metrics_ = {};
        store_latency_.reset();
        retrieve_latency_.reset();
        start_time_ = std::chrono::steady_clock::now();
    }

    // 延迟追踪器 (public for direct query)
    LatencyTracker store_latency_;
    LatencyTracker retrieve_latency_;

private:
    mutable std::mutex mu_;
    CacheMetrics metrics_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace cache
} // namespace qwen_thor
