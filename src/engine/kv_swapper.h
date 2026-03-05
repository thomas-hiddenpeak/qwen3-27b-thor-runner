// KV Block Swapper — Thor 统一内存适配版 (GPU ↔ SSD)
//
// 架构借鉴 LMCache, 适配 Jetson AGX Thor 统一内存平台:
//
//   Thor 硬件特性:
//     - 128 GB LPDDR5X 统一内存, CPU 和 GPU 共享同一物理池
//     - 无独立显存, 无 "CPU offload 层" — cudaMalloc ≡ 统一内存
//     - NVMe SSD 是唯一的 offload 目标 (Write ~2 GB/s, Read ~7-8 GB/s)
//     - page cache 与 cudaMalloc 争夺同一物理内存 → 必须用 FADV_DONTNEED
//
//   关键设计 (vs 原始版本的改进):
//     1. 预分配 I/O staging buffer
//        KV blocks 在 paged attention 池中是散列的, 需先合并再顺序写入 SSD.
//        通过 staging buffer 分批提取+写入 (streaming), 支持任意大小的 KV,
//        避免每次 swap 的 heap 分配和内存碎片.
//
//     2. POSIX_FADV_DONTNEED: 写入 SSD 后立即 drop page cache
//        在统一内存上, page cache 占用的物理页 = cudaMalloc 可用的物理页,
//        不 drop 会导致 SSD 写入反而增加内存压力!
//
//     3. SSM/Conv 通过 staging buffer: 先 cudaMemcpy D2H 再 fwrite,
//        在 Thor 统一内存上 cudaMemcpy D2H 等价于 memcpy, 无额外开销
//
//     4. 后台预取线程: swap_in 前可调用 prefetch(),
//        后台线程提前从 SSD 读数据到内存, 减少显式 swap_in 延迟
//
//   数据流:
//     swap_out:
//       KV blocks (scattered, device mem)
//         →[cudaMemcpy D2H]→ staging buffer (consolidated)
//         →[write@2GB/s]→ SSD file →[FADV_DONTNEED]→ drop page cache
//       SSM/Conv (contiguous, device mem)
//         →[cudaMemcpy D2H]→ staging →[write@2GB/s]→ SSD file
//       → free_blocks(KV) + engine frees SSM/Conv
//
//     swap_in:
//       SSD file →[read@7GB/s]→ staging buffer
//         →[cudaMemcpy H2D]→ new KV blocks (device mem)
//       SSD file →[read]→ host buf →[cudaMemcpy H2D]→ SSM/Conv
//
//     prefetch (async):
//       SSD file →[background fread]→ prefetch cache (heap)
//       后续 swap_in 直接从 prefetch cache 注入, 跳过 SSD I/O
//
// 文件布局 (per request):
//   <swap_dir>/req_<id>.kv    — KV cache: [header][batch0][batch1]...
//   <swap_dir>/req_<id>.ssm   — SSM state: [header][layers...]
//   <swap_dir>/req_<id>.conv  — Conv state: [header][layers...]
//
// 性能预期 (Thor NVMe SSD):
//   128 tokens (8 blocks, ~8 MB KV): swap_out ~4ms, swap_in ~1ms
//   4096 tokens (256 blocks, ~256 MB KV): swap_out ~128ms, swap_in ~33ms
//   4096 tokens with prefetch: swap_in ~1ms (data already in memory)
#pragma once

#include "paged_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// SwapRecord: 一个被换出到 SSD 的请求的状态
// ---------------------------------------------------------------------------
struct SwapRecord {
    uint64_t request_id = 0;
    int      num_blocks = 0;
    int      context_len = 0;
    std::string kv_path;
    std::string ssm_path;
    std::string conv_path;
    bool     has_ssm = false;
    double   swap_out_ms = 0;       // 换出总耗时
    size_t   total_bytes = 0;       // 写入 SSD 的总字节数
};

// ---------------------------------------------------------------------------
// SwapStats: 统计信息
// ---------------------------------------------------------------------------
struct SwapStats {
    int     total_swap_out = 0;
    int     total_swap_in  = 0;
    int     total_prefetch = 0;
    int     prefetch_hits  = 0;     // swap_in 命中 prefetch cache
    double  total_swap_out_ms = 0;
    double  total_swap_in_ms  = 0;
    size_t  total_bytes_written = 0;
    size_t  total_bytes_read    = 0;
    size_t  page_cache_dropped  = 0;  // FADV_DONTNEED 释放的字节数

    double avg_swap_out_ms() const { return total_swap_out > 0 ? total_swap_out_ms / total_swap_out : 0; }
    double avg_swap_in_ms()  const { return total_swap_in > 0 ? total_swap_in_ms / total_swap_in : 0; }
};

// ---------------------------------------------------------------------------
// KVSwapper: 请求级 KV 换出/换入管理器
//   借鉴 LMCache 分层存储架构, 适配 Thor 统一内存:
//     - 无 CPU tier (统一内存 CPU≡GPU)
//     - 仅 SSD offload + FADV_DONTNEED
//     - 预分配 staging buffer + 后台预取
// ---------------------------------------------------------------------------
class KVSwapper {
public:
    // swap_dir: SSD 上的换出目录
    // block_bytes: 每个 block 每层的字节数 (block_size × num_heads × head_dim × 2)
    // num_layers: full attention 层数 (16)
    // staging_mb: staging buffer 大小 (MB), 默认 32 MB
    KVSwapper(const std::string& swap_dir, int block_bytes, int num_layers,
              size_t staging_mb = 32);
    ~KVSwapper();

    // ---- 换出: 请求的 KV blocks + SSM/Conv → SSD, 释放 GPU blocks ----
    SwapRecord swap_out(
        uint64_t request_id,
        ops::KVCacheManager& kv_manager,
        const std::vector<int>& block_table,
        int context_len,
        __nv_bfloat16** ssm_states, int num_linear_layers, size_t ssm_size_per_layer,
        __nv_bfloat16** conv_states, size_t conv_size_per_layer,
        cudaStream_t stream);

    // ---- 换入: SSD → 新分配的 KV blocks + SSM/Conv ----
    std::vector<int> swap_in(
        uint64_t request_id,
        ops::KVCacheManager& kv_manager,
        __nv_bfloat16** ssm_states,
        __nv_bfloat16** conv_states,
        cudaStream_t stream);

    // ---- 预取: 后台线程提前从 SSD 读取到内存 ----
    void prefetch(uint64_t request_id);
    bool is_prefetched(uint64_t request_id) const;

    // ---- 查询 ----
    bool is_swapped_out(uint64_t request_id) const;
    int get_swapped_context_len(uint64_t request_id) const;

    // ---- 清理 ----
    void remove(uint64_t request_id);

    // ---- 等待后台预取完成 ----
    void drain();

    // ---- 统计 ----
    SwapStats get_stats() const { return stats_; }
    void print_stats() const;

private:
    // 分批通过 staging buffer 提取 KV blocks 并写入 SSD 文件
    // 返回写入的字节数
    size_t extract_and_write_kv(
        int fd,
        const ops::KVCacheManager& kv_manager,
        const std::vector<int>& block_table);

    // 从 SSD 文件/prefetch 缓存读取并注入 KV blocks
    size_t read_and_inject_kv(
        const uint8_t* data, size_t data_size,
        ops::KVCacheManager& kv_manager,
        const std::vector<int>& block_table);

    // 统一内存直写 SSM/Conv (不经 staging)
    size_t write_ssm_conv(int fd,
        __nv_bfloat16** ssm_states, int num_layers, size_t ssm_per_layer,
        __nv_bfloat16** conv_states, size_t conv_per_layer);
    size_t read_ssm_conv(const uint8_t* data, size_t data_size,
        __nv_bfloat16** ssm_states, __nv_bfloat16** conv_states);

    // drop page cache (统一内存上至关重要!)
    static void drop_page_cache(int fd, size_t size);

    // 预取线程主函数
    void prefetch_thread_func();

    // 预取缓存条目 (整个文件内容)
    struct PrefetchEntry {
        std::vector<uint8_t> kv_data;
        std::vector<uint8_t> ssm_conv_data;
    };

    std::string swap_dir_;
    int block_bytes_;       // 每 block 每层的字节数
    int num_layers_;        // full attention 层数
    int blocks_per_batch_;  // staging buffer 可容纳多少个 block

    // 预分配 staging buffer (4K 对齐, 用于 KV block consolidation)
    void* staging_buffer_ = nullptr;
    size_t staging_capacity_ = 0;

    // 已换出请求
    std::unordered_map<uint64_t, SwapRecord> swapped_requests_;
    mutable std::mutex swap_mutex_;

    // 预取缓存
    std::unordered_map<uint64_t, PrefetchEntry> prefetch_cache_;
    mutable std::mutex prefetch_mutex_;

    // 预取线程
    std::thread prefetch_thread_;
    std::queue<uint64_t> prefetch_queue_;
    std::mutex pf_queue_mutex_;
    std::condition_variable pf_queue_cv_;
    std::atomic<bool> pf_running_{false};

    SwapStats stats_;
};

} // namespace cache
} // namespace qwen_thor
