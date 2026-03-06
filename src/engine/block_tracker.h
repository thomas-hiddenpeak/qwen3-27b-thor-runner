// Block Tracker — 管理 KV blocks 的 GPU/SSD 位置追踪
//
// 当 KV cache budget 不足以容纳全部 context 时 (如 4 GB 预算 + 200K context),
// 部分 blocks 需要 evict 到 SSD, 只保留热区 blocks 在 GPU.
//
// 术语:
//   - GPU blocks: 在 KVCacheManager pool 中, 可被 paged attention 直接访问
//   - SSD blocks: 已写入 SSD 文件, GPU pool 中的 slot 已释放
//
// 每个请求维护一个 block 列表, 每个 block 有:
//   - logical_idx: 在序列中的逻辑位置 (第 0, 1, 2, ... 个 block)
//   - physical_block: KVCacheManager 中的 block ID (GPU-resident 时有效)
//   - on_ssd: 是否已 evict 到 SSD
//
// SSD 文件布局 (per request per layer_batch):
//   <tracker_dir>/req_<id>_layer_<L>.kv
//   每个 block 顺序存储: [K_data | V_data] = block_size × kv_dim × 2(bf16) × 2(K+V)
#pragma once

#include "paged_attention.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace qwen_thor {
namespace cache {

// SSD 上单层单 block 的字节数: block_size × num_kv_heads × head_dim × sizeof(bf16)
// 对于 Qwen3.5-27B: 16 × 4 × 256 × 2 = 32768 bytes = 32 KB (K 或 V)
// K + V = 64 KB per block per layer

struct BlockLocation {
    int physical_block;   // KVCacheManager 中的 block ID (on_ssd 时无效=-1)
    bool on_ssd;          // 是否在 SSD 上
};

// ---------------------------------------------------------------------------
// BlockTracker: 追踪单个请求的 blocks 在 GPU vs SSD 的分布
// ---------------------------------------------------------------------------
class BlockTracker {
public:
    BlockTracker() = default;

    // 初始化: 所有 blocks 在 GPU
    void init(const std::vector<int>& gpu_block_table) {
        locs_.resize(gpu_block_table.size());
        for (size_t i = 0; i < gpu_block_table.size(); ++i) {
            locs_[i].physical_block = gpu_block_table[i];
            locs_[i].on_ssd = false;
        }
    }

    // 添加新 block (GPU-resident)
    void push_block(int physical_block) {
        locs_.push_back({physical_block, false});
    }

    // 标记指定 logical blocks 已 evict 到 SSD (physical_block 无效化)
    void mark_evicted(int start_logical, int count) {
        for (int i = start_logical; i < start_logical + count && i < (int)locs_.size(); ++i) {
            locs_[i].physical_block = -1;
            locs_[i].on_ssd = true;
        }
    }

    // 获取所有 GPU-resident blocks 的 physical IDs (用于标准 paged attention)
    // 返回的 block table 保持逻辑顺序, SSD blocks 用 -1 占位
    std::vector<int> get_full_block_table() const {
        std::vector<int> bt(locs_.size());
        for (size_t i = 0; i < locs_.size(); ++i)
            bt[i] = locs_[i].physical_block;
        return bt;
    }

    // 获取仅 GPU-resident blocks 的 table (紧凑, 用于 hot-pass attention)
    // 同时输出 logical → hot_idx 的映射
    std::vector<int> get_gpu_block_table() const {
        std::vector<int> bt;
        for (auto& loc : locs_)
            if (!loc.on_ssd) bt.push_back(loc.physical_block);
        return bt;
    }

    // 获取 SSD-resident blocks 的 logical indices
    std::vector<int> get_ssd_logical_indices() const {
        std::vector<int> indices;
        for (size_t i = 0; i < locs_.size(); ++i)
            if (locs_[i].on_ssd) indices.push_back((int)i);
        return indices;
    }

    int num_blocks() const { return (int)locs_.size(); }
    int num_gpu_blocks() const {
        int c = 0;
        for (auto& loc : locs_) if (!loc.on_ssd) c++;
        return c;
    }
    int num_ssd_blocks() const {
        int c = 0;
        for (auto& loc : locs_) if (loc.on_ssd) c++;
        return c;
    }
    bool has_ssd_blocks() const {
        for (auto& loc : locs_) if (loc.on_ssd) return true;
        return false;
    }

    const BlockLocation& operator[](int logical_idx) const { return locs_[logical_idx]; }

private:
    std::vector<BlockLocation> locs_;
};

// ---------------------------------------------------------------------------
// BlockSSDStore: 按层按 block 读写 SSD 文件
//
// 文件布局: 每层一个文件, 每个 block 按 logical index 顺序存储
//   offset = logical_block_idx × per_block_bytes
//   per_block_bytes = block_size × num_kv_heads × head_dim × sizeof(bf16) × 2 (K+V)
//
// 支持:
//   - 逐层写入指定 blocks (evict 时)
//   - 逐层读取指定 blocks 到 staging buffer (streaming attention 时)
// ---------------------------------------------------------------------------
class BlockSSDStore {
public:
    BlockSSDStore(const std::string& store_dir,
                  int block_size, int num_kv_heads, int head_dim,
                  int num_layers,
                  void* staging_buffer, size_t staging_capacity)
        : store_dir_(store_dir),
          block_size_(block_size),
          num_kv_heads_(num_kv_heads),
          head_dim_(head_dim),
          num_layers_(num_layers),
          staging_buffer_(staging_buffer),
          staging_capacity_(staging_capacity)
    {
        kv_dim_ = num_kv_heads * head_dim;
        // 每 block 单层: K = block_size × kv_dim × sizeof(bf16)
        k_block_bytes_ = (size_t)block_size * kv_dim_ * sizeof(__nv_bfloat16);
        v_block_bytes_ = k_block_bytes_;
        per_block_bytes_ = k_block_bytes_ + v_block_bytes_;  // K + V

        // staging 能容纳多少 blocks (per layer)
        blocks_per_batch_ = staging_capacity_ / per_block_bytes_;
        if (blocks_per_batch_ < 1) blocks_per_batch_ = 1;

        // 创建目录
        std::string cmd = "mkdir -p " + store_dir_;
        if (system(cmd.c_str()) != 0) {
            fprintf(stderr, "[BlockSSDStore] mkdir -p failed for %s\n", store_dir_.c_str());
        }
    }

    // Evict: 将指定 blocks 的 KV 从 GPU cache 写入 SSD (所有 layers)
    // blocks_to_evict: logical block indices
    void evict_blocks(uint64_t request_id,
                      const std::vector<int>& blocks_to_evict,
                      const std::vector<int>& block_table,  // logical→physical mapping
                      const ops::KVCacheManager& kv_manager) {
        for (int L = 0; L < num_layers_; ++L) {
            std::string path = make_path(request_id, L);
            // 追加模式不行, 我们需要随机写入; 用 O_CREAT|O_RDWR
            int fd = open(path.c_str(), O_CREAT | O_RDWR, 0644);
            if (fd < 0) {
                fprintf(stderr, "[BlockSSDStore] Cannot open %s: %s\n",
                        path.c_str(), strerror(errno));
                continue;
            }

            const __nv_bfloat16* k_cache = kv_manager.get_layer_k_cache(L);
            const __nv_bfloat16* v_cache = kv_manager.get_layer_v_cache(L);
            size_t elems_per_block = (size_t)block_size_ * kv_dim_;

            for (int bi : blocks_to_evict) {
                int phys = block_table[bi];
                if (phys < 0) continue;

                // K: 从 cache 读到 staging
                cudaMemcpy(staging_buffer_,
                           k_cache + phys * elems_per_block,
                           k_block_bytes_, cudaMemcpyDeviceToHost);
                // V: 紧接 K 后面
                cudaMemcpy((uint8_t*)staging_buffer_ + k_block_bytes_,
                           v_cache + phys * elems_per_block,
                           v_block_bytes_, cudaMemcpyDeviceToHost);

                // 写入 SSD 的正确偏移位置
                off_t offset = (off_t)bi * per_block_bytes_;
                if (pwrite(fd, staging_buffer_, per_block_bytes_, offset) < 0) {
                    fprintf(stderr, "[BlockSSDStore] pwrite failed block %d: %s\n", bi, strerror(errno));
                }
            }

            // Drop page cache
            off_t file_size = lseek(fd, 0, SEEK_END);
            if (file_size > 0) {
                posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
            }
            close(fd);
        }
    }

    // Load: 从 SSD 读取指定 blocks 到 staging K/V cache buffers
    // 调用方提供 staging K/V buffer (GPU memory), 本函数填充后可用于 paged attention
    //
    // ssd_logical_indices: 要加载的 logical block indices
    // d_staging_k, d_staging_v: GPU staging cache, 布局同 KVCacheManager 的 per-layer cache
    //   staging block i 对应 d_staging_k[i * elems_per_block ... (i+1)*elems_per_block]
    // layer_idx: 当前层
    //
    // 返回加载的 block 数
    int load_blocks_for_layer(uint64_t request_id,
                              int layer_idx,
                              const std::vector<int>& ssd_logical_indices,
                              __nv_bfloat16* d_staging_k,
                              __nv_bfloat16* d_staging_v) {
        std::string path = make_path(request_id, layer_idx);
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "[BlockSSDStore] Cannot open %s for read\n", path.c_str());
            return 0;
        }

        size_t elems_per_block = (size_t)block_size_ * kv_dim_;
        int loaded = 0;

        // 分批加载
        for (size_t start = 0; start < ssd_logical_indices.size(); start += blocks_per_batch_) {
            size_t end = std::min(start + (size_t)blocks_per_batch_, ssd_logical_indices.size());
            size_t batch = end - start;

            // Phase 1: 批量从 SSD 读取到 staging buffer
            bool read_ok[256];  // blocks_per_batch_ 不会超过 staging 容量
            for (size_t bi = 0; bi < batch; ++bi) {
                int logical_idx = ssd_logical_indices[start + bi];
                off_t offset = (off_t)logical_idx * per_block_bytes_;

                ssize_t nr = pread(fd, (uint8_t*)staging_buffer_ + bi * per_block_bytes_,
                                   per_block_bytes_, offset);
                if (nr < (ssize_t)per_block_bytes_) {
                    fprintf(stderr, "[BlockSSDStore] pread failed block %d: %s\n",
                            logical_idx, nr < 0 ? strerror(errno) : "short read");
                    read_ok[bi] = false;
                } else {
                    read_ok[bi] = true;
                }
            }

            // Phase 2: 注入 GPU staging cache (跳过读取失败的 block)
            for (size_t bi = 0; bi < batch; ++bi) {
                if (!read_ok[bi]) continue;
                int staging_slot = (int)(start + bi);
                uint8_t* src = (uint8_t*)staging_buffer_ + bi * per_block_bytes_;

                // K
                cudaMemcpy(d_staging_k + staging_slot * elems_per_block,
                           src, k_block_bytes_, cudaMemcpyHostToDevice);
                // V
                cudaMemcpy(d_staging_v + staging_slot * elems_per_block,
                           src + k_block_bytes_, v_block_bytes_, cudaMemcpyHostToDevice);
            }
            loaded += (int)batch;
        }

        // Drop page cache
        off_t file_size = lseek(fd, 0, SEEK_END);
        if (file_size > 0)
            posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
        close(fd);
        return loaded;
    }

    // 清理请求的所有 SSD 文件
    void remove(uint64_t request_id) {
        for (int L = 0; L < num_layers_; ++L) {
            std::string path = make_path(request_id, L);
            unlink(path.c_str());
        }
    }

    int per_block_bytes() const { return (int)per_block_bytes_; }
    int blocks_per_batch() const { return blocks_per_batch_; }

private:
    std::string make_path(uint64_t req_id, int layer_idx) const {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s/req_%lu_L%02d.kv",
                 store_dir_.c_str(), req_id, layer_idx);
        return std::string(buf);
    }

    std::string store_dir_;
    int block_size_;
    int num_kv_heads_;
    int head_dim_;
    int kv_dim_;
    int num_layers_;
    size_t k_block_bytes_;
    size_t v_block_bytes_;
    size_t per_block_bytes_;
    int blocks_per_batch_;
    void* staging_buffer_;
    size_t staging_capacity_;
};

} // namespace cache
} // namespace qwen_thor
