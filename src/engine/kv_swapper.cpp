// KV Block Swapper — Thor 统一内存适配版实现
//
// 核心改进 vs 原始版本:
//   1. 预分配 staging buffer (posix_memalign, 4K 对齐)
//   2. 分批 streaming 写入 (支持任意多 blocks, 不受 buffer 大小限制)
//   3. POSIX_FADV_DONTNEED (drop page cache, 统一内存上必须!)
//   4. SSM/Conv 直写 (统一内存连续, 跳过 staging)
//   5. 后台预取 (prefetch 线程提前从 SSD 加载)
//   6. POSIX fd-based I/O (fileno 精确控制 fadvise)
//
#include "kv_swapper.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// 辅助: 安全 write/read (检查返回值, 消除 warn_unused_result)
// ---------------------------------------------------------------------------
static inline bool safe_write(int fd, const void* buf, size_t count) {
    ssize_t w = ::write(fd, buf, count);
    if (w < 0 || (size_t)w != count) {
        fprintf(stderr, "[KVSwapper] write error: expected %zu, got %zd\n", count, w);
        return false;
    }
    return true;
}

static inline bool safe_read(int fd, void* buf, size_t count) {
    ssize_t r = ::read(fd, buf, count);
    if (r < 0 || (size_t)r != count) {
        fprintf(stderr, "[KVSwapper] read error: expected %zu, got %zd\n", count, r);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 辅助: 递归创建目录
// ---------------------------------------------------------------------------
static void mkdir_p(const std::string& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] == '/') {
            std::string sub = path.substr(0, i);
            mkdir(sub.c_str(), 0755);
        }
    }
    mkdir(path.c_str(), 0755);
}

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------
KVSwapper::KVSwapper(const std::string& swap_dir, int block_bytes, int num_layers,
                     size_t staging_mb)
    : swap_dir_(swap_dir)
    , block_bytes_(block_bytes)
    , num_layers_(num_layers) {
    mkdir_p(swap_dir_);

    // 预分配 staging buffer (4K 对齐, 便于未来 O_DIRECT)
    staging_capacity_ = staging_mb * 1024 * 1024;
    int ret = posix_memalign(&staging_buffer_, 4096, staging_capacity_);
    if (ret != 0 || !staging_buffer_) {
        fprintf(stderr, "[KVSwapper] WARNING: posix_memalign failed, fallback to malloc\n");
        staging_buffer_ = malloc(staging_capacity_);
    }

    // staging buffer \u5206\u6279: \u6bcf\u6279\u5bb9\u7eb3\u591a\u5c11\u4e2a block (K+V \u6240\u6709\u5c42)
    size_t per_block_total = (size_t)num_layers_ * block_bytes_ * 2;
    blocks_per_batch_ = std::max(1, (int)(staging_capacity_ / per_block_total));

    printf("[KVSwapper] Initialized: staging=%zu MB, %d blocks/batch, "
           "block_bytes=%d, layers=%d, dir=%s\n",
           staging_mb, blocks_per_batch_, block_bytes_, num_layers_,
           swap_dir_.c_str());

    // 启动预取线程
    pf_running_ = true;
    prefetch_thread_ = std::thread(&KVSwapper::prefetch_thread_func, this);
}

KVSwapper::~KVSwapper() {
    // 停止预取线程
    {
        std::lock_guard<std::mutex> lock(pf_queue_mutex_);
        pf_running_ = false;
    }
    pf_queue_cv_.notify_all();
    if (prefetch_thread_.joinable()) prefetch_thread_.join();

    // 释放 staging buffer
    if (staging_buffer_) {
        free(staging_buffer_);
        staging_buffer_ = nullptr;
    }

    // 清理残留 swap 文件
    for (auto& [id, rec] : swapped_requests_) {
        std::remove(rec.kv_path.c_str());
        if (rec.has_ssm) {
            std::remove(rec.ssm_path.c_str());
            std::remove(rec.conv_path.c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// drop_page_cache: 写入 SSD 后立即释放 Linux page cache
//
// 在 Thor 统一内存上至关重要:
//   fwrite → 数据进入 page cache → page cache 占用物理页
//   这些物理页本可被 cudaMalloc 使用!
//   FADV_DONTNEED 提示内核: "我不再需要这些页了, 请释放"
// ---------------------------------------------------------------------------
void KVSwapper::drop_page_cache(int fd, size_t size) {
    // fdatasync 确保数据落盘后再 drop
    fdatasync(fd);
    posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED);
}

// ---------------------------------------------------------------------------
// extract_and_write_kv: KV blocks 经 staging buffer 写入 SSD
//
// 即便在 Thor 统一内存上, cudaMallocManaged / cudaMalloc 地址也不能
// 直接用 memcpy/writev/write 访问 (内核 copy_from_user 和用户态
// memcpy 都不触发 CUDA UVM 页错误, 导致 EFAULT/SEGFAULT).
// 必须使用 cudaMemcpy 作为唯一可靠的数据传输机制.
//
// 流程: KV blocks →[cudaMemcpy D2H]→ staging buffer →[write]→ SSD
// 文件格式: [header: num_blocks, context_len] [block0_K_L0..L15, block0_V_L0..L15, ...]
// ---------------------------------------------------------------------------
size_t KVSwapper::extract_and_write_kv(
    int fd,
    const ops::KVCacheManager& kv_manager,
    const std::vector<int>& block_table) {

    int num_blocks = (int)block_table.size();
    size_t per_block = (size_t)num_layers_ * block_bytes_;
    size_t per_block_total = per_block * 2;
    size_t total_written = 0;
    size_t elems_per_block = block_bytes_ / sizeof(__nv_bfloat16);

    uint8_t* staging = (uint8_t*)staging_buffer_;

    for (int start = 0; start < num_blocks; start += blocks_per_batch_) {
        int batch_count = std::min(blocks_per_batch_, num_blocks - start);
        size_t batch_bytes = 0;

        for (int b = 0; b < batch_count; ++b) {
            int block_id = block_table[start + b];
            uint8_t* dst = staging + b * per_block_total;

            for (int L = 0; L < num_layers_; ++L) {
                const __nv_bfloat16* k_ptr = kv_manager.get_layer_k_cache(L);
                size_t offset = (size_t)block_id * elems_per_block;
                cudaMemcpy(dst + L * block_bytes_,
                           k_ptr + offset,
                           block_bytes_, cudaMemcpyDeviceToHost);
            }
            dst += per_block;

            for (int L = 0; L < num_layers_; ++L) {
                const __nv_bfloat16* v_ptr = kv_manager.get_layer_v_cache(L);
                size_t offset = (size_t)block_id * elems_per_block;
                cudaMemcpy(dst + L * block_bytes_,
                           v_ptr + offset,
                           block_bytes_, cudaMemcpyDeviceToHost);
            }

            batch_bytes += per_block_total;
        }

        safe_write(fd, staging, batch_bytes);
        total_written += batch_bytes;
    }

    return total_written;
}

// ---------------------------------------------------------------------------
// read_and_inject_kv: 从数据缓冲区读取并注入 KV blocks
// (数据来自 prefetch cache, host 堆内存)
// Thor 统一内存: KV cache 是 cudaMallocManaged, 用 memcpy 直接注入
// ---------------------------------------------------------------------------
size_t KVSwapper::read_and_inject_kv(
    const uint8_t* data, size_t data_size,
    ops::KVCacheManager& kv_manager,
    const std::vector<int>& block_table) {

    int num_blocks = (int)block_table.size();
    size_t per_block = (size_t)num_layers_ * block_bytes_;
    size_t per_block_total = per_block * 2;
    size_t elems_per_block = block_bytes_ / sizeof(__nv_bfloat16);
    size_t total_read = 0;

    for (int b = 0; b < num_blocks; ++b) {
        int block_id = block_table[b];
        const uint8_t* src = data + b * per_block_total;

        for (int L = 0; L < num_layers_; ++L) {
            __nv_bfloat16* k_ptr = kv_manager.get_layer_k_cache_mut(L);
            size_t offset = (size_t)block_id * elems_per_block;
            cudaMemcpy(k_ptr + offset, src + L * block_bytes_,
                       block_bytes_, cudaMemcpyHostToDevice);
        }
        src += per_block;

        for (int L = 0; L < num_layers_; ++L) {
            __nv_bfloat16* v_ptr = kv_manager.get_layer_v_cache_mut(L);
            size_t offset = (size_t)block_id * elems_per_block;
            cudaMemcpy(v_ptr + offset, src + L * block_bytes_,
                       block_bytes_, cudaMemcpyHostToDevice);
        }

        total_read += per_block_total;
    }

    return total_read;
}

// ---------------------------------------------------------------------------
// write_ssm_conv: SSM/Conv 经 staging buffer 写入 SSD
//
// cudaMemcpy D2H 是 Jetson Thor 上唯一可靠的 GPU→CPU 传输机制
// ---------------------------------------------------------------------------
size_t KVSwapper::write_ssm_conv(int fd,
    float** ssm_states, int num_layers, size_t ssm_per_layer,
    __nv_bfloat16** conv_states, size_t conv_per_layer) {

    size_t total = 0;
    uint8_t* staging = (uint8_t*)staging_buffer_;

    if (ssm_states && num_layers > 0) {
        int ssm_sz = (int)ssm_per_layer;
        safe_write(fd, &num_layers, sizeof(int));
        safe_write(fd, &ssm_sz, sizeof(int));
        total += 2 * sizeof(int);

        for (int li = 0; li < num_layers; ++li) {
            cudaMemcpy(staging, ssm_states[li], ssm_per_layer, cudaMemcpyDeviceToHost);
            safe_write(fd, staging, ssm_per_layer);
            total += ssm_per_layer;
        }
    }

    if (conv_states && num_layers > 0) {
        int conv_sz = (int)conv_per_layer;
        safe_write(fd, &num_layers, sizeof(int));
        safe_write(fd, &conv_sz, sizeof(int));
        total += 2 * sizeof(int);

        for (int li = 0; li < num_layers; ++li) {
            cudaMemcpy(staging, conv_states[li], conv_per_layer, cudaMemcpyDeviceToHost);
            safe_write(fd, staging, conv_per_layer);
            total += conv_per_layer;
        }
    }

    return total;
}

// ---------------------------------------------------------------------------
// read_ssm_conv: 从 prefetch 缓冲区读取 SSM/Conv 并写入 managed memory
// Thor 统一内存: SSM/Conv 是 cudaMallocManaged, 用 memcpy 直接注入
// ---------------------------------------------------------------------------
size_t KVSwapper::read_ssm_conv(const uint8_t* data, size_t data_size,
    float** ssm_states, __nv_bfloat16** conv_states) {

    size_t offset = 0;

    // SSM
    if (ssm_states && offset + 2 * sizeof(int) <= data_size) {
        int num_layers, ssm_sz;
        memcpy(&num_layers, data + offset, sizeof(int)); offset += sizeof(int);
        memcpy(&ssm_sz, data + offset, sizeof(int)); offset += sizeof(int);

        for (int li = 0; li < num_layers && offset + (size_t)ssm_sz <= data_size; ++li) {
            cudaMemcpy(ssm_states[li], data + offset, ssm_sz, cudaMemcpyHostToDevice);
            offset += ssm_sz;
        }
    }

    // Conv
    if (conv_states && offset + 2 * sizeof(int) <= data_size) {
        int num_layers, conv_sz;
        memcpy(&num_layers, data + offset, sizeof(int)); offset += sizeof(int);
        memcpy(&conv_sz, data + offset, sizeof(int)); offset += sizeof(int);

        for (int li = 0; li < num_layers && offset + (size_t)conv_sz <= data_size; ++li) {
            cudaMemcpy(conv_states[li], data + offset, conv_sz, cudaMemcpyHostToDevice);
            offset += conv_sz;
        }
    }

    return offset;
}

// ---------------------------------------------------------------------------
// swap_out: 请求级换出 (GPU → SSD)
// ---------------------------------------------------------------------------
SwapRecord KVSwapper::swap_out(
    uint64_t request_id,
    ops::KVCacheManager& kv_manager,
    const std::vector<int>& block_table,
    int context_len,
    float** ssm_states, int num_linear_layers, size_t ssm_size_per_layer,
    __nv_bfloat16** conv_states, size_t conv_size_per_layer,
    cudaStream_t stream) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // 确保 GPU 操作完成 (统一内存一致性)
    cudaStreamSynchronize(stream);

    SwapRecord rec;
    rec.request_id = request_id;
    rec.num_blocks = (int)block_table.size();
    rec.context_len = context_len;
    rec.kv_path  = swap_dir_ + "/req_" + std::to_string(request_id) + ".kv";
    rec.ssm_path = swap_dir_ + "/req_" + std::to_string(request_id) + ".ssm";
    rec.conv_path = swap_dir_ + "/req_" + std::to_string(request_id) + ".conv";
    rec.has_ssm = (ssm_states != nullptr && num_linear_layers > 0);

    size_t total_bytes = 0;

    // ---- 写 KV blocks (streaming through staging buffer) ----
    {
        int fd = open(rec.kv_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            fprintf(stderr, "[KVSwapper] ERROR: cannot open %s\n", rec.kv_path.c_str());
            return rec;
        }

        // 写 header
        safe_write(fd, &rec.num_blocks, sizeof(int));
        safe_write(fd, &rec.context_len, sizeof(int));
        total_bytes += 2 * sizeof(int);

        // 分批提取+写入 KV blocks
        total_bytes += extract_and_write_kv(fd, kv_manager, block_table);

        // drop page cache (统一内存: 释放物理页给 cudaMalloc!)
        drop_page_cache(fd, total_bytes);
        stats_.page_cache_dropped += total_bytes;

        close(fd);
    }

    // ---- 写 SSM/Conv (统一内存直写, 不经 staging) ----
    if (rec.has_ssm) {
        // SSM 和 Conv 写入同一个文件 (减少文件数)
        int fd = open(rec.ssm_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            size_t ssm_conv_bytes = write_ssm_conv(fd,
                ssm_states, num_linear_layers, ssm_size_per_layer,
                conv_states, conv_size_per_layer);
            drop_page_cache(fd, ssm_conv_bytes);
            stats_.page_cache_dropped += ssm_conv_bytes;
            total_bytes += ssm_conv_bytes;
            close(fd);
        }
    }

    // ---- 释放 KV blocks ----
    kv_manager.free_blocks(block_table);

    auto t1 = std::chrono::high_resolution_clock::now();
    rec.swap_out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rec.total_bytes = total_bytes;

    // 更新统计
    {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        swapped_requests_[request_id] = rec;
        stats_.total_swap_out++;
        stats_.total_swap_out_ms += rec.swap_out_ms;
        stats_.total_bytes_written += total_bytes;
    }

    printf("[KVSwapper] Swap out req %lu: %d blocks, %zu bytes → SSD (%.1f ms, FADV_DONTNEED)\n",
           request_id, rec.num_blocks, total_bytes, rec.swap_out_ms);

    return rec;
}

// ---------------------------------------------------------------------------
// swap_in: 请求级换入 (SSD → GPU)
// ---------------------------------------------------------------------------
std::vector<int> KVSwapper::swap_in(
    uint64_t request_id,
    ops::KVCacheManager& kv_manager,
    float** ssm_states,
    __nv_bfloat16** conv_states,
    cudaStream_t stream) {

    auto t0 = std::chrono::high_resolution_clock::now();

    SwapRecord rec;
    {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        auto it = swapped_requests_.find(request_id);
        if (it == swapped_requests_.end()) {
            fprintf(stderr, "[KVSwapper] ERROR: req %lu not found in swapped set\n", request_id);
            return {};
        }
        rec = it->second;
    }

    size_t total_bytes = 0;

    // ---- 检查 prefetch cache ----
    PrefetchEntry* pf = nullptr;
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        auto it = prefetch_cache_.find(request_id);
        if (it != prefetch_cache_.end()) {
            pf = &it->second;
        }
    }

    // ---- 分配新 KV blocks ----
    auto new_blocks = kv_manager.allocate_blocks(rec.num_blocks);
    if ((int)new_blocks.size() < rec.num_blocks) {
        fprintf(stderr, "[KVSwapper] ERROR: cannot allocate %d blocks for swap_in\n",
                rec.num_blocks);
        return {};
    }

    if (pf) {
        // ---- Prefetch 命中: 从内存注入 (跳过 SSD I/O!) ----
        total_bytes += read_and_inject_kv(
            pf->kv_data.data(), pf->kv_data.size(),
            kv_manager, new_blocks);

        if (ssm_states || conv_states) {
            read_ssm_conv(pf->ssm_conv_data.data(), pf->ssm_conv_data.size(),
                          ssm_states, conv_states);
            total_bytes += pf->ssm_conv_data.size();
        }

        stats_.prefetch_hits++;
    } else {
        // ---- 从 SSD 读取 KV ----
        {
            int fd = open(rec.kv_path.c_str(), O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "[KVSwapper] ERROR: cannot open %s\n", rec.kv_path.c_str());
                kv_manager.free_blocks(new_blocks);
                return {};
            }

            // 跳过 header
            int nb, cl;
            safe_read(fd, &nb, sizeof(int));
            safe_read(fd, &cl, sizeof(int));

            // 分批读取 + 注入 (staging buffer + memcpy)
            // 内核 readv/read 无法直接写入 managed memory, 先读到 staging 再 memcpy
            size_t per_block = (size_t)num_layers_ * block_bytes_;
            size_t per_block_total = per_block * 2;
            size_t elems_per_block = block_bytes_ / sizeof(__nv_bfloat16);
            uint8_t* staging = (uint8_t*)staging_buffer_;

            for (int start = 0; start < rec.num_blocks; start += blocks_per_batch_) {
                int batch_count = std::min(blocks_per_batch_, rec.num_blocks - start);
                size_t batch_bytes = (size_t)batch_count * per_block_total;

                safe_read(fd, staging, batch_bytes);

                // 注入 batch (cudaMemcpy H2D)
                for (int b = 0; b < batch_count; ++b) {
                    int block_id = new_blocks[start + b];
                    uint8_t* src = staging + b * per_block_total;

                    for (int L = 0; L < num_layers_; ++L) {
                        __nv_bfloat16* k_ptr = kv_manager.get_layer_k_cache_mut(L);
                        size_t offset = (size_t)block_id * elems_per_block;
                        cudaMemcpy(k_ptr + offset, src + L * block_bytes_,
                                   block_bytes_, cudaMemcpyHostToDevice);
                    }
                    src += per_block;
                    for (int L = 0; L < num_layers_; ++L) {
                        __nv_bfloat16* v_ptr = kv_manager.get_layer_v_cache_mut(L);
                        size_t offset = (size_t)block_id * elems_per_block;
                        cudaMemcpy(v_ptr + offset, src + L * block_bytes_,
                                   block_bytes_, cudaMemcpyHostToDevice);
                    }
                }

                total_bytes += batch_bytes;
            }

            // 读完后 drop page cache (避免 SSD 读取的数据残留在内存)
            drop_page_cache(fd, total_bytes);

            close(fd);
        }

        // ---- 读 SSM/Conv (经 staging buffer + cudaMemcpy) ----
        if (rec.has_ssm && (ssm_states || conv_states)) {
            int fd = open(rec.ssm_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                uint8_t* staging = (uint8_t*)staging_buffer_;
                size_t ssm_conv_bytes = 0;

                if (ssm_states) {
                    int num_layers_ssm, ssm_sz;
                    safe_read(fd, &num_layers_ssm, sizeof(int));
                    safe_read(fd, &ssm_sz, sizeof(int));
                    ssm_conv_bytes += 2 * sizeof(int);
                    for (int li = 0; li < num_layers_ssm; ++li) {
                        safe_read(fd, staging, ssm_sz);
                        cudaMemcpy(ssm_states[li], staging, ssm_sz, cudaMemcpyHostToDevice);
                        ssm_conv_bytes += ssm_sz;
                    }
                }

                if (conv_states) {
                    int num_layers_conv, conv_sz;
                    safe_read(fd, &num_layers_conv, sizeof(int));
                    safe_read(fd, &conv_sz, sizeof(int));
                    ssm_conv_bytes += 2 * sizeof(int);
                    for (int li = 0; li < num_layers_conv; ++li) {
                        safe_read(fd, staging, conv_sz);
                        cudaMemcpy(conv_states[li], staging, conv_sz, cudaMemcpyHostToDevice);
                        ssm_conv_bytes += conv_sz;
                    }
                }

                total_bytes += ssm_conv_bytes;
                drop_page_cache(fd, ssm_conv_bytes);
                close(fd);
            }
        }
    }

    // 同步确保数据在 GPU 可见 (统一内存一致性)
    cudaStreamSynchronize(stream);

    // ---- 清理 ----
    std::remove(rec.kv_path.c_str());
    if (rec.has_ssm) {
        std::remove(rec.ssm_path.c_str());
        // conv 已合并到 ssm 文件, 但删除旧 conv 文件以防万一
        std::remove(rec.conv_path.c_str());
    }

    // 清理 prefetch cache
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        prefetch_cache_.erase(request_id);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double swap_in_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 更新统计
    {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        swapped_requests_.erase(request_id);
        stats_.total_swap_in++;
        stats_.total_swap_in_ms += swap_in_ms;
        stats_.total_bytes_read += total_bytes;
    }

    printf("[KVSwapper] Swap in req %lu: %d blocks, %zu bytes ← %s (%.1f ms)\n",
           request_id, rec.num_blocks, total_bytes,
           pf ? "PREFETCH" : "SSD", swap_in_ms);

    return new_blocks;
}

// ---------------------------------------------------------------------------
// prefetch: 后台异步预取
// ---------------------------------------------------------------------------
void KVSwapper::prefetch(uint64_t request_id) {
    // 检查是否已在 prefetch cache 或队列中
    {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        if (prefetch_cache_.count(request_id)) return;
    }

    {
        std::lock_guard<std::mutex> lock(pf_queue_mutex_);
        prefetch_queue_.push(request_id);
    }
    pf_queue_cv_.notify_one();
    stats_.total_prefetch++;
}

bool KVSwapper::is_prefetched(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    return prefetch_cache_.count(request_id) > 0;
}

// ---------------------------------------------------------------------------
// prefetch_thread_func: 后台预取线程
// ---------------------------------------------------------------------------
void KVSwapper::prefetch_thread_func() {
    while (true) {
        uint64_t req_id;
        {
            std::unique_lock<std::mutex> lock(pf_queue_mutex_);
            pf_queue_cv_.wait(lock, [this] {
                return !prefetch_queue_.empty() || !pf_running_;
            });
            if (!pf_running_ && prefetch_queue_.empty()) return;
            req_id = prefetch_queue_.front();
            prefetch_queue_.pop();
        }

        // 获取 swap record
        SwapRecord rec;
        {
            std::lock_guard<std::mutex> lock(swap_mutex_);
            auto it = swapped_requests_.find(req_id);
            if (it == swapped_requests_.end()) continue;
            rec = it->second;
        }

        // 读取 KV 文件到内存
        PrefetchEntry entry;
        {
            int fd = open(rec.kv_path.c_str(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st;
            fstat(fd, &st);
            entry.kv_data.resize(st.st_size - 2 * sizeof(int));  // 去掉 header

            // 跳过 header
            int nb, cl;
            safe_read(fd, &nb, sizeof(int));
            safe_read(fd, &cl, sizeof(int));
            safe_read(fd, entry.kv_data.data(), entry.kv_data.size());

            // drop page cache after prefetch read
            drop_page_cache(fd, st.st_size);
            close(fd);
        }

        // 读取 SSM/Conv 文件
        if (rec.has_ssm) {
            int fd = open(rec.ssm_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                fstat(fd, &st);
                entry.ssm_conv_data.resize(st.st_size);
                safe_read(fd, entry.ssm_conv_data.data(), st.st_size);
                drop_page_cache(fd, st.st_size);
                close(fd);
            }
        }

        // 存入 prefetch cache
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            prefetch_cache_[req_id] = std::move(entry);
        }
    }
}

// ---------------------------------------------------------------------------
// 辅助方法
// ---------------------------------------------------------------------------
bool KVSwapper::is_swapped_out(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(swap_mutex_);
    return swapped_requests_.count(request_id) > 0;
}

int KVSwapper::get_swapped_context_len(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(swap_mutex_);
    auto it = swapped_requests_.find(request_id);
    return (it != swapped_requests_.end()) ? it->second.context_len : 0;
}

void KVSwapper::remove(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(swap_mutex_);
    auto it = swapped_requests_.find(request_id);
    if (it != swapped_requests_.end()) {
        std::remove(it->second.kv_path.c_str());
        if (it->second.has_ssm) {
            std::remove(it->second.ssm_path.c_str());
            std::remove(it->second.conv_path.c_str());
        }
        swapped_requests_.erase(it);
    }
    // 也清理 prefetch cache
    {
        std::lock_guard<std::mutex> lock2(prefetch_mutex_);
        prefetch_cache_.erase(request_id);
    }
}

void KVSwapper::drain() {
    // 等待 prefetch 队列清空
    while (true) {
        std::lock_guard<std::mutex> lock(pf_queue_mutex_);
        if (prefetch_queue_.empty()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void KVSwapper::print_stats() const {
    printf("[KVSwapper] Stats: swap_out=%d (avg %.1fms, %.1f MB written), "
           "swap_in=%d (avg %.1fms, %.1f MB read), "
           "prefetch=%d (hits=%d), page_cache_dropped=%.1f MB\n",
           stats_.total_swap_out, stats_.avg_swap_out_ms(),
           stats_.total_bytes_written / (1024.0 * 1024),
           stats_.total_swap_in, stats_.avg_swap_in_ms(),
           stats_.total_bytes_read / (1024.0 * 1024),
           stats_.total_prefetch, stats_.prefetch_hits,
           stats_.page_cache_dropped / (1024.0 * 1024));
}

} // namespace cache
} // namespace qwen_thor
