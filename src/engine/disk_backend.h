// KV Cache Offload — Disk 后端 (Binary Files + LRU)
// 对标 LMCache 的 LocalDiskBackend
#pragma once

#include "storage_backend.h"
#include <unordered_map>
#include <mutex>
#include <list>
#include <string>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// DiskBackend: 磁盘文件存储 + LRU 驱逐
//
// 存储格式:
//   每个 CacheEntry 存为一个二进制文件: <cache_dir>/<hex_hash>.bin
//   文件头: [magic, version, num_tokens, kv_bytes, ssm_bytes, conv_bytes, tokens...]
//   文件体: [KV data, SSM data, Conv data]
//
// 特点:
//   - 容量可远大于内存 (SSD 存储)
//   - 读取时按需加载到内存 (get 返回 CacheEntryPtr)
//   - 适合缓存不常用的前缀
//
// 线程安全: 所有公开方法均持有 mutex_
// ---------------------------------------------------------------------------
class DiskBackend : public StorageBackend {
public:
    DiskBackend(const std::string& cache_dir, size_t max_bytes);
    ~DiskBackend() override;

    // ---- StorageBackend interface ----
    bool contains(const CacheKey& key) const override;
    int  prefix_match(const std::vector<CacheKey>& keys) const override;
    bool put(const CacheKey& key, CacheEntryPtr entry) override;
    CacheEntryPtr get(const CacheKey& key) override;
    bool remove(const CacheKey& key) override;

    size_t current_size_bytes() const override;
    size_t max_size_bytes() const override;
    size_t num_entries() const override;

    std::vector<CacheKey> get_evict_candidates(int n) const override;

    const char* name() const override { return "DiskBackend"; }

private:
    // 获取 key 对应的文件路径
    std::string key_to_path(const CacheKey& key) const;

    // 序列化 / 反序列化
    bool write_entry(const std::string& path, const CacheEntry& entry);
    CacheEntryPtr read_entry(const std::string& path);

    // 驱逐
    void evict_until_fit(size_t needed_bytes);

    std::string cache_dir_;
    size_t max_bytes_;

    // 元数据索引 (仅跟踪文件大小, 不缓存内容)
    struct FileInfo {
        size_t file_bytes;
    };

    using LRUList = std::list<std::pair<CacheKey, FileInfo>>;
    using LRUIterator = LRUList::iterator;

    LRUList lru_list_;
    std::unordered_map<CacheKey, LRUIterator, CacheKeyHash> index_;

    size_t current_bytes_ = 0;
    mutable std::mutex mutex_;
};

} // namespace cache
} // namespace qwen_thor
