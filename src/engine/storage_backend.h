// KV Cache Offload — 存储后端抽象接口
// 对标 LMCache 的 StorageBackendInterface
#pragma once

#include "cache_key.h"
#include "cache_entry.h"
#include <vector>
#include <functional>

namespace qwen_thor {
namespace cache {

// 驱逐回调: (evicted_bytes) → void
using EvictionCallback = std::function<void(uint64_t)>;

// ---------------------------------------------------------------------------
// StorageBackend: 存储后端抽象接口
//
// 实现:
//   - DiskBackend: SSD 磁盘文件 (二进制格式) + LRU 驱逐
//
// 所有方法默认线程安全 (内部 mutex)
// ---------------------------------------------------------------------------
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    // ---- 驱逐回调 ----
    void set_eviction_callback(EvictionCallback cb) { eviction_cb_ = std::move(cb); }

    // ---- 查询 ----

    // 检查 key 是否存在
    virtual bool contains(const CacheKey& key) const = 0;

    // 前缀匹配: 给定一系列 chunk key, 返回连续匹配的 chunk 数
    // keys[0] 必须匹配, keys[1] 必须在 keys[0] 之后匹配, 以此类推
    // 返回 0 表示无匹配
    virtual int prefix_match(const std::vector<CacheKey>& keys) const = 0;

    // ---- 读写 ----

    // 存入一个缓存条目
    // 如果 key 已存在, 返回 false (不覆盖)
    virtual bool put(const CacheKey& key, CacheEntryPtr entry) = 0;

    // 取出一个缓存条目
    // 如果不存在, 返回 nullptr
    // 成功取出后, 更新 LRU 访问记录
    virtual CacheEntryPtr get(const CacheKey& key) = 0;

    // 删除一个缓存条目
    virtual bool remove(const CacheKey& key) = 0;

    // 批量删除
    virtual int remove_batch(const std::vector<CacheKey>& keys) {
        int removed = 0;
        for (auto& k : keys) removed += remove(k) ? 1 : 0;
        return removed;
    }

    // ---- 容量 ----

    virtual size_t current_size_bytes() const = 0;
    virtual size_t max_size_bytes() const = 0;
    virtual size_t num_entries() const = 0;

    // ---- 驱逐 ----

    // 获取 LRU 驱逐候选 (最久未使用的 n 个 key)
    virtual std::vector<CacheKey> get_evict_candidates(int n) const = 0;

    // ---- 信息 ----

    virtual const char* name() const = 0;

protected:
    EvictionCallback eviction_cb_;
};

} // namespace cache
} // namespace qwen_thor
