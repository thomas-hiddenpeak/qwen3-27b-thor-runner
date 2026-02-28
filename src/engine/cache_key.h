// KV Cache Offload — Token 前缀哈希 + CacheKey
// 基于 LMCache 的 ChunkedTokenDatabase 设计:
//   将 Token 序列按固定 chunk_size 切分,
//   每个 chunk 的哈希依赖前一个 chunk 的哈希 → 形成前缀链
//   CacheKey = chunk 的前缀链哈希值
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <algorithm>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash — 简单、快速、无依赖
// ---------------------------------------------------------------------------
inline uint64_t fnv1a_hash(const void* data, size_t len,
                           uint64_t seed = 0xcbf29ce484222325ULL) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    uint64_t hash = seed;
    for (size_t i = 0; i < len; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// ---------------------------------------------------------------------------
// CacheKey: 通过前缀链哈希唯一标识一个 KV 缓存 chunk
// ---------------------------------------------------------------------------
struct CacheKey {
    uint64_t hash;

    bool operator==(const CacheKey& o) const { return hash == o.hash; }
    bool operator!=(const CacheKey& o) const { return hash != o.hash; }
    bool operator<(const CacheKey& o)  const { return hash <  o.hash; }
};

struct CacheKeyHash {
    size_t operator()(const CacheKey& key) const { return key.hash; }
};

// ---------------------------------------------------------------------------
// TokenHasher: 将 token 序列转换为前缀链 CacheKey 序列
//
// 算法 (与 LMCache prefix_hash 等价):
//   prefix_hash_0 = 0
//   for i in 0..num_chunks:
//       chunk_hash_i = fnv1a(prefix_hash_{i-1} || tokens[chunk_i])
//       keys[i] = { chunk_hash_i }
//       prefix_hash_i = chunk_hash_i
//
// 性质:
//   - 相同前缀 → 相同 key 序列 (精确匹配)
//   - 不同前缀从第一个不同 chunk 开始产生不同 key
//   - O(n) 计算, n = token 数量
// ---------------------------------------------------------------------------
class TokenHasher {
public:
    explicit TokenHasher(int chunk_size) : chunk_size_(chunk_size) {}

    // 计算 token 序列的所有 chunk 键
    // 返回: 每个 chunk 一个 CacheKey, 共 ceil(num_tokens / chunk_size) 个
    std::vector<CacheKey> compute_keys(const int* tokens, int num_tokens) const {
        std::vector<CacheKey> keys;
        uint64_t prefix_hash = 0;  // 初始种子

        int offset = 0;
        while (offset < num_tokens) {
            int chunk_len = std::min(chunk_size_, num_tokens - offset);

            // 先混入前缀哈希, 再混入 chunk 内容
            uint64_t h = fnv1a_hash(&prefix_hash, sizeof(prefix_hash));
            h = fnv1a_hash(tokens + offset, chunk_len * sizeof(int), h);

            keys.push_back({h});
            prefix_hash = h;
            offset += chunk_len;
        }

        return keys;
    }

    // 计算整个前缀的最终哈希 (最后一个 chunk 的 key)
    CacheKey compute_prefix_key(const int* tokens, int num_tokens) const {
        auto keys = compute_keys(tokens, num_tokens);
        if (keys.empty()) return {0};
        return keys.back();
    }

    int get_chunk_size() const { return chunk_size_; }

private:
    int chunk_size_;
};

} // namespace cache
} // namespace qwen_thor
