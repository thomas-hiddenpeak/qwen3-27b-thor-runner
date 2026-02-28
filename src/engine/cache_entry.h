// KV Cache Offload — 缓存条目 (CacheEntry)
// 存储一个前缀的完整 KV + SSM/Conv 状态, 支持 store/retrieve
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// CacheEntry: 一个已缓存前缀的所有数据
//
// 包含:
//   1. KV 数据:  所有 full attention 层的 K/V tensor
//      布局: [num_full_attn_layers, 2(K/V), num_tokens, num_kv_heads, head_dim] BF16
//      = 16 * 2 * T * 4 * 256 * 2 bytes  (T=token数)
//      例: T=1024 → 64 MB
//
//   2. SSM 状态:  处理完所有 token 后的 DeltaNet SSM 状态快照
//      布局: [num_linear_attn_layers][nkh * kd * v_per_kh] FP32
//      = 48 * 3 MB ≈ 144 MB (固定, 与 T 无关)
//
//   3. Conv 状态: 处理完所有 token 后的 Conv1d 状态快照
//      布局: [num_linear_attn_layers][in_qkv * (conv_k-1)] BF16
//      = 48 * 60 KB ≈ 2.88 MB (固定, 与 T 无关)
//
// 所有 buffer 均分配在 host 侧 (malloc), 通过 cudaMemcpy 与 device 交互.
// ---------------------------------------------------------------------------
struct CacheEntry {
    // 前缀 token 序列 (用于校验)
    std::vector<int> tokens;
    int num_tokens = 0;

    // KV data (host memory, flat layout)
    std::vector<uint8_t> kv_data;

    // SSM state snapshot (host memory)
    // 空 vector = 未缓存 SSM
    std::vector<uint8_t> ssm_data;

    // Conv state snapshot (host memory)
    std::vector<uint8_t> conv_data;

    // 总字节数
    size_t total_bytes() const {
        return kv_data.size() + ssm_data.size() + conv_data.size();
    }

    bool has_ssm_state() const { return !ssm_data.empty(); }
};

using CacheEntryPtr = std::shared_ptr<CacheEntry>;

} // namespace cache
} // namespace qwen_thor
