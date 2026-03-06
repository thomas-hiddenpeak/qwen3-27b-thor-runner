# Cache 子系统重构计划

## 1. 现状分析

### 1.1 现有组件 (6 个独立实现)

| 组件 | 文件 | 职责 | 管理维度 |
|------|------|------|----------|
| KVCacheManager | paged_attention.h/cpp | GPU KV block pool 的 alloc/free | block 级 |
| BlockTracker | block_tracker.h | 追踪单请求的 logical block 在 GPU 还是 SSD | block 级 |
| BlockSSDStore | block_tracker.h | 按层按块的 SSD pread/pwrite | block 级 |
| CacheEngine | cache_engine.h/cpp | Prefix 缓存 (token chunk hash → KV+SSM 快照) | chunk(256t) 级 |
| KVSwapper | kv_swapper.h/cpp | 请求级 KV+SSM 整体换出/换入 | request 级 |
| DiskBackend | disk_backend.h/cpp | CacheEngine 的 SSD 存储后端 (LRU) | file 级 |

加上 engine.cpp 本身有 **~500 行** 内联缓存管理逻辑。

### 1.2 核心架构问题

1. **三重 block 管控**: KVCacheManager / BlockTracker / BlockSSDStore 都参与 block 生命周期，ownership 不清晰
2. **双状态机冲突**: `ctx->block_tracker` (per-block GPU/SSD 位置) vs `ctx->is_swapped` (整请求换出) 可能不一致
3. **SSM 状态碎片化**: 池化 pool、CacheEntry 快照、KVSwapper 换出文件、BlockSSDStore 完全不管 SSM —— 四处分布
4. **三种 SSD 目录各自为政**: `cache_dir/` (prefix)、`cache_dir/kv_swap/` (swap)、`cache_dir/block_store/` (streaming) 无统一管理
5. **context_len 计算碎片化**: 至少 4 种来源，downstream 必须自行拼凑
6. **engine.cpp 违反 SRP**: 直接做 block eviction 决策 + SSD I/O 调度 + streaming attention 组装

### 1.3 性能问题

- Streaming attention SSD I/O: 每层串行 pread → ~100s/chunk (3856 SSD blocks)
- 无 prefetch / 无 pipeline: load_blocks_for_layer 是同步阻塞
- 无 block 级 LRU: evict 总是驱逐最旧块，无热度追踪

## 2. 参考架构总结

### vLLM v1
- **KVCacheManager** → **KVCacheCoordinator** → **BlockPool**
- BlockPool: 统一的物理 block 池，支持 hash-based prefix sharing + LRU eviction
- Coordinator: 跨 KV cache group 协调 (full attn / sliding window / cross attn)
- 关键设计: `KVCacheBlock` 有 `ref_cnt`，共享前缀只存一份
- 混合注意力模型 (Qwen3.5 有 FullAttn + LinearAttn) 需要类似 HybridKVCacheCoordinator

### SGLang
- 两级内存池: `ReqToTokenPool` (请求→token位置映射) + `TokenToKVPoolAllocator` (token→物理 KV)
- `HybridReqToTokenPool`: 同时管理 KV 和 Mamba/SSM pool
- `MambaPool`: SSM/Conv 状态与 KV 完全独立管理，有自己的 alloc/free/copy_from
- 关键启示: **SSM 状态作为独立池，与 KV 解耦但由同一 ReqToTokenPool 协调一致分配**

### LMCache
- **LMCBackendInterface**: 抽象接口 `put(key, chunk)` / `get(key)` / `contains(key)`
- **LMCHybridBackend**: 组合 local (GPU/CPU) + remote (Redis/SSD)，write-through / read-through
- 关键思想: **存储后端完全透明，上层不关心数据在哪一层**

## 3. 重构方案

### 3.1 整体架构

```
┌────────────────────────────────────┐
│           InferenceEngine          │
│   (只调 CacheManager 的公共 API)    │
└──────────────┬─────────────────────┘
               │
┌──────────────▼─────────────────────┐
│         CacheManager               │  ← 新增: 统一入口
│  - allocate / free / evict         │
│  - prefix lookup / store           │
│  - swap_out / swap_in              │
│  - streaming attention setup       │
│  - SSM pool 管理                   │
│  - context_len 单一来源            │
└──────────────┬─────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼───────┐     ┌───────▼────────┐
│ GPU Tier  │     │   SSD Tier     │
│ BlockPool │     │ BlockStore     │
│ (alloc/   │     │ (evict/load/   │
│  free/    │     │  prefix/swap)  │
│  KV ptrs) │     │                │
└───────────┘     └────────────────┘
```

### 3.2 核心新类

#### `CacheManager` (统一入口)

```cpp
class CacheManager {
public:
    struct Config {
        float kv_cache_budget_gb;     // GPU KV 预算
        int block_size;               // 16 tokens/block
        int max_ssm_slots;            // SSM 池大小
        std::string cache_dir;        // SSD 缓存目录
        size_t ssd_cache_max_bytes;   // SSD 缓存上限
        bool enable_prefix_cache;     // 是否启用前缀缓存
        int prefix_chunk_size;        // prefix hash 粒度 (256)
        int staging_num_blocks;       // GPU staging buffer blocks
    };

    // ---- 请求生命周期 ----
    struct RequestCacheState {
        std::vector<int> gpu_block_table;   // GPU 上的 physical block IDs
        std::vector<int> ssd_logical_ids;   // SSD 上的 logical block indices
        int context_len;                    // 统一来源: total tokens
        int gpu_tokens;                     // = gpu_blocks × block_size
        int ssd_tokens;                     // = ssd_blocks × block_size
        int ssm_slot;                       // SSM pool slot
        bool has_ssd_blocks;                // 快速判断
    };

    // 为新请求分配 KV blocks + SSM slot
    RequestCacheState allocate_request(uint64_t req_id, int num_tokens);

    // 扩展 blocks (decode / chunked prefill 新 tokens)
    bool extend_blocks(uint64_t req_id, int new_tokens, RequestCacheState& state);

    // 释放请求所有资源
    void free_request(uint64_t req_id, RequestCacheState& state);

    // ---- Eviction (GPU → SSD) ----
    // 将请求旧 blocks 驱逐到 SSD, 释放 GPU blocks
    int evict_blocks(uint64_t req_id, int num_blocks_to_free, RequestCacheState& state);

    // 请求级整体换出 (multi-request scheduling)
    bool swap_out(uint64_t req_id, RequestCacheState& state,
                  const SSMState& ssm, const ConvState& conv);

    // 请求级换入
    bool swap_in(uint64_t req_id, RequestCacheState& state,
                 SSMState& ssm, ConvState& conv);

    // ---- Prefix Cache ----
    int lookup_prefix(const int* tokens, int num_tokens);
    int restore_prefix(const int* tokens, int num_tokens,
                       RequestCacheState& state,
                       SSMState& ssm, ConvState& conv);
    void store_prefix(const int* tokens, int num_tokens,
                      const RequestCacheState& state,
                      const SSMState& ssm, const ConvState& conv);

    // ---- Streaming Attention 支持 ----
    // 为有 SSD blocks 的请求配置 StreamingAttnCtx
    StreamingAttnCtx build_streaming_ctx(uint64_t req_id,
                                          const RequestCacheState& state,
                                          int num_tokens);

    // 每层加载 SSD blocks 到 staging buffer (streaming forward 回调)
    int load_ssd_blocks_for_layer(uint64_t req_id, int layer_idx,
                                  int batch_start, int batch_count);

    // ---- SSM Pool ----
    int allocate_ssm_slot();
    void free_ssm_slot(int slot);
    __nv_bfloat16* get_ssm_state(int slot, int layer);
    __nv_bfloat16* get_conv_state(int slot, int layer);

    // ---- 查询 ----
    int num_free_gpu_blocks() const;
    int total_gpu_blocks() const;
    const __nv_bfloat16* get_layer_k_cache(int layer) const;
    const __nv_bfloat16* get_layer_v_cache(int layer) const;
    __nv_bfloat16* get_layer_k_cache_mut(int layer);
    __nv_bfloat16* get_layer_v_cache_mut(int layer);

    // ---- 统计 ----
    struct Stats { /* hit/miss/eviction/swap/SSD I/O 统计 */ };
    Stats get_stats() const;

private:
    // 内部组件 (engine 不直接访问)
    std::unique_ptr<GPUBlockPool> gpu_pool_;     // 原 KVCacheManager，精简
    std::unique_ptr<SSDBlockStore> ssd_store_;   // 统一 BlockSSDStore + DiskBackend + KVSwapper
    std::unique_ptr<PrefixCache> prefix_cache_;  // 原 CacheEngine，简化接口

    // SSM/Conv pool
    __nv_bfloat16* ssm_pool_base_;
    __nv_bfloat16* conv_pool_base_;
    std::vector<std::vector<__nv_bfloat16*>> ssm_states_; // [slot][layer]
    std::vector<std::vector<__nv_bfloat16*>> conv_states_;
    std::vector<int> free_ssm_slots_;

    // Staging buffers (streaming attention)
    __nv_bfloat16* d_staging_k_;
    __nv_bfloat16* d_staging_v_;
    int staging_capacity_;  // blocks

    // Per-request block tracking (替代散布在 RequestContext 中的 block_tracker)
    struct RequestBlockState {
        std::vector<int> gpu_blocks;        // physical block IDs on GPU
        std::vector<int> ssd_logical_ids;   // logical block indices on SSD
        int context_len;
        bool is_swapped;
    };
    std::unordered_map<uint64_t, RequestBlockState> request_states_;

    // 模型参数 (构造时传入)
    int num_layers_;          // full_attn 层数 (16)
    int num_kv_heads_;
    int head_dim_;
    int block_size_;          // 16
    int num_linear_layers_;   // 48
    size_t ssm_size_per_layer_;
    size_t conv_size_per_layer_;
};
```

### 3.3 内部组件

#### `GPUBlockPool` (替代 KVCacheManager)

保留核心功能，去掉 engine 不应该直接调用的 API：
- `allocate(n) → vector<int>`
- `free(vector<int>)`
- `num_free() → int`
- `get_k_cache(layer)` / `get_v_cache(layer)` → 物理指针

```cpp
class GPUBlockPool {
    // 与现有 KVCacheManager 基本相同
    // 但 ONLY CacheManager 可访问
};
```

#### `SSDBlockStore` (统一 3 种 SSD 操作)

合并 BlockSSDStore + KVSwapper + 部分 DiskBackend：
```cpp
class SSDBlockStore {
public:
    // Block 级 I/O (streaming attention 用)
    void evict_blocks(uint64_t req_id, const std::vector<int>& logical_indices,
                      const std::vector<int>& physical_blocks,
                      const GPUBlockPool& pool);
    int load_blocks_for_layer(uint64_t req_id, int layer_idx,
                              const std::vector<int>& ssd_logical_ids,
                              int batch_start, int batch_count,
                              __nv_bfloat16* d_staging_k,
                              __nv_bfloat16* d_staging_v);
    void remove_request(uint64_t req_id);

    // Request 级 I/O (swap 用)
    void swap_out_request(uint64_t req_id, const std::vector<int>& block_table,
                          int context_len, const GPUBlockPool& pool,
                          const SSMState& ssm, const ConvState& conv);
    bool swap_in_request(uint64_t req_id, std::vector<int>& new_blocks,
                         int& context_len, GPUBlockPool& pool,
                         SSMState& ssm, ConvState& conv);

private:
    std::string store_dir_;
    void* host_staging_;      // 4K-aligned host buffer
    size_t staging_size_;
    int per_block_kv_bytes_;  // per layer
    int num_layers_;
};
```

#### `PrefixCache` (精简 CacheEngine)

```cpp
class PrefixCache {
public:
    int lookup(const int* tokens, int n) const;
    int restore(const int* tokens, int n, GPUBlockPool& pool,
                std::vector<int>& out_blocks,
                SSMState& ssm, ConvState& conv,
                cudaStream_t stream);
    void store(const int* tokens, int n,
               const GPUBlockPool& pool,
               const std::vector<int>& block_table,
               const SSMState& ssm, const ConvState& conv,
               cudaStream_t stream);
    void record_stats(int prompt_tokens, int restored, int computed);
};
```

### 3.4 RequestContext 简化

**Before** (当前):
```cpp
struct RequestContext {
    std::vector<int> block_table;          // ← 移除
    int context_len;                        // ← 移除
    cache::BlockTracker block_tracker;      // ← 移除
    bool uses_ssd_blocks;                   // ← 移除
    std::vector<__nv_bfloat16*> ssm_states; // ← 移除
    std::vector<__nv_bfloat16*> conv_states;// ← 移除
    int ssm_slot;                           // ← 移除
    bool is_swapped;                        // ← 移除
    // ... 大量 cache 状态
};
```

**After** (重构后):
```cpp
struct RequestContext {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int max_new_tokens;

    // 采样参数
    SamplingParams sampling;

    // Cache 状态: 通过 CacheManager 管理, 只存一个句柄
    CacheManager::RequestCacheState cache_state;

    // MTP
    MTPState mtp_state;

    // 多模态
    std::vector<core::ProcessedImage> processed_images;

    bool is_finished = false;

    // Helper: 快速访问
    int context_len() const { return cache_state.context_len; }
    bool has_ssd_blocks() const { return cache_state.has_ssd_blocks; }
    const std::vector<int>& block_table() const { return cache_state.gpu_block_table; }
    int ssm_slot() const { return cache_state.ssm_slot; }
};
```

### 3.5 Engine.cpp 变化

**Before**: ~500 行内联缓存管理 (分散在 process_prefill / process_decode / cleanup / init)
**After**: 通过 `cache_manager_->xxx()` 调用，engine.cpp 中缓存相关代码预计 <100 行

关键简化点:
1. `InferenceEngine 构造函数`: `cache_manager_ = make_unique<CacheManager>(config)` 替代 5 个独立 make_unique
2. `process_prefill`: `cache_manager_->allocate_request()` + `cache_manager_->extend_blocks()` + `cache_manager_->evict_blocks()` 替代内联 block 管理
3. `process_decode streaming`: `auto streaming_ctx = cache_manager_->build_streaming_ctx()` 替代 ~50 行手动组装
4. `cleanup_request`: `cache_manager_->free_request()` 替代分散的 free_blocks + remove_ssd + remove_swap
5. `try_swap_out_victim`: 委托给 `cache_manager_->swap_out()`

## 4. 实施计划

### Phase 1: CacheManager 骨架 (新文件, 不改 engine)
- [ ] 创建 `src/engine/cache_manager.h` — 定义 CacheManager 类 + RequestCacheState + 所有公共接口
- [ ] 创建 `src/engine/cache_manager.cpp` — 实现构造/析构/SSM pool
- [ ] 保留现有组件作为 private 成员 (先包装, 后替换)

### Phase 2: 实现 CacheManager 核心 API
- [ ] `allocate_request` / `free_request` — 包装 KVCacheManager::allocate/free + SSM pool
- [ ] `extend_blocks` — 包装 allocate_blocks + block_tracker
- [ ] `evict_blocks` — 包装 BlockSSDStore::evict + BlockTracker::mark_evicted + KVCacheManager::free
- [ ] `swap_out` / `swap_in` — 包装 KVSwapper
- [ ] `build_streaming_ctx` — 移动 engine.cpp ~50 行 streaming context 组装
- [ ] `load_ssd_blocks_for_layer` — 包装 BlockSSDStore::load_blocks_for_layer

### Phase 3: 集成到 Engine
- [ ] 替换 engine.h 中的 5 个独立成员为 1 个 `cache_manager_`
- [ ] 修改 RequestContext: 使用 `cache_state` 替代散布字段
- [ ] 修改 engine.cpp 构造函数
- [ ] 修改 process_prefill: 使用 CacheManager API
- [ ] 修改 process_decode: 使用 CacheManager API
- [ ] 修改 cleanup / swap 逻辑

### Phase 4: 测试 & 基准
- [ ] 编译通过
- [ ] 现有单元测试全部通过 (`./build/qwen3-27b-thor test`)
- [ ] serve 模式单轮对话正确
- [ ] 长上下文 SSD streaming 测试通过
- [ ] 性能基准不退化 (decode tok/s, TTFT)

## 5. 风险与约束

### 必须保留的硬约束
- `max_chunk_size_ = 256` (SMMU 稳定性)
- `cudaStreamSynchronize` per-layer (统一内存安全)
- 权重用 `cudaMalloc`, staging 用 `cudaMalloc`, host staging 需 4K 对齐
- `loaders_.clear()` 防双份权重

### 不改动的组件
- paged_attention.cu (kernel 代码不变)
- streaming_attention.cu (kernel 代码不变)
- cache_kernels.cu (extract/inject kernel 不变)
- model.h/cpp (forward path 不变)
- layer.cu (层实现不变)

### 迁移策略
采用**包装递进**策略，而非一次性重写:
1. Phase 1: CacheManager 包装现有组件，engine 仍直接调用原组件
2. Phase 2: CacheManager 内部调用原组件，API 完备
3. Phase 3: engine 切换到 CacheManager API，原组件降为 private
4. Phase 4: (未来) 可选：重写内部组件

这样每个 Phase 都可以编译测试，不会出现大面积断裂。
