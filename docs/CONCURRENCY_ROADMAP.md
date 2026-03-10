# 并发架构演进计划

> 目标: 支持两极化推理场景 — 单请求超长上下文 + 32-64 并发共享 system prompt

## 背景与战略约束

### 两极化场景
1. **单请求超长上下文** (100K+): 当前已基本具备 (Streaming Attention + SSD KV eviction + chunked prefill)
2. **32-64 并发 + 10K-20K 共享 system prompt**: 需要新架构能力, 核心要求 "能共享, 不能窜"

### KV + SSD 策略
- GPU KV: 做**热缓存** — 共享 prefix blocks (CoW 只读) + 每请求 decode 增量
- SSD: 做**冷存储 + 初始化源** — prefix cache 大方给 (50-100 GB), 不怕浪费
- Prefix cache 命中后, SSD → GPU 一次性 inject, 之后 decode 全在 GPU

### 稳定性红线
- Per-layer `cudaStreamSynchronize` 不可移除 (UMA 并发保护)
- `loaders_.clear()` 不可移除 (防双份权重 OOM)
- `max_chunk_size ≤ 4096` (CUTLASS TMA 描述符限制)
- 内存分配失败 → 拒绝/swap-out, 绝不 OOM

## 当前架构基线

### 已有能力
- 请求级隔离: 每请求独立 `block_table` + `ssm_slot` + `block_tracker` (`RequestCacheState`)
- KV 分页管理: `KVCacheManager` 简单 free list + mutex
- Prefix cache: `CacheEngine` + `DiskBackend` + `TokenHasher` (FNV-1a), SSD 增量存储
- SSD swap: `KVSwapper` 请求级 KV+SSM/Conv 整体换出/换入
- Streaming attention: GPU+SSD hybrid (超长上下文)
- `forward_decode` API: 已支持 `batch_size > 1` (stride-based SSM 索引), 但 engine 始终传 1

### 关键缺口

| 优先级 | 缺口 | 当前状态 | 影响 |
|--------|------|----------|------|
| P0 → Phase 1 | SSM slot 硬编码 8 | `MAX_CACHE_SSM_SLOTS = 8` | 最多 8 并发 |
| P1 → Phase 2 | batch decode | `step()` 只处理 `active_requests[0]` | N 请求串行 = N× 延迟 |
| P2 → Phase 3 | CoW KV 共享 | `KVCacheManager` 无 ref counting | N×20K 各自一份, 内存爆 |
| P3 → Phase 4 | 并发调度 | FIFO 串行拉取 | 无抢占/公平调度 |

## 内存规划 (27B BF16)

```
128 GB 总物理内存 (LPDDR5X)
 - 51.2 GB  模型权重 (BF16)
 -  5.0 GB  系统 + workspace + buffers
 -  N × 75 MB  SSM/Conv pool (SSM 72 MB + Conv 2.88 MB per slot)
 = 可用于 KV cache + 其他
```

| 并发数 N | SSM pool | KV 可用 (approx) | 说明 |
|----------|----------|---------|------|
| 8 (当前) | 0.6 GB | ~71 GB | 足够单请求 100K+ |
| 32 | 2.4 GB | ~69 GB | decode-only ~4 GB, 剩余做 prefix 共享 |
| 64 | 4.8 GB | ~67 GB | 需要 CoW, 否则 20K×64×64KB = 80 GB 爆 |

Per-token KV 大小: 64 KB (16 full attn layers × 4 KV heads × 256 dim × 2 bytes × 2 K/V)

### GPU vs SSD 功能分工

```
GPU KV (cudaMalloc, 带宽 ~220 GB/s):
├── 共享 prefix blocks (CoW 只读, 20K tokens = 1.25 GB, 一份)
├── 每请求 decode 增量 blocks (2K tokens × 64 KB = 128 MB/请求)
└── 预算: ~8-16 GB (视并发数动态调整)

SSD (NVMe, 顺序 ~2 GB/s):
├── Prefix cache: 所有唯一 system prompt 的 KV+SSM
├── Swap store: 被驱逐请求的完整 KV + SSM/Conv
├── Streaming: 超长上下文历史 blocks
└── 预算: 50-100 GB (不限, 给够)
```

## 实施计划

### Phase 1: SSM Slot 可配置化 (风险: 低)

**目标**: 解除 8 并发硬编码限制, 支持最多 64 并发

**改动范围**:
- `src/engine/cache_manager.h`: `MAX_CACHE_SSM_SLOTS` → 从 `CacheConfig` 读取, 默认 64
- `src/engine/cache_config.h`: 新增 `max_ssm_slots` 配置项
- `src/engine/cache_manager.cpp`: 池初始化使用配置值
- 配置文件: 新增 `max_ssm_slots=64` (可覆盖)

**内存影响**: 64 slots × 75 MB = 4.8 GB (在预算内, 71.8 GB → 67 GB)

**验证**:
- 现有测试通过 (默认值兼容)
- 编译通过, 无内存泄漏

**依赖**: 无

---

### Phase 2: Batch Decode (风险: 中)

**目标**: `step()` 同时处理多个 decode 请求, 权重只读一次服务 N 个 token

**改动范围**:
- `src/engine/engine.cpp`: `step()` 从只处理 `active_requests[0]` → 收集所有 decode-ready 请求
  - 新请求仍然串行 prefill (不变)
  - 多个已 prefill 完成的请求同时 batch decode
- `src/engine/model.h/cpp`: `forward_decode` 已支持 `batch_size` 参数, 需验证 N>1 路径
- `src/engine/paged_attention.cu`: split-K decode attention 需支持 batch, 每请求不同 block_table
- `src/engine/light_ops.cu`: GPU sampling 需 batch 化 (N 个 logits → N 个 token)

**关键设计**:
```
step() 逻辑:
1. 分离 prefill 请求 (generated_tokens 为空) 和 decode 请求
2. 如果有 prefill 请求: 处理一个 prefill (不变)
3. 收集所有 decode 请求的 token + block_table + ssm_slot
4. 调用 forward_decode(batch_size=N, ...)
5. dispatch 每个请求的结果 token
```

**性能收益**: N 请求共读一次权重 → decode 吞吐接近 N× (权重带宽瓶颈下)

**验证**:
- 多请求并发: 每请求结果与单请求串行一致
- MTP verify 路径: batch_size > 1 × T > 1 组合的正确性
- 内存不增长 (batch workspace 预分配)

**依赖**: Phase 1 (需要足够 SSM slots)

---

### Phase 3: CoW KV 共享 (风险: 高) — ✅ 已实现

**目标**: 相同 system prompt 的多个请求共享 prefix KV blocks, 写时复制

**实现**:
- `src/engine/paged_attention.h/cpp`: `KVCacheManager` ref counting
  - `ref_count_[]` per-block 引用计数 (0=空闲, 1=独占, >1=共享)
  - `allocate_blocks()` → ref_count = 1
  - `share_blocks(block_ids)` → ref_count++ (用于前缀共享)
  - `free_blocks()` → ref_count--, 仅 ref_count 降至 0 时回收
  - `get_ref_count(block_id)` → 查询
- `src/engine/cache_manager.h/cpp`:
  - `SharedGPUPrefix` struct: block_ids + num_tokens + active_users
  - `gpu_prefix_registry_` (hash → SharedGPUPrefix): GPU 上已驻留的共享前缀
  - `register_gpu_prefix()` 注册 / `try_share_gpu_prefix()` 共享 / `release_gpu_prefix()` 释放
  - `restore_ssm_only()`: GPU prefix sharing 时仅从 SSD 恢复 SSM/Conv (跳过 KV I/O)
  - `compute_prefix_hash()`: 使用 TokenHasher 计算 chunk-aligned 前缀哈希
- `src/engine/cache_engine.h/cpp`: `restore_ssm_only()` — SSD 读取 SSM/Conv 状態のみ
- `src/engine/engine.cpp`:
  - prefill 前: 先尝试 GPU prefix sharing (零 SSD KV I/O), 回退到 SSD restore
  - prefill 后: register_gpu_prefix 注册前缀 blocks 为可共享
  - cleanup: release_gpu_prefix 释放引用
  - SSM 恢复失败时自动回滚 KV sharing

**隔离保证 ("不能窜")**:
- 共享 prefix blocks 是**只读**的 (所有请求的前 N tokens 相同, KV 值相同)
- decode 新 token 写入的 block 一定是 ref_count=1 的独占 block (新分配)
- prefix 边界始终对齐 block_size (chunk_size=256, block_size=16 → 16 blocks/chunk)
- SSM/Conv slot: 每请求始终独立, 从 SSD prefix cache 恢复各自终态

**内存收益**: N×20K prefix 从 N×1.25 GB → 1×1.25 GB + N×(decode_only)

---

### Phase 4: 动态调度与抢占 (风险: 中) — ✅ 已实现

**目标**: decode 优先于 prefill, 内存压力时智能 swap-out

**实现**:
- `src/engine/engine.h`: `RequestContext` 增加调度字段
  - `last_active_step`: 最近活跃步 (用于 LRU swap-out)
  - `prefill_chunk_idx`, `prefill_cached_tokens`, `prefill_ssd_cursor`: chunk 级 prefill 进度
  - `step_counter_`: InferenceEngine 全局步数
- `src/engine/engine.cpp`:
  - **LRU swap-out**: `try_swap_out_victim()` 从"最大 blocks" → "最久未活跃" (最小 `last_active_step`)
  - **Decode-first 调度**: 请求选择循环优先 decode/streaming/swap-in, 其次 prefill
  - **Chunk-level 抢占**: 有 decode 请求等待时, prefill 每次只处理 1 个 chunk 然后 yield
    - 首次进入: cache lookup → 保存 `prefill_cached_tokens` → `prefill_chunk_idx = 0`
    - 后续进入: 从保存的 `prefill_chunk_idx` 继续
    - 所有 chunk 完成: 执行 post-processing (store prefix, lm_head, sample)
  - **Admission control**: `inference_loop` 内存压力时暂停接入新请求
    - 阈值: `free_blocks < max_chunk_size/16` (128 blocks) 或 SSM slots 耗尽
    - 不丢弃请求, 而是保留在 IPC 队列中等候

**Decode 延迟改善**: 20K prompt (10 chunks) 期间, decode 等待从 ~2-5s → ~200-500ms/chunk

## 风险与回退

| Phase | 风险等级 | 主要风险 | 回退策略 |
|-------|---------|---------|---------|
| 1 | 低 | 内存浪费 (预分配过多 SSM) | 减小 max_ssm_slots 配置 |
| 2 | 中 | batch decode 正确性 | 回退到 batch_size=1 串行 |
| 3 | 高 | CoW race condition / 引用泄漏 | 关闭 CoW, 回退到独立 allocate |
| 4 | 中 | 调度导致饿死或延迟不稳 | 回退到 FIFO |

每个 Phase 独立可交付, 有独立 commit, 可单独回退。

---

## 性能验证结果 (2026-03-10)

### 发现并修复的 Bug

1. **B=1 阻塞 prefill**: `step()` 路由中, B=1 走 MTP 路径时不做 pending prefill,
   导致所有请求串行处理。修复: B=1 且有 pending prefill 时走 batch decode (牺牲 MTP 换取快速 ramp-up)。
2. **IPC 队列容量不足**: 请求队列容量 8 (可用 7), B=8 时第 8 个请求失败。修复: 容量 8→128。

### GEMV→CUTLASS 阈值优化

发现 `gemv_multirow_kernel_scattered<16>` 有严重寄存器压力 (每线程 16 FMA 累加器,
低占用率), 导致 B=16 反而比 B=8 慢。将 CUTLASS GEMM 接管阈值从 M≥17 降到 M≥9,
B=16 吞吐 +87%。

### Scaling 曲线

| Batch Size | tok/s | Scaling | GEMM 路径 | per_req_tps | TTFT (avg) |
|-----------|-------|---------|----------|-------------|------------|
| 1 (MTP)   | 12.1  | 1.0×    | GEMV<4>  | 12.1        | 233ms      |
| 4         | 15.5  | 1.28×   | GEMV<4>  | 4.1         | 903ms      |
| 8         | 19.6  | 1.62×   | GEMV<8>  | 2.7         | 1836ms     |
| 16        | 29.2  | 2.41×   | CUTLASS  | 2.1         | 4029ms     |
| 32        | 44.3  | 3.64×   | CUTLASS  | 2.0         | ~10s       |
| 64        | 69.2  | 6.09×   | CUTLASS  | 1.8         | ~16s       |

### 效率分析

理论极限: 51.2 GB 权重 / 220 GB/s 实测带宽 = 233ms/step
- B=64 理论: 64 tok / 233ms = 275 tok/s
- B=64 实测: 69.2 tok/s (25% of theoretical)
- 差距主因: 64 ×串行 prefill ramp-up (~32s of 46s wall time)

### 下一步优化方向

| 优先级 | 方向 | 预期收益 | 复杂度 |
|--------|------|----------|--------|
| P0 | 并行/批量 Prefill | 消除 ramp-up, B=64 可达 100+ tok/s | 中 |
| P1 | Head-Group Batch Attention | 6 Q heads per KV group 合并, 减少 FA 时间 | 中 |
| P2 | Batch MTP (B=4-8 开启投机) | 小 batch +50-80% | 高 |
| P3 | SSM State 压缩 | 减少每 slot 75 MB, 支持更多并发 | 中 |

### Batch MTP 可行性分析

**结论: 暂不实现, 标记为 future work**

难点:
- SSM 检查点内存: B × N × 48 层, B=8 需 1.7 GB, B=64 需 13.8 GB
- 异步验证: 各请求 accept count 不同, 无法 batch rollback
- forward 重构: B 请求各 T 个 token, 需拆分为 per-request block/position 映射
- 对目标场景 (B=32-64) 收益有限 (已接近带宽极限)
