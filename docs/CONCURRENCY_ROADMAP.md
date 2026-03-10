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

### Phase 3: CoW KV 共享 (风险: 高)

**目标**: 相同 system prompt 的多个请求共享 prefix KV blocks, 写时复制

**改动范围**:
- `src/engine/paged_attention.h`: `KVCacheManager` 增加 `ref_count_[]` (per-block 原子引用计数)
  - `allocate_blocks()` → ref_count = 1
  - `share_blocks(block_ids)` → ref_count++
  - `release_block(block_id)` → ref_count--, 为 0 时回收到 free list
  - `copy_on_write(block_id)` → 如果 ref_count > 1: allocate 新 block, 拷贝数据, 旧 block ref_count--
- `src/engine/cache_manager.h/cpp`: 
  - 识别相同 prefix (通过 TokenHasher 已有能力)
  - 首次: SSD → GPU inject 完整 prefix blocks
  - 后续: 直接 share_blocks() 引用已有 GPU blocks
- `src/engine/engine.cpp`:
  - decode 时 `write_kv_cache` 前: 检查当前 block 是否 shared → CoW
  - 新请求匹配已有 prefix → 跳过 prefill, 共享 + 恢复 SSM

**隔离保证 ("不能窜")**:
- 共享 prefix blocks 是**只读**的 (所有请求的前 20K tokens 相同, KV 值相同)
- decode 新 token 写入的 block 一定是 ref_count=1 的独占 block
- CoW 触发条件: `write_kv_cache` 的目标 block ref_count > 1
- SSM/Conv slot: 每请求始终独立, 从 prefix cache 恢复各自终态

**内存收益**: N×20K prefix 从 N×1.25 GB → 1×1.25 GB + N×(decode_only)

**验证**:
- 正确性: 两个请求共享 prefix, decode 不同 token, 输出完全独立
- 引用计数: 请求结束后 ref_count 正确递减, 无泄漏
- CoW: 写入时正确触发拷贝, 原 block 不被修改
- 压力: 64 请求同时共享 + 并发结束, 无 race condition

**依赖**: Phase 2 (共享才有意义需要先能并发 decode)

---

### Phase 4: 动态调度与抢占 (风险: 中)

**目标**: decode 优先于 prefill, 内存压力时智能 swap-out

**改动范围**:
- `src/engine/engine.cpp`: `inference_loop` 调度逻辑
  - Decode 请求优先 (Short-Job-First: 减少平均延迟)
  - 新请求 prefill 与已有 decode 交错 (不饿死新请求)
  - 内存压力阈值: free_blocks < threshold → 暂停接入新请求
- Swap-out 策略: LRU (最久未产出 token 的请求优先换出)
- 抢占支持: prefill 可被 decode 中断 (在 chunk 边界)

**验证**:
- 混合负载: 持续新请求 + 并发 decode, 无饿死, 无 OOM
- Swap 往返: 换出→换入后推理结果正确
- 性能: 平均 TTFT 和 decode 延迟在可接受范围

**依赖**: Phase 2, Phase 3

## 风险与回退

| Phase | 风险等级 | 主要风险 | 回退策略 |
|-------|---------|---------|---------|
| 1 | 低 | 内存浪费 (预分配过多 SSM) | 减小 max_ssm_slots 配置 |
| 2 | 中 | batch decode 正确性 | 回退到 batch_size=1 串行 |
| 3 | 高 | CoW race condition / 引用泄漏 | 关闭 CoW, 回退到独立 allocate |
| 4 | 中 | 调度导致饿死或延迟不稳 | 回退到 FIFO |

每个 Phase 独立可交付, 有独立 commit, 可单独回退。
