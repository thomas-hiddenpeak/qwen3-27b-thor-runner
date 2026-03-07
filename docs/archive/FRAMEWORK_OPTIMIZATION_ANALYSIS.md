# 现代 LLM 推理框架优化技术评估报告

## 目标系统概览

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA Jetson AGX Thor, SM110a Blackwell, 20 SM |
| 内存 | 128 GB LPDDR5X UMA, 273 GB/s 峰值, ~230 GB/s GEMV 实测 |
| 模型 | Qwen3.5-27B, 64 层 (48 Linear Attn + 16 Full Attn), BF16 |
| 权重总量 | ~51.2 GB BF16 |
| 当前性能 (B=1) | 218ms/step, 4.58 tok/s, 235 GB/s 有效带宽 (86% 利用率) |
| 当前性能 (B=32) | 267ms/step, 119.8 tok/s, 192 GB/s 权重带宽 |
| 当前性能 (B=64) | 319ms/step, 200.8 tok/s, 161 GB/s 权重带宽 |
| 瓶颈 (B=1) | **纯 DRAM 内存带宽** (Decode M=1 GEMV) |
| 瓶颈 (B≥32) | **GEMM 计算吞吐** (Tensor Core throughput) |
| 理论极限 (B=1) | 191.7ms → 5.22 tok/s (含 ~4ms non-GEMV 计算) |
| 差距 (B=1) | 27.5ms (12.5%), 来自 DRAM bank conflict 和访问模式 |

### 已有优化

- GEMV: float4 向量化, shared memory A 广播, warp shuffle reduce, scattered-tiled (K>8192)
- Dual GEMV: MLP gate+up 共享 A 输入
- QKV 权重合并 (FullAttn T=1), ZAB 权重合并 (LinearAttn T=1)
- Fused Add+RMSNorm, Fused Deinterleave+Q_RMSNorm, Fused Norm+SiLU+Gate
- DeltaNet SSM alpha/sigmoid(beta) 内联
- GPU Argmax (managed memory, 0.07ms)
- CUTLASS 4.4 SM100 TMA GEMM (Prefill)
- 868 kernel launches/step, CPU launch 完全被 GPU 掩藏 (0 pipeline bubbles)
- **连续批处理 (Continuous Batching)**: GEMV→GEMM 转换, 支持 B=1~256 并发
- **cudaDeviceScheduleBlockingSync**: CPU 空闲时 yield 而非 spin-wait
- **Batched Paged Attention**: 2D block_table [B, max_blocks], per-seq context_lens
- **Batched DeltaNet/Conv1d**: device pointer array 索引 per-request SSM/Conv 状态

---

## Phase 8 更新: 连续批处理实测与评估方法论修正

### 多并发基准测试结果

> **Phase 11 更新 (2026-02-27)**: 以下数据为最新 sweep_batch.sh 扫描结果, 包含 CUTLASS per-GEMM sync 移除和 CUDA Graph 优化。

| Batch | 延迟 (ms) | Forward (ms) | 吞吐 (tok/s) | Scaling | Efficiency | 权重BW (GB/s) | BW利用率 | 瓶颈类型 |
|------:|----------:|-------------:|-------------:|--------:|-----------:|--------------:|---------:|----------|
| 1     | 218.2     | ~208         | 4.58         | 1.00x   | 100.0%     | 234.8         | 86%      | **带宽** |
| 2     | 224.1     | ~214         | 8.92         | 1.95x   | 97.4%      | 228.6         | 84%      | 带宽     |
| 4     | 226.5     | ~216         | 17.66        | 3.86x   | 96.4%      | 226.3         | 83%      | 带宽     |
| 8     | 231.9     | ~222         | 34.50        | 7.53x   | 94.2%      | 221.0         | 81%      | 带宽     |
| 16    | 240.9     | ~231         | 66.42        | 14.50x  | 90.6%      | 212.7         | 78%      | 带宽+计算 |
| 32    | 267.1     | ~257         | 119.80       | 26.16x  | 81.7%      | 191.8         | 70%      | 带宽+计算 |
| 48    | 291.9     | ~282         | 164.46       | 35.91x  | 74.8%      | 175.6         | 64%      | 计算     |
| 64    | 318.7     | ~309         | 200.80       | 43.84x  | 68.5%      | 160.8         | 59%      | **计算** |

### 关键发现

1. **近线性 Scaling (B=1-8)**: Efficiency 94-100%, ITL 仅从 218→232ms (+6%)。系统完全 memory-bound, 权重只读一次服务 N 个 token
2. **Roofline 交叉 (B=16-32)**: MLP GEMM `[M, 5120] × [5120, 17408]` 在 M=28-32 时 compute ≈ bandwidth, 效率开始下降
3. **Compute-bound (B=48-64)**: 效率 69-75%, ITL 明显上升但吞吐仍线性增长。B=64 达 200.8 tok/s
4. **GEMM tile 效率**: B=2/4 时 M 需 pad 到 8 对齐, 少量 padding overhead; B≥8 后 CUTLASS tile 利用率高
5. **SSM 状态内存**: 每请求 ~147 MB (48层 × 3MB ssm + 60KB conv), B=64 消耗 9.4 GB

### 评估方法论修正

原文以 "单请求 decode M=1 纯带宽瓶颈" 为前提。**多并发服务场景下, 前提根本性改变**:

| 场景 | 瓶颈 | 有效优化方向 |
|------|------|-------------|
| B=1 decode | DRAM 带宽 | 减少权重读取量、提升 DRAM 利用率 |
| B=2~16 decode | 仍为带宽主导 | GEMM tile 选择、权重读取效率 |
| B=32~64 decode | 带宽+计算 | GEMM 计算优化、Tensor Core 利用率提升 |
| B≥128 decode | 计算主导 | Tensor Core 吞吐、GEMM tile tuning、减少非 GEMM 开销 |
| 多用户服务 | 调度+内存 | Chunked prefill、prefix caching、KV 内存管理 |

以下重新评估各技术的结论。

---

## 单请求 GEMV→GEMM 转换可行性分析

### 核心问题

batch=2 的 GEMM 读同样 51GB 权重却产出 2 个 token, 等效吞吐翻倍。能否在单请求 decode 中实现类似的 GEMV→GEMM 转换?

### 逐方向分析

| 方向 | 可行性 | 分析 |
|------|--------|------|
| **CUTLASS GEMM at M=1** | ❌ **更慢** | 实测 GEMV 234 GB/s vs GEMM (B=2) 190 GB/s。GEMM 的 tile 管理、TMA 初始化开销在 M=1 时无法被计算隐藏。 |
| **合并更多权重矩阵** | ✅ **已穷尽** | QKV merged (N=14336), ZAB merged (N=6240), gate+up dual GEMV。所有共享输入 A 的投影已合并。 |
| **跨层批处理** | ❌ **不可能** | Transformer 层间有严格的数据依赖 (residual + norm + state update)。 |
| **DeltaNet 内部并行** | ❌ **不可能** | SSM 递推 $S_t = \alpha S_{t-1} + \beta k_t v_t^T$ 是强序列依赖。 |
| **TMA 用于 GEMV** | ❌ **不适配** | TMA 硬件设计为 2D tile 批量加载到 shared memory, GEMV 的访问模式是 1D 流式列读取, TMA 开销大于收益。 |
| **投机解码** | 🚫 **禁止** | 项目规则明确禁止。 |

### 结论

> **单请求下 GEMV→GEMM 转换不可行。** GEMV 在 M=1 时的带宽效率 (234 GB/s, 86%) 已超过 GEMM (190 GB/s)。GEMM 的优势仅存在于 M>1 的权重复用场景。唯一突破单请求带宽墙的方式是多并发 batched decode, 这已经实现。

---

## 技术评估

### 评估方法论

对于**多并发服务场景**:
- B=1: **DRAM 带宽瓶颈** — 有效优化 = 减少 DRAM 传输或提高利用率
- B=2~64: **带宽主导, 计算渐增** — 有效优化 = GEMM tile 优化、减少非 GEMM 开销
- B≥128: **计算瓶颈** — 有效优化 = Tensor Core 吞吐、GEMM auto-tune
- 多用户：**调度与内存** — 有效优化 = Chunked prefill, prefix caching, KV 内存管理
- 时间估算公式: `时间节省(ms) = 字节节省(GB) / 有效带宽(234 GB/s)` (B=1)
- 时间估算公式: `时间节省(ms) = FLOP节省 / TFLOPS` (B≥128)

---

## 一、FlashInfer 技术

### 1. Cascade Attention / Tree Attention

**原理**: 将 KV cache 拆分为共享前缀 (shared prefix) 和独有后缀 (unique suffix)，多请求复用前缀的 attention 计算结果，避免重复 KV 读取。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | **多并发下 (B>1) 有共享前缀时有意义**。B=32 共享 system prompt → 32 个请求复用前缀 KV cache 读取。但 Paged Attention 在 decode 中仍远小于 GEMM 时间。 |
| 带宽节省 | 单请求: **0 bytes**。B=32 共享 100 token 前缀: 前缀 KV 读取 = 100 × 4 heads × 256 dim × 2B × 2 (K+V) × 16 layers = 65.5 MB, 从 32× 读降为 1× → 省 ~63 MB |
| 时间节省 | B=1: **0ms**。B=32, ctx=100: 63 MB / 234 GB/s ≈ **0.27ms**。ctx=1000: **~2.7ms** |
| 难度 | 高 — 需要 prefix tree 管理, attention kernel 拆分为两阶段 + merge |
| 代码现状 | ❌ 未实现 |
| **结论** | **🟡 条件 GO** — B>1 长上下文共享前缀场景有价值; 短上下文收益仍被 GEMM 掩盖 |

### 2. JIT Compilation of Attention Variants

**原理**: FlashInfer 在运行时根据 head_dim, causal mask, page_size 等参数 JIT 编译最优 attention kernel，避免通用 kernel 的分支和性能损失。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 本系统模型参数固定 (head_dim=256, 4 KV heads, block_size=16)，不需要运行时适配多种配置。 |
| 带宽节省 | **0 bytes** — JIT 优化的是计算效率，不是内存访问量 |
| 时间节省 | Paged Attention 仅 ~0.5ms/step, 即使 JIT kernel 快 30%, 也只省 **~0.15ms** |
| 难度 | 高 — 需要 NVRTC 或 cubin 缓存机制 |
| 代码现状 | ❌ 未实现 (AOT 编译, 固定架构) |
| **结论** | **🔴 NO-GO** — 配置固定无需运行时特化; attention 不是瓶颈 |

### 3. RaggedTensor / Page Table 优化

**原理**: FlashInfer 使用 RaggedTensor 一维紧凑存储多序列的变长 KV，消除 padding 浪费；page table 使用 GPU 友好的索引结构减少间接访问开销。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 当前单序列推理，无 padding 浪费。Page table 已是简单的 `int[]` 线性 block_table。 |
| 带宽节省 | **0 bytes** — 单序列无 padding; page table 本身极小 (几十个 int) |
| 时间节省 | **~0ms** |
| 难度 | 低 (数据结构改动) |
| 代码现状 | ✅ 已有简单 Paged KV cache (block_size=16, per-layer independent) |
| **结论** | **🔴 NO-GO** — 单序列场景下无优化空间 |

### 4. Fused RoPE in Attention Kernel

**原理**: 将 RoPE 旋转嵌入直接融合到 attention kernel 的 QK 读取路径中，避免对 Q 和 K 做一次额外的全局内存 read+write。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 仅影响 16 层 FullAttn。当前 RoPE 是独立 kernel `invoke_rope_partial()`。 |
| 带宽节省 | Partial RoPE 只旋转 64/256 维。数据量 = 16 层 × (24 Q heads + 4 KV heads) × 64 dim × 2B × 2 (read+write) = 16 × 28 × 64 × 4 = **114,688 bytes ≈ 0.11 MB** |
| 时间节省 | 0.11 MB / 216 GB/s = **~0.0005ms** |
| 难度 | 中 — 需修改 paged_attention_kernel 和 write_kv_cache_kernel 以内联 RoPE |
| 代码现状 | ❌ RoPE 是独立 kernel |
| **结论** | **🔴 NO-GO** — 0.11 MB 节省在 51 GB 权重面前可忽略 |

### 5. Head-Group Batch GEMV (GQA Decode)

**原理**: 对于 GQA decode，将 Q heads 按 KV head 分组，每组 Q heads 共享同一个 KV head 的 cache 读取，以 batched GEMV 方式执行 $\text{score}_{h} = q_h^T \cdot K_{\text{kv\_head}}$ ，减少 KV cache 的重复读取。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | FullAttn: 24 Q heads / 4 KV heads = 6 Q heads/group。当前 kernel 每 block 处理 1 个 (token, Q_head)，KV head 被每个 Q head 的 block 独立读取 → 同一 KV head 的 K/V cache 被读 6 次。**多并发下冗余放大 B 倍**。 |
| 带宽节省 | 每层每 context_token: 4 KV heads × 256 dim × 2B × 2 (K+V) = 4 KB。当前读 6× = 24 KB/token/layer。优化后读 1× = 4 KB/token/layer。节省 = 20 KB/token/layer × 16 层 = 320 KB/token。B=32, ctx=100: 32 × 100 × 320 KB = **1 GB → ~4.3ms**。B=64, ctx=100: **~8.5ms**。 |
| 时间节省 | B=1, ctx=100: **~0.15ms**。B=32, ctx=100: **~4.3ms**。B=64, ctx=200: **~17ms** |
| 难度 | 中 — 重写 paged_attention_kernel，block 维度改为 (token, kv_head)，内部循环处理 6 个 Q heads |
| 代码现状 | ❌ 当前逐 Q head 独立处理 |
| **结论** | **🟢 GO** — 多并发 + 长上下文场景下收益显著 (4-17ms); 即使 B=1 也有微小收益 |

### 6. Persistent Kernels

**原理**: 让 kernel 的 thread blocks 常驻 SM，通过循环处理多个工作项，避免反复 launch kernel 的开销。TensorRT-LLM 和 FlashInfer 用于 attention 和 GEMM。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | CUDA Graph 分析已证明: CPU launch 速率 (~3µs/kernel) 远快于 GPU 执行 (~273µs/kernel)，**零 pipeline bubbles**。persistent kernel 解决的是 launch overhead 问题——本系统不存在此问题。 |
| 带宽节省 | **0 bytes** |
| 时间节省 | **~0ms** (launch overhead 已被掩藏) |
| 难度 | 高 — 需要整体架构重写, cooperative groups, 手工调度 |
| 代码现状 | ❌ 未实现 |
| **结论** | **🔴 NO-GO** — 本系统没有 kernel launch 瓶颈; 868 kernels × 3µs = 2.6ms CPU 时间 vs 237ms GPU 时间 |

### 7. FP8 KV Cache (权重保持 BF16)

**原理**: 将 KV cache 中的 K 和 V 值从 BF16 (2B) 量化为 FP8 (1B)，减少 attention 阶段的 cache 读取带宽。权重和计算仍保持 BF16。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 仅影响 16 层 FullAttn 的 paged_attention 读取。KV cache 数据量 = 16 层 × ctx × 4 KV_heads × 256 dim × 2 (K+V) × 2B = 65.5 KB/token。BF16→FP8 节省 50% = 32.8 KB/token。 |
| 带宽节省 | ctx=100: 3.2 MB → 0.015ms; ctx=1000: 32 MB → 0.15ms; ctx=4096: 131 MB → **0.6ms** |
| **多并发影响** | B=32, ctx=200: 32×200×32.8KB = **210 MB 节省 → ~0.9ms**。B=64, ctx=500: 64×500×32.8KB = **1.05 GB → ~4.6ms**。更关键的是 **内存占用**: B=256, ctx=500: KV cache = 256×500×65.5KB = **8.4 GB (BF16) → 4.2 GB (FP8)**。FP8 直接决定 max_batch 或 max_context。 |
| 时间节省 | B=1 短上下文: **<0.1ms**。B=64, ctx=500: **~4.6ms**。**内存节省与吞吐天花板直接相关**。 |
| 难度 | 中 — 需要 FP8 write kernel (BF16→FP8 quantize), attention kernel 中 FP8→FP32 dequant |
| 代码现状 | ❌ 未实现 |
| 精度风险 | 低 — FP8 E4M3 动态范围对 softmax 后的 attention values 足够; K 需要 per-head scale 校准 |
| **结论** | **🟢 GO** — 多并发下 KV cache 内存是硬约束: FP8 直接将最大并发数/上下文长度翻倍; 长上下文 + 高并发可节省数 ms |

---

## 二、SGLang / vLLM 技术

### 8. RadixAttention / Prefix Caching

**原理**: 用 radix tree 管理 KV cache key (prompt token sequence)，当多个请求共享相同的 system prompt 或 few-shot 前缀时，跳过前缀部分的 prefill 计算，直接复用已有的 KV cache。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | **多并发服务场景彻底改变评估**。多用户共享 system prompt (如 500 tokens) 时: 每用户 prefill 500 tok = 500×219ms/tok = ~110s (全 forward); 使用 prefix cache 可跳过共享前缀的 KV 计算。同时节省 **KV cache 内存**: 16 层 × 500 × 4 × 256 × 2 × 2B = 32.8 MB/用户 → 共享后仅需 1 份。 |
| 带宽节省 | Decode: **0 bytes/step**。Prefill (每新用户): **跳过前缀的全部 forward 计算** |
| 时间节省 | Decode: **0ms**。Prefill: **system_prompt_len × ~0.22s/token**。500 tok prefix → **~110s → 0s (复用)**。B=32 用户同一 prompt: 节省 31 × 110s = **~57 分钟** |
| 内存节省 | 共享 500 tok prefix × 32 用户: **31 × 32.8 MB = ~1 GB KV cache 内存** |
| 难度 | 高 — radix tree for KV blocks, hash/compare prefix, eviction policy |
| 代码现状 | ❌ 未实现 |
| **结论** | **🟢 GO** — 多用户服务的 **必需功能**: 共享 system prompt 场景下节省巨量 prefill 时间和 KV cache 内存; 实现复杂度高但回报极大 |

### 9. Chunked Prefill

**原理**: 将长 prefill 拆分为多个 chunk，每个 chunk 与 decode batch 交替执行，避免长 prefill 阻塞 decode 请求的时延。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | **多并发 continuous batching 已实现，chunked prefill 变为关键需求**。场景: B=32 并发 decode 运行中，新用户到达需要 prefill 1000 tokens。不分块: prefill 独占 GPU ~220s, 32 个 decode 用户全部停滞。分块 (chunk=64): prefill 拆为 16 次, 每次 ~14s, 交替执行 decode → decode 用户感知延迟从 220s 降到 ~14s。 |
| 带宽节省 | **0 bytes** (总计算量不变) |
| 时间节省 | 不减少总时间, 但 **decode 延迟尖刺从 ~220s 降到 ~14s (chunk=64)**。对 TTFT: 首 chunk 后即可开始返回部分 attention 结果。 |
| 难度 | 中-高 — 引擎调度循环拆分: prefill_chunk() + batched_decode() 交替; SSM/Conv state 需要在 chunk 间正确续接 |
| 代码现状 | ❌ 未实现 |
| **结论** | **🟢 GO (生产必需)** — 多并发服务下不做 chunked prefill = decode 用户遭遇秒级停顿, 不可接受; 是 continuous batching 的必要补充 |

### 10. Tensor Parallelism on Single Device (Stream-Level Parallelism)

**原理**: 在单 GPU 上使用多个 CUDA stream 并行执行独立的计算（如不同 head group 的 attention 或 MLP 的 gate+up 两个分支），利用 SM 分区实现类似 tensor parallel 的效果。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | Decode M=1 场景中，每个 GEMV 已经消耗全部 20 个 SM (grid size = N/8 warps, 最小 GEMV N=48 → 6 blocks, 最大 N=248320 → 31040 blocks)。在 DRAM 带宽瓶颈下，**多 stream 并行不增加带宽**。 |
| 带宽节省 | **0 bytes** — 权重读取总量不变 |
| 时间节省 | **0ms 或负** — 多 stream 争抢 DRAM 带宽会导致单个 stream 的 GEMV 带宽利用率下降 |
| 难度 | 中 — stream 分配和同步逻辑 |
| 代码现状 | ❌ 单 stream |
| **结论** | **🔴 NO-GO** — 带宽瓶颈下多 stream 无法增加吞吐; 会增加调度复杂性和带宽争抢 |

### 11. Custom All-Reduce / Scheduling Insights

**原理**: vLLM/SGLang 针对多 GPU tensor parallel 优化 all-reduce 通信。单设备不适用 all-reduce，但其调度洞见（如 layer-by-layer weight prefetch）可借鉴。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 单设备无 all-reduce。权重预取 (prefetch) 在 UMA 架构上无意义——CPU 和 GPU 共享物理内存，没有 PCIe 传输或 NVLink。 |
| **唯一可借鉴点** | **L2 cache persistence policy**: 可设置 `cudaAccessPolicyWindow` 让高频访问的小 tensor (norm weights: 64 × 5120 × 2B = 640 KB, 适配 32 MB L2) 常驻 L2。 |
| 带宽节省 | Norm weights L2 hit: 640 KB × 64 layers × 2 reads = 82 MB → **0.38ms** (如果从 L2 读则省去 DRAM 访问) |
| 时间节省 | **≤0.38ms** (理论上限; 实际可能更少因为 norm weights 已被自然缓存) |
| 难度 | 低 — 几行 CUDA API 调用 |
| 代码现状 | ❌ 未设置 L2 persistence |
| **结论** | **🟢 GO (L2 persistence)** — 实现简单, 可能省 0.1-0.3ms. 主策略 NO-GO |

---

## 三、TensorRT-LLM 技术

### 12. Weight-Only GEMV with Fast Dequant (Layout Insights)

**原理**: TRT-LLM 对 INT8/INT4 权重使用特殊的交错布局 (interleaved layout)，使得 dequant 指令和 FMA 可以流水执行。不做量化时，其**布局洞见**仍有参考价值。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 当前权重是 Column Major BF16 `[K, N]`, 即 B(k,n) = B[n*K + k]。Warp 内 32 线程读 32 个连续 float4 → 512B transaction → 完美 coalesced。 |
| 布局分析 | TRT-LLM 的交错布局是为 INT4 dequant 设计的, 在 BF16 下反而会破坏 coalesced 访问。当前 Column Major 是 GEMV 的最优布局。 |
| 带宽节省 | **0 bytes** |
| 时间节省 | **0ms** |
| 代码现状 | ✅ 已是最优布局 |
| **结论** | **🔴 NO-GO** — 当前布局已是 BF16 GEMV 最优; TRT-LLM 交错布局不适用 |

### 13. Fused MHA (Multi-Head Attention in One Kernel)

**原理**: 将 Q projection → Q/K norm → RoPE → KV write → Attention → Gate → O projection 合并为一个 mega-kernel，消除所有中间缓冲区的 DRAM 读写。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 仅 16 层 FullAttn。当前 attention 通路涉及的中间数据:</br>- QKV proj 输出: 1×(12288+1024+1024)×2B = 28 KB</br>- Deinterleave+Norm 输出: 1×6144×2B = 12 KB</br>- RoPE in-place: 0 额外</br>- Attn output: 1×6144×2B = 12 KB</br>- Gate sigmoid: 1×6144×2B = 12 KB</br>- O proj output: 1×5120×2B = 10 KB</br>**每层中间数据 ≈ 74 KB**, 16 层 = 1.2 MB |
| 带宽节省 | 避免中间写+读 ≈ 1.2 MB × 2 = **2.4 MB** |
| 时间节省 | 2.4 MB / 216 GB/s = **~0.011ms** |
| 难度 | **极高** — 需要在一个 kernel 中管理 GEMV (需全部 SM), attention (per-head), norm (per-token), RoPE (per-pair)，thread block 职责完全不同 |
| 代码现状 | ❌ 各步骤独立 kernel |
| **结论** | **🔴 NO-GO** — 2.4 MB 中间数据在 51 GB 权重面前微不足道 (0.005%), 工程量巨大 |

### 14. Plugin-Based Kernel Selection / Auto-Tuning

**原理**: TRT-LLM 在模型加载时针对每个 GEMM/GEMV 的形状 (M,N,K) 自动搜索最优的 tile size, block size, 以及 kernel variant (tiled vs scattered vs 8-warp vs 4-warp)。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 当前 GEMV 使用 if/else 在 `invoke_dense_gemv()` 中选择:</br>- K>8192 → `scattered_tiled` (4 warps, tile_k=4096)</br>- K≤8192, smem≤48KB → `gemv_kernel` (8 warps)</br>- 否则 → `tiled` (8 warps, tile_k=4096)</br>这覆盖了 3 种 variant, 但 tile_k=4096 是固定值。 |
| 可调参数 | tile_k (2048/4096/8192), WARPS_PER_BLOCK (4/8), scattered vs sequential |
| **多并发影响** | **极大**。当 B>1 时 GEMV→GEMM, 需要为不同 M 值 (1,2,4,8,16,32,64,128,256) 选择不同 CUTLASS tile size。基准数据显示 **B=16 异常加速 (250ms < B=8 的 279ms)**, 表明 tile/SM 利用率对性能影响巨大。B=128 时 compute-bound, tile 选择直接决定 TFLOPS。 |
| 潜在收益 | B=1: 0-1ms (GEMV variant 微调)。**B=16: 如果能将 B=8/32 也调到 B=16 的效率 → 每用户延迟减少 10-20%**。B≥128: GEMM tile tuning 影响 TFLOPS 利用率。 |
| 时间节省 | B=1: **0-1ms**。B=16: **~30ms** (如果消除异常)。B=128+: **潜在 10-20%** |
| 难度 | 低-中 — 加载时对每个 (M,N,K) 形状遍历 CUTLASS tile variants, 记录最优 |
| 代码现状 | ⚠️ 部分实现 (GEMV heuristic; GEMM 使用固定 CUTLASS tile) |
| **结论** | **🟢 GO** — 多并发场景下 GEMM tile 选择直接决定吞吐; B=16 异常数据证明 auto-tune 有显著收益空间; 低风险低成本 |

### 15. Layer Fusion Beyond Attention (Norm+Linear, Linear+Activation)

**原理**: TRT-LLM 和 FasterTransformer 将 RMSNorm → GEMV、GEMV → SiLU、GEMV → residual add 等融合为单个 kernel, 消除中间缓冲区。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 已有融合 | ✅ Fused Add+RMSNorm (64 层), ✅ Fused Deinterleave+Q_RMSNorm (16 层), ✅ Fused Norm+SiLU+Gate (48 层), ✅ DeltaNet alpha/beta 内联 |
| 剩余可融合项 | |
| **RMSNorm → GEMV** | 将 norm 输出直接 feed 到 GEMV 的 shared memory 加载中。节省 norm_out 的 DRAM write+read = 64 层 × 5120 × 2B × 2 = **1.25 MB → 0.006ms** |
| **SwiGLU → Down GEMV** | 将 silu(gate)×up 直接在 GEMV 读 A 时计算。节省 swiglu_out 的 DRAM write+read = 64 层 × 17408 × 2B × 2 = **4.5 MB → 0.02ms** |
| **Down GEMV → residual Add** | GEMV 输出直接加到 hidden_states。节省 down_out 的 DRAM write+read = 64 层 × 5120 × 2B × 2 = **1.25 MB → 0.006ms** |
| 总带宽节省 | **~7 MB → ~0.03ms** |
| 时间节省 | **~0.03ms** (DRAM) + **~0.6ms** (launch savings, 但已被掩藏 → 实际 0ms) = **~0.03ms** |
| 难度 | 高 — 需要 GEMV kernel 支持 epilogue callback (norm/silu/add), 模板化重写 |
| 代码现状 | ⚠️ 部分实现 (已有 5 种融合; 剩余 3 种收益极低) |
| **结论** | **🔴 NO-GO** — 所有剩余融合合计 <0.05ms, 不值得工程投入 |

---

## 四、通用 CUDA 优化技术

### 16. Persistent Thread Blocks

**原理**: 线程块创建后不退出, 通过 atomic work-stealing 或 cooperative groups 的 `grid.sync()` 反复获取新任务。避免 kernel launch 的 block 调度开销。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | CUDA Graph 分析已证明:**零 pipeline bubbles**。GPU 237ms vs CPU launch 2.6ms。Persistent blocks 和 CUDA Graphs 解决同一问题 (launch overhead), 该问题在本系统不存在。 |
| 带宽节省 | **0 bytes** |
| 时间节省 | **~0ms** |
| 难度 | 极高 — 需要完全重写所有 kernel 为 persistent 模式 |
| 代码现状 | ❌ 未实现 |
| **结论** | **🔴 NO-GO** — Launch overhead 不是瓶颈; 极高复杂度零收益 |

### 17. Software Pipelining (Prefetch / Double Buffering)

**原理**: 在 GEMV 的 K 维循环中, 使用 `cp.async` 或手动 double-buffer 预取下一轮的 B 列数据到 shared memory, 同时计算当前轮, 隐藏 DRAM latency。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 当前 `gemv_kernel` 中 A 通过 shared memory 广播, B 直接从 global memory float4 读取。每 warp 每轮读 `float4` 相当于 128B/warp → 完美 coalesced。关键问题: **LPDDR5X 的问题不是 latency 而是 bandwidth**。 |
| 分析 | SM110 L1 cache (128 KB/SM) 已自动缓存 global memory 读取。float4 向量化读取已充分利用 128-bit bus。double buffering 需要更多 shared memory, 会降低 SM occupancy。 |
| 权衡 | 当前 GEMV 38 registers, 100% occupancy (6 blocks×256/SM)。引入 double buffering → shared memory 翻倍 (K×4 bytes) → K=5120 时需 20 KB→40 KB/block → occupancy 从 6 blocks 降到 2-3 blocks → 更差的 latency hiding。 |
| 时间节省 | **0ms 或负** (在 bandwidth-bound 场景下, 增加 shared memory 压力反而有害) |
| 难度 | 中 |
| 代码现状 | ❌ 未实现 (也不应实现) |
| **结论** | **🔴 NO-GO** — GEMV 是带宽瓶颈而非延迟瓶颈; software pipelining 适用于 compute-bound 或 latency-hiding 场景 |

### 18. Warp Specialization

**原理**: 在同一 block 中, 部分 warp 专门负责数据加载 (producer), 另一部分 warp 专门计算 (consumer), 通过 shared memory 传递数据。CUTLASS 3.x 的 GEMM 核心使用此模式。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | Warp specialization 在 **GEMM** (compute-bound, M>1) 中非常有效, CUTLASS 已用此模式。但 **GEMV** (M=1, bandwidth-bound) 中: 计算 = N 次 dot(K), 算术强度 ≈ 2 FLOP/2B = 1 FLOP/byte → 极低。生产者 warp 的加载速度受限于 DRAM 带宽, 不存在计算可以 overlap 的对象。 |
| 时间节省 | **0ms** (加载和计算已经完全串行化在带宽通道上) |
| 难度 | 高 |
| 代码现状 | ❌ GEMV 中未使用 (CUTLASS GEMM 已使用) |
| **结论** | **🔴 NO-GO** — GEMV 算术强度太低, producer/consumer 分离无法隐藏任何开销 |

### 19. Vectorized bfloat162 Arithmetic

**原理**: 使用 `__nv_bfloat162` 类型和 `__hfma2`/`__hadd2` 等指令, 在一条 CUDA 指令中同时处理两个 BF16 值, 提高计算吞吐。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 当前实现 | GEMV 已使用 `__bfloat1622float2()` 将 bf162 解包为 float2 进行 FMA 计算。结果用 `af.x * bf.x + af.y * bf.y` 在 FP32 精度累加。 |
| 替代方案 | 使用 `__hmul2` + `__hadd2` 直接做 BF16 点积, 省去 bf16↔f32 转换。但: (1) BF16 精度 7 bit mantissa, K=5120~17408 维累加会有精度损失; (2) 本系统不是 compute-bound, FP32→BF16 转换指令不是瓶颈。 |
| 计算分析 | 每步 FP32 FMA 数量 = 51.2 GB / 2B = 25.6G elements → ~25.6 GFLOP。SM110 FP32 吞吐 = 8.06 TFLOP → **计算仅用 0.3% 的 FP32 FLOPS**。即使 BF16×2 快 2×, 也不影响瓶颈。 |
| 时间节省 | **0ms** (compute 不是瓶颈) |
| 精度风险 | 中 — 长维度累加精度损失 |
| 代码现状 | ⚠️ 已使用 bfloat162 解包+FP32 累加 (正确且无精度损失) |
| **结论** | **🔴 NO-GO** — 0.3% 计算利用率下, 2× 计算加速 = 0% 实际加速 |

### 20. Read-Only Cache (`__ldg`) for Weight Loads

**原理**: 使用 `__ldg()` intrinsic (或 `const __restrict__` 提示) 通过 texture/read-only cache 路径加载数据, 可能获得更好的缓存行为和带宽利用率。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 当前实现 | GEMV kernel 参数已使用 `const __nv_bfloat16* __restrict__ B`。在 SM110 上, `__restrict__` 提示编译器可使用 `LDG` (read-only/texture cache) 加载。`nvcc -O3` 通常自动将 `const __restrict__` 指针的读取编译为 `LDG`。 |
| 显式 `__ldg` 影响 | SM80+ 以后, L1/texture cache 已统一, `LDG` vs `LD` 性能差异很小。SM110 的 L1 cache (128 KB/SM) 对 GEMV 的大权重矩阵几乎无命中率 (每个 block 读 K×2B = 10-35 KB 的 B 列, 远超 L1)。 |
| 时间节省 | **~0ms** (编译器已自动优化; L1 cache 对流式大数据无效) |
| 难度 | 极低 (已隐式实现) |
| 代码现状 | ✅ 已使用 `const __restrict__` (等效) |
| **结论** | **🔴 NO-GO** — 已隐式实现; SM110 统一 cache 下无额外收益 |

### 21. Non-Temporal Stores

**原理**: 使用 `__stwt()` 或 `__stcg()` 绕过 L1/L2 cache 直接写入 DRAM, 避免写回缓存行的 cache pollution, 为后续读取留出更多 cache 空间。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | GEMV 输出写: 每个 warp lane0 写 1 个 BF16 (2 bytes) 到 C。这是极低带宽写操作 (每步 ~102 KB 总写出 = 51.2M 个输出 × 2B, 分散在 868 个 kernel 中)。 |
| 分析 | non-temporal store 主要帮助流式大写操作 (如 memcpy), 对 GEMV 逐元素写无实际帮助。且 intermediate buffers 马上被下一个 kernel 读取 → 被 L2 cache 是**有利的** (保持数据热度)。 |
| 风险 | NT store 会让下一个 kernel 读取时 L2 miss, 反而增加延迟 |
| 时间节省 | **0ms 或负** |
| 代码现状 | ❌ 未使用 (正确决策) |
| **结论** | **🔴 NO-GO** — 写操作量微不足道; NT store 会伤害 intermediate buffer 的缓存局部性 |

### 22. Register Blocking for Small GEMV

**原理**: 让每个线程在寄存器中同时累积多个输出元素的 partial sum, 共享 A 向量的寄存器读取, 减少 A 的 shared memory 访问次数。例如每 warp 同时处理 4 个输出列而非 1 个。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 当前: 1 warp → 1 输出, warp 内 32 threads 遍历 K/8 个 float4。A 从 shared memory 读, B 从 global memory 读。A 的读带宽 = 每 warp 每迭代 16B (1 float4) 从 smem → warp 内广播。 |
| Register blocking 方案 | 1 warp → R 个输出。A 的 float4 读 1 次, B 的 float4 读 R 次。寄存器: R 个 `float sum` 额外。shared memory A 访问减少到 1/R。 |
| 实际收益分析 | A 从 shared memory 读是 **free** (无 bank conflict, 广播模式)。瓶颈是 B 从 DRAM 读。R>1 时 B 的读取变为 R×, A 读只节省 smem 带宽 (不是 DRAM)。**总 DRAM 读取量不变 (B 总量 = N×K×2B)**。唯一的节省是减少 warp 数量 → 减少 warp scheduling overhead → 在 bandwidth-bound 下不可测。 |
| 时间节省 | **~0ms** (DRAM 读取量不变, warp scheduling 不是瓶颈) |
| 难度 | 低-中 |
| 代码现状 | ❌ 未实现 |
| **结论** | **🔴 NO-GO** — 不减少 DRAM 带宽消耗; shared memory 带宽不是瓶颈 |

### 23. Cooperative Groups (Cross-Block Synchronization)

**原理**: CUDA cooperative groups 允许跨 block 的 grid-level 同步 (`grid.sync()`), 可用于 persistent kernel 或跨 block reduction。

**对本系统的评估**:

| 维度 | 分析 |
|------|------|
| 适用性 | 潜在用例: (1) persistent kernel 跨层 (→ 已在 #16 排除); (2) split-K GEMV (将大 N 的 GEMV 拆分为多 block K 维并行 → cross-block reduce 合并); (3) argmax cross-block (当前已用单 block 解决)。 |
| Split-K GEMV 分析 | 当前 GEMV: 每 warp 处理 1 个 N 维输出, K 维在 warp 内 reduce。Split-K: 每个输出由多个 block 在 K 维并行处理, 最终跨 block 合并。但 GEMV bandwidth-bound, K 维已足够并行 (K/256 = 20-68 迭代/warp → 每 warp 工作量充分)。Split-K 增加额外的 cross-block 合并写和 sync 开销。 |
| 时间节省 | **0ms 或负** |
| 难度 | 中-高 |
| 代码现状 | ❌ 未使用 |
| **结论** | **🔴 NO-GO** — 所有 cooperative groups 用例在本系统的 bandwidth-bound 场景下无收益 |

---

## 五、未在列表中但值得注意的方向

### 24. Dual/Quad GEMV 扩展 (已部分实现)

**原理**: 将共享输入 A 的多个 GEMV 合并为一个 kernel, 共享 A 的 shared memory 加载。

| 维度 | 分析 |
|------|------|
| 当前实现 | ✅ `invoke_dense_dual_gemv()` 用于 MLP gate+up (共享 post_norm_out A 输入) |
| 扩展方向 | FullAttn 已有 QKV merged GEMV (单次 GEMV N=14336)。LinearAttn 已有 ZAB merged (N=6240)。**所有主要的 A 共享已完成**。 |
| 剩余可合并 | down_proj + next_layer_input_norm 的 A 不共享 (不同输入), 无法合并。 |
| **结论** | **✅ 已优化** — 所有可共享 A 的 GEMV 已合并 |

### 25. DRAM 访问模式优化 (Bank-Level Parallelism)

**原理**: LPDDR5X 有 16 banks, 8 bank groups。连续地址落在同一 bank 会导致 bank conflict, 降低有效带宽。权重矩阵的列访问模式 (Column Major) 决定了 DRAM bank 的命中模式。

| 维度 | 分析 |
|------|------|
| 适用性 | 当前 GEMV 有效带宽 216 GB/s vs 峰值 273 GB/s = 79.1%。**~20% 的 gap 最可能来自 DRAM bank conflict 和 row buffer miss**。 |
| 可尝试方向 | (1) 权重矩阵 pad 到特定对齐 (避免相邻 warp 访问同一 bank); (2) 修改 scattered GEMV 的映射策略使同一 block 的 warp 访问不同 bank group; (3) 实验 256B vs 512B aligned 列起始地址。 |
| 潜在收益 | 如果消除一半的 bank conflict: 216→~240 GB/s → 51.2/240 = 213ms → **省 ~24ms (10%)** |
| 难度 | 中 — 需要对 LPDDR5X 的 bank mapping 做逆向分析和实验 |
| 代码现状 | ❌ 未专门优化 DRAM 访问模式 |
| **结论** | **🟢 GO** — 这是目前最大的优化空间; 需要系统性的 microbenchmark 实验 |

### 26. 多 Batch (B>1) Decode

| 维度 | 分析 |
|------|------|
| 适用性 | B=2 时, 每步读同样 51 GB 权重但产出 2 个 token → 有效吞吐翻倍。这是 **唯一能突破带宽墙的 BF16 方案** (除量化外)。 |
| 实测结果 | B=1: 4.57 tok/s, 234 GB/s │ B=32: 97.5 tok/s │ B=64: 192.4 tok/s │ B=128: 282.7 tok/s │ B=256: 365.4 tok/s |
| 代码现状 | ✅ **已完全实现** — Phase 8 完成 continuous batching, 所有 kernel (PA, DeltaNet, Conv1d, write_kv_cache) 支持 batch_size, GEMV→CUTLASS GEMM 自动切换 |
| **结论** | **✅ 已完成** — 吞吐从 4.57 tok/s 提升到 365 tok/s (80×), 验证了权重复用的核心价值 |

### 27. Batched Argmax (采样优化)

**原理**: 当前 batched decode 时, 每个 batch 元素单独调用 `invoke_argmax()` (1 block × 1024 threads), B=256 时需要 256 次 kernel launch。改为单次 launch、grid=batch_size blocks 的 batched argmax kernel。

| 维度 | 分析 |
|------|------|
| 适用性 | 基准数据: B=1: 0.05ms, B=32: 4.3ms, B=64: 7.0ms, B=128: 17.0ms, **B=256: 33.1ms (4.7% of step time)**。每次 launch overhead ~0.13ms × 256 = 33ms。 |
| 优化方案 | 单 kernel: grid=(batch_size,1,1), block=(1024,1,1)。每 block 对 `logits[b * vocab_size ... (b+1) * vocab_size - 1]` 做 argmax。结果写入 `result_idx[b]`。 |
| 时间节省 | B=256: **33ms → <0.5ms (节省 ~32ms, 4.6%)**。B=128: **17ms → <0.3ms**。B=64: **7ms → <0.2ms**。 |
| 难度 | **极低** — 现有 argmax kernel 每 block 逻辑不变, 只需外加 batch 维度的 grid indexing |
| 代码现状 | ❌ 逐请求串行调用 |
| **结论** | **🟢 GO (P0 quick win)** — 极低成本换 4.6% 性能提升 (B=256); 应立即实现 |

### 28. SSM State 内存优化

**原理**: LinearAttn (Gated DeltaNet) 每个请求需要维护 SSM state 和 Conv state。当前每请求内存: 48 层 × (16 heads × 128 key_dim × 128 value_dim × 4B + 10240 × 4 × 2B) ≈ **147 MB**。B=256 时 SSM 状态总共 **37.6 GB**。

| 维度 | 分析 |
|------|------|
| 适用性 | 128 GB 统一内存中: 权重 51.2 GB + B=256 SSM 37.6 GB + B=256 KV cache (~1-8 GB) + workspace → **内存在 B≥256 时接近极限** |
| 优化方向 | (1) **BF16 SSM state**: 目前 FP32 (4B/element), 改为 BF16 (2B) 可将 147MB→74MB/request, 37.6→18.8 GB @B=256。精度风险需评估。 (2) **SSM state offload**: 不活跃请求的 SSM state swap 到 CPU-pinned 内存或 NVMe。 (3) **Lazy allocation**: 仅在请求活跃时分配 SSM state。 |
| 内存节省 | BF16 方案: **~18.8 GB** @B=256。允许支持更多并发请求。 |
| 难度 | 低 (BF16 转换) 到 中 (offload 策略) |
| 代码现状 | ❌ FP32 全量分配 |
| **结论** | **🟢 GO** — SSM 内存是多并发的第二大约束 (仅次于权重本身); BF16 化或 offload 可将最大并发数翻倍 |

---

## 综合评估总表

| # | 技术 | 来源 | DRAM 节省 | 时间节省 | 难度 | 状态 | 结论 |
|---|------|------|----------|---------|------|------|------|
| 1 | Cascade Attention | FlashInfer | 0 | 0ms (B=1) | 高 | ❌ | 🟡 条件GO |
| 2 | JIT Attention | FlashInfer | 0 | ~0ms | 高 | ❌ | 🔴 NO-GO |
| 3 | RaggedTensor/PageTable | FlashInfer | 0 | 0ms | 低 | ✅ | 🔴 NO-GO |
| 4 | Fused RoPE in Attn | FlashInfer | 0.11 MB | ~0ms | 中 | ❌ | 🔴 NO-GO |
| 5 | Head-Group Batch GEMV | FlashInfer | 1GB@B32ctx100 | 4-17ms | 中 | ❌ | 🟢 GO |
| 6 | Persistent Kernels | FlashInfer | 0 | 0ms | 极高 | ❌ | 🔴 NO-GO |
| 7 | FP8 KV Cache | FlashInfer | 1GB@B64ctx500 | 0.9-4.6ms | 中 | ❌ | 🟢 GO |
| 8 | Prefix Caching | SGLang/vLLM | 1GB@B32 | prefill 省分钟级 | 高 | ❌ | 🟢 GO |
| 9 | Chunked Prefill | SGLang/vLLM | 0 | 延迟尖刺 220s→14s | 中-高 | ❌ | 🟢 GO |
| 10 | Stream Parallelism | SGLang/vLLM | 0 | 0ms/负 | 中 | ❌ | 🔴 NO-GO |
| 11 | L2 Persistence Policy | (调度洞见) | 82 MB | <0.1ms | 低 | ✅ | ✅ 已完成 |
| 12 | Weight Layout (BF16) | TRT-LLM | 0 | 0ms | — | ✅ | 🔴 NO-GO |
| 13 | Fused MHA | TRT-LLM | 2.4 MB | ~0.01ms | 极高 | ❌ | 🔴 NO-GO |
| 14 | Auto-Tune GEMV/GEMM | TRT-LLM | 0 | 0-30ms | 低-中 | ⚠️ | 🟢 GO |
| 15 | Layer Fusion (剩余) | TRT-LLM | 7 MB | ~0.03ms | 高 | ⚠️ | 🔴 NO-GO |
| 16 | Persistent Thread Blocks | CUDA | 0 | 0ms | 极高 | ❌ | 🔴 NO-GO |
| 17 | Software Pipelining | CUDA | 0 | 0ms/负 | 中 | ❌ | 🔴 NO-GO |
| 18 | Warp Specialization | CUDA | 0 | 0ms | 高 | ❌ | 🔴 NO-GO |
| 19 | BF16x2 Arithmetic | CUDA | 0 | 0ms | 低 | ⚠️ | 🔴 NO-GO |
| 20 | `__ldg` Read-Only | CUDA | 0 | 0ms | — | ✅ | 🔴 NO-GO |
| 21 | Non-Temporal Stores | CUDA | 0 | 0ms/负 | 低 | ❌ | 🔴 NO-GO |
| 22 | Register Blocking | CUDA | 0 | 0ms | 低-中 | ❌ | 🔴 NO-GO |
| 23 | Cooperative Groups | CUDA | 0 | 0ms | 中-高 | ❌ | 🔴 NO-GO |
| 24 | Dual/Quad GEMV | 自有 | — | — | — | ✅ | ✅ 已完成 |
| **25** | **DRAM Bank-Level优化** | **自研** | **0** | **10-24ms** | **中** | ❌ | **🟢 GO** |
| **26** | **多 Batch Decode (B>1)** | **通用** | **0/token** | **80× 吞吐** | **中** | ✅ | **✅ 已完成** |
| **27** | **Batched Argmax** | **自研** | **0** | **32ms@B256** | **极低** | ✅ | **✅ 已完成** |
| **28** | **SSM State 内存优化** | **自研** | **18.8GB@B256** | **扩容** | **低-中** | ❌ | **🟢 GO** |
| **29** | **CUDA Graph** | **CUDA** | **0** | **3-14ms@B32-64** | **中** | ✅ | **✅ 已完成** |

---

## 结论与建议

### 核心发现

**28+1 个技术中有 15 个 NO-GO, 12 个 GO/已完成, 2 个条件 GO**。

Phase 8-9 后评估基础发生根本变化:

> **单请求 (B=1): 纯 DRAM 带宽瓶颈, 已达 234 GB/s (86% 峰值), 优化空间有限。**
> **多并发 (B≥32): 带宽→计算过渡, 调度/内存管理成为新瓶颈。**
> **之前被判 NO-GO 的 6 项技术 (#5,#7,#8,#9,#14,#26) 在多并发下变为 GO 或已完成。**

### 已完成优化 (Phase 1-9)

- ✅ GEMV 核心优化 (vectorized, smem broadcast, warp reduce) — 234 GB/s (86%)
- ✅ 所有可合并权重的 GEMV 合并 (QKV, ZAB, dual gate+up)
- ✅ 所有有意义的算子融合 (5 种 fused kernels)
- ✅ Launch overhead 完全被 GPU 执行掩藏 (0 pipeline bubbles)
- ✅ GPU argmax in managed memory (0.07ms)
- ✅ CPU spin-wait 修复 (cudaDeviceScheduleBlockingSync)
- ✅ **Continuous Batching** — 全部 kernel 支持 batch_size, GEMV→GEMM 自动切换
- ✅ **多 Batch Decode** — B=256: 386 tok/s, 84× 单请求吞吐
- ✅ **CUDA Graph** — 800+ kernel launches → 1 graph launch, B=64 节省 14ms (+2.0%)
- ✅ **L2 Cache Persistence** — norm weights 持久化 4MB L2
- ✅ **Batched Argmax** — B=256 采样 33ms → 1.7ms

### 推荐优先级 (Phase 10)

| 优先级 | 方向 | 预期收益 | 备注 |
|--------|------|---------|------|
| **P0** | **Chunked Prefill (#9)** | **decode 延迟尖刺 220s → 14s** | 生产环境必需, 否则新请求到达时所有并发用户停顿 |
| **P1** | **Auto-Tune GEMM/GEMV (#14)** | **0-30ms per step** | B=16 异常加速证明 tile 选择空间大; 需 per-(M,N,K) sweep |
| **P1** | **DRAM Bank-Level 优化 (#25)** | **10-24ms (B=1 only)** | 单请求下唯一的挤性能方向, 需 LPDDR5X bank mapping 实验 |
| **P1** | **Prefix Caching (#8)** | **prefill 省分钟级** | 多用户共享 system prompt 场景下 ROI 极高 |
| **P2** | **Head-Group Paged Attn (#5)** | **4-17ms @B32-64** | KV 冗余读取在多并发+长上下文下放大 |
| **P2** | **FP8 KV Cache (#7)** | **内存减半, 扩容** | 最大并发数/上下文长度翻倍, 长上下文带宽收益 |
| **P2** | **SSM State 内存优化 (#28)** | **18.8 GB 节省 @B256** | FP32→BF16 SSM state, max batch 接近翻倍 |

### 基准数据总览

```
当前性能 (Phase 9, CUDA Graph + L2 Persistence):

B=1:    218.2ms/step  →   4.58 tok/s  (235 GB/s, 86% peak)  [bandwidth-bound]
B=32:   268.8ms/step  → 119.06 tok/s  (191 GB/s)            [transition zone]
B=64:   321.7ms/step  → 199.0  tok/s  (159 GB/s)            [transition zone]
B=128:  433.0ms/step  → 295.6  tok/s  (118 GB/s)            [compute-bound]
B=256:  663.0ms/step  → 386.1  tok/s  ( 77 GB/s)            [compute-bound]

Phase 8 → Phase 9 改进:
B=1:   +0.2%  B=32: +0.9%  B=64: +2.0%  B=128: +1.3%  B=256: +0.8%

BF16 单请求理论极限:
B=1:    187.6ms       →  5.33 tok/s   (273 GB/s, 100%)

结论:
- 单请求已达天花板 ~86%, 剩余空间 ~14% 需 bank-level 优化
- 多并发是主战场: B=64 最佳效率 (199 tok/s, 每用户 3.11 tok/s)
- B≥128 进入 compute-bound, Tensor Core tile 优化成关键
- 生产化需要 chunked prefill + prefix caching 保障 SLA
- CUDA Graph 主要价值: CPU 侧 (释放 CPU 用于 IPC/调度), GPU 侧改进 1-2%
```
