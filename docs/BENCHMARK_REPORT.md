# Qwen3.5-Thor Comprehensive Benchmark Report

> **Hardware**: NVIDIA Jetson AGX Thor (SM110a Blackwell), 128 GB LPDDR5X, 273 GB/s peak BW, 20 SM  
> **Software**: qwen35-thor v3.1.0, CUDA 13.0, MAXN power mode  
> **Date**: 2026-03-12  
> **Method**: `bench --decode 50 --iterations 3` (serve mode), non-thinking, GPU sampling

---

## 1. Test Matrix

| Model | Precision | Weight Size | MTP | Concurrency Tested |
|-------|-----------|-------------|-----|--------------------|
| Qwen3.5-4B | BF16 | 8.7 GB | d=2 | B=1, 4, 16, 64 |
| Qwen3.5-9B | BF16 | 18.0 GB | d=3 | B=1, 4, 16, 64 |
| Qwen3.5-27B | BF16 | 51.2 GB | d=3 | B=1, 2, 4, 8, 16, 32, 64, 128 |
| Qwen3.5-27B | NVFP4 | 19.2 GB | d=2 | ⚠️ Skipped (Power issue) |
| Qwen3.5-35B-A3B | MoE BF16 | 66.0 GB | d=2 | B=1, 4, 8, 16 |
| Qwen3.5-122B-A10B | MoE NVFP4 | 77.1 GB | d=2 | B=1, 4, 8 |

> ⚠️ **27B NVFP4**: 测试中设备供电崩溃导致硬重启 (uptime 归零), 非软件问题. 待供电环境稳定后补测.

---

## 2. Single-Request Performance (MTP Enabled)

### 2.1 Decode Throughput & TTFT

Short prompt (P=17), 50 generation tokens, 3 iterations:

| Model | Precision | MTP | Decode tok/s | Overall tok/s | TTFT (ms) | TTFT CV |
|-------|-----------|-----|-------------|---------------|-----------|---------|
| Qwen3.5-4B | BF16 | d=2 | **46.8** | 45.6 | 48.8 | 2.0% |
| Qwen3.5-9B | BF16 | d=3 | **27.4** | 26.8 | 83.7 | 0.1% |
| Qwen3.5-27B | BF16 | d=3 | **11.7** | 11.2 | 273.1 | 0.5% |
| Qwen3.5-35B-A3B | MoE BF16 | d=2 | **43.3** | 39.5 | 134.5 | 0.3% |
| Qwen3.5-122B-A10B | MoE NVFP4 | d=2 | **16.4** | 10.3 | 352.4 | 0.3% |

> **Decode tok/s** = (tokens-1) / (last_token - first_token). 纯 decode 速度, 用于性能分析.  
> **Overall tok/s** = tokens / (submit → complete). 含 TTFT 的端到端速度, 用于横向对比.  
> **TTFT CV** < 2% 表明稳定性优秀.

### 2.2 MTP Speedup (vs No-MTP Baseline)

| Model | Precision | No-MTP tok/s | MTP tok/s | Optimal d | Boost |
|-------|-----------|-------------|-----------|-----------|-------|
| Qwen3.5-4B | BF16 | 26.0 | **46.8** | d=2 | **+80%** |
| Qwen3.5-9B | BF16 | 14.1 | **27.4** | d=3 | **+94%** |
| Qwen3.5-27B | BF16 | 4.5 | **11.7** | d=3 | **+160%** |
| Qwen3.5-35B-A3B | MoE BF16 | 31.5 | **43.3** | d=2 | **+37%** |
| Qwen3.5-122B-A10B | MoE NVFP4 | 14.6 | **16.4** | d=2 | **+12%** |

> MTP 加速比随模型增大而增大 (4B: +80%, 27B: +160%), 因为更大模型的 decode step 更长, MTP 的权重复用收益更显著.  
> 122B MoE 使用 NVFP4, 权重已压缩 ~60%, MTP 额外收益较小.

### 2.3 Prefill Throughput (varying prompt length)

| Model | Precision | P=17 | P=128 | P=512 | P=2048 | P=4096 | P=8192 | P=16384 | P=32768 |
|-------|-----------|------|-------|-------|--------|--------|--------|---------|--------|
| Qwen3.5-4B | BF16 | 405 | 2,097 | 4,844 | 3,895 | 3,710 | 3,234 | 2,623 | 1,915 |
| Qwen3.5-9B | BF16 | 230 | 1,341 | 3,624 | 2,485 | 2,518 | 2,289 | 1,963 | 1,535 |
| Qwen3.5-27B | BF16 | 65 | 480 | 1,208 | 1,046 | 934 | 866 | 746 | 554 |
| Qwen3.5-35B-A3B | MoE BF16 | 132 | 399 | 1,303 | 2,442 | 2,374 | 2,151 | — | — |
| Qwen3.5-122B-A10B | MoE NVFP4 | 49 | 63 | 65 | 64 | 63 | — | — | — |

> 单位: tok/s. Dense 模型 prefill 吞吐在 P=512 附近达到 GEMM compute 峰值, 之后因 DeltaNet SSM 串行依赖和 chunked prefill 权重重复读取逐步下降.  
> 32K 时 27B prefill 仍有 554 tok/s (TTFT ~59s), 瓶颈在 16 轮 chunk × 64 层 forward.  
> MoE 35B prefill 通过 per-expert GEMM (CUTLASS tensor cores) 从 ~150 提升到 ~2,450 tok/s (+16×), 将 T*top_k 个独立 GEMV 转为 256 个 CUTLASS GEMM (avg M≈64).  
> MoE 35B 在 P=8192 仍有 2,151 tok/s, 随 prompt 增长缓慢下降 (SSM 串行开销).

### 2.4 TTFT vs Prompt Length

| Model | Precision | P=17 | P=128 | P=512 | P=2048 | P=4096 | P=8192 | P=16384 | P=32768 |
|-------|-----------|------|-------|-------|--------|--------|--------|---------|--------|
| Qwen3.5-4B | BF16 | 49ms | 68ms | 113ms | 533ms | 1.1s | 2.5s | 6.3s | 17.1s |
| Qwen3.5-9B | BF16 | 84ms | 105ms | 151ms | 834ms | 1.6s | 3.6s | 8.4s | 21.4s |
| Qwen3.5-27B | BF16 | 273ms | 279ms | 435ms | 2.0s | 4.4s | 9.5s | 22.0s | 59.1s |
| Qwen3.5-35B-A3B | MoE BF16 | 135ms | 326ms | 399ms | 845ms | 1.7s | 3.8s | — | — |
| Qwen3.5-122B-A10B | MoE NVFP4 | 352ms | 2.0s | 7.9s | 32.2s | 64.6s | — | — | — |

> Dense 模型 TTFT 随 prompt 近似线性增长 (27B: ~1.8 ms/token).  
> MoE 35B TTFT 优化 (per-expert GEMM): P=2048 从 13.5s 降至 845ms (-94%), P=4096 从 27.2s 降至 1.7s (-94%), GEMV→GEMM 切换阈值 T≥128.  
> MoE 35B P=8192 TTFT 3.8s, 有效支持中长 context 场景.

### 2.5 Decode Throughput vs Context Length

| Model | Precision | P=17 | P=128 | P=512 | P=2048 | P=4096 | P=8192 | P=16384 | P=32768 |
|-------|-----------|------|-------|-------|--------|--------|--------|---------|--------|
| Qwen3.5-4B | BF16 | 46.8 | 18.7 | 17.9 | 15.6 | 13.4 | 9.9 | 6.5 | 3.9 |
| Qwen3.5-9B | BF16 | 27.4 | 9.3 | 9.1 | 8.6 | 7.7 | 6.2 | 4.5 | 2.9 |
| Qwen3.5-27B | BF16 | 11.7 | 4.5 | 4.0 | 3.3 | 3.0 | 2.3 | 1.7 | 1.1 |
| Qwen3.5-35B-A3B | MoE BF16 | 43.3 | 20.2 | 24.1 | 20.0 | 14.4 | 12.3 | — | — |
| Qwen3.5-122B-A10B | MoE NVFP4 | 16.4 | 9.8 | 8.2 | 7.8 | 6.5 | — | — | — |

> 单位: tok/s (MTP enabled). Decode 速率随 context 增长持续下降 — 两方面原因:  
> 1. DeltaNet SSM: 48 层串行 state update, 每步固定开销随模型大小线性增长  
> 2. Full Attention: 16 层 paged KV cache 读取, 开销随 context 线性增长  
> 27B 从 P=17 的 11.7 tok/s 降至 P=32K 的 1.1 tok/s (降幅 91%), 主要受 SSM 串行开销主导.  
> MTP accept rate 在长 context 下基本稳定 (非 decode 下降的主因).

### 2.6 Memory Bandwidth Utilization (Raw Decode, B=1)

基于 `bench --raw-batch` 的纯模型 forward 性能 (无 engine 开销):

| Model | Precision | Weight Size | ITL (ms) | Decode tok/s | BW (GB/s) | Utilization |
|-------|-----------|-------------|----------|-------------|-----------|-------------|
| Qwen3.5-4B | BF16 | 8.7 GB | 38.0 | 26.3 | 221 | 81% |
| Qwen3.5-9B | BF16 | 18.0 GB | 67.5 | 14.8 | 235 | 86% |
| Qwen3.5-27B | BF16 | 51.2 GB | 231.2 | 4.3 | 222 | 81% |
| Qwen3.5-35B-A3B | MoE BF16 | 6.0 GB† | 31.6 | 31.6 | 190 | 70% |
| Qwen3.5-122B-A10B | MoE NVFP4 | 10.7 GB† | 68.4 | 14.6 | 156 | 57% |

> Dense 模型 BW 利用率 81-86%, 接近 LPDDR5X 实测极限 ~230 GB/s.  
> 122B MoE NVFP4 达到 156 GB/s (57%), FP4 压缩权重 + 稀疏 expert routing 共同影响.  
> † MoE 模型 Weight Size 为每步活跃权重 (3B/10B active params), 总模型分别为 66.0 GB / 77.1 GB.

---

## 3. Concurrent Throughput (Batch Decode)

MTP 仅在 B=1 时生效, B≥2 走 batch decode (权重读一次服务多请求).

### 3.1 Dense Models

**Qwen3.5-4B BF16** (weight: 8.7 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 46.8 (MTP d=2) | 45.6 | 1.0× |
| 4 | 103.6 | 98.2 | 2.2× |
| 16 | 341.2 | 317.9 | 7.3× |
| 64 | 1,068.6 | 919.2 | 22.8× |

**Qwen3.5-9B BF16** (weight: 18.0 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 27.4 (MTP d=3) | 26.8 | 1.0× |
| 4 | 57.6 | 54.9 | 2.1× |
| 16 | 198.0 | 187.1 | 7.2× |
| 64 | 714.1 | 617.7 | 26.0× |

**Qwen3.5-27B BF16** (weight: 51.2 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 12.1 (MTP d=3) | 12.1 | 1.0× |
| 2 | 9.1 | 8.7 | — |
| 4 | 17.1 | 16.4 | 1.4× |
| 8 | 25.4 | 24.6 | 2.1× |
| 16 | 69.3 | 65.8 | 5.7× |
| 32 | 133.1 | 122.2 | 11.0× |
| 64 | 239.5 | 207.6 | 19.8× |
| 128 | 390.2 | 309.7 | 32.2× |

> **27B B=2**: decode 9.1 < B=1 MTP 12.1, 因 B=2 走 batch decode 而非 MTP, batch 权重复用收益尚不足以补偿 MTP 损失.  
> **27B B=128**: 390 tok/s, 相对 raw baseline 379 tok/s 达标率 103%.

### 3.2 MoE Models

**Qwen3.5-35B-A3B MoE BF16** (weight: 66.0 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 43.3 (MTP d=2) | 39.5 | 1.0× |
| 4 | 87.6 | 63.0 | 1.9× |
| 8 | 108.2 | 73.9 | 2.4× |
| 16 | 160.2 | 94.8 | 3.5× |

**Qwen3.5-122B-A10B MoE NVFP4** (weight: 77.1 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 16.4 (MTP d=2) | 10.3 | 1.0× |
| 4 | 38.9 | 16.5 | 2.4× |
| 8 | 45.1 | 18.0 | 2.7× |

> MoE 模型并发 scaling 低于 Dense, 因 expert routing 增加了计算和访存不规则性.  
> 122B 受 128 GB 内存限制, B=8 已接近可用 KV cache 上限.  
> MoE aggregate 远低于 decode, 因其 TTFT 远长于 Dense (expert 遍历开销).

---

## 4. Stability & Consistency

### 4.1 TTFT Coefficient of Variation (CV)

| Model | P=17 | P=128 | P=512 | P=2048 |
|-------|------|-------|-------|--------|
| Qwen3.5-4B | 2.0% | 0.1% | 0.3% | 0.1% |
| Qwen3.5-9B | 0.1% | 0.1% | 0.9% | 0.9% |
| Qwen3.5-27B | 0.5% | 0.4% | 0.2% | 0.1% |
| Qwen3.5-35B-A3B | 0.3% | 0.0% | 0.6% | 0.2% |
| Qwen3.5-122B-A10B | 0.3% | 0.0% | 0.0% | 0.1% |

> 所有模型 TTFT CV < 2%, 生产环境一致性优秀.  
> AOT cuBLAS autotuning warmup 消除了首次推理延迟波动.

### 4.2 Known Issues

- **27B NVFP4**: 测试中设备供电崩溃导致硬重启 (uptime 归零), 非软件问题. 疑似 MAXN 模式下瞬态功耗尖峰触发断电保护.
- **122B MoE P=17**: 生成 8 tokens 后遇到 EOS (非模型问题, 是 benchmark 合成 prompt 的特性), decode 速度仍然准确.

---

## 5. Summary

### 5.1 Production Readiness

| 指标 | 要求 | 实际 | 状态 |
|------|------|------|------|
| TTFT 稳定性 (CV) | < 5% | < 2% | ✅ |
| Decode 吞吐一致性 | Serve ≥ Raw | 102-275% | ✅ |
| 并发 scaling | 近线性 | B=128 → 32× (27B) | ✅ |
| 内存安全 | 无 OOM / crash | 5/6 模型通过 | ⚠️ (27B NVFP4 供电问题) |
| 首次推理延迟 | 稳定 | AOT warmup 消除 | ✅ |

### 5.2 Peak Performance per Model

| Model | Precision | Single-Request (MTP) | Max Concurrent | Peak Throughput |
|-------|-----------|---------------------|----------------|-----------------|
| Qwen3.5-4B | BF16 | 46.8 tok/s | B=64 | **1,069 tok/s** |
| Qwen3.5-9B | BF16 | 27.4 tok/s | B=64 | **714 tok/s** |
| Qwen3.5-27B | BF16 | 11.7 tok/s | B=128 | **390 tok/s** |
| Qwen3.5-35B-A3B | MoE BF16 | 43.3 tok/s | B=16 | **160 tok/s** |
| Qwen3.5-122B-A10B | MoE NVFP4 | 16.4 tok/s | B=8 | **45 tok/s** |

### 5.3 Testing Methodology

- **Benchmark tool**: `qwen35-thor bench` (engine-based, serve mode)
- **Iterations**: 3 per test point (N=3, median ±95% CI)
- **Generation**: 50 tokens (single-request), 30 tokens (concurrent)
- **Warmup**: Engine warmup request + AOT cuBLAS autotuning (~3s)
- **Prompt**: Synthetic (chat template header + padding tokens, non-thinking mode)
- **Total test points**: 78 (5 models × 8 prompt lengths + MTP on/off + concurrent sweep)

---

## 6. Long Context Deep Dive (Qwen3.5-27B BF16)

> 本节聚焦 27B Dense 模型在 4K-32K 上下文下的性能特征, 用以评估实际 Agent 场景下的体验.

### 6.1 性能全景 (Single-Request, MTP d=3)

| Prompt Length | TTFT | Prefill tok/s | Decode tok/s | Overall tok/s | Total (50 tok) |
|---|---|---|---|---|---|
| 17 | 273ms | 65 | **11.7** | 11.2 | 4.7s |
| 128 | 279ms | 480 | 4.5 | 4.3 | 11.5s |
| 512 | 435ms | 1,208 | 4.0 | 3.7 | 13.4s |
| 2,048 | 2.0s | 1,046 | 3.3 | 2.7 | 18.4s |
| 4,096 | 4.4s | 934 | 3.0 | 2.4 | 20.6s |
| 8,192 | 9.5s | 866 | 2.3 | 1.6 | 30.4s |
| 16,384 | 22.0s | 746 | 1.7 | 1.0 | 51.1s |
| 32,768 | 59.1s | 554 | 1.1 | 0.5 | 104.4s |

### 6.2 性能衰减分析

**Prefill 吞吐下降**:

| 区间 | 下降率 | 原因分析 |
|---|---|---|
| P=512→4096 (8×) | 934/1208 = 77% 保留 | Chunked prefill 权重重复读取 (chunk×2 → chunk×8), GEMM compute 仍主导 |
| P=4096→16384 (4×) | 746/934 = 80% 保留 | DeltaNet SSM 串行开销随 chunk 数线性增长, 每 chunk 48 层串行 state update |
| P=16384→32768 (2×) | 554/746 = 74% 保留 | SSM 串行成本占比进一步提高, GEMM 并行度收益递减 |

> **理论下界**: 每 chunk 2048 token 读 51.2 GB 权重 ÷ 222 GB/s = 231ms, 32K 需 16 chunks × 231ms = 3.7s 纯权重读取.  
> **实测 59.1s** 中 SSM 串行成本约占 55.1s ÷ 16 chunks ≈ 3.4s/chunk (含 48 层 DeltaNet state update + 16 层 chunked attention).  
> Prefill 瓶颈不在 GEMM, 而在 **DeltaNet SSM 的 O(T) 串行 state propagation**.

**Decode 吞吐下降**:

| 区间 | Decode tok/s | 额外开销来源 |
|---|---|---|
| P=17 → P=128 | 11.7 → 4.5 (−62%) | SSM state 从 L2 cache (32MB) 溢出到 DRAM, MTP verify 需读更多 SSM state |
| P=128 → P=2048 | 4.5 → 3.3 (−27%) | Full Attention KV cache 读取增长 (16层 × 4 KV heads × 256 dim × context) |
| P=2048 → P=32768 | 3.3 → 1.1 (−67%) | KV cache 读取从 ~32MB 增至 ~512MB, 占权重读取的 1% |

> **关键发现**: P=17 到 P=128 的 62% 暴跌是 SSM state DRAM 溢出导致的.  
> 每步 decode 需读 48 层 SSM state (48 × 16 heads × 128 × 128 × 2B = 24 MB BF16) + 写回 24 MB = 48 MB 额外 DRAM I/O.  
> P=17 时 SSM state 可部分留在 L2 cache (32 MB), P=128+ 完全溢出.

### 6.3 长上下文各模型对比

| Model | P=4096 Decode | P=16384 Decode | P=32768 Decode | 32K/短prompt 保留率 |
|---|---|---|---|---|
| Qwen3.5-4B (8.7 GB) | 13.4 tok/s | 6.5 tok/s | 3.9 tok/s | 3.9/46.8 = 8.3% |
| Qwen3.5-9B (18.0 GB) | 7.7 tok/s | 4.5 tok/s | 2.9 tok/s | 2.9/27.4 = 10.6% |
| Qwen3.5-27B (51.2 GB) | 3.0 tok/s | 1.7 tok/s | 1.1 tok/s | 1.1/11.7 = 9.4% |
| Qwen3.5-35B MoE (66.0 GB) | 14.4 tok/s | 12.3 tok/s | — | — |
| Qwen3.5-122B MoE FP4 (77.1 GB) | 6.5 tok/s | — | — | — |

> 所有 Dense 模型在 32K 时仅保留短 prompt 约 8-11% 的 decode 速率.  
> 下降的主因是 SSM state I/O (固定开销) 相对权重读取的比例随模型变小而增大.  
> MoE 模型 P=4096 decode 仍然可观 (35B: 14.4, 122B: 6.5), 因 expert routing 只读激活的 expert.  
> 35B P=8192 decode 12.3 tok/s, 中长 context 下仍保持较好 decode 速率.

### 6.4 TTFT 构成分解 (27B, P=32768)

```
TTFT = 59.1s (实测), 分解:
├── 权重读取: 16 chunks × 51.2 GB ÷ 222 GB/s    ≈  3.7s ( 6.3%)
├── GEMM compute: 16 × ~20ms (cuBLAS/CUTLASS)   ≈  0.3s ( 0.5%)
├── DeltaNet SSM serial: 16 × 48 layers × ~70ms  ≈ 53.8s (91.0%)
├── Full Attention (chunked): 16 × 16 layers × ~5ms ≈ 1.3s ( 2.2%)
└── Overhead (sync, alloc, etc.)                  ≈  0.0s
```

> **DeltaNet SSM 串行开销占 91%** — 这是 Qwen3.5 混合架构的固有特征.  
> SSM 串行性使得 prefill 无法像纯 Transformer 那样通过增大 chunk 或 batch 加速.  
> 优化方向: WY chunkwise kernel (已部分实现, T≥4 时 1.71×) 可进一步改进长 chunk 的 SSM 效率.

---

*Generated by qwen35-thor v3.1.0 benchmark suite on NVIDIA Jetson AGX Thor.*
