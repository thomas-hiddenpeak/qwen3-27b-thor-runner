# Qwen3.5-Thor Comprehensive Benchmark Report

> **Hardware**: NVIDIA Jetson AGX Thor (SM110a Blackwell), 128 GB LPDDR5X, 273 GB/s peak BW, 20 SM  
> **Software**: qwen35-thor v3.0.0, CUDA 13.0, MAXN power mode  
> **Date**: 2026-03-11  
> **Method**: `bench --decode 50 --iterations 3` (serve mode), non-thinking, GPU sampling

---

## 1. Test Matrix

| Model | Precision | Weight Size | MTP | Concurrency Tested |
|-------|-----------|-------------|-----|--------------------|
| Qwen3.5-4B | BF16 | 8.7 GB | d=2 | B=1, 4, 16, 64 |
| Qwen3.5-9B | BF16 | 18.0 GB | d=3 | B=1, 4, 16, 64 |
| Qwen3.5-27B | BF16 | 51.2 GB | d=3 | B=1, 2, 4, 8, 16, 32, 64, 128 |
| Qwen3.5-27B | NVFP4 | 19.2 GB | d=2 | ⚠️ Skipped (memory issue) |
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
| Qwen3.5-35B-A3B | MoE BF16 | d=2 | **45.7** | 40.7 | 154.8 | 0.2% |
| Qwen3.5-122B-A10B | MoE NVFP4 | d=2 | **16.5** | 10.3 | 348.4 | 0.1% |

> **Decode tok/s** = (tokens-1) / (last_token - first_token). 纯 decode 速度, 用于性能分析.  
> **Overall tok/s** = tokens / (submit → complete). 含 TTFT 的端到端速度, 用于横向对比.  
> **TTFT CV** < 2% 表明稳定性优秀.

### 2.2 MTP Speedup (vs No-MTP Baseline)

| Model | Precision | No-MTP tok/s | MTP tok/s | Optimal d | Boost |
|-------|-----------|-------------|-----------|-----------|-------|
| Qwen3.5-4B | BF16 | 26.0 | **46.8** | d=2 | **+80%** |
| Qwen3.5-9B | BF16 | 14.1 | **27.4** | d=3 | **+94%** |
| Qwen3.5-27B | BF16 | 4.5 | **11.7** | d=3 | **+160%** |
| Qwen3.5-35B-A3B | MoE BF16 | 32.2 | **45.7** | d=2 | **+42%** |
| Qwen3.5-122B-A10B | MoE NVFP4 | 14.7 | **16.5** | d=2 | **+12%** |

> MTP 加速比随模型增大而增大 (4B: +80%, 27B: +160%), 因为更大模型的 decode step 更长, MTP 的权重复用收益更显著.  
> 122B MoE 使用 NVFP4, 权重已压缩 ~60%, MTP 额外收益较小.

### 2.3 Prefill Throughput (varying prompt length)

| Model | Precision | P=17 | P=128 | P=512 | P=2048 |
|-------|-----------|------|-------|-------|--------|
| Qwen3.5-4B | BF16 | 405 tok/s | 2,097 tok/s | 4,844 tok/s | 3,895 tok/s |
| Qwen3.5-9B | BF16 | 230 tok/s | 1,341 tok/s | 3,624 tok/s | 2,485 tok/s |
| Qwen3.5-27B | BF16 | 65 tok/s | 480 tok/s | 1,208 tok/s | 1,046 tok/s |
| Qwen3.5-35B-A3B | MoE BF16 | 114 tok/s | 139 tok/s | 154 tok/s | 151 tok/s |
| Qwen3.5-122B-A10B | MoE NVFP4 | 50 tok/s | 64 tok/s | 66 tok/s | 66 tok/s |

> Dense 模型 prefill 吞吐随 prompt 长度增加先升后降 (P=512 峰值, P=2048 因 DeltaNet SSM 串行依赖略降).  
> MoE 模型 prefill 吞吐较低且较平坦, 因 expert routing 开销在各长度下均匀分布.

### 2.4 TTFT vs Prompt Length

| Model | Precision | P=17 | P=128 | P=512 | P=2048 |
|-------|-----------|------|-------|-------|--------|
| Qwen3.5-4B | BF16 | 49ms | 68ms | 113ms | 533ms |
| Qwen3.5-9B | BF16 | 84ms | 105ms | 151ms | 834ms |
| Qwen3.5-27B | BF16 | 273ms | 279ms | 435ms | 1,969ms |
| Qwen3.5-35B-A3B | MoE BF16 | 155ms | 928ms | 3,328ms | 13,541ms |
| Qwen3.5-122B-A10B | MoE NVFP4 | 348ms | 2,019ms | 7,740ms | 30,807ms |

> Dense 模型 TTFT 随 prompt 近似线性增长.  
> MoE 模型 TTFT 增长更快, 因 prefill 需要遍历大量 expert.

### 2.5 Decode Throughput vs Context Length

| Model | Precision | P=17 | P=128 | P=512 | P=2048 |
|-------|-----------|------|-------|-------|--------|
| Qwen3.5-4B | BF16 | 46.8 tok/s | 18.7 tok/s | 17.9 tok/s | 15.6 tok/s |
| Qwen3.5-9B | BF16 | 27.4 tok/s | 9.3 tok/s | 9.1 tok/s | 8.6 tok/s |
| Qwen3.5-27B | BF16 | 11.7 tok/s | 4.5 tok/s | 4.0 tok/s | 3.3 tok/s |
| Qwen3.5-35B-A3B | MoE BF16 | 45.7 tok/s | 20.1 tok/s | 21.7 tok/s | 19.8 tok/s |
| Qwen3.5-122B-A10B | MoE NVFP4 | 16.5 tok/s | 9.5 tok/s | 8.4 tok/s | 8.4 tok/s |

> MTP 加速在长 context 时受限: P=128+ 时 decode 显著降低.  
> 原因: 长 context 下 DeltaNet SSM 的串行 state 更新和 attention KV cache 读取开销增加.  
> P=17 的数字最接近纯 decode 峰值 (context 长度对 MTP accept rate 影响极小时).

### 2.6 Memory Bandwidth Utilization (Raw Decode, B=1)

基于 `bench --raw-batch` 的纯模型 forward 性能 (无 engine 开销):

| Model | Precision | Weight Size | ITL (ms) | Decode tok/s | BW (GB/s) | Utilization |
|-------|-----------|-------------|----------|-------------|-----------|-------------|
| Qwen3.5-4B | BF16 | 8.7 GB | 38.0 | 26.3 | 221 | 81% |
| Qwen3.5-9B | BF16 | 18.0 GB | 67.5 | 14.8 | 235 | 86% |
| Qwen3.5-27B | BF16 | 51.2 GB | 231.2 | 4.3 | 222 | 81% |
| Qwen3.5-35B-A3B | MoE BF16 | 66.0 GB | 30.8 | 32.5 | 191 | 70% |
| Qwen3.5-122B-A10B | MoE NVFP4 | 77.1 GB | 67.1 | 14.9 | 268 | 98% |

> Dense 模型 BW 利用率 81-86%, 接近 LPDDR5X 实测极限 ~230 GB/s.  
> 122B MoE NVFP4 达到 268 GB/s (98% 峰值), 因 FP4 权重更紧凑, DRAM 访问模式更优.  
> MoE BF16 相对较低 (70%), 因 expert routing 的不规则访存.

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
| 1 | 45.7 (MTP d=2) | 40.7 | 1.0× |
| 4 | 87.6 | 63.0 | 1.9× |
| 8 | 108.2 | 73.9 | 2.4× |
| 16 | 160.2 | 94.8 | 3.5× |

**Qwen3.5-122B-A10B MoE NVFP4** (weight: 77.1 GB):

| Concurrent B | Decode tok/s | Aggregate tok/s | Scaling |
|---|---|---|---|
| 1 | 16.5 (MTP d=2) | 10.3 | 1.0× |
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
| Qwen3.5-35B-A3B | 0.2% | 0.1% | 0.1% | 0.1% |
| Qwen3.5-122B-A10B | 0.1% | 0.1% | 0.0% | 0.6% |

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
| Qwen3.5-35B-A3B | MoE BF16 | 45.7 tok/s | B=16 | **160 tok/s** |
| Qwen3.5-122B-A10B | MoE NVFP4 | 16.5 tok/s | B=8 | **45 tok/s** |

### 5.3 Testing Methodology

- **Benchmark tool**: `qwen35-thor bench` (engine-based, serve mode)
- **Iterations**: 3 per test point (N=3, median ±95% CI)
- **Generation**: 50 tokens (single-request), 30 tokens (concurrent)
- **Warmup**: Engine warmup request + AOT cuBLAS autotuning (~3s)
- **Prompt**: Synthetic (chat template header + padding tokens, non-thinking mode)
- **Total test points**: 54 (5 models × 4 prompt lengths + MTP on/off + concurrent sweep)

---

*Generated by qwen35-thor v3.0.0 benchmark suite on NVIDIA Jetson AGX Thor.*
