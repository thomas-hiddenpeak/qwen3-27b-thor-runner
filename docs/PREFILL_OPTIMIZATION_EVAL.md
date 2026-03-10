# Prefill 优化空间评估报告

## 1. 当前 Prefill 性能基线 (27B BF16)

| Prompt Len (T) | TTFT (ms) | Forward (ms) | LM Head (ms) | Prefill tok/s | ms/tok |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 17 | 357.0 | 345.9 | 11.0 | 48 | 20.3 |
| 32 | 360.7 | 349.8 | 10.8 | 89 | 10.9 |
| 64 | 379.7 | 368.5 | 11.0 | 169 | 5.8 |
| 128 | 332.1 | 320.9 | 10.9 | 385 | 2.5 |
| 256 | 343.5 | 332.8 | 10.7 | 745 | 1.3 |

**关键观察**: Forward 时间在 T=17→256 范围内高度稳定 (~320-370ms), ms/tok 随 T 线性下降。T=128 的 TTFT **低于** T=17 和 T=64, 说明在此范围内 cuBLAS 效率随 M 增大而改善。

---

## 2. 性能瓶颈分析

### 2.1 理论下界计算

**权重数据量** (27B BF16):
- 每层 LinearAttn 权重: (16480 + 6144 + 34816 + 17408) × 5120 × 2B ≈ 733 MB
- 每层 FullAttn 权重: (14336 + 6144 + 34816 + 17408) × 5120 × 2B ≈ 713 MB
- 总权重: 48 × 733 + 16 × 713 ≈ 46.6 GB

**带宽瓶颈** (LPDDR5X):
- 峰值带宽: 273 GB/s
- 实测 decode 有效带宽: ~224 GB/s (82%)
- 理论最低 forward 时间: 46.6 GB / 273 = **170.7ms** (峰值), 46.6 / 224 = **208ms** (实测效率)

**实测 vs 理论**:
| T | Forward 实测 | 理论下界 (224 GB/s) | 效率 | 有效 BW |
|:-:|:-:|:-:|:-:|:-:|
| 17 | 345.9ms | 208ms | 60.1% | 134.7 GB/s |
| 128 | 320.9ms | 208ms | 64.8% | 145.2 GB/s |
| 256 | 332.8ms | 208ms | 62.5% | 140.0 GB/s |

**关键结论**: Prefill forward 有效带宽仅 135-145 GB/s, 远低于 decode 路径的 224 GB/s (~60-65% 效率)。

### 2.2 效率差距根因

#### A. GEMM 路径低效 (M=9-127 → cuBLAS, M≥128 → CUTLASS)

对于 T=17, 每个 GEMM 的 M=17, 走 cuBLAS 路径 (M=9-127 分支)。cuBLAS 对小 M:
- 内部 tile 可能导致权重多次读取
- kernel 调度开销比例高
- 无法像 decode (M=1) 的 scattered GEMV 那样优化 DRAM 访问模式

对于 T=256, M=256 走 CUTLASS SM110 路径 (TileShape=128×128×64, Cluster=2×2×1):
- CUTLASS 对 M=256 仅能切 2 个 tile (M=128×2), 硬件利用不佳
- 但效率反而低于 T=128 (332.8 > 320.9), 可能因为 CUTLASS TMA 开销和 M padding

T=128 是"甜蜜点" — cuBLAS 对 M=128 的 tile 效率最优。

#### B. 非 GEMM 计算开销

per-layer overhead 估算 (T=17):
- 64 × cudaStreamSynchronize ≈ 0.64ms
- ~1264 kernel launches × 5μs ≈ 6.3ms
- Conv1d + DeltaNet SSM + Norms + Attention ≈ 10-15ms
- 总非 GEMM 开销: ~20ms

这仅占 forward 的 ~6%, **非主要瓶颈**。

#### C. FullAttn Prefill Attention (多 kernel 实现)

对于 T≥256 首块, `invoke_prefill_attention` 使用 per-KV-group 循环:
```
per KV group (4 groups):
  extract_kv_head: 1 kernel
  transpose: 1 kernel
  gather_q: 1 kernel
  score GEMM: 1 cuBLAS/CUTLASS call
  causal_softmax: 1 kernel
  output GEMM: 1 call
  scatter: 1 kernel
```
每层 FullAttn ≈ 28 kernel launches × 16 层 = **448 kernel launches** (仅 attention 部分)。

对于 T<256, 首块走 `invoke_paged_attention` (decode 风格 paged attention), 适用性有限。

---

## 3. 优化机会评估

### 3.1 高收益 — 值得投入

#### ① 扩展 Multi-Row GEMV 到 M=9-64 范围

**现状**: M=2-8 使用自研 `gemv_multirow_kernel_scattered` (register-based, L2 cache for A, zero SMEM), M=9-127 回退到 cuBLAS。

**方案**: 将 multi-row GEMV 扩展到 M=9-64 (或更大):
- M=9-16: 寄存器可直接扩展 (register pressure 可控)
- M=17-64: 需要 A rows tiling (L2 cache 大小 = 32 MB, A[64, 5120] = 640KB fits in L2)
- 核心优势: **每列 B 只读一次** (vs cuBLAS 可能多次)

**预期收益**:
- T=17 forward: 345.9ms → ~270ms (cuBLAS→custom GEMV 30% 提升, 假设 BW 从 135→175 GB/s)
- T=64 forward: 368.5ms → ~290ms (类似)
- 对 T=128+ 影响小 (CUTLASS/cuBLAS 已高效)

**风险**: 中。M > 8 时寄存器文件压力增大, 可能降低 occupancy。需要 A rows SMEM/L2 tiling 策略。

**工作量**: 中。基于现有 multi-row GEMV 框架扩展, 估计需要 3-5 天调优。

#### ② Fused Prefill Flash Attention (替代 multi-kernel 循环)

**现状**: 
- T≥256: `invoke_prefill_attention` 内多 kernel 循环 (28 launches/层 × 16 层 = 448 launches)
- 已实现 fused prefill attention 但**未启用** (copilot-instructions.md 记载)
- T<256: 用 `invoke_paged_attention` (decode 风格)

**方案**: 
1. 启用已有的 fused prefill attention kernel
2. 或实现 flash-attention 风格的 tiled attention (Q/K/V tile-based, online softmax)
3. GQA: 同 KV head 的 6 Q head 可合并计算

**预期收益**:
- 448 launches → 16 launches (每层 1 个 attention kernel)
- Launch overhead 节省: ~2.2ms
- 计算效率提升: 减少中间结果 GMEM 读写 (score GEMM 输出 [T,T] 不需要写回)
- T=256: forward 332.8 → ~325ms (attention 占比不大, ~2-3%)
- T=1024+ (chunked prefill): 显著收益

**风险**: 低-中。已有实现可参考, 但需验证数值正确性。

**工作量**: 中。已有实现基础, 需要调优 tile size 和 SM110a 适配。

### 3.2 中等收益 — 可选投入

#### ③ LinearAttn T>1 A+B 投影超级合并

**现状**: T>1 时 QKV+Z 合并为一个 GEMM [T, 5120]→[T, 16384](已优化), 但 A [T,48] 和 B [T,48] 仍各自一个 cuBLAS call。

**方案**: 将 A+B 追加到 QKV+Z 合并中: [T, 5120]→[T, 16480], 与 T=1 路径一致。

**预期收益**:
- 节省 2 cuBLAS launches × 48 layers = 96 launches (~0.5ms)
- 减少一次 norm_out [T, 5120] 读取 (A+B 复用 GEMM 输入)
- T=128: 节省 ~1-2ms

**风险**: 低。权重已连续排列 (T=1 已证明), 只需调整 T>1 分支的 GEMM N 维度和输出分割。

**工作量**: 低。~1 天。

#### ④ cuBLAS → CUTLASS 切换阈值调优

**现状**: M < 128 全部走 cuBLAS, M ≥ 128 走 CUTLASS。但 CUTLASS SM110 的 TileShape=128×128×64, 理论上 M=64 也可以 (1 个 M tile)。

**方案**: 
- 尝试降低 CUTLASS 阈值到 M=64
- 或为 M=32-127 实现专用 TileShape (如 64×128×64)
- 对齐 M padding 策略

**预期收益**:
- 可能在 M=64-127 范围获得 10-20% GEMM 加速
- T=64-128 forward: ~5-10ms 改善

**风险**: 中。CUTLASS 对小 M 的 TMA 描述符创建可能失败 (已有 guard)。

**工作量**: 中。需要测试多种 TileShape 配置。

### 3.3 低收益 — 不推荐当前投入

#### ⑤ per-layer sync 消除
- 已验证不可移除 (GPU hard-reset 风险)
- 成本仅 0.64ms/forward
- **不推荐**

#### ⑥ Fused RMSNorm + GEMM for T>1
- T>1 已经将 RMSNorm 输出用作 GEMM 输入
- 融合需要修改 CUTLASS/cuBLAS epilogue, 复杂度高
- RMSNorm 本身仅 ~0.02ms
- **收益不抵复杂度**

#### ⑦ LM Head 优化
- 当前 ~10.7ms (fused RMSNorm + GEMV, 即 decode 路径的单 token GEMV)
- 已是最优实现
- **无优化空间**

---

## 4. Prefill 数据流架构 (当前实现总览)

```
engine.cpp::process_request()
│
├─ Prefix Cache Lookup (SSD) → cached_tokens
├─ Chunked Prefill Loop (max_chunk_size=256):
│  ├─ Block 分配 + Block Table 上传
│  ├─ Embedding Lookup
│  ├─ Vision Encoder (if applicable)
│  ├─ Position IDs + Context Lens
│  └─ model.forward_prefill() ← 主要耗时
│     └─ 64 layers × {
│        ├─ FullAttn (16 layers):
│        │  ├─ RMSNorm                    [1 kernel]
│        │  ├─ Merged QKV GEMM            [1 CUTLASS/cuBLAS]
│        │  ├─ Deinterleave 3-way         [1 kernel]
│        │  ├─ Fused QK_norm + RoPE       [1 kernel]
│        │  ├─ Write KV Cache             [1 kernel]
│        │  ├─ Attention                  [1-28 kernels] ← 优化重点
│        │  ├─ Sigmoid_mul + O_proj       [1-2 kernels]
│        │  ├─ Fused Add+RMSNorm          [1 kernel]
│        │  └─ MLP (fused gate_up+SwiGLU+down) [3 calls]
│        │
│        └─ LinearAttn (48 layers):
│           ├─ RMSNorm                    [1 kernel]
│           ├─ QKV+Z merged GEMM          [1 CUTLASS/cuBLAS]
│           ├─ Deinterleave               [1 kernel]
│           ├─ A GEMM [T,48]              [1 cuBLAS] ← 可合并
│           ├─ B GEMM [T,48]              [1 cuBLAS] ← 可合并
│           ├─ Conv1d + SiLU              [1 kernel]
│           ├─ GDN DeltaNet (WY/serial)   [1 kernel]
│           ├─ Fused norm+silu gate       [1 kernel]
│           ├─ Out projection GEMM        [1 CUTLASS/cuBLAS]
│           ├─ Fused Add+RMSNorm          [1 kernel]
│           └─ MLP (fused gate_up+SwiGLU+down) [3 calls]
│        }
│        + cudaStreamSynchronize          [per layer]
│
├─ Fused Final RMSNorm + LM Head GEMV    [1 call]
├─ GPU Sampling                           [1 kernel]
└─ Cache Store (SSD)
```

**Kernel Launch 统计** (T=256, 首块):
- FullAttn: ~13 launches/层 × 16 层 = 208 (T≥256 attention: ~28 → 更多)
- LinearAttn: ~13 launches/层 × 48 层 = 624
- 总计: ~1000-1300 launches
- cudaStreamSynchronize: 64 次

---

## 5. 业务插件接口建议

用户计划在 prefill 与 decode 前后添加业务插件。建议在以下位置提供 hook 点:

### Prefill 阶段 Hook
1. **pre_prefill_hook**: 在 embedding 之后, forward_prefill 之前
   - 用途: 输入预处理、隐藏状态注入、prompt 增强
   - 数据: `hidden_states [T, 5120]`, `pos_ids [T]`

2. **per_layer_hook**: 在每层 forward 之后, cudaStreamSynchronize 之后
   - 用途: 中间层监控、特征提取、动态路由
   - 数据: `hidden_states [T, 5120]`, `layer_idx`

3. **post_prefill_hook**: 在 forward_prefill 完成之后, lm_head 之前
   - 用途: 输出后处理、hidden state 缓存
   - 数据: `hidden_states [T, 5120]`

### Decode 阶段 Hook
1. **pre_decode_hook**: 在 embedding 之后, forward_decode 之前
2. **post_decode_hook**: 在 forward_decode 之后, sampling 之前

### 实现建议
- 使用函数指针或 `std::function` 回调, 默认为空 (零开销)
- Hook 在 cudaStreamSynchronize 之后调用 (数据已在 GPU, 可安全访问)
- 提供 CPU 和 GPU 两种 hook 接口

---

## 6. 总结与优先级建议

### 当前状态评分
| 维度 | 评分 | 说明 |
|------|------|------|
| 正确性 | ✅ 95% | 16 tests 全通过, 多轮验证稳定 |
| 稳定性 | ✅ 90% | per-layer sync 保障, max_chunk_size=256 经验值 |
| Decode 性能 | ✅ 85% | 224 GB/s / 273 峰值 = 82% |
| Prefill 性能 | ⚠️ 55% | 135-145 GB/s / 273 峰值 = 50-53% |

### 优化优先级
| 优先级 | 优化项 | 预期 TTFT 改善 | 工作量 | 风险 |
|--------|--------|----------------|--------|------|
| P0 | 扩展 multi-row GEMV (M=9-64) | -15~25% (T≤64) | 中 | 中 |
| P1 | Fused flash attention | -2~5% (T≥256) | 中 | 低-中 |
| P2 | A+B 投影超级合并 (T>1) | -0.5~1% | 低 | 低 |
| P3 | CUTLASS 切换阈值调优 | -2~5% (M=64-127) | 中 | 中 |

### 关键结论

1. **Prefill 性能的主要瓶颈是 GEMM 效率, 不是 kernel launch 或 sync 开销**。cuBLAS 在 M=9-127 范围的带宽利用率仅 50-53%, 远低于 decode 路径的 82%。

2. **Forward 时间高度稳定** (~330-370ms), 与 prompt 长度几乎无关 (T=17→256 仅变化 8%)。这意味着 TTFT 的改善主要依赖于 GEMM 效率提升, 而非减少计算量。

3. **T=128 是当前最高效的 prompt 长度** (320.9ms, TTFT 最低), 因为 cuBLAS 在 M=128 的 tile 效率最佳。

4. **对于短 prompt (T<64), 优化空间最大** — 自研 small-M GEMM 有望获得 15-25% TTFT 改善。

5. **Prefill 路径代码质量高, 架构清晰**, 支持 chunked prefill + prefix cache + SSD streaming + vision 等完整功能。适合在当前架构上添加业务插件 hook。
