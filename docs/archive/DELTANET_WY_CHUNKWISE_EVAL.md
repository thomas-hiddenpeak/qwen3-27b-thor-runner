# DeltaNet WY Chunkwise 并行算法 — SM110 硬件适配评估

## 结论 (TL;DR)

WY chunkwise 在 SM110a (Jetson Thor) 上经过多轮优化后仍比串行 kernel **慢 ~2.5×**（初始原型慢 7×，优化后提升 2.8×）。核心瓶颈不是算法的二次方开销，而是**数据加载和 S_local 读取**（占总时间 52%），这些是该算法固有的。**目前不适合替代串行 kernel 投入生产。**

### 已完成的优化及收益
| 优化 | 技术手段 | 收益 |
|------|---------|------|
| Bank conflict 消除 | chunk_K/Q padding (kd→kd_pad=129) | +78% (14.6→8.2ms@T=256) |
| 全线程协作 IKK/QK_mask | 128 线程合作代替 64 线程行并行 | 含上 |
| Chunk size 调优 | C=64→C=16, smem 178→90KB, 2 blocks/SM | +40% (8.2→5.8ms@T=256) |
| **累计** | | **从 14.6ms 降至 5.8ms (2.5× 提升)** |

---

## 1. 算法概述

WY chunkwise 将 DeltaNet 的 O(T) 串行 token 递推分解为 O(T/C) 个 chunk，每个 chunk 内 C 个 token 通过矩阵运算并行处理:

```
Per chunk (C=16):
  P0: 加载 K, Q, V, 计算 L2 norms, α, β, γ_cum
  P1: V'[t,j] = V[t,j] - exp(γ[t]) * dot(K[t,:], S[:,j])         tiled KS
  P2a: IKK[s,t] = δ(s,t) + β·exp(Γ)·dot(K[s],K[t])              C×C 下三角
  P2b: T_mat = IKK^{-1} · diag(β)                                 forward substitution
  P3: new_v = T_mat @ V'                                           C×vd
  P4: QK_mask + output = exp(γ)·Q@S + QK_mask @ new_v             C×vd
  P5: S_new = exp(γ_block) S_old + Σ K[s]·new_v[s]               kd×vd 状态更新
```

参考: FlashInfer `blockwise_delta_rule()` (reference_delta_rule.py), fla DeltaNet WY 论文。

---

## 2. 优化历程

### Round 1: Bank Conflict + 全线程协作

**问题**: chunk_K/Q [C, kd=128] 的行步长 128, 128 mod 32 = 0 → 所有行的同一列映射到同一 bank → **N-way bank conflict**。IKK/QK_mask 构建仅 64/128 线程活跃。

**修复**:
- chunk_K/Q 使用 kd_pad = kd + 1 = 129 的 padded 步长。bank(s,d) = (s*129+d) % 32 = (s+d) % 32 → 无冲突
- IKK/QK_mask 从 `if (j < C) { for t in C }` (64线程行并行) 改为 `for (idx = j; idx < C*C; idx += 128)` (全128线程协作)

**效果** (T=256):
```
Chunkwise: 14.6ms → 8.2ms (+78%)
T=64 单 chunk: 从 0.46× 提升到 1.17× (超越串行!)
```

### Round 2: 分阶段计时 + Chunk Size 调优

添加 clock64() 分阶段计时后发现:

| Phase | C=64 占比 | 说明 |
|-------|-----------|------|
| P0: Data loading | 22.2% | 逐 token norm 不可避免 |
| **P1: KS/V' compute** | **29.2%** | **#1 瓶颈: tiled S_local 读取** |
| P2a: IKK construct | 5.8% | bank fix 后效果好 |
| P2b: Fwd sub + beta | 4.4% | 已足够快 |
| P3: T_mat @ V' | 3.8% | 已足够快 |
| **P4: QK_mask+output** | **20.1%** | QS recompute + intra-attn |
| P5: State update | 14.5% | S_local 状态更新 |

**关键发现**: P2a+P2b+P3 仅占 14%, 这些是与 C² 相关的开销。**真正的瓶颈是 P0+P1+P4+P5 (86%)**, 它们与 C 无关或仅线性相关。

据此进行 chunk size 扫描:

| C | Smem (KB) | Blocks/SM | T=256 (ms) | Speedup |
|---|-----------|-----------|------------|---------|
| 8 | 76.9 | **2** | 6.36 | 0.34× |
| **16** | **89.8** | **2** | **5.85** | **0.40×** |
| 32 | 117.1 | 1 | 8.19 | 0.29× |
| 64 | 177.8 | 1 | 8.74 | 0.27× |

**C=16 最优**: smem = 90KB → 2 blocks/SM (匹配串行 kernel 的占用率), 将 wave 数从 3 降至 2。

### Round 2.5: Tiled QS 实验 (失败)

尝试将 P4 的 QS 计算从逐 token 改为 tiled (与 P1 的 KS 对称), 但 **反而变慢 85%** (921→1710μs)。原因: 打破了原始代码中 QS 和 output 写入之间的自然流水线重叠。已回退。

---

## 3. 最终性能数据

### 3.1 正确性验证 (所有 chunk size)
```
Output: max abs error = 1.68e-4, relative error = 2.86e-3
SSM state: max abs error = 6.86e-5, relative error = 2.74e-4
Status: ✅ PASS (C=8/16/32/64 均通过)
```

### 3.2 单层性能对比 (最优 C=16)

| T (tokens) | Serial (ms) | C=16 (ms) | Speedup | C=64 (ms) | C=64 Speedup |
|------------|-------------|-----------|---------|-----------|-------------|
| 64         | 2.43        | 1.54      | 1.58×   | 2.20      | 1.10×       |
| 128        | 1.23        | 3.04      | 0.41×   | 4.28      | 0.28×       |
| 256        | 2.34        | 5.85      | 0.40×   | 8.73      | 0.27×       |
| 512        | 4.19        | 11.52     | 0.36×   | 17.18     | 0.24×       |
| 1024       | 8.52        | 22.89     | 0.37×   | 34.88     | 0.24×       |

注: T=64 serial 时间受冷启动影响波动较大

### 3.3 Phase 开销分析 (C=16, T=256, block 0)

| Phase | 耗时(μs) | 占比 | Per-chunk (μs) | 特征 |
|-------|----------|------|----------------|------|
| P0: Data loading | 634 | 22.1% | 39.6 | O(T), 逐 token norm |
| P1: KS/V' compute | 857 | 29.9% | 53.6 | O(T·kd), tiled S 读取 |
| P2a: IKK construct | 48 | 1.7% | 3.0 | O(C²·kd), bank-free |
| P2b: Fwd sub + β | 79 | 2.8% | 4.9 | O(C²), 列并行 |
| P3: T_mat @ V' | 60 | 2.1% | 3.8 | O(C²·vd) |
| P4: QK+output | 494 | 17.2% | 30.8 | O(C²·kd + C·kd·vd) |
| P5: State update | 698 | 24.3% | 43.6 | O(C·kd·vd), per-chunk S |
| **Total** | **2869** | **100%** | **179.3** | |

### 3.4 48层外推 (T=1024)
```
Serial  (48 layers): 407 ms
Chunkwise (48 layers): 1099 ms  (C=16)
                       1674 ms  (C=64)
额外开销: +692 ms (C=16) vs +1267 ms (C=64)
```

---

## 4. 根因分析

### 4.1 为什么仍然慢 2.5×?

**占时间 52% 的 P0+P1 是算法固有开销, 与串行 kernel 等价但不能更快。**

| 操作 | Serial (per token) | Chunkwise (per chunk C=16) | 对比 |
|------|-------------------|---------------------------|------|
| S_local 读取 (kS/QS) | kd × 2 = 256 次 | kd × (C/8 + C) ≈ 2050 次 | 相当 |
| S_local 写入 (update) | kd = 128 次 | kd = 128 次 | 相同 |
| 全局数据加载 | 3 BF16/thread/token | 同 | 相同 |
| 额外: IKK+fwd+T@V' | 0 | ~12μs/chunk | **纯开销** |
| 额外: QK_mask | 0 | ~10μs/chunk | **纯开销** |
| 同步次数 | ~5/token | ~40/chunk | 更多 |

串行 kernel 的 y_j 计算 "免费" 搭乘 S 更新: `y_j += new_s * q_hat_s[i]` 使用已经在寄存器中的 `new_s`。Chunkwise 必须单独计算 QS = Q·S_old (P4), 增加一整遍 S_local 读取。

### 4.2 占用率已匹配

| 配置 | Smem/block | Blocks/SM | Warps/SM |
|------|-----------|-----------|----------|
| Serial | 66 KB | 2 | 8 |
| Chunkwise C=16 | 90 KB | 2 | 8 |
| Chunkwise C=64 | 178 KB | 1 | 4 |

C=16 已匹配串行 kernel 的占用率。剩余差距完全来自算法层面的额外工作。

### 4.3 FlashInfer SM90 为什么快？

FlashInfer 的 SM90 kernel 使用:
- **WGMMA**: 512 线程的 WarpGroup MMA, IKK/QK_mask 的小矩阵乘法通过 Tensor Core 高效计算
- **TMA**: 硬件异步 bulk 数据加载, 与计算完全重叠
- **更高占用率**: SM90 配置灵活

SM110 有 5th-gen Tensor Core 和 TMA, 但不是 SM90 的 WGMMA。移植 FlashInfer 的 SM90 tiling 到 SM110 MMA 指令需要重写整个 kernel。

---

## 5. 已尝试的优化和结论

| 优化 | 状态 | 效果 | 教训 |
|------|------|------|------|
| Bank conflict fix (kd_pad=129) | ✅ 成功 | +78% | N-way smem bank conflict 是重大性能杀手 |
| 全线程协作 IKK/QK_mask | ✅ 成功 | 含上 | 让所有 128 线程参与, 不要浪费 50% |
| Chunk size C=16 | ✅ 成功 | +40% | 小 C 减少 smem→更高占用率 > 二次方开销节省 |
| Tiled QS (P4) | ❌ 失败 | -85% | 打破流水线重叠; 结构改变不一定好 |
| C=8 (更小 chunk) | ❌ 不如C=16 | P5 增加 | 太多 chunk boundary 开销 |

---

## 6. 后续可探索方向

### 6.1 当前 kernel 优化 (低风险, 低收益)
| 方向 | 预期收益 | 说明 |
|------|---------|------|
| 合并 P1 KS + P4 QS | ~5% | 共享 S_local 读取, 但增加 P1 指令密度 |
| P0 异步数据加载 | ~5% | cp.async 预取 K/Q/V, 与 norm 计算重叠 |
| P5 双缓冲 | ~3% | 重叠当前 chunk 的 P5 写入与下一 chunk 的 P0 读取 |

### 6.2 架构级优化 (高风险)
| 方向 | 预期收益 | 说明 |
|------|---------|------|
| SM110 wmma for IKK/QK_mask | ~2% | P2a 仅 1.7%, 即使 10× 加速也只省 0.2μs/chunk |
| SM110 TMA for 数据加载 | ~10% | 消除 P0 手动加载开销 |

### 6.3 非 WY 方向 (推荐)
| 方向 | 预期收益 | 说明 |
|------|---------|------|
| 串行 kernel 双缓冲预取 | 10-20% | cp.async K/Q/V overlap (比 WY 更佳 ROI) |
| Prefill GEMM 优化 | 独立项 | GEMM 占 TTFT ~55% |

---

## 7. 实验配置

- **硬件**: Jetson AGX Thor, SM110a, 20 SMs, 228KB smem/SM, 128GB LPDDR5X
- **GPU 时钟**: 1575 MHz (GPC), MAXN 功耗模式
- **CUDA**: 13.0 V13.0.48
- **编译**: `--use_fast_math -O3 -gencode arch=compute_110a,code=sm_110a`
- **模型参数**: nkh=16, kd=128, nv_per_kh=3, vd=128, nv=48

### Kernel 资源使用

| Kernel | 寄存器 | Stack | Spill | Smem | Blocks/SM |
|--------|--------|-------|-------|------|-----------|
| Chunkwise (C=16) | 131 | 64B | 0 | 90KB | 2 |
| Chunkwise (C=64) | 131 | 64B | 0 | 178KB | 1 |
| Serial | 56 | 0 | 0 | 66KB | 2 |

### 文件
- 原型代码: `src/engine/deltanet_chunkwise.cu`
- 生成二进制: `build/deltanet_bench`
- 使用: `./build/deltanet_bench --tokens 256 --chunk-size 16 --profile`

---

## 8. 决策建议

| 问题 | 答案 |
|------|------|
| WY chunkwise 是否适合生产替代串行 kernel? | **否**, 最优配置仍慢 2.5× |
| 优化是否有价值? | **有**, 从 7× 优化到 2.5× 慢, 积累了宝贵的 SM110 优化经验 |
| 是否应继续优化 WY? | 低优先级: 52% 时间来自算法固有开销, 难以进一步缩小 |
| DeltaNet prefill 最佳优化路线? | 在**串行 kernel** 上做微优化 (预取, 流水线) |
| 最大 TTFT 收益来源? | GEMM 优化 (占 55%) > Attention 优化 > DeltaNet 优化 |
