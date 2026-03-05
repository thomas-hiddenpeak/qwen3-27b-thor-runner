# GatedDeltaNet UMMA Kernel Design — SM110a (Jetson AGX Thor)

## 1. 目标

将 FlashInfer 的 GatedDeltaNet (GDN) 分块并行 prefill 算法移植到 SM110a，
利用 UMMA (tcgen05.mma) + TMEM (256KB/SM) 实现高效的 SSM 状态持久化和分块矩阵运算。

**当前瓶颈**：串行 prefill kernel 逐 token 递推 → O(T × kd²) 串行工作量，
大 prefill (4K+ tokens) 吞吐极低。

**目标**：分块并行 → O(T/B × B²) + UMMA GEMM，将串行递推转化为矩阵乘法。

## 2. 模型参数

| 参数 | 值 | 说明 |
|------|---|------|
| nkh | 16 | Key heads |
| nv | 48 | Value heads |
| nv_per_kh | 3 | Value heads per key head |
| kd | 128 | Key head dimension |
| vd | 128 | Value head dimension |
| State S | [kd=128, vd=128] FP32 | 每个 value head 独立 |
| Grid | 48 blocks | 1 block per value head |

## 3. SM110a 硬件能力

| 特性 | 规格 | GDN 相关 |
|------|------|---------|
| UMMA (tcgen05.mma) | SS/TS 模式 | 核心 GEMM 引擎 |
| TMEM | 256KB/SM = 128行 × 512列 × 32bit | 持久化 State |
| SMEM | 228KB/SM | TMA 缓冲 + 中间数据 |
| SM80 HMMA | ✅ 可用 (__CUDA_ARCH__ >= 800) | 矩阵逆 |
| TMA | ✅ cp.async.bulk | 异步加载 Q/K/V |
| SM 数量 | 20 | 48 heads / 20 SM ≈ 2.4 波 |

## 4. 核心算法: WY 分块 Delta Rule

### 4.1 数学基础

逐 token 递推:
```
S_i = α_i · S_{i-1} + β_i · k_i ⊗ (v_i - α_i · k_i^T S_{i-1})
o_i = scale · q_i^T S_i
```

WY 分块 (chunk_size = B) 将 B 个 token 的递推转化为矩阵运算:

**IKK 矩阵** [B, B]: 下三角依赖矩阵
```
IKK[s,t] = { 1                              if s == t
           { β_s · exp(Γ_{s,t}) · k_s^T k_t  if s > t
           { 0                              if s < t
```

**T 矩阵** [B, B]: `T = IKK^{-1} · diag(β)` — WY 变换核心

### 4.2 每个 Chunk 的运算流程

给定 `Q[B,kd], K[B,kd], V[B,vd], α[B], β[B]` 和持久化 `S^T[vd, kd]`:

#### Phase A: 辅助计算 (KK, QK, 矩阵逆)
```
A1: KK = K × K^T         [B,kd] × [kd,B] → [B,B]       UMMA SS
A2: QK = Q × K^T         [B,kd] × [kd,B] → [B,B]       UMMA SS
A3: mask + scale KK,QK   element-wise (registers)
A4: T = inv(IKK)         [B,B] → [B,B]                  SM80 HMMA
A5: T *= diag(β)         column scale                    registers
A6: Write T, QK → SMEM
```

#### Phase B: 主计算 (State 操作)
```
B1: O_inter = S^T × Q_norm   [vd,kd] × [kd,B] → [vd,B]    UMMA TS (A=TMEM)
B2: SK = S^T × K_norm        [vd,kd] × [kd,B] → [vd,B]    UMMA TS
B3: V_corr = V^T - α·SK      [vd,B]                        element-wise
B4: NewV = V_corr × T^T      [vd,B] × [B,B] → [vd,B]      UMMA TS
B5: O += NewV × QK^T         [vd,B] × [B,B] → [vd,B]      UMMA TS (accumulate)
B6: Store O → global
B7: S^T *= exp(γ_{B-1})      state decay                   element-wise
B8: NewV decay                 position-dependent            element-wise
B9: S^T += NewV_d × K^T      [vd,B] × [B,kd] → [vd,kd]   UMMA TS (accumulate)
```

## 5. UMMA 操作映射

### 5.1 BF16 UMMA Tile 约束

| 参数 | 约束 |
|------|------|
| M | 64 或 128 (1SM atom) |
| N | 8~256 (步长 8) |
| K (per iteration) | 16 (BF16: 256bit / 16bit) |
| 累加器 | FP32 in TMEM |
| SS 模式 | A=SMEM, B=SMEM → C=TMEM |
| TS 模式 | A=TMEM(BF16), B=SMEM → C=TMEM(FP32) |

### 5.2 GDN 操作 → UMMA Tile 映射

| 操作 | 维度 | UMMA模式 | M | N | K_total | K迭代数 |
|------|------|---------|---|---|---------|--------|
| KK = K×K^T | [64,128]×[128,64] | SS | 64 | 64 | 128 | 8 |
| QK = Q×K^T | [64,128]×[128,64] | SS | 64 | 64 | 128 | 8 |
| O = S^T×Q | [128,128]×[128,64] | TS | 128 | 64 | 128 | 8 |
| SK = S^T×K | [128,128]×[128,64] | TS | 128 | 64 | 128 | 8 |
| NewV = V_c×T^T | [128,64]×[64,64] | TS | 128 | 64 | 64 | 4 |
| O += NewV×QK^T | [128,64]×[64,64] | TS | 128 | 64 | 64 | 4 |
| S += NewV_d×K^T | [128,64]×[64,128] | TS | 128 | 128 | 64 | 4 |

### 5.3 TS 模式的 BF16 转换要求

**关键约束**: UMMA TS 的 A 操作数必须是 BF16 (在 TMEM 中)。
而 State S^T 以 FP32 持久化在 TMEM。

**解决方案**: 使用 TMEM_LOAD + TMEM_STORE 进行 FP32 → BF16 转换:
```
1. S^T (FP32, 128 TMEM cols) → TMEM_LOAD → RF (FP32)
2. RF: FP32 → BF16 转换
3. TMEM_STORE → S^T_bf16 (BF16 packed, 64 TMEM cols)
4. UMMA TS: A = S^T_bf16 (TMEM), B = Q (SMEM) → O (TMEM)
```

或者使用 **TF32 UMMA** (无需转换, 但 K=8 per iteration = 2×慢):
```
SM100_MMA_TF32_TS: A from TMEM (32-bit), K=8 per iteration
  → S^T ×  Q: 128/8 = 16 iterations (vs BF16 的 8)
```

**决策**: v1 先用 BF16 + 转换 (与 FMHA 一致的模式), 性能更好。

## 6. TMEM 内存布局

```
TMEM Columns:  [0           128         192         256         320    384    512]
                |  S^T FP32  | O (FP32)  |  temp1    |  temp2    | PhA  | free |
                | 128 cols   | 64 cols   |  64 cols  |  64 cols  | 64c  |      |
                             
Phase A:        [  S^T (128) ..................... | KK/QK (64) |      |      ]
                Peak: 192 cols (37.5%)

Phase B prep:   [  S^T (128) | S^T_bf16 (64) ..... |           |      |      ]  
                FP32 → BF16 转换, 使用 temp1 区域存 BF16 版本

Phase B main:   [  S^T (128) | O (64) | SK/Vc (64) | NV/NVd(64)| free |      ]
                Peak: 320 cols (62.5%)  ✅ 充裕
```

### 各区域用途

| 区域 | TMEM地址 | 大小 | 生命周期 |
|------|---------|------|---------|
| S^T FP32 | 0-127 | 128 cols | 永久 (跨 chunk) |
| O_inter/O_total | 128-191 | 64 cols | Phase B: B1→B6 |
| temp1 (SK, V_corr, S^T_bf16) | 192-255 | 64 cols | 复用 |
| temp2 (NewV, NewV_d) | 256-319 | 64 cols | Phase B: B4→B9 |
| Phase A temp (KK/QK) | 320-383 | 64 cols | Phase A only |
| Free | 384-511 | 128 cols | 可扩展 |

## 7. SMEM 布局

```
SMEM:
  Q_buf   [B, kd]    = [64, 128] BF16 = 16 KB  (已 L2-norm 归一化)
  K_buf   [B, kd]    = [64, 128] BF16 = 16 KB
  V_buf   [B, vd]    = [64, 128] BF16 = 16 KB
  T_mat   [B, B]     = [64, 64]  BF16 = 8 KB   (Phase A 输出)
  QK_mat  [B, B]     = [64, 64]  BF16 = 8 KB   (Phase A 输出)
  alpha   [B]        = [64]      FP32 = 256 B   (cumulative decay)
  beta    [B]        = [64]      FP32 = 256 B
  k_norms [B]        = [64]      FP32 = 256 B   (L2 norm per token)
  q_norms [B]        = [64]      FP32 = 256 B
  tmem_ptr            = 4 B                     (TMEM base addr)
  Total              ≈ 65 KB (单缓冲) ✅ 远低于 228KB 限制
```

## 8. 线程布局 (v1: 简化非流水线版)

```
Block: 256 threads = 8 warps = 2 WarpGroups
  WG0 (Warps 0-3, 128 threads): Element-wise 计算 + TMEM_LOAD/STORE
  WG1 (Warps 4-7, 128 threads): MMA 发射 + 矩阵逆 + TMA 加载

v1 简化: 所有 phases 串行执行, 无 warp specialization
  → 所有线程参与数据加载(cooperative)
  → elect_one 发射 UMMA
  → 全线程参与 element-wise / TMEM_LOAD/STORE
  → 4 warps 参与 SM80 HMMA 矩阵逆 (FlashInfer 风格)
```

## 9. Q/K L2 归一化处理

Qwen3.5 的 DeltaNet 使用 QK-Norm (per-token L2 normalization):
```
k_hat = k / ||k||₂        (ε = 1e-6)
q_hat = q / ||q||₂ × scale   (scale = 1/√kd)
```

**实现**: 加载 Q/K chunk 到 SMEM 后, 用 block-wide reduction 计算 L2 norm,
然后原地归一化。归一化后的 Q/K 直接用于 UMMA 操作。

## 10. 矩阵逆 (64×64 IKK → T)

采用 FlashInfer `CollectiveInverse` 的递归分块法:
```
L1: 8×8 对角块 × 8 — warp shuffle Gauss 消元
L2: 16×16 — SM80 HMMA 16×8×8
L3: 32×32 — SM80 HMMA 16×8×16, 2 warps
L4: 64×64 — SM80 HMMA 16×8×16, 4 warps, cross-warp reduce
```

SM80 HMMA 在 SM110 上可用 (`__CUDA_ARCH__ 1100 >= 800` → `CUTLASS_ARCH_MMA_SM80_ENABLED` ✅)。

## 11. 实现计划

### Phase 1: UMMA + TMEM 验证原型
- 最小 kernel: TMEM 分配 → SS UMMA (K^T×V) → TMEM_LOAD 验证结果
- 目标: 确认 CUTLASS SM100 atoms 在 SM110a 上工作
- 文件: `src/engine/gdn_umma_sm110.cu`

### Phase 2: 完整分块 GDN (无流水线)
- 实现所有 A/B phase 操作, 串行执行
- 正确性对比: 与 `gated_delta_net_prefill_kernel` 在相同输入上逐 bit 比较
- chunk_size = 64

### Phase 3: 矩阵逆移植
- 从 FlashInfer `flat_collective_inverse.hpp` 移植到我们的 kernel
- SM80 HMMA 实现

### Phase 4: 流水线 + Warp 特化
- TMA 异步加载 + pipeline 双缓冲
- Warp specialization: MMA warp + element-wise warpgroup
- 多 chunk 流水线重叠

### Phase 5: 集成 + 性能调优
- 替换现有 prefill kernel (条件编译, SM110 路径)
- 性能基准测试: 1K/4K/8K/16K tokens
- 与串行 kernel 对比 tok/s

## 12. 性能估算

### 理论分析 (单 chunk, B=64, kd=vd=128)

**UMMA 计算量 (Phase B 主要操作)**:
- B1: S^T×Q: 128×64×128×2 = 2.1M FLOP
- B2: S^T×K: 同上 = 2.1M
- B4: NewV = V_c×T: 128×64×64×2 = 1.0M
- B5: O += NewV×QK: 同上 = 1.0M
- B9: S += NewV×K: 128×128×64×2 = 2.1M
- **Phase B 总计 ≈ 8.4M FLOP**

**Phase A (KK, QK, inverse)**:
- KK+QK: 2 × 64×64×128×2 = 2.1M FLOP
- Inverse: ~1M FLOP (SM80 HMMA)
- **Phase A 总计 ≈ 3.1M FLOP**

**每 chunk 总计 ≈ 11.5M FLOP**

**SM110a Tensor Core 吞吐** (估算):
- UMMA BF16: ~4 TFLOPS/SM (保守估计, 实际可能更高)
- 11.5M FLOP / 4 TFLOPS = 2.9 μs/chunk (计算时间)
- 加上 TMEM_LOAD/STORE + element-wise: ~10 μs/chunk (估计)

**T=4096 tokens, B=64 → 64 chunks**:
- 理论: 64 × 10 μs = 640 μs/head
- 48 heads / 20 SM × 3 waves = 1.9 ms/layer

**对比串行 kernel**: 4K tokens 当前 ~31 ms/layer (从 benchmark)
→ **潜在加速: 16×** (如果 UMMA 计算不成为瓶颈)

## 13. 关键风险

1. **TMEM BF16 转换开销**: FP32→BF16 的 TMEM_LOAD/STORE 可能占显著时间
2. **矩阵逆精度**: FP16 下 64×64 逆矩阵的数值稳定性需验证
3. **Q/K 归一化**: Block-wide reduction + SMEM rewrite 的延迟
4. **SM110a UMMA 吞吐**: Thor 的 UMMA 实测吞吐可能低于预期 (20 SM vs 数据中心)
5. **首次 TMEM 分配延迟**: `tcgen05.alloc` 可能有 ~1μs 初始化开销
