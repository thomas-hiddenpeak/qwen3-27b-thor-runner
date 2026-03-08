# FA4 启发的优化方案

基于 FlashAttention-4 (SM100/SM110 Blackwell) 代码分析，以下为可落地到我们 SM110a AGX Thor 平台的优化。

## 当前基线 (B=1, 30 decode steps)

| 模型 | ITL (median) | Forward | tok/s | BW |
|------|-------------|---------|-------|-----|
| 27B BF16 | 229.81 ms | 218.65 ms | 4.35 | 223.0 GB/s |
| 27B NVFP4 | 98.98 ms | 87.76 ms | 10.10 | N/A |

## 优化 1: expf → exp2f (Softmax 快速路径)

### 原理

`exp2f(x)` 比 `expf(x)` 快，因为 GPU 硬件直接支持 base-2 指数。
等价变换: `expf(x) = exp2f(x * LOG2_E)`, 其中 `LOG2_E = log2(e) ≈ 1.4426950f`

FA4 全面使用 `exp2` + `scale_log2` 模式:
```python
# FA4: softmax scale 预乘 log2(e)
softmax_scale_log2 = softmax_scale * log2(e)
# 计算: exp2(x * scale_log2 - max * scale_log2)
# 等价: exp(x * scale - max * scale)
```

### 影响范围

paged_attention.cu 中 13 处 `expf` 调用 (softmax 相关)：
- `paged_attention_kernel` (decode split-K): 2 处 (L126-127)
- `split_k_reduce_kernel`: 4 处 (L257-258, L303-304) 
- `causal_softmax_interleaved_kernel`: 2 处 (L529, L541)
- `fused_prefill_attention_kernel`: 2 处 (L629, L634)
- `tiled_causal_softmax_kernel`: 1 处 (L806)
- `merge_attention_tile_kernel`: 2 处 (L844-845)

### 实现方式

引入常量 `LOG2E = 1.4426950408889634f`，将所有 softmax 中的 `expf(a - b)` 替换为 `exp2f((a - b) * LOG2E)`。
注意：sm_scale 可以预乘 LOG2E 一次，避免每次乘法开销。

### 预期收益

- Decode: 微量 (~0.1-0.3%), 因为 softmax 占比小 (decode 是 GEMV bandwidth-bound)
- Prefill: 可能 1-3%, softmax 在 prefill attention 中占比更高

### 风险

- 精度: exp2f 和 expf 的 ULP 误差不同, 但 softmax 本身是 approximate 操作, FP32 中差异可忽略
- 不影响 light_ops.cu 中的 SiLU/sigmoid/DeltaNet — 这些不是 softmax 路径

---

## 优化 2: __expf / __exp2f 快速数学内建函数

### 原理

CUDA 提供 `__expf()` / `__exp2f()` 快速版本, 比标准 `expf`/`exp2f` 少约 2-4 个指令周期, 
精度从 ~1 ULP 降到 ~2 ULP, 对 softmax 完全可接受。

### 实现

在优化 1 基础上, 将 `exp2f` 进一步替换为 `__exp2f`。

### 预期收益

- 与优化 1 叠加, 额外 ~0.1%

---

## 实施计划

**Phase 1** (本次): 优化 1 + 2 — expf → __exp2f
- 修改 paged_attention.cu 中所有 softmax 相关的 expf
- 预乘 sm_scale * LOG2E 减少运行时乘法
- 编译验证 + 正确性测试 + 性能对比

**待定**: 
- Prefill fused attention kernel 启用 + 按 FA4 模式优化 (需要更深入分析)
- GEMM SMEM descriptor 预计算 (CUTLASS 内部已处理, 手动优化空间有限)
