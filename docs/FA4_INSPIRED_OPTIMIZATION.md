# FA4 启发的优化方案 — 实施报告

基于 FlashAttention-4 (SM100/SM110 Blackwell) 代码分析，以下为已落地到 SM110a AGX Thor 平台的优化及其实际效果。

## 基线 vs 优化后 (B=1, 30 decode steps, MTP off)

| 版本 | ITL (ms) | Forward (ms) | BW (GB/s) | 变化 |
|------|----------|-------------|-----------|------|
| 基线 | 229.81 | 218.65 | 223.0 | — |
| + expf→exp2f | 227.96 | 217.03 | 224.8 | -0.8% |
| + sm_scale×LOG2E pre-multiply | 227.62 | 216.56 | 225.1 | -0.95% |
| + light_ops/gdn_umma exp2f | 227.26 | 216.34 | 225.5 | -1.1% |

**总计**: ITL -2.55ms (-1.1%), BW +2.5 GB/s

---

## 优化 1: expf → exp2f (已实施 ✅)

### 原理

`exp2f(x)` 比 `expf(x)` 快，GPU 硬件直接支持 base-2 指数。
FA4 全面使用 `exp2` + `scale_log2` 模式。

### 实施范围

**paged_attention.cu**: 13 处 `expf` → `exp2f(...*LOG2E)`, 覆盖全部 7 个 attention kernel。

**light_ops.cu**: ~22 处 `expf` 转换:
- SiLU/sigmoid: conv1d (×3), SwiGLU merged (×2), deinterleave (×3), norm+gate (×1)
- DeltaNet: softplus + alpha + beta (×2 instances: decode + prefill)
- MoE router softmax (×2), vision gate sigmoid (×2)

**gdn_umma_sm110.cu**: 3 处转换, 含链式优化:
- `expf(-dt_v * expf(a_l))` → `log2f(...)` 链简化为直接 log2-scale 计算, 省 2 个 transcendental

### 实测效果

- Decode: -0.8% ITL (paged_attention only), light_ops 无额外贡献 (bandwidth-bound)
- Prefill: 17-token benchmark 中占比太小无法单独测量

---

## 优化 2: sm_scale × LOG2E 预乘 (已实施 ✅)

### 原理

FA4 在 Q 加载时就将 `softmax_scale * log2(e)` 合并, 使 QK 点积结果直接处于 log2 尺度,
所有 `exp2f(score - max)` 调用去掉 `* LOG2E` 运行时乘法。

### 实施

4 个 Q 加载点改为 `sm_scale * LOG2E`:
- `paged_attention_kernel`: `s_q[tid] = ... * (sm_scale * LOG2E)`
- `paged_attention_split_k_kernel`: 同上
- `gather_q_group_kernel`: GEMM prefill Q extraction
- `fused_prefill_attention_kernel`: fused prefill Q loading

13 个 `exp2f((x - y) * LOG2E)` 简化为 `exp2f(x - y)`:
- Decode inner loop: 每 KV token 省 1 FMA (split-k + merge 共 6 处)
- Prefill softmax: 5 处
- Fused prefill: 2 处

### 实测效果

- 额外 -0.15% on decode (与 exp2f 叠加后 -0.95%)
- Inner loop 每 KV token 少 1 multiply, 但 decode 是 bandwidth-bound, 计算节省被带宽掩盖

---

## 未实施优化评估

### Fused Prefill Attention Kernel (❌ 暂不启用)

`fused_prefill_attention_kernel` 已实现 (Grid=(num_heads, T), Block=256, online softmax),
但当前 dispatch 逻辑用 GEMM-based 方式 (CUTLASS GEMM + causal softmax, 28 launches/layer)。

**不启用原因**:
- max_chunk_size=256, 对应 T≤256; CUTLASS GEMM 对 [1536, 256] × [256, 256] 大小矩阵已高度优化
- Fused kernel 用 per-thread scalar reduction + blockReduceSum, 远不如 CUTLASS Tensor Core throughput
- 28 kernel launches 开销 ~0.15ms/layer, 但 GEMM throughput 优势更大
- 仅首 chunk prefill (T≥256) 使用此路径, 不影响 decode

### GEMM SMEM Descriptor 预计算 (❌ 不适用)

CUTLASS SM110 GEMM 内部管理 SMEM descriptor, 无法外部预计算。

### __exp2f 快速内建 (❌ 不可用)

`__exp2f()` 是 host-only 函数, 不可用于 device code。标准 `exp2f()` 已被 NVCC 优化为最优指令。

---

## 总结

FA4 中对 SM110 有直接价值的技术主要是 **exp2f + log2-scale 一致性**模式。
其最先进的技术 (UMMA + TMEM + 2CTA persistent + 16-warp pipeline) 面向数据中心 Blackwell,
在 AGX Thor (20 SM, 无 cluster) 上价值有限。

**净效果**: Decode ITL -1.1% (229.81→227.26ms), 带宽利用 223→225.5 GB/s。
改进幅度小但方向正确, 符合 decode bandwidth-bound 的理论预期:
softmax/exp 计算在 decode 中占比 <1%, 优化空间天然有限。
