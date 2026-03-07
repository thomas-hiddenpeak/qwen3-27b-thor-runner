# Prefill 综合优化方案 — Phase 11

## 背景

当前 prefill 性能与 vLLM/SGLang 相比差距巨大 (10x+)。本文档基于全面的管线分析、FlashInfer/fla 社区调研、Qwen3-Next 官方博客、Gated DeltaNet 论文研究，制定完整的优化路线图。

### 当前状态 (2026-02-27 更新)

**已完成优化**:
- ✅ GEMM-based Prefill Attention (Phase 10b): T≥256 用 CUTLASS GEMM 替代 O(T²) paged attention, 2.1-3.0× 加速
- ✅ KV Cache DeviceAllocator (Phase 11A): `cudaMalloc` 替代 `cudaMallocManaged`, 消除 page fault 延迟
- ✅ CUTLASS Per-GEMM Sync 移除 (Phase 11B): 消除 312×2=624 次 host-device 同步, T=17 TTFT -8.5%
- ✅ Benchmark 2x Warmup Prefill (Phase 11D): 解决 T=1024 双模态测量问题
- ✅ 连续批处理 Benchmark (Phase 11F): 1/2/4/8/16/32/48/64 并发扫描完成 (B=64 达 200.8 tok/s)

**尝试后放弃**:
- ❌ DeltaNet kd 并行化 (Phase 11C): 4-way bank conflict (stride-32 → 同 bank)
- ❌ DeltaNet norm 预计算 (Phase 11C): 16MB FP32 输出污染 32MB L2 cache

**当前 Prefill 性能**:

| T | TTFT (ms) | Prefill tok/s |
|---:|---:|---:|
| 17 | 268 | 63 |
| 256 | 445 | 575 |
| 1024 | 1143 | 896 |

---

## 一、调研总结

### 1.1 FlashInfer 调研结论

| 项目 | 结论 |
|------|------|
| **SM 支持** | SM120/121 = Jetson Thor。我们的 SM110a 在 SM100 和 SM120 之间，需验证兼容性 |
| **GDN Prefill** | ✅ `include/flashinfer/flat/` 目录, Hopper (`sm_90`) + Ampere 后端均有 |
| **GDN Decode** | ✅ `gdn_decode.py` + CuTe-DSL kernel，已支持 bf16 state |
| **Attention** | ✅ FlashAttention-2/3, paged attention, POD attention (P+D 混合) |
| **GEMM** | ✅ BF16 GEMM for SM100+, CUTLASS backend |
| **Conv1d** | ❌ 无内置 causal conv1d (需 `causal-conv1d` 库或自实现) |
| **C++ API** | 有 header-only `.cuh` 文件可直接集成，但主要为 PyTorch TVM FFI binding |
| **集成路径** | 提取 `include/flashinfer/flat/` 下的 C++ kernel 头文件，AOT 编译 |

**关键发现**: FlashInfer 最近几周 (2025-10) 在持续优化 GDN kernel:
- `Perf: Optimize GDN decode pretranspose kernel for all batch sizes (#2588)` — 2 小时前
- `refactor: reduce hopper's gdn prefill compilation time` — 3 周前
- `Ameyn/gdn bf16 tolerance parallel reduction (#2610)` — 3 天前

### 1.2 flash-linear-attention (fla) 调研结论

| 项目 | 结论 |
|------|------|
| **GDN 实现** | ✅ `fla/ops/gated_delta_rule/` 包含 chunk.py, fused_recurrent.py, wy_fast.py |
| **算法** | WY representation 将 Householder 乘积转为矩阵乘法, chunk-parallel |
| **语言** | 纯 Python/Triton，无法直接在 C++ 项目中使用 |
| **价值** | 提供完整的 chunkwise 并行算法参考，可将算法移植为 CUDA kernel |
| **Qwen 关系** | Qwen 官方推荐的推理库: "通过安装 flash-linear-attention 和 causal-conv1d 获得更佳效率" |

### 1.3 Qwen3-Next 官方博客结论

| 项目 | 结论 |
|------|------|
| **架构** | 3:1 混合 (75% GDN + 25% Gated Attention) = 我们的 48:16 |
| **Prefill 提速** | 官方报告 vs dense model: 4k context 7×, 32k+ context 10× |
| **推理框架** | 推荐 SGLang 和 vLLM，均已支持 Qwen3-Next |
| **关键依赖** | flash-linear-attention + causal-conv1d |
| **MTP** | Multi-Token Prediction 用于投机解码 (我们暂不做) |
| **Zero-Centered RMSNorm** | 已实现 `(1+weight)` |

### 1.4 DeltaNet Chunkwise Parallel 算法

来自 fla 的 DeltaNet 论文 Appendix B, chunkwise 算法核心:

```
输入: Q[T,dk], K[T,dk], V[T,dv], β[T] (gate), S0[dk,dv] (初始状态)
分块: C = chunk_size (如 64)

对每个 chunk c:
  1. T = (I + tril(diag(β)⊙KK^T, -1))^{-1} ⊙ diag(β)  [C×C, 下三角求逆]
  2. W = T @ K          [C, dk]    — 使用 GEMM
  3. U = T @ V          [C, dv]    — 使用 GEMM
  4. S_new = S_old + K^T(U - W·S_old)  [dk, dv] — 使用 GEMM
  5. O = Q·S_old + (Q·K^T ⊙ M) · (U - W·S_old)  [C, dv] — 使用 GEMM
  6. S_old = S_new
```

**关键**: 每个 chunk 内部完全可以并行 (所有 C 个 token 同时处理), 只需 chunk 间串行传递 S。
- 现状: T 步全串行 → O(T) 
- 优化后: T/C 步串行 × O(C²) GEMM → O(T/C · C²) = O(T·C)
- C=64 时 GEMM 内部并行, 有效并行度提升 64×

---

## 二、瓶颈优先级排序

### Tier 1 🔴 — 决定性瓶颈 (预期 5-20× prefill 提速)

| # | 优化项 | 现状 | 目标 | 预估加速 | **实施状态** |
|---|--------|------|------|----------|---|
| 1 | **DeltaNet 串行→chunkwise 并行** | 48层 × T步全串行, 128线程/block | chunk_size=64, GEMM 内部并行 | 10-50× per layer | ⚠️ kernel 级微优化失败 (bank conflict + L2 pollution), 需 WY 算法级重构 |
| 2 | **Conv1d 串行→并行** | 10240通道 × T步串行 loop | 矩阵化: `x_slide[T,4] × w[4,1]` 或 warp 级并行 | 5-10× | 未开始 |
| 3 | **合并 QKV/ZAB GEMM (T>1)** | 每层 3 次 GEMM 读 input 3× | 单次合并 GEMM, scatter 到 workspace | 2-3× for proj stage | 未开始 |

### Tier 2 🟠 — 重要优化 (预期 2-5× for specific stages)

| # | 优化项 | 现状 | 目标 | 预估加速 | **实施状态** |
|---|--------|------|------|----------|---|
| 4 | **FlashAttention 替代 GEMM attention** | ~~材料化 [T,T] score~~ | Tiled FA, O(1) score memory | 2-4× for 16 FullAttn layers | ✅ Phase 10b: CUTLASS GEMM attention 已替代 O(T²) paged attention |
| 5 | **合并 Gate+Up GEMM (T>1)** | 每层 2 次 GEMM 读 norm_out 2× | 单次 GEMM → split | 1.5-2× for MLP stage | 未开始 |
| 6 | **MLP Down GEMM+Add 融合 (T>1)** | ~~`invoke_dense_gemm` + 独立 `invoke_add`~~ | 使用已有的 `invoke_dense_gemm_add` | 1.2-1.5× for MLP stage | ✅ 已实现: `run_mlp` T>1 路径使用 `invoke_dense_gemm_add` |

### Tier 2.5 🟠 — 新发现的系统瓶颈

| # | 优化项 | 现状 | 目标 | 预估加速 | **实施状态** |
|---|--------|------|------|----------|---|
| 6a | **CUTLASS Per-GEMM Sync 移除** | 312 GEMM × 2 sync = 624 sync | 0 sync | T=17 -8.5%, 大 T 流水线更紧凑 | ✅ Phase 11B |
| 6b | **KV Cache DeviceAllocator** | cudaMallocManaged lazy page fault | cudaMalloc 直接映射 | TTFT 5.4× (消除 12s+ 延迟) | ✅ Phase 11A |

### Tier 3 🟡 — 系统级优化 (预期 10-50% overall)

| # | 优化项 | 现状 | 目标 | 预估加速 | **实施状态** |
|---|--------|------|------|----------|---|
| 7 | **P/D 分离 + Continuous Batching** | ~~单请求串行处理~~ | 混合批次, prefill 和 decode 可交替 | 2-3× throughput | ✅ Phase 8-9: Benchmark 已支持 B=1~64 (200.8 tok/s @B=64) |
| 8 | **Chunked Prefill** | 全序列一次 prefill | 分 chunk 处理, 插入 decode 步 | 降低 TTFT p99, 提升并发 | 未开始 |
| 9 | **SSM/Conv 状态池化** | 每请求 cudaMalloc/cudaFree | 预分配池 | 减少延迟 | 未开始 |
| 10 | **CUDA Graph for Prefill** | 仅 decode 有 graph | 固定大小 padded graph | 减少 launch overhead | ❌ 不适用: prefill 的 T 变化导致 kernel 参数不固定 |
| 11 | **Prefix Caching** | 无 | 系统提示 KV 缓存复用 | 减少重复 prefill | 未开始 |

---

## 三、实施方案 (按优先级排序)

### Phase 11a: DeltaNet Chunkwise Parallel (最高优先级)

**目标**: 将 48 层 LinearAttn 的 DeltaNet 从 O(T) 串行变为 O(T/C) chunk 并行

**⚠️ Phase 11C 实施记录**:

尝试了两种 kernel 级微优化方案，均失败:

1. **512-thread KD 并行化**: 将 kd=128 的 S-update 拆分给 4 个 kd group 并行处理  
   → **4-way bank conflict**: `S_smem[i*129+j]` → bank=(i+j)%32, stride-32 的 kd groups 恰好映射到相同 banks  
   → 与原始 128-thread kernel 持平 (8ms/head)

2. **128-thread 预计算 norm**: prepare kernel 预计算全部 T 个 token 的 k_norm/q_norm  
   → **L2 pollution**: 16MB FP32 中间结果刷掉 32MB L2 中的 GEMM 权重缓存  
   → 总体 forward 时间无改善

**根本限制**: S-update 内循环 (128 次 kd iterations × smem read/FMA/write) 占执行时间 99%，norm 计算 <1%。  
Kernel 级微优化已无空间，**只有算法级 WY chunkwise decomposition 才能突破**。

**算法**: 基于 fla 的 WY representation (参见上方 §1.4)

**实现方案** (待实施):

```
新增 kernel: gated_delta_net_chunkwise_kernel

Grid: [num_chunks, nkh * nv_per_kh]
Block: [128]  (处理一个 chunk 的一个 head)

Shared Memory:
  - S_local[kd, vd]   = 64KB (float32, 128×128)  — SSM state
  - chunk_K[C, kd]     = 16KB (bf16, 64×128)      — chunk keys
  - chunk_V[C, vd]     = 16KB (bf16, 64×128)      — chunk values
  - T_mat[C, C]        = 16KB (float32, 64×64)     — 下三角矩阵
  Total ≈ 112KB/block — 需要 extended shared memory (228 KB/SM available)

每个 chunk:
  1. 加载 K[c*C:(c+1)*C, dk], V[c*C:(c+1)*C, dv] 到 smem
  2. 计算 T = (I + tril(diag(β)KK^T, -1))^{-1} diag(β)
     - KK^T 用 warp-level matmul in smem
     - 下三角求逆用 forward substitution (C=64, 可展开)
  3. W = T @ K, U = T @ V  (smem GEMM)
  4. S_contrib = K^T(U - W·S_local)  (state update via GEMM)
  5. O = Q·S_local + (QK^T ⊙ mask)(U - W·S_local)
  6. S_local += S_contrib
  7. 写 O 到 global
最后写回 S_local 到 ssm_state
```

**选型考虑**:
- **方案 A: 自实现 CUDA kernel** — 完全控制, 可动态 chunk_size, 最适合 SM110a
- **方案 B: 移植 FlashInfer `flat/` 头文件** — 需验证 SM110 兼容性, 有 CuTe 依赖
- **方案 C: Triton kernel (fla 风格)** — 需要 Triton 编译器, 不符合纯 C++ 约束

**推荐: 方案 A** — 自实现 chunkwise kernel, 参考 fla 的 WY algorithm

**理论分析** (T=1024, kd=128, vd=128, C=64):
- 现状: 1024 步串行, 每步 ~128 次 S 行更新 + 128 次 y 累加 = ~128K FMA/step
- 优化后: 16 chunks, 每 chunk:
  - T_mat: C²=4096 乘加
  - W,U: C×dk=8K 乘加
  - O: C×dv + C²×dv = 8K + 512K 乘加
  - 总: ~530K FMA/chunk, 但大部分可并行 (warp-level gemm)
- 关键: 530K FMA 在 128 线程 × 16 chunks 上可并行, vs 128K FMA × 1024 步串行

### Phase 11b: Conv1d 并行化

**目标**: 消除 `for(t=0; t<T; t++)` 串行循环

**方案**:
```cuda
// 现状: 每线程处理一个 channel, 串行遍历 T 个 token
// 优化: 矩阵化视角
//   x_slide[t, 0..conv_k-1] = [conv_state[t-3], conv_state[t-2], conv_state[t-1], x[t]]
//   y[t] = dot(x_slide[t], conv_w)
//
// 但 causal conv1d 有状态依赖 (每步更新 conv_state)
// 实际上 conv_state 只是过去 conv_k-1 个 input 的滑动窗口!
// 对于 prefill: 历史在 t=0,1,2 来自初始 conv_state, t>=3 时全来自 input
//
// 优化: 一次性构建滑动窗口矩阵, 然后 GEMV 并行
// Grid: [num_tokens, ceil(channels/256)]
// Block: [256]
// 每线程处理一个 (t, ch) 对
```

**Dao-AILab causal-conv1d 参考**: Qwen 官方推荐安装 `causal-conv1d` 库,
该库提供了高效的 CUDA kernel. 可以考虑:
1. 直接集成 `causal-conv1d` 的 CUDA kernel (header-only C++)
2. 自实现并行 conv1d (更可控)

### Phase 11c: 合并 QKV/ZAB GEMM

**目标**: T>1 时将 3 次独立 GEMM 合并为 1 次

**现状问题**: 注释说 "T>1 时 RowMajor GEMM 行主序交错 vs workspace 独立块布局不兼容"

**解决方案**: 
```
方案 1 (推荐): 合并 GEMM 输出到连续 buffer, 然后用 scatter kernel 分发
  - GEMM: [T, hs] × [hs, N_merged] → [T, N_merged]
    其中 N_merged = qp_dim + 2*kv_dim (FullAttn) 或 in_qkv + lin_v + 2*nv (LinearAttn)
  - scatter: 从 [T, N_merged] 分段复制到 workspace 的各子区域
  - 节省: 2 次 input 读取 (每次 T×hs×2B), 总省 2×T×5120×2B per layer

方案 2: strided GEMM 
  - CUTLASS 支持 StridedBatch GEMM, 但这里是 split-K 不是 batch
  - 不太适用

方案 3: 合并权重矩阵, 输出直接到 merged buffer
  - 权重: 在 load 时拼接 [q_proj_w; k_proj_w; v_proj_w] 为 [qp+2kv, hs]
  - 输出: [T, qp+2kv] 行主序, 直接用指针偏移 ← 这就是 T=1 已在做的!
  - T>1 问题: CUTLASS GEMM 输出 RowMajor [T, qp+2kv], 第 t 行的数据是
    [q[t,0..qp-1], k[t,0..kv-1], v[t,0..kv-1]], 下游 kernel 期望连续的 Q/K/V 块
  - 解决: 下游 kernel 改用 strided 访问, 或添加一个转置/scatter kernel
  
推荐方案 3: 合并权重 + merged output + scatter kernel
```

### Phase 11d: FlashAttention for Prefill

**目标**: 替代当前 GEMM-based prefill attention (16 FullAttn layers)

**现状**: 4 KV groups × 8 kernels = 32 kernels per layer, 材料化 [T,T] score

**方案**:
```
选项 A: FlashInfer attention kernel
  - include/flashinfer/attention/ 有完整的 prefill + decode kernel
  - 需要验证 SM110a 兼容性
  - 支持 GQA (num_q=24, num_kv=4)

选项 B: 自实现简化版 FlashAttention-2
  - Tile Q[Br,hd], K[Bc,hd], V[Bc,hd]
  - Online softmax (Milakov & Gimelshein 2018)
  - 对 hd=256 需要特殊处理 (标准 FA 按 hd=128 设计)

选项 C: cuDNN SDPA (cudnnSetAttnDescriptor)
  - NVIDIA 官方库, SM110 支持有保障
  - 但 API 较重, 不确定 GQA 支持

推荐: 先试 选项 A (FlashInfer), 如不兼容 再 选项 B (自实现)
```

### Phase 11e: 合并 Gate+Up GEMM & Down+Add 融合

**Gate+Up 合并** (未开始):
```
现状: 
  invoke_dense_gemm(norm_out, gate_proj_w, gate_out, T, is, hs)  // 读 norm_out
  invoke_dense_gemm(norm_out, up_proj_w,   up_out,  T, is, hs)  // 再读 norm_out

优化:
  gate_up_merged_w = vstack(gate_proj_w, up_proj_w)  // [2*is, hs]
  invoke_dense_gemm(norm_out, gate_up_merged_w, gate_up_out, T, 2*is, hs)
  // 然后 SiLU gate 内部做 split (offset is)
```

**Down+Add 融合 (T>1) — ✅ 已完成**:
```
// run_mlp() 的 T>1 分支已使用 invoke_dense_gemm_add:
invoke_dense_gemm_add(swiglu, down_proj_w, hidden, hidden, num_tokens, hs, is, stream);
// T=1 分支也使用 invoke_dense_gemv_add
```

### Phase 11f: P/D 分离 + Continuous Batching

**目标**: 引擎支持 prefill 和 decode 交替执行

**方案**:
```
核心修改 engine.cpp:

1. 请求队列分类:
   - pending_prefill_: 等待 prefill 的新请求
   - active_decode_:   正在 decode 的请求

2. 每次 step:
   a. 如有 pending_prefill 且当前 decode batch < max_batch:
      - 将一个 pending 请求做 prefill (或 chunked prefill)
      - 完成后移入 active_decode_
   b. 对所有 active_decode_ 做一步 batched decode

3. Chunked Prefill:
   - 长 prompt 分 chunk_size=512 的段
   - 每个 chunk 做完后检查 active_decode_ 是否要插一步 decode
   - 避免长 prefill 阻塞 decode
```

### Phase 11g: Prefix Caching

**目标**: 系统提示的 KV Cache 可复用

**方案**: 实现基本的前缀命中检测
```
1. 维护 prefix_cache_: hash(token_ids[0..L]) → (block_table, conv_states, ssm_states)
2. 新请求到来时:
   a. 检查 prefix_cache_ 是否命中
   b. 命中: 复用 KV blocks, 仅处理新增 tokens
   c. 未命中: 全量 prefill
3. 特别注意: DeltaNet SSM state 也需要缓存! 不仅是 KV cache
   - 48 层 × 48 heads × 128×128 = 144 MB float32 per prefix
   - 这是 hybrid 架构的特有挑战
```

---

## 四、实施顺序与依赖关系

```
Phase 11a (DeltaNet chunkwise)  ┐
Phase 11b (Conv1d parallel)     ├── 独立, 可同时进行
Phase 11e (Gate+Up merge, Down+Add fusion) ┘

Phase 11c (QKV merge) ── 依赖 11a (接口可能变化)

Phase 11d (FlashAttention) ── 独立, 但需先评估 FlashInfer SM110 兼容性

Phase 11f (P/D separation) ── 依赖上述优化稳定后
Phase 11g (Prefix Caching) ── 依赖 11f
```

**推荐实施顺序** (2026-02-27 更新):
1. ~~**11e** (Down+Add 融合)~~ — ✅ 已完成
2. ~~**11d** (Attention 优化)~~ — ✅ Phase 10b GEMM attention 已完成
3. ~~**CUTLASS sync 移除**~~ — ✅ Phase 11B 已完成
4. ~~**连续批处理 benchmark**~~ — ✅ Phase 11F, B=64 达 200.8 tok/s
5. **11b** (Conv1d 并行) — 相对简单, 中等收益
6. **11a** (DeltaNet WY chunkwise) — 最大收益, 但实现复杂度极高
7. **11c** (QKV merge) — 中等复杂度, 需修改权重加载
8. **11e.gate_up** (Gate+Up merge) — 需修改权重加载
9. **11f** (P/D separation) — Engine 调度器重构
10. **11g** (Prefix Caching) — 需 SSM state caching 设计

---

## 五、性能分析 (2026-02-27 实测更新)

### Prefill T=1024, B=1 开销分解 (Phase 11 实测 — TTFT=1143ms)

基于 nsys profiling (GPU kernel wall span = 1129ms):

| 阶段 | 耗时估算 | 占比 | 状态 |
|------|---------|------|------|
| DeltaNet recurrence (48层, 串行) | ~384ms | ~34% | 🔴 待 WY chunkwise |
| GEMM projections (64层 × 5-8 GEMM) | ~625ms | ~55% | ✅ CUTLASS sync 已移除 |
| Attention (16层, GEMM-based) | ~80ms | ~7% | ✅ Phase 10b GEMM attention |
| Conv1d (48层, 串行) | ~20ms | ~2% | 🟡 可优化但收益小 |
| MLP misc (SiLU, RMSNorm, add) | ~15ms | ~1% | ✅ 已融合 |
| Kernel launch overhead | ~5ms | ~0.4% | ✅ CUDA Graph (decode only) |
| **总计** | **~1129ms** | **100%** | |

### 剩余优化空间

| 优化 | 预期省时 | 难度 |
|------|----------|------|
| DeltaNet WY chunkwise | 384ms → ~50ms = **-334ms** | 🔴 极高 (算法级重构) |
| QKV/ZAB GEMM 合并 | 625ms × 0.15 = **-94ms** | 🟡 中等 (权重预拼接) |
| Gate+Up GEMM 合并 (T>1) | ~20ms per layer × 64 = **-40ms** | 🟡 中等 |
| Conv1d 并行 | 20ms → ~2ms = **-18ms** | 🟢 低 |
| **总计可省** | **~486ms** | |
| **优化后预期 TTFT** | **~657ms** (1.74× speedup from current) | |

注: DeltaNet WY 是唯一能带来显著加速的优化 (占剩余可省时间的 69%)。
如果仅做 GEMM 合并 + Conv1d 并行 (不做 DeltaNet WY), 预期 TTFT ~990ms (~1.15× speedup), 收益有限。

---

## 六、FlashInfer 集成评估

### SM110a 兼容性

**问题**: FlashInfer 官方列表中 SM110 未明确出现:
- SM100/103 = B200/B300 (data center Blackwell)
- SM120/121 = RTX 50/DGX Spark/Jetson Thor

SM110a 位于二者之间。可能的情况:
1. SM110a 后向兼容 SM100 功能 → FlashInfer SM100 kernel 可直接使用
2. SM110a 是独立架构 → 需要针对性编译

**验证方法**: 
```bash
# 1. 检查 CUDA Toolkit 对 SM110a 的功能级别
nvcc --list-gpu-arch | grep 110
# 2. 尝试编译 FlashInfer 的 attention header
nvcc -arch=sm_110a -c test_flashinfer.cu
```

**建议**: 先尝试编译 FlashInfer 的 `include/flashinfer/attention/` 头文件, 
看是否能通过 SM110a 编译。如果可以, 直接集成; 否则自实现 FA2。

### 集成方式 (AOT)

FlashInfer 的 C++ kernel 在 `include/flashinfer/` 下作为 header-only 使用。
AOT 编译步骤:
1. `git submodule add https://github.com/flashinfer-ai/flashinfer.git third_party/flashinfer`
2. CMakeLists.txt 添加 `include_directories(third_party/flashinfer/include)`
3. 在 attention wrapper 中 `#include <flashinfer/attention/...>` 并调用

---

## 七、关键注意事项

### 不要引入的回退

1. ❌ RMSNorm `(1+weight)` centered 规则不可破坏
2. ❌ RoPE 半旋转配对不可改为交错
3. ❌ q_proj 的 Q+Gate deinterleave 逻辑
4. ❌ KV cache 每层独立布局
5. ❌ Conv1d 对全部 10240 通道 (不只是 Q)
6. ❌ DeltaNet attn_norm 使用 plain weight (非 centered)

### Qwen3.5 vs Qwen3-Next

我们的模型是 **Qwen3.5-27B** (dense, 非 MoE), 与 Qwen3-Next-80B-A3B 共享:
- 相同的 GDN + Gated Attention 混合架构
- 相同的 3:1 比例 (48 GDN + 16 FullAttn)
- 相同的 kd=128, vd=128, conv_k=4 参数
- 不同: Qwen3.5 是 dense MLP, Qwen3-Next 是 MoE

因此 vLLM/SGLang 对 Qwen3-Next 的优化 (特别是 GDN chunk 和 FA) 直接适用于我们。
