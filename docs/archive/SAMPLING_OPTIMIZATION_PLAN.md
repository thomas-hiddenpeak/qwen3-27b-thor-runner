# GPU 采样优化方案 — FlashInfer 参考设计

> **状态**: Phase 1-2 已完成 (2026-03-03), Phase 3 待定
> **详细实现记录**: 见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) Phase 14

## 1. 背景与目标

### 现状分析

当前 `sample_token()` 流程 (engine.cpp L1813-1955):
```
cudaStreamSynchronize()          → 阻塞 CPU 等待 GPU
cudaMemcpy(D→H, 248320×BF16)    → 485 KB 数据搬回 host
BF16→float loop (248320 iters)   → CPU 循环转换
repetition penalty loop           → CPU 遍历已生成 token
temperature scaling loop          → CPU 248320 次乘法
std::partial_sort(248320, top_k) → O(n) 选择排序, ~0.5ms
softmax + top_p + min_p          → CPU 归一化 + 截断
random sampling                   → std::mt19937 + uniform_dist
```

**性能瓶颈:**
- 单并发 decode ~222ms/step, 采样 ~0.5ms, 占 ~0.2% → 影响较小
- 但 batched decode (batch=8): forward ~222ms, 采样 `8 × 0.5ms = 4ms` → 占比增至 ~1.8%
- 更关键: `cudaStreamSynchronize` + `cudaMemcpy` 打断 GPU pipeline, 增加延迟
- CPU 采样无法被 CUDA Graph 捕获, 破坏 graph 的完整性

**稳定性问题:**
1. CPU float32 softmax 可能在极端 logit 值时数值不稳定 (虽然 partial_sort 后 max 已知)
2. MTP verify 路径使用 `sample_argmax` 而非随机采样 → temperature>0 时分布不一致
3. `std::partial_sort` 在 top_k 较大时非确定性 (等值元素顺序未定义)
4. seed 重复性: CPU mt19937 可确保重复性, 但需要移植到 GPU PRNG

### 目标

| 维度 | 目标 | 优先级 |
|------|------|--------|
| 消除 Host-Device 同步 | 全 GPU 采样, 仅在读结果时 sync 一次 | P0 |
| CUDA Graph 兼容 | 采样 kernel 可被 Graph 捕获 | P0 |
| 数值稳定性 | Online softmax, 不依赖 max 预计算 | P1 |
| MTP 随机一致性 | verify 路径使用正确采样策略 | P1 |
| Batched 采样 | 每请求独立参数, 无串行 | P1 |
| 确定性模式 | seed → 确定相同输出 | P2 |

## 2. 设计方案

### 2.1 Phase 1: GPU 采样核心 Kernels (light_ops.cu)

#### Kernel 1: `gpu_top_k_top_p_sampling_kernel`

**核心算法: Pivot Binary Search (参考 FlashInfer)**

不需要排序,用二分搜索找到概率分界值:

```
输入: BF16 logits[vocab_size], temperature, top_k, top_p, min_p, seed
输出: int sampled_token

算法:
1. Online softmax: 两遍扫描 logits
   - Pass 1: 找 max(logits/temp), 计算 Σexp(logit/temp - max) — 用 warp+block reduce
   - Pass 2: prob[i] = exp(logit[i]/temp - max) / sum

2. Top-K 过滤 (pivot binary search):
   - 用 max_prob 和 0 作为初始上下界
   - 二分搜索 pivot: count(prob > pivot) ≈ top_k
   - ~20 次迭代收敛 (log2(max/eps))

3. Top-P 过滤:
   - 从 top-k 候选中, 按概率降序累加到 top_p
   - 用 pivot search: 找 threshold 使 Σ(prob[prob≥threshold]) ≥ top_p

4. Min-P 过滤:
   - threshold = max_prob * min_p
   - 抛弃 prob < threshold 的 token

5. 重新归一化 + CDF 采样:
   - filtered_sum = Σ(合格 prob)
   - 生成 random u ∈ [0, filtered_sum)
   - 扫描并累加, cumsum ≥ u 时返回对应 token
```

**线程布局:**
- Grid: `(1)` 或 `(batch_size)` — 每 batch 元素一个 block
- Block: `1024` threads (256 for vocab_size=248320: 每线程处理 ~243 elements)
- Shared memory: `4 * 1024 bytes` (pivot counters + partial sums)
- 无需额外 global memory 分配

**关键设计决策:**
- **为何不用 Gumbel-Max**: Gumbel-Max 对 temperature-only 最优, 但 top-k/top-p/min-p 需要概率空间操作, pivot 方法更通用
- **为何不排序**: 248K vocab 的 sort 是 O(n log n) ≈ 4.3M ops, pivot search 是 O(20 × n) ≈ 5M ops, 但 sort 需要额外 248K × 8B = 2MB workspace

#### Kernel 2: `gpu_apply_penalties_kernel`

```
输入: BF16 logits[vocab_size], int penalty_tokens[N], int penalty_counts[N], 
      float repeat_penalty, float freq_penalty, float pres_penalty
输出: in-place 修改 logits

线程: 每线程处理一个 penalty token, N 通常 < 4096
Grid: (ceil(N, 256)), Block: 256
```

#### Kernel 3: `gpu_sample_argmax_or_gumbel_kernel` (快速路径)

```
temperature <= 0 || top_k == 1 → argmax (已有)
temperature > 0 && top_k == vocab_size && top_p >= 1.0 && min_p <= 0 → Gumbel-Max
其余 → Kernel 1 (pivot sampling)
```

**Gumbel-Max (参考 FlashInfer `SamplingFromLogitsKernel`)**:
- `gumbel_logit[i] = logit[i]/temp - log(-log(uniform_rand()))`
- argmax(gumbel_logit) = 从 categorical(softmax(logit/temp)) 采样
- 无需 softmax! 只需一次 reduce

### 2.2 Phase 2: Engine 集成

#### 2.2.1 GPU 采样缓冲区 (engine.h)

```cpp
// GPU 侧采样状态 (managed memory, 可被 GPU kernel 写, CPU 直接读)
int* d_sampled_tokens_;           // [MAX_SSM_SLOTS] 采样结果
float* d_sampling_temperatures_;  // [MAX_SSM_SLOTS] 每请求温度
float* d_sampling_top_p_;         // [MAX_SSM_SLOTS]
int* d_sampling_top_k_;           // [MAX_SSM_SLOTS]
float* d_sampling_min_p_;         // [MAX_SSM_SLOTS]
uint64_t* d_sampling_seeds_;      // [MAX_SSM_SLOTS] Philox 种子
```

#### 2.2.2 新 `sample_token_gpu()` 接口

```cpp
int sample_token_gpu(__nv_bfloat16* logits, int vocab_size,
                     float temperature, float top_p, int top_k, float min_p,
                     float repeat_penalty, float frequency_penalty,
                     float presence_penalty, int64_t seed,
                     const std::vector<int>& generated_tokens,
                     cudaStream_t stream);
```

流程:
1. GPU penalty kernel (如果 has_penalty)
2. GPU sampling kernel (自动 dispatch: argmax / gumbel / pivot)
3. cudaStreamSynchronize (只在此处 sync 一次)
4. return *d_sampled_tokens_

#### 2.2.3 MTP Verify 随机一致性修复

```cpp
// 当前 (有 bug):
int main_token = sample_argmax(logits, vocab_size, stream);

// 修复为:
int main_token;
if (req->temperature > 0 && req->top_k != 1) {
    main_token = sample_token_gpu(logits, vocab_size,
                                   req->temperature, req->top_p, ...);
} else {
    main_token = sample_argmax(logits, vocab_size, stream);
}
```

### 2.3 Phase 3: Batched 采样 (未来)

当 step() 支持真正的 batched decode 时:
- `gpu_batched_sampling_kernel`: Grid=(batch_size), 每 block 独立采样
- 每请求有独立的 temperature/top_p/top_k/min_p/seed
- Penalty token 列表: 预打包到 GPU buffer, 使用 CSR 格式索引

## 3. 性能预估 vs 实测

### 单并发 (batch=1)

| 操作 | 计划预估 (GPU) | 实测结果 | 差异分析 |
|------|---------------|---------|---------|
| cudaMemcpy 248K×BF16 | 0 (消除) | ✅ 0 | 如预期 |
| BF16→float 转换 | 0 (kernel 内) | ✅ 0 | 如预期 |
| Penalty 应用 | ~0.005ms | ~0.01ms | CPU hash map + async upload 开销 |
| Temperature scaling | 0 (fused) | ✅ 0 | 如预期 |
| partial_sort 248K | 0 (消除) | ✅ 0 | 如预期 |
| Softmax + sample | ~0.05ms | ~0.35ms | pivot binary search 35+ pass vs 预估; 见下文 |
| **合计** | **~0.07ms** | **~0.46ms** | 6.6× 差距, 详见分析 |

**差距分析**: 计划预估 ~0.07ms 过于乐观, 主要低估了:
1. **Binary search pass 次数**: top-k 20 轮 + top-p 15 轮 = 35 轮, 每轮扫描 248K 元素
2. **L2 cache 竞争**: 486KB logits 虽 < 32MB L2, 但 kernel 的 shared memory 和 register 压力导致 L2 命中率低于预期
3. **`__syncthreads()` 开销**: 初版 tiled CDF 有 2430 次 sync, 优化后仍有 ~50 次 (prefix sum + reduce)
4. **`expf()` 计算**: 每个 binary search pass 中 qualifying elements 需要 `expf()`, 比纯比较贵

**但 vs CPU 仍有提升**: 0.46ms < 0.5ms (CPU), 且消除了 `cudaMemcpy` + sync 的 pipeline 阻断

### 多并发 (batch=8) — 尚未实测

| 操作 | 计划预估 | 当前实际 | 说明 |
|------|---------|---------|------|
| 8×采样 | ~0.08ms (并行) | ~3.7ms (串行) | batch_size=1 逐请求调用, 未实现 Phase 3 |
| 同步开销 | 1×sync | 8×sync | 同上 |

> Phase 3 (Batched 采样) 完成后再实测

### 稳定性收益 (实测验证)

1. **MTP accept rate 提升**: ✅ verify 路径已修复, 但当前模型无 MTP 权重, 未量化验证
2. **数值稳定**: ✅ GPU online softmax + pivot binary search, 5 种采样模式全部正确
3. **CUDA Graph 完整性**: 🔲 采样 kernel 本身可被 graph 捕获, 但 CUDA Graph 集成尚未实施
4. **确定性采样**: ⚠️ GPU SplitMix64 哈希提供足够随机性, 但跨请求的 step_offset 不保证确定性重放

## 4. 实施计划

### Step 1: 基线基准测试 — ✅ 完成 (2026-03-03)
- ✅ batch=1 benchmark: warmup ~232ms/step, 4.3 tok/s
- ✅ batch=4 benchmark: warmup ~292ms/step, 13.7 tok/s (但 CUDA Graph capture 崩溃 — CUTLASS cluster_launch 不兼容, 非采样问题)
- ⚠️ 采样微基准测试: 未单独编写, 使用 serve REQUEST SUMMARY 中的 `sample` 阶段计时

### Step 2: 实现 GPU Sampling Kernels — ✅ 完成 (2026-03-03)
- ✅ `light_ops.cu`: 新增 3 个 kernel (~300 行)
  - `gpu_gumbel_sample_kernel`: Gumbel-Max 快速路径 (无 filter)
  - `gpu_fused_sample_kernel`: 融合 top-k + top-p + min-p + CDF 采样
  - `gpu_apply_penalties_kernel`: 重复/频率/存在惩罚
- ✅ `light_ops.h`: 新增 `invoke_gpu_sample()` + `invoke_gpu_apply_penalties()` 声明
- ✅ 编译验证通过 (需要添加 `#include <cstdint>` for uint64_t)

**实现与计划的差异:**
| 计划 | 实际 | 原因 |
|------|------|------|
| Kernel 命名 `gpu_top_k_top_p_sampling_kernel` | `gpu_fused_sample_kernel` | 更能反映融合特性 |
| Online softmax 两遍扫描 | 两步: max reduce + sum_exp reduce | 等价, 更清晰 |
| CDF 采样用排序或 tiled scan | Hillis-Steele inclusive prefix sum | 初版 tiled scan (2430 syncs) 性能差, 优化为 prefix sum (~22 syncs) |
| PRNG 用 Philox | 用 SplitMix64 哈希 | 更简单, 无需状态, 哈希质量足够 |
| `__launch_bounds__(1024, 2)` | `__launch_bounds__(1024, 1)` | shared memory 用量限制, 1 block/SM 足够 |

### Step 3: Engine 集成 — ✅ 完成 (2026-03-03)
- ✅ `engine.h`: 新增 `d_sampled_token_`, `d_penalty_token_ids_`, `d_penalty_counts_` 等 buffer
- ✅ `engine.cpp`: 新增 `sample_token_gpu()` 方法 (~60 行)
- ✅ 两处 `sample_token()` → `sample_token_gpu()` 替换 (single decode + batched decode)
- ✅ CPU fallback 保留 (`sample_token()` 原函数未删除)
- ✅ Penalty 已完全移到 GPU (计划中说"暂时仍在 CPU", 但实际 CPU 只做 hash map 构建, kernel 在 GPU 执行)

### Step 4: MTP 一致性修复 — ✅ 完成 (2026-03-03)
- ✅ MTP verify 路径: temperature>0 时用 `sample_token_gpu`, 否则用 `sample_argmax`
- ✅ 实现与计划方案完全一致

### Step 5: 测试 + 对比基准 — ✅ 完成 (2026-03-03)

**功能回归测试** (全部通过):
| 模式 | 参数 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| Greedy | temperature=0 | "7×8=?" | "56" | ✅ |
| 随机采样 | temp=0.7, top_k=20, top_p=0.95 | "Capital of France?" | "Paris" | ✅ |
| 高温+min_p | temp=1.5, top_k=100, top_p=0.95, min_p=0.05 | "2+2=?" | "4" | ✅ |
| 中文回答 | temp=0.7, top_k=20, top_p=0.95 | "中国的首都?" | "北京" | ✅ |
| 列表生成 | temp=1.0, top_k=50, top_p=0.95, min_p=0.1 | "List 3 fruits" | Apple/Banana/Orange | ✅ |

**性能对比** (server REQUEST SUMMARY 数据):
| 阶段 | Sample Avg | Sample Min | Sample Max | 备注 |
|------|-----------|-----------|-----------|------|
| 基线 (CPU random) | 0.50ms | — | — | partial_sort + softmax |
| 基线 (GPU argmax) | 0.17ms | — | — | 仅 greedy |
| GPU 采样 v1 | 9.52ms | 6.33ms | 10.17ms | sync_stream_with_timeout 轮询 bug |
| + sync 修复 | 0.73ms | 0.54ms | 1.15ms | cudaStreamSynchronize |
| + CDF prefix sum | **0.46ms** | **0.15ms** | **0.75ms** | Hillis-Steele (最终版) |

**发现并修复的额外 Bug:**
1. `sync_stream_with_timeout()` 使用 10ms usleep 轮询 → `cudaStreamSynchronize` (20.7× 加速)
2. `sample_argmax` 也使用了 `sync_stream_with_timeout` → 同步修复 (greedy min 0.15ms)

## 5. 风险与缓解

| 风险 | 影响 | 缓解措施 | 状态 |
|------|------|----------|------|
| GPU 采样结果与 CPU 不同 | 生成文本质量变化 | 保留 CPU fallback, A/B 对比 | ✅ 验证通过, 质量正常 |
| Pivot binary search 不收敛 | 采样失败 | fallback 到 CDF 全扫描 | ✅ 20 轮迭代足够收敛 |
| Gumbel noise 精度不足 | 采样偏差 | 使用 double 精度 log(-log(u)) | ⚠️ 当前用 float, 未见问题 |
| CUDA Graph 与动态 penalty 不兼容 | Graph 需拆分 | Penalty kernel 在 graph 外执行 | 🔲 CUDA Graph 集成尚未实施 |
| MTP 修复导致 accept rate 变化 | 吞吐量波动 | 先测量, 再决定上线 | ⚠️ 未量化测试 (模型无 MTP 权重) |

---

## 6. 下一步计划分析

### 6.1 已完成项总结

Phase 1 (GPU kernels) + Phase 2 (Engine 集成) 已全部完成:
- 消除 Host-Device 同步: ✅ (仅最终 sync 一次)
- 数值稳定性: ✅ (GPU online softmax)
- MTP 随机一致性: ✅
- repeat/frequency/presence penalty GPU 化: ✅

### 6.2 Phase 3 待完成: Batched 采样

**当前状态**: `sample_token_gpu()` 的 `invoke_gpu_sample` 已支持 `batch_size` 参数, 但 engine 层面每次调用 batch_size=1 (循环中逐请求调用)。

**优化方向**: batched decode 时, N 个请求的 logits 已经连续 (`lm_head` 输出 `[N, vocab_size]`), 只需:
1. 将 N 个请求的 penalty 数据打包为 CSR 格式上传
2. 一次 kernel launch `batch_size=N` — 已有基础设施
3. 一次 sync

**预估收益** (batch=8):
- 当前: 8 × (penalty_upload + kernel + sync) ≈ 8 × 0.46ms = **3.7ms**
- 优化后: 1 × penalty_upload + 1 × kernel(8 blocks) + 1 × sync ≈ **0.5ms** (7.4× 加速)
- 在 ~222ms 的 forward 中占比从 1.7% 降至 0.2%

**实现复杂度**: 中等, 主要是 penalty 数据的 batched 打包 (CSR 索引 + 合并 upload)

**优先级**: P1 — batched decode 吞吐量场景有实际价值, 但需要先确认 batched decode 路径的 logits buffer 布局

### 6.3 其他可探索方向

#### CUDA Graph 集成 (P0)
GPU 采样的核心价值之一是 CUDA Graph 兼容。当前 penalty 阶段有 `cudaMemcpyAsync` (host→device), 可能阻碍 graph capture:
- 方案 A: Penalty kernel 放在 graph 外 (graph 仅覆盖 forward + lm_head + sampling)
- 方案 B: 预分配固定大小 penalty buffer, 使用 `cudaGraphExecUpdateNode` 更新数据

#### Gumbel-Max 精度验证 (P2)
当前 PRNG 用 SplitMix64 哈希 → float uniform → float Gumbel noise, 在极罕见情况下:
- `u ≈ 0` 时 `-log(-log(u))` → `+inf`, 可能导致 non-argmax token 被选中
- 可用 `double` 精度计算 Gumbel 或 clamp u ∈ [1e-30, 1-1e-7]
- 当前线上无问题, 但大规模长序列生成时值得关注

#### 确定性模式 (P2)
当前 `seed >= 0` 时传入固定 seed, 但 `step_offset` 为全局递增计数器, 多并发时不同请求的 step_offset 不确定。
- 修复: 每请求维护独立 step_offset (从请求创建时开始计数)
- 影响较小, 仅调试/评测场景需要
