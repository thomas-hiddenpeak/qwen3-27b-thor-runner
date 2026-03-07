# SM110a 自底向上优化路线图

> 目标: Qwen3.5-27B 在 Jetson AGX Thor (SM110a) 上的极致推理性能
> 创建: 2026-03-05
> 状态: 规划中

## 当前基线

| 指标 | 值 |
|---|---|
| Decode 吞吐 | ~4.31 tok/s |
| 有效带宽 | ~220 GB/s (峰值 273 GB/s 的 80%) |
| 每步权重读取 | ~51 GB |
| GDN State 开销 | 72 MB/request (BF16, 48 层 × 128 × 128 × 2B × 48 heads) |
| 最大并发 | 受限于 128 GB 统一内存 |

## 核心设计原则

1. **自底向上**: 先探测硬件原语，再定内存布局，再写 kernel，最后做系统调度。每一层为上层提供稳定抽象，避免反复重构。
2. **SM110a 由我们定义**: 不等 NVIDIA 官方文档，通过 micro-benchmark 实测硬件行为，建立第一手数据。
3. **硬件×模型交叉驱动**: 优化决策基于硬件特性与模型架构的具体交叉分析，而非通用经验。

## 硬件×模型交叉分析

### 模型架构特征

| | Linear Attention (GDN) | Full Attention (GQA) |
|---|---|---|
| 层数 | 48 / 64 = **75%** | 16 / 64 = **25%** |
| Decode 每层 State I/O | 128×128×FP32×48 heads = **3 MB** | KV cache 读取 (随序列长度增长) |
| Decode 核心操作 | rank-1 update + 向量内积 | 标准 QK^TV + online softmax |
| Prefill 计算密度 | WY 矩阵乘 (compute-bound T≥8) | 标准 attention (compute-bound) |

### SM110a 硬件原语 × 应用场景

| 原语 | 状态 | GDN 适用性 | Attention 适用性 |
|---|---|---|---|
| TMEM | ✅ 实测 | State 累加器, 128×128=25% 容量 ✅ | Softmax partial sum |
| UMMA SS (5th-gen TC) | ✅ 实测 1.6T/SM | rank-1 K=1 < K_ATOM=16 ❌ | QK^T/PV 标准 GEMM ✅ |
| UMMA TS | ✅ 实测 = SS | WY chain MMA (TMEM→TC) ✅ | FMHA pipeline ✅ |
| TMA | ✅ 实测 88.6 GB/s/SM | State tile 异步加载 ✅ | KV block (<16KB 不适用) |
| UTCCP | ✅ 实测 25.2 GB/s/SM | SMEM→TMEM 前置 TS 输入 ✅ | FMHA SMEM→TMEM ✅ |
| TMEM pack/unpack | ✅ pack 1.4×, unpack 5.5× | FP32→BF16 epilogue ✅ / BF16→FP32 ⚠️ | accumulator 管理 |
| f32x2 SIMD | ✅ 实测 1.97× | 标量操作加速 ✅ | Norms/Activation ✅ |
| LDSM/STSM b8 | ✅ 实测 222/45 GB/s/SM | INT8 路径基线 | INT8 路径基线 |
| SMEM 228 KB/SM | 已使用 (~162 KB for WY) | WY 已接近极限 | 多 stage pipeline 需要空间 |
| L2 32 MB | 部分利用 | 48 层 state 144 MB >> L2 | 16 层 KV 也远超 L2 |
| L1::no_allocate | ✅ 实测无差异 | 不适用 | 不适用 |
| Reg Reconfig | ✅ 57ns | warp-specialization ✅ | warp-specialization ✅ |
| PDL | ✅ 6.3ns | 64 层 kernel 间隙消除 ✅ | kernel 间隙消除 ✅ |
| UMMA Fence/Commit | ✅ <13ns | pipeline 同步免费 ✅ | pipeline 同步免费 ✅ |

### 关键推论

- GDN 影响 75% 的层, 每提升 GDN 1% ≈ 提升 Attention 3% 等效收益
- Decode 瓶颈是带宽 (51 GB 权重 + 144 MB state + KV cache, 经 273 GB/s 通道)
- State BF16 化同时省内存 (72 MB) 和省带宽 — **实测: B=128 吞吐 +42.6%**
- TMA 对 GDN state 更友好 (固定 shape), 对 paged KV cache 更复杂 (散列 block)
- ~~UMMA 对 GDN rank-1 update 的适用性需实测验证~~ → **实测确认: K=1 < K_ATOM=16, 无法映射**

---

## 优化路线

### Level 0: SM110a 硬件原语探测

**目标**: 实测 TMEM/UMMA/TMA 的真实能力, 建立硬件抽象层

**产出**: `sm110a_primitives.h` + `sm110a_probe.cu` micro-benchmark 数据

#### 实测结果 (2026-03-05, MAXN, Statistical Edition: 20 trials, 5 warmup)

运行命令: `./build/qwen3-27b-thor probe`

##### TMEM / UMMA / TMA (片上原语, CV=0.0% ★★★)

| 指标 | 实测值 (median) | CV | 备注 |
|---|---|---|---|
| TMEM 分配延迟 | 221.6 ns | 0.0% | 列数无关 (32/64/128 均一致) |
| TMEM 释放延迟 | 119.4 ns | 0.0% | |
| TMEM Store @128cols | 5.3 ns/col | 0.0% | 总 677.5 ns |
| TMEM Load @128cols | 10.5 ns/col | 0.0% | 总 1344.1 ns |
| TMEM 128cols roundtrip | ~2.0 μs | — | store+load, 远快于 GMEM |
| UMMA SS [64,16]×[16,64] | 1.60 TFLOPS/SM | 0.0% | 100~10000 iters 均稳定 |
| TMA LOAD 32KB | 369.8 ns | 0.8% | 88.6 GB/s/SM |
| TMA LOAD 16KB | 286.3 ns | 0.2% | 交叉点: TMA ≈ manual |
| TMA STORE 32KB | 704.5 ns | 0.0% | 46.5 GB/s/SM |
| TMA 交叉点 | ≥16KB | — | TMA wins vs manual copy |
| TMA 最大 tile | 32KB | — | 48KB 触发 SMEM/block limit |

##### DRAM 带宽 (cudaEvent timing, 全 20 SM)

| 指标 | 实测值 (median) | CV | 峰值% | 备注 |
|---|---|---|---|---|
| Read 64MB | 263.0 GB/s | 9.5% | 96% | |
| Read 256MB | 228.8 GB/s | 11.4% | 84% | CV>10%: 统一内存调度抖动 |
| Read 512MB | 228.1 GB/s | 8.6% | 84% | |
| **Write 64MB** | **169.0 GB/s** | **3.8%** | **62%** | ⚠️ 读写非对称! |
| **Write 256MB** | **176.6 GB/s** | **3.7%** | **65%** | |
| **Write 512MB** | **149.3 GB/s** | **1.9%** | **55%** | |
| L2 BW 1MB | 3244 GB/s | 4.9% | — | 纯 L2 |
| L2 BW 32MB | 1425 GB/s | 0.8% | — | 32MB = L2 size |

**⚠️ 新发现: LPDDR5X 写带宽仅为读取的 55-65%** (149-177 vs 228-263 GB/s)

##### Multi-SM Scaling (256MB read)

| SM 数 | 中值 BW | 峰值% | per-SM 效率 |
|---|---|---|---|
| 1 | 27.1 GB/s | 10% | 100% (baseline) |
| 5 | 111.7 GB/s | 41% | 82% |
| 10 | 180.2 GB/s | 66% | 66% |
| 20 | 248.5 GB/s | 91% | 46% |

线性缩放到 ~10 SM，之后 DRAM 带宽饱和，边际递减。

##### 扩展 SM110a 特性 (Probes 12-20, CV=0.0% ★★★)

| 指标 | 实测值 (median) | CV | 备注 |
|---|---|---|---|
| **UMMA TS [64,16]×[16,64]** | **1.60 TFLOPS/SM** | 0.0% | = SS 吞吐, A 从 TMEM 读取 |
| UTCCP 128x256b (4KB) | 162.6 ns | 0.0% | 25.2 GB/s/SM, SMEM→TMEM |
| UTCCP 4x256b (128B) | 10.2 ns | 0.0% | 12.6 GB/s/SM, 小块灵活 |
| TMEM normal load | 1.0 ns/op | 0.0% | 32x32b.x4 基线 |
| TMEM pack::16b load | 1.4 ns/op | 0.0% | FP32→BF16 仅 1.4× 开销 |
| TMEM normal store | 2.0 ns/op | 0.0% | 基线 |
| **TMEM unpack::16b store** | **10.9 ns/op** | 0.0% | **5.5× 基线! BF16→FP32 昂贵** |
| DRAM read L1::no_allocate | 260.5 GB/s | — | ≈ standard (261 GB/s), 无差异 |
| f32 scalar FMA | 2.8 ns/1024ops | 0.0% | 基线 |
| **f32x2 FMA** | **2.9 ns/1024ops** | 0.0% | **1.97× throughput 几乎理论值** |
| LDSM b8 m16n16 | 1.2 ns | 0.0% | 222 GB/s/SM |
| STSM b8 m16n8 | 2.9 ns | 0.0% | 45 GB/s/SM |
| UMMA wait::ld | 1.1 ns | 0.0% | 近零同步 |
| UMMA wait::st | 7.5 ns | 0.0% | |
| UMMA commit | 12.6 ns | 0.0% | pipeline 同步免费 |
| **Reg Reconfig (inc+dec)** | **57.3 ns** | 0.0% | warp-specialization 可行 |
| PDL launch_dependents | 5.2 ns | 0.0% | 64 层 kernel 间隙可消除 |
| PDL wait | 1.1 ns | 0.0% | ultra-low |

##### Memory Access Latency (pointer chasing, 1 thread)

| 工作集 | 延迟 (median) | CV | 层级 |
|---|---|---|---|
| 4KB | 24.7 ns | 0.0% | L1 |
| 16KB | 24.8 ns | 0.0% | L1/L2 |
| 64KB | 26.4 ns | 0.0% | L2 near |
| 256KB | 85.5 ns | 0.1% | L2 |
| 4MB | 159.7 ns | 0.0% | L2 deep |
| 32MB | 161.9 ns | 0.0% | L2=32MB |
| 256MB | 169.9 ns | 0.3% | DRAM |

**L1→L2 断崖**: 64KB→256KB (26→86 ns, 3.3×)
**L2→DRAM 过渡**: 渐进式 (32MB→256MB, 162→170 ns, 仅5%)
★ 统一内存下 DRAM 延迟接近 L2 deep — TLB walk 主导

#### 关键结论

1. ✅ **TMEM 可存放完整 GDN State**: 128×128 FP32 = 128 列 = 25% TMEM 容量, 剩余 384 列可供 WY 中间结果
2. ❌ **UMMA 不适用于 GDN decode rank-1 update**: K=1 不满足 K_ATOM=16, 必须保持标量
3. ✅ **UMMA 适用于 WY prefill GEMMs**: inner dim = kd = 128, AI=128 ≈ 临界点, compute-bound
4. ✅ **UMMA 适用于 FullAttn QK^T/PV**: inner dim = head_dim = 256 >> K_ATOM=16
5. ✅ **TMA 32KB tile 比手动 copy 快 4.2×**: GDN state tile 应使用 TMA, KV block (8KB) 保持手动
6. ✅ **DRAM 读 249 GB/s (91%), 写仅 149-177 GB/s (55-65%)**: LPDDR5X 读写非对称
7. ⚠️ **State 写回受限**: 144 MB state × 写 BW 175 GB/s = 0.82 ms → BF16 化可减半至 0.41 ms
8. ✅ **L1→L2 断崖 3.3×**: 25ns → 86ns, 关键数据必须 fit L1 (≤16KB/线程)
9. ✅ **SM 线性缩放到 10 SM**: 之后 DRAM 饱和, 20 SM per-SM 效率仅 46%
10. ✅ **UMMA TS = SS 吞吐 (均 1.6 T/SM)**: TS 从 TMEM 读 A, 省 SMEM 往返, GDN WY 可 chain MMA
11. ✅ **UTCCP 25.2 GB/s/SM @4KB**: SMEM→TMEM 拷贝可行, TS 模式前置操作
12. ✅ **TMEM pack::16b 几乎免费 (1.4×)**: FP32 accumulator→BF16 输出零开销; ⚠️ unpack 5.5× 慢, 避免 BF16→FP32 TMEM 回写
13. ✅ **L1::no_allocate 无效果**: 统一内存 LPDDR5X 下 L1 bypass 无额外收益 (260≈261 GB/s)
14. ✅ **f32x2 SIMD 1.97× 加速**: RMSNorm/SiLU/sigmoid 等逐元素操作改用 f32x2 可消减 50% 指令数
15. ✅ **LDSM b8 极快 222 GB/s/SM**: INT8 SMEM→Reg 传输基线; STSM b8 较慢 45 GB/s/SM
16. ✅ **UMMA fence/commit 开销极低 (<13ns)**: Pipeline 同步可放心使用
17. ✅ **Reg Reconfig 57ns**: warp-specialization 每层仅 1 次切换, <0.03% decode step
18. ✅ **PDL 5.2+1.1ns**: 64 层 forward 的 kernel launch gap 可用 PDL 消除 (理论省 ~320μs)

#### Roofline Model

```
算力天花板:
  UMMA BF16→FP32:  32.0 TFLOPS (1.60 T/SM × 20 SM, CV=0.0%, TS=SS)
  CUDA FP32:       ~5.0 TFLOPS (保守估计, 理论 8.06T)
  f32x2 SIMD:      ~10.0 TFLOPS (f32x2 ≈ 2× scalar, 实测 1.97×)

带宽天花板:
  DRAM Read:   249 GB/s (20 SM, 91% peak, CV=3.3%)
  DRAM Write:  149~177 GB/s (55-65% peak) ← 非对称!
  DRAM Read L1::no_allocate: 260 GB/s (≈ standard, 无差异)
  L2:         3244 GB/s (@1MB, fits-in-L2)
  TMEM:       片上, 不走 memory hierarchy
  UTCCP:      25.2 GB/s/SM (504 GB/s @20SM, SMEM→TMEM)
  LDSM b8:    222 GB/s/SM (4440 GB/s @20SM, SMEM→Reg INT8)

同步开销 (可忽略):
  UMMA fence/commit: <13 ns → pipeline 调度免费
  Reg Reconfig:      57 ns → warp-specialization 每层 1×
  PDL launch+wait:   6.3 ns → 消除 kernel 间隙

算术强度临界点 (FLOP/Byte):
  UMMA:  32000/249 = 128.5 F/B (DRAM read limited)
  FP32:   5000/249 =  20.1 F/B

各操作分析:
  Decode GEMV:    AI=2.0  → BW-bound   → 249 GB/s 限制
  GDN rank-1:    AI=0.5  → BW-bound (GMEM) / ∞ (TMEM)
  WY GEMM:       AI=128  → Compute-bound (UMMA, 恰好>128.5 临界)
  FullAttn QK^T: AI=2T   → BW/Compute depends on T
  Norms/Activ:   AI≈1.5  → L2-bound, <10ns, 可忽略
  State 写回:    AI→0    → Write BW-bound (149-177 GB/s)
```

| 子任务 | 内容 | 关键问题 | 状态 |
|---|---|---|---|
| TMEM 探测 | 容量 / 读写延迟 / 与 SMEM 和寄存器的交互模式 | 128×128 FP32 state 能否放入? ✅ YES | ✅ 完成 |
| UMMA 探测 | BF16 吞吐实测 / 支持的 tile shape / TS/SS 模式 | rank-1 update 能否映射? ❌ NO | ✅ 完成 |
| TMA 探测 | cp.async.bulk 延迟/BW / vs 手动 copy 对比 | State tile 最优传输? ✅ TMA ≥16KB | ✅ 完成 |
| 综合 Roofline | 基于实测数据建立 SM110a roofline model | 各操作的理论上限? | ✅ 完成 |
| UMMA TS 模式 | TMEM→TC 直通, TS vs SS 吞吐对比 | TS 能否省 SMEM 往返? ✅ TS=SS | ✅ 完成 |
| UTCCP | SMEM→TMEM 拷贝带宽 (128x256b / 4x256b) | TS 前置可行? ✅ 25.2 GB/s/SM | ✅ 完成 |
| TMEM pack/unpack | FP32↔BF16 精度转换开销 | 是否免费? pack ✅ 1.4×, unpack ⚠️ 5.5× | ✅ 完成 |
| L1::no_allocate | L1 bypass 对统一内存的效果 | 有额外收益? ❌ 无差异 | ✅ 完成 |
| f32x2 SIMD | f32x2 FMA vs scalar 吞吐对比 | 加速比? ✅ 1.97× | ✅ 完成 |
| LDSM/STSM b8 | INT8 SMEM 矩阵传输 | 基线带宽? LDSM 222/STSM 45 GB/s/SM | ✅ 完成 |
| UMMA Fence/Commit | Pipeline 同步开销 | 是否免费? ✅ <13ns | ✅ 完成 |
| Reg Reconfig | setmaxnreg inc/dec 延迟 | warp-specialization 可行? ✅ 57ns | ✅ 完成 |
| PDL | griddepcontrol 延迟 | kernel 间隙可消除? ✅ 6.3ns total | ✅ 完成 |

**参考资源**:
- `third_party/cutlass/` 中的 SM100/SM110 arch 定义
- FlashInfer `blackwell/` 目录中的 UMMA/TMA 用法 (SM100 实现)
- 已有的 `gdn_umma_sm110.cu` 中 UMMA atom 类型定义 (未启用)

**状态**: ✅ 完成

---

### Profiling 方法论: NVTX + Nsight Systems / Compute

**原则**: 放弃手写 CUDA event Profiler，全面引入 NVTX 标记 + Nsight 工具链。

#### 工具职责划分

| 工具 | 职责 | 对应优化层 |
|---|---|---|
| **NVTX + nsys** | 宏观耗时分布、kernel launch 间隙、CPU/GPU 重叠、内存事件 | Level 1: 识别热点层和瓶颈类型 |
| **ncu** | 单 kernel 硬件利用率 (DRAM BW%、L2 hit%、SM occupancy、warp stall) | Level 2+: kernel 重写的优化指导 |
| **PerfStats** (保留) | 生产级 per-request 统计 (TTFT、TPOT、tok/s) | 运行时监控, serve 模式 |

#### NVTX 层级设计 (4 级嵌套)

```
L0: Request    — request_id, 覆盖完整请求生命周期
L1: Phase      — prefill / decode step N
L2: Layer      — layer_0 ~ layer_63, 标注类型 (GDN / GQA)
L3: Op         — GEMV / DeltaNet / Attention / RMSNorm / SiLU 等
```

- 不挂 profiler 时 NVTX 标记是 NOP, 生产环境零开销
- 比手写 CUDA event timing 更轻量, 且能捕获 CPU/GPU 重叠、统一内存 page fault 等系统级信息

#### 关键注意事项

1. **Jetson 兼容性**: AGX Thor JetPack 自带 `nsys` CLI; GUI 需 x86 主机远程分析 `.nsys-rep` 文件
2. **SSM State 与 NCU Replay**: GDN 的 SSM state 有状态, Nsight Compute 默认 kernel replay 会破坏 state 一致性 → 必须使用 `--replay-mode application` 或隔离单层测试
3. **保留最小生产级监控**: `PerfStats` 保留请求级聚合 (TTFT/tok/s), 层级计时由 NVTX 替代
4. **CMake 集成**: 链接 `nvToolsExt` (或 CUDA 12+ 内置 `nvtx3`), 编译开关 `-DENABLE_NVTX`

#### 典型工作流

```bash
# Level 1: 宏观 timeline — 识别热点层
nsys profile --trace=cuda,nvtx -o baseline ./build/qwen3-27b-thor chat --rounds 3

# Level 2: 微观 kernel 分析 — 锁定热点后
ncu --replay-mode application --set full \
    --kernel-name "gated_delta_net_kernel|paged_attention_kernel" \
    ./build/qwen3-27b-thor chat --rounds 1
```

**状态**: 未开始 — 需插入 NVTX 标记 (~20 处) + CMake 配置

---

### Level 1: 内存布局基础设施

**目标**: 为 GDN state 建立最优存储方案

**依赖**: Level 0 (TMA 对齐要求, TMEM 容量决定是否需要 tile)

| 子任务 | 内容 | 状态 |
|---|---|---|
| State BF16 化 | GMEM 存储 BF16, kernel 内 FP32 计算, load 升精度 / store 降精度 | ✅ 已完成 |
| State 布局调整 | 评估 pretranspose `[B*HV, V, K]` (K 连续) vs 当前布局, 对齐 TMA tile 要求 | 未开始 |
| 精度验证 | BF16 state 对 DeltaNet 的精度影响, perplexity 回归测试 | ✅ 已验证 |

**产出**: state allocator + load/store 原语 + 精度报告

**实测收益 (2026-03-05)**:
- 内存: 144 → 72 MB/request (**50% 节省**)
- B=1 decode: 无回归 (218ms, 4.58 tok/s)
- B=128 decode: 吞吐 **+42.6%** (264→377 tok/s)
- 并发: 最大请求数约翻倍 (464→928)

**状态**: ✅ State BF16 化已完成, 布局调整待定

---

### Level 2: GDN Decode Kernel 重写

**目标**: 基于 Level 0/1 的硬件抽象和内存布局, 重写 GDN decode kernel

**依赖**: Level 0 (硬件原语), Level 1 (State BF16 + 布局)

| 子任务 | 内容 |
|---|---|
| TMA pipeline | State tile 用 TMA 异步加载, 2-stage 双缓冲 |
| K-persistent | T=1 时 K 加载一次到寄存器 (32 lanes × 4 FP32 = kd=128), 跨所有 V-tile 复用 |
| TMEM 累加 | 如果 Level 0 验证可行: State 更新走 TMEM, 省 SMEM 空间 |
| UMMA 评估 | 如果 Level 0 验证 rank-1 update 可映射: 用 UMMA 加速状态更新 |

**参考**:
- FlashInfer `gdn_decode.py`: K-persistent 模式, cp.async 2-stage, pretranspose 布局
- FlashInfer `gdn_decode_bf16_state.py`: BF16 state + 4-chunk pipeline

**当前实现差距**:
- 现有 `gated_delta_net_kernel` 直接全局访存 state (每 token 每层 64 KB global I/O)
- 无 pipeline, 无 K 复用, 逐元素标量加载

**状态**: 未开始

---

### Level 3: Attention Decode Kernel 升级

**目标**: 基于 Level 0 的硬件抽象, 升级 paged attention decode kernel

**依赖**: Level 0 (TMA/UMMA)

| 子任务 | 内容 |
|---|---|
| 向量化 | vec_size=8 (128-bit per thread), 替代当前逐元素标量加载 |
| TMA/cp_async pipeline | KV tile 异步预取, 2+ stage, 计算与访存重叠 |
| GQA head 合并 | 6 Q heads 共享 1 KV head 的 SMEM tile, 减少 KV 重复读取 |
| ptx_exp2 | 替代 expf, scale 预乘 log2e |
| UMMA QK^T/PV | 如果 Level 0 验证: TC 加速矩阵乘 |

**参考**:
- FlashInfer `decode.cuh`: state_t{o,m,d} + merge, cp_async K/V 交替 pipeline, vec_t 模板
- FlashInfer `blackwell/`: SM100 FMHA warp-specialized 架构

**当前实现差距**:
- 零 pipeline, 零向量化, 无 GQA head 合并
- 标量 global load, expf 而非 ptx_exp2

**状态**: 未开始

---

### Level 4: Prefill 优化

**目标**: 降低 TTFT

**依赖**: Level 0 (UMMA/TMA), Level 2 (GDN 基础)

| 子任务 | 内容 |
|---|---|
| GDN WY Phase A | UMMA 替代标量 dot product + forward substitution (如果 Level 0 验证) |
| Attention FMHA | UMMA-based fused multi-head attention (如果 TMEM 验证可行) |
| TMA weight load | 权重 tile 用 TMA 预取 (GDN + Attention 共享) |

**状态**: 未开始

---

### Level 5: 系统级调度

**目标**: 多并发吞吐最大化

**依赖**: Level 2-4 (稳定 kernel)

| 子任务 | 内容 |
|---|---|
| Persistent kernel | Cooperative launch + dual runner, 20 SM 利用率优化 |
| GPU-side planner | 多请求负载均衡, work stealing |
| Batched decode | GEMV → GEMM 切换 (batch>1 时权重只读一次) |

**状态**: 未开始

---

## 可参考的 FlashInfer 代码索引

| 我们的模块 | FlashInfer 参考文件 | 关键技术点 |
|---|---|---|
| GDN Decode | `flashinfer/gdn_decode.py` | K-persistent registers, pretranspose layout, cp.async 2-stage |
| GDN State BF16 | `flashinfer/gdn_kernels/gdn_decode_bf16_state.py` | BF16↔FP32 转换, 128-bit async copy, 4-chunk pipeline |
| Attn Decode | `include/flashinfer/attention/decode.cuh` | vec_t, state_t{o,m,d}, cp_async K/V 交替 pipeline, GQA bdy |
| Attn Blackwell | `include/flashinfer/attention/blackwell/` | UMMA+TMA+TMEM, warp-specialized, persistent |
| GDN Prefill | `include/flashinfer/flat/hopper/` | TMA WarpSpecialized (SM90, 仅参考架构) |
| GPU Sampling | `include/flashinfer/sampling.cuh` | Gumbel-Max (已借鉴), Top-K pivot 二分, chain speculative |

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-03-05 | 创建路线图, 基于 FlashInfer 调研和硬件×模型交叉分析 |
| 2026-03-05 | Level 0 TMEM/UMMA 探测完成: TMEM 可存 GDN state (25% 容量), UMMA 1.61 TFLOPS/SM, rank-1 update 不适用 UMMA |
| 2026-03-05 | Level 0 TMA + Roofline 完成: TMA 32KB 4.3× vs manual, 跨越点 16KB; DRAM 升至 256 GB/s (94%); Roofline 确认 decode BW-bound (AI=2.0), WY compute-bound (AI=128) |
| 2026-03-05 | Level 0 统计学升级: 20 trials + 5 warmup, 全指标报告 min/P5/median/mean/P95/max/σ/CV; 新增 DRAM 写带宽 (149-177 GB/s, 读的 55-65%, 非对称!), 12 级 size sweep, multi-SM scaling (1→20 SM), memory latency 分层 (L1=25ns/L2=86-162ns/DRAM=170ns), strided access pattern |
| 2026-03-05 | Level 0 扩展至 21 probes: 新增 9 项 SM110a 特性全面测试 (Probes 12-20, 全部 CV=0.0% ★★★): UMMA TS/UTCCP/TMEM pack:unpack/f32x2/LDSM:STSM/UMMA fence/Reg Reconfig/PDL |
| 2026-03-05 | Level 1 State BF16 化完成: 15 文件修改, 144→72 MB/req, B=1 无回归, B=128 吞吐 +42.6% (264→377 tok/s) |
