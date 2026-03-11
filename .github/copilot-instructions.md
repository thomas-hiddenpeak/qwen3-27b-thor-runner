# Qwen3.5-Thor — Copilot 项目指令

## 项目概述

运行在 NVIDIA Jetson AGX Thor (SM110a Blackwell) 上的 Qwen3.5 推理引擎。支持 4B/9B/27B Dense + 35B-A3B/122B-A10B MoE 模型，BF16 + NVFP4 (W4A16) 精度。C++17 / CUDA，目标是极致性能与稳定。

## 硬件规格 (Jetson AGX Thor)

- **GPU**: Blackwell SM110a, 20 SM (10 TPC × 2), 2560 CUDA Cores, 5th-gen Tensor Cores
- **时钟**: GPC 1575 MHz, NVD 1692 MHz, Power Mode = MAXN
- **FP32 峰值**: 8.064 TFLOPS
- **内存**: 128 GB LPDDR5X 统一内存, 4266 MHz, 256-bit bus
  - 峰值带宽 273 GB/s, 实测 GEMV ~220 GB/s (80%)
  - **无独立显存**: CPU 和 GPU 共享同一物理内存
  - `cudaMalloc` 由 GPU driver 管理, 不可 CPU 访问, 不计入进程 RSS, jtop 也可能不显示
  - `cudaMallocManaged` 走 OS VM, 有 lazy page fault, 计入进程 RSS
  - 系统级可通过 tegrastats / `/proc/meminfo` MemAvailable 监控实际物理占用
- **L2 Cache**: 32 MB
- **Shared Memory**: 228 KB/SM, 48 KB/block
- **Registers**: 65536/SM (= 65536/block)
- **Threads**: 1536/SM (48 warps), 1024/block, max 24 blocks/SM
- **CPU**: 14-core Arm Neoverse V3AE @ 2601 MHz
- **功耗**: 40-130W (当前 MAXN)
- **SM110 特性**: Cluster Launch ✅, TMA ✅, Cooperative Launch ✅, ATS ✅
- 不是数据中心 GPU, 不能假设 PCIe/NVLink 或独立 HBM

### 统一内存的关键约束

权重 (~51 GB) + merged 权重 (~5 GB) + KV Cache + SSM 状态 + Workspace 全在 128 GB 物理内存中:
- 必须 `loaders_.clear()` 释放 mmap 防止双份权重 (~54 GB)
- `max_chunk_size_ = 2048` 默认值, 上限 4096 (>4096 触发 CUTLASS TMA 描述符错误 → SIGSEGV)
- 每层 forward 需 `cudaStreamSynchronize` 防止统一内存并发访问超载

## 模型架构 (Qwen3.5-27B)

- **64 层混合架构**:
  - 48 层 **Linear Attention** (Gated DeltaNet SSM) — `layer_idx % 4 != 3`
  - 16 层 **Full Attention** (GQA + Paged KV Cache) — `layer_idx % 4 == 3`
- **Hidden Size**: 5120, **Intermediate Size**: 17408, **Vocab**: 248320
- **Full Attn**: 24 Q heads, 4 KV heads, head_dim=256, RoPE partial (64/256)
- **Linear Attn**: 16 key heads, 48 value heads, key_dim=128, value_dim=128, conv_k=4
- **精度**: BF16 (权重/激活/KV cache/SSM state GMEM), FP32 (SSM state kernel 内计算, A_log)
- **Norms**: `Qwen3_5RMSNorm` 使用 centered weight `(1+w)`, 层内 attn_norm 使用 plain weight

### 官方推荐采样参数

| 模式 | temperature | top_p | top_k | min_p | presence_penalty |
|------|-------------|-------|-------|-------|------------------|
| 思考模式 — 通用 | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| 思考模式 — 编码 | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| 非思考模式 — 通用 | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| 非思考模式 — 推理 | 1.0 | 1.0 | 40 | 0.0 | 2.0 |

## 代码结构

```
src/
├── main.cpp              — 统一入口 (serve/chat/bench/test/probe/version 子命令)
├── tests.cpp             — 测试框架 (16 tests, 3 categories, --list/--filter/--category/--all)
├── benchmark.cpp         — 性能评估 (engine bench + --raw-batch 参考基线)
├── engine/
│   ├── engine.h/cpp      — 推理引擎: prefill/decode 循环, 连续批处理, MTP
│   ├── backend.h/cpp     — 独立后端接口 (线程安全, 与传输层解耦)
│   ├── model.h/cpp       — 64 层 forward, safetensors 权重加载, MTP 模块
│   ├── layer.h/cu        — Qwen35Config, FullAttn/LinearAttn 层实现
│   ├── light_ops.h/cu    — 融合算子 (RMSNorm, RoPE, SiLU, Conv1d, DeltaNet, GPU Sampling)
│   ├── dense_gemm.h      — GEMM/GEMV 接口 (BF16)
│   ├── dense_gemm_sm110.cu — CUTLASS SM110 GEMM + 散列 GEMV + Dual GEMV + GEMV+Add
│   ├── dense_gemm_fp4.h  — NVFP4 GEMM/GEMV 接口
│   ├── dense_gemm_fp4_sm110.cu — FP4 E2M1 GEMV V2 (SMEM LUT + 向量化读取)
│   ├── gdn_umma_sm110.cu/h — GDN WY 分块 Prefill Kernel
│   ├── paged_attention.h/cpp/cu — KV Cache 管理 + Paged/Split-K/Chunked Attention
│   ├── streaming_attention.h/cu — GPU+SSD 混合 Streaming Attention
│   ├── cache_config.h    — 缓存配置 + 容量规划器
│   ├── cache_engine.h/cpp — SSD 前缀缓存
│   ├── kv_swapper.h/cpp  — 请求级状态换出/换入 SSD
│   ├── allocator.h/cpp   — UnifiedAllocator (cudaMallocManaged) / DeviceAllocator (cudaMalloc)
│   ├── tokenizer.h/cpp   — BPE tokenizer
│   ├── vision.h/cu       — ViT 视觉编码器
│   ├── perf_stats.h/cpp  — CUDA 事件计时/阶段统计/利用率监控
│   ├── safetensors.h/cpp — Safetensors 零拷贝加载
│   ├── tensor.h/cpp      — Tensor 封装
│   ├── shm_queue.h       — POSIX 共享内存 SPSC 环形队列
│   ├── pdl.h             — PDL (Programmatic Dependent Launch) 宏 (SM90+)
│   ├── tma_utils.h       — TMA bulk copy helpers (cp.async.bulk, mbarrier)
│   ├── sm110a_primitives.h — SM110a 硬件特性常量
│   ├── sm110a_probe.cu   — SM110a 硬件能力探测
│   ├── deltanet_chunkwise.cu — WY chunkwise 评估原型 (独立 micro-benchmark, 不参与推理)
│   └── moe_*.h/cpp, grouped_gemm.h, cutlass_grouped_gemm_sm110.cuh — MoE 预留
├── serve/
│   └── serve.h/cpp       — HTTP API 服务 (Ollama/OpenAI 兼容)
└── tui/
    └── tui.h/cpp         — TUI 交互式 Chat 界面
```

## 已实现的核心优化

### 内存管理
- 权重/KV/SSM/Workspace: `cudaMalloc` (GPU driver 管理, 无 page fault)
- 少量 CPU 需访问的数据 (argmax result, pointer arrays): `cudaMallocManaged`
- 权重加载后释放 mmap (`loaders_.clear()`)

### GEMV/GEMM
- 散列映射 GEMV, Dual GEMV, GEMV+Add 融合
- Multi-row GEMV: M=2-8 模板, B 权重读一次, A 行从 L2 cache 复用, 零 SMEM, MTP verify 38%/层加速
- Level 2 投影合并: Init-time 权重合并 + 单 GEMV 替代多次
  - FullAttn QKV: [12288+1024+1024, 5120] → 3 GEMV→1, 16 层 × 2 = 32 launches saved
  - LinearAttn QKVZAB 超级合并: [10240+6144+48+48, 5120] → 4 GEMV→1, 48 层 × 3 = 144 launches saved
  - 合并后释放原始权重, net zero 内存; T>1 GEMM 用子指针偏移
- Fused RMSNorm+GEMV: Input RMSNorm 在 GEMV SMEM 内完成, 省 norm_out GMEM I/O + 64 launches
- GEMM Dispatch: M=1 GEMV, M=2-8 Multi-row GEMV, M=9-16 cuBLAS, M≥17 CUTLASS SM110 (M pad 到 8 对齐), can_implement() 失败自动回退 cuBLAS
- AOT cuBLAS Autotuning: engine/serve 启动时预热 7 (N,K)×3 M×20 reps (~3s), 消除首次推理延迟

### Kernel Fusion
- Fused Add+RMSNorm, Deinterleave+RMSNorm, RMSNorm+SiLU Gate
- Fused QK_norm+RoPE: deinterleave+Q_norm + K_norm + partial RoPE → 单 kernel (32 launches saved)
- Fused SwiGLU, Sigmoid-Mul, Deinterleave 3-Way Split

### DeltaNet SSM
- SSM State BF16 化: GMEM BF16 存储, kernel 内 FP32 计算 (Level 1 已完成)
- Serial prefill: SSM state 全量缓存 SMEM
- WY 分块 prefill (Phase 16): T≥4 启用, 1.71× 加速
- Conv1d prefill 全并行
- MTP checkpoint 用于 reject 回滚

### Attention
- Split-K decode paged attention (含 causal masking 支持 MTP verify T≤8)
- FullAttn small-T paged attention: T≤8 用 paged split-K 替代 chunked prefill, attention -86%
- Chunked prefill tiled GEMM attention (T>8)
- Fused prefill attention kernel (已实现, 未启用)
- SSD streaming attention (256K+)

### GPU Sampling (参考 FlashInfer)
- Gumbel-Max 快速路径 + GPU top-k/top-p/min_p/presence_penalty

### SM110a 硬件原语
- PDL (Programmatic Dependent Launch): 全量 kernel launch 转 `PDL_LAUNCH()`, launch overlap -1.8%
- f32x2 SIMD FMA: 14 个 BF16 GEMV kernel 使用 `fma.rn.f32x2` PTX, 50% fewer FMA instructions
- TMA bulk copy: SSM state GMEM↔SMEM 使用 `cp.async.bulk`, 32KB 4.31× 加速
- exp2f: FA4 启发, 全量 `expf` → `exp2f(x * LOG2E)`, softmax LOG2E 预乘

### 其他
- Batched argmax, MTP 投机解码, KV/SSM 状态 SSD offload, L2 persistence

## 关键实现陷阱 (绝对不可回退)

- RMSNorm 使用 `(1+weight)`, 除 DeltaNet attn_norm 用 plain weight
- RoPE 半旋转 `(d, d+rot_dim/2)`, **不是**交错 `(2i, 2i+1)`
- q_proj 输出 = Q + Gate, 需 deinterleave 后 Gate 做 sigmoid
- KV cache 每层独立, 布局 `[block, slot, head, dim]`
- paged_attention read 和 write_kv_cache 偏移一致
- Conv1d 操作全部 10240 通道 (Q+K+V), 不只是 Q
- CUTLASS output RowMajor, `can_implement()` 失败必须 cuBLAS 回退
- Chunked prefill chunk 1+ 用 tiled GEMM attention
- `max_chunk_size_ = 2048` 默认值, 可配置 64-4096, 不可超 4096 (CUTLASS TMA → SIGSEGV)
- `loaders_.clear()` 不可移除 (双份权重 → OOM)
- per-layer `cudaStreamSynchronize` 不可移除 (forward_decode/forward_prefill)

## 绝对禁止

- **不做进一步量化** (INT8/INT4/MX 等, NVFP4 已实现)
- **不做剪枝**
- **不引入外部 draft model** (仅用模型自带 MTP)
- **不许说"已达极致"** — 距理论峰值 273 GB/s 还有 ~20%

## 性能优化方向

### 单请求 Decode (带宽瓶颈)
- 当前 ~4.4 tok/s, ~227 GB/s (83% 峰值), 每步读 ~51 GB 权重
- 方向: DRAM bank-level 访问模式, GEMV kernel 微调

### Prefill
- WY 已加速 DeltaNet 1.71×; Fused prefill attention 可替代 28 次 launch/层
- TTFT 优化空间显著

### 多并发吞吐 (核心方向)
- batched decode GEMV→GEMM, 权重只读一次服务多 token
- Head-Group Batch Attention: 同 KV head 的 6 Q head 合并读取
- SSM State BF16 化: ✅ 已完成, 72MB/request, B=128 吞吐 +42.6%
- Batch MTP: 在 batch decode 模式下启用 MTP (T=4×B GEMM), 预估 2-3× 提升
- Batch Prefill: 不同请求的 prefill 合并为单次 forward (block-diagonal attention)
  - 可将 N 次串行 prefill (N×270ms) 压缩为 ~1-2 次 forward, ramp-up 时间降 10-30×

### 已完成优化清单
- ✅ Level 1: SSM State BF16化 (GMEM BF16, kernel FP32), B=128 +42.6%
- ✅ Level 2: FullAttn QKV merge (3→1 GEMV, 16层 ×2=32 launches)
- ✅ Level 2b: LinearAttn QKVZAB super-merge (4→1 GEMV, 48层 ×3=144 launches)
- ✅ Fused QK_norm + RoPE (3→1 kernel, 32 launches)
- ✅ Fused RMSNorm + GEMV (norm in SMEM, 64 launches, ~1ms)
- ✅ MTP Partial Accept (d=3, 逐位置 verify + SSM/Conv checkpoint), +21.6%
- ✅ Batched Argmax (verify 路径 4 sync → 1 sync), sample 37→7ms
- ✅ GPU-Resident MTP Draft Chain (3 sync → 1, pre-alloc blocks), +18.5%
- ✅ NVFP4 (W4A16) 推理支持 (FP4 E2M1 GEMV V2, SMEM LUT, 向量化读取)
- ✅ FP4 QKV/GateUp 投影合并, NVFP4 decode +17% over BF16
- ✅ 多模型支持 (27B/9B/4B, config.json 自动检测架构)
- ✅ Benchmark 重构 (Engine-based, 通过 InferenceBackend 走完整推理路径, TTFT/吞吐量/统计/JSON)
- ✅ Test 框架 (16 tests, 3 categories, --list/--filter/--category/--all)- ✔️ exp2f + LOG2E 预乘 (FA4 启发, 全量 expf→exp2f), -1.1%
- ✔️ PDL (Programmatic Dependent Launch, 10 files, ~70 launch sites), -1.8%
- ✔️ f32x2 SIMD FMA (14 BF16 GEMV kernels, fma.rn.f32x2), noise-neutral (BW-bound)
- ✔️ TMA bulk copy (SSM state cp.async.bulk, 32KB 4.31×), prefill 加速
- ✔️ Multi-row GEMV (M=2-8 register-based, L2 cache, zero SMEM), MTP 5.2→7.3-9.0 tok/s
- ✔️ FullAttn small-T paged attention + split-K causal masking, attention -86%, MTP +3.3%
- ✅ 权重加载: adaptive mmap + scalar bypass + direct-to-packed expert loading
  - 122B: 151.2→120.2s (-20.5%), 657 MB/s (+25.9%), cudaMalloc -97%
  - 27B: 36.9s/1437 MB/s, 9B: 11.7s/1576 MB/s, 4B: 6.2s/1435 MB/s
- ❌ ShardPool (阻止 release_raw → 2× peak memory OOM, reverted)
- ❌ SMEM multi-row GEMV (occupancy 6→5 blocks/SM, -24%, reverted)
- ❌ GDN SMEM caching (occupancy drop, reverted)
- ❌ Dual GEMV + SwiGLU fusion (block count halved, +4.6%, reverted)
- ✅ Prefill max_chunk_size 256→2048 (减少权重重复读取, TTFT -17%~-37%)
  - T~256: 694→438ms (-37%), T~512: 980→734ms (-25%), T~1024: 1720→1432ms (-17%)
  - CLI: --max-chunk-size, 配置: max_chunk_size=2048, 上限 4096
- ✅ CUTLASS 接管 M=9+ (M=9-16 CUTLASS 替代 GEMV<16>, B=16 吞吐 +87%)
- ✅ Batch decode routing fix (B=1 让步 prefill ramp-up, IPC 队列 8→128)
- ✅ Engine pipeline 优化:
  - Batched argmax: batch_decode_step greedy 采样 B×sync→1 sync
  - Prefill-first ramp-up: 有 pending prefill 时跳过 batch decode, 1× 权重读取
  - Bulk IPC admission: 一次 pop 所有请求, 消除队列延迟
  - Cleanup fast sync: 替代 polling+sleep, 利用 step 已 sync 特性
- ✅ Batch Prefill: B 请求合并单次 forward, Q aliasing fix
  - B=32: TTFT 3836→929ms (-75.8%), 90.6→123.8 tok/s (+36.6%)
  - B=64: TTFT ~17s→1868ms (-89.0%), 123.2→212.5 tok/s (+72.5%)
- ✅ Batched LM Head: B×GEMV → gather+RMSNorm+single GEMM
  - B=32: TTFT 929→596ms (-35.8%), 128.0 tok/s (raw 达标基线 130.1, 98.4%)
  - B=64: TTFT 1868→1201ms (-35.7%), 218.5 tok/s (raw 达标基线 232.3, 94.1%)
- ✅ SSM/Conv 指针缓存: 跳过 steady-state 下 managed memory 重建, ATS 一致性开销消除
  - B=4: 16.3→17.2 tok/s (93.7%→99.4%), B=16: 36.5→39.4 tok/s (92.9%→100.5%)
  - 所有 B=1~16 达标率 ≥99%
- ✅ cuBLAS routing M=9-16: 替代 GEMV<16>, B=16 BW 116→220 GB/s (+89%)
- ✅ AOT cuBLAS autotuning warmup: 7 (N,K)×3 M×20 reps ~3s, 消除首次推理延迟
- ✅ Benchmark: decode-only + overall throughput 双指标, concurrent benchmark decode_tps 排除 TTFT

### 并发吞吐 (27B BF16, d=30, 3 iterations)
| B | Raw tok/s | Serve Decode tok/s | Serve/Raw |
|---|-----------|--------------------|-----------|
| 1 | 4.4 | 12.1 (MTP) | 275% |
| 2 | 8.8 | 9.0 | 102% |
| 4 | 16.5 | 17.1 | 104% |
| 8 | 25.6 | 26.3 | 103% |
| 16 | 65.7 | 68.9 | 105% |
| 32 | 127.9 | 133.0 | 104% |
| 64 | 230.0 | 239.5 | 104% |
| 128 | 379.0 | 390.1 | 103% |

### 稳定性
- 统一内存 SMMU 资源有限, 大规模并发访问可致 GPU hard-reset
- 压力测试覆盖多轮、长上下文、多并发
- 内存监控: cudaMalloc 不计进程 RSS, 需 tegrastats 或 CUDA API

## 编码规范

- C++17 + CUDA, kernel 使用 `__nv_bfloat16`
- Decode T=1 GEMV, Prefill T>1 CUTLASS GEMM
- 预分配 workspace, 推理时**禁止**动态 malloc
- 单 CUDA stream, 连续批处理

## 构建

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
# 产物: build/qwen35-thor
# 运行 (推荐使用统一配置文件 configs/qwen3.5-27b.conf):
#   ./build/qwen35-thor serve --config configs/qwen3.5-27b.conf
#   ./build/qwen35-thor chat  --config configs/qwen3.5-27b.conf
#   ./build/qwen35-thor bench --config configs/qwen3.5-27b.conf --decode 30 --iterations 3 --json results.json
#   ./build/qwen35-thor bench --config configs/qwen3.5-27b.conf --raw-batch 1,32,64,128 --raw-decode 10 --iterations 3
#   ./build/qwen35-thor test --list
#   ./build/qwen35-thor test --all
# 多模型 (自动从 config.json 检测架构):
#   ./build/qwen35-thor serve --config configs/qwen3.5-4b.conf
#   ./build/qwen35-thor serve --config configs/qwen3.5-27b-nvfp4.conf
# 也可单独覆盖 serve 配置:
#   ./build/qwen35-thor serve --config configs/qwen3.5-27b.conf --serve-config configs/serve.conf
```

## 工作流规范

### Git Commit（强制）

每次取得阶段性成果时**必须** `git commit` 记录:
- **编译通过 + 测试通过**: 立即 commit
- **性能测量完成 (A/B 对比有结论)**: 立即 commit，commit message 包含关键数值
- **新优化实现**: 实现 + 验证后 commit，不要积累多个优化再一次性提交
- **失败回退**: 回退后也要 commit，注明失败原因
- **Benchmark 基线更新**: commit message 包含 ITL/Forward/BW 数值

示例 commit message 格式:
```
perf: fused RMSNorm+GEMV saves 64 launches, ITL 230→229ms (-0.5%)
revert: dual GEMV+SwiGLU fusion — block count halved, +4.6% regression
bench: prompt=17 TTFT=467ms gen=10.4 tok/s (engine-based, MTP enabled)
```

### Benchmark 基本要求

- Benchmark 通过 InferenceBackend 走完整推理路径 (含 MTP, chunked prefill, GPU 采样)
- 最少 `--decode 30 --iterations 3`, N≥3 才有统计意义
- 每次测量前 `pkill` 之前的进程, 确保 GPU 空闲
- 对比必须控制相同参数 (config, kv-cache-gb, decode steps, mtp-disable/enable)
- `docs/OPTIMIZATION_LOG.md` 记录每次优化的 A/B 结果

## 临时文件

- 调试日志、临时输出一律写到 `tmp/` 目录 (已加入 `.gitignore`)
- 例如: `./build/qwen35-thor serve ... > tmp/debug.log 2>&1`
- 不要使用 `/tmp/` 等系统目录，避免需要额外授权

## 沟通规范

- 使用中文
- 内存尺寸明确单位 (bytes / elements / BF16 count)
- kernel 维度: `[M, K] x [K, N] -> [M, N]`
- 修改 kernel 注明线程布局 (grid, block, shared memory)
- 性能改动附带理论计算 (FLOPS, 带宽, roofline)
- 每次执行程序前先 kill 之前的进程
- 调试新功能时，禁止使用 tail 过滤输出
- 在调试过程中，如果用户没有明确提示出现了热节流，不要考虑热节流。