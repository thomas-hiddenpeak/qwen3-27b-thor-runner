# Qwen3.5-27B CUDA Inference Engine — Copilot 项目指令

## 项目概述

运行在 NVIDIA Jetson AGX Thor (SM110a Blackwell) 上的 Qwen3.5-27B 推理引擎。C++17 / CUDA，BF16 精度，目标是极致性能与稳定。

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
- 大 prefill chunk (>256 tokens) 的 MLP GEMM 同时访问 >300 MB 权重, 超过 SMMU 资源 → GPU hard-reset
- 必须 `loaders_.clear()` 释放 mmap 防止双份权重 (~54 GB)
- `max_chunk_size_ = 256` 是稳定性硬约束
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
├── main.cpp              — 统一入口 (serve/chat/bench/test 子命令)
├── tests.cpp             — 单元测试集合
├── benchmark.cpp         — 性能评估 (--warmup/--decode/--batch/--csv)
├── engine/
│   ├── engine.h/cpp      — 推理引擎: prefill/decode 循环, 连续批处理, MTP
│   ├── backend.h/cpp     — 独立后端接口 (线程安全, 与传输层解耦)
│   ├── model.h/cpp       — 64 层 forward, safetensors 权重加载, MTP 模块
│   ├── layer.h/cu        — Qwen35Config, FullAttn/LinearAttn 层实现
│   ├── light_ops.h/cu    — 融合算子 (RMSNorm, RoPE, SiLU, Conv1d, DeltaNet, GPU Sampling)
│   ├── dense_gemm.h      — GEMM/GEMV 接口
│   ├── dense_gemm_sm110.cu — CUTLASS SM110 GEMM + 散列 GEMV + Dual GEMV + GEMV+Add
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
- Level 2 投影合并: Init-time 权重合并 + 单 GEMV 替代多次
  - FullAttn QKV: [12288+1024+1024, 5120] → 3 GEMV→1, 16 层 × 2 = 32 launches saved
  - LinearAttn QKVZAB 超级合并: [10240+6144+48+48, 5120] → 4 GEMV→1, 48 层 × 3 = 144 launches saved
  - 合并后释放原始权重, net zero 内存; T>1 GEMM 用子指针偏移
- Fused RMSNorm+GEMV: Input RMSNorm 在 GEMV SMEM 内完成, 省 norm_out GMEM I/O + 64 launches
- CUTLASS SM110 GEMM, can_implement() 失败自动回退 cuBLAS

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
- Split-K decode paged attention
- Chunked prefill tiled GEMM attention
- Fused prefill attention kernel (已实现, 未启用)
- SSD streaming attention (256K+)

### GPU Sampling (参考 FlashInfer)
- Gumbel-Max 快速路径 + GPU top-k/top-p/min_p/presence_penalty

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
- `max_chunk_size_ = 256` 不可放宽 (GPU hard-reset)
- `loaders_.clear()` 不可移除 (双份权重 → OOM)
- per-layer `cudaStreamSynchronize` 不可移除 (forward_decode/forward_prefill)

## 绝对禁止

- **不做量化** (INT8/INT4/FP4/MX 等)
- **不做剪枝**
- **不引入外部 draft model** (仅用模型自带 MTP)
- **不许说"已达极致"** — 距理论峰值 273 GB/s 还有 ~20%

## 性能优化方向

### 单请求 Decode (带宽瓶颈)
- 当前 ~4.3 tok/s, ~220 GB/s (80% 峰值), 每步读 ~51 GB 权重
- 方向: DRAM bank-level 访问模式, GEMV kernel 微调

### Prefill
- WY 已加速 DeltaNet 1.71×; Fused prefill attention 可替代 28 次 launch/层
- TTFT 优化空间显著

### 多并发吞吐 (核心方向)
- batched decode GEMV→GEMM, 权重只读一次服务多 token
- Head-Group Batch Attention: 同 KV head 的 6 Q head 合并读取
- SSM State BF16 化: ✅ 已完成, 72MB/request, B=128 吞吐 +42.6%

### 已完成优化清单
- ✅ Level 1: SSM State BF16化 (GMEM BF16, kernel FP32), B=128 +42.6%
- ✅ Level 2: FullAttn QKV merge (3→1 GEMV, 16层 ×2=32 launches)
- ✅ Level 2b: LinearAttn QKVZAB super-merge (4→1 GEMV, 48层 ×3=144 launches)
- ✅ Fused QK_norm + RoPE (3→1 kernel, 32 launches)
- ✅ Fused RMSNorm + GEMV (norm in SMEM, 64 launches, ~1ms)
- ✅ MTP Partial Accept (d=3, 逐位置 verify + SSM/Conv checkpoint), +21.6%
- ✅ Batched Argmax (verify 路径 4 sync → 1 sync), sample 37→7ms
- ✅ GPU-Resident MTP Draft Chain (3 sync → 1, pre-alloc blocks), +18.5%
- ❌ GDN SMEM caching (occupancy drop, reverted)
- ❌ Dual GEMV + SwiGLU fusion (block count halved, +4.6%, reverted)

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
# 产物: build/qwen3-27b-thor
# 运行: ./build/qwen3-27b-thor serve --kv-cache-gb 8
#       ./build/qwen3-27b-thor chat  --kv-cache-gb 4
#       ./build/qwen3-27b-thor bench --decode 30
#       ./build/qwen3-27b-thor test
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
bench: B=1 baseline 229.2ms ITL / 218.3ms fwd / 223.6 GB/s
```

### Benchmark 基本要求

- 最少 `--decode 30 --warmup 5`, N≥30 才有统计意义
- 每次测量前 `pkill` 之前的进程, 确保 GPU 空闲
- 对比必须控制相同参数 (kv-cache-gb, batch, decode steps)
- `docs/OPTIMIZATION_LOG.md` 记录每次优化的 A/B 结果

## 临时文件

- 调试日志、临时输出一律写到 `tmp/` 目录 (已加入 `.gitignore`)
- 例如: `./build/qwen3-27b-thor serve ... > tmp/debug.log 2>&1`
- 不要使用 `/tmp/` 等系统目录，避免需要额外授权

## 沟通规范

- 使用中文
- 内存尺寸明确单位 (bytes / elements / BF16 count)
- kernel 维度: `[M, K] x [K, N] -> [M, N]`
- 修改 kernel 注明线程布局 (grid, block, shared memory)
- 性能改动附带理论计算 (FLOPS, 带宽, roofline)
- 每次执行程序前先 kill 之前的进程
