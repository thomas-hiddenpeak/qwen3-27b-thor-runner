# Qwen3.5-27B CUDA 推理引擎 — 构建与实现要点

> 目标平台: NVIDIA Blackwell / Thor (SM110a)  
> 精度: BF16  
> 模型: Qwen/Qwen3.5-27B (64 层混合架构)

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境要求与构建](#2-环境要求与构建)
3. [模型架构概述](#3-模型架构概述)
4. [运行方式](#4-运行方式)
5. [关键实现要点 & 踩坑记录](#5-关键实现要点--踩坑记录)
   - [5.1 RMSNorm 中心化权重](#51-rmsnorm-中心化权重)
   - [5.2 RoPE: 半旋转配对而非交错](#52-rope-半旋转配对而非交错)
   - [5.3 Full Attention 输出门控](#53-full-attention-输出门控)
   - [5.4 Q 投影输出包含 Q + Gate](#54-q-投影输出包含-q--gate)
   - [5.5 Per-head Q/K RMSNorm (centered)](#55-per-head-qk-rmsnorm-centered)
   - [5.6 Conv1d 应用于 Q+K+V 全部通道](#56-conv1d-应用于-qkv-全部通道)
   - [5.7 KV Cache 必须按层独立](#57-kv-cache-必须按层独立)
   - [5.8 KV Cache 内存布局一致性](#58-kv-cache-内存布局一致性)
   - [5.9 Paged Attention HEAD_DIM 支持](#59-paged-attention-head_dim-支持)
   - [5.10 CUTLASS GEMM 输出布局](#510-cutlass-gemm-输出布局)
   - [5.11 BF16 全链路](#511-bf16-全链路)
   - [5.12 Gated DeltaNet 线性注意力](#512-gated-deltanet-线性注意力)
6. [调试方法论](#6-调试方法论)
7. [性能说明](#7-性能说明)

---

## 1. 项目结构

```
runner/
├── CMakeLists.txt                   # 构建配置 (SM110a, C++17, CUDA 17)
├── src/
│   ├── main.cpp                     # 入口, test/server 模式切换
│   ├── core/
│   │   ├── engine.h/cpp             # 推理引擎: IPC、prefill/decode 循环
│   │   ├── model.h/cpp              # 64 层 forward 循环、权重加载
│   │   ├── layer.h                  # 模型配置 (Qwen35Config)、层类声明
│   │   ├── layer.cu                 # Full Attention 和 Linear Attention forward
│   │   ├── tensor.h/cpp             # Tensor 抽象
│   │   └── allocator.h/cpp          # GPU 内存分配器 (Unified Memory)
│   ├── ops/
│   │   ├── paged_attention.h/cpp/cu # KV Cache 管理器 + Paged Attention kernel
│   │   ├── light_ops.h/cu           # 基础算子 (RMSNorm, RoPE, SiLU, Conv1d, DeltaNet 等)
│   │   ├── dense_gemm.h             # GEMM / GEMV 接口
│   │   ├── dense_gemm_sm110.cu      # CUTLASS SM110 GEMM 实现
│   │   ├── moe_layer.h/cpp          # MoE 层 (当前模型未使用)
│   │   └── grouped_gemm.h           # Grouped GEMM 接口
│   ├── io/
│   │   └── safetensors.h            # Safetensors 零拷贝加载器
│   └── ipc/
│       └── shm_queue.h              # POSIX 共享内存 SPSC 队列
├── third_party/
│   └── cutlass/                     # CUTLASS header-only 库
├── models/Qwen/Qwen3.5-27B/        # 模型权重 (safetensors 格式)
├── test_chat.py                     # 非交互式 IPC 测试客户端
├── test_hf_generate.py              # HuggingFace 参考生成脚本
└── docs/
    └── BUILD_AND_IMPLEMENTATION.md  # 本文档
```

---

## 2. 环境要求与构建

### 2.1 依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| CUDA Toolkit | ≥ 12.x (支持 SM110a) | `nvcc` 编译器 |
| CMake | ≥ 3.24 | 构建系统 |
| GCC / G++ | 支持 C++17 | 宿主编译器 |
| CUTLASS | (submodule) | Header-only, 放在 `third_party/cutlass/` |
| Python 3.12 | (可选) | 测试客户端 |
| PyTorch + Transformers | (可选) | HuggingFace 参考对比 |

### 2.2 构建步骤

```bash
# 1. 创建构建目录
mkdir -p build && cd build

# 2. CMake 配置
cmake ..

# 3. 编译 (并行)
make -j$(nproc)

# 产物: build/test_runner
```

### 2.3 关键 CMake 配置

```cmake
# SM110a (Blackwell/Thor) — 必须使用 compute_110a
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_110a,code=sm_110a --expt-relaxed-constexpr")
```

> **注意**: 不能使用 `CMAKE_CUDA_ARCHITECTURES=110a`，因为 CMake 可能生成冲突的 `-arch` 标志。需要手动关闭 `CMAKE_CUDA_ARCHITECTURES` 然后通过 `CMAKE_CUDA_FLAGS` 指定。

---

## 3. 模型架构概述

Qwen3.5-27B 是一个 **混合架构** Transformer:

| 属性 | 值 |
|------|-----|
| 总层数 | 64 |
| Full Attention (GQA) 层 | 16 层 (layer_idx % 4 == 3: 即 3, 7, 11, ..., 63) |
| Linear Attention (Gated DeltaNet) 层 | 48 层 (其余) |
| Hidden Size | 5120 |
| Intermediate Size (MLP) | 17408 |
| Vocab Size | 248320 |
| RMSNorm eps | 1e-6 |

### Full Attention (GQA) 参数

| 属性 | 值 |
|------|-----|
| Q Heads | 24 |
| KV Heads | 4 (GQA, 每个 KV head 服务 6 个 Q head) |
| Head Dim | 256 |
| RoPE theta | 1e7 |
| RoPE 旋转维度 | 64 (partial_rotary_factor=0.25, 只旋转前 64 维) |
| q_proj 输出维度 | 12288 (= 24 × 256 × 2, 包含 Q + Gate) |

### Linear Attention (Gated DeltaNet) 参数

| 属性 | 值 |
|------|-----|
| Key Heads | 16 |
| Key Head Dim | 128 |
| Value Heads | 48 |
| Value Head Dim | 128 |
| Conv Kernel Size | 4 |
| in_proj_qkv 输出维度 | 10240 (= 2×2048 + 6144 = Q+K+V) |

### 每层共享

- `input_layernorm` (RMSNorm, centered weight)
- `post_attention_layernorm` (RMSNorm, centered weight)
- Dense MLP: gate_proj → up_proj → SiLU → down_proj → residual

---

## 4. 运行方式

### 4.1 启动服务器

```bash
cd build
./test_runner server > /tmp/server.log 2>&1 &
```

服务器通过 POSIX 共享内存 IPC 通信:
- `/dev/shm/qwen_thor_ipc` — 请求队列 (容量 128)
- `/dev/shm/qwen_thor_resp` — 响应队列 (容量 512)

### 4.2 运行测试

```bash
python3 test_chat.py "What is 2+2?" 200
```

### 4.3 清理

```bash
pkill -9 test_runner
rm -f /dev/shm/qwen_thor_ipc /dev/shm/qwen_thor_resp
```

---

## 5. 关键实现要点 & 踩坑记录

以下是在调试过程中发现的所有关键实现细节。每一条都曾导致输出错误（NaN、乱码或语义不正确），最终通过与 HuggingFace `transformers` 参考实现逐步对比修复。

---

### 5.1 RMSNorm 中心化权重

**问题**: Qwen3.5 的 `Qwen3_5RMSNorm` 使用 **centered weight** 公式:

```python
# HuggingFace 参考
output = hidden_states * (1 + self.weight)  # 注意 1 + weight
```

**错误做法**: `output = hidden_states * weight` (标准 RMSNorm)

**正确做法**: 对所有 `input_layernorm` 和 `post_attention_layernorm` 使用 `(1 + weight)` 公式。

> **例外**: 线性注意力层内部的 `attn_norm` (per-head RMSNorm on DeltaNet output) 使用 **普通** weight (不加 1)。需要区分 `Qwen3_5RMSNorm` (centered) 和 `Qwen3_5RMSNormGated` (plain)。

---

### 5.2 RoPE: 半旋转配对而非交错

**问题**: Qwen3.5 的 RoPE 使用 **half-rotation** (前半/后半) 配对，而不是交错 (0,1), (2,3), ... 配对。

```python
# 正确 (Qwen3.5): 半旋转配对
# 对于 rotary_dim=64, head_dim=256:
# 配对: (dim[0], dim[32]), (dim[1], dim[33]), ..., (dim[31], dim[63])
# dim[64..255] 不旋转 (pass through)

cos_val = cos(pos * freq[d])
sin_val = sin(pos * freq[d])
x_rot[d]              =  x[d] * cos_val - x[d + rotary_dim/2] * sin_val
x_rot[d + rotary_dim/2] =  x[d] * sin_val + x[d + rotary_dim/2] * cos_val
```

**错误做法**: 交错配对 `(x[2i], x[2i+1])` — 这是 GPT-NeoX 风格，Qwen3.5 不使用。

---

### 5.3 Full Attention 输出门控

**问题**: Qwen3.5 的 Full Attention 层在 attention 输出送入 `o_proj` 之前，会乘以一个 **sigmoid 门控**:

```python
# HuggingFace 参考
query, gate = self.q_proj(x).split([q_dim, q_dim], dim=-1)
# ... attention computation ...
attn_output = attn_output * F.sigmoid(gate)
output = self.o_proj(attn_output)
```

**关键**: `q_proj` 的输出维度是 `2 × q_dim`，前半是 Q，后半是 Gate。需要先 deinterleave 分离出 Q 和 Gate，然后 Gate 通过 sigmoid 后与 attention 输出逐元素相乘。

---

### 5.4 Q 投影输出包含 Q + Gate

**问题**: `q_proj` 的权重矩阵维度是 `[12288, 5120]` 而非 `[6144, 5120]`。输出按 **交错** 方式排列:

```
q_proj output: [T, num_heads, 2, head_dim]
             = 每个 head 内: [Q_head_dim | Gate_head_dim]
```

需要一个 **deinterleave** 操作将其分离为:
- Q: `[T, num_heads, head_dim]` → 用于 RMSNorm + RoPE + Attention
- Gate: `[T, num_heads, head_dim]` → 用于 sigmoid 门控

---

### 5.5 Per-head Q/K RMSNorm (centered)

**问题**: Full Attention 层在 RoPE 之前，对 Q 和 K 分别做 **per-head RMSNorm**:

```python
query = self.q_norm(query)  # [T, num_q, head_dim] → per-head normalize
key   = self.k_norm(key)    # [T, num_kv, head_dim] → per-head normalize
```

- 权重维度: `[head_dim]` (= 256)，所有 head 共享同一组权重
- **使用 centered weight**: `output = normalized * (1 + weight)`
- 这与层级 RMSNorm 使用 centered weight 一致

---

### 5.6 Conv1d 应用于 Q+K+V 全部通道

**问题**: 线性注意力层的 Causal Conv1d 应用于 **Q+K+V 的所有通道** (10240 维)，而非只对 Q 或 K。

```python
# HuggingFace 参考
qkv = self.in_proj(x)           # [T, 10240]
qkv = causal_conv1d(qkv, ...)   # 在 10240 个通道上全部做 conv1d + SiLU
q, k, v = qkv.split(...)
```

- Conv1d 权重维度: `[10240, 1, 4]` (depthwise, kernel_size=4)
- Conv 后紧跟 SiLU 激活
- 需要维护一个 `[10240, 3]` 的 conv state (kernel_size - 1 = 3) 用于 decode

---

### 5.7 KV Cache 必须按层独立

**问题**: 16 个 Full Attention 层必须各自拥有 **独立的** KV cache 空间。如果共享同一个 cache pool，不同层会互相覆盖已写入的 K/V 值。

**错误做法**: 所有 16 层使用同一个 `k_cache` 和 `v_cache` 指针。

**正确做法**: 分配 `num_layers × num_blocks` 的总 block 数，每层通过偏移访问自己的切片:

```cpp
// KVCacheManager 构造
KVCacheManager(num_blocks=4096, ..., num_layers=16)
// 实际分配: 16 × 4096 = 65536 blocks

// 每层获取自己的 cache 指针
const bf16* k_cache = kv_manager.get_layer_k_cache(full_attn_idx);  // 0..15
const bf16* v_cache = kv_manager.get_layer_v_cache(full_attn_idx);
```

> **内存开销**: 每层约 256MB (K + V)，16 层共约 4GB。

---

### 5.8 KV Cache 内存布局一致性

**问题**: KV cache 的 **写入** kernel 和 **读取** kernel 必须使用完全相同的内存布局。

**正确布局**: `[num_blocks, block_size, num_kv_heads, head_dim]`

```cpp
// 写入 (write_kv_cache_kernel):
int offset = block_idx * (block_size * num_kv_heads * head_dim)
           + slot_in_block * (num_kv_heads * head_dim)
           + kv_head * head_dim
           + dim_idx;

// 读取 (paged_attention_kernel): 必须使用完全相同的计算方式
int kv_offset = physical_block * (block_size * num_kv_heads * head_dim)
              + token_in_block * (num_kv_heads * head_dim)
              + kv_head_idx * head_dim
              + tid;
```

**错误做法**: 读取时使用 `[num_blocks, num_kv_heads, block_size, head_dim]` 的访问模式 — 这会导致读到错误的 K/V 值，表现为 decode 输出与 prefill 近似但逐渐发散。

---

### 5.9 Paged Attention HEAD_DIM 支持

**问题**: Qwen3.5 的 head_dim = 256，比常见模型的 128 大。Paged attention kernel 的 **线程数 = head_dim**，需要确保 kernel launch 参数支持 256 线程 (即 8 warps)。

某些硬编码 `HEAD_DIM=128` 的模板 kernel 需要扩展为支持 256。

---

### 5.10 CUTLASS GEMM 输出布局

**问题**: CUTLASS GEMM 的输出 layout 必须是 **RowMajor**。

```cpp
// 正确:
using LayoutC = cutlass::layout::RowMajor;

// 错误:
using LayoutC = cutlass::layout::ColumnMajor;  // → 输出转置，下游全部错误
```

所有矩阵 (A, B, C) 均使用 RowMajor，与 PyTorch 的默认布局一致。GEMM 计算 `C = A × B^T` (B 权重以转置形式存储)。

---

### 5.11 BF16 全链路

**问题**: 必须在整个推理链路中使用 **BF16** (bfloat16)，而非 FP16 (float16):

- 模型权重: BF16 (safetensors 中存储为 `bf16`)
- 激活值: BF16
- GEMM: BF16 输入/输出
- KV Cache: BF16
- Embedding: BF16

**例外**: 
- SSM state (Gated DeltaNet): **FP32** (累积精度需要)
- `A_log` 参数: **FP32** (模型原始存储即为 float32)
- Alpha (decay) 计算: **FP32** 中间值

如果在某些 kernel 中误用 FP16 类型或函数 (`__half` 而非 `__nv_bfloat16`)，会导致 NaN 或数值不正确。

---

### 5.12 Gated DeltaNet 线性注意力

**问题**: Gated DeltaNet 是本模型的核心创新，实现时需注意:

1. **QKV 来源**: 从 `in_proj_qkv` 的输出中按 offset 切分 (stride = in_qkv = 10240):
   - Q: `[0 .. qk_dim)` 每 token
   - K: `[qk_dim .. 2*qk_dim)` 每 token
   - V: `[2*qk_dim .. 2*qk_dim + lin_v)` 每 token

2. **Q 归一化**: Q 需要 L2 normalize，然后乘以 `1/sqrt(key_dim)`

3. **Key head 与 Value head 的映射**: 每个 key head 对应 `nv/nkh = 3` 个 value head (类似 GQA 的分组)

4. **Alpha 计算**: `alpha = exp(-softplus(a + dt_bias) * exp(A_log))` — 注意 softplus 和 exp 的嵌套

5. **Beta**: 通过 sigmoid 激活

6. **SSM 递推**: 状态矩阵 `S[kd, vpk]` (float32)，递归更新:
   ```
   S = alpha * S + beta * (k ⊗ v)
   y = Q @ S
   ```

7. **输出后处理**: RMSNorm (per-head, plain weight) → SiLU gate (与 z_proj 输出) → out_proj

---

## 6. 调试方法论

在整个调试过程中，以下方法论被证明是高效的:

### 6.1 逐层对比

编写 Python 脚本，使用 HuggingFace `transformers` 作为参考，逐层对比 hidden_states:

```python
# 在 HuggingFace 模型中注册 hook:
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(lambda m, inp, out, idx=i: 
        print(f"Layer {idx}: {out[0][0, -1, :3]}")
    )
```

在 C++ 端对应层添加 `cudaMemcpy` + `printf` 打印相同位置的值。找到 **第一个发散的层**，然后在该层内部逐步打印中间值定位问题。

### 6.2 Prefill 先行, Decode 后续

先确保 prefill (多 token 并行处理) 输出正确:
- 对比最后一个 token 的 logits
- 确认 top-1 token 与 HuggingFace 一致

再测试 decode (单 token 自回归):
- 逐 token 对比，找到第一个发散点
- 发散往往意味着 KV cache 相关问题

### 6.3 NaN 检测

在关键位置插入 NaN peek 函数:
```cpp
auto peek_nan = [](const char* tag, const bf16* buf, int n) {
    std::vector<bf16> tmp(n);
    cudaMemcpy(tmp.data(), buf, n*sizeof(bf16), D2H);
    for (auto& x : tmp)
        if (isnan(bf162float(x))) { printf("%s: NaN!\n", tag); break; }
};
```

### 6.4 单步 vs 自回归

如果 token 2 正确但 token 3 错误，问题很可能在 **KV cache 读取** 而非写入。因为 token 2 的 K/V 只在 token 3 才被读取。

---

## 7. 性能说明

当前状态 (Phase 11, 2026-02-27):

| 指标 | 值 |
|------|-----|
| Decode 速度 (B=1) | 4.58 tok/s (218ms/step, 235 GB/s 带宽) |
| Decode 速度 (B=64) | 200.8 tok/s (319ms/step, 43.84× scaling) |
| Prefill TTFT (T=17) | 268 ms |
| Prefill TTFT (T=1024) | 1143 ms |
| 主要瓶颈 (B=1) | DRAM 带宽 (86% 利用率) |
| 预分配显存 | workspace + KV cache ≈ 10GB |
| 模型权重 | ~51.2GB (BF16) |

### 已完成优化

1. ✅ **GEMV warp 协作 + 向量化**: 1570ms → 245ms (6.4×)
2. ✅ **GPU Argmax**: 3.7ms → 0.17ms (21.7×)
3. ✅ **QKV/ZAB GEMV 合并 (T=1)**: 减少 128 kernel launches
4. ✅ **算子融合**: Fused Add+RMSNorm, Deinterleave+Q_RMSNorm, Norm+SiLU+Gate, Alpha/Sigmoid 内联
5. ✅ **CUDA Graph**: Decode 路径 ~800 launches → 1 graph launch
6. ✅ **Continuous Batching**: B=1~64 多并发 decode, GEMV→GEMM 自动切换
7. ✅ **CUTLASS GEMM attention (prefill)**: O(T²) paged attention → CUTLASS GEMM (T≥256)
8. ✅ **CUTLASS per-GEMM sync 移除**: 312 GEMM × 2 sync → 0, TTFT T=17 -8.5%
9. ✅ **KV Cache DeviceAllocator**: cudaMallocManaged → cudaMalloc, 消除 page fault 延迟
10. ✅ **Dual GEMV gate+up**: MLP gate/up 共享输入向量, 1 kernel 替代 2

### 剩余优化方向

1. **DeltaNet WY chunkwise 并行**: 当前 SSM 递推完全串行, WY decomposition 可实现 chunk 并行 (预期 -334ms @T=1024)
2. **Conv1d 并行化**: T 维串行循环可并行化 (预期 -18ms @T=1024)
3. **Prefill QKV/Gate+Up GEMM 合并**: 减少 input 重复读取 (预期 -134ms @T=1024)

---

*文档版本: 2026-02-27*  
*基于 Qwen3.5-27B 推理引擎 Phase 0-11 完整优化经验总结*
