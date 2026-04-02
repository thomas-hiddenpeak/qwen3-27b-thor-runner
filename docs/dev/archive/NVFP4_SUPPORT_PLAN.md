# NVFP4 量化模型兼容性支持计划

## 目标

在当前 BF16 runner 基础上，以 **additive** 方式兼容 Sehyo-Qwen3.5-27B-NVFP4 量化模型。
同一个二进制，通过 `model_dir` 配置自动检测量化格式，BF16 路径零改动。

## 硬件与格式匹配

| 项目 | 值 |
|------|-----|
| 硬件 | Jetson AGX Thor SM110a, CUDA 13.0 |
| 模型格式 | `nvfp4-pack-quantized` (compressed-tensors) |
| 权重数据 | FP4 E2M1, 每 U8 字节打包 2 个值 |
| 权重 scale | F8_E4M3, per-group (group_size=16) |
| 全局 scale | F32 标量 × 2 (weight_global_scale + input_global_scale) |
| cuBLASLt 匹配 | `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` ← 精确匹配 |
| CUTLASS blockscaled | 要求 UE8M0 scale → **不兼容**, 不用 |

## 量化覆盖范围

| 模块 | 层数 | 量化? | BF16 大小 | NVFP4 大小 |
|------|------|-------|----------|-----------|
| MLP (gate/up/down_proj) | 64 | **是** | 31.9 GB | 9.0 GB |
| Self-Attn (q/k/v/o_proj) | 16 | **是** | 4.1 GB | 1.1 GB |
| Linear Attn (in_proj_qkv/z/a/b, out_proj) | 48 | 否 | ~10.4 GB | 10.4 GB |
| Embeddings + LM head | 1 | 否 | 4.7 GB | 4.7 GB |
| MTP 模块 | 1 | 否 | ~0.8 GB | 0.8 GB |
| **总计** | | | **~51 GB** | **~25.8 GB** |

## 实施阶段

### Phase 0 — 基础设施扩展

#### 0.1 allocator.h: DataType 加 U8
- `DataType::U8`, `get_dtype_size(U8) = 1`
- 不改现有枚举项

#### 0.2 safetensors.cpp: parse_dtype 加 U8 映射
- `"U8" → DataType::U8`
- 已有 F8_E4M3, F4 映射可用

#### 0.3 tensor.cpp: 新增构造函数重载
- `Tensor(shape, dtype, data_ptr, explicit_nbytes)` — 用于 U8/FP4
- 不改已有构造函数

### Phase 1 — 量化权重结构

#### 1.1 layer.h: QuantizedWeight 结构体
```cpp
struct QuantizedWeight {
    uint8_t* packed = nullptr;   // [N, K/2] FP4 packed
    uint8_t* scale = nullptr;    // [N, K/16] F8_E4M3
    float global_scale = 1.0f;
    float input_scale = 1.0f;
    int N = 0, K = 0;
};
```

#### 1.2 FullAttnLayer / LinearAttnLayer 扩展
- 新增 `quantized_` 标志 + `QuantizedWeight` 成员
- 新增 `set_quantized_weights()` / `set_quantized_mlp()` 方法
- 现有成员和方法不改

### Phase 2 — 权重加载 (model.cpp)

- 自动检测: 扫描 tensor name 含 `weight_packed` → NVFP4 模式
- 新增 `raw_tensor_map` 存储 U8/F8 tensor（不影响 `tensor_map<bf16*>`）
- NVFP4 权重绑定: 按 4-tuple (packed/scale/global/input) 组装 QuantizedWeight
- NVFP4 模式不合并 QKV / Gate+Up（global_scale 不同）
- Linear Attn 投影是 BF16，继续 super-merge 不改
- MTP / Vision 是 BF16，走原路径不改

### Phase 3 — FP4 计算核心 (新文件)

#### 3.1 dense_gemm_fp4.h — 接口声明
```
invoke_fp4_gemv()
invoke_fp4_gemv_add()
invoke_fp4_dual_gemv()
invoke_fp4_gemv_with_rmsnorm()
invoke_fp4_gemm()          // via cuBLASLt
invoke_fp4_gemm_add()
```

#### 3.2 dense_gemm_fp4_sm110.cu — 实现
- FP4 GEMV (W4A16): 读 packed U8 → 解包 FP4 → ×F8 scale → ×global_scale → FMA with BF16 activation
- FP4 GEMM: cuBLASLt `CUDA_R_4F_E2M1` + `VEC16_UE4M3` scale 原生路径
- 融合变体: dual_gemv, gemv_add, gemv_with_rmsnorm

### Phase 4 — Forward 分发 (layer.cu)

- 每个 GEMV/GEMM 调用点加 `if (quantized_)` 分支
- BF16 分支不变, FP4 分支调新接口
- 不影响: RoPE, attention, conv1d, DeltaNet, sampling, KV cache

### Phase 5 — 构建系统

- CMakeLists.txt 添加新 .cu 文件
- 链接 cublasLt

### Phase 6 — 验证

- 精度: HF 参考 logits 对比
- 性能: bench --decode 30 对比 BF16 vs NVFP4

## 预期收益

| 指标 | BF16 | NVFP4 | 提升 |
|------|------|-------|------|
| 权重内存 | 51 GB | 25.8 GB | 释放 25 GB |
| Decode 权重读取/步 | ~51 GB | ~26 GB | ~2× 带宽减少 |
| Decode 速度 (预估) | ~4.3 tok/s | ~7-8 tok/s | ~1.7× |
| 多并发容量 | 有限 | 大幅提升 | +25 GB headroom |

## 新增文件

| 文件 | 用途 |
|------|------|
| `src/engine/dense_gemm_fp4.h` | FP4 接口声明 |
| `src/engine/dense_gemm_fp4_sm110.cu` | FP4 GEMV kernel + cuBLASLt GEMM |

## 修改文件 (additive only)

| 文件 | 改动 |
|------|------|
| `src/engine/allocator.h` | +U8 枚举 |
| `src/engine/safetensors.cpp` | +U8 映射 |
| `src/engine/tensor.h/cpp` | +构造函数重载 |
| `src/engine/layer.h` | +QuantizedWeight 结构体 + 成员 |
| `src/engine/layer.cu` | +if(quantized_) 分支 |
| `src/engine/model.cpp` | +NVFP4 检测/加载分支 |
| `CMakeLists.txt` | +新 .cu 编译 |

## 不受影响 (零改动)

light_ops, paged_attention, streaming_attention, gdn_umma_sm110,
engine, backend, serve, tui, sampling, tokenizer, vision, cache_engine, kv_swapper
