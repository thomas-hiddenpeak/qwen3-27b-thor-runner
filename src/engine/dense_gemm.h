#pragma once

#include "tensor.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace ops {

// 基础的 Dense GEMM 算子接口
// 用于 Qwen3.5-27B (稠密模型) 的 QKV 投影和 MLP 层
class DenseGEMM {
public:
    DenseGEMM() = default;
    virtual ~DenseGEMM() = default;

    // 执行矩阵乘法: C = A * B
    // A: [M, K] (通常是 Hidden States)
    // B: [K, N] (通常是 Weight)
    // C: [M, N] (Output)
    virtual void forward(
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor& C,
        void* stream = nullptr
    ) = 0;
};

// 基于 CUTLASS 3.x 的 SM110 优化实现
class DenseGEMMCUTLASS : public DenseGEMM {
public:
    DenseGEMMCUTLASS();
    ~DenseGEMMCUTLASS() override;

    void forward(
        const core::Tensor& A,
        const core::Tensor& B,
        core::Tensor& C,
        void* stream = nullptr
    ) override;
};

// 辅助函数：直接使用裸指针调用 Dense GEMM
void invoke_dense_gemm(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream = nullptr
);

// 辅助函数：针对 M=1 的矩阵向量乘法 (GEMV)
// 用于 Decode 阶段和 LM Head
void invoke_dense_gemv(
    const __nv_bfloat16* A, // [1, K]
    const __nv_bfloat16* B, // [K, N] (Column Major)
    __nv_bfloat16* C,       // [1, N]
    int N,
    int K,
    cudaStream_t stream = nullptr
);

// GEMV + Residual Add: C[i] = (A × B)[i] + residual[i]
// 融合 down_proj GEMV 和 residual add, 消除额外的 add kernel launch + 内存写读
void invoke_dense_gemv_add(
    const __nv_bfloat16* A,        // [1, K]
    const __nv_bfloat16* B,        // [K, N] (Column Major)
    __nv_bfloat16* C,              // [1, N] output = GEMV + residual
    const __nv_bfloat16* residual, // [1, N] 要加的 residual
    int N,
    int K,
    cudaStream_t stream = nullptr
);

// Dual-output GEMV: 一次 kernel 同时计算 C1 = A × B1 和 C2 = A × B2
// 共享 A 的 shared memory 加载，节省 launch overhead + A 重复读取
// 用于 MLP 的 gate_proj + up_proj (共享 post_norm_out 输入)
void invoke_dense_dual_gemv(
    const __nv_bfloat16* A,  // [1, K] 共享输入
    const __nv_bfloat16* B1, // [K, N] (Column Major) — gate_proj
    const __nv_bfloat16* B2, // [K, N] (Column Major) — up_proj
    __nv_bfloat16* C1,       // [1, N] — gate output
    __nv_bfloat16* C2,       // [1, N] — up output
    int N,
    int K,
    cudaStream_t stream = nullptr
);

// GEMM + Residual Add: D = A × B + residual
// 融合 down_proj GEMM 和 residual add, 使用 CUTLASS beta=1 epilogue
// 消除独立的 add kernel launch + 额外的内存读写
void invoke_dense_gemm_add(
    const __nv_bfloat16* A,        // [M, K]
    const __nv_bfloat16* B,        // [K, N] (Column Major)
    __nv_bfloat16* D,              // [M, N] output = GEMM + residual (can be same as residual)
    const __nv_bfloat16* residual, // [M, N] 要加的 residual
    int M, int N, int K,
    cudaStream_t stream = nullptr
);

} // namespace ops
} // namespace qwen_thor
