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

// Fused RMSNorm + GEMV: Input RMSNorm 在 SMEM 内完成后直接开始 GEMV
// 省去 norm_out 的 GMEM write+read + 1 kernel launch
// hidden_states → SMEM load → in-SMEM RMSNorm(centered) → GEMV
// 仅用于 T=1 decode, K 必须能装入 SMEM (K*2 ≤ 48KB)
void invoke_dense_gemv_with_rmsnorm(
    const __nv_bfloat16* hidden_states,  // [1, K]
    const __nv_bfloat16* norm_weight,    // [K] centered RMSNorm weight
    float eps,
    const __nv_bfloat16* B,              // [K, N]
    __nv_bfloat16* C,                    // [1, N] output
    int N, int K,
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

// Grouped Expert GEMV: 单次 launch 计算 top_k 个 expert 的 GEMV
// shared_input=true: all experts share the same input (gate_up projection)
// shared_input=false: per-expert inputs at inputs[rank*K] (down projection)
void invoke_grouped_expert_gemv(
    const __nv_bfloat16* inputs,         // [1, K] (shared) or [top_k, K] (per-expert)
    const __nv_bfloat16* packed_weights, // [E, N, K] all experts packed contiguous
    __nv_bfloat16* outputs,              // [top_k, N]
    const int* expert_indices,           // [top_k] on device
    int N, int K, size_t expert_stride,  // stride between experts in bf16 elements
    int top_k, bool shared_input,
    cudaStream_t stream = nullptr
);

// Fused SwiGLU + Grouped Expert Down GEMV
// gate_up_outputs[top_k, 2*K] → SwiGLU in SMEM → GEMV → outputs[top_k, N]
void invoke_grouped_expert_gemv_swiglu(
    const __nv_bfloat16* gate_up_outputs,  // [top_k, 2*K]
    const __nv_bfloat16* packed_weights,   // [E, N, K] expert down weights
    __nv_bfloat16* outputs,                // [top_k, N]
    const int* expert_indices,             // [top_k] on device
    int N, int K, size_t expert_stride,
    int top_k,
    cudaStream_t stream = nullptr
);

// Fused SwiGLU + GEMV for shared expert down projection
// swiglu(gate[K], up[K]) → SMEM → GEMV with weight[N, K] → output[N]
void invoke_dense_gemv_swiglu(
    const __nv_bfloat16* gate_out,     // [K]
    const __nv_bfloat16* up_out,       // [K]
    const __nv_bfloat16* weight,       // [N, K]
    __nv_bfloat16* output,             // [N]
    int N, int K,
    cudaStream_t stream = nullptr
);

// Weighted Expert Reduce: accum[i] = sum_k(weights[k] * outputs[k*hs + i])
void invoke_weighted_expert_reduce(
    __nv_bfloat16* accum,                 // [hs] output
    const __nv_bfloat16* expert_outputs,  // [top_k, hs]
    const float* expert_weights,          // [top_k] on device
    int hs, int top_k,
    cudaStream_t stream = nullptr
);

} // namespace ops
} // namespace qwen_thor
