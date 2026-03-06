#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace core { struct QuantizedWeight; }
namespace ops {

// ============================================================================
// NVFP4 W4A16 GEMV/GEMM 接口
//
// 权重格式: FP4 E2M1 packed as U8 (2 values/byte) +
//           F8_E4M3 per-group-16 scale +
//           F32 weight_global_scale
//
// Dequant: W_real = fp4_value × e4m3_scale / weight_global_scale
// GEMV:    C[n] = sum_k(A[k] × W_real[n,k])       i.e. M=1
// GEMM:    C[m,n] = sum_k(A[m,k] × W_real[n,k])   i.e. M>1
//
// input_global_scale cancels in W4A16 (不用于推理).
// ============================================================================

// FP4 GEMV: C[1,N] = A[1,K] × W[N,K]
void invoke_fp4_gemv(
    const __nv_bfloat16* A,             // [1, K]
    const core::QuantizedWeight& W,     // packed[N,K/2], scale[N,K/16], N, K, global_scale
    __nv_bfloat16* C,                   // [1, N]
    cudaStream_t stream = nullptr
);

// FP4 GEMV + Residual Add: C = A × W + residual
void invoke_fp4_gemv_add(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* C,
    const __nv_bfloat16* residual,      // [1, N]
    cudaStream_t stream = nullptr
);

// FP4 Dual GEMV: C1 = A × W1, C2 = A × W2 (共享 A SMEM load)
void invoke_fp4_dual_gemv(
    const __nv_bfloat16* A,             // [1, K] 共享输入
    const core::QuantizedWeight& W1,    // gate_proj
    const core::QuantizedWeight& W2,    // up_proj
    __nv_bfloat16* C1,                  // [1, N1]
    __nv_bfloat16* C2,                  // [1, N2]
    cudaStream_t stream = nullptr
);

// FP4 GEMM via cuBLASLt: C[M,N] = A[M,K] × W[N,K]
void invoke_fp4_gemm(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* C,
    int M,
    cudaStream_t stream = nullptr
);

// FP4 GEMM + Residual Add: D = A × W + residual
void invoke_fp4_gemm_add(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* D,
    const __nv_bfloat16* residual,      // [M, N]
    int M,
    cudaStream_t stream = nullptr
);

// cuBLASLt FP4 handle 初始化/清理 (在 model 加载后调用一次)
void init_fp4_cublaslt();
void cleanup_fp4_cublaslt();

} // namespace ops
} // namespace qwen_thor
