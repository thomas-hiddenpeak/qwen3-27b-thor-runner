#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

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

// ============================================================================
// FP4 Grouped Expert GEMV — MoE expert 路径
// 权重布局: 所有 expert 的行连续打包
//   packed: [E * N_per_expert, K/2]   (contiguous, expert e at row offset e*N)
//   scale:  [E * N_per_expert, K/16]  (contiguous, expert e at row offset e*N)
// ============================================================================

// FP4 Grouped Expert Gate+Up GEMV:
// 对 T*top_k 个 assignment, 用 expert_indices 选择 expert
// shared_input=true: post_norm[T, K] → 每个 token 共享输入
// 输出: outputs[T*top_k, 2*moe_is]
void invoke_fp4_grouped_expert_gemv(
    const __nv_bfloat16* inputs,         // [T, K] (shared) or [T*top_k, K]
    const uint8_t* packed_weights,       // [E * N, K/2] all experts
    const uint8_t* packed_scales,        // [E * N, K/16]
    const float* inv_global_scales,      // [E] per-expert 1/global_scale
    __nv_bfloat16* outputs,              // [T*top_k, N]
    const int* expert_indices,           // [T*top_k] on device
    int N, int K,                        // N = per-expert output dim, K = input dim
    int top_k, bool shared_input,
    cudaStream_t stream = nullptr,
    int num_tokens = 1
);

// FP4 Grouped Expert SwiGLU + Down GEMV:
// gate_up[T*top_k, 2*K_down] → SwiGLU → GEMV with down weights → outputs[T*top_k, N]
void invoke_fp4_grouped_expert_gemv_swiglu(
    const __nv_bfloat16* gate_up_outputs,  // [T*top_k, 2*K_down]
    const uint8_t* packed_weights,         // [E * N, K_down/2]
    const uint8_t* packed_scales,          // [E * N, K_down/16]
    const float* inv_global_scales,        // [E] per-expert 1/global_scale
    __nv_bfloat16* outputs,                // [T*top_k, N]
    const int* expert_indices,             // [T*top_k]
    int N, int K,                          // N = hs, K = moe_is
    int top_k,
    cudaStream_t stream = nullptr,
    int num_tokens = 1
);

// FP4 Dual GEMV for shared expert: gate + up, shared input A
void invoke_fp4_dual_gemv_shared_expert(
    const __nv_bfloat16* A,              // [1, K]
    const core::QuantizedWeight& gate_qw,
    const core::QuantizedWeight& up_qw,
    __nv_bfloat16* C_gate,               // [1, N]
    __nv_bfloat16* C_up,                 // [1, N]
    cudaStream_t stream = nullptr
);

} // namespace ops
} // namespace qwen_thor
