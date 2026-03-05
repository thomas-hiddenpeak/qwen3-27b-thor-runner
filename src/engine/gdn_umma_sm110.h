#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gdn_umma {

// WY chunk-wise GDN prefill kernel
// Algorithm: WY factorization converts sequential B-token delta rule update
// into matrix operations: KK, QK GEMMs + IKK inverse + state GEMMs.
//
// Parameters match invoke_gated_delta_net (serial kernel):
//   q, k, v: [T, token_stride] with Q at offset 0, K at qk_dim, V at 2*qk_dim
//   a_raw, beta_raw: [T, nv] BF16 — input-dependent decay and gate
//   dt_bias: [nv] BF16 — per-head bias for decay (nullptr = 0)
//   A_log: [nv] FP32 — log decay parameter (nullptr = 0)
//   ssm_state: [nv, kd, vd] FP32 — in/out SSM state
//   y_out: [T, nv, vd] BF16 — output
//   token_stride: stride between tokens in Q/K/V (= in_qkv = 10240 in production)
void invoke_gdn_wy_prefill(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* a_raw,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    const __nv_bfloat16* beta_raw,
    __nv_bfloat16* ssm_state,
    __nv_bfloat16* y_out,
    int num_tokens,
    int nkh,
    int kd,
    int nv_per_kh,
    int vd,
    cudaStream_t stream = 0,
    int token_stride = 0,
    __nv_bfloat16* ssm_state_checkpoint = nullptr);

} // namespace gdn_umma
