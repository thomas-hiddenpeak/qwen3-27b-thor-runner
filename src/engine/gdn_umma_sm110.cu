// =============================================================================
// GDN UMMA SM110 — GatedDeltaNet Chunk-wise Prefill with WY Factorization
// =============================================================================
//
// Phase 1: Scalar WY with SMEM-cached Q/K for correctness + baseline perf.
// Phase 2 (TODO): Replace scalar GEMMs with UMMA SS/TS + TMEM persistence.
//
// Optimizations vs initial prototype:
//   - K_hat[B, kd_pad] and Q_hat[B, kd_pad] fully cached in SMEM
//   - Alpha/beta computation parallelized across threads
//   - Pre-computed nv_decayed avoids redundant exp2f in state update
//   - Bank-conflict-free SMEM layout (kd_pad=129, vd_pad=129)
//
// Qwen3.5-27B DeltaNet: nkh=16, nv=48, kd=128, vd=128, nv_per_kh=3
//

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// CUTLASS CuTe includes (for UMMA type definitions — Phase 2 readiness)
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

using namespace cute;

namespace gdn_umma {

// =============================================================================
// Constants
// =============================================================================
static constexpr int KD = 128;
static constexpr int VD = 128;
static constexpr int CHUNK_SIZE = 8;
static constexpr int NUM_THREADS = 128;  // 1 thread per vd element

// =============================================================================
// UMMA Atom Type Definitions (Phase 2 readiness — validates compilation)
// =============================================================================
// SS mode: SMEM x SMEM -> TMEM
using UmmaAtom_SS_64x64 = SM100_MMA_F16BF16_SS<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    64, 64, UMMA::Major::K, UMMA::Major::K>;

// TS mode: TMEM x SMEM -> TMEM
using UmmaAtom_TS_128x64 = SM100_MMA_F16BF16_TS<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    128, 64, UMMA::Major::K, UMMA::Major::K>;

using UmmaAtom_TS_128x128 = SM100_MMA_F16BF16_TS<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float,
    128, 128, UMMA::Major::K, UMMA::Major::K>;

// TiledMMA types (to be used in Phase 2)
using TiledMma_SS_64x64   = decltype(make_tiled_mma(UmmaAtom_SS_64x64{}));
using TiledMma_TS_128x64  = decltype(make_tiled_mma(UmmaAtom_TS_128x64{}));
using TiledMma_TS_128x128 = decltype(make_tiled_mma(UmmaAtom_TS_128x128{}));

// SMEM layout atoms for UMMA (Phase 2 readiness)
using SmemLayoutAtomK = GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;

// =============================================================================
// Phase B Device Function Template — compile-time CS for full unroll
// =============================================================================
// Template parameter CS is the chunk size (compile-time constant).
// Called via switch(B) to ensure all loops are fully unrollable.
//
template <int CS>
__device__ __forceinline__ void gdn_wy_phase_b(
    float* __restrict__ S_smem,
    const float* __restrict__ K_hat,
    const float* __restrict__ Q_hat,
    const float* __restrict__ alpha_cp,
    const float* __restrict__ alpha_cl,
    const float* __restrict__ ikk_t,
    const float* __restrict__ qk_sc,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ y_out,
    int t_start, int j, int kd, int vd,
    int vd_pad, int kd_pad, int token_stride,
    int h_v, int nv)
{
    // B1+B2: O_inter = alpha_cp * S^T @ Q_hat,  SK = S^T @ K_hat
    // Loop swap: kd outer, B inner. S[i,j] read ONCE per i.
    float o_arr[CS];
    float sk_arr[CS];
    #pragma unroll
    for (int b = 0; b < CS; b++) {
        o_arr[b] = 0.f;
        sk_arr[b] = 0.f;
    }
    for (int i = 0; i < kd; i++) {
        float s_ij = S_smem[i * vd_pad + j];
        #pragma unroll
        for (int b = 0; b < CS; b++) {
            o_arr[b]  += s_ij * Q_hat[b * kd_pad + i];
            sk_arr[b] += s_ij * K_hat[b * kd_pad + i];
        }
    }
    #pragma unroll
    for (int b = 0; b < CS; b++)
        o_arr[b] *= alpha_cp[b];

    // B3: V_corr = V - alpha_cp * SK
    float vc_arr[CS];
    #pragma unroll
    for (int b = 0; b < CS; b++) {
        int v_idx = (t_start + b) * token_stride + h_v * vd + j;
        vc_arr[b] = __bfloat162float(v[v_idx]) - alpha_cp[b] * sk_arr[b];
    }

    // B4: NewV = V_corr @ T^T  (T is lower-triangular, stride=CHUNK_SIZE)
    float nv_arr[CS];
    #pragma unroll
    for (int b = 0; b < CS; b++) {
        float sum = 0.f;
        for (int t = 0; t <= b; t++)
            sum += vc_arr[t] * ikk_t[b * CHUNK_SIZE + t];
        nv_arr[b] = sum;
    }

    // B5: O_intra = NewV @ QK^T  (QK is lower-triangular, stride=CHUNK_SIZE)
    #pragma unroll
    for (int b = 0; b < CS; b++) {
        float o_intra = 0.f;
        for (int t = 0; t <= b; t++)
            o_intra += nv_arr[t] * qk_sc[b * CHUNK_SIZE + t];
        o_arr[b] += o_intra;
    }

    // B6: Write output
    #pragma unroll
    for (int b = 0; b < CS; b++)
        y_out[((t_start + b) * nv + h_v) * vd + j] =
            __float2bfloat16(o_arr[b]);

    // B7+B8+B9: Fused state decay + update (single kd loop)
    float nv_d[CS];
    #pragma unroll
    for (int b = 0; b < CS; b++)
        nv_d[b] = nv_arr[b] * exp2f(alpha_cl[CS - 1] - alpha_cl[b]);

    float block_decay = alpha_cp[CS - 1];
    for (int i = 0; i < kd; i++) {
        float s_upd = 0.f;
        #pragma unroll
        for (int b = 0; b < CS; b++)
            s_upd += nv_d[b] * K_hat[b * kd_pad + i];
        S_smem[i * vd_pad + j] = S_smem[i * vd_pad + j] * block_decay + s_upd;
    }
}

// =============================================================================
// WY Chunk-wise GDN Prefill Kernel (SMEM-cached Q/K)
// =============================================================================
// Grid: 1 block per value head (48 blocks)
// Block: 128 threads (1 per vd element)
// SMEM: S[kd, vd_pad] + K_hat[CS, kd_pad] + Q_hat[CS, kd_pad]
//       + IKK[CS,CS] + QK[CS,CS] + scalars  (~162 KB, 1 block/SM)
//
// Key optimization: Q_hat/K_hat pre-loaded and L2-normalized in SMEM.
// KK, QK, O_inter, SK, state_update all read from SMEM — zero global reads
// in hot inner loops.
//
__global__ void __launch_bounds__(NUM_THREADS, 2)
gdn_wy_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ a_raw,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ beta_raw,
    __nv_bfloat16* __restrict__ ssm_state,
    __nv_bfloat16* __restrict__ y_out,
    int num_tokens, int kd, int nv_per_kh, int vd,
    int token_stride, int nkh_x_nvpkh,
    __nv_bfloat16* __restrict__ ssm_state_checkpoint)
{
    const int h_v = blockIdx.x;
    const int h_k = h_v / nv_per_kh;
    const int nv  = nkh_x_nvpkh;
    const int j   = threadIdx.x;  // [0, 128)

    extern __shared__ float smem[];
    const int vd_pad = vd + 1;   // 129 — bank-conflict-free for S
    const int kd_pad = kd + 1;   // 129 — bank-conflict-free for K_hat/Q_hat

    // SMEM layout (total ~162 KB, fits 228 KB/SM with 1 block)
    float* S_smem   = smem;                                // [kd, vd_pad]  16512 floats
    float* K_hat    = S_smem + kd * vd_pad;                // [CS, kd_pad]   8256 floats
    float* Q_hat    = K_hat + CHUNK_SIZE * kd_pad;         // [CS, kd_pad]   8256 floats
    float* alpha_cl = Q_hat + CHUNK_SIZE * kd_pad;         // [CS]             64
    float* alpha_cp = alpha_cl + CHUNK_SIZE;                // [CS]             64
    float* beta_v   = alpha_cp + CHUNK_SIZE;                // [CS]             64
    float* scratch  = beta_v + CHUNK_SIZE;                  // [CS] temp        64
    float* ikk_t    = scratch + CHUNK_SIZE;                 // [CS, CS]       4096
    float* qk_sc    = ikk_t + CHUNK_SIZE * CHUNK_SIZE;     // [CS, CS]       4096

    float q_scale = rsqrtf((float)kd);
    int ss_base = h_v * kd * vd;

    // Load initial state S[kd, vd] -> S_smem[kd, vd_pad] (BF16 GMEM -> FP32 SMEM)
    for (int i = 0; i < kd; i++)
        S_smem[i * vd_pad + j] = __bfloat162float(ssm_state[ss_base + i * vd + j]);
    __syncthreads();

    int num_chunks = (num_tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int t_start = chunk * CHUNK_SIZE;
        int t_end   = min(t_start + CHUNK_SIZE, num_tokens);
        int B       = t_end - t_start;
        if (B <= 0) break;

        // =============================================================
        // Step 0: Parallel alpha/beta computation
        // =============================================================
        // Each of B threads computes one token's softplus(a) and sigmoid(beta)
        if (j < B) {
            int tidx   = (t_start + j) * nv + h_v;
            float a_val = __bfloat162float(a_raw[tidx]);
            float bias  = dt_bias ? __bfloat162float(dt_bias[h_v]) : 0.f;
            float a_l   = A_log   ? A_log[h_v] : 0.f;
            float ab    = a_val + bias;
            float dt_v  = (ab > 20.f) ? ab : log1pf(expf(ab));
            float alpha = expf(-dt_v * expf(a_l));
            alpha_cl[j] = log2f(fmaxf(alpha, 1e-10f));
            beta_v[j]   = 1.0f / (1.0f + expf(-__bfloat162float(beta_raw[tidx])));
        }
        __syncthreads();

        // Sequential prefix sum for cumulative log2-alpha (thread 0)
        if (j == 0) {
            float cumlog = 0.f;
            for (int t = 0; t < B; t++) {
                cumlog += alpha_cl[t];
                alpha_cl[t] = cumlog;
                alpha_cp[t] = exp2f(cumlog);
            }
        }
        __syncthreads();

        // =============================================================
        // Step 1: Load & L2-normalize K -> K_hat[B, kd_pad] in SMEM
        // =============================================================
        for (int t = 0; t < B; t++) {
            K_hat[t * kd_pad + j] = __bfloat162float(
                k[(t_start + t) * token_stride + h_k * kd + j]);
        }
        __syncthreads();

        // Compute norms (1 thread per token, B <= 64 <= 128 threads)
        if (j < B) {
            float sum = 0.f;
            for (int d = 0; d < kd; d++) {
                float val = K_hat[j * kd_pad + d];
                sum += val * val;
            }
            scratch[j] = rsqrtf(sum + 1e-6f);
        }
        __syncthreads();

        for (int t = 0; t < B; t++)
            K_hat[t * kd_pad + j] *= scratch[t];
        __syncthreads();

        // =============================================================
        // Step 2: Load & L2-normalize Q -> Q_hat[B, kd_pad] (with q_scale)
        // =============================================================
        for (int t = 0; t < B; t++) {
            Q_hat[t * kd_pad + j] = __bfloat162float(
                q[(t_start + t) * token_stride + h_k * kd + j]);
        }
        __syncthreads();

        if (j < B) {
            float sum = 0.f;
            for (int d = 0; d < kd; d++) {
                float val = Q_hat[j * kd_pad + d];
                sum += val * val;
            }
            scratch[j] = rsqrtf(sum + 1e-6f) * q_scale;
        }
        __syncthreads();

        for (int t = 0; t < B; t++)
            Q_hat[t * kd_pad + j] *= scratch[t];
        __syncthreads();

        // =============================================================
        // Phase A: Fused KK+QK, IKK inverse — ALL from SMEM
        // =============================================================
        // Fused KK+QK: K_hat[t,d] read ONCE, used for both KK and QK.
        // Saves kd reads per lower-tri entry + 1 syncthreads vs separate.
        // Note: stride = CHUNK_SIZE (compile-time constant) for ikk_t/qk_sc,
        //       matching the template Phase B which uses the same stride.

        // A1: Initialize + fused KK+QK
        // Zero the full CS×CS buffer first for clean partial-chunk handling
        for (int idx = j; idx < CHUNK_SIZE * CHUNK_SIZE; idx += NUM_THREADS) {
            ikk_t[idx] = 0.f;
            qk_sc[idx] = 0.f;
        }
        __syncthreads();

        for (int idx = j; idx < B * B; idx += NUM_THREADS) {
            int s = idx / B;
            int t = idx % B;
            if (s < t) continue;
            float kk_dot = 0.f, qk_dot = 0.f;
            for (int d = 0; d < kd; d++) {
                float kt = K_hat[t * kd_pad + d];
                kk_dot += K_hat[s * kd_pad + d] * kt;
                qk_dot += Q_hat[s * kd_pad + d] * kt;
            }
            float gamma_st = alpha_cl[s] - alpha_cl[t];
            float decay = exp2f(gamma_st);
            ikk_t[s * CHUNK_SIZE + t] = (s == t) ? 1.f : beta_v[s] * decay * kk_dot;
            qk_sc[s * CHUNK_SIZE + t] = decay * qk_dot;
        }
        __syncthreads();

        // A2: T = inv(IKK) * diag(beta)  — forward substitution (thread 0)
        if (j == 0) {
            for (int col = 0; col < B; col++) {
                for (int row = col + 1; row < B; row++) {
                    float sum = 0.f;
                    for (int kk = col; kk < row; kk++)
                        sum += ikk_t[row * CHUNK_SIZE + kk] * ikk_t[kk * CHUNK_SIZE + col];
                    ikk_t[row * CHUNK_SIZE + col] = -sum;
                }
            }
            for (int col = 0; col < B; col++)
                for (int row = 0; row < B; row++)
                    ikk_t[row * CHUNK_SIZE + col] *= beta_v[col];
        }
        __syncthreads();

        // =============================================================
        // Phase B: State operations — dispatched via template for
        // compile-time loop unrolling. Switch on B selects CS == B.
        // =============================================================
        if (j < vd) {
            switch (B) {
            case 8: gdn_wy_phase_b<8>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 7: gdn_wy_phase_b<7>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 6: gdn_wy_phase_b<6>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 5: gdn_wy_phase_b<5>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 4: gdn_wy_phase_b<4>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 3: gdn_wy_phase_b<3>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 2: gdn_wy_phase_b<2>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            case 1: gdn_wy_phase_b<1>(S_smem, K_hat, Q_hat, alpha_cp, alpha_cl,
                        ikk_t, qk_sc, v, y_out, t_start, j, kd, vd,
                        vd_pad, kd_pad, token_stride, h_v, nv); break;
            }
        }
        __syncthreads();

        // MTP speculative checkpoint: save SSM state after first chunk
        // for T=2 verify rollback (matches serial kernel behavior)
        if (ssm_state_checkpoint && chunk == 0 && num_tokens > CHUNK_SIZE) {
            for (int i = 0; i < kd; i++)
                ssm_state_checkpoint[ss_base + i * vd + j] = __float2bfloat16(S_smem[i * vd_pad + j]);
            // No __syncthreads needed — checkpoint is per-thread write, next chunk
            // iteration starts with __syncthreads at Step 0
        }

    } // end chunk loop

    // Write final state (FP32 SMEM -> BF16 GMEM)
    for (int i = 0; i < kd; i++)
        ssm_state[ss_base + i * vd + j] = __float2bfloat16(S_smem[i * vd_pad + j]);
}

// =============================================================================
// Host Launch Function
// =============================================================================
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
    cudaStream_t stream,
    int token_stride,
    __nv_bfloat16* ssm_state_checkpoint)
{
    if (token_stride <= 0) token_stride = nkh * kd;
    int nv = nkh * nv_per_kh;
    int grid = nv;
    int threads = std::min(vd, NUM_THREADS);
    const int vd_pad = vd + 1;
    const int kd_pad = kd + 1;

    // SMEM: S[kd, vd_pad] + K_hat[CS, kd_pad] + Q_hat[CS, kd_pad]
    //       + alpha_cl + alpha_cp + beta_v + scratch (4 * CS)
    //       + ikk_t[CS, CS] + qk_sc[CS, CS]
    size_t smem_floats = (size_t)(kd * vd_pad)
                       + 2 * CHUNK_SIZE * kd_pad
                       + 4 * CHUNK_SIZE
                       + 2 * CHUNK_SIZE * CHUNK_SIZE;
    size_t smem_bytes = smem_floats * sizeof(float);

    cudaError_t err = cudaFuncSetAttribute(
        gdn_wy_prefill_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GDN-WY] cudaFuncSetAttribute failed: %s (need %zu bytes)\n",
                cudaGetErrorString(err), smem_bytes);
        return;
    }

    gdn_wy_prefill_kernel<<<grid, threads, smem_bytes, stream>>>(
        q, k, v, a_raw, dt_bias, A_log, beta_raw,
        ssm_state, y_out,
        num_tokens, kd, nv_per_kh, vd, token_stride, nv,
        ssm_state_checkpoint);
}

} // namespace gdn_umma
