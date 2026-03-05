// ============================================================================
// DeltaNet WY Chunkwise Parallel Kernel — Evaluation Prototype
//
// 算法: 基于 fla/FlashInfer 的 WY representation chunkwise decomposition
// 目标: 替代 gated_delta_net_prefill_kernel 的 O(T) 串行 token loop
//       使用 O(T/C) 个 chunk, 每 chunk 内 C 个 token 通过矩阵运算并行处理
//
// 关键变换:
//   KS[t,j] = dot(K[t,:], S[:,j])  → V'[t,j] = V[t,j] - exp(γ[t]) * KS[t,j]
//   new_v = T_mat @ V'  (消除显式 w 缓冲区)
//
// Shared Memory 布局 (~177 KB, 228 KB/SM available):
//   S_local [kd, vd_pad]  = 128×129×4 = 64.5 KB
//   chunk_K [C,  kd]      = 64×128×4  = 32  KB
//   chunk_Q [C,  kd]      = 64×128×4  = 32  KB
//   chunk_V [C,  vd]      = 64×128×4  = 32  KB  (重用: V → V' → new_v)
//   mat     [C,  C]       = 64× 64×4  = 16  KB  (重用: IKK → T_mat → QK_mask)
//   scalars [C, 3]        = 64× 3×4   = 0.75 KB (alpha, beta, gamma_cum)
//   reduce_tmp[2]         = 8 B
//   Total ≈ 177 KB
//
// Grid: [nkh * nv_per_kh]  (= 48 blocks, one per value head)
// Block: [128]  (one thread per vd element)
//
// 参考:
//   - flashinfer/tests/gdn/reference_delta_rule.py: blockwise_delta_rule()
//   - fla/ops/gated_delta_rule/wy_fast.py
//   - DeltaNet 论文 Appendix B
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <chrono>

namespace deltanet_eval {

// ============================================================================
// 辅助: warp/block 归约
// ============================================================================
__device__ __forceinline__ float warpReduceSum_cw(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum_cw(float val) {
    __shared__ float warp_sums[4]; // 128 threads = 4 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warpReduceSum_cw(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    val = (threadIdx.x < 4) ? warp_sums[threadIdx.x] : 0.f;
    if (wid == 0) val = warpReduceSum_cw(val);
    return val;
}

// ============================================================================
// WY Chunkwise DeltaNet Kernel (Optimized Prototype)
// ============================================================================

// Phase timing macros (block 0, thread 0 only, zero overhead otherwise)
#define PHASE_TIMER_START() do { if (_do_pt) _ts = clock64(); } while(0)
#define PHASE_TIMER_MARK(n) do { if (_do_pt) { long long _now = clock64(); _pt[n] += _now - _ts; _ts = _now; } } while(0)

__global__ void __launch_bounds__(128, 1)
gated_delta_net_chunkwise_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ a_raw,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ beta_raw,
    float* __restrict__ ssm_state,       // [nv, kd, vd]
    __nv_bfloat16* __restrict__ y_out,   // [T, nv, vd]
    int num_tokens,
    int kd,           // 128
    int nv_per_kh,    // 3
    int vd,           // 128
    int token_stride,
    int nkh_x_nvpkh,  // 48
    int chunk_size_param,                // C: 16, 32, or 64
    long long* __restrict__ phase_times) // 7 slots (nullable)
{
    const int h_v = blockIdx.x;
    const int h_k = h_v / nv_per_kh;
    const int nv  = nkh_x_nvpkh;
    const int j   = threadIdx.x;  // vd 维度索引 [0, 127]

    const int vd_pad = vd + 1;  // 129, 减少 bank conflict
    const int kd_pad = kd + 1;  // 129, 消除 chunk_K/Q 的 N-way bank conflict
    const int C = chunk_size_param;

    // Phase profiling (block 0, thread 0 only — zero cost for other blocks)
    long long _pt[7] = {0,0,0,0,0,0,0};
    const bool _do_pt = (phase_times != nullptr && blockIdx.x == 0 && j == 0);
    long long _ts = 0;

    // ---- Shared Memory 布局 ----
    extern __shared__ float smem[];
    float* S_local   = smem;                              // [kd, vd_pad] = 66048 B
    float* chunk_K   = S_local + kd * vd_pad;             // [C, kd_pad]  = 33024 B  ← padded!
    float* chunk_Q   = chunk_K + C * kd_pad;              // [C, kd_pad]  = 33024 B  ← padded!
    float* chunk_V   = chunk_Q + C * kd_pad;              // [C, vd]      = 32768 B
    float* mat       = chunk_V + C * vd;                  // [C, C]       = 16384 B
    float* alpha_arr = mat + C * C;                       // [C]
    float* beta_arr  = alpha_arr + C;                     // [C]
    float* gamma_cum = beta_arr + C;                      // [C]
    float* reduce_tmp = gamma_cum + C;                    // [2]

    const float q_scale = rsqrtf((float)kd);
    const int ss_base = h_v * kd * vd;

    // ---- 加载初始 SSM state 到 smem ----
    for (int i = 0; i < kd; i++)
        S_local[i * vd_pad + j] = ssm_state[ss_base + i * vd + j];
    __syncthreads();

    // ---- Chunk 循环 ----
    const int num_chunks = (num_tokens + C - 1) / C;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        PHASE_TIMER_START();
        const int chunk_start = chunk * C;
        const int chunk_len = min(C, num_tokens - chunk_start);

        // ================================================================
        // Phase 0: 加载 K, Q, V, 计算 α, β, γ_cum
        //   每 token 需要 L2 norm 归约 → blockReduceSum
        // ================================================================
        for (int ti = 0; ti < chunk_len; ti++) {
            int t = chunk_start + ti;
            int q_base = t * token_stride + h_k * kd;
            int k_base = t * token_stride + h_k * kd;
            int v_base = t * token_stride + h_v * vd;

            float local_k_sq = 0.f, local_q_sq = 0.f;
            for (int i = j; i < kd; i += 128) {
                float kv = __bfloat162float(k[k_base + i]);
                local_k_sq += kv * kv;
                float qv = __bfloat162float(q[q_base + i]);
                local_q_sq += qv * qv;
            }
            local_k_sq = blockReduceSum_cw(local_k_sq);
            local_q_sq = blockReduceSum_cw(local_q_sq);

            if (j == 0) {
                reduce_tmp[0] = rsqrtf(local_k_sq + 1e-6f);
                reduce_tmp[1] = rsqrtf(local_q_sq + 1e-6f) * q_scale;
            }
            __syncthreads();
            float k_ns = reduce_tmp[0];
            float q_ns = reduce_tmp[1];

            // Store normalized K, Q (padded stride kd_pad to avoid bank conflict)
            for (int i = j; i < kd; i += 128) {
                chunk_K[ti * kd_pad + i] = __bfloat162float(k[k_base + i]) * k_ns;
                chunk_Q[ti * kd_pad + i] = __bfloat162float(q[q_base + i]) * q_ns;
            }
            // Store V
            chunk_V[ti * vd + j] = __bfloat162float(v[v_base + j]);

            // α, β, γ_cum (serial, thread 0 only)
            if (j == 0) {
                float a_val = __bfloat162float(a_raw[t * nv + h_v]);
                float bias  = dt_bias ? __bfloat162float(dt_bias[h_v]) : 0.f;
                float a_l   = A_log ? A_log[h_v] : 0.f;
                float ab    = a_val + bias;
                float dt_v  = (ab > 20.f) ? ab : log1pf(expf(ab));
                float log_alpha = -dt_v * expf(a_l);
                alpha_arr[ti] = expf(log_alpha);
                beta_arr[ti]  = 1.0f / (1.0f + expf(-__bfloat162float(beta_raw[t * nv + h_v])));
                gamma_cum[ti] = (ti == 0) ? log_alpha : gamma_cum[ti - 1] + log_alpha;
            }
            __syncthreads();
        }
        // Pad short chunk
        for (int ti = chunk_len + j; ti < C; ti += 128) {
            if (ti < C) {
                for (int i = 0; i < kd; i++) {
                    chunk_K[ti * kd_pad + i] = 0.f;
                    chunk_Q[ti * kd_pad + i] = 0.f;
                }
                chunk_V[ti * vd + (j < vd ? j : 0)] = 0.f;
            }
        }
        if (j == 0) {
            for (int ti = chunk_len; ti < C; ti++) {
                alpha_arr[ti] = 1.f;
                beta_arr[ti]  = 0.f;
                gamma_cum[ti] = (chunk_len > 0) ? gamma_cum[chunk_len - 1] : 0.f;
            }
        }
        __syncthreads();
        PHASE_TIMER_MARK(0);  // Phase 0 done

        // ================================================================
        // Phase 1: 计算 V'[t,j] = V[t,j] - exp(γ[t]) * dot(K[t,:], S[:,j])
        //   使用 tiled KS 避免 64-float 寄存器数组 → 消除 stack spill
        //   预计算 exp(gamma_cum[t]) → 复用 reduce_tmp 放不下, 用 beta_arr 后面空间
        // ================================================================
        // 预计算 exp_gamma[t] 到 alpha_arr (复用: 后面 Phase 2 之后 alpha_arr 不再需要)
        // alpha_arr 当前存的是 exp(log_alpha[t]), 但我们需要 exp(gamma_cum[t])
        // 不能覆盖 alpha_arr (Phase 2 的 IKK 需要 beta_arr 但不需要 alpha_arr? 让我检查)
        // 实际上 IKK 使用的是 beta_arr 和 gamma_cum, 不直接用 alpha_arr
        // → 安全覆盖 alpha_arr 为 exp(gamma_cum[t])
        if (j < chunk_len) {
            alpha_arr[j] = expf(gamma_cum[j]);
        }
        __syncthreads();
        {
            constexpr int TILE_T = 8;
            for (int t_base = 0; t_base < chunk_len; t_base += TILE_T) {
                int tile_end = min(t_base + TILE_T, chunk_len);
                int tile_len = tile_end - t_base;
                float ks_tile[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < kd; i++) {
                    float s_val = S_local[i * vd_pad + j];
                    #pragma unroll
                    for (int dt = 0; dt < 8 && dt < tile_len; dt++)
                        ks_tile[dt] += chunk_K[(t_base + dt) * kd_pad + i] * s_val;
                }
                // V' = V - exp(γ) * KS for this tile
                for (int dt = 0; dt < tile_len; dt++)
                    chunk_V[(t_base + dt) * vd + j] -= alpha_arr[t_base + dt] * ks_tile[dt];
            }
        }
        __syncthreads();
        PHASE_TIMER_MARK(1);  // Phase 1 done

        // ================================================================
        // Phase 2: 构建 IKK, 求逆 → T_mat
        //   IKK[s,t] = δ(s,t) + (s>t) * β[s] * exp(Γ_st) * dot(K[s], K[t])
        //   T_mat = IKK^{-1} · diag(β)
        //
        //   2a: 线程 s (s < C) 负责 IKK 第 s 行的构建 (行并行)
        //   2b: 列并行 forward substitution — 所有 128 线程参与 sync
        // ================================================================
        // 2a: 构建 IKK 矩阵 — 全 128 线程协作, 每线程处理 C*C/128 = 32 个元素
        for (int idx = j; idx < C * C; idx += 128) {
            int s = idx / C;
            int t = idx % C;
            float val = 0.f;
            if (s == t) {
                val = 1.f;
            } else if (s > t && s < chunk_len && t < chunk_len) {
                float dot = 0.f;
                for (int d = 0; d < kd; d++)
                    dot += chunk_K[s * kd_pad + d] * chunk_K[t * kd_pad + d];
                val = beta_arr[s] * expf(gamma_cum[s] - gamma_cum[t]) * dot;
            }
            mat[s * C + t] = val;
        }
        __syncthreads();
        PHASE_TIMER_MARK(2);  // Phase 2a (IKK construction) done

        // 2b: 列并行 forward substitution  (ALL 128 threads participate in sync)
        //     Thread j handles column j (if j < chunk_len), threads >= chunk_len idle but sync
        //     逐行处理: row s 的 T_inv[s, col] = δ(s,col) - sum_{t<s} IKK[s,t] * T_inv[t, col]
        //     mat[s*C+t] 中 row s 的 IKK 值在 row s 被处理时才被读, 然后覆写为 T_inv
        for (int s = 0; s < chunk_len; s++) {
            float result = 0.f;
            if (j < chunk_len) {
                result = (s == j) ? 1.f : 0.f;
                for (int t = 0; t < s; t++)
                    result -= mat[s * C + t] * mat[t * C + j]; // IKK[s,t] * T_inv[t,j]
            }
            __syncthreads();  // 所有线程同步: 确保 IKK[s,:] 已读完
            if (j < chunk_len)
                mat[s * C + j] = result;
            __syncthreads();  // 所有线程同步: 确保 T_inv[s,:] 已写完
        }

        // β 列缩放: T_mat[s, col] *= β[col]
        if (j < chunk_len) {
            float b_col = beta_arr[j];
            for (int s = j; s < chunk_len; s++)
                mat[s * C + j] *= b_col;
        }
        __syncthreads();
        PHASE_TIMER_MARK(3);  // Phase 2b (forward sub + β-scaling) done

        // ================================================================
        // Phase 3: new_v = T_mat @ V'
        //   new_v[s, j] = sum_{t<=s} T_mat[s,t] * V'[t, j]
        //   覆写 chunk_V (V' → new_v), 使用 tiled 避免大寄存器数组
        // ================================================================
        {
            constexpr int TILE_S = 8;
            for (int s_base = 0; s_base < chunk_len; s_base += TILE_S) {
                int tile_end = min(s_base + TILE_S, chunk_len);
                float nv_tile[8];
                for (int ds = 0; ds < tile_end - s_base; ds++) {
                    int s = s_base + ds;
                    float nv_sj = 0.f;
                    for (int t = 0; t <= s; t++)
                        nv_sj += mat[s * C + t] * chunk_V[t * vd + j];
                    nv_tile[ds] = nv_sj;
                }
                // 等所有线程算完这个 tile 后再覆写 chunk_V
                // (因为 chunk_V[t] 可能被后面的 s 行需要)
                __syncthreads();
                for (int ds = 0; ds < tile_end - s_base; ds++)
                    chunk_V[(s_base + ds) * vd + j] = nv_tile[ds];
                __syncthreads();
            }
        }
        PHASE_TIMER_MARK(4);  // Phase 3 (T@V') done

        // ================================================================
        // Phase 4: 构建 QK_mask 并计算 output
        //   QK_mask[s,t] = (s>=t) * exp(Γ_st) * dot(Q[s,:], K[t,:])
        //   o_inter[s,j] = exp(γ[s]) * QS[s]
        //   o_intra[s,j] = sum_{t<=s} QK_mask[s,t] * new_v[t,j]
        //   y_out = o_inter + o_intra
        //
        //   QK_mask 在 mat中构建, 线程 s (< C) 负责行 s
        // ================================================================
        // 构建 QK_mask — 全 128 线程协作 (padded stride)
        for (int idx = j; idx < C * C; idx += 128) {
            int s = idx / C;
            int t = idx % C;
            float val = 0.f;
            if (s >= t && s < chunk_len && t < chunk_len) {
                float dot = 0.f;
                for (int d = 0; d < kd; d++)
                    dot += chunk_Q[s * kd_pad + d] * chunk_K[t * kd_pad + d];
                val = expf(gamma_cum[s] - gamma_cum[t]) * dot;
            }
            mat[s * C + t] = val;
        }
        __syncthreads();

        // 计算输出: QS[s] 实时计算, exp(gamma) 预计算到 alpha_arr (复用)
        // alpha_arr 在 Phase 2 后已不需要, 重新填充 exp(gamma_cum[s])
        if (j < chunk_len)
            alpha_arr[j] = expf(gamma_cum[j]);
        __syncthreads();
        for (int s = 0; s < chunk_len; s++) {
            // o_inter = exp(γ[s]) * dot(Q[s,:], S_old[:,j])  — 实时计算
            float qs_val = 0.f;
            for (int i = 0; i < kd; i++)
                qs_val += chunk_Q[s * kd_pad + i] * S_local[i * vd_pad + j];
            float o = alpha_arr[s] * qs_val;
            // o_intra = sum_t QK_mask[s,t] * new_v[t,j]
            for (int t = 0; t <= s; t++)
                o += mat[s * C + t] * chunk_V[t * vd + j];
            int out_base = ((chunk_start + s) * nv + h_v) * vd;
            y_out[out_base + j] = __float2bfloat16(o);
        }
        __syncthreads();
        PHASE_TIMER_MARK(5);  // Phase 4 (QK_mask + output) done

        // ================================================================
        // Phase 5: 更新 SSM state
        //   S_new[i,j] = exp(γ_block) * S_old[i,j]
        //       + sum_{s} exp(γ_block - γ[s]) * K[s,i] * new_v[s,j]
        //   预计算 exp 衰减因子到 alpha_arr (复用, 不再需要)
        // ================================================================
        float gamma_block = gamma_cum[chunk_len - 1];
        // 预计算到 alpha_arr[s] = exp(γ_block - γ_cum[s])
        if (j < chunk_len)
            alpha_arr[j] = expf(gamma_block - gamma_cum[j]);
        float exp_gamma_block = expf(gamma_block);
        __syncthreads();
        for (int i = 0; i < kd; i++) {
            float s_val = exp_gamma_block * S_local[i * vd_pad + j];
            for (int s = 0; s < chunk_len; s++) {
                s_val += alpha_arr[s] * chunk_K[s * kd_pad + i] * chunk_V[s * vd + j];
            }
            S_local[i * vd_pad + j] = s_val;
        }
        __syncthreads();
        PHASE_TIMER_MARK(6);  // Phase 5 (state update) done

    }  // end chunk loop

    // ---- Phase timing write-back (block 0, thread 0 only) ----
    if (_do_pt) {
        for (int p = 0; p < 7; p++)
            phase_times[p] = _pt[p];
    }

    // ---- 写回 SSM state ----
    for (int i = 0; i < kd; i++)
        ssm_state[ss_base + i * vd + j] = S_local[i * vd_pad + j];
}


// ============================================================================
// 现有串行 kernel (复制自 light_ops.cu, 用于本文件独立对比)
// ============================================================================
__global__ void __launch_bounds__(128, 2)
gated_delta_net_serial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ a_raw,
    const __nv_bfloat16* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ beta_raw,
    float* __restrict__ ssm_state,
    __nv_bfloat16* __restrict__ y_out,
    int num_tokens, int kd, int nv_per_kh, int vd,
    int token_stride, int nkh_x_nvpkh)
{
    int h_v = blockIdx.x;
    int h_k = h_v / nv_per_kh;
    int nv  = nkh_x_nvpkh;
    int j   = threadIdx.x;

    extern __shared__ float smem[];
    const int vd_pad = vd + 1;
    float* S_smem  = smem;
    float* k_hat_s = S_smem + kd * vd_pad;
    float* q_hat_s = k_hat_s + kd;
    float* s_norms = q_hat_s + kd;

    int ss_base = h_v * kd * vd;
    float q_scale = rsqrtf((float)kd);

    for (int i = 0; i < kd; i++)
        S_smem[i * vd_pad + j] = ssm_state[ss_base + i * vd + j];
    __syncthreads();

    for (int t = 0; t < num_tokens; t++) {
        int q_base = t * token_stride + h_k * kd;
        int k_base = t * token_stride + h_k * kd;

        if (threadIdx.x == 0) { s_norms[0] = 0.f; s_norms[1] = 0.f; }
        __syncthreads();
        float local_k_sq = 0.f, local_q_sq = 0.f;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            float kv = __bfloat162float(k[k_base + i]);
            local_k_sq += kv * kv;
            float qv = __bfloat162float(q[q_base + i]);
            local_q_sq += qv * qv;
        }

        // blockReduceSum inline
        {
            float val = local_k_sq;
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);
            __shared__ float warp_buf_k[4];
            if ((threadIdx.x & 31) == 0) warp_buf_k[threadIdx.x >> 5] = val;
            __syncthreads();
            val = (threadIdx.x < 4) ? warp_buf_k[threadIdx.x] : 0.f;
            for (int offset = 2; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);
            if (threadIdx.x == 0) s_norms[0] = val;
        }
        __syncthreads();
        {
            float val = local_q_sq;
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);
            __shared__ float warp_buf_q[4];
            if ((threadIdx.x & 31) == 0) warp_buf_q[threadIdx.x >> 5] = val;
            __syncthreads();
            val = (threadIdx.x < 4) ? warp_buf_q[threadIdx.x] : 0.f;
            for (int offset = 2; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xffffffff, val, offset);
            if (threadIdx.x == 0) s_norms[1] = val;
        }
        __syncthreads();

        float k_norm = rsqrtf(s_norms[0] + 1e-6f);
        float q_norm = rsqrtf(s_norms[1] + 1e-6f) * q_scale;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            k_hat_s[i] = __bfloat162float(k[k_base + i]) * k_norm;
            q_hat_s[i] = __bfloat162float(q[q_base + i]) * q_norm;
        }
        __syncthreads();

        float a_val = __bfloat162float(a_raw[t * nv + h_v]);
        float bias  = dt_bias ? __bfloat162float(dt_bias[h_v]) : 0.f;
        float a_l   = A_log   ? A_log[h_v]                    : 0.f;
        float ab    = a_val + bias;
        float dt_v  = (ab > 20.f) ? ab : log1pf(expf(ab));
        float alpha_v = expf(-dt_v * expf(a_l));
        float beta_v = 1.0f / (1.0f + expf(-__bfloat162float(beta_raw[t * nv + h_v])));

        int v_base = t * token_stride + h_v * vd;
        float kS_j = 0.f;
        for (int i = 0; i < kd; i++)
            kS_j += k_hat_s[i] * S_smem[i * vd_pad + j];

        float v_j = __bfloat162float(v[v_base + j]);
        float delta_j = v_j - alpha_v * kS_j;

        float y_j = 0.f;
        for (int i = 0; i < kd; i++) {
            float beta_k_i = beta_v * k_hat_s[i];
            float old_s = S_smem[i * vd_pad + j];
            float new_s = alpha_v * old_s + beta_k_i * delta_j;
            S_smem[i * vd_pad + j] = new_s;
            y_j += new_s * q_hat_s[i];
        }

        int y_base = (t * nv + h_v) * vd;
        y_out[y_base + j] = __float2bfloat16(y_j);
        __syncthreads();
    }

    for (int i = 0; i < kd; i++)
        ssm_state[ss_base + i * vd + j] = S_smem[i * vd_pad + j];
}

// ============================================================================
// Dispatch 函数
// ============================================================================
void invoke_chunkwise(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias,
    const float* A_log, const __nv_bfloat16* beta_raw,
    float* ssm_state, __nv_bfloat16* y_out,
    int num_tokens, int nkh, int kd, int nv_per_kh, int vd,
    cudaStream_t stream, int token_stride,
    int chunk_size = 64,
    long long* phase_times = nullptr)
{
    if (token_stride <= 0) token_stride = nkh * kd;
    int nkh_x_nvpkh = nkh * nv_per_kh;
    int threads = 128;
    int grid = nkh_x_nvpkh;

    const int vd_pad = vd + 1;
    const int kd_pad = kd + 1;  // padded to avoid bank conflict
    const int C = chunk_size;
    size_t smem_bytes =
        (size_t)(kd * vd_pad
                 + C * kd_pad       // chunk_K (padded)
                 + C * kd_pad       // chunk_Q (padded)
                 + C * vd           // chunk_V / new_v
                 + C * C            // mat (IKK/T_mat/QK_mask)
                 + C * 3            // alpha, beta, gamma_cum
                 + 2                // reduce_tmp
                 ) * sizeof(float);

    cudaFuncSetAttribute(
        gated_delta_net_chunkwise_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);

    gated_delta_net_chunkwise_kernel<<<grid, threads, smem_bytes, stream>>>(
        q, k, v, a_raw, dt_bias, A_log, beta_raw,
        ssm_state, y_out,
        num_tokens, kd, nv_per_kh, vd, token_stride, nkh_x_nvpkh,
        C, phase_times);
}

void invoke_serial(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias,
    const float* A_log, const __nv_bfloat16* beta_raw,
    float* ssm_state, __nv_bfloat16* y_out,
    int num_tokens, int nkh, int kd, int nv_per_kh, int vd,
    cudaStream_t stream, int token_stride)
{
    if (token_stride <= 0) token_stride = nkh * kd;
    int nkh_x_nvpkh = nkh * nv_per_kh;
    int threads = std::min(vd, 128);
    int grid = nkh_x_nvpkh;
    const int vd_pad = vd + 1;
    size_t smem_bytes = (size_t)(kd * vd_pad + 2 * kd + 2) * sizeof(float);

    cudaFuncSetAttribute(
        gated_delta_net_serial_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem_bytes);

    gated_delta_net_serial_kernel<<<grid, threads, smem_bytes, stream>>>(
        q, k, v, a_raw, dt_bias, A_log, beta_raw,
        ssm_state, y_out,
        num_tokens, kd, nv_per_kh, vd, token_stride, nkh_x_nvpkh);
}

}  // namespace deltanet_eval


// ============================================================================
// Standalone micro-benchmark main()
// ============================================================================
int main(int argc, char* argv[]) {
    // Qwen3.5-27B DeltaNet 参数
    constexpr int NKH = 16;          // key heads
    constexpr int NV_PER_KH = 3;     // value heads per key head
    constexpr int NV = NKH * NV_PER_KH;  // 48
    constexpr int KD = 128;          // key dim
    constexpr int VD = 128;          // value dim
    // token_stride: 每个 token 在 q/k 中的步长 (= nkh * kd = 2048 for q/k)

    int num_tokens = 256;
    int warmup_iters = 3;
    int bench_iters = 10;
    int chunk_size = 64;
    bool do_profile = false;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i+1 < argc) num_tokens = atoi(argv[++i]);
        if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) warmup_iters = atoi(argv[++i]);
        if (strcmp(argv[i], "--iters")  == 0 && i+1 < argc) bench_iters = atoi(argv[++i]);
        if (strcmp(argv[i], "--chunk-size") == 0 && i+1 < argc) chunk_size = atoi(argv[++i]);
        if (strcmp(argv[i], "--profile") == 0) do_profile = true;
    }

    printf("=== DeltaNet WY Chunkwise Evaluation ===\n");
    printf("Model dims: NKH=%d, NV_PER_KH=%d, NV=%d, KD=%d, VD=%d\n",
           NKH, NV_PER_KH, NV, KD, VD);
    printf("Test config: tokens=%d, warmup=%d, iters=%d, chunk_size=%d\n\n",
           num_tokens, warmup_iters, bench_iters, chunk_size);

    // 注: q/k 的 token_stride = NKH * KD = 2048 (key heads 连续)
    //     v 的 token_stride 实际是 NV * VD = 6144 (value heads 连续)
    //     但为简化, 本测试使用统一 token_stride
    //     实际模型中 q, k, v 可以有不同 stride
    //     这里使用 q/k/v 分离的布局: q[T, NKH*KD], k[T, NKH*KD], v[T, NV*VD]
    //     token_stride 仅影响 q/k (= NKH*KD), v 的 stride 是 NV*VD (硬编码)
    const int q_stride = NKH * KD;    // 2048
    const int k_stride = NKH * KD;    // 同上 (实际模型中 q/k 共享 stride)
    const int v_stride = NV * VD;     // 6144

    // 为了使 kernel 中 `t * token_stride + h_k * kd` 和 `t * token_stride + h_v * vd`
    // 在同一 token_stride 下正确, 我们需要使 q/k/v 使用各自独立的缓冲区
    // 但 kernel 当前设计是 q,k 共用 token_stride, v 硬编码 `t * token_stride + h_v * vd`
    // 实际模型中: q 和 k 的 token_stride = total_q_dim (或 interleaved stride)
    //           v 的 token_stride 的偏移计算方式不同
    // 为了正确性, 使用 token_stride = max(NKH*KD, NV*VD) 并在一个大 buffer 中存放所有

    // 简化方案: token_stride = NV * VD = 6144 (足够存下 NKH*KD=2048 和 NV*VD=6144)
    // q[t] 起始于 t * 6144, k[t] 起始于同, v[t] 起始于同
    // 但这浪费内存. 更好的方案: 分离 q, k, v 指针.
    // 当前 kernel 用同一 token_stride, v 偏移 = t * token_stride + h_v * vd
    // 如果 token_stride = NV * VD = 6144, 则 v[t][h_v] = data[t*6144 + h_v*128] ✓
    // q/k 偏移 = t * token_stride + h_k * kd
    // 如果 token_stride = 6144, 则 q[t][h_k] = data[t*6144 + h_k*128] ✓ (前 2048 字节)
    // 只要 NKH*KD <= NV*VD, 没有越界. NKH*KD=2048 < NV*VD=6144 ✓

    const int token_stride = NV * VD;  // 6144, 确保 q/k/v 都不越界

    // Allocate unified memory
    size_t q_bytes = (size_t)num_tokens * token_stride * sizeof(__nv_bfloat16);
    size_t k_bytes = q_bytes;
    size_t v_bytes = q_bytes;
    size_t a_bytes = (size_t)num_tokens * NV * sizeof(__nv_bfloat16);
    size_t beta_bytes = a_bytes;
    size_t dt_bias_bytes = NV * sizeof(__nv_bfloat16);
    size_t A_log_bytes   = NV * sizeof(float);
    size_t ssm_bytes     = NV * KD * VD * sizeof(float);
    size_t y_bytes       = (size_t)num_tokens * NV * VD * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_q, *d_k, *d_v, *d_a, *d_beta, *d_dt;
    float *d_A_log;
    float *d_ssm_serial, *d_ssm_chunk;
    __nv_bfloat16 *d_y_serial, *d_y_chunk;

    cudaMallocManaged(&d_q, q_bytes);
    cudaMallocManaged(&d_k, k_bytes);
    cudaMallocManaged(&d_v, v_bytes);
    cudaMallocManaged(&d_a, a_bytes);
    cudaMallocManaged(&d_beta, beta_bytes);
    cudaMallocManaged(&d_dt, dt_bias_bytes);
    cudaMallocManaged(&d_A_log, A_log_bytes);
    cudaMallocManaged(&d_ssm_serial, ssm_bytes);
    cudaMallocManaged(&d_ssm_chunk,  ssm_bytes);
    cudaMallocManaged(&d_y_serial, y_bytes);
    cudaMallocManaged(&d_y_chunk,  y_bytes);

    printf("Memory allocated:\n");
    printf("  q/k/v: %.1f MB each\n", q_bytes / 1048576.0);
    printf("  SSM state: %.1f MB\n",  ssm_bytes / 1048576.0);
    printf("  Output: %.1f MB\n",     y_bytes / 1048576.0);

    // Initialize with random data
    srand(42);
    auto randf = [](){ return (float)(rand() % 1000 - 500) / 1000.0f; };

    for (size_t i = 0; i < q_bytes / sizeof(__nv_bfloat16); i++)
        d_q[i] = __float2bfloat16(randf());
    for (size_t i = 0; i < k_bytes / sizeof(__nv_bfloat16); i++)
        d_k[i] = __float2bfloat16(randf());
    for (size_t i = 0; i < v_bytes / sizeof(__nv_bfloat16); i++)
        d_v[i] = __float2bfloat16(randf());
    for (size_t i = 0; i < a_bytes / sizeof(__nv_bfloat16); i++)
        d_a[i] = __float2bfloat16(randf() * 0.5f);
    for (size_t i = 0; i < beta_bytes / sizeof(__nv_bfloat16); i++)
        d_beta[i] = __float2bfloat16(randf());
    for (int i = 0; i < NV; i++) {
        d_dt[i] = __float2bfloat16(randf() * 0.1f);
        d_A_log[i] = randf() * 0.5f;
    }
    // SSM state 初始化为 0
    memset(d_ssm_serial, 0, ssm_bytes);
    memset(d_ssm_chunk,  0, ssm_bytes);

    cudaDeviceSynchronize();
    printf("  Data initialized.\n\n");

    // ================================================================
    // Step 1: 正确性验证
    // ================================================================
    printf("--- Correctness Check ---\n");

    // Run serial kernel
    deltanet_eval::invoke_serial(
        d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
        d_ssm_serial, d_y_serial,
        num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride);
    cudaError_t err1 = cudaDeviceSynchronize();
    printf("  Serial kernel: %s\n",
           err1 == cudaSuccess ? "OK" : cudaGetErrorString(err1));

    // Run chunkwise kernel
    deltanet_eval::invoke_chunkwise(
        d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
        d_ssm_chunk, d_y_chunk,
        num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride,
        chunk_size);
    cudaError_t err2 = cudaDeviceSynchronize();
    printf("  Chunkwise kernel: %s\n",
           err2 == cudaSuccess ? "OK" : cudaGetErrorString(err2));

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("  KERNEL ERROR — skipping correctness check\n");
    } else {
        // Compare outputs
        int total_elements = num_tokens * NV * VD;
        double max_abs_err = 0.0;
        double sum_abs_err = 0.0;
        double sum_ref_abs = 0.0;
        int num_mismatch = 0;
        for (int i = 0; i < total_elements; i++) {
            float ref = __bfloat162float(d_y_serial[i]);
            float tst = __bfloat162float(d_y_chunk[i]);
            double err = fabs((double)ref - (double)tst);
            max_abs_err = fmax(max_abs_err, err);
            sum_abs_err += err;
            sum_ref_abs += fabs((double)ref);
            if (err > 0.1 * (fabs((double)ref) + 1e-6))
                num_mismatch++;
        }
        double avg_abs_err = sum_abs_err / total_elements;
        double rel_err = sum_abs_err / (sum_ref_abs + 1e-12);
        printf("  Output comparison (%d elements):\n", total_elements);
        printf("    Max abs error:  %.6e\n", max_abs_err);
        printf("    Avg abs error:  %.6e\n", avg_abs_err);
        printf("    Relative error: %.6e\n", rel_err);
        printf("    Mismatches (>10%%): %d\n", num_mismatch);

        // Compare SSM states
        int ss_elements = NV * KD * VD;
        double ss_max_err = 0.0, ss_sum_err = 0.0, ss_sum_ref = 0.0;
        for (int i = 0; i < ss_elements; i++) {
            double ref = d_ssm_serial[i];
            double tst = d_ssm_chunk[i];
            double err = fabs(ref - tst);
            ss_max_err = fmax(ss_max_err, err);
            ss_sum_err += err;
            ss_sum_ref += fabs(ref);
        }
        printf("  SSM state comparison (%d elements):\n", ss_elements);
        printf("    Max abs error:  %.6e\n", ss_max_err);
        printf("    Relative error: %.6e\n", ss_sum_err / (ss_sum_ref + 1e-12));
        if (rel_err < 0.01 && ss_sum_err / (ss_sum_ref + 1e-12) < 0.01)
            printf("  ✅ PASS\n");
        else
            printf("  ❌ FAIL (error too high)\n");
    }
    printf("\n");

    // ================================================================
    // Step 2: 性能基准
    // ================================================================
    printf("--- Performance Benchmark ---\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark serial
    for (int w = 0; w < warmup_iters; w++) {
        memset(d_ssm_serial, 0, ssm_bytes);
        cudaDeviceSynchronize();
        deltanet_eval::invoke_serial(
            d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
            d_ssm_serial, d_y_serial,
            num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride);
        cudaDeviceSynchronize();
    }

    float serial_ms = 0.f;
    for (int it = 0; it < bench_iters; it++) {
        memset(d_ssm_serial, 0, ssm_bytes);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        deltanet_eval::invoke_serial(
            d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
            d_ssm_serial, d_y_serial,
            num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        serial_ms += ms;
    }
    serial_ms /= bench_iters;

    // Benchmark chunkwise
    // Allocate phase timing buffer if profiling requested
    long long* d_phase_times = nullptr;
    long long h_phase_times[7] = {0};
    if (do_profile) {
        cudaMallocManaged(&d_phase_times, 7 * sizeof(long long));
    }

    for (int w = 0; w < warmup_iters; w++) {
        memset(d_ssm_chunk, 0, ssm_bytes);
        cudaDeviceSynchronize();
        deltanet_eval::invoke_chunkwise(
            d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
            d_ssm_chunk, d_y_chunk,
            num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride,
            chunk_size);
        cudaDeviceSynchronize();
    }

    float chunk_ms = 0.f;
    for (int it = 0; it < bench_iters; it++) {
        memset(d_ssm_chunk, 0, ssm_bytes);
        if (d_phase_times) memset(d_phase_times, 0, 7 * sizeof(long long));
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        deltanet_eval::invoke_chunkwise(
            d_q, d_k, d_v, d_a, d_dt, d_A_log, d_beta,
            d_ssm_chunk, d_y_chunk,
            num_tokens, NKH, KD, NV_PER_KH, VD, 0, token_stride,
            chunk_size, d_phase_times);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        chunk_ms += ms;
        // Accumulate phase times from last iteration
        if (d_phase_times && it == bench_iters - 1) {
            cudaDeviceSynchronize();
            for (int p = 0; p < 7; p++)
                h_phase_times[p] = d_phase_times[p];
        }
    }
    chunk_ms /= bench_iters;

    printf("  %-20s %8.3f ms (T=%d, 1 layer)\n", "Serial:", serial_ms, num_tokens);
    printf("  %-20s %8.3f ms (T=%d, 1 layer)\n", "Chunkwise:", chunk_ms, num_tokens);
    printf("  Speedup:            %.2fx\n", serial_ms / chunk_ms);
    printf("  Serial per-head:    %.3f ms (%d heads)\n", serial_ms / NV, NV);
    printf("  Chunk per-head:     %.3f ms (%d heads)\n", chunk_ms / NV, NV);
    printf("\n");

    // ================================================================
    // Step 3: Shared Memory 分析
    // ================================================================
    const int vd_pad = VD + 1;
    const int C = chunk_size;
    size_t smem_serial = (size_t)(KD * vd_pad + 2 * KD + 2) * sizeof(float);
    const int kd_pad_print = KD + 1;
    size_t smem_chunk  = (size_t)(KD * vd_pad + C*kd_pad_print + C*kd_pad_print + C*VD + C*C + C*3 + 2) * sizeof(float);
    printf("--- Shared Memory Usage ---\n");
    printf("  Serial:    %6zu bytes (%.1f KB)\n", smem_serial, smem_serial / 1024.0);
    printf("  Chunkwise: %6zu bytes (%.1f KB) [C=%d]\n", smem_chunk, smem_chunk / 1024.0, C);
    printf("  SM110 max: 228 KB (233,472 bytes)\n");
    printf("  Chunkwise headroom: %.1f KB\n", (233472 - (int)smem_chunk) / 1024.0);
    printf("  Max blocks/SM: %d\n", (int)(233472 / smem_chunk));
    printf("\n");

    // ================================================================
    // Step 3.5: Phase Timing (if --profile)
    // ================================================================
    if (do_profile) {
        const char* phase_names[7] = {
            "P0: Data loading   ",
            "P1: KS/V' compute  ",
            "P2a: IKK construct ",
            "P2b: Fwd sub + beta",
            "P3: T_mat @ V'     ",
            "P4: QK_mask+output ",
            "P5: State update   "
        };
        // Convert clock cycles to microseconds (GPU clock ~1575 MHz)
        const double gpu_freq_mhz = 1575.0;  // SM110 GPC clock
        long long total_cycles = 0;
        for (int p = 0; p < 7; p++) total_cycles += h_phase_times[p];
        int num_chunks = (num_tokens + C - 1) / C;
        printf("--- Phase Timing (block 0, last iteration, %d chunks) ---\n", num_chunks);
        printf("  %-21s %10s %8s %8s\n", "Phase", "Cycles", "us", "%%");
        for (int p = 0; p < 7; p++) {
            double us = h_phase_times[p] / gpu_freq_mhz;
            double pct = total_cycles > 0 ? 100.0 * h_phase_times[p] / total_cycles : 0.0;
            printf("  %s %10lld %8.1f %7.1f%%\n", phase_names[p], h_phase_times[p], us, pct);
        }
        double total_us = total_cycles / gpu_freq_mhz;
        printf("  %-21s %10lld %8.1f\n", "Total (block 0)", total_cycles, total_us);
        printf("  Per-chunk avg: %.1f us\n", total_us / num_chunks);
        printf("\n");
    }
    printf("\n");

    // ================================================================
    // Step 4: 48 层外推
    // ================================================================
    printf("--- 48-Layer Extrapolation ---\n");
    printf("  Serial  (48 layers): %.1f ms\n", serial_ms * 48);
    printf("  Chunkwise (48 layers): %.1f ms\n", chunk_ms * 48);
    printf("  Savings: %.1f ms (%.1f%%)\n",
           (serial_ms - chunk_ms) * 48,
           (1.0 - chunk_ms / serial_ms) * 100);
    printf("\n");

    // Cleanup
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_a); cudaFree(d_beta); cudaFree(d_dt);
    cudaFree(d_A_log);
    cudaFree(d_ssm_serial); cudaFree(d_ssm_chunk);
    cudaFree(d_y_serial); cudaFree(d_y_chunk);
    if (d_phase_times) cudaFree(d_phase_times);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
