// ============================================================================
// NVFP4 W4A16 GEMV Kernels + cuBLASLt GEMM
//
// FP4 E2M1 packed weight (2 values/byte) + F8_E4M3 per-group-16 scale
// Dequant: W = fp4_val × e4m3_scale / weight_global_scale
//
// GEMV: hand-written scattered kernels (T=1 decode)
// GEMM: cuBLASLt CUDA_R_4F_E2M1 + VEC16_UE4M3 (T>1 prefill)
// ============================================================================

#include "dense_gemm_fp4.h"
#include "layer.h"   // QuantizedWeight
#include <cublas_v2.h>
#include <cublasLt.h>
#include <iostream>

namespace qwen_thor {
namespace ops {

// ============================================================================
// FP4 E2M1 dequantization lookup table
// 4 bits → float: {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
// ============================================================================
__constant__ float c_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// E4M3 (unsigned, bias=7) → float inline device function
__device__ __forceinline__ float e4m3_to_float(uint8_t v) {
    int exp = (v >> 3) & 0xF;
    int man = v & 0x7;
    if (exp == 0) {
        // Subnormal: (man/8) × 2^(1-7) = man × 2^(-9)
        return (float)man * 1.953125e-3f;  // 1/512
    }
    // Normal: (1 + man/8) × 2^(exp-7)
    float fval = (1.0f + (float)man * 0.125f);
    // Use ldexpf for 2^(exp-7)
    return ldexpf(fval, exp - 7);
}

// ============================================================================
// FP4 GEMV Kernel — Scattered mapping, process per-group-16
//
// Grid:  [ceil(N / WARPS_PER_BLOCK)]
// Block: [WARPS_PER_BLOCK × 32]
// SMEM:  [K] × sizeof(bf16) for activation A
//
// Each warp computes one output element C[n]
// Weight layout: packed[N, K/2] U8, scale[N, K/16] U8 (E4M3)
// ============================================================================
template <bool ADD_RESIDUAL>
__global__ void fp4_gemv_kernel(
    const __nv_bfloat16* __restrict__ A,       // [1, K]
    const uint8_t* __restrict__ packed,        // [N, K/2]
    const uint8_t* __restrict__ scale,         // [N, K/16]
    __nv_bfloat16* __restrict__ C,             // [1, N]
    const __nv_bfloat16* __restrict__ residual,// [1, N] or nullptr
    float inv_global_scale,
    int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int out_idx = blockIdx.x + warp_id * num_blocks;

    // Cooperative load A → shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_A[i] = A[i];
    __syncthreads();

    if (out_idx >= N) return;

    const int K_half = K / 2;      // packed dimension
    const int K_groups = K / 16;   // number of scale groups
    const uint8_t* w_row = packed + (size_t)out_idx * K_half;
    const uint8_t* s_row = scale + (size_t)out_idx * K_groups;

    float sum = 0.0f;

    // Each lane processes groups in stride of WARP_SIZE
    // Each group: 16 FP4 values (8 bytes packed), 1 E4M3 scale byte
    for (int g = lane_id; g < K_groups; g += WARP_SIZE) {
        float group_scale = e4m3_to_float(s_row[g]) * inv_global_scale;
        int k_base = g * 16;

        // Load 8 bytes of packed weight (16 FP4 values)
        const uint8_t* w_ptr = w_row + g * 8;
        // Process 16 FP4 values
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t byte = w_ptr[j];
            int k0 = k_base + j * 2;
            float a0 = __bfloat162float(s_A[k0]);
            float a1 = __bfloat162float(s_A[k0 + 1]);
            float w0 = c_fp4_lut[byte & 0xF] * group_scale;
            float w1 = c_fp4_lut[(byte >> 4) & 0xF] * group_scale;
            sum += a0 * w0 + a1 * w1;
        }
    }

    // Warp reduce
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0) {
        if constexpr (ADD_RESIDUAL) {
            sum += __bfloat162float(residual[out_idx]);
        }
        C[out_idx] = __float2bfloat16(sum);
    }
}

// ============================================================================
// FP4 Dual GEMV Kernel — Gate + Up sharing A in SMEM
// Grid front half → W1/C1, back half → W2/C2
// ============================================================================
__global__ void fp4_dual_gemv_kernel(
    const __nv_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ packed1,
    const uint8_t* __restrict__ scale1,
    float inv_gs1,
    const uint8_t* __restrict__ packed2,
    const uint8_t* __restrict__ scale2,
    float inv_gs2,
    __nv_bfloat16* __restrict__ C1,
    __nv_bfloat16* __restrict__ C2,
    int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int blocks_per_output = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    bool is_second = (blockIdx.x >= blocks_per_output);
    int local_block = is_second ? (blockIdx.x - blocks_per_output) : blockIdx.x;
    int out_idx = local_block * WARPS_PER_BLOCK + warp_id;

    // Cooperative load A → shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_A[i] = A[i];
    __syncthreads();

    if (out_idx >= N) return;

    const int K_half = K / 2;
    const int K_groups = K / 16;
    const uint8_t* w_row = is_second
        ? (packed2 + (size_t)out_idx * K_half)
        : (packed1 + (size_t)out_idx * K_half);
    const uint8_t* s_row = is_second
        ? (scale2 + (size_t)out_idx * K_groups)
        : (scale1 + (size_t)out_idx * K_groups);
    float inv_gs = is_second ? inv_gs2 : inv_gs1;

    float sum = 0.0f;

    for (int g = lane_id; g < K_groups; g += WARP_SIZE) {
        float group_scale = e4m3_to_float(s_row[g]) * inv_gs;
        int k_base = g * 16;
        const uint8_t* w_ptr = w_row + g * 8;

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t byte = w_ptr[j];
            int k0 = k_base + j * 2;
            float a0 = __bfloat162float(s_A[k0]);
            float a1 = __bfloat162float(s_A[k0 + 1]);
            sum += a0 * (c_fp4_lut[byte & 0xF] * group_scale)
                 + a1 * (c_fp4_lut[(byte >> 4) & 0xF] * group_scale);
        }
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0) {
        __nv_bfloat16* out = is_second ? C2 : C1;
        out[out_idx] = __float2bfloat16(sum);
    }
}

// ============================================================================
// Host dispatch functions
// ============================================================================

void invoke_fp4_gemv(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* C,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS = BLOCK_THREADS / 32;
    int N = W.N, K = W.K;
    float inv_gs = 1.0f / W.global_scale;
    int blocks = (N + WARPS - 1) / WARPS;
    size_t smem = K * sizeof(__nv_bfloat16);

    fp4_gemv_kernel<false><<<blocks, BLOCK_THREADS, smem, stream>>>(
        A, W.packed, W.scale, C, nullptr, inv_gs, N, K);
}

void invoke_fp4_gemv_add(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* C,
    const __nv_bfloat16* residual,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS = BLOCK_THREADS / 32;
    int N = W.N, K = W.K;
    float inv_gs = 1.0f / W.global_scale;
    int blocks = (N + WARPS - 1) / WARPS;
    size_t smem = K * sizeof(__nv_bfloat16);

    fp4_gemv_kernel<true><<<blocks, BLOCK_THREADS, smem, stream>>>(
        A, W.packed, W.scale, C, residual, inv_gs, N, K);
}

void invoke_fp4_dual_gemv(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W1,
    const core::QuantizedWeight& W2,
    __nv_bfloat16* C1,
    __nv_bfloat16* C2,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS = BLOCK_THREADS / 32;
    int N = W1.N;  // W1, W2 should have same N for gate+up
    int K = W1.K;
    int blocks_per_output = (N + WARPS - 1) / WARPS;
    int total_blocks = blocks_per_output * 2;
    size_t smem = K * sizeof(__nv_bfloat16);

    fp4_dual_gemv_kernel<<<total_blocks, BLOCK_THREADS, smem, stream>>>(
        A,
        W1.packed, W1.scale, 1.0f / W1.global_scale,
        W2.packed, W2.scale, 1.0f / W2.global_scale,
        C1, C2, N, K);
}

// ============================================================================
// cuBLASLt FP4 GEMM (T>1 prefill)
//
// cuBLASLt native path:
//   CUDA_R_4F_E2M1 weight + CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
//   alpha = 1/global_scale, beta = 0 (or 1 for add variant)
//
// Fallback: dequantize to BF16 then cuBLAS GEMM (if native FP4 not supported)
// ============================================================================

static cublasLtHandle_t s_cublaslt_handle = nullptr;

void init_fp4_cublaslt() {
    if (!s_cublaslt_handle) {
        cublasLtCreate(&s_cublaslt_handle);
    }
}

void cleanup_fp4_cublaslt() {
    if (s_cublaslt_handle) {
        cublasLtDestroy(s_cublaslt_handle);
        s_cublaslt_handle = nullptr;
    }
}

// Dequantize FP4 packed to BF16 kernel (fallback path)
__global__ void fp4_dequant_kernel(
    __nv_bfloat16* __restrict__ out,       // [N, K]
    const uint8_t* __restrict__ packed,    // [N, K/2]
    const uint8_t* __restrict__ scale,     // [N, K/16]
    float inv_global_scale,
    int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx % K;

    int byte_idx = n * (K / 2) + k / 2;
    uint8_t byte = packed[byte_idx];
    int nibble = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
    float fp4_val = c_fp4_lut[nibble];

    int group_idx = n * (K / 16) + k / 16;
    float gs = e4m3_to_float(scale[group_idx]) * inv_global_scale;

    out[idx] = __float2bfloat16(fp4_val * gs);
}

// Static workspace for dequantized weights (reused across calls)
static __nv_bfloat16* s_dequant_buf = nullptr;
static size_t s_dequant_buf_size = 0;

static void ensure_dequant_buf(size_t need, cudaStream_t stream) {
    if (need > s_dequant_buf_size) {
        if (s_dequant_buf) {
            cudaStreamSynchronize(stream);
            cudaFree(s_dequant_buf);
        }
        cudaMalloc(&s_dequant_buf, need);
        s_dequant_buf_size = need;
    }
}

// cuBLAS handle for fallback GEMM
static cublasHandle_t s_cublas_handle = nullptr;
static cublasHandle_t get_cublas() {
    if (!s_cublas_handle) cublasCreate(&s_cublas_handle);
    return s_cublas_handle;
}

void invoke_fp4_gemm(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* C,
    int M,
    cudaStream_t stream)
{
    int N = W.N, K = W.K;
    float inv_gs = 1.0f / W.global_scale;

    // Fallback: dequantize to BF16 then cuBLAS GEMM
    // TODO: Try native cuBLASLt FP4 path when API is verified on SM110
    size_t need = (size_t)N * K * sizeof(__nv_bfloat16);
    ensure_dequant_buf(need, stream);

    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
        s_dequant_buf, W.packed, W.scale, inv_gs, N, K);

    // cuBLAS GEMM: C[M,N] = A[M,K] × W_dequant[K,N]^T
    // W_dequant is [N,K] row-major = [K,N] col-major
    // A is [M,K] row-major
    // For cuBLAS (col-major): C^T[N,M] = W^T[N,K] × A^T[K,M]
    //   → op(W) = N, op(A) = T, m=N, n=M, k=K
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle = get_cublas();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 s_dequant_buf, CUDA_R_16BF, K,
                 A, CUDA_R_16BF, K,
                 &beta,
                 C, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void invoke_fp4_gemm_add(
    const __nv_bfloat16* A,
    const core::QuantizedWeight& W,
    __nv_bfloat16* D,
    const __nv_bfloat16* residual,
    int M,
    cudaStream_t stream)
{
    int N = W.N, K = W.K;
    float inv_gs = 1.0f / W.global_scale;

    size_t need = (size_t)N * K * sizeof(__nv_bfloat16);
    ensure_dequant_buf(need, stream);

    int total = N * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fp4_dequant_kernel<<<blocks, threads, 0, stream>>>(
        s_dequant_buf, W.packed, W.scale, inv_gs, N, K);

    // Copy residual to D if they differ
    if (D != residual) {
        cudaMemcpyAsync(D, residual, (size_t)M * N * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // cuBLAS GEMM: D = A × W_dequant + D (beta=1)
    float alpha = 1.0f, beta = 1.0f;
    cublasHandle_t handle = get_cublas();
    cublasSetStream(handle, stream);
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 s_dequant_buf, CUDA_R_16BF, K,
                 A, CUDA_R_16BF, K,
                 &beta,
                 D, CUDA_R_16BF, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

} // namespace ops
} // namespace qwen_thor
