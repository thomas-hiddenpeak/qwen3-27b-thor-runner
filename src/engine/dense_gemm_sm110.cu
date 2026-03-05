#include "dense_gemm.h"
#include <stdexcept>
#include <iostream>
#include <cublas_v2.h>

// Warp/Block reduce helpers (for gemv_rmsnorm_kernel)
template <typename T>
__inline__ __device__ T gemv_warpReduceSum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
template <typename T>
__inline__ __device__ T gemv_blockReduceSum(T val) {
    static __shared__ T s_reduce[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = gemv_warpReduceSum(val);
    if (lane == 0) s_reduce[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? s_reduce[lane] : (T)0;
    if (wid == 0) val = gemv_warpReduceSum(val);
    return val;
}

// Safe cudaMalloc: on failure, keep old pointer and size, log error
#define SAFE_CUDA_REALLOC(ptr, sz_var, need, stream) do { \
    if ((need) > (sz_var)) { \
        if (ptr) { cudaStreamSynchronize(stream); cudaFree(ptr); ptr = nullptr; } \
        cudaError_t _err = cudaMalloc(&(ptr), (need)); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "[CUDA] cudaMalloc failed (%zu bytes): %s\n", \
                    (size_t)(need), cudaGetErrorString(_err)); \
            fflush(stderr); \
            ptr = nullptr; sz_var = 0; \
        } else { \
            sz_var = (need); \
        } \
    } \
} while(0)

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cutlass;
using namespace cute;

namespace qwen_thor {
namespace ops {

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementC = bfloat16_t;
using ElementD = bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _2, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using GemmType = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

DenseGEMMCUTLASS::DenseGEMMCUTLASS() {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    if (props.major < 9) {
        std::cerr << "Warning: DenseGEMMCUTLASS is optimized for SM100+ (Blackwell). Current SM: " 
                  << props.major << "." << props.minor << std::endl;
    }
}

DenseGEMMCUTLASS::~DenseGEMMCUTLASS() = default;

void DenseGEMMCUTLASS::forward(
    const core::Tensor& A,
    const core::Tensor& B,
    core::Tensor& C,
    void* stream
) {
    int M = A.shape()[0];
    int K = A.shape()[1];
    int N = B.shape()[1];

    // 使用裸指针接口 (支持 M padding)
    invoke_dense_gemm(
        static_cast<const __nv_bfloat16*>(A.data()),
        static_cast<const __nv_bfloat16*>(B.data()),
        static_cast<__nv_bfloat16*>(C.data()),
        M, N, K, static_cast<cudaStream_t>(stream)
    );
}

// ============================================================================
// cuBLAS 全局 handle (惰性初始化, 用于小尺寸 GEMM)
// ============================================================================
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

// ============================================================================
// invoke_dense_gemm: CUTLASS SM110 BF16 GEMM + 统一内存 per-GEMM sync
// C[M,N] = A[M,K] × B[K,N]
// A: RowMajor, B: ColumnMajor, C: RowMajor
// ============================================================================
void invoke_dense_gemm(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    // M padding to 8-aligned (required by CUTLASS tensor core)
    int M_padded = M;
    const __nv_bfloat16* A_eff = A;
    __nv_bfloat16* C_eff = C;

    static __nv_bfloat16* s_A_pad = nullptr;
    static __nv_bfloat16* s_C_pad = nullptr;
    static size_t s_A_pad_sz = 0;
    static size_t s_C_pad_sz = 0;

    if (M % 8 != 0) {
        M_padded = ((M + 7) / 8) * 8;
        size_t a_need = (size_t)M_padded * K * sizeof(__nv_bfloat16);
        size_t c_need = (size_t)M_padded * N * sizeof(__nv_bfloat16);
        SAFE_CUDA_REALLOC(s_A_pad, s_A_pad_sz, a_need, stream);
        SAFE_CUDA_REALLOC(s_C_pad, s_C_pad_sz, c_need, stream);
        if (!s_A_pad || !s_C_pad) return; // OOM guard
        cudaMemcpyAsync(s_A_pad, A, (size_t)M * K * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(s_A_pad + (size_t)M * K, 0,
                        (size_t)(M_padded - M) * K * sizeof(__nv_bfloat16), stream);
        A_eff = s_A_pad;
        C_eff = s_C_pad;
    }

    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;
    using ProblemShapeType = typename GemmType::GemmKernel::ProblemShape;

    ProblemShapeType problem_size = ProblemShapeType{M_padded, N, K, 1};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    static int s_cached_sm_count = -1;
    if (s_cached_sm_count < 0) {
        s_cached_sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }
    hw_info.sm_count = s_cached_sm_count;

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<const ElementA*>(A_eff), cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_padded, K, 1)),
         reinterpret_cast<const ElementB*>(B), cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1))},
        {{1.0f, 0.0f},
         reinterpret_cast<const ElementC*>(C_eff), cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_padded, N, 1)),
         reinterpret_cast<ElementD*>(C_eff), cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M_padded, N, 1))},
        hw_info
    };

    GemmType gemm;
    
    static void* s_workspace = nullptr;
    static size_t s_workspace_size = 0;
    size_t workspace_size = GemmType::get_workspace_size(args);
    SAFE_CUDA_REALLOC(s_workspace, s_workspace_size, workspace_size, stream);

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        // CUTLASS 无法处理此配置 (通常是 N 或 K 未对齐到 8)
        // Fallback 到 cuBLAS: C[M,N] = A[M,K] × B[K,N]
        // cuBLAS 是 column-major, 利用 C^T = B^T × A^T:
        //   A row-major [M,K] = col-major [K,M], op_N
        //   B col-major [K,N], op_T → [N,K]
        //   C row-major [M_padded,N] = col-major [N,M_padded]
        auto h = get_cublas_handle();
        cublasSetStream(h, stream);
        float alpha = 1.0f, beta_val = 0.0f;
        cublasGemmEx(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            A_eff == A ? B : B, CUDA_R_16BF, K,   // B [K,N] col-major → op_T
            A_eff, CUDA_R_16BF, K,                 // A [M,K] row-major = [K,M] col-major
            &beta_val,
            C_eff, CUDA_R_16BF, N,                 // C [M,N] row-major = [N,M] col-major
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        if (M_padded != M) {
            cudaMemcpy2DAsync(C, (size_t)N * sizeof(__nv_bfloat16),
                              C_eff, (size_t)N * sizeof(__nv_bfloat16),
                              (size_t)N * sizeof(__nv_bfloat16), M,
                              cudaMemcpyDeviceToDevice, stream);
        }
        return;
    }

    // 关键修复: 必须传 stream 给 initialize()!
    // CUTLASS initialize() 内部使用 cudaMemcpyAsync 写入 TMA 描述符到 workspace。
    // 如果不传 stream，默认使用 stream 0，与 gemm.run(stream) 在不同 stream 上，
    // 导致跨 stream 竞态 — TMA 描述符可能未写完就被 GPU kernel 读取。
    // 传入相同 stream 保证 stream 内串行执行顺序。
    status = gemm.initialize(args, s_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialization failed.");
    }

    // CUTLASS initialize(args, workspace, stream) 用 stream-ordered ops 设置 workspace:
    //   - Mainloop: no-op (TMA descriptor 嵌入 kernel args, 不写 workspace)
    //   - TileScheduler: cudaMemsetAsync on stream
    // 因此 gemm.run(stream) 在同一 stream 上保证顺序, 无需额外 sync.
    // 之前的 cudaDeviceSynchronize + cudaStreamSynchronize 在 T=1024 (312 GEMM)
    // 导致 ~1000ms 的流水线阻塞, 已验证可安全移除.

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM execution failed.");
    }

    if (M_padded != M) {
        cudaMemcpy2DAsync(C, (size_t)N * sizeof(__nv_bfloat16),
                          C_eff, (size_t)N * sizeof(__nv_bfloat16),
                          (size_t)N * sizeof(__nv_bfloat16), M,
                          cudaMemcpyDeviceToDevice, stream);
    }
}

// ============================================================================
// invoke_dense_gemm_add: CUTLASS SM110 BF16 GEMM + Residual Add
// D[M,N] = A[M,K] × B[K,N] + residual[M,N]
// Uses CUTLASS epilogue beta=1
// ============================================================================
void invoke_dense_gemm_add(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* D,
    const __nv_bfloat16* residual,
    int M, int N, int K,
    cudaStream_t stream
) {
    int M_padded = M;
    const __nv_bfloat16* A_eff = A;
    __nv_bfloat16* D_eff = D;
    const __nv_bfloat16* C_eff = residual;

    static __nv_bfloat16* s_A_pad2 = nullptr;
    static __nv_bfloat16* s_D_pad2 = nullptr;
    static size_t s_A_pad2_sz = 0;
    static size_t s_D_pad2_sz = 0;

    if (M % 8 != 0) {
        M_padded = ((M + 7) / 8) * 8;
        size_t a_need = (size_t)M_padded * K * sizeof(__nv_bfloat16);
        size_t d_need = (size_t)M_padded * N * sizeof(__nv_bfloat16);
        SAFE_CUDA_REALLOC(s_A_pad2, s_A_pad2_sz, a_need, stream);
        SAFE_CUDA_REALLOC(s_D_pad2, s_D_pad2_sz, d_need, stream);
        if (!s_A_pad2 || !s_D_pad2) return; // OOM guard
        cudaMemcpyAsync(s_A_pad2, A, (size_t)M * K * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(s_A_pad2 + (size_t)M * K, 0,
                        (size_t)(M_padded - M) * K * sizeof(__nv_bfloat16), stream);
        cudaMemcpyAsync(s_D_pad2, residual, (size_t)M * N * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(s_D_pad2 + (size_t)M * N, 0,
                        (size_t)(M_padded - M) * N * sizeof(__nv_bfloat16), stream);
        A_eff = s_A_pad2;
        D_eff = s_D_pad2;
        C_eff = s_D_pad2;
    } else if (D == residual) {
        C_eff = D;
        D_eff = D;
    }

    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;
    using ProblemShapeType = typename GemmType::GemmKernel::ProblemShape;

    ProblemShapeType problem_size = ProblemShapeType{M_padded, N, K, 1};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    static int s_cached_sm_count2 = -1;
    if (s_cached_sm_count2 < 0) {
        s_cached_sm_count2 = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    }
    hw_info.sm_count = s_cached_sm_count2;

    typename GemmType::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {reinterpret_cast<const ElementA*>(A_eff), cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M_padded, K, 1)),
         reinterpret_cast<const ElementB*>(B), cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1))},
        {{1.0f, 1.0f},
         reinterpret_cast<const ElementC*>(C_eff), cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M_padded, N, 1)),
         reinterpret_cast<ElementD*>(D_eff), cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M_padded, N, 1))},
        hw_info
    };

    GemmType gemm;

    static void* s_workspace2 = nullptr;
    static size_t s_workspace2_size = 0;
    size_t workspace_size = GemmType::get_workspace_size(args);
    SAFE_CUDA_REALLOC(s_workspace2, s_workspace2_size, workspace_size, stream);

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        // cuBLAS fallback: D[M,N] = A[M,K] × B[K,N] + residual[M,N]
        // cuBLAS col-major: C_cublas^T = B^T × A^T + C_cublas^T  (alpha=1, beta=1)
        // 先把 residual 拷到 D (作为 beta*C 的 C), 然后 GEMM alpha=1 beta=1
        auto h = get_cublas_handle();
        cublasSetStream(h, stream);
        if (D != residual) {
            cudaMemcpyAsync(D, residual, (size_t)M * N * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream);
        }
        float alpha = 1.0f, beta_val = 1.0f;
        cublasGemmEx(h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            B, CUDA_R_16BF, K,
            A, CUDA_R_16BF, K,
            &beta_val,
            D, CUDA_R_16BF, N,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
        return;
    }
    // 同 invoke_dense_gemm: 必须传 stream 给 initialize() 避免跨 stream 竞态
    status = gemm.initialize(args, s_workspace2, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM+Add initialization failed.");
    }

    // 同 invoke_dense_gemm: stream-ordered workspace init, 无需额外 sync

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM+Add execution failed.");
    }

    if (M_padded != M) {
        cudaMemcpy2DAsync(D, (size_t)N * sizeof(__nv_bfloat16),
                          D_eff, (size_t)N * sizeof(__nv_bfloat16),
                          (size_t)N * sizeof(__nv_bfloat16), M,
                          cudaMemcpyDeviceToDevice, stream);
    }
}

// ----------------------------------------------------------------------------
// GEMV (M=1) 高性能实现
// ----------------------------------------------------------------------------
// B 是 Column Major [K, N]，即 B(k, n) = B[n*K + k]
// A 是 [1, K] 向量， C 是 [1, N] 向量
//
// 优化策略：
// - 1 warp (32 threads) 协作产出 1 个输出元素，K 维并行 + warp shuffle reduce
// - A 向量加载到 shared memory，同 block 8 个 warp 共享
// - float4 向量化读取 (每次 16 bytes = 8 BF16)，B 列访问 coalesced
// - Grid: ceil(N / WARPS_PER_BLOCK)，Block: 256 threads (8 warps)
//
// 理论带宽利用: ~85%（受 DRAM 带宽限制）
// 主要数据读取: N*K*2 bytes (权重 B) + K*2 bytes (向量 A, shared/cached)

__global__ void gemv_kernel(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B,
                            __nv_bfloat16* __restrict__ C,
                            int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;  // 256 / 32

    extern __shared__ __nv_bfloat16 s_A[];  // [K] — dynamic shared memory

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // --- 协作加载 A 到 shared memory ---
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_A[i] = A[i];
    }
    __syncthreads();

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    // --- 主循环: float4 向量化 (每次读 8 个 BF16) ---
    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];     // shared memory (无 bank conflict: 128-bit)
        float4 b4 = b_col_v4[i];   // global memory, warp 内 coalesced

        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    // --- 标量尾部处理 (K % 8 != 0 的情况) ---
    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE) {
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);
    }

    // --- Warp shuffle reduce ---
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    if (lane_id == 0) {
        C[out_idx] = __float2bfloat16(sum);
    }
}

// ----------------------------------------------------------------------------
// K 分块 GEMV — 当 K 维度很大导致 shared memory 超过阈值时使用
// 将 A 向量分块加载到 shared memory，每个块只需 TILE_K * sizeof(BF16) 字节
// 典型场景: down_proj (K=17408) 需要 35 KB shared → 限制 SM 占用率
// 分块后每次只需 TILE_K*2 字节 (e.g. 4096*2 = 8 KB) → 更多 blocks/SM
// ----------------------------------------------------------------------------
__global__ void gemv_kernel_tiled(const __nv_bfloat16* __restrict__ A,
                                   const __nv_bfloat16* __restrict__ B,
                                   __nv_bfloat16* __restrict__ C,
                                   int N, int K, int tile_k)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];  // [tile_k]

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    for (int k_start = 0; k_start < K; k_start += tile_k) {
        int tile_end = min(k_start + tile_k, K);
        int tile_len = tile_end - k_start;

        // 协作加载 A[k_start..tile_end) 到 shared memory
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x) {
            s_A[i] = A[k_start + i];
        }
        __syncthreads();

        // float4 向量化主循环
        int t8 = tile_len / 8;
        const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
        const float4* b_tile_v4 = reinterpret_cast<const float4*>(b_col + k_start);

        for (int i = lane_id; i < t8; i += WARP_SIZE) {
            float4 a4 = s_A_v4[i];
            float4 b4 = b_tile_v4[i];
            const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 af = __bfloat1622float2(a2[j]);
                float2 bf = __bfloat1622float2(b2[j]);
                sum += af.x * bf.x + af.y * bf.y;
            }
        }

        // 尾部处理
        int tail_start = t8 * 8;
        for (int k = tail_start + lane_id; k < tile_len; k += WARP_SIZE) {
            sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k_start + k]);
        }
        __syncthreads();  // 下一轮 tile 前的同步
    }

    // Warp shuffle reduce
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    if (lane_id == 0) {
        C[out_idx] = __float2bfloat16(sum);
    }
}

// ----------------------------------------------------------------------------
// 散列映射 GEMV — 8 warps, 全 K 在 shared memory
// 适用于 K 较大但 K*2 <= 48KB (可放入 1 个 block 的 smem limit)
// 核心: out_idx = blockIdx.x + warp_id * num_blocks (散列映射)
// 同 block 的 8 warp 各访问 DRAM 中相距甚远的列, 降低 bank conflict
// 微基准: down_proj (N=5120, K=17408) 206.2 → 225.7 GB/s (+9.5%)
// 比 scattered_tiled (V3, 212.7 GB/s) 更快: 避免了多次 tile sync 开销
// ----------------------------------------------------------------------------
__global__ void gemv_kernel_scattered(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int out_idx = blockIdx.x + warp_id * num_blocks;

    // 协作加载 A → shared memory (一次性, 无 tiling)
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_A[i] = A[i];
    __syncthreads();

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];
        float4 b4 = b_col_v4[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE)
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum);
}

// ----------------------------------------------------------------------------
// 散列映射 + K 分块 GEMV — 针对 K 极大且 smem 超 block limit 的场景
// 核心优化思路:
//   1. 散列 warp 映射: out_idx = blockIdx.x + warp_id * num_blocks
//      → 同 block 的 warp 访问 DRAM 远端列, 避免 bank 冲突
//   2. K 分块: tile_k=4096 → smem=8KB → 可装入 block smem limit
// 保留用于 K*2 > 48KB 的极端情况 (当前模型不触发)
// ----------------------------------------------------------------------------
__global__ void gemv_kernel_scattered_tiled(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int N, int K, int tile_k)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 4;  // 128 threads

    extern __shared__ __nv_bfloat16 s_A[];  // [tile_k]

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int out_idx = blockIdx.x + warp_id * num_blocks;

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    for (int k_start = 0; k_start < K; k_start += tile_k) {
        int tile_end = min(k_start + tile_k, K);
        int tile_len = tile_end - k_start;

        // 协作加载 A → shared memory (128 threads)
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
            s_A[i] = A[k_start + i];
        __syncthreads();

        int t8 = tile_len / 8;
        const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
        const float4* b_tile_v4 = reinterpret_cast<const float4*>(b_col + k_start);

        for (int i = lane_id; i < t8; i += WARP_SIZE) {
            float4 a4 = s_A_v4[i];
            float4 b4 = b_tile_v4[i];
            const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 af = __bfloat1622float2(a2[j]);
                float2 bf = __bfloat1622float2(b2[j]);
                sum += af.x * bf.x + af.y * bf.y;
            }
        }

        int tail_start = t8 * 8;
        for (int k = tail_start + lane_id; k < tile_len; k += WARP_SIZE)
            sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k_start + k]);
        __syncthreads();
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum);
}

// ============================================================================
// GEMV + Residual Add 变体: C[i] = dot(A, B[:,i]) + residual[i]
// 融合 down_proj GEMV 和残差加法, 消除额外的 add kernel launch + 内存写读
// ============================================================================

// 散列映射 + 残差加法 GEMV — 8 warps, 全 K 在 shared memory
// 适用于 down_proj (K=17408): smem = 34 KB < 48 KB block limit
// 比 scattered_tiled_add 更快: 避免多次 tile sync 开销
// 微基准 (non-add variant 对比): 225.7 GB/s (scattered) vs 212.7 GB/s (scattered_tiled)
__global__ void gemv_kernel_scattered_add(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual,
    int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int out_idx = blockIdx.x + warp_id * num_blocks;

    // 协作加载 A → shared memory (一次性, 无 tiling)
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_A[i] = A[i];
    __syncthreads();

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];
        float4 b4 = b_col_v4[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE)
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum + __bfloat162float(residual[out_idx]));
}

__global__ void gemv_kernel_add(const __nv_bfloat16* __restrict__ A,
                                const __nv_bfloat16* __restrict__ B,
                                __nv_bfloat16* __restrict__ C,
                                const __nv_bfloat16* __restrict__ residual,
                                int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    for (int i = threadIdx.x; i < K; i += blockDim.x)
        s_A[i] = A[i];
    __syncthreads();

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;
    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];
        float4 b4 = b_col_v4[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }
    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE)
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum + __bfloat162float(residual[out_idx]));
}

__global__ void gemv_kernel_tiled_add(const __nv_bfloat16* __restrict__ A,
                                       const __nv_bfloat16* __restrict__ B,
                                       __nv_bfloat16* __restrict__ C,
                                       const __nv_bfloat16* __restrict__ residual,
                                       int N, int K, int tile_k)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    for (int k_start = 0; k_start < K; k_start += tile_k) {
        int tile_end = min(k_start + tile_k, K);
        int tile_len = tile_end - k_start;
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
            s_A[i] = A[k_start + i];
        __syncthreads();

        int t8 = tile_len / 8;
        const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
        const float4* b_tile_v4 = reinterpret_cast<const float4*>(b_col + k_start);
        for (int i = lane_id; i < t8; i += WARP_SIZE) {
            float4 a4 = s_A_v4[i];
            float4 b4 = b_tile_v4[i];
            const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 af = __bfloat1622float2(a2[j]);
                float2 bf = __bfloat1622float2(b2[j]);
                sum += af.x * bf.x + af.y * bf.y;
            }
        }
        int tail_start = t8 * 8;
        for (int k = tail_start + lane_id; k < tile_len; k += WARP_SIZE)
            sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k_start + k]);
        __syncthreads();
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum + __bfloat162float(residual[out_idx]));
}

__global__ void gemv_kernel_scattered_tiled_add(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    const __nv_bfloat16* __restrict__ residual,
    int N, int K, int tile_k)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int num_blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int out_idx = blockIdx.x + warp_id * num_blocks;

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    for (int k_start = 0; k_start < K; k_start += tile_k) {
        int tile_end = min(k_start + tile_k, K);
        int tile_len = tile_end - k_start;
        for (int i = threadIdx.x; i < tile_len; i += blockDim.x)
            s_A[i] = A[k_start + i];
        __syncthreads();

        int t8 = tile_len / 8;
        const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
        const float4* b_tile_v4 = reinterpret_cast<const float4*>(b_col + k_start);
        for (int i = lane_id; i < t8; i += WARP_SIZE) {
            float4 a4 = s_A_v4[i];
            float4 b4 = b_tile_v4[i];
            const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 af = __bfloat1622float2(a2[j]);
                float2 bf = __bfloat1622float2(b2[j]);
                sum += af.x * bf.x + af.y * bf.y;
            }
        }
        int tail_start = t8 * 8;
        for (int k = tail_start + lane_id; k < tile_len; k += WARP_SIZE)
            sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k_start + k]);
        __syncthreads();
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    if (lane_id == 0)
        C[out_idx] = __float2bfloat16(sum + __bfloat162float(residual[out_idx]));
}

// ----------------------------------------------------------------------------
// Fused RMSNorm + GEMV: Input RMSNorm 在 SMEM 内完成后直接开始 GEMV
// 省去 norm_out 的 GMEM write+read + 1 kernel launch
// RMSNorm 使用 centered weight: out = x * rsqrt(var+eps) * (1+w)
// 限制: K <= 24576 (smem 48KB limit / 2 bytes) 且 M=1 (T=1 decode only)
// 线程布局: Grid(ceil(N/8)), Block(256 = 8 warps), Dynamic SMEM = K*sizeof(BF16)
// ----------------------------------------------------------------------------
__global__ void gemv_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const __nv_bfloat16* __restrict__ norm_weight,
    float eps,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];  // [K] will hold normalized hidden states

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);
    int out_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    // Phase 1: Load hidden_states → SMEM + compute sum-of-squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        __nv_bfloat16 val = hidden_states[i];
        s_A[i] = val;
        float fv = __bfloat162float(val);
        sum_sq += fv * fv;
    }
    sum_sq = gemv_blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)K + eps);
    }
    __syncthreads();

    // Phase 2: Apply RMSNorm in-place in SMEM
    float inv_rms = s_inv_rms;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(s_A[i]);
        float w = __bfloat162float(norm_weight[i]);
        s_A[i] = __float2bfloat16(v * inv_rms * (1.0f + w));
    }
    __syncthreads();

    // Phase 3: Standard GEMV from normalized SMEM
    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = B + (size_t)out_idx * K;
    float sum = 0.0f;

    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];
        float4 b4 = b_col_v4[i];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE) {
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    if (lane_id == 0) {
        C[out_idx] = __float2bfloat16(sum);
    }
}

void invoke_dense_gemv_with_rmsnorm(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* norm_weight,
    float eps,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int N, int K,
    cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    size_t smem_bytes = K * sizeof(__nv_bfloat16);
    gemv_rmsnorm_kernel<<<blocks, BLOCK_THREADS, smem_bytes, stream>>>(
        hidden_states, norm_weight, eps, B, C, N, K);
}

void invoke_dense_gemv(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS_8W = 256;  // 8 warps
    constexpr int WARPS_8W = BLOCK_THREADS_8W / 32;
    constexpr int BLOCK_THREADS_4W = 128;  // 4 warps
    constexpr int WARPS_4W = BLOCK_THREADS_4W / 32;
    size_t smem_bytes = K * sizeof(__nv_bfloat16);

    // SM110 Thor: 228 KB shared/SM, 48 KB/block 硬件上限
    constexpr size_t SMEM_BLOCK_LIMIT = 48 * 1024;  // 48 KB = Thor hardware limit per block
    constexpr int SCATTER_K_THRESHOLD = 8192;
    constexpr int TILE_K = 4096;

    if (smem_bytes <= SMEM_BLOCK_LIMIT) {
        // 全 K 放入 shared memory: 优先散列映射 (8 warps, 无 tiling 开销)
        // 微基准 (K=17408): scattered 225.7 GB/s vs scattered_tiled 212.7 GB/s (+6.1%)
        int blocks = (N + WARPS_8W - 1) / WARPS_8W;
        if (K > SCATTER_K_THRESHOLD) {
            gemv_kernel_scattered<<<blocks, BLOCK_THREADS_8W, smem_bytes, stream>>>(
                A, B, C, N, K);
        } else {
            gemv_kernel<<<blocks, BLOCK_THREADS_8W, smem_bytes, stream>>>(A, B, C, N, K);
        }
    } else {
        // K 太大无法放入单 block smem → 散列 + K 分块 (4 warps)
        int blocks = (N + WARPS_4W - 1) / WARPS_4W;
        size_t tiled_smem = TILE_K * sizeof(__nv_bfloat16);
        gemv_kernel_scattered_tiled<<<blocks, BLOCK_THREADS_4W, tiled_smem, stream>>>(
            A, B, C, N, K, TILE_K);
    }
}

void invoke_dense_gemv_add(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    const __nv_bfloat16* residual,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS_8W = 256;
    constexpr int WARPS_8W = BLOCK_THREADS_8W / 32;
    constexpr int BLOCK_THREADS_4W = 128;
    constexpr int WARPS_4W = BLOCK_THREADS_4W / 32;
    size_t smem_bytes = K * sizeof(__nv_bfloat16);

    constexpr size_t SMEM_BLOCK_LIMIT = 48 * 1024;
    constexpr int SCATTER_K_THRESHOLD = 8192;
    constexpr int TILE_K = 4096;

    if (K > SCATTER_K_THRESHOLD) {
        // K > 8192: 散列 + K 分块 (4 warps, tile_k=4096, smem=8KB)
        // 实测比 scattered_add (8 warps, 34KB smem) 更快:
        //   12 blocks/SM vs 6 blocks/SM → 更好的调度 + 更大 L1
        int blocks = (N + WARPS_4W - 1) / WARPS_4W;
        size_t tiled_smem = TILE_K * sizeof(__nv_bfloat16);
        gemv_kernel_scattered_tiled_add<<<blocks, BLOCK_THREADS_4W, tiled_smem, stream>>>(
            A, B, C, residual, N, K, TILE_K);
    } else if (smem_bytes > SMEM_BLOCK_LIMIT) {
        int blocks = (N + WARPS_8W - 1) / WARPS_8W;
        int tile_k = 4096;
        size_t tiled_smem = tile_k * sizeof(__nv_bfloat16);
        gemv_kernel_tiled_add<<<blocks, BLOCK_THREADS_8W, tiled_smem, stream>>>(
            A, B, C, residual, N, K, tile_k);
    } else {
        int blocks = (N + WARPS_8W - 1) / WARPS_8W;
        gemv_kernel_add<<<blocks, BLOCK_THREADS_8W, smem_bytes, stream>>>(
            A, B, C, residual, N, K);
    }
}

// ----------------------------------------------------------------------------
// Dual-output GEMV: C1 = A × B1,  C2 = A × B2
// A 只加载一次到 shared memory, grid 前半负责 B1/C1, 后半负责 B2/C2
// 节省: 1 次 kernel launch + 1 次 A 向量 DRAM 读取
// ----------------------------------------------------------------------------
__global__ void dual_gemv_kernel(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B1,
                                  const __nv_bfloat16* __restrict__ B2,
                                  __nv_bfloat16* __restrict__ C1,
                                  __nv_bfloat16* __restrict__ C2,
                                  int N, int K)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = 8;

    extern __shared__ __nv_bfloat16 s_A[];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x & (WARP_SIZE - 1);

    // Grid 前半 (blockIdx.x < blocks_per_output) → B1/C1
    // Grid 后半 → B2/C2
    int blocks_per_output = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    bool is_second = (blockIdx.x >= blocks_per_output);
    int local_block = is_second ? (blockIdx.x - blocks_per_output) : blockIdx.x;
    int out_idx = local_block * WARPS_PER_BLOCK + warp_id;

    // 协作加载 A 到 shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        s_A[i] = A[i];
    }
    __syncthreads();

    if (out_idx >= N) return;

    const __nv_bfloat16* b_col = is_second
        ? (B2 + (size_t)out_idx * K)
        : (B1 + (size_t)out_idx * K);
    float sum = 0.0f;

    int k8 = K / 8;
    const float4* s_A_v4 = reinterpret_cast<const float4*>(s_A);
    const float4* b_col_v4 = reinterpret_cast<const float4*>(b_col);

    for (int i = lane_id; i < k8; i += WARP_SIZE) {
        float4 a4 = s_A_v4[i];
        float4 b4 = b_col_v4[i];

        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            sum += af.x * bf.x + af.y * bf.y;
        }
    }

    int k_tail_start = k8 * 8;
    for (int k = k_tail_start + lane_id; k < K; k += WARP_SIZE) {
        sum += __bfloat162float(s_A[k]) * __bfloat162float(b_col[k]);
    }

    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    if (lane_id == 0) {
        __nv_bfloat16* out = is_second ? C2 : C1;
        out[out_idx] = __float2bfloat16(sum);
    }
}

void invoke_dense_dual_gemv(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B1,
    const __nv_bfloat16* B2,
    __nv_bfloat16* C1,
    __nv_bfloat16* C2,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    int blocks_per_output = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int total_blocks = blocks_per_output * 2;  // 前半 B1, 后半 B2
    size_t smem_bytes = K * sizeof(__nv_bfloat16);

    dual_gemv_kernel<<<total_blocks, BLOCK_THREADS, smem_bytes, stream>>>(
        A, B1, B2, C1, C2, N, K);
}

} // namespace ops
} // namespace qwen_thor
