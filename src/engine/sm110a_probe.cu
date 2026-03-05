// =============================================================================
// SM110a Hardware Primitives — Statistically Rigorous Micro-benchmark Suite
//
// Level 0: 实测 TMEM / UMMA / TMA / DRAM BW / L2 / Memory Latency
//          + UMMA TS / UTCCP / pack / L1::no_alloc / f32x2 / LDSM / Fence / PDL
//          在 SM110a 上的真实表现
//
// 统计方法: 每项测试 NUM_TRIALS 次 (默认 20), 报告:
//   min / P5 / median / mean / P95 / max / stddev / CV(%)
//   > CV < 2%:  极高可信度 (★★★), 可直接用于优化决策
//   > CV < 5%:  高可信度   (★★☆), 适合作为基准
//   > CV < 15%: 中等可信度 (★☆☆), 需结合其他数据
//   > CV ≥ 15%: 低可信度   (☆☆☆), 可能有系统干扰
//
// 运行: ./build/qwen3-27b-thor probe
//
// 架构: Jetson AGX Thor, SM110a (Blackwell), 20 SM, 128 GB LPDDR5X 统一内存
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cstring>

// CUTLASS CuTe headers for TMEM/UMMA/TMA
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>       // TMEM load/store
#include <cute/arch/copy_sm90_tma.hpp>    // TMA LOAD/STORE, BULK_COPY G2S/S2G
#include <cute/arch/mma_sm100_umma.hpp>   // UMMA SS/TS
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/mma_sm100_desc.hpp>   // SmemDescriptor
#include <cutlass/arch/barrier.h>         // ClusterBarrier, ClusterTransactionBarrier

using namespace cute;

namespace sm110a_probe {

// =============================================================================
// Constants
// =============================================================================
static constexpr int NUM_SM       = 20;
static constexpr int NUM_TRIALS   = 20;   // 统计显著性: 20 次采样
static constexpr int WARMUP_RUNS  = 5;    // 预热次数 (不计入统计)
static constexpr float CLOCK_GHZ  = 1.575f; // GPC clock (base)
static constexpr float PEAK_BW_GBS = 273.0f; // 理论 DRAM 峰值 (4266 MT/s × 256-bit / 8)

// =============================================================================
// Error checking macro
// =============================================================================
#define PROBE_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("[SM110a probe] CUDA error %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// =============================================================================
// Statistical utilities
// =============================================================================
struct ProbeStats {
    float min, p5, median, mean, p95, max;
    float stddev, cv;   // cv = coefficient of variation (stddev/mean × 100%)
    int   n;
};

static ProbeStats compute_stats(std::vector<float>& samples) {
    ProbeStats s{};
    s.n = (int)samples.size();
    if (s.n == 0) return s;

    std::sort(samples.begin(), samples.end());
    s.min    = samples.front();
    s.max    = samples.back();
    s.median = (s.n & 1) ? samples[s.n / 2]
                         : (samples[s.n / 2 - 1] + samples[s.n / 2]) * 0.5f;
    // P5, P95 (nearest-rank method)
    s.p5  = samples[std::max(0, (int)(s.n * 0.05f))];
    s.p95 = samples[std::min(s.n - 1, (int)(s.n * 0.95f))];

    double sum = 0;
    for (float v : samples) sum += v;
    s.mean = (float)(sum / s.n);

    double var = 0;
    for (float v : samples) var += (v - s.mean) * (v - s.mean);
    s.stddev = (float)std::sqrt(var / s.n);  // population stddev
    s.cv     = (s.mean > 1e-9f) ? (s.stddev / s.mean * 100.0f) : 0.0f;

    return s;
}

// Confidence rating based on CV
static const char* cv_rating(float cv) {
    if (cv < 2.0f)  return "\xe2\x98\x85\xe2\x98\x85\xe2\x98\x85";   // ★★★
    if (cv < 5.0f)  return "\xe2\x98\x85\xe2\x98\x85\xe2\x98\x86";   // ★★☆
    if (cv < 15.0f) return "\xe2\x98\x85\xe2\x98\x86\xe2\x98\x86";   // ★☆☆
    return "\xe2\x98\x86\xe2\x98\x86\xe2\x98\x86";                    // ☆☆☆
}

static void print_stats_line(const char* label, ProbeStats& s, const char* unit) {
    printf("  %-22s  mean=%8.1f  med=%8.1f  std=%6.1f  CV=%5.1f%% %s  "
           "[%7.1f .. %7.1f .. %7.1f] %s  (n=%d)\n",
           label, s.mean, s.median, s.stddev, s.cv, cv_rating(s.cv),
           s.p5, s.median, s.p95, unit, s.n);
}

static void print_stats_bw(const char* label, ProbeStats& s) {
    printf("  %-22s  mean=%8.1f  med=%8.1f  std=%6.1f  CV=%5.1f%% %s  "
           "[%7.1f .. %7.1f .. %7.1f] GB/s  (%4.0f%% peak)\n",
           label, s.mean, s.median, s.stddev, s.cv, cv_rating(s.cv),
           s.p5, s.median, s.p95, s.mean / PEAK_BW_GBS * 100.0f);
}

// =============================================================================
// =====================  DEVICE KERNELS  =====================================
// =============================================================================

// =============================================================================
// K1. TMEM Lifecycle Kernel
// =============================================================================
__global__ void tmem_probe_kernel(
    float* __restrict__ results,   // [8] output timings
    int num_columns)
{
    __shared__ uint32_t tmem_addr_smem;
    __shared__ uint64_t clocks[8];

    int tid  = threadIdx.x;
    int warp = tid >> 5;

    __syncthreads();

    // Allocate
    long long t0, t1;
    if (warp == 0) {
        t0 = clock64();
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        {
            uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(&tmem_addr_smem);
            asm volatile(
                "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                : : "r"(smem_ptr), "r"(num_columns));
        }
#endif
        t1 = clock64();
        if ((tid & 31) == 0) clocks[0] = t1 - t0;
    }
    __syncthreads();

    uint32_t tmem_base = tmem_addr_smem;

    // Store (registers -> TMEM)
    {
        long long ts = clock64();
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        for (int col = 0; col < num_columns; col++) {
            uint32_t addr = tmem_base + col;
            float val = (float)(tid * 1000 + col);
            uint32_t val_bits;
            memcpy(&val_bits, &val, 4);
            asm volatile(
                "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                : : "r"(addr), "r"(val_bits));
        }
#endif
        long long te = clock64();
        if (tid == 0) clocks[1] = te - ts;
    }
    __syncthreads();

    // Load (TMEM -> registers) + verify
    {
        int errors = 0;
        long long ts = clock64();
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        for (int col = 0; col < num_columns; col++) {
            uint32_t addr = tmem_base + col;
            uint32_t loaded_bits;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
                : "=r"(loaded_bits) : "r"(addr));
            float loaded_val, expected = (float)(tid * 1000 + col);
            memcpy(&loaded_val, &loaded_bits, 4);
            if (fabsf(loaded_val - expected) > 0.001f) errors++;
        }
#endif
        long long te = clock64();
        if (tid == 0) clocks[2] = te - ts;

        __shared__ int total_errors;
        if (tid == 0) total_errors = 0;
        __syncthreads();
        if (errors > 0) atomicAdd(&total_errors, errors);
        __syncthreads();
        if (tid == 0) clocks[3] = total_errors;
    }

    // Deallocate
    __syncthreads();
    if (warp == 0) {
        long long ts = clock64();
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : : "r"(tmem_base), "r"(num_columns));
#endif
        long long te = clock64();
        if ((tid & 31) == 0) clocks[6] = te - ts;
    }
    __syncthreads();

    if (tid == 0) {
        results[0] = (float)clocks[0];  // alloc cycles
        results[1] = (float)clocks[1];  // store total cycles
        results[2] = (float)clocks[2];  // load total cycles
        results[3] = (float)clocks[3];  // verification errors
        results[6] = (float)clocks[6];  // dealloc cycles
        results[7] = (float)num_columns;
    }
}

// =============================================================================
// K2. UMMA SS Throughput Kernel
// =============================================================================
__global__ void umma_ss_probe_kernel(
    float* __restrict__ results,
    int num_iterations)
{
    __shared__ uint32_t tmem_c_addr_smem;
    int tid  = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* A_smem = (__nv_bfloat16*)smem_raw;
    __nv_bfloat16* B_smem = A_smem + 64 * 16;

    for (int i = tid; i < 64 * 16; i += blockDim.x) {
        A_smem[i] = __float2bfloat16(1.0f);
        B_smem[i] = __float2bfloat16(1.0f);
    }
    __syncthreads();

    if (warp == 0) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(&tmem_c_addr_smem);
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            : : "r"(smem_ptr), "r"(64));
#endif
    }
    __syncthreads();

    uint32_t tmem_c = tmem_c_addr_smem;

#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    UMMA::SmemDescriptor desc_a_val, desc_b_val;
    desc_a_val.desc_ = 0;
    desc_b_val.desc_ = 0;

    uint32_t a_smem_uint = cute::cast_smem_ptr_to_uint(A_smem);
    uint32_t b_smem_uint = cute::cast_smem_ptr_to_uint(B_smem);

    desc_a_val.start_address_ = a_smem_uint >> 4;
    desc_a_val.leading_byte_offset_ = 128 >> 4;
    desc_a_val.stride_byte_offset_ = 128 >> 4;
    desc_a_val.layout_type_ = 0;

    desc_b_val.start_address_ = b_smem_uint >> 4;
    desc_b_val.leading_byte_offset_ = 128 >> 4;
    desc_b_val.stride_byte_offset_ = 128 >> 4;
    desc_b_val.layout_type_ = 0;

    uint64_t desc_a = desc_a_val.desc_;
    uint64_t desc_b = desc_b_val.desc_;

    UMMA::InstrDescriptor idesc_i = UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        64, 64, UMMA::Major::K, UMMA::Major::K>();
    uint64_t idescE = UMMA::make_runtime_instr_desc<>(idesc_i);

    // Warmup
    for (int i = 0; i < 10; i++) {
        if (cute::elect_one_sync()) {
            uint32_t mask[4] = {0, 0, 0, 0};
            uint32_t scaleC = (i > 0) ? 1u : 0u;
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                "}\n"
                : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                    "r"(uint32_t(idescE >> 32)), "r"(scaleC),
                    "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();

    // Timed loop
    long long t_start = clock64();
    for (int i = 0; i < num_iterations; i++) {
        if (cute::elect_one_sync()) {
            uint32_t mask[4] = {0, 0, 0, 0};
            uint32_t scaleC = 1u;
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"
                "}\n"
                : : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
                    "r"(uint32_t(idescE >> 32)), "r"(scaleC),
                    "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();
    long long t_end = clock64();
#endif // CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED

    float verify_val = 0.0f;
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    {
        uint32_t val_bits;
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
            : "=r"(val_bits) : "r"(tmem_c));
        memcpy(&verify_val, &val_bits, 4);
    }
#endif

    if (tid == 0) {
#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
        long long elapsed = t_end - t_start;
        double total_flops = (double)num_iterations * 64.0 * 64.0 * 16.0 * 2.0;
        double seconds = (double)elapsed / (1.575e9);
        double tflops = total_flops / seconds / 1e12;
        results[0] = (float)elapsed;
        results[1] = (float)tflops;
        results[2] = verify_val;
        results[3] = (float)num_iterations;
#else
        results[0] = -1.0f;
#endif
    }

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    __syncthreads();
    if (warp == 0) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            : : "r"(tmem_c), "r"(64));
    }
#endif
}

// =============================================================================
// K3. DRAM Read BW Kernel (float4 vectorized)
// =============================================================================
__global__ void dram_read_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    int num_float4, int iterations)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = gid; i < num_float4; i += stride) {
            float4 v = data[i];
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
    }
    if (gid == 0) sink[0] = acc.x + acc.y + acc.z + acc.w;
}

// =============================================================================
// K4. DRAM Write BW Kernel (float4 vectorized)
// =============================================================================
__global__ void dram_write_kernel(
    float4* __restrict__ data,
    int num_float4, int iterations)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 val = make_float4((float)gid, 1.0f, 2.0f, 3.0f);
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = gid; i < num_float4; i += stride) {
            data[i] = val;
        }
    }
}

// =============================================================================
// K5. SMEM Bandwidth Kernel
// =============================================================================
__global__ void smem_bw_kernel(float* __restrict__ results, int iterations)
{
    extern __shared__ char smem_raw[];
    float* smem_f = reinterpret_cast<float*>(smem_raw);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    for (int i = tid; i < 4096; i += bs) smem_f[i] = (float)i;
    __syncthreads();

    long long t0 = clock64();
    float acc = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < 4096; i += bs) acc += smem_f[i];
    }
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = acc;
        results[2] = (float)iterations;
    }
}

// =============================================================================
// K6. TMA Load Probe (cp.async.bulk GMEM->SMEM)
// =============================================================================
__global__ void __launch_bounds__(128)
tma_load_probe_kernel(
    const void* __restrict__ gmem_src,
    float* __restrict__ results,
    int tile_bytes, int iterations)
{
    extern __shared__ char smem_raw[];
    int mbar_offset = (tile_bytes + 7) & ~7;
    auto* mbar = reinterpret_cast<uint64_t*>(smem_raw + mbar_offset);
    int tid = threadIdx.x;

    if (tid == 0) cutlass::arch::ClusterBarrier::init(mbar, 1);
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    __syncthreads();

    uint32_t phase = 0;

    // Warmup
    for (int i = 0; i < 5; i++) {
        if (tid == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(mbar, (uint32_t)tile_bytes);
            cute::SM90_BULK_COPY_G2S::copy(gmem_src, mbar, smem_raw, tile_bytes);
        }
        cutlass::arch::ClusterBarrier::wait(mbar, phase);
        phase ^= 1;
    }
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < iterations; i++) {
        if (tid == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(mbar, (uint32_t)tile_bytes);
            cute::SM90_BULK_COPY_G2S::copy(gmem_src, mbar, smem_raw, tile_bytes);
        }
        cutlass::arch::ClusterBarrier::wait(mbar, phase);
        phase ^= 1;
    }
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)iterations;
    }
}

// =============================================================================
// K7. TMA Store Probe (cp.async.bulk SMEM->GMEM)
// =============================================================================
__global__ void __launch_bounds__(128)
tma_store_probe_kernel(
    void* __restrict__ gmem_dst,
    float* __restrict__ results,
    int tile_bytes, int iterations)
{
    extern __shared__ char smem_raw[];
    int tid = threadIdx.x;

    for (int i = tid * 4; i < tile_bytes; i += blockDim.x * 4) {
        if (i + 4 <= tile_bytes) *reinterpret_cast<int*>(smem_raw + i) = i;
    }
    __syncthreads();

    for (int i = 0; i < 5; i++) {
        if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            cute::SM90_BULK_COPY_S2G::copy(smem_raw, gmem_dst, tile_bytes);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
        __syncthreads();
    }

    __syncthreads();
    long long t0 = clock64();
    for (int i = 0; i < iterations; i++) {
        if (tid == 0) {
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            cute::SM90_BULK_COPY_S2G::copy(smem_raw, gmem_dst, tile_bytes);
            cute::tma_store_arrive();
            cute::tma_store_wait<0>();
        }
    }
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)iterations;
    }
}

// =============================================================================
// K8. Manual Cooperative Copy Probe (baseline for TMA comparison)
// =============================================================================
__global__ void __launch_bounds__(128)
manual_copy_probe_kernel(
    const float4* __restrict__ gmem_src,
    float* __restrict__ results,
    int tile_float4s, int iterations)
{
    extern __shared__ char smem_raw[];
    float4* smem_f4 = reinterpret_cast<float4*>(smem_raw);
    int tid = threadIdx.x;
    int bs = blockDim.x;

    for (int w = 0; w < 5; w++) {
        for (int i = tid; i < tile_float4s; i += bs) smem_f4[i] = gmem_src[i];
        __syncthreads();
    }

    long long t0 = clock64();
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < tile_float4s; i += bs) smem_f4[i] = gmem_src[i];
        __syncthreads();
    }
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)iterations;
    }
}

// =============================================================================
// K9. Memory Latency Probe (pointer chasing)
//
// Builds a randomized linked list in GMEM, traverses it serially from 1 thread.
// Latency = cycles per chase step, measuring L1/L2/DRAM latency tiers.
// =============================================================================
__global__ void mem_latency_kernel(
    const int* __restrict__ chain,
    float* __restrict__ results,
    int steps)
{
    int idx = 0;
    // Warmup
    for (int i = 0; i < 1000; i++) idx = chain[idx];

    long long t0 = clock64();
    for (int i = 0; i < steps; i++) {
        idx = chain[idx];
    }
    long long t1 = clock64();

    // Prevent dead code elimination
    if (idx == -999999) results[2] = (float)idx;

    results[0] = (float)(t1 - t0);
    results[1] = (float)steps;
}

// =============================================================================
// K10. DRAM Read Kernel — Strided access pattern
//
// Each thread reads with a stride of `stride_float4` between accesses,
// simulating non-contiguous memory access patterns (e.g., gather/scatter).
// =============================================================================
__global__ void dram_strided_read_kernel(
    const float4* __restrict__ data,
    float* __restrict__ sink,
    int total_float4, int stride_float4, int read_count)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

    // Each thread reads `read_count` elements, spaced by stride_float4
    int start = gid * stride_float4;
    for (int i = 0; i < read_count; i++) {
        int idx = (start + i * stride_float4) % total_float4;
        float4 v = data[idx];
        acc.x += v.x; acc.y += v.y;
    }

    if (gid == 0) sink[0] = acc.x + acc.y;
}


// =============================================================================
// K12. UMMA TS (A=TMEM, B=SMEM) Throughput
//
// Compare against SS (K2) with same tile [M=64, K=16]×[K=16, N=64] BF16→FP32.
// TS sources A from TMEM instead of SMEM descriptor.
// =============================================================================
__global__ void umma_ts_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    __shared__ uint32_t tmem_addr_smem;
    int tid  = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* B_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // B[K=16, N=64] in SMEM
    for (int i = tid; i < 16 * 64; i += blockDim.x)
        B_smem[i] = __float2bfloat16(1.0f);
    __syncthreads();

    // Allocate 128 TMEM cols: [0..63] for A data, [64..127] for C accumulator
    if (warp == 0) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
        uint32_t sp = cute::cast_smem_ptr_to_uint(&tmem_addr_smem);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(sp), "r"(128));
#endif
    }
    __syncthreads();

    uint32_t tmem_base = tmem_addr_smem;
    uint32_t tmem_a = tmem_base;
    uint32_t tmem_c = tmem_base + 64;

    // Store A data into TMEM (BF16 packed as uint32)
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    for (int col = 0; col < 16; col++) {
        uint32_t addr = tmem_a + col;
        uint32_t packed = 0x3F803F80u; // two bfloat16(1.0)
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                     : : "r"(addr), "r"(packed));
    }
#endif
    __syncthreads();

#if defined(CUTE_ARCH_TCGEN05_F16F32_MMA_ENABLED)
    UMMA::SmemDescriptor desc_b_val;
    desc_b_val.desc_ = 0;
    uint32_t b_smem = cute::cast_smem_ptr_to_uint(B_smem);
    desc_b_val.start_address_ = b_smem >> 4;
    desc_b_val.leading_byte_offset_ = 128 >> 4;
    desc_b_val.stride_byte_offset_ = 128 >> 4;
    desc_b_val.layout_type_ = 0;
    uint64_t desc_b = desc_b_val.desc_;

    UMMA::InstrDescriptor idesc = UMMA::make_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float,
        64, 64, UMMA::Major::K, UMMA::Major::K>();
    uint64_t idescE = UMMA::make_runtime_instr_desc<>(idesc);

    // Warmup (TS mode: [tmem_c], [tmem_a], desc_b)
    for (int i = 0; i < 10; i++) {
        if (cute::elect_one_sync()) {
            uint32_t mask[4] = {0, 0, 0, 0};
            uint32_t scaleC = (i > 0) ? 1u : 0u;
            asm volatile(
                "{\n\t.reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
                "{%5, %6, %7, %8}, p;\n\t}\n"
                : : "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
                    "r"(uint32_t(idescE >> 32)), "r"(scaleC),
                    "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++) {
        if (cute::elect_one_sync()) {
            uint32_t mask[4] = {0, 0, 0, 0};
            uint32_t scaleC = 1u;
            asm volatile(
                "{\n\t.reg .pred p;\n\t"
                "setp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
                "{%5, %6, %7, %8}, p;\n\t}\n"
                : : "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
                    "r"(uint32_t(idescE >> 32)), "r"(scaleC),
                    "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();
    long long t1 = clock64();

    if (tid == 0) {
        long long elapsed = t1 - t0;
        double total_flops = (double)num_iterations * 64.0 * 64.0 * 16.0 * 2.0;
        double seconds = (double)elapsed / (CLOCK_GHZ * 1e9);
        results[0] = (float)elapsed;
        results[1] = (float)(total_flops / seconds / 1e12);
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(128));
    }
#endif
}

// =============================================================================
// K13. UTCCP (SMEM→TMEM Copy) Latency
//
// 128x256b mode: 128 DPs × 256 bits = 4 KB per op
// 4x256b mode:     4 DPs × 256 bits = 128 B per op
// =============================================================================
template <int Mode>  // 0 = 128x256b, 1 = 4x256b
__global__ void utccp_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    __shared__ uint32_t tmem_addr_smem;
    int tid  = threadIdx.x;
    int warp = tid >> 5;

    extern __shared__ char smem_raw[];
    float* smem_f = reinterpret_cast<float*>(smem_raw);
    for (int i = tid; i < 1024; i += blockDim.x) smem_f[i] = (float)i;
    __syncthreads();

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    if (warp == 0) {
        uint32_t sp = cute::cast_smem_ptr_to_uint(&tmem_addr_smem);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(sp), "r"(128));
    }
    __syncthreads();
    uint32_t tmem_dst = tmem_addr_smem;

    UMMA::SmemDescriptor desc;
    desc.desc_ = 0;
    uint32_t su = cute::cast_smem_ptr_to_uint(smem_f);
    desc.start_address_ = su >> 4;
    desc.leading_byte_offset_ = 128 >> 4;
    desc.stride_byte_offset_ = 128 >> 4;
    desc.layout_type_ = 0;
    uint64_t src_desc = desc.desc_;

    for (int i = 0; i < 5; i++) {
        if (cute::elect_one_sync()) {
            if constexpr (Mode == 0)
                asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                             : : "r"(tmem_dst), "l"(src_desc));
            else
                asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;"
                             : : "r"(tmem_dst), "l"(src_desc));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++) {
        if (cute::elect_one_sync()) {
            if constexpr (Mode == 0)
                asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;"
                             : : "r"(tmem_dst), "l"(src_desc));
            else
                asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;"
                             : : "r"(tmem_dst), "l"(src_desc));
        }
    }
#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
#endif
    __syncthreads();
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)num_iterations;
    }

    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_dst), "r"(128));
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

// =============================================================================
// K14. TMEM pack::16b (FP32→BF16) Load / unpack::16b (BF16→FP32) Store
//
// Compare: normal 32x32b.x4 vs 16x256b.x1.pack::16b (both give 4 regs)
// =============================================================================
__global__ void tmem_pack_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    __shared__ uint32_t tmem_addr_smem;
    int tid  = threadIdx.x;
    int warp = tid >> 5;

#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    if (warp == 0) {
        uint32_t sp = cute::cast_smem_ptr_to_uint(&tmem_addr_smem);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(sp), "r"(128));
    }
    __syncthreads();
    uint32_t tmem_base = tmem_addr_smem;

    // Fill 128 TMEM columns with FP32 data
    for (int col = 0; col < 128; col++) {
        uint32_t addr = tmem_base + col;
        float val = (float)(tid * 128 + col);
        uint32_t bits;
        memcpy(&bits, &val, 4);
        asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
                     : : "r"(addr), "r"(bits));
    }
    __syncthreads();

    uint32_t r0, r1, r2, r3;

    // ---- Normal load: 32x32b.x4 ----
    for (int i = 0; i < 10; i++)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_base));
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(tmem_base));
    __syncthreads();
    long long t1 = clock64();

    // ---- Pack load: 16x256b.x1.pack::16b ----
    uint32_t p0, p1, p2, p3;
    for (int i = 0; i < 10; i++)
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3) : "r"(tmem_base));
    __syncthreads();

    long long t2 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.pack::16b.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(p0), "=r"(p1), "=r"(p2), "=r"(p3) : "r"(tmem_base));
    __syncthreads();
    long long t3 = clock64();

    // ---- Normal store: 32x32b.x4 ----
    for (int i = 0; i < 10; i++)
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_base), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    __syncthreads();

    long long t4 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_base), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
    __syncthreads();
    long long t5 = clock64();

    // ---- Unpack store: 16x256b.x1.unpack::16b ----
    for (int i = 0; i < 10; i++)
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_base), "r"(p0), "r"(p1), "r"(p2), "r"(p3));
    __syncthreads();

    long long t6 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.st.sync.aligned.16x256b.x1.unpack::16b.b32 [%0], {%1,%2,%3,%4};"
                     : : "r"(tmem_base), "r"(p0), "r"(p1), "r"(p2), "r"(p3));
    __syncthreads();
    long long t7 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);  // normal load
        results[1] = (float)(t3 - t2);  // pack load
        results[2] = (float)(t5 - t4);  // normal store
        results[3] = (float)(t7 - t6);  // unpack store
        results[4] = (float)num_iterations;
    }

    __syncthreads();
    if (warp == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(tmem_base), "r"(128));
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

// =============================================================================
// K15. L1::no_allocate DRAM Read
//
// 256-bit (v8.f32) global load with L1 cache bypass for streaming patterns.
// =============================================================================
__global__ void dram_read_noalloc_kernel(
    const float* __restrict__ data,
    float* __restrict__ sink,
    int num_float8, int iterations)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float acc = 0.0f;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = gid; i < num_float8; i += stride) {
            const float* p = data + i * 8;
#if defined(CUTE_ARCH_LOAD256_SM100A_ENABLED)
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile(
                "ld.global.L1::no_allocate.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
                  "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
                : "l"(p));
            float f0; memcpy(&f0, &r0, 4);
            acc += f0;
#else
            const float4* p4 = reinterpret_cast<const float4*>(p);
            float4 a = p4[0]; float4 b = p4[1];
            acc += a.x + b.x;
#endif
        }
    }
    if (gid == 0) sink[0] = acc;
}

// =============================================================================
// K16. f32x2 SIMD FMA Throughput
//
// Compare fma.rn.f32x2 (2 FMAs/instr) vs fma.rn.f32 (1 FMA/instr)
// =============================================================================
__global__ void f32x2_fma_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;

    // ---- Scalar FMA ----
    float sa = 1.0001f, sb = 0.9999f, sc = 0.0f;
    for (int i = 0; i < 10; i++)
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sc) : "f"(sa), "f"(sb), "f"(sc));
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(sc) : "f"(sa), "f"(sb), "f"(sc));
    __syncthreads();
    long long t1 = clock64();

    // ---- f32x2 FMA ----
    uint64_t va, vb, vc;
    {
        uint32_t ai1, ai2, bi1, bi2;
        float a1 = 1.0001f, a2 = 1.0002f, b1 = 0.9999f, b2 = 0.9998f;
        memcpy(&ai1, &a1, 4); memcpy(&ai2, &a2, 4);
        memcpy(&bi1, &b1, 4); memcpy(&bi2, &b2, 4);
        va = ((uint64_t)ai2 << 32) | ai1;
        vb = ((uint64_t)bi2 << 32) | bi1;
        vc = 0;
    }

#if defined(CUTE_ARCH_FLOAT2_MATH_ENABLED)
    for (int i = 0; i < 10; i++)
        asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(vc) : "l"(va), "l"(vb), "l"(vc));
    __syncthreads();

    long long t2 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(vc) : "l"(va), "l"(vb), "l"(vc));
    __syncthreads();
    long long t3 = clock64();
#else
    long long t2 = clock64(), t3 = t2;
#endif

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)(t3 - t2);
        results[2] = sc;                   // anti-DCE
        results[3] = (float)num_iterations;
        uint32_t lo = (uint32_t)(vc & 0xFFFFFFFF);
        float f; memcpy(&f, &lo, 4);
        results[4] = f;                    // anti-DCE
    }
}

// =============================================================================
// K17. LDSM/STSM b8 (SM100 8-bit matrix load/store from SMEM)
// =============================================================================
__global__ void ldsm_b8_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;
    int lane = tid & 31;

    extern __shared__ char smem_raw[];
    uint8_t* smem_u8 = reinterpret_cast<uint8_t*>(smem_raw);
    for (int i = tid; i < 512; i += blockDim.x) smem_u8[i] = (uint8_t)(i & 0xFF);
    __syncthreads();

    // Each thread points to its row (16 bytes per row for m16n16)
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(smem_u8 + (lane % 16) * 16);

#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t r0, r1;
    for (int i = 0; i < 10; i++)
        asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
                     : "=r"(r0), "=r"(r1) : "r"(smem_ptr));
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0, %1}, [%2];"
                     : "=r"(r0), "=r"(r1) : "r"(smem_ptr));
    __syncthreads();
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)num_iterations;
        results[2] = (float)r0;  // anti-DCE
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

__global__ void stsm_b8_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;
    int lane = tid & 31;

    extern __shared__ char smem_raw[];
    // stmatrix m16n8 b8: 16 rows, each row pointer must be 16-byte aligned
    uint32_t smem_ptr = cute::cast_smem_ptr_to_uint(smem_raw + (lane % 16) * 16);

#if defined(CUTE_ARCH_STSM_SM100A_ENABLED)
    uint32_t val = (uint32_t)(tid * 257 + 42);
    for (int i = 0; i < 10; i++)
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
                     : : "r"(smem_ptr), "r"(val));
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};"
                     : : "r"(smem_ptr), "r"(val));
    __syncthreads();
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)num_iterations;
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

// =============================================================================
// K18. UMMA Fence (tcgen05.wait::ld / wait::st) + Commit Overhead
// =============================================================================
__global__ void umma_fence_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw);
    if (tid == 0) cutlass::arch::ClusterBarrier::init(mbar, blockDim.x);
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    __syncthreads();

#if defined(CUTLASS_ARCH_TCGEN_ENABLED)
    uint32_t mbar_ptr = cute::cast_smem_ptr_to_uint(mbar);

    // --- wait::ld ---
    for (int i = 0; i < 10; i++)
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    __syncthreads();
    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
    long long t1 = clock64();

    // --- wait::st ---
    __syncthreads();
    long long t2 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
    long long t3 = clock64();

    // --- commit ---
    for (int i = 0; i < 5; i++) {
        if (cute::elect_one_sync())
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                : : "r"(mbar_ptr));
    }
    __syncthreads();
    long long t4 = clock64();
    for (int i = 0; i < num_iterations; i++) {
        if (cute::elect_one_sync())
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                : : "r"(mbar_ptr));
    }
    long long t5 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)(t3 - t2);
        results[2] = (float)(t5 - t4);
        results[3] = (float)num_iterations;
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

// =============================================================================
// K19. Warpgroup Register Reconfiguration Latency
//
// setmaxnreg.inc/dec cycle cost. Requires 128 threads (1 warpgroup).
// =============================================================================
__global__ void __launch_bounds__(128, 8)
reg_reconfig_probe_kernel(float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;

#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < 5; i++) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n" ::);
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::);
    }
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 32;\n" ::);
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\n" ::);
    }
    __syncthreads();
    long long t1 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)num_iterations;
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}

// =============================================================================
// K20. Grid Dependency Control (PDL) Instruction Latency
//
// griddepcontrol.launch_dependents / griddepcontrol.wait
// =============================================================================
__global__ void pdl_probe_kernel(
    float* __restrict__ results, int num_iterations)
{
    int tid = threadIdx.x;

#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < 5; i++)
        asm volatile("griddepcontrol.launch_dependents;\n" ::);
    __syncthreads();

    long long t0 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("griddepcontrol.launch_dependents;\n" ::);
    __syncthreads();
    long long t1 = clock64();

    for (int i = 0; i < 5; i++)
        asm volatile("griddepcontrol.wait;\n" ::);
    __syncthreads();

    long long t2 = clock64();
    for (int i = 0; i < num_iterations; i++)
        asm volatile("griddepcontrol.wait;\n" ::);
    __syncthreads();
    long long t3 = clock64();

    if (tid == 0) {
        results[0] = (float)(t1 - t0);
        results[1] = (float)(t3 - t2);
        results[2] = (float)num_iterations;
    }
#else
    if (tid == 0) results[0] = -1.0f;
#endif
}


// =============================================================================
// =====================  HOST RUNNER  ========================================
// =============================================================================

// Helper: build a random pointer-chasing chain of given element count
// Creates a random Hamiltonian cycle so every element is visited exactly once.
static void build_chase_chain(int* h_chain, int count) {
    std::vector<int> perm(count);
    std::iota(perm.begin(), perm.end(), 0);
    // Simple LCG for determinism
    uint64_t rng = 0xDEADBEEF42ULL;
    for (int i = count - 1; i > 0; i--) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int j = (int)(rng % (i + 1));
        std::swap(perm[i], perm[j]);
    }
    for (int i = 0; i < count - 1; i++) h_chain[perm[i]] = perm[i + 1];
    h_chain[perm[count - 1]] = perm[0];
}

void run_sm110a_probes() {
    printf("\n");
    printf("================================================================"
           "========\n");
    printf("  SM110a Hardware Primitives Probe  --  Level 0 (Statistical "
           "Edition)\n");
    printf("  Jetson AGX Thor, 20 SM, 128 GB LPDDR5X\n");
    printf("  Trials=%d, Warmup=%d, Report: "
           "min/P5/median/mean/P95/max/stddev/CV\n", NUM_TRIALS, WARMUP_RUNS);
    printf("================================================================"
           "========\n\n");

    float* d_results;
    PROBE_CHECK(cudaMalloc(&d_results, 64 * sizeof(float)));
    std::vector<float> h_results(64);

    // =========================================================================
    // Probe 1: TMEM Lifecycle (multi-trial)
    // =========================================================================
    printf("--- Probe 1: TMEM Lifecycle (x%d trials) ---\n", NUM_TRIALS);
    {
        // cta_group::1 最大 128 列 (TMEM_DP_PER_CTA=128), 必须 2 的幂次
        int test_cols[] = {32, 64, 128};
        for (int cols : test_cols) {
            std::vector<float> alloc_ns_v, store_ns_v, load_ns_v, dealloc_ns_v;
            std::vector<float> store_pc_v, load_pc_v;
            int total_errors = 0;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                tmem_probe_kernel<<<1, 128>>>(d_results, cols);
                PROBE_CHECK(cudaDeviceSynchronize());
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       8 * sizeof(float), cudaMemcpyDeviceToHost));

                if (trial < WARMUP_RUNS) continue;

                float alloc_ns = h_results[0] / CLOCK_GHZ;
                float store_ns = h_results[1] / CLOCK_GHZ;
                float load_ns  = h_results[2] / CLOCK_GHZ;
                float dealloc_ns = h_results[6] / CLOCK_GHZ;

                alloc_ns_v.push_back(alloc_ns);
                store_ns_v.push_back(store_ns);
                load_ns_v.push_back(load_ns);
                dealloc_ns_v.push_back(dealloc_ns);
                store_pc_v.push_back(store_ns / cols);
                load_pc_v.push_back(load_ns / cols);
                total_errors += (int)h_results[3];
            }

            printf("\n  columns=%d (verify_errors=%d):\n", cols, total_errors);
            auto sa = compute_stats(alloc_ns_v);   print_stats_line("alloc", sa, "ns");
            auto ss = compute_stats(store_ns_v);    print_stats_line("store_total", ss, "ns");
            auto sl = compute_stats(load_ns_v);     print_stats_line("load_total", sl, "ns");
            auto ssp = compute_stats(store_pc_v);   print_stats_line("store/col", ssp, "ns/col");
            auto slp = compute_stats(load_pc_v);    print_stats_line("load/col", slp, "ns/col");
            auto sd = compute_stats(dealloc_ns_v);  print_stats_line("dealloc", sd, "ns");
        }
    }

    // =========================================================================
    // Probe 2: UMMA SS Throughput (multi-trial, multi-iter)
    // =========================================================================
    printf("\n--- Probe 2: UMMA BF16 SS [64x16]x[16x64] Throughput (x%d trials)"
           " ---\n", NUM_TRIALS);
    {
        int iters_list[] = {100, 500, 1000, 5000, 10000};
        int smem_bytes = 64 * 16 * 2 * 2;  // 4 KB

        for (int ni : iters_list) {
            std::vector<float> tflops_v;
            bool ok = true;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                umma_ss_probe_kernel<<<1, 128, smem_bytes>>>(d_results, ni);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  iters=%5d: CUDA error: %s\n", ni,
                           cudaGetErrorString(err));
                    ok = false;
                    break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));

                if (h_results[0] < 0) {
                    printf("  UMMA not available on this device\n");
                    ok = false;
                    break;
                }

                if (trial < WARMUP_RUNS) continue;
                tflops_v.push_back(h_results[1]);
            }

            if (!ok) break;

            char label[32];
            snprintf(label, sizeof(label), "iters=%d", ni);
            auto st = compute_stats(tflops_v);
            print_stats_line(label, st, "TFLOPS");
        }
    }

    // =========================================================================
    // Probe 3: DRAM Read Bandwidth (extensive size sweep)
    // =========================================================================
    printf("\n--- Probe 3: DRAM Read Bandwidth (x%d trials per size) ---\n",
           NUM_TRIALS);
    {
        struct SizeEntry { size_t bytes; const char* label; };
        SizeEntry test_sizes[] = {
            {   256ULL << 10, "256KB"},
            {   512ULL << 10, "512KB"},
            {     1ULL << 20, "1MB"},
            {     2ULL << 20, "2MB"},
            {     4ULL << 20, "4MB"},
            {     8ULL << 20, "8MB"},
            {    16ULL << 20, "16MB"},
            {    32ULL << 20, "32MB"},
            {    64ULL << 20, "64MB"},
            {   128ULL << 20, "128MB"},
            {   256ULL << 20, "256MB"},
            {   512ULL << 20, "512MB"},
        };

        float* d_sink;
        PROBE_CHECK(cudaMalloc(&d_sink, sizeof(float)));

        for (auto& entry : test_sizes) {
            size_t sz = entry.bytes;
            float4* d_data;
            cudaError_t alloc_err = cudaMalloc(&d_data, sz);
            if (alloc_err != cudaSuccess) {
                printf("  read %-6s: skip (alloc failed: %s)\n",
                       entry.label, cudaGetErrorString(alloc_err));
                continue;
            }
            PROBE_CHECK(cudaMemset(d_data, 0, sz));

            int num_f4  = (int)(sz / sizeof(float4));
            int threads = 256;
            int blocks  = std::min(NUM_SM * 4,
                                   (num_f4 + threads - 1) / threads);
            // Target ~128MB total data per timed launch
            int iters = std::max(1, (int)(128ULL * 1024 * 1024 / sz));

            std::vector<float> bw_v;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                // Warmup (also helps with GPU clock ramp-up)
                dram_read_kernel<<<blocks, threads>>>(d_data, d_sink, num_f4, 1);
                PROBE_CHECK(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_read_kernel<<<blocks, threads>>>(d_data, d_sink, num_f4,
                                                      iters);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (trial < WARMUP_RUNS) continue;

                double bw = (double)sz * iters / (ms / 1000.0) / 1e9;
                bw_v.push_back((float)bw);
            }

            auto s = compute_stats(bw_v);
            char label[32];
            snprintf(label, sizeof(label), "read %-6s", entry.label);
            print_stats_bw(label, s);

            cudaFree(d_data);
        }
        cudaFree(d_sink);
    }

    // =========================================================================
    // Probe 4: DRAM Write Bandwidth
    // =========================================================================
    printf("\n--- Probe 4: DRAM Write Bandwidth (x%d trials per size) ---\n",
           NUM_TRIALS);
    {
        struct SizeEntry { size_t bytes; const char* label; };
        SizeEntry test_sizes[] = {
            {     1ULL << 20, "1MB"},
            {     4ULL << 20, "4MB"},
            {    16ULL << 20, "16MB"},
            {    64ULL << 20, "64MB"},
            {   256ULL << 20, "256MB"},
            {   512ULL << 20, "512MB"},
        };

        for (auto& entry : test_sizes) {
            size_t sz = entry.bytes;
            float4* d_data;
            cudaError_t alloc_err = cudaMalloc(&d_data, sz);
            if (alloc_err != cudaSuccess) {
                printf("  write %-6s: skip (alloc failed)\n", entry.label);
                continue;
            }

            int num_f4  = (int)(sz / sizeof(float4));
            int threads = 256;
            int blocks  = std::min(NUM_SM * 4,
                                   (num_f4 + threads - 1) / threads);
            int iters = std::max(1, (int)(128ULL * 1024 * 1024 / sz));

            std::vector<float> bw_v;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                dram_write_kernel<<<blocks, threads>>>(d_data, num_f4, 1);
                PROBE_CHECK(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_write_kernel<<<blocks, threads>>>(d_data, num_f4, iters);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (trial < WARMUP_RUNS) continue;
                double bw = (double)sz * iters / (ms / 1000.0) / 1e9;
                bw_v.push_back((float)bw);
            }

            auto s = compute_stats(bw_v);
            char label[32];
            snprintf(label, sizeof(label), "write %-6s", entry.label);
            print_stats_bw(label, s);

            cudaFree(d_data);
        }
    }

    // =========================================================================
    // Probe 5: DRAM Multi-SM Scaling (read BW vs active SM count)
    // =========================================================================
    printf("\n--- Probe 5: DRAM Read BW vs SM Count (256MB, x%d trials) ---\n",
           NUM_TRIALS);
    {
        size_t sz = 256ULL << 20;
        float4* d_data;
        float* d_sink;
        PROBE_CHECK(cudaMalloc(&d_data, sz));
        PROBE_CHECK(cudaMemset(d_data, 0, sz));
        PROBE_CHECK(cudaMalloc(&d_sink, sizeof(float)));

        int num_f4  = (int)(sz / sizeof(float4));
        int threads = 256;
        int sm_counts[] = {1, 2, 4, 5, 8, 10, 15, 20};

        for (int sm_target : sm_counts) {
            int blocks = sm_target;
            std::vector<float> bw_v;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                dram_read_kernel<<<blocks, threads>>>(d_data, d_sink, num_f4, 1);
                PROBE_CHECK(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_read_kernel<<<blocks, threads>>>(d_data, d_sink, num_f4, 1);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (trial < WARMUP_RUNS) continue;
                double bw = (double)sz / (ms / 1000.0) / 1e9;
                bw_v.push_back((float)bw);
            }

            auto s = compute_stats(bw_v);
            char label[32];
            snprintf(label, sizeof(label), "%2d SM(s)", sm_target);
            print_stats_bw(label, s);
        }

        cudaFree(d_data);
        cudaFree(d_sink);
    }

    // =========================================================================
    // Probe 6: DRAM Strided Access Pattern
    // =========================================================================
    printf("\n--- Probe 6: DRAM Strided Read BW (64MB, x%d trials) ---\n",
           NUM_TRIALS);
    {
        size_t sz = 64ULL << 20;
        float4* d_data;
        float* d_sink;
        PROBE_CHECK(cudaMalloc(&d_data, sz));
        PROBE_CHECK(cudaMemset(d_data, 0, sz));
        PROBE_CHECK(cudaMalloc(&d_sink, sizeof(float)));

        int total_f4 = (int)(sz / sizeof(float4));
        int threads  = 256;
        int blocks   = NUM_SM * 4;
        // reads_per_thread determines total data read
        int reads_per_thread = 256;

        // Stride values: 1 (sequential), 2, 4, 8, 16, 64, 256 (scattering)
        int strides[] = {1, 2, 4, 8, 16, 64, 256, 1024};

        for (int stride : strides) {
            std::vector<float> bw_v;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                dram_strided_read_kernel<<<blocks, threads>>>(
                    d_data, d_sink, total_f4, stride, reads_per_thread);
                PROBE_CHECK(cudaDeviceSynchronize());

                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_strided_read_kernel<<<blocks, threads>>>(
                    d_data, d_sink, total_f4, stride, reads_per_thread);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                if (trial < WARMUP_RUNS) continue;
                // total bytes read = blocks * threads * reads_per_thread * 16
                double total_bytes = (double)blocks * threads * reads_per_thread
                                     * sizeof(float4);
                double bw = total_bytes / (ms / 1000.0) / 1e9;
                bw_v.push_back((float)bw);
            }

            auto s = compute_stats(bw_v);
            char label[32];
            snprintf(label, sizeof(label), "stride=%4d", stride);
            print_stats_bw(label, s);
        }

        cudaFree(d_data);
        cudaFree(d_sink);
    }

    // =========================================================================
    // Probe 7: SMEM Bandwidth (multi-trial)
    // =========================================================================
    printf("\n--- Probe 7: SMEM Bandwidth (x%d trials) ---\n", NUM_TRIALS);
    {
        int iters = 10000;
        std::vector<float> bw_v;

        for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
            PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
            smem_bw_kernel<<<1, 128, 4096 * sizeof(float)>>>(d_results, iters);
            PROBE_CHECK(cudaDeviceSynchronize());
            PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                   3 * sizeof(float), cudaMemcpyDeviceToHost));

            if (trial < WARMUP_RUNS) continue;

            float cycles = h_results[0];
            float time_s = cycles / CLOCK_GHZ / 1e9f;
            double total_bytes = 4096.0 * 4.0 * iters;  // 16KB × iters
            double bw_tbs = total_bytes / time_s / 1e12;
            bw_v.push_back((float)bw_tbs);
        }

        auto s = compute_stats(bw_v);
        print_stats_line("16KB read", s, "TB/s");
    }

    // =========================================================================
    // Probe 8: TMA Load vs Manual Copy (multi-tile, multi-trial)
    // =========================================================================
    printf("\n--- Probe 8: TMA Load vs Manual Copy (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int tile_sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 49152};

        char* d_tma_src;
        PROBE_CHECK(cudaMalloc(&d_tma_src, 256 * 1024));
        PROBE_CHECK(cudaMemset(d_tma_src, 0xAB, 256 * 1024));

        printf("  %-5s | %-36s | %-36s | %s\n",
               "", "TMA Load (ns/iter)", "Manual Copy (ns/iter)", "");
        printf("  %-5s | %7s %7s %7s %5s | %7s %7s %7s %5s | %s\n",
               "tile", "med", "mean", "std", "CV%",
               "med", "mean", "std", "CV%", "speedup");
        printf("  ------+--------------------------------------+"
               "--------------------------------------+---------\n");

        for (int sz : tile_sizes) {
            int tma_iters = std::max(200, 32768 / sz * 200);
            int smem_bytes_tma = ((sz + 7) & ~7) + 8;

            std::vector<float> tma_ns_v, man_ns_v;
            bool ok = true;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                // TMA Load
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                tma_load_probe_kernel<<<1, 128, smem_bytes_tma>>>(
                    d_tma_src, d_results, sz, tma_iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  %4dKB: TMA error: %s\n", sz / 1024,
                           cudaGetErrorString(err));
                    ok = false;
                    break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                float tma_ns = h_results[0] / CLOCK_GHZ / tma_iters;

                // Manual Copy
                int tile_f4s = sz / (int)sizeof(float4);
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                manual_copy_probe_kernel<<<1, 128, sz>>>(
                    reinterpret_cast<const float4*>(d_tma_src),
                    d_results, tile_f4s, tma_iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  %4dKB: Manual error: %s\n", sz / 1024,
                           cudaGetErrorString(err));
                    ok = false;
                    break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                float man_ns = h_results[0] / CLOCK_GHZ / tma_iters;

                if (trial < WARMUP_RUNS) continue;
                tma_ns_v.push_back(tma_ns);
                man_ns_v.push_back(man_ns);
            }

            if (!ok) continue;

            auto st = compute_stats(tma_ns_v);
            auto sm = compute_stats(man_ns_v);
            float speedup = sm.median / st.median;

            char tile_label[16];
            if (sz >= 1024)
                snprintf(tile_label, sizeof(tile_label), "%dKB", sz / 1024);
            else
                snprintf(tile_label, sizeof(tile_label), "%dB", sz);

            printf("  %-5s | %7.1f %7.1f %7.1f %4.1f%% | %7.1f %7.1f %7.1f "
                   "%4.1f%% | %.2fx %s\n",
                   tile_label,
                   st.median, st.mean, st.stddev, st.cv,
                   sm.median, sm.mean, sm.stddev, sm.cv,
                   speedup,
                   (speedup > 1.0f) ? "TMA wins" : "Manual wins");
        }

        cudaFree(d_tma_src);
    }

    // =========================================================================
    // Probe 9: TMA Store (multi-tile, multi-trial)
    // =========================================================================
    printf("\n--- Probe 9: TMA Store SMEM->GMEM (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int tile_sizes[] = {1024, 2048, 4096, 8192, 16384, 32768};

        char* d_tma_dst;
        PROBE_CHECK(cudaMalloc(&d_tma_dst, 256 * 1024));

        printf("  %-5s | %7s %7s %7s %5s | %s\n",
               "tile", "med", "mean", "std", "CV%", "eff BW (med)");

        for (int sz : tile_sizes) {
            int store_iters = std::max(200, 32768 / sz * 200);
            std::vector<float> ns_v;
            bool ok = true;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                tma_store_probe_kernel<<<1, 128, sz>>>(
                    d_tma_dst, d_results, sz, store_iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  %4dKB: Error: %s\n", sz / 1024,
                           cudaGetErrorString(err));
                    ok = false;
                    break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));

                if (trial < WARMUP_RUNS) continue;
                float ns_per_iter = h_results[0] / CLOCK_GHZ / store_iters;
                ns_v.push_back(ns_per_iter);
            }

            if (!ok) continue;

            auto s = compute_stats(ns_v);
            float gbs = (float)sz / s.median * 1.0f;   // bytes / ns = GB/s
            char tile_label[16];
            snprintf(tile_label, sizeof(tile_label), "%dKB", sz / 1024);
            printf("  %-5s | %7.1f %7.1f %7.1f %4.1f%% | %6.1f GB/s\n",
                   tile_label, s.median, s.mean, s.stddev, s.cv, gbs);
        }

        cudaFree(d_tma_dst);
    }

    // =========================================================================
    // Probe 10: Memory Access Latency (pointer chasing)
    // =========================================================================
    printf("\n--- Probe 10: Memory Access Latency (pointer chasing, x%d trials)"
           " ---\n", NUM_TRIALS);
    {
        struct LatencyTest {
            const char* name;
            size_t count;  // number of int elements
        };
        LatencyTest tests[] = {
            {"4KB  (L1 hit)",         1024},       // 4 KB
            {"16KB (L1/L2)",          4096},       // 16 KB
            {"64KB (L2)",            16384},       // 64 KB
            {"256KB (L2)",           65536},       // 256 KB
            {"1MB  (L2)",           262144},       // 1 MB
            {"4MB  (L2)",          1048576},       // 4 MB
            {"16MB (L2 edge)",     4194304},       // 16 MB
            {"32MB (L2=32MB)",     8388608},       // 32 MB
            {"64MB (DRAM)",       16777216},       // 64 MB
            {"256MB (DRAM)",      67108864},       // 256 MB
        };

        int steps = 100000;

        for (auto& test : tests) {
            size_t bytes = test.count * sizeof(int);

            std::vector<int> h_chain(test.count);
            build_chase_chain(h_chain.data(), (int)test.count);

            int* d_chain;
            cudaError_t alloc_err = cudaMalloc(&d_chain, bytes);
            if (alloc_err != cudaSuccess) {
                printf("  %-22s: skip (alloc %zu MB failed)\n",
                       test.name, bytes >> 20);
                continue;
            }
            PROBE_CHECK(cudaMemcpy(d_chain, h_chain.data(), bytes,
                                   cudaMemcpyHostToDevice));

            std::vector<float> latency_ns_v;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                mem_latency_kernel<<<1, 1>>>(d_chain, d_results, steps);
                PROBE_CHECK(cudaDeviceSynchronize());
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));

                if (trial < WARMUP_RUNS) continue;

                float cycles = h_results[0];
                float latency_ns = cycles / CLOCK_GHZ / steps;
                latency_ns_v.push_back(latency_ns);
            }

            auto s = compute_stats(latency_ns_v);
            print_stats_line(test.name, s, "ns/access");

            cudaFree(d_chain);
        }
    }

    // =========================================================================
    // Probe 12: UMMA TS (A=TMEM, B=SMEM) Throughput (x20 trials)
    // =========================================================================
    printf("\n--- Probe 12: UMMA BF16 TS Mode [64x16]x[16x64] (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int iters_list[] = {100, 1000, 10000};
        int smem_bytes = 16 * 64 * 2;  // B[16,64] BF16 = 2048 B

        for (int ni : iters_list) {
            std::vector<float> tflops_v;
            bool ok = true;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                umma_ts_probe_kernel<<<1, 128, smem_bytes>>>(d_results, ni);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  iters=%5d: CUDA error: %s\n", ni, cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                if (h_results[0] < 0) { printf("  UMMA TS not available\n"); ok = false; break; }
                if (trial < WARMUP_RUNS) continue;
                tflops_v.push_back(h_results[1]);
            }
            if (!ok) break;

            char label[32];
            snprintf(label, sizeof(label), "TS iters=%d", ni);
            auto st = compute_stats(tflops_v);
            print_stats_line(label, st, "TFLOPS");
        }
    }

    // =========================================================================
    // Probe 13: UTCCP (SMEM→TMEM) Copy Latency (x20 trials)
    // =========================================================================
    printf("\n--- Probe 13: UTCCP SMEM->TMEM Copy Latency (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int smem_bytes = 4096;
        int iters = 10000;

        // 128x256b mode (4 KB / op)
        {
            std::vector<float> ns_v;
            bool ok = true;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                utccp_probe_kernel<0><<<1, 128, smem_bytes>>>(d_results, iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  128x256b: CUDA error: %s\n", cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                if (h_results[0] < 0) { printf("  UTCCP not available\n"); ok = false; break; }
                if (trial < WARMUP_RUNS) continue;
                ns_v.push_back(h_results[0] / CLOCK_GHZ / iters);
            }
            if (ok) {
                auto s = compute_stats(ns_v);
                print_stats_line("128x256b (4KB/op)", s, "ns/op");
                printf("    -> eff BW: %.1f GB/s/SM\n", 4096.0f / s.median);
            }
        }

        // 4x256b mode (128 B / op)
        {
            std::vector<float> ns_v;
            bool ok = true;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                utccp_probe_kernel<1><<<1, 128, smem_bytes>>>(d_results, iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  4x256b: CUDA error: %s\n", cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                if (h_results[0] < 0) { ok = false; break; }
                if (trial < WARMUP_RUNS) continue;
                ns_v.push_back(h_results[0] / CLOCK_GHZ / iters);
            }
            if (ok) {
                auto s = compute_stats(ns_v);
                print_stats_line("4x256b  (128B/op)", s, "ns/op");
                printf("    -> eff BW: %.1f GB/s/SM\n", 128.0f / s.median);
            }
        }
    }

    // =========================================================================
    // Probe 14: TMEM pack::16b / unpack::16b (x20 trials)
    // =========================================================================
    printf("\n--- Probe 14: TMEM pack::16b Load / unpack::16b Store (x%d trials)"
           " ---\n", NUM_TRIALS);
    {
        int iters = 10000;
        std::vector<float> norm_ld_v, pack_ld_v, norm_st_v, unpack_st_v;
        bool ok = true;

        for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
            PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
            tmem_pack_probe_kernel<<<1, 128>>>(d_results, iters);
            PROBE_CHECK(cudaDeviceSynchronize());
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("  CUDA error: %s\n", cudaGetErrorString(err));
                ok = false; break;
            }
            PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                   8 * sizeof(float), cudaMemcpyDeviceToHost));
            if (h_results[0] < 0) { printf("  TMEM pack not available\n"); ok = false; break; }
            if (trial < WARMUP_RUNS) continue;
            float n = h_results[4];
            norm_ld_v.push_back(h_results[0] / CLOCK_GHZ / n);
            pack_ld_v.push_back(h_results[1] / CLOCK_GHZ / n);
            norm_st_v.push_back(h_results[2] / CLOCK_GHZ / n);
            unpack_st_v.push_back(h_results[3] / CLOCK_GHZ / n);
        }
        if (ok) {
            auto s1 = compute_stats(norm_ld_v);   print_stats_line("normal ld (32x32b.x4)", s1, "ns/op");
            auto s2 = compute_stats(pack_ld_v);    print_stats_line("pack::16b ld", s2, "ns/op");
            auto s3 = compute_stats(norm_st_v);    print_stats_line("normal st (32x32b.x4)", s3, "ns/op");
            auto s4 = compute_stats(unpack_st_v);  print_stats_line("unpack::16b st", s4, "ns/op");
            printf("    pack ratio: %.2fx  |  unpack ratio: %.2fx\n",
                   s1.median / s2.median, s3.median / s4.median);
        }
    }

    // =========================================================================
    // Probe 15: L1::no_allocate DRAM Read BW (x20 trials)
    // =========================================================================
    printf("\n--- Probe 15: L1::no_allocate DRAM Read (64MB, x%d trials) ---\n",
           NUM_TRIALS);
    {
        size_t sz = 64ULL << 20;
        float* d_data_f; float* d_sink;
        PROBE_CHECK(cudaMalloc(&d_data_f, sz));
        PROBE_CHECK(cudaMemset(d_data_f, 0, sz));
        PROBE_CHECK(cudaMalloc(&d_sink, sizeof(float)));

        int threads = 256;
        int blocks  = std::min(NUM_SM * 4, (int)(sz / (8 * sizeof(float)) / threads));
        int iters   = 2;

        // Standard (float4, 128-bit)
        {
            int num_f4 = (int)(sz / sizeof(float4));
            std::vector<float> bw_v;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                dram_read_kernel<<<blocks, threads>>>((const float4*)d_data_f,
                                                      d_sink, num_f4, 1);
                PROBE_CHECK(cudaDeviceSynchronize());
                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_read_kernel<<<blocks, threads>>>((const float4*)d_data_f,
                                                      d_sink, num_f4, iters);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start); cudaEventDestroy(stop);
                if (trial < WARMUP_RUNS) continue;
                bw_v.push_back((float)((double)sz * iters / (ms / 1000.0) / 1e9));
            }
            auto s = compute_stats(bw_v);
            print_stats_bw("standard (v4.f32)", s);
        }

        // L1::no_allocate (v8.f32, 256-bit)
        {
            int num_f8 = (int)(sz / (8 * sizeof(float)));
            std::vector<float> bw_v;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                dram_read_noalloc_kernel<<<blocks, threads>>>(d_data_f,
                                                              d_sink, num_f8, 1);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  L1::no_allocate not available: %s\n",
                           cudaGetErrorString(err));
                    break;
                }
                cudaEvent_t start, stop;
                PROBE_CHECK(cudaEventCreate(&start));
                PROBE_CHECK(cudaEventCreate(&stop));
                PROBE_CHECK(cudaEventRecord(start));
                dram_read_noalloc_kernel<<<blocks, threads>>>(d_data_f,
                                                              d_sink, num_f8, iters);
                PROBE_CHECK(cudaEventRecord(stop));
                PROBE_CHECK(cudaEventSynchronize(stop));
                float ms;
                PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
                cudaEventDestroy(start); cudaEventDestroy(stop);
                if (trial < WARMUP_RUNS) continue;
                bw_v.push_back((float)((double)sz * iters / (ms / 1000.0) / 1e9));
            }
            if (!bw_v.empty()) {
                auto s = compute_stats(bw_v);
                print_stats_bw("L1::no_alloc (v8)", s);
            }
        }
        cudaFree(d_data_f); cudaFree(d_sink);
    }

    // =========================================================================
    // Probe 16: f32x2 SIMD FMA vs Scalar (x20 trials)
    // =========================================================================
    printf("\n--- Probe 16: f32x2 SIMD FMA vs Scalar (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int iters_list[] = {100000, 1000000};
        for (int ni : iters_list) {
            std::vector<float> scalar_v, simd_v;
            bool ok = true;

            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                f32x2_fma_probe_kernel<<<1, 128>>>(d_results, ni);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  CUDA error: %s\n", cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       8 * sizeof(float), cudaMemcpyDeviceToHost));
                if (trial < WARMUP_RUNS) continue;
                float n = h_results[3];
                scalar_v.push_back(h_results[0] / CLOCK_GHZ / n);
                simd_v.push_back(h_results[1] / CLOCK_GHZ / n);
            }
            if (!ok) break;

            char sl[32], sl2[32];
            snprintf(sl, sizeof(sl), "scalar N=%dK", ni / 1000);
            snprintf(sl2, sizeof(sl2), "f32x2  N=%dK", ni / 1000);
            auto s1 = compute_stats(scalar_v);
            auto s2 = compute_stats(simd_v);
            print_stats_line(sl, s1, "ns/FMA");
            print_stats_line(sl2, s2, "ns/op(2FMA)");
            if (s2.median > 0)
                printf("    -> f32x2 speedup: %.2fx\n", (s1.median * 2.0f) / s2.median);
        }
    }

    // =========================================================================
    // Probe 17: LDSM/STSM b8 Bandwidth (x20 trials)
    // =========================================================================
    printf("\n--- Probe 17: LDSM/STSM b8 (x%d trials) ---\n", NUM_TRIALS);
    {
        int iters = 100000;
        int smem_bytes = 512;

        // LDSM b8 m16n16.x1 (256 B per load)
        {
            std::vector<float> ns_v;
            bool ok = true;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                ldsm_b8_probe_kernel<<<1, 128, smem_bytes>>>(d_results, iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  LDSM b8: CUDA error: %s\n", cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                if (h_results[0] < 0) { printf("  LDSM b8 not available\n"); ok = false; break; }
                if (trial < WARMUP_RUNS) continue;
                ns_v.push_back(h_results[0] / CLOCK_GHZ / iters);
            }
            if (ok) {
                auto s = compute_stats(ns_v);
                print_stats_line("LDSM b8 m16n16x1", s, "ns/op");
                printf("    -> eff BW: %.1f GB/s/SM (256B / %.1f ns)\n",
                       256.0f / s.median, s.median);
            }
        }

        // STSM b8 m16n8.x1 (128 B per store)
        {
            std::vector<float> ns_v;
            bool ok = true;
            for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
                PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
                stsm_b8_probe_kernel<<<1, 128, smem_bytes>>>(d_results, iters);
                PROBE_CHECK(cudaDeviceSynchronize());
                auto err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("  STSM b8: CUDA error: %s\n", cudaGetErrorString(err));
                    ok = false; break;
                }
                PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                       4 * sizeof(float), cudaMemcpyDeviceToHost));
                if (h_results[0] < 0) { printf("  STSM b8 not available\n"); ok = false; break; }
                if (trial < WARMUP_RUNS) continue;
                ns_v.push_back(h_results[0] / CLOCK_GHZ / iters);
            }
            if (ok) {
                auto s = compute_stats(ns_v);
                print_stats_line("STSM b8 m16n8x1", s, "ns/op");
                printf("    -> eff BW: %.1f GB/s/SM (128B / %.1f ns)\n",
                       128.0f / s.median, s.median);
            }
        }
    }

    // =========================================================================
    // Probe 18: UMMA Fence + Commit Overhead (x20 trials)
    // =========================================================================
    printf("\n--- Probe 18: UMMA Fence/Commit Overhead (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int iters = 10000;
        int smem_bytes = 64;
        std::vector<float> ld_ns_v, st_ns_v, commit_ns_v;
        bool ok = true;

        for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
            PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
            umma_fence_probe_kernel<<<1, 128, smem_bytes>>>(d_results, iters);
            PROBE_CHECK(cudaDeviceSynchronize());
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("  CUDA error: %s\n", cudaGetErrorString(err));
                ok = false; break;
            }
            PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                   8 * sizeof(float), cudaMemcpyDeviceToHost));
            if (h_results[0] < 0) { printf("  UMMA fence not available\n"); ok = false; break; }
            if (trial < WARMUP_RUNS) continue;
            float n = h_results[3];
            ld_ns_v.push_back(h_results[0] / CLOCK_GHZ / n);
            st_ns_v.push_back(h_results[1] / CLOCK_GHZ / n);
            commit_ns_v.push_back(h_results[2] / CLOCK_GHZ / n);
        }
        if (ok) {
            auto s1 = compute_stats(ld_ns_v);    print_stats_line("wait::ld", s1, "ns/op");
            auto s2 = compute_stats(st_ns_v);    print_stats_line("wait::st", s2, "ns/op");
            auto s3 = compute_stats(commit_ns_v); print_stats_line("commit", s3, "ns/op");
        }
    }

    // =========================================================================
    // Probe 19: Warpgroup Register Reconfiguration (x20 trials)
    // =========================================================================
    printf("\n--- Probe 19: Warpgroup Reg Reconfig (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int iters = 10000;
        std::vector<float> ns_v;
        bool ok = true;

        for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
            PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
            reg_reconfig_probe_kernel<<<1, 128>>>(d_results, iters);
            PROBE_CHECK(cudaDeviceSynchronize());
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("  Not supported on SM110a: %s\n", cudaGetErrorString(err));
                ok = false; break;
            }
            PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                   4 * sizeof(float), cudaMemcpyDeviceToHost));
            if (h_results[0] < 0) { printf("  setmaxnreg not available\n"); ok = false; break; }
            if (trial < WARMUP_RUNS) continue;
            ns_v.push_back(h_results[0] / CLOCK_GHZ / iters);
        }
        if (ok) {
            auto s = compute_stats(ns_v);
            print_stats_line("inc+dec (32 regs)", s, "ns/pair");
        }
    }

    // =========================================================================
    // Probe 20: Grid Dependency Control / PDL (x20 trials)
    // =========================================================================
    printf("\n--- Probe 20: Grid Dependency Control / PDL (x%d trials) ---\n",
           NUM_TRIALS);
    {
        int iters = 10000;
        std::vector<float> launch_ns_v, wait_ns_v;
        bool ok = true;

        for (int trial = 0; trial < WARMUP_RUNS + NUM_TRIALS; trial++) {
            PROBE_CHECK(cudaMemset(d_results, 0, 64 * sizeof(float)));
            pdl_probe_kernel<<<1, 128>>>(d_results, iters);
            PROBE_CHECK(cudaDeviceSynchronize());
            auto err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("  Not supported on SM110a: %s\n", cudaGetErrorString(err));
                ok = false; break;
            }
            PROBE_CHECK(cudaMemcpy(h_results.data(), d_results,
                                   4 * sizeof(float), cudaMemcpyDeviceToHost));
            if (h_results[0] < 0) { printf("  PDL not available\n"); ok = false; break; }
            if (trial < WARMUP_RUNS) continue;
            float n = h_results[2];
            launch_ns_v.push_back(h_results[0] / CLOCK_GHZ / n);
            wait_ns_v.push_back(h_results[1] / CLOCK_GHZ / n);
        }
        if (ok) {
            auto s1 = compute_stats(launch_ns_v);
            auto s2 = compute_stats(wait_ns_v);
            print_stats_line("launch_dependents", s1, "ns/op");
            print_stats_line("wait", s2, "ns/op");
        }
    }

    // =========================================================================
    // Probe 21: Hardware Config (reference)
    // =========================================================================
    printf("\n--- Probe 21: Hardware Config ---\n");
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("  Device: %s\n", prop.name);
        printf("  SM count: %d\n", prop.multiProcessorCount);
        printf("  SMEM/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  SMEM/block max: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Registers/SM: %d\n", prop.regsPerMultiprocessor);
        printf("  Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Warp size: %d\n", prop.warpSize);
        int clock_khz = 0, mem_clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
        printf("  Clock: %.0f MHz (base)\n", clock_khz / 1000.0);
        printf("  Mem clock: %.0f MHz (base, actual 4266 MT/s)\n",
               mem_clock_khz / 1000.0);
        printf("  Memory bus: %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache: %d KB = %d MB\n",
               prop.l2CacheSize / 1024, prop.l2CacheSize / 1024 / 1024);
        printf("  Compute: %d.%d\n", prop.major, prop.minor);
        printf("  DRAM peak BW: %.1f GB/s (theoretical)\n", PEAK_BW_GBS);
    }

    // =========================================================================
    //  Statistical Quality Guide
    // =========================================================================
    printf("\n--- Statistical Quality Guide ---\n");
    printf("  CV%% = Coefficient of Variation = stddev / mean * 100%%\n");
    printf("  *** CV < 2%%  : Excellent - direct optimization target\n");
    printf("  **  CV < 5%%  : Good      - reliable baseline\n");
    printf("  *   CV < 15%% : Fair      - combine with other data\n");
    printf("      CV >= 15%%: Poor      - potential system interference\n");
    printf("\n  Config: %d trials, %d warmup, clock=%.3f GHz, "
           "peak BW=%.0f GB/s\n\n",
           NUM_TRIALS, WARMUP_RUNS, CLOCK_GHZ, PEAK_BW_GBS);

    cudaFree(d_results);
}

} // namespace sm110a_probe
