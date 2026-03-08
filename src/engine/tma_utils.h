#pragma once
// TMA bulk copy helpers for SSM state load/store (SM90+/SM110a)
// cp.async.bulk GMEM↔SMEM with mbarrier synchronization.
#include <cstdint>

__device__ __forceinline__ void tma_mbar_init(uint64_t* mbar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(smem_addr));
#endif
}

__device__ __forceinline__ void tma_mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(smem_addr), "r"(bytes));
#endif
}

__device__ __forceinline__ void tma_bulk_g2s(const void* gmem, void* smem, uint64_t* mbar, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    uint32_t smem_mbar = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
#endif
}

__device__ __forceinline__ void tma_mbar_wait(uint64_t* mbar, uint32_t phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("{\n"
                 ".reg .pred P;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
                 "@!P bra LAB_WAIT;\n"
                 "}\n" :: "r"(smem_addr), "r"(phase));
#endif
}

__device__ __forceinline__ void tma_bulk_s2g(const void* smem, void* gmem, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                 :: "l"(gmem), "r"(smem_ptr), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
}
