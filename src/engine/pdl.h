#pragma once
// =============================================================================
// PDL (Programmatic Dependent Launch) — SM90+ Grid Dependency Control
//
// 允许同一 stream 上连续 kernel 的 launch gap 与尾部重叠:
//   - 前驱 kernel 完成全局内存写入后调用 PDL_SIGNAL()
//   - 后继 kernel 在读取全局内存前调用 PDL_WAIT()
//   - Host 端使用 PDL_LAUNCH() 替代 <<<>>> 启用 overlap
//
// 约束:
//   - cudaStreamSynchronize 重置依赖链 (每层同步不受影响)
//   - 链中某个 kernel 不使用 PDL 则该位置退化为传统串行
//   - 仅 SM90+ (≥ __CUDA_ARCH__ 900) 支持 griddepcontrol 指令
// =============================================================================

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// Device-side: kernel 入口/出口指令
// ─────────────────────────────────────────────────────────────────────────────
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 900

#define PDL_WAIT()   asm volatile("griddepcontrol.wait;")
#define PDL_SIGNAL() asm volatile("griddepcontrol.launch_dependents;")

#else
#define PDL_WAIT()
#define PDL_SIGNAL()
#endif
#else
// Host compilation — macros are no-ops
#define PDL_WAIT()
#define PDL_SIGNAL()
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Host-side: cudaLaunchKernelEx wrapper with PDL attribute
// ─────────────────────────────────────────────────────────────────────────────
// Usage:  PDL_LAUNCH(kernel, grid, block, smem, stream, arg1, arg2, ...)
//
// Note: cudaLaunchKernelEx requires CUDA 12.0+.
// The macro sets ProgrammaticStreamSerialization = 1 so the runtime allows
// overlapping launch of the next kernel before this one fully retires.
//
#define PDL_LAUNCH(kernel, grid, block, smem, stream, ...)              \
  do {                                                                   \
    cudaLaunchConfig_t __pdl_cfg = {};                                   \
    __pdl_cfg.gridDim = (grid);                                          \
    __pdl_cfg.blockDim = (block);                                        \
    __pdl_cfg.dynamicSmemBytes = (smem);                                 \
    __pdl_cfg.stream = (stream);                                         \
    cudaLaunchAttribute __pdl_attr[1];                                   \
    __pdl_attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\
    __pdl_attr[0].val.programmaticStreamSerializationAllowed = 1;        \
    __pdl_cfg.numAttrs = 1;                                              \
    __pdl_cfg.attrs = __pdl_attr;                                        \
    cudaLaunchKernelEx(&__pdl_cfg, (kernel), ##__VA_ARGS__);             \
  } while(0)
