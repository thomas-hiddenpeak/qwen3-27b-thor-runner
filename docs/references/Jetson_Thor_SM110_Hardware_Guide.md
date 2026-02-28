# NVIDIA Jetson Thor AGX (SM110) 硬件特性与优化指南

## 1. 平台概述
Jetson Thor 是专为物理 AI 和机器人设计的边缘计算平台，其底层架构为 Blackwell (SM110)，TDP 在 40W-130W 之间。运行在 L4T R38 和 JetPack 7 之上，向服务器基础系统架构 (SBSA) 对齐。

## 2. 核心硬件特性
*   **统一内存架构 (UMA)**：搭载高达 **128 GB LPDDR5X** 内存。与独立显卡不同，Jetson 采用真正的物理统一内存架构，CPU 和 GPU 共享这 128GB 的物理 RAM，提供极高的内存带宽，彻底消除 PCIe 传输瓶颈。
*   **TMA (Tensor Memory Accelerator)**：Blackwell 架构的核心异步内存搬运引擎。TMA 可以将多维张量数据直接从全局内存 (Global Memory) 异步预取到共享内存 (Shared Memory) 中，**完全绕过 SM 寄存器**，极大释放了线程用于计算。
*   **WGMMA (Warp Group Matrix Multiply Accumulate)**：针对 128 线程 (Warp Group) 的异步矩阵乘加指令。WGMMA 直接从共享内存读取操作数，与 TMA 形成完美的异步流水线 (`TMA fetch -> Shared Memory -> WGMMA compute`)。
*   **FP8/FP4 支持**：搭载第二代 Transformer 引擎，原生支持 FP8 和全新的 **FP4** 数据格式，提供高达 2070 FP4 TFLOPS 的算力。对于超大参数模型，FP4 量化是降低显存占用和提升吞吐量的关键。

## 3. JetPack 7 / L4T R38 的零拷贝 (Zero-Copy) 内存访问
*   **机制暴露**：在 L4T R38 中，CUDA 引擎可以通过 `cudaHostAlloc`（配合 `cudaHostAllocMapped` 标志）或直接使用 `cudaMallocManaged` 分配内存。
*   **物理直达**：由于物理内存是统一的，GPU 的内存控制器可以直接通过页表访问 CPU 填充的 LPDDR5X 物理地址，**无需调用 `cudaMemcpy`**。

## 4. C++ CUDA 推理引擎设计启示
1.  **零拷贝权重流式传输**：利用 JetPack 7 的零拷贝特性，将庞大的模型权重文件通过 CPU 内存映射 (mmap) 直接暴露给 GPU。GPU 仅在需要时通过统一内存地址读取数据，彻底消除 CPU 到 GPU 的 PCIe 传输延迟。
2.  **TMA + WGMMA 异步 MoE 算子**：编写自定义的 MoE Kernel。当路由器选出专家后，利用 `cuda::ptx::cp_async_bulk` (TMA) 异步地将专家的权重从 LPDDR5X 全局内存直接拉取到 SM 的 Shared Memory 中，随后立即触发 `wgmma.mma_async` 进行张量核心计算。