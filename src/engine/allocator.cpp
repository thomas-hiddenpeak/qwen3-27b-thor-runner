#include "allocator.h"
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

namespace qwen_thor {
namespace core {

// 统一内存分配器实现
UnifiedAllocator::UnifiedAllocator() {}

UnifiedAllocator::~UnifiedAllocator() {}

void* UnifiedAllocator::allocate(size_t size) {
    void* ptr = nullptr;
    // 在 Jetson Thor 上，cudaMallocManaged 分配的内存是真正的物理统一内存
    // CPU 和 GPU 都可以直接访问，无需显式拷贝
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void UnifiedAllocator::deallocate(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// 纯设备内存分配器实现
void* DeviceAllocator::allocate(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

void DeviceAllocator::deallocate(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

// 内存映射分配器实现
MmapAllocator::MmapAllocator(const std::string& file_path) : file_path_(file_path), fd_(-1), base_ptr_(MAP_FAILED), size_(0) {
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file for mmap: " + file_path_);
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to get file size for mmap: " + file_path_);
    }
    size_ = sb.st_size;

    // 使用 MAP_PRIVATE 映射文件 — 允许 THP (Transparent Huge Pages) 产生 2MB 大页
    // MAP_SHARED 的文件映射不支持 THP; MAP_PRIVATE 在只读场景下不会额外复制
    // MAP_POPULATE 提前触发所有页面的 fault，避免推理时 page fault
    base_ptr_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for file: " + file_path_);
    }

    // 请求内核使用 Transparent Huge Pages (2MB) 来支持此区域
    // 这可以显著减少 TLB miss，提高 GPU 访问大权重矩阵的带宽
    madvise(base_ptr_, size_, MADV_HUGEPAGE);

    // 注册内存，使其对 GPU 可见（零拷贝）
    // 注意：在某些 JetPack 版本中，mmap 的内存可能需要显式注册才能被 GPU 访问
    cudaError_t err = cudaHostRegister(base_ptr_, size_, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        std::cerr << "Warning: cudaHostRegister failed for mmap memory. "
                  << "This might be expected on some Jetson configurations if UMA handles it automatically. "
                  << "Error: " << cudaGetErrorString(err) << std::endl;
    }
}

MmapAllocator::~MmapAllocator() {
    if (base_ptr_ != MAP_FAILED) {
        // 取消注册
        cudaHostUnregister(base_ptr_);
        munmap(base_ptr_, size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

void* MmapAllocator::allocate(size_t size) {
    // MmapAllocator 不支持动态分配，它只管理整个文件的映射
    throw std::runtime_error("MmapAllocator does not support dynamic allocation.");
}

void MmapAllocator::deallocate(void* ptr) {
    // MmapAllocator 不支持动态释放
    throw std::runtime_error("MmapAllocator does not support dynamic deallocation.");
}

} // namespace core
} // namespace qwen_thor
