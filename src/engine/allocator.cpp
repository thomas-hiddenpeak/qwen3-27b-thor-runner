#include "allocator.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
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

    // 自适应 mmap 策略: 根据文件大小 vs 可用内存决定是否预读
    //  - 小文件 (< 可用内存 25%): MAP_POPULATE 批量预读, 最快
    //  - 大文件: 按需 fault + MADV_WILLNEED 后台异步预读, 避免峰值内存暴涨
    size_t avail_kb = 0;
    {
        FILE* f = fopen("/proc/meminfo", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "MemAvailable:", 13) == 0) {
                    sscanf(line + 13, " %zu", &avail_kb);
                    break;
                }
            }
            fclose(f);
        }
    }
    size_t avail_bytes = avail_kb * 1024;
    bool use_populate = (avail_bytes > 0 && size_ < avail_bytes / 4);

    int flags = MAP_PRIVATE;
    if (use_populate) flags |= MAP_POPULATE;

    base_ptr_ = mmap(nullptr, size_, PROT_READ, flags, fd_, 0);
    if (base_ptr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for file: " + file_path_);
    }

    if (use_populate) {
        // 小文件已 MAP_POPULATE 预读, 用 MADV_HUGEPAGE 减少 TLB miss
        madvise(base_ptr_, size_, MADV_HUGEPAGE);
    } else {
        // 大文件: 顺序预读 hint + 后台异步预读
        madvise(base_ptr_, size_, MADV_SEQUENTIAL);
        madvise(base_ptr_, size_, MADV_WILLNEED);
    }
}

MmapAllocator::~MmapAllocator() {
    if (base_ptr_ != MAP_FAILED) {
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
