#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen_thor {
namespace core {

// 数据类型枚举，支持 Qwen3.5 和 Blackwell 架构的特性
enum class DataType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8_E4M3, // Blackwell 原生支持
    FP8_E5M2,
    FP4,      // Blackwell 极低精度支持
    UNKNOWN
};

// 获取数据类型的字节大小
inline size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        case DataType::FP4: return 0; // 特殊处理，通常 2 个 FP4 占 1 字节
        default: throw std::runtime_error("Unknown data type");
    }
}

// 内存分配器接口
class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t size) = 0;
    virtual void deallocate(void* ptr) = 0;
};

// 针对 Jetson Thor 统一内存架构 (UMA) 的分配器
// 利用 cudaMallocManaged 实现 CPU/GPU 共享的零拷贝内存
class UnifiedAllocator : public Allocator {
public:
    UnifiedAllocator();
    ~UnifiedAllocator() override;

    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;

    // 禁用拷贝和赋值
    UnifiedAllocator(const UnifiedAllocator&) = delete;
    UnifiedAllocator& operator=(const UnifiedAllocator&) = delete;
};

// 纯设备内存分配器，用于不需要 CPU 访问的大缓冲区 (如 KV Cache)
// cudaMalloc 立即建立 GPU 页表映射，无 lazy page fault 开销
class DeviceAllocator : public Allocator {
public:
    DeviceAllocator() = default;
    ~DeviceAllocator() override = default;

    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;

    DeviceAllocator(const DeviceAllocator&) = delete;
    DeviceAllocator& operator=(const DeviceAllocator&) = delete;
};

// 内存映射分配器，用于直接从磁盘零拷贝加载 Safetensors 权重
class MmapAllocator : public Allocator {
public:
    MmapAllocator(const std::string& file_path);
    ~MmapAllocator() override;

    void* allocate(size_t size) override;
    void deallocate(void* ptr) override;

    void* get_base_ptr() const { return base_ptr_; }
    size_t get_size() const { return size_; }

private:
    std::string file_path_;
    int fd_;
    void* base_ptr_;
    size_t size_;
};

} // namespace core
} // namespace qwen_thor
