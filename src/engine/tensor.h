#pragma once

#include "allocator.h"
#include <vector>
#include <memory>
#include <numeric>

namespace qwen_thor {
namespace core {

// 针对 Jetson Thor 统一内存架构优化的 Tensor 类
class Tensor {
public:
    // 构造函数：使用指定的分配器分配内存
    Tensor(const std::vector<int64_t>& shape, DataType dtype, std::shared_ptr<Allocator> allocator);

    // 构造函数：从已有的内存指针创建（例如从 mmap 映射的 Safetensors 权重）
    // 这种方式不负责释放内存，生命周期由外部管理
    Tensor(const std::vector<int64_t>& shape, DataType dtype, void* data_ptr);

    ~Tensor();

    // 禁用拷贝，允许移动
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // 获取张量属性
    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    void* data() const { return data_ptr_; }
    size_t numel() const { return numel_; }
    size_t nbytes() const { return nbytes_; }

    // 辅助函数：计算元素总数
    static size_t compute_numel(const std::vector<int64_t>& shape) {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    }

private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    size_t numel_;
    size_t nbytes_;

    void* data_ptr_;
    std::shared_ptr<Allocator> allocator_; // 如果为空，则表示内存由外部管理
};

} // namespace core
} // namespace qwen_thor
