#pragma once

#include "tensor.h"
#include "allocator.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace qwen_thor {
namespace io {

// Safetensors 文件中的 Tensor 元数据
struct TensorMetadata {
    std::string dtype_str;
    core::DataType dtype;
    std::vector<int64_t> shape;
    size_t data_offsets[2]; // [start, end]
};

// Safetensors 加载器，利用 MmapAllocator 实现零拷贝加载
class SafetensorsLoader {
public:
    SafetensorsLoader(const std::string& file_path);
    ~SafetensorsLoader();

    // 禁用拷贝
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;

    // 获取指定名称的 Tensor
    // 返回的 Tensor 对象不拥有内存，其 data_ptr 指向 mmap 区域
    std::unique_ptr<core::Tensor> get_tensor(const std::string& name) const;

    // 获取所有 Tensor 的名称
    std::vector<std::string> get_tensor_names() const;

    // 检查是否包含某个 Tensor
    bool has_tensor(const std::string& name) const;

private:
    void parse_header();
    core::DataType parse_dtype(const std::string& dtype_str) const;

    std::string file_path_;
    std::unique_ptr<core::MmapAllocator> mmap_allocator_;
    std::unordered_map<std::string, TensorMetadata> metadata_map_;
    size_t header_size_;
};

} // namespace io
} // namespace qwen_thor
