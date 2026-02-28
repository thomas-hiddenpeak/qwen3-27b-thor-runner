#include "safetensors.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <sys/mman.h>

// 极简的 JSON 解析器，仅用于解析 Safetensors 头部
// 在实际生产环境中，应替换为 nlohmann/json 或 simdjson
namespace {
    // 辅助函数：查找字符串中的键值对
    std::string extract_json_value(const std::string& json, const std::string& key) {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = json.find(":", pos);
        if (pos == std::string::npos) return "";
        
        size_t start = json.find_first_not_of(" \t\n\r", pos + 1);
        if (start == std::string::npos) return "";

        if (json[start] == '"') {
            size_t end = json.find("\"", start + 1);
            return json.substr(start + 1, end - start - 1);
        } else if (json[start] == '[') {
            size_t end = json.find("]", start + 1);
            return json.substr(start, end - start + 1);
        }
        return "";
    }

    // 辅助函数：解析 JSON 数组为 vector<int64_t>
    std::vector<int64_t> parse_json_array(const std::string& array_str) {
        std::vector<int64_t> result;
        size_t start = 1; // 跳过 '['
        while (start < array_str.length() - 1) {
            size_t end = array_str.find_first_of(",]", start);
            if (end == std::string::npos) break;
            std::string num_str = array_str.substr(start, end - start);
            // 去除空格
            num_str.erase(std::remove_if(num_str.begin(), num_str.end(), ::isspace), num_str.end());
            if (!num_str.empty()) {
                result.push_back(std::stoll(num_str));
            }
            start = end + 1;
        }
        return result;
    }
}

namespace qwen_thor {
namespace io {

SafetensorsLoader::SafetensorsLoader(const std::string& file_path) : file_path_(file_path), header_size_(0) {
    mmap_allocator_ = std::make_unique<core::MmapAllocator>(file_path_);
    parse_header();
}

SafetensorsLoader::~SafetensorsLoader() {}

void SafetensorsLoader::parse_header() {
    void* base_ptr = mmap_allocator_->get_base_ptr();
    if (base_ptr == MAP_FAILED) {
        throw std::runtime_error("Cannot parse header: mmap failed.");
    }

    // Safetensors 格式：前 8 个字节是 header 的长度 (uint64_t)
    uint64_t header_len = 0;
    std::memcpy(&header_len, base_ptr, sizeof(uint64_t));
    header_size_ = sizeof(uint64_t) + header_len;

    // 读取 JSON header
    const char* json_start = static_cast<const char*>(base_ptr) + sizeof(uint64_t);
    std::string header_json(json_start, header_len);

    // 极简解析逻辑：按顶层键分割
    // 注意：这只是一个原型解析器，假设 JSON 格式非常规整
    size_t pos = 1; // 跳过最外层的 '{'
    while (pos < header_json.length() - 1) {
        size_t key_start = header_json.find("\"", pos);
        if (key_start == std::string::npos) break;
        size_t key_end = header_json.find("\"", key_start + 1);
        std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);

        if (key == "__metadata__") {
            // 跳过 metadata
            size_t obj_start = header_json.find("{", key_end);
            size_t obj_end = header_json.find("}", obj_start);
            pos = obj_end + 1;
            continue;
        }

        size_t obj_start = header_json.find("{", key_end);
        size_t obj_end = header_json.find("}", obj_start);
        std::string obj_json = header_json.substr(obj_start, obj_end - obj_start + 1);

        TensorMetadata meta;
        meta.dtype_str = extract_json_value(obj_json, "dtype");
        meta.dtype = parse_dtype(meta.dtype_str);
        
        std::string shape_str = extract_json_value(obj_json, "shape");
        meta.shape = parse_json_array(shape_str);

        std::string offsets_str = extract_json_value(obj_json, "data_offsets");
        std::vector<int64_t> offsets = parse_json_array(offsets_str);
        if (offsets.size() == 2) {
            meta.data_offsets[0] = offsets[0];
            meta.data_offsets[1] = offsets[1];
        }

        metadata_map_[key] = meta;
        pos = obj_end + 1;
    }
}

core::DataType SafetensorsLoader::parse_dtype(const std::string& dtype_str) const {
    if (dtype_str == "F32") return core::DataType::FP32;
    if (dtype_str == "F16") return core::DataType::FP16;
    if (dtype_str == "BF16") return core::DataType::BF16;
    if (dtype_str == "I8") return core::DataType::INT8;
    // 扩展支持 Blackwell 的 FP8/FP4
    if (dtype_str == "F8_E4M3") return core::DataType::FP8_E4M3;
    if (dtype_str == "F8_E5M2") return core::DataType::FP8_E5M2;
    if (dtype_str == "F4") return core::DataType::FP4;
    
    std::cerr << "Warning: Unknown dtype '" << dtype_str << "', defaulting to UNKNOWN." << std::endl;
    return core::DataType::UNKNOWN;
}

std::unique_ptr<core::Tensor> SafetensorsLoader::get_tensor(const std::string& name) const {
    auto it = metadata_map_.find(name);
    if (it == metadata_map_.end()) {
        throw std::runtime_error("Tensor not found in safetensors: " + name);
    }

    const TensorMetadata& meta = it->second;
    
    // 计算数据在 mmap 区域中的绝对指针
    // 偏移量是相对于 header 结束位置的
    void* base_ptr = mmap_allocator_->get_base_ptr();
    char* data_ptr = static_cast<char*>(base_ptr) + header_size_ + meta.data_offsets[0];

    // 创建 Tensor 对象，接管 mmap 内存指针
    // 注意：这个 Tensor 不拥有内存，析构时不会释放
    return std::make_unique<core::Tensor>(meta.shape, meta.dtype, static_cast<void*>(data_ptr));
}

std::vector<std::string> SafetensorsLoader::get_tensor_names() const {
    std::vector<std::string> names;
    for (const auto& pair : metadata_map_) {
        names.push_back(pair.first);
    }
    return names;
}

bool SafetensorsLoader::has_tensor(const std::string& name) const {
    return metadata_map_.find(name) != metadata_map_.end();
}

} // namespace io
} // namespace qwen_thor
