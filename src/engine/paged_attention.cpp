#include "paged_attention.h"
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace qwen_thor {
namespace ops {

KVCacheManager::KVCacheManager(
    int num_blocks, 
    int block_size, 
    int num_heads, 
    int head_dim, 
    core::DataType dtype, 
    std::shared_ptr<core::Allocator> allocator,
    int num_layers
) : num_blocks_(num_blocks),
    block_size_(block_size),
    num_heads_(num_heads),
    head_dim_(head_dim),
    num_layers_(num_layers),
    dtype_(dtype) {
    
    // 初始化物理 Cache Tensor
    // 形状: [num_layers * num_blocks, block_size, num_heads, head_dim]
    // 每层独立使用 num_blocks 个 block, 通过指针偏移区分层
    int64_t total_blocks = (int64_t)num_layers_ * num_blocks_;
    std::vector<int64_t> shape = {total_blocks, (int64_t)block_size_, (int64_t)num_heads_, (int64_t)head_dim_};
    
    // KV Cache 仅 GPU 访问，使用 DeviceAllocator (cudaMalloc) 而非 UnifiedAllocator (cudaMallocManaged)
    // cudaMalloc 立即建立 GPU 页表映射，无 lazy page fault 开销
    // 在 Jetson Thor 上同样分配 LPDDR5X 物理内存，但避免了 cudaMallocManaged 的 12s+ 初始化延迟
    auto device_allocator = std::make_shared<core::DeviceAllocator>();
    k_cache_ = std::make_unique<core::Tensor>(shape, dtype_, device_allocator);
    v_cache_ = std::make_unique<core::Tensor>(shape, dtype_, device_allocator);
    
    std::cout << "KVCacheManager: " << num_layers_ << " layers x " << num_blocks_ << " blocks = " 
              << total_blocks << " total blocks, "
              << (total_blocks * block_size_ * num_heads_ * head_dim_ * 2 * 2 / (1024*1024)) << " MB (device)" << std::endl;

    // 初始化空闲 Block 列表 (0 到 num_blocks - 1)
    // Block IDs are per-layer; the layer offset is added when computing cache addresses
    free_blocks_.resize(num_blocks_);
    std::iota(free_blocks_.begin(), free_blocks_.end(), 0);
}

KVCacheManager::~KVCacheManager() {}

const __nv_bfloat16* KVCacheManager::get_layer_k_cache(int layer_idx) const {
    // Each layer has num_blocks_ blocks of size [block_size, num_heads, head_dim]
    size_t per_layer_elements = (size_t)num_blocks_ * block_size_ * num_heads_ * head_dim_;
    return static_cast<const __nv_bfloat16*>(k_cache_->data()) + layer_idx * per_layer_elements;
}

const __nv_bfloat16* KVCacheManager::get_layer_v_cache(int layer_idx) const {
    size_t per_layer_elements = (size_t)num_blocks_ * block_size_ * num_heads_ * head_dim_;
    return static_cast<const __nv_bfloat16*>(v_cache_->data()) + layer_idx * per_layer_elements;
}

__nv_bfloat16* KVCacheManager::get_layer_k_cache_mut(int layer_idx) {
    size_t per_layer_elements = (size_t)num_blocks_ * block_size_ * num_heads_ * head_dim_;
    return static_cast<__nv_bfloat16*>(k_cache_->data()) + layer_idx * per_layer_elements;
}

__nv_bfloat16* KVCacheManager::get_layer_v_cache_mut(int layer_idx) {
    size_t per_layer_elements = (size_t)num_blocks_ * block_size_ * num_heads_ * head_dim_;
    return static_cast<__nv_bfloat16*>(v_cache_->data()) + layer_idx * per_layer_elements;
}

std::vector<int> KVCacheManager::allocate_blocks(int num_blocks_needed) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (free_blocks_.size() < static_cast<size_t>(num_blocks_needed)) {
        throw std::runtime_error("Out of memory: Not enough free blocks in KVCacheManager.");
    }

    std::vector<int> allocated;
    allocated.reserve(num_blocks_needed);
    
    for (int i = 0; i < num_blocks_needed; ++i) {
        allocated.push_back(free_blocks_.back());
        free_blocks_.pop_back();
    }
    
    return allocated;
}

void KVCacheManager::free_blocks(const std::vector<int>& block_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 将释放的 blocks 放回空闲列表
    for (int idx : block_indices) {
        if (idx >= 0 && idx < num_blocks_) {
            free_blocks_.push_back(idx);
        }
    }
}

} // namespace ops
} // namespace qwen_thor
