#include "tensor.h"
#include <stdexcept>

namespace qwen_thor {
namespace core {

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype, std::shared_ptr<Allocator> allocator)
    : shape_(shape), dtype_(dtype), allocator_(allocator) {
    numel_ = compute_numel(shape_);
    nbytes_ = numel_ * get_dtype_size(dtype_);

    if (allocator_) {
        data_ptr_ = allocator_->allocate(nbytes_);
    } else {
        throw std::invalid_argument("Allocator cannot be null when allocating memory.");
    }
}

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype, void* data_ptr)
    : shape_(shape), dtype_(dtype), data_ptr_(data_ptr), allocator_(nullptr) {
    numel_ = compute_numel(shape_);
    nbytes_ = numel_ * get_dtype_size(dtype_);
}

Tensor::~Tensor() {
    if (allocator_ && data_ptr_) {
        allocator_->deallocate(data_ptr_);
        data_ptr_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      numel_(other.numel_),
      nbytes_(other.nbytes_),
      data_ptr_(other.data_ptr_),
      allocator_(std::move(other.allocator_)) {
    other.data_ptr_ = nullptr;
    other.numel_ = 0;
    other.nbytes_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (allocator_ && data_ptr_) {
            allocator_->deallocate(data_ptr_);
        }

        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        nbytes_ = other.nbytes_;
        data_ptr_ = other.data_ptr_;
        allocator_ = std::move(other.allocator_);

        other.data_ptr_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
    }
    return *this;
}

} // namespace core
} // namespace qwen_thor
