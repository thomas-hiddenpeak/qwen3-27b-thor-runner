#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <stdexcept>

namespace qwen_thor {
namespace ipc {

// -----------------------------------------------------------------------------
// 通信数据结构定义
// -----------------------------------------------------------------------------

constexpr int MAX_PROMPT_LEN = 262144;  // 256K tokens (= model max_position_embeddings)

// 前端发送给后端的推理请求
struct InferenceRequest {
    uint64_t request_id;
    int32_t prompt_tokens[MAX_PROMPT_LEN];
    int32_t prompt_len;
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;               // min-p 过滤阈值
    float repeat_penalty;      // 重复惩罚 (1.0=无惩罚)
    float frequency_penalty;   // 频率惩罚 (OpenAI 风格)
    float presence_penalty;    // 存在性惩罚 (OpenAI 风格)
    int64_t seed;              // 随机种子 (-1=随机)
    // 标志位：是否是流式输出
    bool stream; 
};

// 后端返回给前端的推理响应
struct InferenceResponse {
    uint64_t request_id;
    int32_t token_id;
    // 标志位：是否是该请求的最后一个 token
    bool is_finished; 
    // 错误码，0 表示成功
    int32_t error_code; 
};

// -----------------------------------------------------------------------------
// 基于 POSIX 共享内存的单生产者单消费者 (SPSC) 无锁环形队列
// -----------------------------------------------------------------------------
template <typename T, size_t Capacity>
class ShmRingBuffer {
public:
    // 队列头部控制信息，存放在共享内存的最前面
    struct Header {
        std::atomic<size_t> head{0}; // 生产者写入位置
        std::atomic<size_t> tail{0}; // 消费者读取位置
    };

    ShmRingBuffer(const std::string& shm_name, bool is_creator);
    ~ShmRingBuffer();

    // 禁用拷贝
    ShmRingBuffer(const ShmRingBuffer&) = delete;
    ShmRingBuffer& operator=(const ShmRingBuffer&) = delete;

    // 生产者写入 (非阻塞，如果满了返回 false)
    bool push(const T& item);

    // 消费者读取 (非阻塞，如果空了返回 false)
    bool pop(T& item);

private:
    std::string shm_name_;
    bool is_creator_;
    int shm_fd_;
    void* mapped_addr_;
    size_t mapped_size_;

    Header* header_;
    T* data_;
};

// -----------------------------------------------------------------------------
// 模板实现部分 (通常放在头文件中)
// -----------------------------------------------------------------------------
} // namespace ipc
} // namespace qwen_thor

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

namespace qwen_thor {
namespace ipc {

template <typename T, size_t Capacity>
ShmRingBuffer<T, Capacity>::ShmRingBuffer(const std::string& shm_name, bool is_creator)
    : shm_name_(shm_name), is_creator_(is_creator), shm_fd_(-1), mapped_addr_(MAP_FAILED) {
    
    mapped_size_ = sizeof(Header) + sizeof(T) * Capacity;

    if (is_creator_) {
        // 创建者：创建并初始化共享内存
        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (shm_fd_ == -1) throw std::runtime_error("shm_open failed for creator");
        
        if (ftruncate(shm_fd_, mapped_size_) == -1) {
            close(shm_fd_);
            shm_unlink(shm_name_.c_str());
            throw std::runtime_error("ftruncate failed");
        }
    } else {
        // 消费者：打开已存在的共享内存
        shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
        if (shm_fd_ == -1) throw std::runtime_error("shm_open failed for consumer");
    }

    mapped_addr_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (mapped_addr_ == MAP_FAILED) {
        close(shm_fd_);
        if (is_creator_) shm_unlink(shm_name_.c_str());
        throw std::runtime_error("mmap failed");
    }

    header_ = static_cast<Header*>(mapped_addr_);
    data_ = reinterpret_cast<T*>(static_cast<char*>(mapped_addr_) + sizeof(Header));

    if (is_creator_) {
        // 初始化原子变量 (Placement new)
        new (&header_->head) std::atomic<size_t>(0);
        new (&header_->tail) std::atomic<size_t>(0);
    }
}

template <typename T, size_t Capacity>
ShmRingBuffer<T, Capacity>::~ShmRingBuffer() {
    if (mapped_addr_ != MAP_FAILED) {
        munmap(mapped_addr_, mapped_size_);
    }
    if (shm_fd_ != -1) {
        close(shm_fd_);
    }
    if (is_creator_) {
        shm_unlink(shm_name_.c_str());
    }
}

template <typename T, size_t Capacity>
bool ShmRingBuffer<T, Capacity>::push(const T& item) {
    size_t current_head = header_->head.load(std::memory_order_relaxed);
    size_t next_head = (current_head + 1) % Capacity;

    if (next_head == header_->tail.load(std::memory_order_acquire)) {
        return false; // 队列满
    }

    data_[current_head] = item;
    header_->head.store(next_head, std::memory_order_release);
    return true;
}

template <typename T, size_t Capacity>
bool ShmRingBuffer<T, Capacity>::pop(T& item) {
    size_t current_tail = header_->tail.load(std::memory_order_relaxed);

    if (current_tail == header_->head.load(std::memory_order_acquire)) {
        return false; // 队列空
    }

    item = data_[current_tail];
    size_t next_tail = (current_tail + 1) % Capacity;
    header_->tail.store(next_tail, std::memory_order_release);
    return true;
}

} // namespace ipc
} // namespace qwen_thor
