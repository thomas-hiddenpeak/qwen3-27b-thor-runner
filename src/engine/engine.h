#pragma once

#include "model.h"
#include "perf_stats.h"
#include "shm_queue.h"
#include "paged_attention.h"
#include "cache_config.h"
#include "cache_manager.h"
#include "block_tracker.h"
#include "vision.h"
#include "tokenizer.h"
#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <mutex>
#include <unordered_set>

namespace qwen_thor {
namespace core {

// SSM/Conv 状态池最大槽位数 (每个活跃请求占一个独立 slot, 互不干扰)
static constexpr int MAX_SSM_SLOTS = 8;

// 表示一个正在处理的推理请求
struct RequestContext {
    uint64_t request_id;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    int max_new_tokens;
    
    // 采样参数
    float temperature = 1.0f;
    float top_p       = 0.95f;
    int   top_k       = 20;
    float min_p       = 0.0f;
    float repeat_penalty    = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty  = 0.0f;
    int64_t seed            = -1;
    
    // Paged Attention 相关的状态
    std::vector<int> block_table;
    int context_len = 0;

    // Block Tracker: 追踪 GPU vs SSD blocks (当 context 超过 KV budget 时启用)
    cache::BlockTracker block_tracker;
    bool uses_ssd_blocks = false;  // 是否有 blocks 在 SSD 上
    
    // Gated DeltaNet SSM 状态 (每个 linear_attn 层一份)
    std::vector<__nv_bfloat16*> ssm_states;
    std::vector<__nv_bfloat16*> conv_states;
    int ssm_slot = -1;  // 该请求持有的 SSM/Conv 状态池槽位 (-1 = 未分配)
    
    // MTP 投机解码状态
    std::vector<int> draft_tokens;  // 当前 draft tokens (empty = 无 draft)
    int mtp_context_len = 0;       // MTP KV cache 中已有的 token 数
    std::vector<int> mtp_block_table;  // MTP KV cache block table
    int mtp_pos = 0;               // MTP 下一次调用的绝对位置 (用于 RoPE)

    // 多模态: 预处理后的图像数据
    std::vector<core::ProcessedImage> processed_images;

    bool is_finished = false;
    bool is_swapped  = false;   // 该请求的 KV+SSM/Conv 已换出到 SSD
};

class InferenceEngine {
public:
    InferenceEngine(const Qwen35Config& config, const std::string& model_dir,
                    const cache::CacheConfig& cache_config = cache::CacheConfig());
    ~InferenceEngine();

    // 启动后台推理线程
    void start();
    
    // 停止后台推理线程
    void stop();

    // 供测试用的单步执行接口
    void step();
    void step(std::vector<RequestContext*>& active_requests);

    // ======================================================================
    // 直接推送接口 (供 InferenceBackend 进程内调用, 绕过 IPC shm)
    // ======================================================================
    bool push_request(const ipc::InferenceRequest& req)  { return ipc_queue_->push(req); }
    bool pop_response(ipc::InferenceResponse& resp)       { return ipc_resp_queue_->pop(resp); }

    // 多模态: 附加预处理后的图像到待处理请求 (线程安全)
    void attach_images(uint64_t request_id, std::vector<core::ProcessedImage>&& images) {
        std::lock_guard<std::mutex> lock(pending_images_mutex_);
        pending_images_[request_id] = std::move(images);
    }

    // 取消请求 (线程安全, 从 serve 层调用)
    void cancel_request(uint64_t request_id) {
        std::lock_guard<std::mutex> lock(cancel_mutex_);
        cancel_set_.insert(request_id);
    }

private:
    void inference_loop();
    
    // 处理新请求 (Prefill)
    void process_prefill(std::vector<RequestContext*>& active_requests);
    
    // 处理生成 (Decode)
    void process_decode(std::vector<RequestContext*>& active_requests);

    // 采样逻辑
    int sample_argmax(__nv_bfloat16* logits, int vocab_size, cudaStream_t stream);
    int sample_token(__nv_bfloat16* logits, int vocab_size,
                     float temperature, float top_p, int top_k, float min_p,
                     float repeat_penalty, float frequency_penalty,
                     float presence_penalty, int64_t seed,
                     const std::vector<int>& generated_tokens,
                     cudaStream_t stream);

    // KV 换出辅助: 当 allocate_blocks 失败时, 换出最大请求腾出空间
    // 返回成功释放的 block 数, 0 = 无可换出请求
    int try_swap_out_victim(std::vector<RequestContext*>& active_requests,
                            int blocks_needed);

    // MTP draft 生成: 链式调用 N 次 mtp_forward → argmax, GPU-resident chain
    void generate_mtp_drafts(RequestContext* ctx, __nv_bfloat16* main_hidden,
                             int first_token, int start_pos, int N, int vocab_size);

    std::string token_to_log_text(int token_id) const;

    Qwen35Config config_;
    int num_linear_layers_ = 0;    // 线性注意力层数 (48)
    size_t ssm_size_per_layer_ = 0;
    size_t conv_size_per_layer_ = 0;

    // ======== 统一缓存管理器 (替代原先 5 个独立组件) ========
    std::unique_ptr<cache::CacheManager> cache_manager_;

    std::unique_ptr<Qwen35Model> model_;
    std::unique_ptr<ipc::ShmRingBuffer<ipc::InferenceRequest,  8>> ipc_queue_;
    std::unique_ptr<ipc::ShmRingBuffer<ipc::InferenceResponse, 512>> ipc_resp_queue_;
    
    std::vector<std::unique_ptr<RequestContext>> all_requests_;
    
    // 预分配的显存
    __nv_bfloat16* d_workspace_ = nullptr;
    __nv_bfloat16* d_hidden_states_ = nullptr;
    int* d_pos_ids_ = nullptr;
    int* d_block_tables_ = nullptr;
    int* d_context_lens_ = nullptr;
    int* d_argmax_result_ = nullptr;  // GPU argmax 结果 (managed memory, 16 ints for batched)
    
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    cudaStream_t compute_stream_;

    // 请求取消集合 (serve 层写, 引擎线程读)
    std::mutex cancel_mutex_;
    std::unordered_set<uint64_t> cancel_set_;

    // 多模态: 待附加的预处理图像 (按 request_id 索引)
    std::mutex pending_images_mutex_;
    std::unordered_map<uint64_t, std::vector<core::ProcessedImage>> pending_images_;

    // Chunked prefill: 当 prompt 超过此值时自动分块处理
    int max_chunk_size_ = 256;  // Jetson SMMU 硬约束: >256 的 chunk 在长上下文时 →  GPU illegal memory access

    // 容量上限: 单请求最大 token 数 (= gpu_kv_blocks × block_size)
    int gpu_max_tokens_ = 0;

    // Vision encoder workspace (预分配, 用于最大图像尺寸)
    __nv_bfloat16* d_vision_workspace_ = nullptr;
    size_t vision_workspace_bytes_ = 0;

    // MTP 投机解码相关
    std::unique_ptr<ops::KVCacheManager> mtp_kv_manager_;  // 1 层 KV cache
    __nv_bfloat16* d_ssm_checkpoints_ = nullptr;         // SSM 状态 checkpoint [48 layers contiguous]
    __nv_bfloat16* d_conv_checkpoints_ = nullptr;// Conv 状态 checkpoint [48 layers contiguous]
    int* d_mtp_block_tables_ = nullptr;          // MTP KV block table (device)
    int* d_mtp_context_lens_ = nullptr;          // MTP KV context lens (device)
    size_t ssm_elems_per_layer_ = 0;             // SSM state elements per layer
    size_t conv_elems_per_layer_ = 0;            // Conv state elements per layer
    int num_mtp_drafts_ = 3;                   // 每步 draft token 数 (从配置读取, 1~8)
    int mtp_verify_count_ = 0;                   // verify 步数
    int mtp_accept_count_ = 0;                   // full accept (N/N) 步数
    int mtp_total_accepted_ = 0;                 // 累计接受的 draft tokens
    int mtp_total_emitted_ = 0;                  // 累计产出的 tokens (accept + bonus)

    // 采样用预分配缓冲区 (CPU 侧)
    std::vector<float> sampling_logits_;
    std::vector<int>   sampling_indices_;
    std::vector<__nv_bfloat16> sampling_logits_bf16_; // host staging for GPU logits
    std::mt19937       sampling_rng_;

    // 调试日志: token_id -> 文本
    Tokenizer log_tokenizer_;
    bool log_tokenizer_ready_ = false;

    // 性能统计
    perf::PerfProfiler profiler_;
};

} // namespace core
} // namespace qwen_thor