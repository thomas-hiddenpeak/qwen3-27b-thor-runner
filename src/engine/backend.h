// backend.h — 独立后端引擎接口
//
// InferenceBackend 封装了完整的推理引擎生命周期:
//   1. 初始化 (加载模型权重、分配 KV Cache、创建 CUDA stream)
//   2. 提交推理请求 (线程安全)
//   3. 轮询生成结果 (非阻塞)
//   4. 停止与清理
//
// 此接口与传输层 (IPC/HTTP/TUI) 无关，可嵌入任意前端。

#pragma once

#include "engine.h"
#include "cache_config.h"
#include "layer.h"
#include "tokenizer.h"
#include <functional>
#include <mutex>
#include <queue>
#include <atomic>
#include <string>
#include <condition_variable>
#include <unordered_set>

namespace qwen_thor {

// ============================================================================
// 图像数据 (多模态输入)
// ============================================================================
struct ImageData {
    std::vector<uint8_t> pixels;  // RGB raw pixels [height, width, 3]
    int width  = 0;
    int height = 0;
};

// ============================================================================
// 视频数据 (多模态输入 — 帧序列)
// ============================================================================
struct VideoData {
    std::vector<std::vector<uint8_t>> frames;  // 帧序列, 每帧 RGB [height, width, 3]
    int width  = 0;                            // 统一帧宽度
    int height = 0;                            // 统一帧高度
    float source_fps = 24.0f;                  // 原始视频 FPS (用于时间戳计算)
};

// ============================================================================
// 推理请求 (提交给 Backend)
// ============================================================================
struct InferRequest {
    uint64_t    request_id;
    std::vector<int> prompt_tokens;
    int         max_new_tokens  = 512;
    float       temperature     = 1.0f;
    float       top_p           = 0.95f;
    int         top_k           = 20;
    float       repeat_penalty  = 1.0f;   // 重复惩罚 (1.0=无惩罚, >1.0 抑制重复)
    float       frequency_penalty = 0.0f; // 频率惩罚 (OpenAI 风格, 0.0=无惩罚)
    float       presence_penalty  = 0.0f; // 存在性惩罚 (OpenAI 风格, 0.0=无惩罚)
    int64_t     seed            = -1;     // 随机种子 (-1=随机, >=0 确定性采样)
    bool        stream          = true;

    // 多模态: 图像输入 (可选, 0 或多张图)
    std::vector<ImageData> images;

    // 多模态: 视频输入 (可选, 0 或多段视频, 每段为帧序列)
    std::vector<VideoData> videos;
};

// ============================================================================
// 推理响应 (从 Backend 轮询)
// ============================================================================
struct InferResponse {
    uint64_t    request_id;
    int32_t     token_id;
    bool        is_finished;
    int32_t     error_code;     // 0 = 成功
};

// ============================================================================
// 后端配置 (聚合 Engine 配置 + Cache 配置)
// ============================================================================
struct BackendConfig {
    std::string model_dir   = "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";
    double      kv_cache_gb = 4.0;

    // SSD 前缀缓存
    bool        cache_enabled    = false;
    std::string cache_dir        = "/tmp/qwen_kv_cache";
    double      cache_max_gb     = 20.0;
    int         cache_chunk_size = 256;
    bool        cache_ssm_state  = true;
    std::string eviction_policy  = "lru";

    // MTP 投机解码
    std::string mtp_mode         = "auto";
    int         mtp_kv_blocks    = 256;

    // 从 CLI 参数构建
    static BackendConfig from_args(int argc, char** argv);

    // 从配置文件构建
    static BackendConfig from_file(const std::string& path);

    // 转换为内部 CacheConfig
    cache::CacheConfig to_cache_config() const;

    void print() const;
};

// ============================================================================
// InferenceBackend — 独立推理后端
// ============================================================================
class InferenceBackend {
public:
    explicit InferenceBackend(const BackendConfig& config);
    ~InferenceBackend();

    // 禁用拷贝
    InferenceBackend(const InferenceBackend&) = delete;
    InferenceBackend& operator=(const InferenceBackend&) = delete;

    // 启动后台推理线程
    void start();

    // 停止后台推理线程 (阻塞直到线程退出)
    void stop();

    // 提交推理请求 (线程安全, 非阻塞)
    // 返回 true 表示成功入队, false 表示队列已满
    bool submit(const InferRequest& request);

    // 轮询生成结果 (线程安全, 非阻塞)
    // 返回 true 表示成功取出一个 response, false 表示当前无可用结果
    bool poll(InferResponse& response);

    // 取消请求
    void cancel(uint64_t request_id);

    // 是否正在运行
    bool is_running() const { return running_.load(); }

    // 活跃请求数
    int active_request_count() const;

    // 获取配置
    const BackendConfig& config() const { return config_; }

    // 获取 tokenizer (由 Backend 拥有)
    const Tokenizer& tokenizer() const { return tokenizer_; }

private:
    void inference_loop();

    BackendConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    core::Qwen35Config model_config_;
    Tokenizer tokenizer_;

    // 请求队列 (前端 → 后端)
    std::queue<InferRequest> request_queue_;
    std::mutex request_mutex_;

    // 响应队列 (后端 → 前端)
    std::queue<InferResponse> response_queue_;
    std::mutex response_mutex_;

    std::thread worker_;
    std::atomic<bool> running_{false};
    std::condition_variable cv_;
};

} // namespace qwen_thor
