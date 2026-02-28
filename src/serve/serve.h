// serve.h — HTTP API 服务 (Ollama / OpenAI 兼容接口)
//
// 提供 RESTful API 端点:
//   POST /v1/chat/completions     — OpenAI Chat Completions (streaming/non-streaming)
//   POST /v1/completions          — OpenAI Completions
//   POST /api/generate            — Ollama Generate
//   POST /api/chat                — Ollama Chat
//   GET  /v1/models               — 模型列表
//   GET  /api/tags                — Ollama 模型列表
//   GET  /health                  — 健康检查
//
// 使用 POSIX socket 实现轻量级 HTTP 服务, 无外部依赖。

#pragma once

#include "../engine/backend.h"
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <deque>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <condition_variable>

namespace qwen_thor {
namespace serve {

// ============================================================================
// 服务配置
// ============================================================================
struct ServeConfig {
    std::string host      = "0.0.0.0";
    int         ollama_port = 11434;        // Ollama API 端口
    int         openai_port = 8080;         // OpenAI API 端口
    int         max_conns = 64;             // 最大并发连接
    std::string model_name = "qwen3.5-27b"; // 模型显示名称
    int         timeout_s  = 300;           // 请求超时 (秒)

    // 从 CLI 参数解析 (全新配置)
    static ServeConfig from_args(int argc, char** argv);
    // 在已有配置基础上应用 CLI 覆盖
    static ServeConfig merge_args(const ServeConfig& base, int argc, char** argv);
    // 从配置文件解析
    static ServeConfig from_file(const std::string& path);

    void print() const;
};

// ============================================================================
// HTTP 请求/响应结构
// ============================================================================
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    int client_fd = -1;
};

struct HttpResponse {
    int         status_code = 200;
    std::string status_text = "OK";
    std::string content_type = "application/json";
    std::string body;
    bool        is_streaming = false;
};

// ============================================================================
// ServeApp — HTTP API 服务
// ============================================================================
class ServeApp {
public:
    ServeApp(const ServeConfig& config, InferenceBackend& backend);
    ~ServeApp();

    // 启动 HTTP 服务 (阻塞主线程)
    void run();

    // 停止服务
    void stop();

private:
    // 接受连接并分发到 handler
    void accept_loop();

    // 处理单个 HTTP 连接 (protocol: 0=ollama, 1=openai)
    void handle_connection(int client_fd, int protocol);

    // 解析 HTTP 请求
    HttpRequest parse_request(int client_fd);

    // 发送 HTTP 响应
    void send_response(int client_fd, const HttpResponse& resp);

    // 发送 SSE (Server-Sent Events) 事件
    void send_sse_event(int client_fd, const std::string& data);
    void send_sse_done(int client_fd);

    // NDJSON chunked streaming (Ollama protocol)
    void send_ndjson_chunk(int client_fd, const std::string& json_line);
    void send_chunked_end(int client_fd);

    // ---- Tool Call 信息结构 ----
    struct ToolCallInfo {
        std::string id;         // "call_xxxxxxxxxxxx"
        std::string name;       // 函数名
        std::string arguments;  // 参数 JSON 字符串
    };

    // Common inference polling — calls on_token for each decoded piece,
    // returns completion token count.
    // on_reasoning: 可选回调, 接收 thinking 内容 (用于 reasoning_content 输出)
    // on_tool_call: 可选回调, 当检测到完整 <tool_call>...</tool_call> 时调用
    // out_finish_reason: 输出 finish_reason ("stop" 或 "tool_calls")
    int poll_tokens(uint64_t request_id,
                    const std::function<void(const std::string&)>& on_token,
                    int timeout_s = 300,
                    bool start_in_thinking = true,
                    const std::vector<std::string>& stop_seqs = {},
                    const std::function<void(const std::string&)>& on_reasoning = {},
                    const std::function<void(const ToolCallInfo&)>& on_tool_call = {},
                    std::string* out_finish_reason = nullptr);

    // ---- API 路由 ----
    void handle_health(const HttpRequest& req, int client_fd);
    void handle_models(const HttpRequest& req, int client_fd);
    void handle_model_retrieve(const HttpRequest& req, int client_fd);
    void handle_ollama_tags(const HttpRequest& req, int client_fd);
    void handle_ollama_show(const HttpRequest& req, int client_fd);
    void handle_ollama_ps(const HttpRequest& req, int client_fd);
    void handle_ollama_version(const HttpRequest& req, int client_fd);
    void handle_openai_chat(const HttpRequest& req, int client_fd);
    void handle_openai_completions(const HttpRequest& req, int client_fd);
    void handle_ollama_generate(const HttpRequest& req, int client_fd);
    void handle_ollama_chat(const HttpRequest& req, int client_fd);
    void handle_cors_preflight(const HttpRequest& req, int client_fd);

    // JSON 辅助
    std::string make_chat_chunk(const std::string& model, const std::string& content,
                                const std::string& finish_reason, const std::string& id,
                                int64_t created);
    std::string make_chat_reasoning_chunk(const std::string& model, const std::string& reasoning,
                                          const std::string& id, int64_t created);
    std::string make_chat_tool_call_chunk(const std::string& model, const ToolCallInfo& tc,
                                           int index, const std::string& id, int64_t created);
    std::string make_completion_chunk(const std::string& model, const std::string& text,
                                      const std::string& finish_reason, const std::string& id,
                                      int64_t created);

    // 生成唯一请求 ID
    uint64_t next_request_id();

    // 响应分发线程: 从 backend_ 单消费者队列读取, 路由到 per-request queues
    void response_dispatch_loop();

    // 注册/注销 per-request 响应队列
    void register_request(uint64_t request_id);
    void unregister_request(uint64_t request_id);
    // 从 per-request 队列取一条响应 (阻塞至有数据或超时)
    bool poll_request(uint64_t request_id, InferResponse& resp, int timeout_ms = 100);

    ServeConfig config_;
    InferenceBackend& backend_;
    int ollama_fd_ = -1;
    int openai_fd_ = -1;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> req_id_counter_{1};
    std::atomic<int> active_workers_{0};
    std::thread resp_dispatcher_;  // 响应分发线程

    // Per-request 响应队列 (解决并发 token 窃取问题)
    std::mutex resp_mutex_;
    std::condition_variable resp_cv_;
    std::unordered_map<uint64_t, std::deque<InferResponse>> resp_queues_;
};

} // namespace serve
} // namespace qwen_thor
