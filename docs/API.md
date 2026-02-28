# Backend Engine API Reference

## Overview

Qwen3.5-27B Thor 推理引擎提供了一个独立的 C++ 后端 (`InferenceBackend`), 与传输层 (HTTP/TUI) 完全解耦。后端可通过配置文件或 CLI 参数启动，支持线程安全的请求提交与结果轮询。

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend Layer                        │
│   ┌─────────────┐   ┌────────────┐   ┌──────────────┐   │
│   │  HTTP Serve  │   │  TUI Chat  │   │  Benchmark   │   │
│   │  (Ollama/    │   │            │   │              │   │
│   │   OpenAI)    │   │            │   │              │   │
│   └──────┬───────┘   └──────┬─────┘   └──────┬───────┘   │
│          │                  │                │            │
│          └──────────────────┼────────────────┘            │
│                             │                             │
│                    ┌────────▼────────┐                    │
│                    │ InferenceBackend │                    │
│                    │   (Thread-Safe)  │                    │
│                    └────────┬────────┘                    │
│                             │                             │
├─────────────────────────────┼────────────────────────────┤
│                    Engine Layer                           │
│                    ┌────────▼────────┐                    │
│                    │InferenceEngine  │                    │
│                    │  ┌───────────┐  │                    │
│                    │  │Qwen35Model│  │                    │
│                    │  │ (64 layers)│  │                    │
│                    │  └───────────┘  │                    │
│                    │  ┌───────────┐  │                    │
│                    │  │KVCacheManager│                    │
│                    │  └───────────┘  │                    │
│                    │  ┌───────────┐  │                    │
│                    │  │CacheEngine│  │                    │
│                    │  │ (SSD)     │  │                    │
│                    │  └───────────┘  │                    │
│                    └────────────────┘                    │
└──────────────────────────────────────────────────────────┘
```

## Core Classes

### `InferenceBackend`

**Header:** `src/engine/backend.h`

独立推理后端封装，提供线程安全的推理接口。

```cpp
class InferenceBackend {
public:
    explicit InferenceBackend(const BackendConfig& config);
    
    void start();                              // 启动推理引擎
    void stop();                               // 停止推理引擎
    bool submit(const InferRequest& request);  // 提交推理请求 (线程安全)
    bool poll(InferResponse& response);        // 轮询结果 (非阻塞)
    void cancel(uint64_t request_id);          // 取消请求
    bool is_running() const;                   // 查询运行状态
};
```

### `BackendConfig`

后端配置结构体，聚合引擎配置与缓存配置。

```cpp
struct BackendConfig {
    std::string model_dir;          // 模型权重目录
    double      kv_cache_gb;        // GPU KV Cache 预算 (GB)
    bool        cache_enabled;      // SSD 前缀缓存开关
    std::string cache_dir;          // SSD 缓存目录
    double      cache_max_gb;       // SSD 最大缓存 (GB)
    int         cache_chunk_size;   // Chunk 大小 (tokens)
    bool        cache_ssm_state;    // 缓存 SSM/Conv 状态
    std::string mtp_mode;           // MTP 模式: "auto"/"on"/"off"

    static BackendConfig from_args(int argc, char** argv);
    static BackendConfig from_file(const std::string& path);
};
```

### `InferRequest` / `InferResponse`

```cpp
struct InferRequest {
    uint64_t    request_id;
    std::vector<int> prompt_tokens; // Token IDs
    int         max_new_tokens;
    float       temperature;
    float       top_p;
    bool        stream;
};

struct InferResponse {
    uint64_t    request_id;
    int32_t     token_id;
    bool        is_finished;
    int32_t     error_code;         // 0 = 成功
};
```

### `InferenceEngine`

**Header:** `src/engine/engine.h`

核心推理引擎，管理模型、KV Cache、SSD 缓存、IPC 队列。

### `Qwen35Model`

**Header:** `src/engine/model.h`

64 层混合架构模型 (48 DeltaNet + 16 GQA)，支持 P/D 分离:
- `forward_prefill()` — 单请求, T>1, per-layer sync
- `forward_decode()` — batched decode, 无 per-layer sync
- `mtp_forward()` — MTP 投机解码

### `KVCacheManager`

**Header:** `src/engine/paged_attention.h`

Paged KV Cache, 每层独立的 block 池。

### `CacheEngine`

**Header:** `src/engine/cache_engine.h`

SSD 前缀缓存: 序列化/反序列化 KV + SSM/Conv 状态。

---

## HTTP API Endpoints

### OpenAI Compatible

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat Completions (streaming/non-streaming) |
| POST | `/v1/completions` | Text Completions |
| GET | `/v1/models` | List available models |

### Ollama Compatible

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate` | Generate text |
| POST | `/api/chat` | Chat (messages format) |
| GET | `/api/tags` | List model tags |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |

---

## Request Format

### OpenAI Chat Completions

```json
{
  "model": "qwen3.5-27b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### OpenAI Completions

```json
{
  "model": "qwen3.5-27b",
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "stream": false
}
```

### Ollama Generate

```json
{
  "model": "qwen3.5-27b",
  "prompt": "Why is the sky blue?",
  "stream": true
}
```

### Ollama Chat

```json
{
  "model": "qwen3.5-27b",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true
}
```

---

## Response Format

### OpenAI Chat (Non-Streaming)

```json
{
  "id": "chatcmpl-0",
  "object": "chat.completion",
  "created": 1740000000,
  "model": "qwen3.5-27b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

### OpenAI Chat (Streaming SSE)

```
data: {"id":"chatcmpl-0","object":"chat.completion.chunk","model":"qwen3.5-27b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-0","object":"chat.completion.chunk","model":"qwen3.5-27b","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: [DONE]
```

### Ollama Generate

```json
{"model":"qwen3.5-27b","response":"Hello!","done":true}
```

---

## Configuration

### Engine Config (`configs/engine.conf`)

```properties
model_dir=/path/to/Qwen3.5-27B
kv_cache_budget_gb=4.0
enabled=false
cache_dir=/tmp/qwen_kv_cache
max_cache_gb=20
chunk_size=256
cache_ssm_state=true
eviction_policy=lru
mtp_mode=auto
mtp_kv_blocks=256
```

### Serve Config (`configs/serve.conf`)

```properties
host=0.0.0.0
port=11434
max_conns=64
model_name=qwen3.5-27b
timeout=300
```

---

## Quick Start

```bash
# 构建
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# 启动 HTTP 服务
./qwen-thor serve --config ../configs/engine.conf --serve-config ../configs/serve.conf

# 交互式 Chat
./qwen-thor chat --kv-cache-gb 4

# 性能基准
./qwen-thor bench --decode 30 --prompt-len 512

# 使用 curl 调用 API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Hello"}]}'
```
