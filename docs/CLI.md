# CLI Reference — qwen-thor

## Synopsis

```
qwen-thor <command> [options]
```

## Commands

| Command | Description |
|---------|-------------|
| `serve` | 启动 HTTP API 服务 (Ollama/OpenAI 兼容接口) |
| `chat` | 启动 TUI 交互式对话终端 |
| `bench` | 运行推理性能基准测试 |
| `test` | 运行单元测试 |
| `version` | 打印版本与环境信息 |
| `--help` | 显示帮助文档 |

---

## Engine Options

以下选项适用于 `serve`、`chat`、`bench` 所有子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config <file>` | string | — | 从配置文件加载引擎参数 |
| `--model-dir <path>` | string | `models/Qwen/Qwen3.5-27B` | 模型权重目录 |
| `--kv-cache-gb <N>` | float | `4.0` | GPU KV Cache 内存预算 (GB) |
| `--cache-enable` | flag | off | 启用 SSD 前缀缓存 |
| `--cache-dir <path>` | string | `/tmp/qwen_kv_cache` | SSD 缓存存储目录 |
| `--cache-max-gb <N>` | float | `20` | SSD 最大缓存容量 (GB) |
| `--cache-chunk-size <N>` | int | `256` | 前缀缓存 chunk 大小 (tokens) |
| `--cache-no-ssm` | flag | off | 禁用 SSM/Conv 状态缓存 |
| `--mtp-enable` | flag | — | 强制启用 MTP 投机解码 |
| `--mtp-disable` | flag | — | 强制禁用 MTP 投机解码 |
| `--mtp-kv-blocks <N>` | int | `256` | MTP KV Cache blocks 数量 |

---

## Serve Options

仅适用于 `serve` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host <addr>` | string | `0.0.0.0` | HTTP 监听地址 |
| `--port <N>` | int | `11434` | HTTP 监听端口 |
| `--max-conns <N>` | int | `64` | 最大并发连接数 |
| `--model-name <name>` | string | `qwen3.5-27b` | 模型显示名称 (API 返回值) |
| `--timeout <N>` | int | `300` | 请求超时 (秒) |
| `--serve-config <file>` | string | — | 从配置文件加载服务参数 |

---

## Chat Options

仅适用于 `chat` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max-tokens <N>` | int | `2048` | 每次回复最大生成 token 数 |
| `--temperature <F>` | float | `0.0` | 采样温度 (0.0 = greedy) |
| `--top-p <F>` | float | `1.0` | Nucleus 采样阈值 |
| `--timeout <N>` | int | `300` | 生成超时 (秒) |
| `--no-stats` | flag | off | 不显示推理性能统计 |

### TUI 内置命令

| Command | Description |
|---------|-------------|
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史 |
| `/tokens N` | 设置 max_new_tokens |
| `/temp F` | 设置 temperature |
| `/stats` | 切换性能统计显示 |
| `/exit` | 退出 |

---

## Bench Options

仅适用于 `bench` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup <N>` | int | `5` | 预热 decode 步数 |
| `--decode <N>` | int | `30` | 测量的 decode 步数 |
| `--prompt-len <N>` | int | `64` | 输入 prompt 长度 (tokens) |
| `--batch <N>` | int | `1` | Batch size |
| `--csv` | flag | off | 以 CSV 格式输出 |
| `--nsys` | flag | off | 启用 NVTX 标记 (配合 nsys profiling) |

---

## Examples

### 基础推理服务

```bash
# 使用默认配置启动 (4GB KV cache, 端口 11434)
qwen-thor serve

# 使用 8GB KV cache, 自定义端口
qwen-thor serve --kv-cache-gb 8 --port 8080

# 使用配置文件
qwen-thor serve --config configs/engine.conf --serve-config configs/serve.conf
```

### SSD 缓存加速

```bash
# 启用 SSD 前缀缓存 (4GB GPU + 20GB SSD)
qwen-thor serve --kv-cache-gb 4 --cache-enable --cache-max-gb 20

# 256K 超长上下文 (8GB GPU + SSD offload)
qwen-thor serve --kv-cache-gb 8 --cache-enable --cache-dir /mnt/nvme/kv_cache
```

### 交互式 Chat

```bash
# 默认设置
qwen-thor chat --kv-cache-gb 4

# 高温采样
qwen-thor chat --temperature 0.8 --top-p 0.95 --max-tokens 4096
```

### 性能评估

```bash
# 基础 benchmark (单并发, 64 token prompt)
qwen-thor bench

# 长上下文 benchmark
qwen-thor bench --prompt-len 4096 --decode 50

# 并发 benchmark
qwen-thor bench --batch 8 --decode 30

# CSV 输出 (用于自动化)
qwen-thor bench --decode 100 --csv > results.csv

# nsys profiling
nsys profile --trace=cuda,nvtx -o profile qwen-thor bench --decode 10 --nsys
```

### API 调用示例

```bash
# OpenAI 格式 (curl)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 512,
    "stream": true
  }'

# Ollama 格式
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3.5-27b", "prompt": "Why is the sky blue?"}'

# 健康检查
curl http://localhost:11434/health

# 模型列表
curl http://localhost:11434/v1/models
```

### Python 客户端 (via OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Configuration Files

### Engine Config (`configs/engine.conf`)

```properties
# 模型与推理引擎配置
model_dir=/home/rm01/runner/models/Qwen/Qwen3.5-27B
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
# HTTP 服务配置
host=0.0.0.0
port=11434
max_conns=64
model_name=qwen3.5-27b
timeout=300
```

配置文件使用简单的 `key=value` 格式, `#` 开头为注释。CLI 参数优先于配置文件。
