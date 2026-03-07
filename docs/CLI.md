# CLI Reference — qwen3-27b-thor

## Synopsis

```
qwen3-27b-thor <command> [options]
```

## Commands

| Command | Description |
|---------|-------------|
| `serve` | 启动 HTTP API 服务 (Ollama/OpenAI 双端口) |
| `chat` | 启动 TUI 交互式对话终端 |
| `bench` | 运行推理性能基准测试 |
| `test` | 运行单元/集成/性能测试 |
| `probe` | SM110a 硬件原语 micro-benchmark |
| `version` | 打印版本与环境信息 |
| `--help` | 显示帮助文档 |

---

## Engine Options

以下选项适用于 `serve`、`chat`、`bench` 所有子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--config <file>` | string | — | 从配置文件加载引擎参数 |
| `--model-dir <path>` | string | (config.json auto) | 模型权重目录 (支持 27B/9B/4B/NVFP4) |
| `--kv-cache-gb <N>` | float | `4.0` | GPU KV Cache 内存预算 (GB) |
| `--cache-enable` | flag | off | 启用 SSD 前缀缓存 |
| `--cache-dir <path>` | string | `/tmp/qwen_kv_cache` | SSD 缓存存储目录 |
| `--cache-max-gb <N>` | float | `20` | SSD 最大缓存容量 (GB) |
| `--cache-chunk-size <N>` | int | `256` | 前缀缓存 chunk 大小 (tokens) |
| `--cache-no-ssm` | flag | off | 禁用 SSM/Conv 状态缓存 |
| `--mtp-enable` | flag | — | 强制启用 MTP 投机解码 |
| `--mtp-disable` | flag | — | 强制禁用 MTP 投机解码 |
| `--mtp-drafts <N>` | int | `1` | MTP draft token 数 (1-8) |
| `--mtp-kv-blocks <N>` | int | `256` | MTP KV Cache blocks 数量 |

---

## Serve Options

仅适用于 `serve` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host <addr>` | string | `0.0.0.0` | HTTP 监听地址 |
| `--ollama-port <N>` | int | `11434` | Ollama 兼容端口 |
| `--openai-port <N>` | int | `8080` | OpenAI 兼容端口 |
| `--port <N>` | int | `11434` | `--ollama-port` 别名 |
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
| `--temperature <F>` | float | `1.0` | 采样温度 |
| `--top-p <F>` | float | `0.95` | Nucleus 采样阈值 |
| `--timeout <N>` | int | `300` | 生成超时 (秒) |
| `--no-stats` | flag | off | 不显示推理性能统计 |

> Chat 模式默认开启 SSD 前缀缓存 (`cache_enabled=true`)。

### TUI 内置命令

| Command | Description |
|---------|-------------|
| `/help` | 显示帮助 |
| `/clear` | 清空对话历史 |
| `/tokens N` | 设置 max_new_tokens |
| `/temp F` | 设置 temperature |
| `/stats` | 切换性能统计显示 |
| `/nothink` | 切换思考模式 (enable_thinking) |
| `/exit` | 退出 |

---

## Bench Options

仅适用于 `bench` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup <N>` | int | `5` | 预热 decode 步数 |
| `--decode <N>` | int | `50` | 测量的 decode 步数 |
| `--prompt-len <N[,N,...]>` | int(s) | `17` | 输入 prompt 长度, **逗号分隔多值扫描** |
| `--batch <N[,N,...]>` | int(s) | `1` | Batch size, **逗号分隔多值扫描** |
| `--iterations <N>` | int | `1` | 每配置独立迭代次数 (≥2 输出 95% CI) |
| `--prefill-repeat <N>` | int | `3` | Prefill 重复测量次数 |
| `--json <file>` | string | — | 结构化 JSON 输出文件 |
| `--csv` | flag | off | 以 CSV 格式输出 |
| `--per-step` | flag | off | 打印每步详情 |
| `--no-graph` | flag | off | 禁用 CUDA Graph |
| `--nsys` | flag | off | 启用 NVTX 标记 (配合 nsys profiling) |

---

## Test Options

仅适用于 `test` 子命令:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--list` | flag | off | 列出所有可用测试, 不运行 |
| `--all` | flag | off | 运行全部测试 (含 integration/benchmark) |
| `--filter <str>` | string | — | 按子串匹配测试名称 |
| `--category <cat>` | string | — | 按类别过滤: `unit` / `integration` / `benchmark` |
| `--model-dir <path>` | string | — | 覆盖测试用模型目录 |

默认只运行 `unit` 类别 (10 个测试, 不需要模型权重)。`--all` 包含 integration (4) 和 benchmark (2)。

---

## Examples

### 多模型推理

```bash
# 27B BF16 (默认)
qwen3-27b-thor serve --config configs/config.conf

# 9B BF16
qwen3-27b-thor serve --config configs/9b.conf

# 4B BF16
qwen3-27b-thor serve --config configs/4b.conf

# 27B NVFP4 (量化)
qwen3-27b-thor serve --config configs/nvfp4.conf
```

### SSD 缓存加速

```bash
# 启用 SSD 前缀缓存 (4GB GPU + 20GB SSD)
qwen3-27b-thor serve --config configs/config.conf --cache-enable --cache-max-gb 20

# 256K 超长上下文 (8GB GPU + SSD offload)
qwen3-27b-thor serve --kv-cache-gb 8 --cache-enable --cache-dir /mnt/nvme/kv_cache
```

### 交互式 Chat

```bash
# 默认设置
qwen3-27b-thor chat --config configs/config.conf

# 高温采样
qwen3-27b-thor chat --temperature 0.8 --top-p 0.95 --max-tokens 4096
```

### 性能评估

```bash
# 基础 benchmark
qwen3-27b-thor bench --decode 30

# 参数扫描 (多 batch × 多 prompt 长度 × 多迭代)
qwen3-27b-thor bench --batch 1,2,4 --prompt-len 64,256 --iterations 3 --json results.json

# 长上下文 benchmark
qwen3-27b-thor bench --prompt-len 4096 --decode 50

# CSV 输出
qwen3-27b-thor bench --decode 100 --csv > results.csv

# nsys profiling
nsys profile --trace=cuda,nvtx -o profile qwen3-27b-thor bench --decode 10 --nsys
```

### 测试

```bash
# 快速 unit 测试
qwen3-27b-thor test

# 列出全部测试
qwen3-27b-thor test --all --list

# 按名称过滤
qwen3-27b-thor test --filter gemm

# 只跑 integration 类别
qwen3-27b-thor test --category integration
```

### API 调用示例

```bash
# OpenAI 格式 (双端口: 8080 或 11434 均可)
curl http://localhost:8080/v1/chat/completions \
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
curl http://localhost:8080/v1/models
```

### Python 客户端 (via OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

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

统一配置文件 (`configs/config.conf`):

```properties
# 模型 (自动检测架构)
model_dir=/path/to/Qwen3.5-27B

# GPU KV Cache
kv_cache_budget_gb=8.0

# SSD 前缀缓存
cache_enabled=true
cache_dir=/tmp/qwen_kv_cache
max_cache_gb=20
chunk_size=256
cache_ssm_state=true
eviction_policy=lru

# MTP 投机解码
mtp_mode=auto
mtp_num_drafts=3
mtp_kv_blocks=256

# HTTP 服务 (双端口)
host=0.0.0.0
ollama_port=11434
openai_port=8080
max_conns=64
model_name=qwen3.5-27b
timeout=300
max_output_tokens=32768
```

预置模型配置: `4b.conf` / `9b.conf` / `nvfp4.conf`。

配置文件使用 `key=value` 格式, `#` 开头为注释。CLI 参数优先于配置文件。
