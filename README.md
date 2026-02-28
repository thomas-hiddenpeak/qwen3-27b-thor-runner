# Qwen3.5-27B Thor Inference Engine

High-performance BF16 inference engine for **Qwen3.5-27B** on **NVIDIA Jetson AGX Thor** (SM110a Blackwell), written in C++17 / CUDA.

> **Note:** Model weights are not included. Download [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) from Hugging Face and set `model_dir` in `configs/config.conf`.

## Features

- **Pure C++17 / CUDA** — zero Python dependency, zero-copy weight loading via `safetensors`
- **BF16 precision** — full BF16 pipeline (weights / activations / KV Cache), FP32 SSM state
- **Hybrid architecture** — 64 layers (48 Gated DeltaNet SSM + 16 GQA Full Attention)
- **Paged KV Cache** — dynamic block allocation with continuous batching
- **Split-K Paged Attention** — decode attention parallelized across partitions
- **SSD prefix caching** — KV + SSM/Conv state offload to NVMe, LRU eviction
- **Streaming Attention** — GPU + SSD hybrid paged attention for 256K+ context
- **MTP speculative decoding** — built-in Multi-Token Prediction (no external draft model)
- **Vision** — image & video understanding (27-layer ViT + merger, ~461M params)
- **Dual-port HTTP API** — Ollama-compatible (port 11434) + OpenAI-compatible (port 8080)
- **TUI Chat** — interactive terminal chat interface
- **Sampling** — temperature, top_k, top_p, repeat/frequency/presence penalty, seed

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA Jetson AGX Thor — Blackwell SM110, 20 SM, 2560 CUDA Cores |
| Memory | 128 GB LPDDR5X unified memory, 273 GB/s peak (~230 GB/s measured) |
| L2 Cache | 32 MB |
| CPU | 14-core Arm Neoverse V3AE @ 2.6 GHz |
| Storage | NVMe SSD (recommended for KV cache offload) |

This engine targets the specific Jetson AGX Thor unified-memory architecture. It is **not** designed for discrete GPU servers.

## Build

```bash
git clone --recursive https://github.com/<your-org>/qwen3-27b-thor.git
cd qwen3-27b-thor
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

**Requirements:**
- CUDA Toolkit 13.0+ (SM110a)
- CMake 3.24+
- C++17 compiler + nvcc
- [CUTLASS](https://github.com/NVIDIA/cutlass) (included as submodule in `third_party/`)

## Quick Start

```bash
# Edit configs/config.conf — set model_dir to your Qwen3.5-27B weights path
# e.g. model_dir=/path/to/Qwen3.5-27B

# Start HTTP API server (Ollama + OpenAI dual-port)
./build/qwen3-27b-thor serve --config configs/config.conf

# Interactive TUI chat
./build/qwen3-27b-thor chat --config configs/config.conf

# Benchmarks
./build/qwen3-27b-thor bench --decode 30 --prompt-len 512

# Unit tests
./build/qwen3-27b-thor test

# Version info
./build/qwen3-27b-thor version
```

## API

The engine exposes two HTTP ports simultaneously:

| Port | Protocol | Endpoints |
|------|----------|-----------|
| 11434 | Ollama | `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` |
| 8080 | OpenAI | `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models` |

Both ports serve `GET /health` for readiness probes.

### Examples

```bash
# OpenAI Chat Completions (streaming)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": true
  }'

# OpenAI Completions with stop sequences
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Count: 1, 2, 3, 4,",
    "stop": ["8", "10"],
    "max_tokens": 50
  }'

# Ollama Generate
curl http://localhost:11434/api/generate \
  -d '{"model": "qwen3.5-27b", "prompt": "Why is the sky blue?"}'

# Ollama Chat with options
curl http://localhost:11434/api/chat \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Explain quicksort"}],
    "options": {"temperature": 0.5, "num_predict": 512}
  }'

# Health check
curl http://localhost:11434/health
```

### Sampling Parameters

| Parameter | OpenAI | Ollama | Default | Description |
|-----------|--------|--------|---------|-------------|
| `temperature` | ✅ | ✅ `options.temperature` | 1.0 (think) / 0.7 | Sampling temperature |
| `top_p` | ✅ | ✅ `options.top_p` | 0.95 / 0.8 | Nucleus sampling |
| `top_k` | ✅ | ✅ `options.top_k` | 20 | Top-K filtering |
| `max_tokens` | ✅ | ✅ `options.num_predict` | 4096 | Max new tokens |
| `stop` | ✅ (string/array) | ✅ `options.stop` (array) | — | Stop sequences |
| `seed` | ✅ | ✅ `options.seed` | -1 (random) | Deterministic sampling |
| `repeat_penalty` | ✅ | ✅ | 1.0 | Repetition penalty |
| `frequency_penalty` | ✅ | ✅ | 0.0 | Token frequency penalty |
| `presence_penalty` | ✅ | ✅ | 0.0 | Token presence penalty |
| `think` / `enable_thinking` | ✅ | ✅ | true | Enable reasoning mode |

## Configuration

The unified config file (`configs/config.conf`) covers both engine and serve settings:

```ini
# Model
model_dir=/path/to/Qwen3.5-27B

# GPU KV Cache
kv_cache_budget_gb=8.0        # GB, ~16K tokens per GB

# SSD Prefix Cache
cache_enabled=false
cache_dir=/tmp/qwen_kv_cache
max_cache_gb=20

# MTP Speculative Decoding
mtp_mode=auto                 # auto / on / off

# HTTP Server
host=0.0.0.0
ollama_port=11434
openai_port=8080
max_conns=64
model_name=qwen3.5-27b
timeout=300
```

CLI arguments always override config file values. Run `qwen3-27b-thor --help` for the full list.

## Project Structure

```
.
├── CMakeLists.txt
├── configs/
│   ├── config.conf             # Unified default config (engine + serve)
│   ├── engine.conf             # Engine-only config
│   └── serve.conf              # Serve-only config
├── docs/
│   └── OPTIMIZATION_LOG.md     # Performance optimization journal
├── src/
│   ├── main.cpp                # Unified entry (serve / chat / bench / test)
│   ├── benchmark.cpp           # Inference benchmarks
│   ├── core/                   # Core inference engine
│   │   ├── engine.h/cpp        # Inference loop, continuous batching, sampling
│   │   ├── backend.h/cpp       # Thread-safe backend interface
│   │   ├── model.h/cpp         # 64-layer forward, weight loading, MTP
│   │   ├── layer.h/cu          # FullAttn / LinearAttn layer impl
│   │   ├── allocator.h/cpp     # GPU unified memory allocator
│   │   └── tensor.h/cpp        # Tensor wrapper
│   ├── ops/                    # CUDA kernels & operators
│   │   ├── light_ops.h/cu      # Fused kernels (RMSNorm, RoPE, SiLU, DeltaNet…)
│   │   ├── dense_gemm*         # CUTLASS GEMM + scattered GEMV (SM110)
│   │   ├── paged_attention.*   # Paged KV Cache + Split-K attention
│   │   └── streaming_attention.* # GPU+SSD hybrid attention
│   ├── cache/                  # SSD caching subsystem
│   │   ├── cache_engine.h/cpp  # Prefix cache with LRU eviction
│   │   ├── kv_swapper.h/cpp    # KV+SSM+Conv state offload
│   │   └── disk_backend.h/cpp  # SSD storage backend
│   ├── io/
│   │   └── safetensors.h/cpp   # Zero-copy safetensors loader
│   ├── ipc/
│   │   └── shm_queue.h         # POSIX shared memory SPSC ring buffer
│   ├── serve/
│   │   └── serve.h/cpp         # Dual-port HTTP server (Ollama + OpenAI)
│   └── tui/
│       └── tui.h/cpp           # Interactive terminal chat
├── tests/
│   └── test_flashinfer_compile.cu
└── third_party/
    ├── cutlass/                # NVIDIA CUTLASS (header-only)
    └── flashinfer/             # FlashInfer headers
```

## Model Architecture

Qwen3.5-27B uses a hybrid attention architecture:

- **48 layers Linear Attention** (Gated DeltaNet SSM) — `layer_idx % 4 != 3`
- **16 layers Full Attention** (GQA + RoPE + Paged KV Cache) — `layer_idx % 4 == 3`

| Parameter | Value |
|-----------|-------|
| Hidden size | 5120 |
| Intermediate size | 17408 |
| Vocab size | 248320 |
| Full Attn Q heads | 24 |
| Full Attn KV heads | 4 (GQA) |
| Full Attn head dim | 256 |
| Linear Attn key heads | 16 |
| Linear Attn value heads | 48 |
| Linear Attn key/value dim | 128 |
| Model size (BF16) | ~51.7 GB |

## Performance

Measured on Jetson AGX Thor (MAXN power mode, BF16, single concurrent request, KV cache 8 GB):

| Context | TTFT (ms) | Decode (tok/s) | Prefill (tok/s) |
|---------|-----------|----------------|-----------------|
| 32 | 290 | 4.5 | 110 |
| 512 | 600 | 4.5 | 796 |
| 1K | 1,191 | 4.5 | 838 |
| 4K | 7,563 | 4.2 | 523 |
| 8K | 16,003 | 3.8 | 497 |
| 16K | 35,873 | 3.3 | 443 |
| 32K | 87,276 | 2.6 | 364 |

Key optimizations applied:
- Scattered GEMV with warp-level hash mapping (+9.5% bandwidth)
- Dual-output GEMV (gate+up share input load)
- GEMV + residual add fusion
- Split-K paged attention (2.6× decode speedup @ 32K context)
- Chunked prefill tiled GEMM attention (4.7× TTFT improvement @ 8K)
- Fused kernels: Add+RMSNorm, Deinterleave+Q-RMSNorm, Norm+SiLU+Gate, SwiGLU
- DeltaNet prefill with extended shared memory

See [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md) for the full optimization journal.

## License

This project is licensed under the [MIT License](LICENSE).

Model weights are subject to the [Qwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-27B).
