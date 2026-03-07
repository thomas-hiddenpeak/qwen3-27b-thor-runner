# Qwen3.5 Thor Inference Engine

High-performance BF16 / NVFP4 inference engine for **Qwen3.5** model family (27B / 9B / 4B) on **NVIDIA Jetson AGX Thor** (SM110a Blackwell), written in C++17 / CUDA.

> **Note:** Model weights are not included. Download from [Hugging Face](https://huggingface.co/Qwen) and set `model_dir` in config.  
> Supported: [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · NVFP4 quantized variants

## Features

- **Pure C++17 / CUDA** — zero Python dependency, zero-copy weight loading via `safetensors`
- **Multi-model** — auto-detects model architecture from `config.json` (27B / 9B / 4B)
- **BF16 precision** — full BF16 pipeline (weights / activations / KV Cache), FP32 SSM state
- **NVFP4 (W4A16)** — FP4 E2M1 quantized inference, +17% decode throughput over BF16
- **Hybrid architecture** — DeltaNet SSM + GQA Full Attention (e.g. 27B: 48+16 = 64 layers)
- **MTP speculative decoding** — built-in Multi-Token Prediction with partial accept (d=3, +27.5%)
- **Paged KV Cache** — dynamic block allocation with continuous batching
- **Split-K Paged Attention** — decode attention parallelized across partitions
- **SSD prefix caching** — KV + SSM/Conv state offload to NVMe, LRU eviction
- **Streaming Attention** — GPU + SSD hybrid paged attention for 256K+ context
- **Vision** — image & video understanding (27-layer ViT + merger, ~461M params)
- **Dual-port HTTP API** — Ollama-compatible (port 11434) + OpenAI-compatible (port 8080)
- **TUI Chat** — interactive terminal chat interface
- **GPU Sampling** — Gumbel-Max fast path + top_k / top_p / min_p / presence penalty
- **Benchmark** — parameter sweep, multi-iteration statistics, 95% CI, JSON export
- **Test Framework** — 16 tests across 3 categories with filtering & timing

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
git clone --recursive https://github.com/thomas-hiddenpeak/qwen3-27b-thor-runner.git
cd qwen3-27b-thor-runner
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
# Edit configs/config.conf — set model_dir to your model weights path
# Supports: Qwen3.5-27B / 9B / 4B (BF16 or NVFP4)

# Start HTTP API server (Ollama + OpenAI dual-port)
./build/qwen3-27b-thor serve --config configs/config.conf

# Interactive TUI chat
./build/qwen3-27b-thor chat --config configs/config.conf

# Benchmarks (parameter sweep with statistics)
./build/qwen3-27b-thor bench --decode 30 --batch 1,2,4 --prompt-len 64,256 --iterations 3

# Unit tests
./build/qwen3-27b-thor test              # run unit tests only
./build/qwen3-27b-thor test --list       # list all tests
./build/qwen3-27b-thor test --all        # run all (unit + integration + benchmark)
./build/qwen3-27b-thor test --filter kv  # filter by name

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
# Model (auto-detects architecture from config.json)
model_dir=/path/to/Qwen3.5-27B      # or 9B, 4B, NVFP4 variants

# GPU KV Cache
kv_cache_budget_gb=8.0               # GB, ~16K tokens per GB

# SSD Prefix Cache
cache_enabled=true
cache_dir=/tmp/qwen_kv_cache
max_cache_gb=20

# MTP Speculative Decoding
mtp_mode=auto                        # auto / on / off
mtp_num_drafts=3                     # draft tokens per step (1-8)

# HTTP Server (dual-port)
host=0.0.0.0
ollama_port=11434
openai_port=8080
max_conns=64
model_name=qwen3.5-27b
timeout=300
```

Pre-built configs for different models: `configs/4b.conf`, `configs/9b.conf`, `configs/nvfp4.conf`.

CLI arguments always override config file values. See [docs/CLI.md](docs/CLI.md) for the full option reference.

## Project Structure

```
.
├── CMakeLists.txt
├── configs/
│   ├── config.conf              # Unified default config (engine + serve)
│   ├── 4b.conf / 9b.conf       # Model-specific configs
│   ├── nvfp4.conf               # NVFP4 quantized model config
│   └── serve.conf               # Serve-only overrides
├── docs/
│   ├── API.md                   # Backend API reference
│   ├── CLI.md                   # CLI option reference
│   ├── BUILD_AND_IMPLEMENTATION.md
│   ├── OPTIMIZATION_LOG.md      # Performance optimization journal
│   └── archive/                 # Completed design docs
├── src/
│   ├── main.cpp                 # Entry point (serve/chat/bench/test/probe/version)
│   ├── benchmark.cpp            # Benchmark with sweep, iterations, JSON
│   ├── tests.cpp                # Test framework (16 tests, 3 categories)
│   ├── engine/                  # Core inference engine (~60 files)
│   │   ├── engine.h/cpp         # Inference loop, continuous batching, MTP
│   │   ├── backend.h/cpp        # Thread-safe backend interface
│   │   ├── model.h/cpp          # Forward pass, weight loading, multi-model
│   │   ├── layer.h/cu           # Qwen35Config, FullAttn/LinearAttn layers
│   │   ├── light_ops.h/cu       # Fused kernels (RMSNorm, RoPE, SiLU, DeltaNet, Sampling)
│   │   ├── dense_gemm*.h/cu     # CUTLASS GEMM + scattered GEMV (BF16 + FP4)
│   │   ├── gdn_umma_sm110.*     # GDN WY chunked prefill kernel
│   │   ├── paged_attention.*    # Paged KV Cache + Split-K/Chunked attention
│   │   ├── streaming_attention.* # GPU+SSD hybrid attention (256K+)
│   │   ├── cache_engine.h/cpp   # SSD prefix cache with LRU eviction
│   │   ├── kv_swapper.h/cpp     # KV+SSM+Conv state SSD offload
│   │   ├── allocator.h/cpp      # GPU memory allocators
│   │   ├── tokenizer.h/cpp      # BPE tokenizer
│   │   ├── vision.h/cu          # ViT vision encoder
│   │   ├── safetensors.h/cpp    # Zero-copy safetensors loader
│   │   └── ...                  # tensor, perf_stats, shm_queue, etc.
│   ├── serve/
│   │   └── serve.h/cpp          # Dual-port HTTP server (Ollama + OpenAI)
│   └── tui/
│       └── tui.h/cpp            # Interactive terminal chat
├── tests/
│   ├── integration/             # Python integration test scripts
│   └── assets/                  # Test media files
└── third_party/
    ├── cutlass/                 # NVIDIA CUTLASS (submodule)
    ├── flashinfer/              # FlashInfer headers (submodule)
    └── stb/                     # stb_image.h
```

## Model Architecture

Qwen3.5 uses a hybrid attention architecture with DeltaNet SSM + Full Attention:

| Parameter | 27B | 9B | 4B |
|-----------|-----|-----|-----|
| Layers | 64 (48 SSM + 16 Attn) | 32 (24+8) | 32 (24+8) |
| Hidden size | 5120 | 4096 | 2560 |
| Intermediate size | 17408 | 12288 | 9216 |
| Vocab size | 248320 | 248320 | 248320 |
| Full Attn Q heads | 24 | 24 | 16 |
| Full Attn KV heads | 4 (GQA) | 4 | 4 |
| Head dim | 256 | 128 | 128 |
| Linear Attn key heads | 16 | 16 | 8 |
| Linear Attn value heads | 48 | 24 | 16 |
| tie_word_embeddings | false | false | true |
| Model size (BF16) | ~51.7 GB | ~18 GB | ~8.7 GB |

- **Linear Attention** layers (Gated DeltaNet SSM): `layer_idx % 4 != 3`
- **Full Attention** layers (GQA + RoPE + Paged KV Cache): `layer_idx % 4 == 3`
- **NVFP4** variants reduce weight memory by ~60% via FP4 E2M1 quantization

## Performance

Measured on Jetson AGX Thor (MAXN, Qwen3.5-27B BF16, MTP d=3, KV cache 8 GB):

| Metric | Value |
|--------|-------|
| Decode throughput (single request) | ~4.3 tok/s (~220 GB/s, 80% peak) |
| MTP accept rate | ~70% (d=3, partial accept) |
| MTP throughput boost | +27.5% over single-token decode |
| NVFP4 decode boost | +17% over BF16 |

Key optimizations applied:
- **GEMV/GEMM**: Scattered GEMV, Dual GEMV, GEMV+Add fusion, CUTLASS SM110 GEMM
- **Weight merging**: QKV merge (32 launches saved), QKVZAB super-merge (144 launches saved)
- **Fused RMSNorm+GEMV**: norm in SMEM, 64 launches & ~1ms saved
- **Fused QK_norm+RoPE**: deinterleave + norm + RoPE in single kernel (32 launches saved)
- **MTP**: GPU-resident draft chain, batched argmax verify, partial accept
- **DeltaNet**: WY chunked prefill (1.71× speedup), SSM state BF16 compression (+42.6% @ B=128)
- **Attention**: Split-K paged attention, chunked prefill tiled GEMM, streaming SSD attention
- **Sampling**: GPU-resident Gumbel-Max + top-k/top-p/min_p/presence penalty
- **FP4**: V2 GEMV with SMEM LUT + vectorized loads, merged FP4 QKV/GateUp projections

See [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md) for the full optimization journal.

## License

This project is licensed under the [MIT License](LICENSE).

Model weights are subject to the [Qwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-27B).
