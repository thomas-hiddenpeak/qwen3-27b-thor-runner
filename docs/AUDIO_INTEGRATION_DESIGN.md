# Qwen3 ASR/TTS 原生集成设计文档

> 版本: v0.2 — 基于实际模型分析的详细架构设计
> 日期: 2026-07
> 参考源码: `qwen-asr==0.0.6`, `qwen-tts==0.1.1` (PyPI)

## 1. 目标与约束

### 1.1 核心目标

在现有 Qwen3.5-Thor runner (C++17/CUDA) 中原生集成 Qwen3-ASR 和 Qwen3-TTS 模型系列,
实现 **语音输入 → 文本理解 → 语音输出** 的端到端推理链路, 对外提供:

1. **OpenAI 兼容 REST API** — `/v1/audio/transcriptions`, `/v1/audio/speech`
2. **WebSocket Realtime API** — 双向流式音频, 参考 OpenAI Realtime API 规范
3. **极低延迟** — 利用 GPU 统一内存/共享调度, 消除进程间通信开销

### 1.2 定位原则

- **本质是 LLM runner**, 不以提供独立 ASR/TTS 服务为目标
- ASR/TTS 作为 LLM 的 **输入编码器** 和 **输出解码器**, 与 vision encoder 同级别的一等公民
- **纯 C++ 实现** — 不可复用 Python 组件, 但参考 `qwen-asr`/`qwen-tts` 的设计实现
- **LLM 无关** — ASR/TTS 可以任意搭配当前适配的所有 LLM (4B/9B/27B/35B/122B)
- **非侵入** — ASR/TTS 的加入不应影响原先 LLM 的输入输出, 可选同步/异步/不输出
- **算子独立** — ASR/TTS 使用独立的 CUDA 算子库 (`audio_ops.h/cu`), 即使数学上与 LLM 算子相同 (RMSNorm, SwiGLU, RoPE 等) 也不共享实现。原因: (1) 参数/行为差异大 (三种 Norm, 三种 RoPE, bias 有无), (2) 音频模型和 LLM 的优化方向不同 (序列长度、batch 形态、精度), (3) 避免改一处破两处, 各自独立演进
- 最终目标: 语音对话 (Voice Chat), LLM 驱动的语音交互

### 1.3 远期目标

- **持续 ASR 输入**: 麦克风常开, 语音持续识别
- **条件触发 LLM→TTS**: 特定条件满足后才启动 TTS 输出 (如 VAD 检测到用户停止说话)
- **双输出策略**: LLM 对同一输入可产出两路不同输出 — 文本回复 vs 语音回复, 各自独立控制

### 1.4 硬件约束

| 资源 | 规格 | 备注 |
|------|------|------|
| GPU | SM110a, 20 SM, 2560 CUDA Cores | 单 GPU, 无 NVLink |
| 内存 | 128 GB LPDDR5X 统一内存 | CPU/GPU 共享, 273 GB/s 峰值 |
| 计算 | FP32 8 TFLOPS, BF16 Tensor Cores | 不是数据中心 GPU |
| CUDA Stream | 单 stream (compute_stream_) | per-layer sync, 无 CUDA graph |

---

## 2. 模型架构分析 (基于实际模型文件 + Python 源码)

### 2.1 Qwen3-ASR-1.7B

**架构**: 编码器-解码器 (`Qwen3ASRForConditionalGeneration`)

#### Audio Encoder (Whisper 风格)

```
PCM 16kHz → Mel [128, T_mel] → chunk 分段 (n_window*2=100 帧)
  → Conv2d1(1→480, k=3, s=2, pad=1) + GELU    → [B, 480, 64, T/2]
  → Conv2d2(480→480, k=3, s=2, pad=1) + GELU   → [B, 480, 32, T/4]
  → Conv2d3(480→480, k=3, s=2, pad=1) + GELU   → [B, 480, 16, T/8]
  → reshape → [B, T/8, 7680]
  → conv_out Linear(7680→1024, no bias)          → [B, T/8, 1024]
  → + sinusoidal_positional_embedding (max=1500, Whisper 风格)
  → 分 window (n_window_infer=800) 构建 cu_seqlens
  → 24 × Encoder Layer:
      Pre-LN: LayerNorm(1024)               ← 标准 LayerNorm, 不是 RMSNorm!
      → MHA(16 heads, head_dim=64, with bias) ← 双向注意力, 非因果
      → residual add
      Pre-LN: LayerNorm(1024)
      → FFN: Linear(1024→4096, bias) + GELU + Linear(4096→1024, bias)
      → residual add
  → ln_post: LayerNorm(1024)
  → proj1: Linear(1024→1024, bias=True) + GELU
  → proj2: Linear(1024→2048, bias=True)
  → output: [total_tokens, 2048]
```

**关键细节**:
- 每 100 帧 chunk 经 3 层 stride=2 conv 后产生 13 token
- 总 token 数 = `(input_frames // 100) * 13 + tail_calc`
- Encoder attention **双向** (非因果), 按 window 分块 (window 间不 attend)
- **有 bias** (Q/K/V/O 投影, FFN, proj1/2)
- 位置编码: 正弦余弦 (非学习), max=1500, channels=1024

#### Text Decoder (Qwen3 变体)

| 参数 | 值 |
|------|-----|
| hidden_size | 2048 |
| num_hidden_layers | 28 |
| num_attention_heads (Q) | 16, num_key_value_heads (KV) | 8 |
| head_dim | 128, intermediate_size | 6144 |
| vocab_size | 151936, max_position | 65536 |
| rope_theta | 1,000,000 |
| mrope_section | [24, 20, 20], interleaved |
| norm | **RMSNorm (plain weight**, 不是 (1+w)) |
| attention_bias | **False** (无 bias) |
| hidden_act | SiLU (SwiGLU) |
| Q/K norm | RMSNorm(128, eps=1e-6) |
| tie_word_embeddings | True |

**音频注入方式 — Token Replacement** (与 Vision 相同, 不是 cross-attention):
```python
audio_mask = (input_ids == 151676)  # <|audio_pad|>
inputs_embeds.masked_scatter_(audio_mask, audio_features)  # [N, 2048]
```

#### 特殊 Token

| Token | ID | 角色 |
|-------|-----|------|
| `<\|audio_start\|>` | 151669 | 音频区域开始 |
| `<\|audio_end\|>` | 151670 | 音频区域结束 |
| `<\|audio_pad\|>` | 151676 | 音频 placeholder (被 encoder 输出替换) |
| `<asr_text>` | 151704 | ASR 文本输出标记 |
| `<\|im_start\|>` / `<\|im_end\|>` | 151644/151645 | Chat 模板 |
| EOS | 151643, 151645 | 生成终止 |

#### 推理流程

1. PCM 16kHz → WhisperFeatureExtractor → log-mel `[128, T]` (n_fft=400, hop=160, 128 mel bins)
2. Encoder 逐样本 forward (不 batch, 保持精度)
3. 构建 prompt: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`
4. Token replacement: `<|audio_pad|>` 位置替换为 encoder output `[N, 2048]`
5. Decoder 自回归生成 → 输出格式: `language Chinese<asr_text>你好世界`
6. 解析: `<asr_text>` 后为实际转录文本

### 2.2 Qwen3-TTS-1.7B

**架构**: 双层自回归 (Talker + Code Predictor + Speech Tokenizer)

#### Talker (主 LM, 28 层)

```
模型结构:
├── text_embedding: Embedding(151936, 2048)    ← 文本 token
├── codec_embedding: Embedding(3072, 2048)      ← codec token (2048 码本 + 1024 特殊)
├── text_projection: ResizeMLP(2048→2048→2048, SiLU, bias=True)  ← 文本→codec 空间
├── layers: 28× Transformer Layer (与 ASR decoder 相同结构)
├── norm: RMSNorm(2048)
├── codec_head: Linear(2048→3072, no bias)      ← 预测 group-0 codec token
└── code_predictor: ...                          ← 预测 groups 1-15
```

**双轨嵌入** (关键设计):
- 每个位置的输入 = `text_projection(text_embedding(text_token))` **+** `codec_embedding(codec_token)` (加法, 非拼接)
- Decode 阶段: 所有 16 个 codebook group 嵌入**求和**为单个向量 + 流式文本残差注入

| 参数 | 1.7B | 0.6B |
|------|------|------|
| hidden_size | 2048 | 1024 |
| num_hidden_layers | 28 | 28 |
| intermediate_size | 6144 | 3072 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 8 | 8 |
| head_dim | 128 | 128 |
| text_hidden_size | 2048 | 2048 |
| rope_theta | 1,000,000 | 1,000,000 |
| mrope_section | [24, 20, 20] | [24, 20, 20] |
| position_id_per_seconds | 13 | 13 |

#### Code Predictor (子 LM, 5 层)

从 Talker hidden state 自回归预测 groups 1-15:

```
输入: past_hidden (talker 上步 hidden) + last_id_hidden (group-0 embedding)
  → small_to_mtp_projection: Linear(2048→1024, bias=True)
  → 5× Transformer Layer (hidden=1024, 16 heads, 8 KV, GQA)
     ← 标准 1D RoPE (非 MRoPE!)
  → 自回归 15 步:
     step i: codec_embedding[i-1](prev_token) → transformer → lm_head[i](hidden) → 采样
```

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| num_hidden_layers | 5 |
| intermediate_size | 3072 |
| num_attention_heads | 16, num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 2048 (codebook_size) |
| 15× codec_embedding | Embedding(2048, 2048) |
| 15× lm_head | Linear(1024, 2048, no bias) |
| RoPE | 标准 1D (非 MRoPE) |

#### 每时间步生成流程

```
Step t:
  1. Talker: codec_head(hidden) → 采样 group_0[t]
  2. CodePredictor: (talker_hidden[t], group_0[t]) → 自回归 15 步 → group_1..15[t]
  3. 下步输入: Σ(embedding_i(group_i[t]) for i=0..15) + text_stream[t+1]
```

#### 特殊 Token

| Token | ID | 域 | 角色 |
|-------|-----|-----|------|
| tts_pad | 151671 | text | 文本 padding |
| tts_bos | 151672 | text | TTS 开始 |
| tts_eos | 151673 | text | TTS 结束 |
| codec_pad | 2148 | codec | Codec padding |
| codec_bos | 2149 | codec | Codec 序列开始 |
| codec_eos | 2150 | codec | Codec 序列结束 (生成终止) |
| codec_think | 2154 | codec | 思考标签 |
| codec_think_bos/eos | 2156/2157 | codec | 思考块边界 |
| 语言: chinese | 2055 | codec | 中文标签 |
| 说话人: vivian | 3065, serena | 3066 | codec | 音色选择 |

#### Prefill 构建 (CustomVoice 模式)

```
talker_input_embeds = cat(
  text_proj(text_emb([<|im_start|>, assistant, \n])),              // [3, D] 角色
  codec_emb([think, think_bos, lang, think_eos]) + tts_pad_emb,   // [4, D] codec 前缀
  codec_emb([spk_id]) + tts_pad_emb,                               // [1, D] 说话人
  codec_emb([codec_pad]) + tts_pad_emb,                             // [1, D]
  text_proj(text_emb(文本tokens)) + codec_emb([codec_pad]×N),      // [N, D] 文本+codec_pad
  text_proj(text_emb(tts_eos)) + codec_emb(codec_pad),             // [1, D]
  tts_pad_emb + codec_emb(codec_bos),                               // [1, D] 生成起点
)
```

流式模式: 文本不在 prefill 全部注入, 而是存入 `trailing_text_hidden`, decode 时逐步 `+=`。

### 2.3 Qwen3-TTS-Tokenizer-12Hz (Speech Codec)

**架构**: Mimi 风格 CNN+Transformer (Encoder) + BigVGAN 风格 (Decoder)

#### Encoder (Audio → Codes)

```
PCM 24kHz [B, 1, T]
  → CNN Encoder (SEANet):
      Conv1d(1, 64, k=7, s=1)
      × 4 stages: ResnetBlock(ELU) + Conv1d(stride=[4,5,6,8])
                   channels: 64→128→256→512→1024
      Conv1d(1024, 512, k=3)                    总下采样: 960×
  → Transformer (8 层, d=512, 8 heads, FFN=2048, GELU)
      sliding_window=250, RoPE theta=10000, LayerScale=0.01
  → Downsample Conv1d(512, 512, k=?, s=2)       总下采样: 1920×
  → RVQ: 32 quantizers (用前 16), codebook_size=2048, dim=256
      input_proj Conv1d(512→256), output_proj Conv1d(256→512)
  → output: codes [B, 16, T_codes]  (12.5 Hz)
```

#### Decoder (Codes → Audio) — **TTS 关键路径**

```
codes [B, 16, T_codes]
  → RVQ dequantize:
      1 semantic quantizer: codebook [2048, 256] → Conv1d(256→512)
      15 acoustic quantizers: 各 codebook [2048, 256] → Conv1d(256→512)
      累加所有量化向量 → [B, 512, T]
  → pre_conv: CausalConv1d(512→1024, k=3)       → [B, 1024, T]
  → pre_transformer (8 层):
      input_proj Linear(1024→512)
      8× SlidingWindowTransformer(d=512, 16 heads MHA, head_dim=64,
           FFN=1024 SwiGLU, sliding_window=72, RoPE θ=10000, LayerScale=0.01)
      output_proj Linear(512→1024)                → [B, 1024, T]
  → Upsample ×2: TransConv(1024, k=2, s=2) + ConvNeXtBlock(dwconv+GELU)  → [B, 1024, 4T]
  → CNN Decoder (BigVGAN):
      Conv1d(1024→1536, k=7)
      × 4 stages:
        SnakeBeta → TransConv1d(stride=[8,5,4,3]) → 3× ResUnit(dil=[1,3,9])
        channels: 1536→768→384→192→96
      SnakeBeta(96) → Conv1d(96→1, k=7)           → [B, 1, 1920T]
  → clamp(-1, 1) → PCM 24kHz
```

**RVQ Codebook 注意事项**: 权重存为 `embedding_sum` + `cluster_usage` (EMA 训练遗留), 推理时需预计算: `embed = embedding_sum / cluster_usage.clamp(min=1e-5)`

**SnakeBeta 激活**: `x + (1/(exp(β)+ε)) * sin²(x * exp(α))`, α/β per-channel learnable

**Chunked Decode**: chunk_size=300 frames, left_context=25 frames, 无 KV Cache (每 chunk 重算)

### 2.4 模型关系图 (LLM 无关架构)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     InferenceBackend (多模型管理)                    │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────────┐   ┌──────────────┐    │
│  │  ASR Engine  │   │    LLM Engine        │   │  TTS Engine  │    │
│  │  (独立推理)   │   │  (任意适配的 LLM)     │   │  (独立推理)   │    │
│  │              │   │                      │   │              │    │
│  │ Encoder 24L  │   │ Qwen3.5-27B/9B/4B   │   │ Talker 28L   │    │
│  │ Decoder 28L  │   │ Qwen3.5-35B/122B    │   │ CodePred 5L  │    │
│  │   1.7B       │   │ BF16/NVFP4          │   │ Tokenizer    │    │
│  │   ~4.7 GB    │   │ 4-52 GB             │   │   ~4.5 GB    │    │
│  └──────┬───────┘   └──────────┬───────────┘   └──────┬───────┘    │
│         │                      │                       │            │
│         └──────────────────────┼───────────────────────┘            │
│                                │                                    │
│                   ┌────────────▼────────────┐                      │
│                   │   GPU SM110a (20 SM)    │                      │
│                   │   128 GB 统一内存        │                      │
│                   │   时分复用调度            │                      │
│                   └─────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

**关键**: ASR/TTS 是 **独立模型**, 不共享 LLM 的权重/KV Cache/tokenizer。

---

## 3. 内存预算分析

### 3.1 当前占用 (Qwen3.5-27B BF16)

| 组件 | 大小 | 分配方式 |
|------|------|----------|
| LLM 权重 | ~52 GB | cudaMalloc |
| KV Cache (64 slot) | ~4 GB | cudaMalloc |
| SSM State | ~4.8 GB | cudaMalloc |
| Vision Encoder | ~0.5 GB | cudaMalloc |
| Workspace | ~2 GB | cudaMalloc |
| **小计** | **~63 GB** | |

### 3.2 新增模型占用 (基于实际 safetensors 文件)

| 组件 | 大小 | 分配方式 |
|------|------|----------|
| ASR 权重 (1.7B, 2 shards) | 4.7 GB | cudaMalloc |
| ASR Decoder KV Cache (16 slot) | ~0.3 GB | cudaMalloc |
| ASR Encoder workspace | ~0.1 GB | cudaMalloc |
| TTS Talker 权重 (1.7B, 1 shard) | 3.8 GB | cudaMalloc |
| TTS Code Predictor (含 talker shard 内) | (已含上方) | |
| TTS Talker KV Cache (16 slot) | ~0.3 GB | cudaMalloc |
| TTS CodePredictor KV Cache | ~0.05 GB | cudaMalloc |
| TTS Tokenizer Decoder 权重 | 0.68 GB | cudaMalloc |
| Audio buffers (mel + PCM I/O) | ~0.1 GB | cudaMalloc |
| **新增小计** | **~10.0 GB** | |

### 3.3 总预算

| 方案 | 总占用 | 剩余 | 可行性 |
|------|--------|------|--------|
| 27B LLM + 1.7B ASR + 1.7B TTS | ~73 GB | ~55 GB | ✅ 可行 |
| 27B LLM (NVFP4) + 1.7B ASR + 1.7B TTS | ~41 GB | ~87 GB | ✅ 非常充裕 |
| 9B LLM + 1.7B ASR + 1.7B TTS | ~27 GB | ~101 GB | ✅ 轻松 |
| 4B LLM + 1.7B ASR + 1.7B TTS | ~18 GB | ~110 GB | ✅ 轻松 |

---

## 4. 集成架构设计

### 4.1 核心原则: 独立 Engine + LLM 无关 + 非侵入

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ServeApp (HTTP + WebSocket)                      │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │ REST API     │  │ WebSocket API    │  │ TUI (Chat)       │     │
│  │ /v1/audio/*  │  │ /v1/realtime     │  │                  │     │
│  │ /v1/chat/*   │  │ (bidirectional)  │  │                  │     │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘     │
│         │                   │                      │               │
│         └───────────────────┼──────────────────────┘               │
│                             │                                      │
│                 ┌───────────▼────────────┐                         │
│                 │     PipelineManager    │   ← 新增: 管线调度      │
│                 │ (ASR→LLM→TTS 编排)     │                         │
│                 └──┬──────────┬───────┬──┘                         │
│                    │          │       │                             │
│         ┌──────────▼──┐ ┌────▼────┐ ┌▼──────────┐                │
│         │ ASR Engine  │ │  LLM    │ │ TTS Engine │                │
│         │ (独立)       │ │ Backend │ │ (独立)      │                │
│         │             │ │ (任意)   │ │            │                │
│         │ load_weights│ │ submit  │ │load_weights│                │
│         │ transcribe  │ │ poll    │ │ synthesize │                │
│         │ stream_*    │ │ cancel  │ │ stream_*   │                │
│         └─────────────┘ └─────────┘ └────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 非侵入设计详述

ASR/TTS 的加入**不影响**现有 LLM 的输入输出:

| 模式 | 行为 | 用例 |
|------|------|------|
| **独立 ASR** | ASR 独立转录, 结果不送 LLM | 纯语音转文字 |
| **独立 TTS** | TTS 独立合成, 输入不来自 LLM | 纯文字转语音 |
| **ASR→LLM** (同步) | ASR 结果作为 LLM 输入, 等待 LLM 完成 | Chat 语音输入 |
| **LLM→TTS** (同步) | LLM 文本输出同步送 TTS | Chat 语音回复 |
| **LLM→TTS** (异步) | LLM 流式输出, TTS 异步消费 | 边生成边播放 |
| **LLM 无输出** | ASR触发 LLM, 但 LLM 仅更新内部状态不输出 | 后台理解 |
| **双输出** | 同一 LLM 输入产出文本+语音两路 | 远期目标 |

### 4.3 实现路径: 三个独立 Engine

```
InferenceBackend (现有, 不改动核心逻辑)
├── engine_ (InferenceEngine, 任意 LLM)
├── tokenizer_
└── [不改动]

ASREngine (新增, 独立)
├── encoder_weights_ (24L audio encoder)
├── decoder_weights_ (28L text decoder)
├── kv_cache_ (共用 decoder KV cache 管理)
├── stream_ (可独立 CUDA stream, 或复用 LLM stream)
└── mel_workspace_ (预分配)

TTSEngine (新增, 独立)
├── talker_weights_ (28L talker)
├── code_predictor_weights_ (5L)
├── tokenizer_decoder_weights_ (CNN+Transformer codec)
├── talker_kv_cache_
├── predictor_kv_cache_ (每步重建)
├── stream_
└── audio_workspace_ (预分配)

PipelineManager (新增)
├── asr_engine_
├── llm_backend_
├── tts_engine_
├── schedule(TaskType) → 时分复用
└── pipeline(AudioIn → TextOut/AudioOut)
```

---

## 5. ASR Engine 详细设计

### 5.1 核心数据结构

```cpp
struct ASRConfig {
    // Audio Encoder
    int mel_bins = 128;
    int n_fft = 400;           // 25ms @ 16kHz
    int hop_length = 160;      // 10ms
    int encoder_layers = 24;
    int encoder_d_model = 1024;
    int encoder_heads = 16;    // head_dim = 64
    int encoder_ffn = 4096;
    int downsample_hidden = 480;       // conv2d output channels
    int conv_out_dim = 7680;           // 480 * 16
    int max_source_positions = 1500;   // sinusoidal PE max
    int n_window = 50;                 // chunk = 100 frames
    int n_window_infer = 800;          // attention window size
    int output_dim = 2048;             // proj2 output

    // Text Decoder
    int decoder_layers = 28;
    int decoder_hidden = 2048;
    int decoder_heads_q = 16;
    int decoder_heads_kv = 8;
    int decoder_head_dim = 128;
    int decoder_ffn = 6144;
    int vocab_size = 151936;
    float rope_theta = 1000000.0f;
    int mrope_sections[3] = {24, 20, 20};  // interleaved
    float rms_norm_eps = 1e-6f;
};
```

### 5.2 Forward Pass (C++ 实现要点)

#### Audio Encoder

```
1. Mel Spectrogram (CPU端):
   - STFT: n_fft=400, hop=160, Hann window
   - Mel filterbank: 128 bins (Whisper 标准)
   - Log-mel + normalize

2. Conv2D Frontend (GPU):
   - 3× Conv2d(k=3, s=2, pad=1) + GELU, channels: 1→480→480→480
   - 所有 conv 有 bias
   - 输出 reshape: [B, 480, 16, T/8] → [B, T/8, 7680]
   - conv_out: Linear(7680→1024, no bias)

3. Positional Embedding:
   - Sinusoidal (Whisper 风格), 预计算 [max_pos=1500, 1024]
   - 加法应用: hidden += PE[:seq_len]

4. Chunk 分段 + Window Attention:
   - 按 n_window_infer=800 分 attention window
   - 构建 cu_seqlens 数组
   - 每个 window 内部双向 attend, window 间不 attend

5. 24× Encoder Layer:
   - PRE-LN: LayerNorm(1024) ← 不是 RMSNorm! 有 bias!
   - MHA: 16 heads, head_dim=64, Q/K/V/O 全有 bias
   - 双向注意力 (非因果), 内部按 cu_seqlens 分段
   - FFN: Linear(1024→4096, bias) + GELU + Linear(4096→1024, bias)
   - Float16 clamp: hidden = clamp(hidden, -65504, 65504)

6. 后处理:
   - ln_post: LayerNorm(1024)
   - proj1: Linear(1024→1024, bias) + GELU
   - proj2: Linear(1024→2048, bias)
   → output [total_tokens, 2048]
```

#### Text Decoder

```
1. Token Replacement:
   - embed_tokens(input_ids) → [B, seq, 2048]
   - 找 <|audio_pad|>(151676) 位置 → 替换为 encoder output

2. 28× Decoder Layer:
   - PRE-NORM: RMSNorm(2048, eps=1e-6) ← plain weight
   - GQA: 16 Q / 8 KV heads, head_dim=128, 无 bias
   - Q/K Norm: per-head RMSNorm(128)
   - MRoPE: 3D interleaved, sections [24,20,20]
   - SwiGLU: gate(2048→6144) * up(2048→6144) → down(6144→2048)

3. Output:
   - norm: RMSNorm(2048)
   - lm_head: Linear(2048→151936, no bias) ← 与 embed_tokens 共享权重
   - 自回归: greedy/beam, EOS=[151643, 151645]
```

### 5.3 与 LLM Decoder 的对比 (代码复用分析)

| 组件 | ASR Decoder | Qwen3.5-27B | 可复用 |
|------|-------------|-------------|--------|
| RMSNorm | plain weight | centered (1+w) | ❌ 不同 |
| Attention | GQA 16Q/8KV, head=128 | GQA 24Q/4KV, head=256 | ❌ 参数不同 |
| Q/K Norm | RMSNorm(128) | RMSNorm(256) | ⚠️ 函数相同 |
| RoPE | MRoPE interleaved | 半旋转 partial | ❌ 不同 |
| MLP | SwiGLU 2048→6144 | SwiGLU 5120→17408 | ⚠️ 函数相同 |
| KV Cache | 标准 paged | Paged + SSM State | ⚠️ 可复用 paged 部分 |

**结论**: 虽然数学上类似, 但参数/行为差异大, **全部在 `audio_ops.cu` 中独立实现**, 不与 LLM `light_ops.cu` 共享代码。
理由: (1) 三种 Norm、三种 RoPE、bias 有无 — 泛化后模板膨胀, 反而增加维护成本; (2) 音频模型优化方向 (小 hidden, 短序列, MHA) 与 LLM (大 hidden, 长序列, GQA+SSM) 差异大; (3) 独立代码库允许各自演进, 避免修改 LLM 算子时破坏音频推理。

### 5.4 Mel Spectrogram 实现

参考 whisper.cpp 的实现:
```cpp
void compute_mel(const float* pcm, int num_samples,
                 float* mel_out, int* mel_frames) {
    // 1. STFT: Hann window(400), hop(160)
    // 2. Power spectrum: |FFT|²
    // 3. Mel filterbank (128 bins)
    // 4. Log-mel: max(val, 1e-10), 10*log10
    // 5. Normalize: (mel - max) / max + 4.0, clamp 0
    *mel_frames = (num_samples - 400) / 160 + 1;
}
```

纯 CPU 实现, 30s 音频计算量 <1ms。

---

## 6. TTS Engine 详细设计

### 6.1 核心数据结构

```cpp
struct TTSConfig {
    // Talker
    int talker_layers = 28;
    int talker_hidden = 2048;
    int talker_heads_q = 16;
    int talker_heads_kv = 8;
    int talker_head_dim = 128;
    int talker_ffn = 6144;
    int text_vocab_size = 151936;
    int codec_vocab_size = 3072;    // 2048 codebook + 1024 special
    int text_hidden_size = 2048;
    float rope_theta = 1000000.0f;
    int mrope_sections[3] = {24, 20, 20};
    int position_id_per_seconds = 13;

    // Code Predictor
    int predictor_layers = 5;
    int predictor_hidden = 1024;
    int predictor_heads_q = 16;
    int predictor_heads_kv = 8;
    int predictor_head_dim = 128;
    int predictor_ffn = 3072;
    int num_code_groups = 16;
    int codebook_size = 2048;

    // Speech Tokenizer Decoder
    int tokenizer_codebook_dim = 512;
    int tokenizer_vq_dim = 256;
    int tokenizer_latent_dim = 1024;
    int tokenizer_transformer_hidden = 512;
    int tokenizer_transformer_layers = 8;
    int tokenizer_transformer_heads = 16;   // MHA
    int tokenizer_transformer_head_dim = 64;
    int tokenizer_transformer_ffn = 1024;
    int tokenizer_sliding_window = 72;
    int tokenizer_decoder_dim = 1536;
    int tokenizer_upsample_rates[4] = {8, 5, 4, 3};  // CNN decoder
    int tokenizer_pre_upsample_rates[2] = {2, 2};     // ConvNeXt
    int decode_upsample_rate = 1920;                   // 总上采样
    int output_sample_rate = 24000;
};
```

### 6.2 Talker Forward Pass

#### Prefill 阶段 — 构建 Input Embeds

```
CustomVoice 模式 (非流式):
  1. role: text_proj(text_emb([im_start, assistant, \n]))      → [3, D]
  2. codec_prefix: codec_emb([think, think_bos, lang_id, think_eos])
                   + tts_pad_emb ×4                             → [4, D]
  3. speaker: codec_emb(spk_id) + tts_pad_emb                  → [1, D]
  4. padding: codec_emb(codec_pad) + tts_pad_emb                → [1, D]
  5. text_content: text_proj(text_emb(文本tokens))
                   + codec_emb(codec_pad) ×N                    → [N, D]
  6. text_eos: text_proj(text_emb(tts_eos))
               + codec_emb(codec_pad)                           → [1, D]
  7. start: tts_pad_emb + codec_emb(codec_bos)                  → [1, D]
  
  总 prefill: [3+4+1+1+N+1+1, D] = [N+11, D]

流式模式:
  - 只 prefill: role + codec_prefix + spk + first_text + codec_bos
  - 剩余文本存入 trailing_text_hidden, decode 时逐步 += 到 inputs_embeds
  - 文本用完后切换为 tts_pad_embed
```

#### Decode 阶段 — 每步

```
1. 取上步采样的 group-0 token → codec_embedding(token) → [1, D]

2. 调用 Code Predictor:
   - 输入: past_hidden[1, D] + last_id_hidden[1, D] = [2, D]
   - small_to_mtp_projection(2048→1024) → [2, 1024]
   - 5 层 Transformer + 自回归 15 步
   - 每步 i: embedding[i-1](prev) → transformer → lm_head[i] → sample
   - 输出: groups 1-15 tokens

3. 计算当前步嵌入:
   - codec_emb(group_0) + Σ(predictor.emb[i](group_i+1) for i=0..14)
   - → sum 到单个 [1, D] 向量
   - += trailing_text_hidden[generation_step] 或 tts_pad_embed

4. 过 Talker 28 层 Transformer → norm → codec_head → sample next group-0

5. 保存 hidden_states[:, -1:] 给下步 Code Predictor
```

#### Key Differences from LLM Engine

| 方面 | TTS Talker | LLM Qwen3.5-27B |
|------|-----------|-----------------|
| 输入嵌入 | text+codec 双轨求和 | 单一 embed_tokens |
| 每步额外计算 | Code Predictor 5L×15 步 | MTP 64L×3 步 |
| RoPE | MRoPE interleaved | 半旋转 partial |
| Norm | RMSNorm plain weight | RMSNorm (1+w) |
| 输出 | codec_head → 3072 vocab | lm_head → 248320 vocab |
| 停止条件 | codec_eos (2150) | EOS tokens |

### 6.3 Speech Tokenizer Decoder — Codes → Waveform

```
codes [B, 16, T]
  │
  ├─ RVQ Dequantize:
  │   1 semantic: codebook[2048, 256] → project_out → output_proj Conv1d(256→512)
  │   15 acoustic: 各 codebook[2048, 256] → 同上
  │   所有累加 → [B, 512, T]
  │   注: embed = embedding_sum / cluster_usage.clamp(1e-5)  ← 预计算!
  │
  ├─ Pre-Conv: CausalConv1d(512→1024, k=3)   → [B, 1024, T]
  │
  ├─ Pre-Transformer:
  │   input_proj Linear(1024→512)
  │   8× Layer:
  │     RMSNorm(512) → SlidingWindowMHA(16h, head_dim=64, window=72)
  │                     → Q/K: RoPE θ=10000, 无 Q/K norm
  │     + LayerScale(0.01)
  │     RMSNorm(512) → SwiGLU(512→1024→512)
  │     + LayerScale(0.01)
  │   norm → output_proj Linear(512→1024)     → [B, 1024, T]
  │
  ├─ Upsample ×2:
  │   2× (CausalTransConv1d(1024, k=2, s=2)
  │       + ConvNeXtBlock(dwconv k=7 + LayerNorm + GELU + FC×2))
  │   → [B, 1024, 4T]
  │
  └─ CNN Decoder (BigVGAN):
      Conv1d(1024→1536, k=7)
      4× DecoderBlock:
        SnakeBeta(dim) → CausalTransConv1d(dim→dim/2, k=2*rate, s=rate)
        → 3× ResUnit(dil=[1,3,9]): SnakeBeta→Conv1d(k=7,dil)→SnakeBeta→Conv1d(k=1)
        Channels: 1536 → 768 → 384 → 192 → 96
        Upsample: s=[8, 5, 4, 3] → 总 480×
      SnakeBeta(96) → Conv1d(96→1, k=7)
      → [B, 1, 1920T]
      → clamp(-1, 1) → PCM 24kHz
```

### 6.4 Chunked Streaming Decode

```
tokenizer_decode_streaming(codes, chunk_size=300, left_context=25):
  for each chunk:
    1. 取 codes[..., start-context : start+chunk_size]
    2. 完整 forward (无 KV cache, 每 chunk 独立)
    3. 裁剪左侧 context*1920 个样本
    4. 回调 on_audio(pcm_chunk)
    5. start += chunk_size
  
  每 chunk: 300 frames = 24s audio, context = 25 frames = 2s overlap
```

### 6.5 Voice 管理 (Phase 1: CustomVoice only)

| 说话人 | codec ID |
|--------|---------|
| Vivian | 3065 |
| Serena | 3066 |
| 其他 7 个 | 3058-3072 |

注入方式: `codec_embedding(spk_id)` 作为 prefill 的一个位置。

### 6.6 CUDA Kernel 需求清单

| Kernel | 用途 | 复杂度 |
|--------|------|--------|
| SnakeBeta | CNN decoder 激活 | 新增, 逐元素 |
| CausalConv1d | 多处使用 | 新增, 左 padding |
| CausalTransConv1d | 上采样 | 新增, 右裁剪 |
| ConvNeXtBlock | upsample 阶段 | 新增, depthwise + GELU |
| LayerScale | Transformer + ConvNeXt | 新增, per-channel multiply |
| RVQ Dequantize | codebook lookup + 累加 | 新增, embedding 查表 |
| LayerNorm | Encoder + ConvNeXt | 独立实现 (audio_ops) |
| RMSNorm (plain weight) | Decoder/Talker/Predictor/Tokenizer | 独立实现 (audio_ops, plain weight) |
| SwiGLU | 多处 MLP | 独立实现 (audio_ops, 不同 dim) |
| GQA Attention | Decoder/Talker/Predictor | 独立实现 (audio_ops, 不同 head config) |
| MHA (bidirectional) | ASR Encoder | 新增, 非因果 |
| SlidingWindow Attention | Tokenizer Transformer | 新增, window=72 |
| MRoPE | ASR Decoder / TTS Talker | 新增, interleaved 3D |
| 1D RoPE | Code Predictor / Tokenizer Transformer | 新增, 标准半旋转 |
| Sinusoidal PE | ASR Encoder | 预计算, 加法 |

---

## 7. GPU 调度策略

### 7.1 时分复用 (Time-Division Multiplexing)

20 SM 单 GPU, 多模型串行执行:

```
独立 ASR 请求:
│ Mel (CPU) │ Encoder (GPU, ~50ms) │ Decoder (GPU, 自回归) │ → Text
              └── 24L, 双向 ──────┘   └── 28L, 自回归 ────┘

独立 TTS 请求:
│ Prefill (GPU, ~10ms) │ Decode loop: │talker│predictor×15│ → codes │ tokenizer decode │ → PCM
                          └── 每步 ~8ms (28L + 5L×15) ──────┘   └── chunk ~50ms ──────┘

语音对话管线 (远期):
│ ASR Encode │ ASR Decode │ LLM Prefill │ LLM Decode (streaming) │ TTS Decode │ TTS Tokenizer │
```

### 7.2 调度优先级

1. **LLM decode batch** — 保证现有文本生成不停顿
2. **ASR encoder** — 非自回归, 一次 forward 完成, 不阻塞
3. **TTS codec decode** — 自回归但 12.5Hz (80ms/帧预算), 可分时
4. **TTS tokenizer decode** — 非自回归, 按 chunk batch 处理
5. **LLM prefill** — 最长, 放空闲时段

### 7.3 条件触发机制 (远期)

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ ASR      │────▶│ Trigger  │────▶│ LLM      │
│ (持续)    │     │ Manager  │     │ → TTS?   │
└──────────┘     └──────────┘     └──────────┘
                      │
                      ▼
               条件判断:
               - VAD 检测用户停止说话?
               - ASR 识别到特定关键词?
               - 累积足够上下文?
               → 触发 LLM 推理
               → 决定是否 TTS 输出
```

---

## 8. WebSocket Realtime API 设计

### 8.1 协议概述

参考 OpenAI Realtime API, 实现 WebSocket 双向通信:

- **端点**: `ws://host:port/v1/realtime`
- **协议**: JSON 事件 (event-based)
- **音频格式**: PCM16 24kHz/16kHz, base64 编码

### 8.2 核心事件 (Phase 1 子集)

#### 客户端 → 服务端

| 事件 | 描述 |
|------|------|
| `session.update` | 配置会话参数 (modalities, voice, temperature) |
| `input_audio_buffer.append` | 追加 base64 音频数据 |
| `input_audio_buffer.commit` | 提交音频缓冲区 (手动模式) |
| `input_audio_buffer.clear` | 清空音频缓冲区 |
| `conversation.item.create` | 创建对话项 (文本/音频) |
| `response.create` | 触发响应生成 |
| `response.cancel` | 取消正在进行的响应 |

#### 服务端 → 客户端

| 事件 | 描述 |
|------|------|
| `session.created` | 会话已创建 |
| `session.updated` | 会话配置已更新 |
| `input_audio_buffer.speech_started` | 检测到语音开始 (VAD) |
| `input_audio_buffer.speech_stopped` | 检测到语音结束 (VAD) |
| `conversation.item.created` | 对话项已创建 |
| `response.created` | 响应已创建 |
| `response.output_item.added` | 响应输出项已添加 |
| `response.text.delta` | 文本增量 |
| `response.audio.delta` | 音频增量 (base64 PCM) |
| `response.audio_transcript.delta` | 音频转录增量 |
| `response.done` | 响应完成 |
| `error` | 错误事件 |

### 8.3 WebSocket 服务器实现

当前 HTTP 服务基于 POSIX socket 自实现。WebSocket 需要:

1. **HTTP Upgrade 握手**: 在现有 HTTP 解析中检测 `Upgrade: websocket`
2. **WebSocket 帧协议**: RFC 6455 帧编解码 (opcode, masking, fragmentation)
3. **会话管理**: 每个 WS 连接对应一个 RealtimeSession

```cpp
// ws_server.h
class WebSocketConnection {
public:
    // 发送 JSON 事件
    void send_event(const std::string& event_json);

    // 接收 JSON 事件 (非阻塞)
    bool recv_event(std::string& event_json);

    // WebSocket 帧层
    void send_frame(uint8_t opcode, const uint8_t* data, size_t len);
    bool recv_frame(uint8_t& opcode, std::vector<uint8_t>& payload);

private:
    int fd_;
    // 帧解析状态机
};

class RealtimeSession {
public:
    RealtimeSession(WebSocketConnection& ws,
                    ASREngine& asr, InferenceBackend& llm, TTSEngine& tts);

    // 主循环
    void run();

    // 事件处理
    void on_session_update(const json& event);
    void on_audio_buffer_append(const json& event);
    void on_audio_buffer_commit(const json& event);
    void on_response_create(const json& event);
    void on_response_cancel(const json& event);

private:
    // 音频缓冲区
    std::vector<int16_t> audio_buffer_;  // PCM16 ring buffer

    // VAD 状态
    bool vad_enabled_ = true;
    bool is_speaking_ = false;

    // 当前响应状态
    bool generating_ = false;

    WebSocketConnection& ws_;
    ASREngine& asr_;
    InferenceBackend& llm_;
    TTSEngine& tts_;
};
```

### 8.4 VAD (语音活动检测)

Phase 1: 简单能量阈值 VAD (CPU 端)
- 计算 PCM 帧能量 (RMS)
- 超过阈值 → speech_started
- 持续低于阈值 N ms → speech_stopped

Phase 2: WebRTC VAD 或 Silero VAD (更准确)

---

## 9. REST API 设计

### 9.1 `/v1/audio/transcriptions` (ASR)

**请求**: `POST`, `multipart/form-data`
```
file: <audio binary>
model: "qwen3-asr-1.7b"     (可选, 默认使用已加载模型)
language: "Chinese"          (可选, 自动检测)
response_format: "json"      (可选, json/text/verbose_json)
```

**响应**:
```json
{
  "text": "今天天气真不错。",
  "language": "Chinese",
  "duration": 3.2
}
```

### 9.2 `/v1/audio/speech` (TTS)

**请求**: `POST`, `application/json`
```json
{
  "model": "qwen3-tts-1.7b",
  "input": "你好，世界！",
  "voice": "Vivian",
  "response_format": "pcm",
  "speed": 1.0
}
```

**响应**: 二进制 PCM/WAV 音频数据

### 9.3 与 LLM API 的协调

当 ASR/TTS 与 LLM 并行服务时, GPU 资源需要协调:
- ASR/TTS 请求期间, LLM decode 可能暂时被阻塞
- 需要优先级机制: LLM 流式输出不应因 ASR/TTS 请求而卡顿
- 方案: ASR/TTS 使用独立 CUDA stream, 由 GPU 硬件调度

---

## 10. 权重加载与模型管理

### 10.1 Safetensors 加载

复用现有 `SafetensorsLoader`:

```cpp
// ASR (2 shards, ~4.7 GB):
//   thinker.audio_tower.conv2d{1,2,3}.*     — Conv2D frontend
//   thinker.audio_tower.layers.0-23.*       — 24 encoder layers (q/k/v/out_proj + bias, fc1/fc2 + bias, 2 layernorm)
//   thinker.audio_tower.conv_out.*          — Linear(7680→1024)
//   thinker.audio_tower.ln_post.*           — LayerNorm(1024)
//   thinker.audio_tower.proj1.*, proj2.*    — 投影层 (有 bias)
//   thinker.model.embed_tokens.*            — Embedding(151936, 2048)
//   thinker.model.layers.0-27.*             — 28 decoder layers
//   thinker.model.norm.*                    — RMSNorm(2048)
//   注: lm_head 与 embed_tokens tied

// TTS (1 shard, ~3.8 GB):
//   talker.model.text_embedding.*           — Embedding(151936, 2048)
//   talker.model.codec_embedding.*          — Embedding(3072, 2048)
//   talker.model.layers.0-27.*              — 28 talker layers
//   talker.model.norm.*                     — RMSNorm(2048)
//   talker.text_projection.*               — ResizeMLP(2048→2048→2048, bias)
//   talker.codec_head.*                     — Linear(2048→3072)
//   talker.code_predictor.model.codec_embedding.0-14.*  — 15× Embedding(2048, 2048)
//   talker.code_predictor.model.layers.0-4.*            — 5 predictor layers
//   talker.code_predictor.lm_head.0-14.*                — 15× Linear(1024, 2048)
//   talker.code_predictor.small_to_mtp_projection.*     — Linear(2048, 1024, bias)

// TTS Tokenizer Decoder (~0.68 GB):
//   decoder.quantizer.rvq_first.layers.0.codebook.*     — semantic [2048, 256] + usage [2048]
//   decoder.quantizer.rvq_rest.layers.0-14.codebook.*   — 15× acoustic
//   decoder.pre_conv.*                                   — CausalConv1d(512→1024)
//   decoder.pre_transformer.layers.0-7.*                 — 8 transformer layers
//   decoder.upsample.0-1.*                               — TransConv + ConvNeXt
//   decoder.model.0-16.*                                 — BigVGAN CNN decoder
```

### 10.2 模型配置

```ini
# configs/qwen3.5-27b.conf (扩展)
# ASR 配置
asr_enabled = true
asr_model = /home/rm01/models/dev/asr/Qwen3-ASR-1.7B

# TTS 配置
tts_enabled = true
tts_model = /home/rm01/models/dev/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice
# tts_tokenizer 自动路径: {tts_model}/speech_tokenizer/
# tts_voice = Vivian
```

### 10.3 加载顺序与 mmap 释放

```
1. LLM 权重加载 (~37s for 27B)
2. Vision Encoder 权重绑定
3. ASR 权重加载 (~3-5s, 2 shards)
4. TTS Talker + CodePredictor 权重加载 (~3s, 1 shard)
5. TTS Tokenizer Decoder 权重加载 (~1s)
   注: speech_tokenizer 在 TTS 模型目录下, 496 tensors
   注: codebook 需预计算 embed = embedding_sum / cluster_usage
6. loaders_.clear()  // 释放所有 mmap, 不可移除
7. 启动推理线程
```

---

## 11. 文件结构规划

```
src/
├── engine/
│   ├── asr_engine.h/cpp       — ASR 推理引擎 (transcribe + stream)
│   ├── asr_encoder.h/cu       — Audio encoder (Conv2D + Transformer, GELU, LayerNorm)
│   ├── asr_decoder.h/cu       — Text decoder (Qwen3 GQA, MRoPE, SwiGLU)
│   ├── asr_config.h           — ASRConfig (从 config.json 解析)
│   │
│   ├── tts_engine.h/cpp       — TTS 推理引擎 (synthesize + stream)
│   ├── tts_talker.h/cu        — Talker LM (dual-track embed, MRoPE, codec_head)
│   ├── tts_predictor.h/cu     — Code Predictor (5L autoregressive, 15 embeddings/heads)
│   ├── tts_tokenizer_decode.h/cu — Speech Tokenizer decoder (RVQ + Transformer + BigVGAN)
│   ├── tts_config.h           — TTSConfig (从 config.json 解析)
│   │
│   ├── audio_utils.h/cpp      — Mel spectrogram, PCM I/O, resampling
│   ├── audio_ops.h/cu         — 音频独立 CUDA 算子库 (不复用 LLM light_ops):
│   │                             基础: RMSNorm(plain), LayerNorm(bias), SwiGLU,
│   │                                   GQA attention, MRoPE, 1D RoPE, sinusoidal PE
│   │                             音频专用: SnakeBeta, CausalConv1d, CausalTransConv1d,
│   │                                       ConvNeXtBlock, LayerScale, RVQ dequantize,
│   │                                       bidirectional MHA, SlidingWindow attention
│   │
│   ├── pipeline.h/cpp         — PipelineManager (ASR→LLM→TTS 编排, 条件触发)
│   └── ... (现有文件不变)
│
├── serve/
│   ├── ws_server.h/cpp        — WebSocket 服务器 (RFC 6455) [Phase 3]
│   ├── realtime_session.h/cpp — Realtime API 会话管理 [Phase 3]
│   ├── vad.h/cpp              — 语音活动检测 [Phase 3]
│   └── serve.h/cpp            — 扩展: ASR/TTS REST 路由 (替代 subprocess 插件)
└── plugins/
    └── (Phase 1 完成后可移除 subprocess 插件)
```

---

## 12. 实现路线图

### Phase 1: 基础设施 + ASR Engine MVP

**目标**: ASR 端到端推理, `/v1/audio/transcriptions` REST API

1. **audio_utils**: CPU 端 Mel spectrogram (参考 whisper.cpp)
2. **asr_config**: 解析 Qwen3-ASR config.json
3. **asr_encoder**: Conv2D frontend + 24L Transformer encoder
   - 关键: LayerNorm (非 RMSNorm, 有 bias), bidirectional attention, GELU
   - 使用 cuBLAS GEMM (序列长度固定, 无需 CUTLASS)
4. **asr_decoder**: 28L Qwen3 decoder
   - 关键: RMSNorm (plain weight), GQA 16Q/8KV, MRoPE interleaved, SwiGLU
   - KV Cache: 可简化版 (单请求, 非 paged, 直接 contiguous)
5. **audio_ops.cu**: 独立算子库 — RMSNorm(plain), LayerNorm, SwiGLU, GQA, MRoPE, bidirectional MHA, sinusoidal PE (不复用 LLM light_ops)
6. **权重加载**: Safetensors → ASR encoder + decoder weights
7. **REST API**: `/v1/audio/transcriptions` (替换 subprocess 插件)
8. **测试**: WAV 文件 → 文字, 与 Python 参考对比精度
9. **Benchmark**: encoder forward 延迟, 端到端 TTFT

### Phase 2: TTS Engine MVP

**目标**: TTS 端到端推理, `/v1/audio/speech` REST API

1. **tts_config**: 解析 Qwen3-TTS config.json
2. **tts_talker**: 28L Talker LM
   - 关键: dual-track embedding (text+codec 求和), MRoPE, codec_head
   - 流式文本注入 (trailing_text_hidden)
3. **tts_predictor**: 5L Code Predictor
   - 关键: 15 独立 embedding + 15 独立 lm_head, small_to_mtp_projection
   - 标准 1D RoPE (非 MRoPE), 每个 talker step 重新 prefill
4. **tts_tokenizer_decode**: Speech Tokenizer decoder
   - **最复杂**: RVQ dequantize + CausalConv + 8L SlidingWindow Transformer
     + 2× ConvNeXt upsample + BigVGAN CNN decoder
   - 新增 kernel: SnakeBeta, CausalConv1d, CausalTransConv1d, ConvNeXtBlock
5. **权重加载**: Safetensors → Talker + Predictor + Tokenizer decoder weights
   - 注意: codebook 需预计算 `embed = sum/usage`
6. **REST API**: `/v1/audio/speech` (替换 subprocess 插件)
7. **流式输出**: Chunked tokenizer decode (chunk=300, context=25)
8. **测试**: 文字 → WAV, 听感测试 + 与 Python 参考波形对比

### Phase 3: WebSocket + Realtime API + Pipeline

**目标**: `/v1/realtime` WebSocket, ASR→LLM→TTS 语音对话

1. **ws_server**: RFC 6455 帧协议 (手动实现, 无外部依赖)
2. **realtime_session**: 事件协议 (参考 OpenAI Realtime API)
3. **vad**: 能量阈值 VAD → 后期升级 WebRTC/Silero
4. **pipeline**: ASR→LLM→TTS 编排, 条件触发, 打断处理
5. **双输出策略**: LLM 文本 + TTS 音频分离输出

### Phase 4: 优化

- ASR encoder + LLM decode 不同 stream 并行
- TTS talker + tokenizer decode 流水线化
- SnakeBeta/Conv kernel 融合优化
- NVFP4 量化支持 (ASR/TTS 模型)
- Batch ASR/TTS 支持 (多路并发)

---

## 13. 风险与已解决事项

### 13.1 已解决 (v0.2 更新)

| 问题 | 状态 | 解决 |
|------|------|------|
| TTS Tokenizer 架构不明 | ✅ 已解决 | BigVGAN CNN decoder + Mimi pre-transformer, 完整逐层分析 |
| ASR encoder 架构不明 | ✅ 已解决 | Whisper Conv2D + 24L bidir transformer, 精确参数 |
| TTS 多码本生成方式不明 | ✅ 已解决 | Talker → group-0 + CodePredictor 15 步自回归 → groups 1-15 |
| Python 参考源码不可用 | ✅ 已解决 | qwen-asr/qwen-tts pip 包, 完整 modeling 代码 |
| Norm 差异不明 | ✅ 已解决 | Encoder=LayerNorm(bias), Decoder/Talker=RMSNorm(plain), LLM=RMSNorm(1+w) |

### 13.2 已识别风险

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| 新增 kernel 数量多 (~10 种) | 开发周期长 | 先用 cuBLAS/naive kernel, 后优化 |
| SnakeBeta + BigVGAN 无现有参考 | 精度难保证 | 数值对比 Python 输出, 逐层验证 |
| MRoPE interleaved 实现复杂 | 可能有 bug | 单元测试对比 Python 旋转矩阵 |
| Code Predictor 每步重 prefill | 性能瓶颈 | 5L 小模型 (~20M), 15 步预计 <5ms |
| 多模型 GPU 调度冲突 | LLM 延迟抖动 | 独立 CUDA stream + 优先级调度 |
| WebSocket 服务器稳定性 | 连接管理复杂 | Phase 3 实现, 控制连接数上限 |
| codebook embed 预计算精度 | 音频质量 | 确认 cluster_usage clamping 阈值 |

### 13.3 模型文件位置

```
/home/rm01/models/dev/
├── asr/
│   ├── Qwen3-ASR-1.7B/           ← 2 shard, 4.7 GB
│   └── Qwen3-ForcedAligner-0.6B/ ← (可选, 对齐用)
└── tts/
    ├── Qwen3-TTS-12Hz-1.7B-CustomVoice/   ← 1 shard, 3.8 GB + speech_tokenizer/
    ├── Qwen3-TTS-12Hz-1.7B-VoiceDesign/   ← (Phase 2+)
    ├── Qwen3-TTS-12Hz-1.7B-Base/          ← (VoiceClone, Phase 2+)
    ├── Qwen3-TTS-12Hz-0.6B-CustomVoice/   ← 小模型选项
    ├── Qwen3-TTS-12Hz-0.6B-Base/
    └── Qwen3-TTS-Tokenizer-12Hz/          ← 独立 tokenizer, 与 speech_tokenizer/ 内容一致
```

---

## 14. C++ 实现关键陷阱总结

以下必须严格遵守, 否则推理结果错误:

1. **三种不同的归一化**:
   - ASR Encoder: `LayerNorm(μ, σ²)`, 有 weight + bias
   - ASR Decoder / TTS Talker / Predictor / Tokenizer: `RMSNorm`, **plain weight** `w * x * rsqrt(variance + eps)`
   - 现有 LLM: `RMSNorm`, **centered weight** `(1+w) * x * rsqrt(variance + eps)`

2. **Bias 使用不一致**:
   - ASR Encoder: Q/K/V/O、FFN、conv_out、proj 全有 bias
   - ASR Decoder: 无 attention bias, 无 MLP bias
   - TTS Talker: 无 attention bias, 无 MLP bias
   - TTS text_projection: 有 bias
   - Code Predictor small_to_mtp_projection: 有 bias

3. **RoPE 类型不一致**:
   - ASR Decoder / TTS Talker: MRoPE interleaved, sections [24,20,20], θ=1e6
   - Code Predictor / Tokenizer Transformer: 标准 1D RoPE, θ=1e4 或 1e6

4. **embed_tokens 与 lm_head 权重共享** (ASR Decoder, `tie_word_embeddings=True`)

5. **TTS 16 group 嵌入求和** — 不是拼接, 不是 concat, 是 element-wise add

6. **RVQ codebook embed 需从 embedding_sum / cluster_usage 预计算**

7. **SnakeBeta 中的 exp(α)/exp(β) 需 per-channel 缓存** (推理时不变)

8. **CausalConv1d 左 padding** = `(kernel_size - 1) * dilation`

9. **CausalTransConv1d 右裁剪** = `kernel_size - stride`

10. **ConvNeXt 用 LayerNorm** (不是 RMSNorm), 有 weight + bias

---

## 15. 总结

v0.2 更新基于实际模型文件分析 (config.json, safetensors) 和 Python 参考实现 (qwen-asr/qwen-tts) 的完整逆向:

1. **ASR**: Whisper 风格 audio encoder (Conv2D + 24L bidir transformer + projection) → Token replacement → Qwen3 decoder (28L GQA + MRoPE + SwiGLU)
2. **TTS**: Dual-track talker (text+codec 嵌入求和, 28L + codec_head) → Code predictor (5L × 15 步自回归) → Speech tokenizer decoder (RVQ + 8L sliding-window transformer + BigVGAN CNN)
3. **独立 Engine 架构**: ASR/TTS 不影响现有 LLM, 可任意搭配, 支持同步/异步/不输出
4. **内存充裕**: 27B LLM + 1.7B ASR + 1.7B TTS ≈ 73 GB, 余 55 GB
5. **纯 C++ 实现 + 算子独立**: 全部 CUDA kernel 在 `audio_ops.cu` 独立实现 (~15 种), 不与 LLM `light_ops.cu` 共享, 各自独立演进
