# ASR 插件全面升级开发计划

> **项目**: Qwen35-Thor ASR Plugin Enhancement  
> **目标**: 将 ASR 插件打造为超越 FunASR 的全栈语音理解引擎，为阿里巴巴 AI 大模型提供原生 C++/CUDA 推理支持  
> **平台**: NVIDIA Jetson AGX Thor (SM110a Blackwell, 128 GB LPDDR5X 统一内存)  
> **创建日期**: 2026-03-14

---

## 一、项目背景与现状

### 1.1 现有 ASR 插件架构

当前 ASR 插件位于 `src/plugins/asr/`，已实现 Qwen3-ASR-1.7B 的 **原生 C++/CUDA 推理**:

| 组件 | 文件 | 说明 |
|------|------|------|
| 引擎 | `asr_engine.h/cpp` | 权重加载 + Encoder/Decoder 编排 + transcribe 入口 |
| 编码器 | `asr_encoder.h/cu` | Whisper-style: Conv2D 前端 + 24 层双向 Transformer, d=2048 |
| 解码器 | `asr_decoder.h/cu` | 28 层 Qwen3 GQA + MRoPE + SwiGLU, d=2048, vocab=151936 |
| Kernel | `audio_ops.h/cu` | LayerNorm, RMSNorm, RoPE, MRoPE, 双向 MHA, Causal GQA 等 |
| 音频 | `audio_utils.h/cpp` | WAV 加载, Mel 频谱图, 重采样 |
| 插件 | `asr_plugin.h/cpp` | 接口层: NativeAsrPlugin (GPU) + SubprocessAsrPlugin |
| 配置 | `asr_config.h/cpp` | config.json 解析 (thinker_config → audio/text config) |

**当前 AsrResult 结构** (仅纯文本):
```cpp
struct AsrResult {
    std::string text;              // 转录文本
    std::string language;          // 检测到的语言
    float       duration_s = 0;    // 音频时长 (秒)
    int         error_code = 0;    // 0 = 成功
    std::string error_message;     // 错误信息
};
```

**当前 VAD** (基于 RMS 能量阈值，在 `serve.cpp` 中实现):
```cpp
constexpr float VAD_ENERGY_THRESHOLD  = 0.01f;   // RMS > 0.01 → 语音
constexpr int   VAD_SILENCE_MS        = 800;      // 静音 800ms → 触发 ASR
constexpr int   VAD_MIN_SPEECH_MS     = 500;      // 最短语音段
constexpr int   VAD_MAX_DURATION_S    = 30;       // 最大时长
constexpr float VAD_MIN_SPEECH_ENERGY = 0.008f;   // 整段最低平均 RMS

// 实时 WebSocket VAD (更紧凑):
constexpr float RT_VAD_ENERGY         = 0.01f;
constexpr int   RT_VAD_SILENCE_MS     = 600;
constexpr int   RT_VAD_MIN_SPEECH_MS  = 300;
constexpr int   RT_VAD_MAX_S          = 30;
constexpr float RT_VAD_MIN_AVG_ENERGY = 0.008f;
```

### 1.2 已有可复用资产

| 资产 | 位置 | 说明 |
|------|------|------|
| Qwen3-ASR-1.7B | `/home/rm01/models/dev/asr/Qwen/Qwen3-ASR-1.7B/` | 已实现完整 C++/CUDA 推理, ~4.7 GB |
| Qwen3-ForcedAligner-0.6B | `/home/rm01/models/dev/asr/Qwen/Qwen3-ForcedAligner-0.6B/` | 磁盘上, 未实现, ~1.8 GB |
| ECAPA-TDNN 说话人编码器 | `src/plugins/tts/tts_speaker_encoder.h/cpp` | TTS 用, CPU 推理, 2048-dim, 已实现 |
| VoiceManager | `src/plugins/tts/tts_voice_manager.h` | JSON 存储说话人 embedding, 线程安全, 可复用 |
| SafetensorsLoader | `src/engine/safetensors.h/cpp` | 零拷贝 mmap 加载, 多分片支持 |
| Mel 频谱图提取 | `src/plugins/asr/audio_utils.cpp` | 80-dim Fbank, 可复用 |
| LLM 推理 | `src/engine/backend.h/cpp` | InferenceBackend, 已在进程内, 可直接调用 |

### 1.3 系统约束

- **无 PyTorch/ONNX Runtime**: 所有模型必须 native C++/CUDA + safetensors 加载
- **统一内存**: CPU/GPU 共享 128 GB LPDDR5X, cudaMalloc 管理, 无独立显存
- **已占用内存**: LLM 权重 (~8.7 GB for 4B / ~51 GB for 27B) + TTS (~4 GB) + ASR (~4.7 GB) + KV/SSM
- **Python 环境**: 3.13.11 via miniconda, qwen_asr 0.0.6 / qwen_tts 0.1.1 (无 torch)

---

## 二、与 FunASR 功能对比

| 功能模块 | FunASR 方案 | 参数量 | 我们的方案 | 我们的优势 |
|----------|-----------|-------|-----------|-----------|
| **ASR** | Paraformer / SenseVoice / Fun-ASR-Nano | 220-800M | Qwen3-ASR-1.7B (✅ 已实现) | Qwen3 架构, 更大模型, 更高精度 |
| **VAD** | FSMN-VAD (PyTorch/ONNX) | 0.4M | FSMN-VAD Native C++/CUDA (✅ 已实现) | 零依赖, GPU 加速, 嵌入式原生 |
| **标点恢复** | CT-Punc (PyTorch) | 290M | LLM-Guided + 轻量规则 (✅ 已实现) | 利用已加载 LLM, 零额外内存 |
| **说话人识别** | CAM++ (PyTorch) | 7.2M | CAM++ Native C++/CUDA (✅ 已实现) | 零依赖, Jetson 优化, 192-dim |
| **关键词识别** | FSMN-KWS (PyTorch + CTC) | 0.7M | ASR 特征复用 + 文本匹配 (✅ 已实现) | 零额外模型, ASR 特征更丰富 |
| **时间戳** | FA-zh / CTC 对齐 | 38M | Qwen3-ForcedAligner-0.6B (✅ 已实现) | Qwen3 原生, 更高精度 |
| **说话人分割** | CAM++ + 聚类流水线 | — | CAM++ + 在线聚类 (✅ 已实现) | 实时流式分割 |
| **情感识别** | emotion2vec+ / SenseVoice | 300M | ASR encoder 特征 + 轻量头 | 复用 encoder, 零新增模型 |
| **部署形态** | Python + ONNX + C++ Runtime | — | 纯 C++/CUDA, 单一二进制 | 零 Python, 启动 <1s |

**核心差异化**: 全栈原生 C++/CUDA, 零 Python/ONNX 依赖, safetensors 直接加载, LLM 同进程协同, Jetson/嵌入式原生支持。

---

## 三、分阶段开发计划

### Phase 1: Neural VAD — FSMN-VAD 替代能量阈值

**优先级: P0** | **新增内存: ~2 MB** | **前置依赖: 获取 FSMN-VAD 权重**

#### 1.1 动机

当前基于 RMS 能量阈值的 VAD 在以下场景失败:
- 轻声细语/低能量语音被误判为静音
- 背景噪声 (空调/风扇/键盘) 被误判为语音
- 音乐/环境声干扰触发误检测
- 无法区分语音和非语音类噪声 (咳嗽声触发 ASR)

#### 1.2 FSMN-VAD 模型架构

参考 FunASR `fsmn-vad` (Deep-FSMN for Large Vocabulary Continuous Speech Recognition, arXiv:1803.05030):

```
输入: 80-dim Fbank 特征, 10ms 帧移, 25ms 帧长
  ↓
4-8 层 DFSMN Block:
  每层: Affine(in→hid) → Memory Block → Affine(hid→out) → ReLU → Skip Connection
  Memory Block: h_t = Σ(a_i · x_{t+i}), i ∈ [-l_order, r_order]
    l_order ≈ 10 (回看), r_order ≈ 1 (前看, 低延迟)
    Hidden dim: 128-256
  ↓
输出: 每帧 [speech_prob, sil_prob], 2-class softmax
总参数: ~0.4M
```

**流式推理**: chunk-by-chunk, 每次输入 N 帧 (nn_eval_block_size=8), 维护 FSMN memory cache。延迟 ~200ms。

#### 1.3 决策状态机

移植 FunASR `FsmnVADStreaming` 的完整状态机逻辑:

```
状态: StartPointNotDetected → InSpeechSegment → EndPointDetected
      ↑______________________|                   |
      |________________________________________________|

转换规则:
  Sil → Speech: 窗口内 speech 帧数 ≥ sil_to_speech_time_thres (150ms)
  Speech → Sil:  窗口内 speech 帧数 ≤ speech_to_sil_time_thres (150ms)
                 且连续静音 ≥ max_end_silence_time (800ms)

可配置参数:
  window_size_ms           = 200     # 判决窗口大小
  sil_to_speech_time_thres = 150     # 静音→语音转换门限 (ms)
  speech_to_sil_time_thres = 150     # 语音→静音转换门限 (ms)
  max_end_silence_time     = 800     # 最大尾部静音 (ms)
  max_single_segment_time  = 60000   # 单段最长时间 (ms)
  speech_noise_thres       = 0.6     # 语音/噪声判决门限
  lookback_time_start_point = 200    # 起点回看 (ms)
  lookahead_time_end_point  = 100    # 终点前看 (ms)
```

#### 1.4 实现细节

**新文件**:
- `src/plugins/asr/vad_engine.h` — VadEngine 接口 + VadConfig
- `src/plugins/asr/vad_engine.cu` — FSMN 前向 + 状态机逻辑

**GPU Kernel** (FSMN Memory Block):
```cpp
// FSMN memory block ≈ 1D causal convolution
// h_t = Σ(a_i · x_{t+i}), i ∈ [-l_order, r_order]
// l_order=10, r_order=1 → 有效 kernel_size = 12
// 可复用 Conv1d kernel 或 SMEM+warp reduce 实现
__global__ void fsmn_memory_block_kernel(
    const float* input,   // [T, hidden_dim]
    const float* weights, // [l_order + r_order + 1, hidden_dim]
    float* output,        // [T, hidden_dim]
    int T, int dim, int l_order, int r_order);
```

**集成改动** (`serve.cpp`):
```cpp
// 替换:
//   constexpr float VAD_ENERGY_THRESHOLD = 0.01f; ...
//   bool vad_triggered = speech_detected && silence_duration_ms >= VAD_SILENCE_MS;
// 为:
//   auto segments = vad_engine_->detect(pcm_chunk, is_final);
//   for (auto& seg : segments) { asr_engine_->transcribe_pcm(seg.data, seg.size, ...); }
```

**Mel 特征**: 复用 `audio_utils.cpp` 中已有的 `compute_mel_spectrogram()`。

#### 1.5 权重获取

```bash
# 临时 conda 环境转换
conda create -n convert python=3.10 && conda activate convert
pip install torch modelscope safetensors

python -c "
from modelscope import snapshot_download
import torch, safetensors.torch as st
model_dir = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
state = torch.load(f'{model_dir}/model.pt', map_location='cpu')
st.save_file({k: v.float() for k, v in state.items()}, 'fsmn_vad.safetensors')
print(f'Saved {len(state)} tensors')
for k, v in state.items():
    print(f'  {k}: {list(v.shape)} {v.dtype}')
"
# 输出文件放到: /home/rm01/models/dev/asr/fsmn_vad/fsmn_vad.safetensors
```

#### 1.6 降级方案

如果 FSMN-VAD 权重获取/架构移植过于复杂，可先实现简化版:
- 3 层 LSTM + Linear head (~1M 参数), 类似 Silero-VAD
- 或保留能量 VAD 作为 fallback, Neural VAD 作为可选增强

---

### Phase 2: CAM++ 说话人识别

**优先级: P1** | **新增内存: ~29 MB** | **前置依赖: 获取 CAM++ 权重**

#### 2.1 CAM++ vs ECAPA-TDNN

| 指标 | ECAPA-TDNN (TTS 已有) | CAM++ |
|------|----------------------|-------|
| 参数量 | 20.8M | **7.2M** (3× 更小) |
| VoxCeleb EER | 0.86% | **0.65%** (24% 改善) |
| 3D-Speaker EER | 8.87% | **7.75%** (13% 改善) |
| 推理速度 | 基准 | **~2× 更快** |
| Embedding dim | 2048 (TTS 定制) | **512** (标准) |
| 实现 | CPU-only | **GPU 加速** |

结论: CAM++ 在**参数量、精度、速度**上全面优于 ECAPA-TDNN。

论文: *CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking* (Interspeech 2023, 阿里达摩院语音实验室)

#### 2.2 CAM++ 架构详解

```
输入: [B, T, 80] Fbank (16kHz, 80-dim)
  ↓
┌─── FCM (Frequency Convolutional Module) ───┐
│ Conv2d(1→32, k=3, s=1) + BN + ReLU        │
│ 2× BasicResBlock(32, stride=2)  → freq ÷4  │
│ 2× BasicResBlock(32, stride=2)  → freq ÷4  │
│ Conv2d(32→32, k=3, s=(2,1)) + BN → freq ÷2│
│ Reshape → [B, 320, T]                      │
└─────────────────────────────────────────────┘
  ↓
TDNN: Conv1d(320→128, k=5, stride=2, d=1) + BN + ReLU
  ↓
┌─── CAMDenseTDNNBlock #1 ───────────────────┐
│ 12 layers, k=3, d=1, growth_rate=32        │
│ Context-Aware Masking on dense connections  │
│ → channels: 128 + 12×32 = 512              │
│ TransitLayer: 512 → 256                    │
└─────────────────────────────────────────────┘
  ↓
┌─── CAMDenseTDNNBlock #2 ───────────────────┐
│ 24 layers, k=3, d=2, growth_rate=32        │
│ → channels: 256 + 24×32 = 1024             │
│ TransitLayer: 1024 → 512                   │
└─────────────────────────────────────────────┘
  ↓
┌─── CAMDenseTDNNBlock #3 ───────────────────┐
│ 16 layers, k=3, d=2, growth_rate=32        │
│ → channels: 512 + 16×32 = 1024             │
│ TransitLayer: 1024 → 512                   │
└─────────────────────────────────────────────┘
  ↓
StatsPool: mean(x) ∥ std(x) over time → [B, 1024]
  ↓
DenseLayer: Linear(1024→512) + BN → L2-normalize
  ↓
输出: 192-dim speaker embedding
```

**CAMDenseTDNNBlock 内部**:
- 每层输出 growth_rate=32 channels, 与之前所有层输出 concatenate
- Context-Aware Masking: 为每个 context 位置学习 attention weight, 对 TDNN output 加权
- BatchNorm + ReLU 激活
- memory_efficient=True: 训练时 checkpoint, 推理时直接计算

#### 2.3 实现计划

**新文件**:
- `src/plugins/asr/speaker_encoder.h` — CamPlusSpeakerEncoder 类定义
- `src/plugins/asr/speaker_encoder.cu` — GPU kernel (Conv2d, Conv1d, BN, StatsPool, Linear)
- `src/plugins/asr/speaker_manager.h` — 说话人注册/识别管理 (复用 VoiceManager 模式)

**GPU Kernel 复用**:
- Conv2d: cuDNN 或自定义 kernel (FCM 部分)
- Conv1d: 复用 ASR encoder 中已有的 audio_ops conv1d 实现
- BatchNorm: 已有 LayerNorm kernel 改造 (无 centering, 有 affine)
- StatsPool: 简单 warp reduce (mean + var)
- Linear: GEMV (复用 engine/dense_gemm)

**说话人识别 API**:
```cpp
class CamPlusSpeakerEncoder {
public:
    void load_weights(const std::string& safetensors_path);

    // 从 Mel 特征提取 192-dim embedding (GPU)
    std::vector<float> extract(const float* mel_80xT, int T);

    // 余弦相似度
    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b);
};

class SpeakerManager {
public:
    // 注册说话人 (embedding 存储到 JSON, 复用 VoiceManager 模式)
    void register_speaker(const std::string& name,
                          const std::vector<float>& embedding);

    // 识别: 返回最匹配的说话人名 + 相似度
    struct MatchResult { std::string name; float similarity; };
    MatchResult identify(const std::vector<float>& embedding,
                         float threshold = 0.65f);

    // 在线更新 (moving average)
    void update_embedding(const std::string& name,
                          const std::vector<float>& new_embedding,
                          float alpha = 0.1f);
};
```

**说话人识别流程**:
1. VAD 检测到语音段 → 提取 Mel 特征
2. CAM++ 编码 → 192-dim embedding (L2 归一化)
3. 与已注册说话人库比较 (cosine similarity)
4. similarity > 0.65 → 匹配; 否则 → 注册新说话人
5. results 中标注 speaker_id

#### 2.4 权重获取

```bash
python -c "
from modelscope import snapshot_download
import torch, safetensors.torch as st
model_dir = snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common')
state = torch.load(f'{model_dir}/campplus_cn_common.bin', map_location='cpu')
st.save_file({k: v.float() for k, v in state.items()}, 'campplus.safetensors')
print(f'Saved {len(state)} tensors, total params: {sum(v.numel() for v in state.values())/1e6:.1f}M')
for k, v in state.items():
    print(f'  {k}: {list(v.shape)}')
"
# 输出: /home/rm01/models/dev/asr/campplus/campplus.safetensors (~29 MB)
```

---

### Phase 3: 关键词识别 (Keyword Spotting / Hotword)

**优先级: P1** | **新增内存: 0** | **前置依赖: 无**

#### 3.1 方案对比

| 方案 | 说明 | 优劣 |
|------|------|------|
| A. 独立 FSMN-KWS 模型 | FunASR 方案, FSMN+CTC, 0.7M | 精确但需额外模型 |
| B. SeACo-Paraformer Hotword | 注意力 bias 注入 | 需改造 decoder |
| **C. ASR 文本流式匹配** ⭐ | ASR decoder 输出流式匹配关键词 | **零模型, 零内存, 最灵活** |

**选择方案 C**: 利用 Qwen3-ASR decoder 逐 token 输出做流式关键词匹配。优势是 ASR encoder 特征 (24L, d=2048) 远比小型 FSMN (0.7M) 丰富，识别精度更有保障。

#### 3.2 实现设计

**新文件**: `src/plugins/asr/keyword_spotter.h`

```cpp
struct KeywordEntry {
    std::string text;            // 关键词文本: "你好小助手"
    std::vector<int> token_ids;  // 分词后 token IDs (ASR tokenizer)
    float threshold;             // 匹配置信度门限 (0-1)
    std::string action;          // 触发动作: "wake" / "stop" / "custom"
};

class KeywordSpotter {
public:
    // 从配置文件加载关键词列表
    void load_config(const std::string& json_path);

    // 添加/移除关键词 (运行时动态)
    void add_keyword(const KeywordEntry& kw);
    void remove_keyword(const std::string& text);

    // ASR 输出文本匹配 (整句)
    struct Hit {
        std::string keyword;
        int char_offset;       // 在文本中的偏移
        float confidence;
    };
    std::vector<Hit> match(const std::string& asr_text);

    // 流式 token 匹配 (逐 token, 低延迟)
    // 在 ASR decoder 每步输出后调用
    std::vector<Hit> on_token(int token_id, float logprob);

private:
    std::vector<KeywordEntry> keywords_;

    // Aho-Corasick 自动机 (多关键词同时匹配)
    // 或简单的文本子串搜索 (关键词少时)
    struct TrieNode { ... };
};
```

**匹配算法**:
1. **精确匹配**: UTF-8 子串搜索 (关键词 ≤ 20 个时 O(N×L))
2. **模糊匹配**: 编辑距离 ≤ 1 (容许一个字差异)
3. **流式前缀匹配**: decoder 每出一个 token 即检查是否构成关键词前缀
4. **Aho-Corasick**: 关键词 > 20 个时启用多模式匹配自动机

**配置文件** (`configs/keywords.json`):
```json
{
    "keywords": [
        {"text": "你好小助手", "threshold": 0.8, "action": "wake"},
        {"text": "开始录音",   "threshold": 0.9, "action": "record_start"},
        {"text": "停止录音",   "threshold": 0.9, "action": "record_stop"},
        {"text": "翻译",       "threshold": 0.85, "action": "translate"}
    ]
}
```

**集成点**: `serve.cpp` 中 ASR 回调后检查 keyword hits, 触发对应 action。

---

### Phase 4: 标点恢复 (Punctuation Restoration)

**优先级: P0** | **新增内存: 0** | **前置依赖: 无**

#### 4.1 方案对比

| 方案 | 说明 | 内存 | 延迟 | 精度 |
|------|------|------|------|------|
| A. CT-Transformer | FunASR 方案, 290M | +1.2 GB | ~50ms | 高 |
| B. CT-Transformer Small | 轻量版 | +200 MB | ~20ms | 中 |
| **C. LLM 辅助** ⭐ | 利用已加载 LLM | **0** | ~100-200ms | **极高** |
| D. 规则 + ASR logits | 启发式 | 0 | <1ms | 低 |

**选择方案 C+D 混合**:
- **实时模式**: 方案 D — 规则方案, 基于语句长度/停顿/常见模式添加基础标点
- **高精度模式**: 方案 C — ASR 文本发送给 LLM 添加标点 (利用已加载 Qwen3.5)

#### 4.2 实现设计

**新文件**: `src/plugins/asr/punctuation.h`

```cpp
class PunctuationRestorer {
public:
    // 快速规则方案 (无外部依赖)
    // - 句末添加句号/问号 (检测疑问词: 吗/呢/什么/怎么/哪...)
    // - 长句 (>15 字无标点) 在自然断点插入逗号
    // - 识别列举 (和/或/与/以及 前后)
    std::string restore_rules(const std::string& text);

    // LLM 方案 (高精度, 需 InferenceBackend)
    // System prompt: "你是标点恢复助手。为以下语音转录文本添加标点符号。
    //                 只添加标点，不修改任何文字内容。直接输出结果。"
    // 使用非思考模式, temperature=0.3, max_tokens=text.length()*1.5
    std::string restore_llm(const std::string& text,
                            InferenceBackend* backend);

    // 自动选择: 短文本用规则, 长文本用 LLM
    std::string restore(const std::string& text,
                        InferenceBackend* backend = nullptr,
                        bool prefer_rules = false);
};
```

**标点规则集**:
| 规则 | 触发条件 | 标点 |
|------|---------|------|
| 句末 | 文本结尾 | 。 |
| 疑问 | 含 吗/呢/什么/怎么/哪/为什么/是否 | ？ |
| 感叹 | 含 太/真/好/啊/哇/哦 | ！ |
| 逗号 | 连续 >15 字无标点, 在连词/转折词处 | ， |
| 顿号 | 并列项 (和/与/跟/或 连接的短语) | 、 |

---

### Phase 5: Qwen3-ForcedAligner 字级时间戳

**优先级: P2** | **新增内存: ~1.8 GB** | **前置依赖: 新建 ForcedAligner Engine**

#### 5.1 模型信息

```
位置: /home/rm01/models/dev/asr/Qwen/Qwen3-ForcedAligner-0.6B/
文件: model.safetensors (~1.8 GB), config.json, tokenizer 等
架构:
  Encoder: 24 层双向 Transformer, hidden=1024 (ASR-1.7B 的一半)
  Decoder: 28 层 Qwen3 GQA + MRoPE + SwiGLU, hidden=1024
  lm_head: [1024, 5000] — 时间分类 (5000 类 × 80ms = 400 秒覆盖)
  tie_word_embeddings = false
  总参数: ~0.6B
  推理模式: NAR (非自回归, 单次 forward)
```

#### 5.2 推理流程

```
1. Audio → Mel (80-dim) → Encoder → audio_features [T_enc, 1024]

2. 构建输入序列:
   <|audio_start|> <|audio_pad|>×N <|audio_end|> word1 <timestamp> word2 <timestamp> ...

3. Decoder single forward → logits at each <timestamp> position

4. argmax(logits[timestamp_pos]) → time_class_id (0-4999)

5. timestamp_ms = time_class_id × 80  (80ms per class)

6. fix_timestamp(): LIS (Longest Increasing Subsequence) 确保单调递增
```

#### 5.3 实现计划

**新文件**:
- `src/plugins/asr/aligner_engine.h/cpp` — ForcedAligner Engine (复用 ASR encoder/decoder 代码)

**代码复用策略**: ForcedAligner 的 encoder/decoder 架构与 ASR-1.7B **完全相同** (只是更小), 可通过模板化/参数化复用:
- 复用 `asr_encoder.h/cu` (参数改为 hidden=1024)
- 复用 `asr_decoder.h/cu` (参数改为 hidden=1024, lm_head=5000)
- 复用所有 `audio_ops.h/cu` kernel

**输出格式** (扩展 AsrResult):
```json
{
    "text": "今天天气真好",
    "words": [
        {"word": "今天", "start_ms": 240, "end_ms": 720, "confidence": 0.95},
        {"word": "天气", "start_ms": 720, "end_ms": 1120, "confidence": 0.92},
        {"word": "真好", "start_ms": 1120, "end_ms": 1600, "confidence": 0.88}
    ]
}
```

**内存考量**: 需额外 ~1.8 GB。对 4B LLM (~8.7 GB) 方案影响不大; 对 27B (~51 GB) 方案需评估。可考虑按需加载/卸载。

---

### Phase 6: 说话人分割 (Speaker Diarization)

**优先级: P2** | **新增内存: 0** | **前置依赖: Phase 1 (VAD) + Phase 2 (CAM++)**

#### 6.1 流水线

```
Audio Stream
  ↓
FSMN-VAD (Phase 1)
  → speech segments: [{start_ms, end_ms, audio_data}, ...]
  ↓
对每个 segment:
  → Mel Spectrogram (复用 audio_utils)
  → CAM++ (Phase 2) → 192-dim embedding
  ↓
在线聚类 (Online Clustering)
  → speaker_id per segment
  ↓
输出: [{start_ms, end_ms, speaker_id, text}, ...]
```

#### 6.2 在线聚类算法

```
SpeakerCluster {
    centroid: float[512]   // L2-normalized
    count: int
    last_seen_ms: int
}

OnIdentify(embedding):
    best_sim = -1, best_idx = -1
    for cluster in clusters:
        sim = cosine_similarity(embedding, cluster.centroid)
        if sim > best_sim: best_sim = sim, best_idx = cluster.idx

    if best_sim >= threshold (0.65):
        // 匹配已有说话人
        cluster[best_idx].centroid =
            normalize(alpha * embedding + (1-alpha) * cluster.centroid)
        cluster[best_idx].count++
        return best_idx
    else:
        // 新说话人
        new_cluster = {centroid: embedding, count: 1}
        clusters.push_back(new_cluster)
        return clusters.size() - 1
```

#### 6.3 实现

**新文件**: `src/plugins/asr/speaker_diarizer.h`

```cpp
class SpeakerDiarizer {
public:
    struct Segment {
        int start_ms;
        int end_ms;
        int speaker_id;
    };

    // 对一段音频执行分割 (需要 VAD + CAM++)
    std::vector<Segment> diarize(const float* pcm, int num_samples,
                                 int sample_rate = 16000);

    // 流式: 每收到一个 VAD segment 调用
    Segment process_segment(const float* pcm, int num_samples,
                            int start_ms);

    // 配置
    float similarity_threshold = 0.65f;
    float update_alpha = 0.1f;        // embedding 更新速率
    int max_speakers = 10;

private:
    CamPlusSpeakerEncoder* encoder_;
    std::vector<SpeakerCluster> clusters_;
};
```

---

### Phase 7: 情感识别 (Speech Emotion Recognition)

**优先级: P3** | **新增内存: ~1 MB** | **前置依赖: 标注数据 + 微调**

#### 7.1 方案

复用 ASR encoder 中间层特征, 添加轻量 classification head:

```
ASR Encoder Layer 12 / 18 输出: [T, 1024]
  ↓
Attentive Statistical Pooling:
  attention_weights = softmax(Linear(1024→1) / √d)
  weighted_mean = Σ(w_t × h_t)
  weighted_std  = √(Σ(w_t × (h_t - mean)²))
  → [2048]
  ↓
MLP Head:
  Linear(2048→256) → ReLU → Dropout(0.3)
  Linear(256→7)    → Softmax
  ↓
输出: 7 类情感概率
  {neutral, happy, sad, angry, fear, surprise, disgust}
```

**参数**: ~0.3M, 总权重 ~1.2 MB。

**暂不实现**: 需要在情感数据 (如 IEMOCAP, MELD, 或中文情感数据) 上训练 head 层。可在后续迭代中用 LLM + ASR 联合 pipeline 做情感标注数据生成。

---

## 四、AsrResult 扩展设计

**当前** (仅 5 个字段):
```cpp
struct AsrResult {
    std::string text;
    std::string language;
    float duration_s;
    int error_code;
    std::string error_message;
};
```

**升级后** (全功能):
```cpp
struct AsrResult {
    // ─── 基础 (已有) ───
    std::string text;                  // 转录文本 (无标点)
    std::string language;              // 语言
    float       duration_s = 0;        // 音频时长
    int         error_code = 0;
    std::string error_message;

    // ─── Phase 4: 标点恢复 ───
    std::string text_with_punc;        // 带标点文本

    // ─── Phase 5: 时间戳 ───
    struct WordInfo {
        std::string word;
        int   start_ms = -1;
        int   end_ms   = -1;
        float confidence = 0;
        int   speaker_id = -1;        // Phase 6
    };
    std::vector<WordInfo> words;

    // ─── Phase 6: 说话人分割 ───
    struct SpeakerSegment {
        int         start_ms;
        int         end_ms;
        int         speaker_id;
        std::string speaker_name;      // 已注册时有名字
        std::string text;
    };
    std::vector<SpeakerSegment> segments;

    // ─── Phase 3: 关键词识别 ───
    struct KeywordHit {
        std::string keyword;
        std::string action;
        int   char_offset;             // UTF-8 字符偏移
        float confidence;
    };
    std::vector<KeywordHit> keyword_hits;

    // ─── Phase 7: 情感 ───
    std::string emotion;               // "neutral"/"happy"/"sad"/...
    float emotion_confidence = 0;
};
```

**向后兼容**: 新字段全部带默认值, 旧代码只访问 `text` 不受影响。

---

## 五、系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      Audio Input (PCM 16kHz)                     │
│                   WebSocket / HTTP / 文件上传                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                          ┌────▼─────┐
                          │  Neural  │  Phase 1
                          │   VAD    │  FSMN-VAD (0.4M, GPU)
                          │          │  替代 RMS 能量阈值
                          └────┬─────┘
                               │ speech segments [{start, end, pcm}]
                  ┌────────────┼────────────┐
                  │            │            │
           ┌──────▼──────┐    │     ┌──────▼──────┐
           │   Speaker   │    │     │   Keyword   │  Phase 3
           │   Encoder   │    │     │   Spotter   │  ASR 文本匹配
           │  (CAM++ GPU)│    │     │  (文本/流式) │  Aho-Corasick
           │  Phase 2    │    │     └──────┬──────┘
           └──────┬──────┘    │            │ keyword_hits
                  │ 192-dim   │            │
           ┌──────▼──────┐    │            │
           │  Speaker    │    │            │  Phase 6
           │  Diarizer   │    │            │  在线聚类
           │  (Cluster)  │    │            │
           └──────┬──────┘    │            │
                  │ spk_id    │            │
                  └────────┬──┘            │
                           │               │
                     ┌─────▼──────┐        │
                     │ ASR Engine │        │  已有: Qwen3-ASR-1.7B
                     │ 24L Enc +  │        │  native C++/CUDA
                     │ 28L Dec    │        │
                     └─────┬──────┘        │
                           │ raw text      │
              ┌────────────┼───────────────┤
              │            │               │
       ┌──────▼──────┐    │        ┌──────▼──────┐
       │ Punctuation │    │        │   Forced    │  Phase 5
       │  Restorer   │    │        │  Aligner    │  Qwen3-FA-0.6B
       │ Phase 4     │    │        │ (时间戳)    │  NAR, GPU
       │ LLM/规则    │    │        └──────┬──────┘
       └──────┬──────┘    │               │ word timestamps
              │            │               │
              └────────────┼───────────────┘
                           │
                     ┌─────▼──────┐
                     │  Emotion   │  Phase 7 (预留)
                     │  Head      │  ASR encoder 中间特征
                     └─────┬──────┘
                           │
                     ┌─────▼──────────────────────────────┐
                     │            AsrResult                │
                     │  {text, text_with_punc, words[],    │
                     │   segments[], keyword_hits[],       │
                     │   emotion, confidence}              │
                     └────────────────────────────────────┘
```

---

## 六、文件结构规划

```
src/plugins/asr/
├── asr_config.h/cpp           # ✅ 已有: ASR 配置
├── asr_engine.h/cpp           # ✅ 已有: ASR 引擎入口
├── asr_encoder.h/cu           # ✅ 已有: 24L 双向 Transformer
├── asr_decoder.h/cu           # ✅ 已有: 28L Qwen3 Decoder
├── asr_plugin.h/cpp           # ✅ 已有: 插件接口
├── audio_ops.h/cu             # ✅ 已有: CUDA 算子
├── audio_utils.h/cpp          # ✅ 已有: 音频预处理
│
├── vad_engine.h/cu            # 🆕 Phase 1: FSMN-VAD Neural VAD
├── vad_config.h               # 🆕 Phase 1: VAD 配置 (阈值/窗口等)
│
├── speaker_encoder.h/cu       # 🆕 Phase 2: CAM++ GPU Speaker Encoder
├── speaker_manager.h          # 🆕 Phase 2: 说话人注册/识别管理
│
├── keyword_spotter.h          # 🆕 Phase 3: 关键词识别 (header-only)
│
├── punctuation.h              # 🆕 Phase 4: 标点恢复 (header-only)
│
├── aligner_engine.h/cpp       # 🆕 Phase 5: Qwen3-ForcedAligner Engine
│
├── speaker_diarizer.h         # 🆕 Phase 6: 说话人分割 (header-only)
│
└── emotion_head.h/cu          # 🆕 Phase 7: 情感分类头 (预留)
```

---

## 七、实施路线图与优先级

```
         无需新模型                    需要权重转换                  需要新 Engine
       ┌──────────┐              ┌────────────────┐           ┌──────────────┐
Week 1 │ Phase 4  │  Week 3-4   │   Phase 1      │  Week 7+  │   Phase 5    │
       │ 标点恢复  │  ──────────→│   Neural VAD   │──────────→│   时间戳     │
       │ (LLM+规则)│             │   (FSMN-VAD)   │           │   (FA-0.6B)  │
       └──────────┘              └────────────────┘           └──────────────┘
            │                          │                            │
            ▼                          ▼                            ▼
       ┌──────────┐              ┌────────────────┐           ┌──────────────┐
Week 2 │ Phase 3  │  Week 5-6   │   Phase 2      │  Week 8+  │   Phase 6    │
       │ 关键词    │  ──────────→│   CAM++ 说话人  │──────────→│   说话人分割  │
       │ (文本匹配)│             │   (GPU Encoder) │           │   (在线聚类)  │
       └──────────┘              └────────────────┘           └──────────────┘
                                                                    │
                                                                    ▼
                                                              ┌──────────────┐
                                                    Future    │   Phase 7    │
                                                              │   情感识别    │
                                                              └──────────────┘
```

### 里程碑

| 里程碑 | 完成阶段 | 交付物 | 状态 |
|--------|---------|--------|------|
| **M1: 基础增强** | Phase 3 + 4 | 标点恢复 + 关键词识别, 零新增模型 | ✅ 完成 |
| **M2: VAD 升级** | Phase 1 | Neural VAD 替代能量阈值, 大幅提升鲁棒性 | ✅ 完成 |
| **M3: 说话人** | Phase 2 | CAM++ 说话人识别, GPU 加速, 192-dim | ✅ 完成 |
| **M4: 完整流水线** | Phase 5 + 6 | 时间戳 + 说话人分割, 完整语音理解 | ✅ 完成 |
| **M5: 情感** | Phase 7 | 情感识别 (需标注数据) | 未开始 |

---

## 八、模型权重获取统一策略

系统无 PyTorch, 所有新模型需 **预转换为 safetensors** 后用 `SafetensorsLoader` 加载:

```bash
# 一次性创建转换环境
conda create -n model_convert python=3.10
conda activate model_convert
pip install torch modelscope safetensors

# ── FSMN-VAD ──
python -c "
from modelscope import snapshot_download
import torch, safetensors.torch as st
d = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
s = torch.load(f'{d}/model.pt', map_location='cpu')
st.save_file({k: v.float() for k, v in s.items()}, 'fsmn_vad.safetensors')
for k, v in s.items(): print(f'  {k}: {list(v.shape)}')
"
# → /home/rm01/models/dev/asr/fsmn_vad/

# ── CAM++ ──
python -c "
from modelscope import snapshot_download
import torch, safetensors.torch as st
d = snapshot_download('iic/speech_campplus_sv_zh-cn_16k-common')
s = torch.load(f'{d}/campplus_cn_common.bin', map_location='cpu')
st.save_file({k: v.float() for k, v in s.items()}, 'campplus.safetensors')
for k, v in s.items(): print(f'  {k}: {list(v.shape)}')
"
# → /home/rm01/models/dev/asr/campplus/
```

**确认清单** (转换后验证):
- [ ] 打印所有 tensor name + shape, 确认与源代码中的 key 名一致
- [ ] 检查 safetensors 文件大小与预期一致
- [ ] 用 SafetensorsLoader 加载测试, 确认无异常

---

## 九、超越 FunASR 的关键差异

### 1. 纯 C++/CUDA, 零 Python 依赖
- **FunASR**: Python + PyTorch + ONNX, pip install 数百依赖, 启动加载 10-30s
- **我们**: 单一 C++ 二进制, safetensors 直接 mmap 加载, 启动 <1s, 内存零冗余

### 2. 统一内存零拷贝 (Jetson AGX Thor)
- **FunASR**: CPU RAM + GPU VRAM 分离, 需显式 H2D/D2H 拷贝
- **我们**: LPDDR5X 统一内存, cudaMalloc 直接管理, 模型权重/音频特征/中间结果全在同一内存空间

### 3. LLM 同进程协同
- **FunASR**: ASR 和 LLM 独立部署, 需 HTTP/gRPC 通信
- **我们**: ASR + LLM + TTS 同进程, 标点恢复直接调用 LLM forward, 关键词触发 LLM 动作, 零网络延迟

### 4. 深度 Qwen3 优化
- **FunASR**: 通用框架, 支持 Paraformer/Whisper/SenseVoice 等多后端
- **我们**: 针对 Qwen3 系列 (ASR-1.7B + ForcedAligner-0.6B) 极致优化, CUTLASS SM110 Tensor Core, 自定义 kernel

### 5. 模型特征复用
- **FunASR**: VAD+ASR+Punc+SPK 各自独立模型, 共 518M+ 参数
- **我们**: ASR encoder 1.7B 特征复用于 KWS/情感, LLM 复用于标点, 新增模型仅 VAD 0.4M + CAM++ 7.2M = 7.6M

### 6. 嵌入式/边缘部署
- **FunASR**: 需要 Python 运行时 + CUDA Toolkit + pip 包管理
- **我们**: 静态编译单一二进制, 可直接部署到 Jetson/嵌入式设备, 无环境依赖

---

## 十、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| FSMN-VAD 权重格式不兼容 | Phase 1 延迟 | 先打印 tensor shapes 确认; 降级用能量 VAD |
| CAM++ 精度不达预期 | Phase 2 返工 | ECAPA-TDNN 已在 TTS 可用作 fallback |
| ForcedAligner 内存不足 (27B LLM 场景) | Phase 5 不可用 | 按需加载, 或只在 4B 场景启用 |
| LLM 标点恢复延迟过高 | Phase 4 用户体验差 | 异步处理, 先返回无标点, 后台补标点 |
| 关键词误触发率高 | Phase 3 体验差 | 提高置信度门限, 加确认步骤 |

---

## 十一、实现状态

> **更新日期**: 2026-03-14

所有 Phase 1-6 已完成实现，功能测试全部通过。

| Phase | 模块 | 文件 | 状态 | 测试 |
|-------|------|------|------|------|
| **Phase 1** | FSMN-VAD | `vad_engine.h`, `vad_config.h` | ✅ 已实现 | 7/7 通过 |
| **Phase 2** | CAM++ 说话人编码器 | `speaker_encoder.h` | ✅ 已实现 | 3/3 通过 |
| **Phase 3** | 关键词识别 | `keyword_spotter.h` | ✅ 已实现 | 10/10 通过 |
| **Phase 4** | 标点恢复 | `punctuation.h` | ✅ 已实现 | 10/10 通过 |
| **Phase 5** | 强制对齐 | `aligner_engine.h` | ✅ 已实现 | 10/10 通过 |
| **Phase 6** | 说话人分割 | `speaker_diarizer.h` | ✅ 已实现 | 5/5 通过 |
| **集成** | 端到端流水线 | — | ✅ 通过 | 1/1 通过 |

**功能测试汇总**: 46 passed, 0 failed, 0 skipped (2959.55 ms)

测试入口: `tests/test_asr_functional.cpp`，运行方式:
```bash
./build/qwen35-thor test --filter asr
```

### CAM++ 修复记录

CAM++ speaker_encoder.h 在首次实现时存在 8 个架构不匹配问题，已全部修复:

1. FCM 操作顺序 (conv2 应在 ResBlocks 之后)
2. ResBlock stride 修正为 (2,1)，仅频率维度下采样
3. TDNN tensor 名称 (`xvector.tdnn.linear.weight`)
4. Transit 层改为 pre-norm (BN→ReLU→Conv1d)
5. out_nonlinear 位置 (StatsPool 之前)
6. Dense 输出维度 512→192，tensor 名称修正
7. CAM 层增加 seg_pooling，context 改为 temporal
8. Conv1d 增加 dilation 支持 (block2/3 dilation=2)

修复后输出 192-dim L2-normalized embeddings，与 FunASR 参考实现一致。

---

## 十二、ASR 性能基线 (Benchmark Baseline)

> **测试日期**: 2026-03-14  
> **测试硬件**: NVIDIA Jetson AGX Thor (SM110a Blackwell), 128 GB LPDDR5X, MAXN  
> **ASR 模型**: Qwen3-ASR-1.7B (708 tensors, 3400.0 MB)  
> **配置**: Encoder 24L×1024, Decoder 28L×2048, vocab=151936, MRoPE=[24,20,20]  
> **采样**: temperature=0.0 (greedy), max_tokens=448  
> **命令**: `./build/qwen35-thor asr --model-dir <path> <audio.wav>`

### 12.1 模型加载

| 指标 | 数值 |
|------|------|
| 权重文件 | 2 shards, 708 tensors |
| 权重大小 | 3400.0 MB |
| 加载时间 | ~1.3s |
| Encoder workspace | 73.4 MB |
| Decoder KV cache | 224.0 MB |
| Decoder workspace | 80.3 MB |
| **总内存占用** | **~3778 MB** |

### 12.2 推理性能

| 音频 | 时长 | 采样率 | Mel 帧数 | Encoder Token | Encode (ms) | Prefill (ms) | Decode (ms) | 输出 Token | Decode tok/s | Total (ms) | RTF |
|------|------|--------|----------|---------------|-------------|--------------|-------------|------------|-------------|------------|-----|
| test_speech_sim.wav | 3.0s | 16kHz | 298 | 39 | 319.7 | 43.5 | 68.6 | 3 | 43.7 | 431.8 | 0.144 |
| test_speech_real.wav† | 4.1s | 8kHz | 409 | 54 | 414.6 | 56.2 | 327.4 | 15 | 45.8 | 798.3 | 0.195 |
| bench_10s.wav | 10.0s | 16kHz | 998 | 130 | 937.9 | 106.9 | 71.1 | 3 | 42.2 | 1115.9 | 0.112 |
| bench_30s.wav | 30.0s | 16kHz | 2998 | 390 | 2773.5 | 532.7 | 78.1 | 3 | 38.4 | 3384.2 | 0.113 |

† test_speech_real.wav 数据为 3 次运行取平均值  
RTF = Real-Time Factor = 处理时间 / 音频时长 (越低越好, <1.0 即实时)

### 12.3 关键性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **Encoder 吞吐** | ~1000 mel_frames/s | 线性扩展, 长音频更高效 (30s: 1081 f/s) |
| **Prefill 吞吐** | ~750 tokens/s | 包含 encoder tokens + prompt template |
| **Decode 速度** | ~42-46 tok/s | ~22 ms/token, 受权重带宽瓶颈 |
| **实时倍率** | 5.1-8.9× | 4s 音频 5.1×, 10-30s 音频 ~8.9× (均远超实时) |
| **RTF** | 0.11-0.20 | 均大幅低于 1.0, 满足实时需求 |

### 12.4 Encoder 扩展性分析

```
Mel Frames    Enc Tokens    Encode (ms)    Throughput (f/s)
   298            39          319.7           932
   409            54          414.6          ~987
   998           130          937.9          1064
  2998           390         2773.5          1081
```

Encoder 时间与 mel 帧数近似线性, 长序列因 GEMM 效率提升略有优势。  
Mel 帧到 encoder token 的压缩比约 **7.7:1** (2× Conv2d stride + Transformer)。

### 12.5 测试音频说明

| 文件 | 时长 | 采样率 | 内容 | 输出文本 |
|------|------|--------|------|----------|
| test_speech_sim.wav | 3.0s | 16kHz | 合成语音 | (无有效识别) |
| test_speech_real.wav | 4.1s | 8kHz | 录制语音 | "Hello, world. This is the top speed transmission system." |
| bench_10s.wav | 10.0s | 16kHz | 重复合成语音 | (无有效识别) |
| bench_30s.wav | 30.0s | 16kHz | 重复合成语音 | (无有效识别) |

> **注**: 合成语音 (由正弦波调制生成) 仅输出 3 个 token, decode 阶段极短。  
> 真实语音场景下 decode 占比更高, RTF 会相应增加但仍远低于 1.0。

### 12.6 优化方向

以下优化已在 v2 中实施:

| 优化 | 实际收益 | 说明 |
|------|----------|------|
| ✅ Radix-2 FFT | Mel -25× (30s: 2340→37ms) | 替换 O(N²) 朴素 DFT, Cooley-Tukey 算法 |
| ✅ 稀疏 Mel Filterbank | Mel -58% | 跳过零值频段, 内积从 201→~15 次/bin |
| ✅ MHA V 并行化 | Prefill -60%, Encode -5% | 修复 V 累加序列化 (thread 0→全 head_dim) |
| ✅ 移除冗余 cudaStreamSync | 52 个 sync 消除 | Encoder/Decoder 层内同步不必要 |
| ✅ Conv2D im2col+GEMM | Conv -68% (246→79ms) | 替换朴素 kernel, cuBLAS tensor core |
| ❌ Encoder CUTLASS SM110 | 无改善 (反而 decoder 回退) | 已评估并回退, cuBLAS 对 ASR 维度更优 |

### 12.7 优化后性能 (v2)

> **测试日期**: 2026-07 (v2 优化后)

| 音频 | 时长 | Encode (ms) | Prefill (ms) | Decode (ms) | Total (ms) | RTF | vs 基线 |
|------|------|-------------|--------------|-------------|------------|-----|---------|
| test_speech_real.wav | 4.1s | 93 | 37 | 425 | 555 | 0.135 | -30.5% |
| bench_10s.wav | 10.0s | 106 | 54 | 69 | 229 | 0.023 | -79.5% |
| bench_30s.wav | 30.0s | 163 | 214 | 78 | 455 | 0.015 | -86.6% |

**Encoder Profile (30s audio, GPU CUDA event timing):**

| 阶段 | 基线 | 优化后 | 改善 |
|------|------|--------|------|
| Conv2D frontend | ~246ms | 79ms | -67.8% |
| 24 Transformer layers | ~46ms | 46ms | — |
| Post-processing | ~0.2ms | 0.2ms | — |
| Mel (CPU) | ~2340ms | 37ms | -98.4% |
| **GPU Total** | ~292ms | 125ms | -57.2% |
| **Encoder Total** | ~2631ms | 163ms | -93.8% |

**实时倍率**: 30s 音频 66× 实时 (RTF=0.015), 基线 8.9× 实时 (RTF=0.113)

### 12.8 剩余优化空间

| 方向 | 预期收益 | 说明 |
|------|----------|------|
| GPU Mel (cuFFT) | Mel 37→5ms | CPU→GPU 迁移, 消除 H2D 往返 |
| Decoder GEMV + MTP | Decode +50~100% | 复用主引擎的 GEMV 优化和 MTP 投机解码 |
| Batch 推理 | 吞吐 N× | 多路音频合并 batch, 权重读一次 |
