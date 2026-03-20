# ASR 插件架构审计报告

> 审计日期: 2026-03-20
> 审计范围: `src/plugins/asr/` (24 文件), `src/serve/serve.cpp` (ASR 相关段落), `src/serve/serve.h`

## 一、组件清单

### 1.1 核心推理

| 文件 | 职责 | 关键接口 |
|------|------|----------|
| `asr_engine.h` | ASR 推理引擎 | `ASREngine::transcribe()`, `transcribe_batch()` |
| `asr_encoder.h/cu` | 音频编码器 (PCM → features) | `AudioEncoder::forward()` |
| `asr_decoder.h/cu` | 文本解码器 | `TextDecoder::forward()` |

### 1.2 音频前端

| 文件 | 职责 | 关键接口 |
|------|------|----------|
| `audio_utils.h` | 音频 I/O (WAV/MP3/M4A → 16kHz PCM) | `load_audio_from_memory()`, `compute_mel()`, `resample()` |
| `audio_ops.h/cu` | CUDA 音频算子 (RMSNorm, GQA, Conv1d) | 独立于 LLM ops (不同 norm 约定) |
| `mel_gpu.h` | GPU Mel (cuFFT + Kaldi 窗口) | `GpuMelExtractor::compute_gpu()` |

### 1.3 VAD

| 文件 | 职责 | 性能 |
|------|------|------|
| `vad_engine.h` | CPU FSMN-VAD (0.4M params) | 80-100s 处理 60min 音频 |
| `vad_gpu.h` | GPU FSMN-VAD (cuFFT batch) | <1s 处理 60min 音频 (>80× 加速) |
| `vad_config.h` | VAD 配置常量 | FSMN 4 层, n_mels=80 |

### 1.4 说话人编码器

| 文件 | 职责 | 性能 |
|------|------|------|
| `speaker_encoder.h` | CPU CAM++ (6.9M params → 192-dim) | 单段提取 |
| `speaker_encoder_gpu.h` | GPU CAM++ (16-stream batch) | 1331 chunks: 33s→3s (10.5×) |

### 1.5 后处理与辅助

| 文件 | 职责 |
|------|------|
| `aligner_engine.h` + `forced_aligner_server.py` | ForcedAligner 子进程 (JSON 协议, model 常驻) |
| `punctuation.h` | 标点恢复 (规则 + 可选 LLM) |
| `keyword_spotter.h` | 关键词唤醒 (**stub, 未实现**) |
| `asr_plugin.h` | 插件抽象 (`SubprocessAsrPlugin` / `NativeAsrPlugin`) |
| `asr_config.h` | ASR 配置加载 |

### 1.6 依赖关系

```
audio_utils → mel_gpu → speaker_encoder_gpu  (PCM → Mel → 192-dim embedding)
vad_gpu → speaker_encoder_gpu               (VAD 段 → CAM++ batch 提取)
aligner_engine → forced_aligner_server.py   (子进程 JSON 通信)
punctuation → serve.cpp LLM callback        (可选, 长文本走 LLM)
```

---

## 二、两个工况

### 工况 1: 实时识别 (ASR → LLM → TTS 级联)

**目标**: 快速识别 + 快速响应 + 说话人区分

**入口**: WebSocket `/v1/realtime` 和 `/v1/voice`

**数据流**:
```
客户端 PCM16LE 16kHz → WebSocket 二进制帧 (连续流入)
         │
    ┌────┴─────────────────────────────────┐
    │ 帧级 VAD (RMS 阈值 0.01f)             │  ← 能量检测, 非 FSMN
    │ 静音 ≥ 600-800ms → 切断语音段          │
    │ 每 2s → partial ASR (asr.partial)     │
    └────┬─────────────────────────────────┘
         │ 完整语音段
         ▼
    transcribe_pcm() → 全段 ASR → 裸文本
         │
         ▼
    identify_speaker() → CAM++ 192-dim → SpeakerManager.identify(cos≥0.65)
         │                                 返回: name + similarity
         ▼
    char_count ≥ 2 → 触发 LLM (全部文本, 无意图过滤)
         │
         ▼
    LLM streaming → 句子分割 → TTS queue → PCM16 24kHz 返回客户端
```

**关键参数**:
- `VAD_ENERGY_THRESHOLD = 0.01f`
- `VAD_SILENCE_MS = 800` (voice) / `600` (realtime)
- `STREAMING_ASR_CHUNK_S = 2.0f`
- `voice_max_output_tokens = 150`
- Barge-in 阈值: `RMS > 0.03f` (3× 正常)

**已有特性**:
- [x] 连续语音流输入
- [x] VAD 切断 + partial ASR
- [x] CAM++ 单段说话人识别
- [x] LLM streaming 生成
- [x] TTS streaming 合成 (producer-consumer, 句子粒度)
- [x] Barge-in 打断 (RMS 阈值 / 显式 interrupt 消息)
- [x] `asr_to_llm` 全局开关
- [x] 多轮对话历史 (voice_max_turns)

### 工况 2: 录音转写 (V4 Pipeline)

**目标**: 高准确率 + 精确说话人分割 + 完善段落格式

**入口**: HTTP `POST /v1/audio/transcriptions` (speaker=true)

**数据流 (6 阶段)**:
```
Phase 1: 音频解析 + 全文 ASR
         • load_audio_from_memory() → PCM
         • Energy-valley 分割 (>100s 音频, 90s 段)
         • 全文 ASR → 统一转录
         
Phase 2: ForcedAligner (并行线程)          ─┐
         • align(pcm_path, full_text)      │ 并行执行
         • → [word, start_ms, end_ms]      │
                                           │
Phase 3a: GPU FSMN-VAD                    ─┘
         • gpu_vad_engine_.detect_all()
         • → VAD segments [start_ms, end_ms]
         
Phase 3b: CAM++ 谱聚类
         • CHUNK_FRAMES=300 (3s) 切片
         • GPU Mel → CAM++ batch → 192-dim embeddings
         • 谱聚类: 余弦相似度 + 时间权重 (TEMPORAL_ALPHA=0.65)
         • NME 自动选 k + K-Means
         • **时间一致性平滑** (±2 窗口, cos_margin=-0.04, 2 轮)
         • → [cluster_id] per chunk
         
Phase 4: Word → Speaker 归属
         • 对齐结果 × VAD speaker 标签
         • 零长度/重叠/间隙处理
         
Phase 5: 段合并
         • gap≤2s 同 speaker 合并
         • 短段 (≤3 chars, <2s) 吸收
         
Phase 6: 后处理
         • 6a-6c: 标点恢复 (。？！)
         • 6d: 按 speaker 变化 + 句子末尾重新分段
         • 6.5: speaker island 平滑 (<3s 孤立段消除)
         • 6.55: 口语归一化 (去除"就是"/"对吧"等 filler)
         • 6.6: Gap 填充 (≥200ms 空白区间, 用最近 VAD chunk speaker)
```

**已有特性**:
- [x] 6 阶段完整 pipeline
- [x] Phase 2/3 并行执行
- [x] GPU 加速 VAD + Mel + CAM++
- [x] 谱聚类 (NME 自动 k 选择 + 时间权重)
- [x] 时间一致性平滑 (+1.4pp, 73.1%)
- [x] ForcedAligner word-level 时间戳
- [x] 标点恢复 + 口语归一化
- [x] Embedding dump 用于离线分析

---

## 三、设计评估

### 3.1 合理的部分 ✅

| 维度 | 评价 |
|------|------|
| **模块化** | `AsrPlugin` 接口抽象干净, subprocess/native 可切换 |
| **GPU 加速** | VAD(cuFFT), Mel(cuFFT), CAM++(16-stream cuBLAS) 全 GPU |
| **V4 Pipeline** | 6 阶段流水线设计成熟, Phase 2/3 并行, 后处理丰富 |
| **谱聚类** | NME 自动选 k + TEMPORAL_ALPHA=0.65 + 时间平滑, 效果稳定 |
| **ForcedAligner** | 子进程长驻, model 加载一次, JSON 协议简洁 |
| **资源管理** | 3 把互斥锁保护有状态组件 (speaker_encoder, vad, aligner) |

### 3.2 需要关注的问题 ⚠️

#### 问题 1: 两个工况组件复用率低 (合理但需意识到)

| 组件 | 实时识别 | 录音转写 | 说明 |
|------|---------|---------|------|
| FSMN-VAD (GPU) | ❌ 不用 | ✅ Phase 3a | 实时用 RMS 能量阈值 |
| CAM++ batch | ❌ 单段 | ✅ 1331 chunks | 实时只做 identify |
| ForcedAligner | ❌ 不用 | ✅ Phase 2 | 实时无 word-level 时间戳 |
| 谱聚类 | ❌ 不用 | ✅ Phase 3b | 实时只做 cos 阈值匹配 |
| 标点恢复 | ❌ 不用 | ✅ Phase 6 | 实时直接输出裸文本 |
| 时间平滑 | ❌ 不用 | ✅ Phase 3b/6.5 | — |

**评价**: 低复用率是合理的 (实时场景延迟约束不同), 但实时模式的 VAD 太简陋 — RMS 阈值在噪声环境下易误判。

#### 问题 2: 实时模式缺乏 "目标说话人" 路由 ★★★ 最关键

当前实现:
- `identify_speaker()` 每段音频 → 返回最近已注册说话人
- `asr_to_llm` 是全局开关 (全送或全不送 LLM)
- **没有 "哪个说话人的话需要 LLM 响应" 的判断**

需求描述: "确认哪些信息是需要响应的说话人, 哪些识别文本是需要作为 prefill 而不做 decode"

**缺失功能**:
- `target_speaker_id` — 系统需要响应的说话人
- 非目标说话人文本 → 上下文 prefill (LLM 可见但不回复)
- 目标说话人文本 → 触发 LLM decode

#### 问题 3: 两个 WebSocket 端点功能重叠

| 特性 | `/v1/voice` | `/v1/realtime` |
|------|-------------|----------------|
| VAD | RMS 0.01, 静音 800ms | RMS 0.01, 静音 600ms |
| 说话人识别 | ✅ | ✅ |
| Barge-in | 显式 interrupt 消息 | 自动 RMS>0.03 |
| 协议命名 | `asr` / `llm.delta` | `input.transcription` / `response.delta` |

核心逻辑 ~90% 重复, 仅 VAD 触发方式和协议命名不同。

#### 问题 4: serve.cpp 体积过大

- `serve.cpp` > 8000 行
- V4 pipeline ~1700 行全部内联
- 两个 WS handler ~2000 行
- 谱聚类、时间平滑、word 归属等逻辑未独立模块

#### 问题 5: 实时 VAD 未用 FSMN

GPU FSMN-VAD 已实现 (<1ms 级别), 但实时模式仍用 RMS 能量检测。FSMN 可提升噪声场景鲁棒性。

#### 问题 6: 无 ASR 增量解码

实时模式每 2s 完整重新编码整段音频做 partial ASR, 非增量解码。导致 encoder 重复计算。

---

## 四、架构建议

### 4.1 短期 (当前框架内)

| 优先级 | 改进 | 影响 | 工作量 | 状态 |
|--------|------|------|--------|------|
| **P0** | 实时模式增加 `target_speaker` | 核心功能需求 | 中 | 🔲 待做 |
| P1 | V4 pipeline 抽取为独立类 | 可维护性 | 大 | 🔲 待做 |
| P1 | 统一两个 WS 端点 | 减少重复代码 | 中 | 🔲 待做 |
| P2 | 实时模式用 FSMN-VAD | 噪声鲁棒性 | 小 | 🔲 待做 |
| P2 | 实时 partial ASR 增量编码 | 降延迟 | 大 | 🔲 待做 |

### 4.2 中期

| 改进 | 说明 |
|------|------|
| 实时意图过滤 | 基于说话人 + 语义判断是否需响应, 非简单 `char_count≥2` |
| 实时多说话人追踪 | 滑动窗口说话人状态, 非每段独立 identify |
| KWS 触发 | 关键词唤醒 (keyword_spotter.h 已预留) |
| 会话持久化 | 断线重连不丢失上下文 |

### 4.3 `target_speaker` 设计方案 (P0)

```jsonc
// 客户端 session.update:
{
    "type": "session.update",
    "target_speaker": "Alice",          // 注册过的说话人名
    "other_speaker_mode": "prefill"     // "prefill" | "ignore" | "respond_all"
}
```

**行为**:
1. 每段 ASR → `identify_speaker()` → `speaker_name`
2. `speaker == target_speaker`:
   - `chat_history.push({"user", text})` → LLM decode → TTS 响应
3. `speaker != target_speaker && mode == "prefill"`:
   - `chat_history.push({"system", f"[{speaker}说]: {text}"})` → prefill only, **不 decode**
4. `speaker != target_speaker && mode == "ignore"`:
   - 不加入上下文

---

## 五、评分总结

| 维度 | 评分 | 说明 |
|------|------|------|
| **录音转写** | ★★★★☆ | V4 pipeline 成熟, 6 阶段完整, 73% 说话人准确率 |
| **实时识别** | ★★★☆☆ | ASR→LLM→TTS 链路可用, 缺目标说话人路由和 FSMN-VAD |
| **代码组织** | ★★☆☆☆ | 核心逻辑全在 serve.cpp, 两个 WS 端点大量重复 |
| **可扩展性** | ★★★☆☆ | 插件接口干净, 但 pipeline 逻辑未模块化 |
| **整体设计** | 🟢 方向正确 | 两工况差异化处理思路对, 主要缺实时说话人路由 |

---

## 六、API 端点清单

| 端点 | 方法 | 用途 | 工况 |
|------|------|------|------|
| `/v1/audio/transcriptions` | POST | 文件转写 + 说话人分割 | 录音转写 |
| `/v1/speakers` | GET | 列出已注册说话人 | 通用 |
| `/v1/speakers/register` | POST | 注册新说话人 | 通用 |
| `/v1/speakers/delete` | POST | 删除说话人 | 通用 |
| `/v1/realtime` | WS | 连续语音对话 (OpenAI 兼容) | 实时识别 |
| `/v1/voice` | WS | Push-to-talk 语音对话 | 实时识别 |
| `/v1/audio/speech` | POST | TTS 合成 | TTS |

---

## 七、模型依赖

| 模型 | 来源 | 大小 | 用途 |
|------|------|------|------|
| Qwen3-ASR-1.7B | 本地 safetensors | ~3.4 GB | 核心 ASR |
| CAM++ | `QWEN_SPEAKER_MODEL` | ~28 MB | 说话人编码 (192-dim) |
| FSMN-VAD | `QWEN_VAD_MODEL` | ~1.6 MB | 语音活动检测 |
| Qwen3-ForcedAligner-0.6B | 本地 | ~1.2 GB | Word-level 时间戳对齐 |

---

## 八、已知限制

1. **CAM++ 非确定性**: 16 CUDA stream 并行导致 embedding 每次略有不同 (±2-3pp 说话人准确率波动)
2. **说话人混淆**: 同性别/年龄说话人 (唐云峰↔石一) CAM++ embedding 物理重叠, 时间信息是唯一额外信号
3. **统一内存约束**: 所有模型共享 128 GB LPDDR5X, ASR/CAM++/VAD/Aligner 需考虑内存占用
4. **ForcedAligner 延迟**: 子进程通信 + Python 推理, 60min 音频 ~10-15s
5. **实时 ASR 非增量**: 每 2s partial 重新编码全段, encoder 计算有冗余
