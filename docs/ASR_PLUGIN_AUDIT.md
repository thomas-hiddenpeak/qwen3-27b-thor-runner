# ASR 插件架构审计报告

> 审计日期: 2026-03-20 (更新: 2026-03-20 — P0/P1/P2 完成, WS 统一 + target_speaker + FSMN-VAD)
> 审计范围: `src/plugins/asr/` (27 文件), `src/serve/` (HTTP + WS 接口层), `src/serve/serve.h`

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

### 1.5 Pipeline 编排层 (新增)

| 文件 | 职责 | 关键接口 |
|------|------|----------|
| `transcription_pipeline.h/cpp` | V4/V2/Plain 三种转写模式统一编排 (1760 行) | `TranscriptionPipeline::transcribe()` |
| `speaker_service.h` | Mel 提取 + CAM++ embedding + SpeakerManager 封装 (235 行) | `SpeakerService::identify()`, `compute_mel_80()` (static) |

`TranscriptionPipeline` 通过 `Dependencies` 结构注入所有组件引用 (不拥有所有权, serve 层负责生命周期)。
`SpeakerService` 持有 `SpeakerManager` + `mutex`, 提供线程安全的说话人识别 API。

### 1.5b WebSocket 会话层 (新增)

| 文件 | 职责 | 关键接口 |
|------|------|----------|
| `voice_session.h/cpp` | 统一 WS 语音会话 (1244 行) | `VoiceSession::run()` |
| `ws_utils.h` | 共享 WS/JSON 工具函数 (332 行, inline) | `ws::send_text()`, `ws::recv_frame()`, `ws::extract_tts_instruct()` |

`VoiceSession` 通过 `ProtocolMode{VOICE, REALTIME}` 枚举统一两个 WS 端点的核心逻辑。
内含 `SpeakerRouting` 结构实现 P0 目标说话人路由 (`target_speaker` + `other_speaker_mode`)。
serve.cpp 中 `handle_websocket_voice/realtime` 退化为 ~6 行薄 wrapper。

### 1.6 后处理与辅助

| 文件 | 职责 |
|------|------|
| `aligner_engine.h` + `forced_aligner_server.py` | ForcedAligner 子进程 (JSON 协议, model 常驻) |
| `punctuation.h` | 标点恢复 (规则 + 可选 LLM) |
| `keyword_spotter.h` | 关键词唤醒 (**stub, 未实现**) |
| `asr_plugin.h` | 插件抽象 (`SubprocessAsrPlugin` / `NativeAsrPlugin`) |
| `asr_config.h` | ASR 配置加载 |

### 1.7 依赖关系

```
serve.cpp (HTTP 接口 + WS thin wrapper)
  ├─ TranscriptionPipeline (录音转写 pipeline)
  │    ├─ NativeAsrPlugin (ASR 推理)
  │    ├─ SpeakerService (Mel + CAM++ + SpeakerManager)
  │    │    ├─ GpuMelExtractor (cuFFT GPU Mel)
  │    │    ├─ GpuSpeakerEncoder (16-stream batch CAM++)
  │    │    └─ SpeakerManager (cos 阈值匹配/注册)
  │    ├─ VadEngine / GpuVadEngine (FSMN-VAD)
  │    ├─ AlignerEngine (ForcedAligner 子进程)
  │    └─ PunctuationRestorer (标点恢复)
  └─ VoiceSession (实时语音会话, 统一 /v1/voice + /v1/realtime)
       ├─ ProtocolMode::VOICE / REALTIME
       ├─ SpeakerRouting (target_speaker + other_mode)
       ├─ NativeAsrPlugin (via ServeApp)
       ├─ SpeakerService (via ServeApp)
       └─ ws_utils.h (共享 WS/JSON 工具)
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
    ┌────┴──────────────────────────────────────┐
    │ FSMN-VAD 流式端点检测 (per-session 副本)     │  ← 神经网络 VAD
    │ max_end_silence: 600-800ms → 语音段输出      │  ← 降级: RMS 阈值
    │ 每 2s → partial ASR (asr.partial)            │
    └────┬──────────────────────────────────────┘
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
- FSMN-VAD: `speech_noise_thres = 0.6`, `max_end_silence_time = 800/600ms`
- RMS 降级: `VAD_ENERGY_THRESHOLD = 0.01f` (FSMN 未加载时)
- `STREAMING_ASR_CHUNK_S = 2.0f`
- `voice_max_output_tokens = 150`
- Barge-in 阈值: `RMS > 0.03f` (3× 正常, 仍用 RMS)

**已有特性**:
- [x] 连续语音流输入
- [x] VAD 切断 + partial ASR
- [x] CAM++ 单段说话人识别
- [x] LLM streaming 生成
- [x] TTS streaming 合成 (producer-consumer, 句子粒度)
- [x] Barge-in 打断 (RMS 阈值 / 显式 interrupt 消息)
- [x] `asr_to_llm` 全局开关
- [x] 多轮对话历史 (voice_max_turns)
- [x] **目标说话人路由** (`target_speaker` + `other_speaker_mode`: respond_all/prefill/ignore) ← P0 新增
- [x] **统一 WS 会话** (VoiceSession 类, ProtocolMode 分派) ← P1 新增
- [x] **FSMN 神经网络 VAD** (per-session CPU VadEngine, 替代 RMS 能量检测, 自动降级) ← P2 新增

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
- [x] 时间一致性平滑 (±2 窗口, cos_margin=-0.04)
- [x] **说话人准确率**: 88.7% avg (86.6-91.2%, 3 run, asrTest2.mp3, 详见 `SPEAKER_DIARIZATION_EXPERIMENTS.md`)
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
| **Pipeline 分层** | `TranscriptionPipeline` + `SpeakerService` 独立于 HTTP 层, 可复用/可测试 |
| **谱聚类** | NME 自动选 k + TEMPORAL_ALPHA=0.65 + 时间平滑, 88.7% avg (3s 非重叠分块为局部最优, 详见实验报告) |
| **ForcedAligner** | 子进程长驻, model 加载一次, JSON 协议简洁 |
| **资源管理** | Dependencies 注入 + 3 把互斥锁保护有状态组件 |

### 3.2 需要关注的问题 ⚠️

#### 问题 1: 两个工况组件复用率低 (合理但需意识到)

| 组件 | 实时识别 | 录音转写 | 说明 |
|------|---------|---------|------|
| FSMN-VAD (CPU) | ✅ 流式端点检测 | — | per-session 副本, 替代 RMS |
| FSMN-VAD (GPU) | — | ✅ Phase 3a | batch 全段检测 |
| CAM++ batch | ❌ 单段 | ✅ 1331 chunks | 实时只做 identify |
| ForcedAligner | ❌ 不用 | ✅ Phase 2 | 实时无 word-level 时间戳 |
| 谱聚类 | ❌ 不用 | ✅ Phase 3b | 实时只做 cos 阈值匹配 |
| 标点恢复 | ❌ 不用 | ✅ Phase 6 | 实时直接输出裸文本 |
| 时间平滑 | ❌ 不用 | ✅ Phase 3b/6.5 | — |

**评价**: 低复用率是合理的 (实时场景延迟约束不同)。实时模式已升级为 CPU FSMN-VAD 流式检测。

#### ~~问题 2: 实时模式缺乏 "目标说话人" 路由~~ ✅ 已解决 (commit d1c0a81)

**实现方案** (VoiceSession `SpeakerRouting`):
- `target_speaker` — 客户端通过 `config` / `session.update` 设置目标说话人名
- `other_speaker_mode` — `respond_all` (默认) / `prefill` / `ignore`
- `evaluate_speaker()` 每段 ASR 后判断: RESPOND → LLM decode, PREFILL → `[{speaker}说]: {text}` 注入 system, IGNORE → 丢弃
- 两个协议均支持 (Voice: `config` 事件, Realtime: `session.update` 事件)

#### ~~问题 3: 两个 WebSocket 端点功能重叠~~ ✅ 已解决 (commit d1c0a81)

- **VoiceSession** 统一两个 WS 端点, `ProtocolMode{VOICE, REALTIME}` 分派差异行为
- 协议差异通过 `MsgNames` 结构映射 (事件名/响应名自动适配)
- VAD 参数通过 `VadConfig` 结构差异化 (Voice: 800ms/500ms, Realtime: 600ms/300ms)
- serve.cpp handler 退化为 ~6 行 wrapper
- **结果**: serve.cpp 6223 → 4692 行 (-24.6%), 消除 ~1551 行重复代码

#### ~~问题 4: serve.cpp 体积过大~~ ✅ 已解决 (累计 -46%)

- ~~`serve.cpp` > 8000 行~~ → ~~6223 行~~ → **4692 行 (-46%)**
- ~~V4 pipeline ~1700 行全部内联~~ → **已抽取到 `transcription_pipeline.cpp` (1760 行)**
- ~~谱聚类、时间平滑、word 归属等逻辑未独立模块~~ → **已模块化**
- `handle_audio_transcriptions` 从 ~2400 行 → ~170 行 (纯 HTTP 委托)
- `compute_mel_80` / `identify_speaker` 委托给 `SpeakerService`
- ~~两个 WS handler ~2000 行仍在 serve.cpp~~ → **已抽取到 `voice_session.cpp` (1244 行) + `ws_utils.h` (332 行)**

#### ~~问题 5: 实时 VAD 未用 FSMN~~ ✅ 已解决 (commit 59e4bf5)

- **方案**: per-session `VadEngine` 副本 (~1.6MB), 从 `ServeApp.vad_engine_` 拷贝权重
- `max_end_silence_time` 按协议设置 (Voice 800ms / Realtime 600ms)
- FSMN 状态机驱动语音段检测 (speech onset + endpoint), 输出 `VadSegment{start_ms, end_ms, pcm}`
- 超时 (30s) 时 flush FSMN 产出最终段
- RMS 保留: barge-in 自动打断 + audio.level 通知
- **降级**: FSMN 模型未加载时自动退回 RMS 能量 VAD

#### 问题 6: 无 ASR 增量解码

实时模式每 2s 完整重新编码整段音频做 partial ASR, 非增量解码。导致 encoder 重复计算。

---

## 四、架构建议

### 4.1 短期 (当前框架内)

| 优先级 | 改进 | 影响 | 工作量 | 状态 |
|--------|------|------|--------|------|
| ~~P0~~ | ~~实时模式增加 `target_speaker`~~ | ~~核心功能需求~~ | ~~中~~ | ✅ **已完成** (commit d1c0a81) |
| ~~P1~~ | ~~V4 pipeline 抽取为独立类~~ | ~~可维护性~~ | ~~大~~ | ✅ **已完成** (commit 196f01f) |
| ~~P1~~ | ~~统一两个 WS 端点~~ | ~~减少重复代码~~ | ~~中~~ | ✅ **已完成** (commit d1c0a81) |
| ~~P2~~ | ~~实时模式用 FSMN-VAD~~ | ~~噪声鲁棒性~~ | ~~小~~ | ✅ **已完成** (commit 59e4bf5) |
| P2 | 实时 partial ASR 增量编码 | 降延迟 | 大 | 🔲 待做 |

### 4.2 中期

| 改进 | 说明 |
|------|------|
| 实时意图过滤 | 基于说话人 + 语义判断是否需响应, 非简单 `char_count≥2` |
| 实时多说话人追踪 | 滑动窗口说话人状态, 非每段独立 identify |
| KWS 触发 | 关键词唤醒 (keyword_spotter.h 已预留) |
| 会话持久化 | 断线重连不丢失上下文 |

### 4.3 ~~`target_speaker` 设计方案~~ ✅ 已实现 (commit d1c0a81)

**实现位置**: `voice_session.h` — `SpeakerRouting` + `evaluate_speaker()`

```jsonc
// Voice 协议 — config 事件:
{ "type": "config", "target_speaker": "Alice", "other_speaker_mode": "prefill" }
// Realtime 协议 — session.update 事件:
{ "type": "session.update", "target_speaker": "Alice", "other_speaker_mode": "prefill" }
```

**已实现行为**:
1. 每段 ASR → `identify_speaker()` → `speaker_name`
2. `evaluate_speaker()` → `SpeakerAction{RESPOND, PREFILL, IGNORE}`
3. RESPOND: `chat_history.push({"user", text})` → LLM decode → TTS 响应
4. PREFILL: `chat_history.push({"system", "[{speaker}说]: {text}"})` → 上下文可见, **不 decode**
5. IGNORE: 丢弃, 不加入上下文
6. 默认 `other_speaker_mode = "respond_all"` (未设置 target_speaker 时所有人都触发响应)

---

## 五、评分总结

| 维度 | 评分 | 说明 |
|------|------|------|
| **录音转写** | ★★★★☆ | V4 pipeline 成熟, 6 阶段完整, 88.7% 说话人准确率 (asrTest2, 4 speakers) |
| **实时识别** | ★★★★★ | ASR→LLM→TTS→VAD 全链路完整, FSMN-VAD + 说话人路由已实现 |
| **代码组织** | ★★★★☆ | Pipeline + WS 会话均已模块化, serve.cpp 4692 行 (-46%) |
| **可扩展性** | ★★★★☆ | 插件接口干净, Pipeline 依赖注入可复用/可测试 |
| **整体设计** | 🟢 架构成熟 | 两工况分层清晰, P0/P1/P2 全部完成, 仅剩增量 ASR 编码 |

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

1. **CAM++ 非确定性**: 16 CUDA stream 并行导致 embedding 每次略有不同 (±2.3pp 波动: 86.6-91.2%, 实测 3 run)
2. **说话人混淆**: 同性别/年龄说话人 (唐云峰↔石一) CAM++ embedding 物理重叠, 时间信息 (α=0.65) 是唯一额外信号
3. **分块策略敏感**: 3s 非重叠 + α=0.65 为局部最优; VAD-segment 粒度 (-10.6pp), 3s 重叠 (-7.5pp), 5s 分块 (-9.7pp), α=0.5 (-23.1pp) 均回归 (详见 `SPEAKER_DIARIZATION_EXPERIMENTS.md`)
4. **统一内存约束**: 所有模型共享 128 GB LPDDR5X, ASR/CAM++/VAD/Aligner 需考虑内存占用
5. **ForcedAligner 延迟**: 子进程通信 + Python 推理, 60min 音频 ~10-15s
6. **实时 ASR 非增量**: 每 2s partial 重新编码全段, encoder 计算有冗余

---

## 九、说话人分割实验记录

> 完整报告: `docs/SPEAKER_DIARIZATION_EXPERIMENTS.md`
> 评估脚本: `tmp/eval_speaker.py` (逐秒对齐 + 最优排列映射)
> 测试音频: `tests/assets/asrTest2.mp3` (60 min, 4 speakers)

| 实验 | 参数变化 | 准确率 (avg) | vs 基线 | 结论 |
|------|---------|-------------|---------|------|
| **基线** | CHUNK=3s, α=0.65, τ=12s | **88.7%** | — | ✅ 当前最优 |
| VAD-segment 粒度 | ≤8s 整段, >8s 4s 分块 | 78.1% | -10.6pp | ❌ 数据点不足 |
| 3s 重叠分块 | 0.75s overlap | 81.2% | -7.5pp | ❌ 相似度膨胀 |
| 5s 分块 | CHUNK=500 | 79.0% | -9.7pp | ❌ 跨说话人污染 |
| α=0.3 | 降低时间权重 | 81.8% | -6.9pp | ❌ 物理重叠失区分 |
| α=0.5 | 中间值 | 65.6% | -23.1pp | ❌ 最差区间 |

**结论**: 当前 3s 非重叠 + α=0.65 是经过协同调优的局部最优。改变分块策略需同步重新搜索 (CHUNK, α, τ, p-prune, smoothing) 多维参数空间。
