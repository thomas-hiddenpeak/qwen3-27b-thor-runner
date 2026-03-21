// transcription_pipeline.h — 转录管线 (V4/Plain 两种模式)
//
// 从 serve.cpp 提取的核心转录逻辑, 属于插件层。
// serve 层只负责 HTTP 解析 + JSON 格式化, 管线逻辑全部在此。

#pragma once

#include "asr_plugin.h"
#include "speaker_encoder.h"
#include "speaker_encoder_gpu.h"
#include "speaker_encoder_eres2netv2.h"
#include "speaker_encoder_eres2netv2_gpu.h"
#include "speaker_service.h"
#include "punctuation.h"
#include "vad_engine.h"
#include "vad_gpu.h"
#include "mel_gpu.h"
#include "aligner_engine.h"
#include "audio_utils.h"

#include <string>
#include <vector>
#include <mutex>

namespace qwen_thor {
namespace asr {

// ============================================================================
// 转录请求参数
// ============================================================================
struct TranscriptionParams {
    std::string language = "auto";
    std::string response_format = "json";  // json / verbose_json / text
    bool suppress_early_eos = false;
    bool punctuate = false;
    bool identify_speaker = false;
    bool clean_oral = false;
    bool want_word_timestamps = false;
};

// ============================================================================
// 转录结果类型
// ============================================================================
struct TranscriptionWord {
    std::string word;
    int start_ms = -1;
    int end_ms = -1;
    int speaker_id = -1;
    std::string speaker_name;
};

struct TranscriptionSegment {
    int start_ms = 0;
    int end_ms = 0;
    int speaker_id = -1;
    std::string speaker_name;
    std::string text;
    float speaker_similarity = 0;
};

struct TranscriptionResult {
    std::string full_text;
    std::string full_text_with_punc;
    float duration_s = 0;
    std::vector<TranscriptionSegment> segments;
    std::vector<TranscriptionWord> words;
    int error_code = 0;
    std::string error_message;
};

// ============================================================================
// TranscriptionPipeline — 将音频转录为文本 + 说话人分割
// ============================================================================
class TranscriptionPipeline {
public:
    // 依赖注入: 不拥有所有权, 由 serve 层管理生命周期
    struct Dependencies {
        plugins::AsrPlugin* asr_plugin = nullptr;
        GpuSpeakerEncoder* speaker_encoder = nullptr;
        SpeakerManager* speaker_manager = nullptr;
        std::mutex* speaker_mutex = nullptr;
        VadEngine* vad_engine = nullptr;
        GpuVadEngine* gpu_vad_engine = nullptr;
        std::mutex* vad_mutex = nullptr;
        GpuMelExtractor* gpu_mel = nullptr;
        PunctuationRestorer* punctuation_restorer = nullptr;
        AlignerEngine* aligner_engine = nullptr;
        std::mutex* aligner_mutex = nullptr;
        ERes2NetV2SpeakerEncoder* eres2netv2_encoder = nullptr;
        GpuERes2NetV2Encoder* eres2netv2_gpu_encoder = nullptr;
    };

    explicit TranscriptionPipeline(const Dependencies& deps) : deps_(deps) {}

    // 统一入口: 根据参数和可用组件自动选择管线
    TranscriptionResult transcribe(const audio::AudioData& wav,
                                   const TranscriptionParams& params);

    // 显式调用各管线 (高级用途)
    TranscriptionResult run_v4_pipeline(const audio::AudioData& wav,
                                        const TranscriptionParams& params);
    TranscriptionResult run_plain_mode(const audio::AudioData& wav,
                                       const TranscriptionParams& params);

private:
    // V4 子阶段
    std::string run_asr_with_energy_split(const audio::AudioData& wav,
                                          const std::string& language);
    std::vector<AlignedWord> run_forced_alignment(const audio::AudioData& wav,
                                                   const std::string& text);

    Dependencies deps_;
};

} // namespace asr
} // namespace qwen_thor
