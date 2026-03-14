// vad_engine.h — FSMN-VAD Neural VAD Engine (Phase 1)
//
// 替代基于 RMS 能量阈值的 VAD, 使用 FSMN 神经网络判决语音/静音。
// 模型: ~0.4M 参数, CPU 推理即可 (无需 GPU), 延迟 <10ms/chunk。
//
// 架构: 80-dim Fbank → LFR(×5) → CMVN → FSMN(4层) → 248-class softmax
// 判决: 滑动窗口检测 speech/silence 转换, 输出语音段 [{start_ms, end_ms, data}]

#pragma once

#include "vad_config.h"
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <fstream>

namespace qwen_thor {
namespace asr {

// VAD 检测到的语音段
struct VadSegment {
    int start_ms = 0;
    int end_ms   = 0;
    std::vector<float> pcm;  // 对应的 PCM 数据
};

// ============================================================================
// VadEngine — FSMN 神经网络 VAD
// ============================================================================
class VadEngine {
public:
    VadEngine() = default;
    ~VadEngine() = default;

    // 加载模型 + CMVN
    bool load(const std::string& model_dir) {
        config_.model_path = model_dir + "/fsmn_vad.safetensors";
        config_.cmvn_path  = model_dir + "/cmvn.safetensors";
        if (!load_weights()) return false;
        if (!load_cmvn()) return false;
        loaded_ = true;
        reset();
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // 重置状态 (新一段音频开始时调用)
    void reset() {
        state_ = State::START_POINT_NOT_DETECTED;
        frame_probs_.clear();
        pcm_cache_.clear();
        frame_idx_ = 0;
        data_buf_start_frame_ = 0;
        speech_start_ms_ = -1;
        speech_end_ms_ = -1;
        // LFR cache
        lfr_cache_.clear();
        fsmn_cache_.clear();
        fsmn_cache_.resize(config_.fsmn_layers);
    }

    // 流式检测: 每次送入一段 PCM (16kHz mono float), 返回完成的语音段
    std::vector<VadSegment> detect(const float* pcm, int num_samples, bool is_final = false) {
        if (!loaded_) return {};

        // 累积 PCM
        pcm_cache_.insert(pcm_cache_.end(), pcm, pcm + num_samples);

        std::vector<VadSegment> segments;

        // 提取帧级 Fbank → LFR 拼接 → CMVN → FSMN → 获取每帧概率
        while (can_extract_frame()) {
            // 提取一帧 Fbank (80-dim)
            std::vector<float> fbank = extract_fbank_frame();
            lfr_cache_.push_back(fbank);

            // LFR: 攒够 lfr_m 帧后拼接
            if ((int)lfr_cache_.size() >= config_.lfr_m) {
                std::vector<float> lfr_feat(config_.input_dim);
                for (int i = 0; i < config_.lfr_m; ++i) {
                    std::copy(lfr_cache_[i].begin(), lfr_cache_[i].end(),
                              lfr_feat.begin() + i * config_.n_mels);
                }
                lfr_cache_.erase(lfr_cache_.begin());

                // CMVN 归一化
                apply_cmvn(lfr_feat);

                // FSMN forward → frame probability
                float speech_prob = fsmn_forward(lfr_feat);
                frame_probs_.push_back(speech_prob);

                // 状态机判决
                auto seg = update_state_machine();
                if (seg.has_value()) {
                    segments.push_back(seg.value());
                }
                frame_idx_++;
            }
        }

        // 最终帧: 强制结束当前语音段
        if (is_final && state_ == State::IN_SPEECH_SEGMENT) {
            int end_ms = frame_idx_ * config_.frame_shift_ms;
            segments.push_back(make_segment(speech_start_ms_, end_ms));
            state_ = State::END_POINT_DETECTED;
        }

        return segments;
    }

    // 单次整段检测 (非流式)
    std::vector<VadSegment> detect_all(const float* pcm, int num_samples) {
        reset();
        auto segs = detect(pcm, num_samples, true);
        return segs;
    }

    const VadConfig& config() const { return config_; }

private:
    VadConfig config_;
    bool loaded_ = false;

    // 判决状态机
    enum class State {
        START_POINT_NOT_DETECTED,
        IN_SPEECH_SEGMENT,
        END_POINT_DETECTED
    };
    State state_ = State::START_POINT_NOT_DETECTED;

    std::vector<float> frame_probs_;  // 每帧语音概率
    std::vector<float> pcm_cache_;    // PCM 缓存
    int frame_idx_ = 0;
    int data_buf_start_frame_ = 0;
    int speech_start_ms_ = -1;
    int speech_end_ms_ = -1;

    // LFR 缓存
    std::vector<std::vector<float>> lfr_cache_;

    // FSMN memory cache (per layer)
    struct FsmnCache {
        std::vector<std::vector<float>> memory;  // [lorder, proj_dim]
    };
    std::vector<FsmnCache> fsmn_cache_;

    // ========== 模型权重 (CPU) ==========
    // in_linear1: [140, 400] + [140]
    std::vector<float> in_linear1_w_, in_linear1_b_;
    // in_linear2: [250, 140] + [250]
    std::vector<float> in_linear2_w_, in_linear2_b_;
    // FSMN layers × 4
    struct FsmnLayerWeights {
        std::vector<float> linear_w;     // [128, 250]
        std::vector<float> fsmn_w;       // [128, 1, lorder, 1] → [lorder, 128]
        std::vector<float> affine_w;     // [250, 128]
        std::vector<float> affine_b;     // [250]
    };
    std::vector<FsmnLayerWeights> fsmn_weights_;
    // out_linear1: [140, 250] + [140]
    std::vector<float> out_linear1_w_, out_linear1_b_;
    // out_linear2: [248, 140] + [248]
    std::vector<float> out_linear2_w_, out_linear2_b_;
    // CMVN
    std::vector<float> cmvn_mean_, cmvn_invstd_;

    // ========== Fbank ==========
    std::vector<float> hann_window_;
    std::vector<float> mel_filterbank_;  // [n_mels, n_fft/2+1]
    int fbank_frame_pos_ = 0; // 已提取的帧位置 (in samples)

    bool can_extract_frame() const {
        int window_size = config_.window_samples();
        return fbank_frame_pos_ + window_size <= (int)pcm_cache_.size();
    }

    std::vector<float> extract_fbank_frame() {
        int n_fft = config_.window_samples();  // 400
        int hop = config_.frame_samples();     // 160

        // 如果还没初始化窗函数和 mel 滤波器
        if (hann_window_.empty()) {
            init_fbank(n_fft);
        }

        // 取一帧, 加窗
        std::vector<float> frame(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            frame[i] = pcm_cache_[fbank_frame_pos_ + i] * hann_window_[i];
        }
        fbank_frame_pos_ += hop;

        // 简化 DFT (实数 → 频域幅度谱)
        int n_freq = n_fft / 2 + 1;
        std::vector<float> power_spec(n_freq);
        for (int k = 0; k < n_freq; ++k) {
            float re = 0, im = 0;
            for (int n = 0; n < n_fft; ++n) {
                float angle = -2.0f * M_PI * k * n / n_fft;
                re += frame[n] * cosf(angle);
                im += frame[n] * sinf(angle);
            }
            power_spec[k] = re * re + im * im;
        }

        // Mel 滤波
        std::vector<float> fbank(config_.n_mels);
        for (int m = 0; m < config_.n_mels; ++m) {
            float sum = 0;
            for (int k = 0; k < n_freq; ++k) {
                sum += mel_filterbank_[m * n_freq + k] * power_spec[k];
            }
            fbank[m] = logf(std::max(sum, 1e-10f));
        }
        return fbank;
    }

    void init_fbank(int n_fft) {
        // Hamming window (config says hamming, not hann)
        hann_window_.resize(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            hann_window_[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (n_fft - 1));
        }

        // Mel filterbank
        int n_freq = n_fft / 2 + 1;
        float fmin = 0, fmax = (float)config_.sample_rate / 2;
        auto hz_to_mel = [](float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); };

        float mel_min = hz_to_mel(fmin), mel_max = hz_to_mel(fmax);
        std::vector<float> mel_points(config_.n_mels + 2);
        for (int i = 0; i < config_.n_mels + 2; ++i) {
            mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (config_.n_mels + 1));
        }

        mel_filterbank_.resize(config_.n_mels * n_freq, 0);
        float freq_step = (float)config_.sample_rate / n_fft;
        for (int m = 0; m < config_.n_mels; ++m) {
            for (int k = 0; k < n_freq; ++k) {
                float freq = k * freq_step;
                if (freq >= mel_points[m] && freq <= mel_points[m + 1]) {
                    mel_filterbank_[m * n_freq + k] =
                        (freq - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
                } else if (freq > mel_points[m + 1] && freq <= mel_points[m + 2]) {
                    mel_filterbank_[m * n_freq + k] =
                        (mel_points[m + 2] - freq) / (mel_points[m + 2] - mel_points[m + 1]);
                }
            }
        }
    }

    // ========== CMVN 归一化 ==========
    void apply_cmvn(std::vector<float>& feat) {
        for (int i = 0; i < config_.input_dim; ++i) {
            feat[i] = (feat[i] - cmvn_mean_[i]) * cmvn_invstd_[i];
        }
    }

    // ========== FSMN Forward ==========
    // 线性层: y = Wx + b
    static void linear_forward(const float* W, const float* b,
                               const float* x, float* y,
                               int out_dim, int in_dim) {
        for (int i = 0; i < out_dim; ++i) {
            float sum = b ? b[i] : 0;
            for (int j = 0; j < in_dim; ++j) {
                sum += W[i * in_dim + j] * x[j];
            }
            y[i] = sum;
        }
    }

    // ReLU
    static void relu(float* x, int n) {
        for (int i = 0; i < n; ++i) {
            x[i] = std::max(0.0f, x[i]);
        }
    }

    float fsmn_forward(const std::vector<float>& feat) {
        // in_linear1: [400] → [140] + ReLU
        std::vector<float> h1(config_.input_affine_dim);
        linear_forward(in_linear1_w_.data(), in_linear1_b_.data(),
                      feat.data(), h1.data(),
                      config_.input_affine_dim, config_.input_dim);
        relu(h1.data(), config_.input_affine_dim);

        // in_linear2: [140] → [250] + ReLU
        std::vector<float> h(config_.linear_dim);
        linear_forward(in_linear2_w_.data(), in_linear2_b_.data(),
                      h1.data(), h.data(),
                      config_.linear_dim, config_.input_affine_dim);
        relu(h.data(), config_.linear_dim);

        // 4 FSMN blocks
        for (int l = 0; l < config_.fsmn_layers; ++l) {
            auto& fw = fsmn_weights_[l];
            auto& cache = fsmn_cache_[l];

            // linear: [250] → [128] (no bias, no activation)
            std::vector<float> p(config_.proj_dim);
            linear_forward(fw.linear_w.data(), nullptr,
                          h.data(), p.data(),
                          config_.proj_dim, config_.linear_dim);

            // FSMN memory block: causal convolution with left context
            // h_t = p_t + Σ(a_i * p_{t-i}), i = 0..lorder-1
            // cache stores last lorder frames of p
            cache.memory.push_back(p);
            if ((int)cache.memory.size() > config_.lorder) {
                cache.memory.erase(cache.memory.begin());
            }

            std::vector<float> mem_out(config_.proj_dim, 0);
            int cache_size = (int)cache.memory.size();
            for (int i = 0; i < cache_size; ++i) {
                int tap = cache_size - 1 - i; // tap 0 = current, tap 1 = previous, etc.
                if (tap < config_.lorder) {
                    for (int d = 0; d < config_.proj_dim; ++d) {
                        // fsmn_w shape: [proj_dim, 1, lorder, 1]
                        // reshaped as [proj_dim][lorder], stored row-major
                        mem_out[d] += fw.fsmn_w[d * config_.lorder + tap] * cache.memory[i][d];
                    }
                }
            }

            // p += memory output (skip connection)
            for (int d = 0; d < config_.proj_dim; ++d) {
                p[d] += mem_out[d];
            }

            // affine: [128] → [250] + ReLU + residual
            std::vector<float> h_new(config_.linear_dim);
            linear_forward(fw.affine_w.data(), fw.affine_b.data(),
                          p.data(), h_new.data(),
                          config_.linear_dim, config_.proj_dim);
            relu(h_new.data(), config_.linear_dim);

            // residual connection
            for (int d = 0; d < config_.linear_dim; ++d) {
                h[d] += h_new[d];
            }
        }

        // out_linear1: [250] → [140] + ReLU
        std::vector<float> o1(config_.output_affine_dim);
        linear_forward(out_linear1_w_.data(), out_linear1_b_.data(),
                      h.data(), o1.data(),
                      config_.output_affine_dim, config_.linear_dim);
        relu(o1.data(), config_.output_affine_dim);

        // out_linear2: [140] → [248]
        std::vector<float> logits(config_.output_dim);
        linear_forward(out_linear2_w_.data(), out_linear2_b_.data(),
                      o1.data(), logits.data(),
                      config_.output_dim, config_.output_affine_dim);

        // Softmax → speech probability
        // speech_prob = 1 - sil_prob, where sil is class 0
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0;
        for (int i = 0; i < config_.output_dim; ++i) {
            logits[i] = expf(logits[i] - max_logit);
            sum_exp += logits[i];
        }

        float sil_prob = logits[config_.sil_pdf_ids[0]] / sum_exp;
        return 1.0f - sil_prob;
    }

    // ========== 状态机 ==========
    struct OptSegment {
        bool has_value_ = false;
        VadSegment val_;
        bool has_value() const { return has_value_; }
        VadSegment& value() { return val_; }
    };

    OptSegment update_state_machine() {
        OptSegment result;
        int cur_ms = frame_idx_ * config_.frame_shift_ms;
        float cur_prob = frame_probs_.back();
        bool is_speech = cur_prob >= config_.speech_noise_thres;

        int window_frames = config_.window_size_ms / config_.frame_shift_ms;

        switch (state_) {
        case State::START_POINT_NOT_DETECTED: {
            if (is_speech) {
                // 检查窗口内 speech 帧数是否超过阈值
                int speech_count = count_speech_frames(
                    std::max(0, frame_idx_ - window_frames), frame_idx_ + 1);
                int thres_frames = config_.sil_to_speech_time_thres / config_.frame_shift_ms;

                if (speech_count >= thres_frames) {
                    state_ = State::IN_SPEECH_SEGMENT;
                    speech_start_ms_ = std::max(0,
                        cur_ms - config_.lookback_time_start_point);
                }
            }
            break;
        }
        case State::IN_SPEECH_SEGMENT: {
            if (!is_speech) {
                // 检查连续静音时长
                int silence_ms = count_trailing_silence_ms();
                if (silence_ms >= config_.max_end_silence_time) {
                    speech_end_ms_ = cur_ms - silence_ms + config_.lookahead_time_end_point;
                    result.has_value_ = true;
                    result.val_ = make_segment(speech_start_ms_, speech_end_ms_);
                    state_ = State::START_POINT_NOT_DETECTED;
                    speech_start_ms_ = -1;
                }
            }
            // 最长时间限制
            if (state_ == State::IN_SPEECH_SEGMENT &&
                cur_ms - speech_start_ms_ >= config_.max_single_segment_time) {
                result.has_value_ = true;
                result.val_ = make_segment(speech_start_ms_, cur_ms);
                state_ = State::START_POINT_NOT_DETECTED;
                speech_start_ms_ = -1;
            }
            break;
        }
        case State::END_POINT_DETECTED:
            // 等待 reset
            break;
        }
        return result;
    }

    int count_speech_frames(int from, int to) const {
        int count = 0;
        for (int i = from; i < to && i < (int)frame_probs_.size(); ++i) {
            if (i >= 0 && frame_probs_[i] >= config_.speech_noise_thres) count++;
        }
        return count;
    }

    int count_trailing_silence_ms() const {
        int count = 0;
        for (int i = (int)frame_probs_.size() - 1; i >= 0; --i) {
            if (frame_probs_[i] < config_.speech_noise_thres) {
                count++;
            } else {
                break;
            }
        }
        return count * config_.frame_shift_ms;
    }

    VadSegment make_segment(int start_ms, int end_ms) const {
        VadSegment seg;
        seg.start_ms = std::max(0, start_ms);
        seg.end_ms = std::min(end_ms, (int)(pcm_cache_.size() * 1000 / config_.sample_rate));

        int start_sample = seg.start_ms * config_.sample_rate / 1000;
        int end_sample = std::min(seg.end_ms * config_.sample_rate / 1000,
                                  (int)pcm_cache_.size());
        if (end_sample > start_sample) {
            seg.pcm.assign(pcm_cache_.begin() + start_sample,
                          pcm_cache_.begin() + end_sample);
        }
        return seg;
    }

    // ========== 权重加载 ==========
    bool load_weights() {
        // 使用简易 safetensors 解析 (模型很小, 直接 mmap+parse)
        auto tensors = load_safetensors_simple(config_.model_path);
        if (tensors.empty()) return false;

        auto get = [&](const std::string& name) -> std::vector<float>* {
            auto it = tensors.find(name);
            return it != tensors.end() ? &it->second : nullptr;
        };

        // 加载所有权重
        auto* w = get("encoder.in_linear1.linear.weight");
        auto* b = get("encoder.in_linear1.linear.bias");
        if (!w || !b) return false;
        in_linear1_w_ = *w;
        in_linear1_b_ = *b;

        w = get("encoder.in_linear2.linear.weight");
        b = get("encoder.in_linear2.linear.bias");
        if (!w || !b) return false;
        in_linear2_w_ = *w;
        in_linear2_b_ = *b;

        fsmn_weights_.resize(config_.fsmn_layers);
        for (int l = 0; l < config_.fsmn_layers; ++l) {
            std::string prefix = "encoder.fsmn." + std::to_string(l) + ".";
            w = get(prefix + "linear.linear.weight");
            if (!w) return false;
            fsmn_weights_[l].linear_w = *w;

            w = get(prefix + "fsmn_block.conv_left.weight");
            if (!w) return false;
            // Reshape [128, 1, 20, 1] → [128, 20] (already contiguous)
            fsmn_weights_[l].fsmn_w = *w;

            w = get(prefix + "affine.linear.weight");
            b = get(prefix + "affine.linear.bias");
            if (!w || !b) return false;
            fsmn_weights_[l].affine_w = *w;
            fsmn_weights_[l].affine_b = *b;
        }

        w = get("encoder.out_linear1.linear.weight");
        b = get("encoder.out_linear1.linear.bias");
        if (!w || !b) return false;
        out_linear1_w_ = *w;
        out_linear1_b_ = *b;

        w = get("encoder.out_linear2.linear.weight");
        b = get("encoder.out_linear2.linear.bias");
        if (!w || !b) return false;
        out_linear2_w_ = *w;
        out_linear2_b_ = *b;

        return true;
    }

    bool load_cmvn() {
        auto tensors = load_safetensors_simple(config_.cmvn_path);
        if (tensors.empty()) return false;
        auto it_m = tensors.find("cmvn_mean");
        auto it_s = tensors.find("cmvn_invstd");
        if (it_m == tensors.end() || it_s == tensors.end()) return false;
        cmvn_mean_ = it_m->second;
        cmvn_invstd_ = it_s->second;
        return (int)cmvn_mean_.size() == config_.input_dim;
    }

    // ========== 简易 safetensors 解析 (CPU, F32 only) ==========
    using TensorMap = std::unordered_map<std::string, std::vector<float>>;

    static TensorMap load_safetensors_simple(const std::string& path) {
        TensorMap result;
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return result;

        // Read header size (first 8 bytes, little-endian uint64)
        uint64_t header_size = 0;
        ifs.read(reinterpret_cast<char*>(&header_size), 8);
        if (header_size > 100000) return result; // sanity check

        // Read header JSON
        std::string header(header_size, '\0');
        ifs.read(&header[0], header_size);

        size_t data_offset_base = 8 + header_size;

        // Parse header: find tensor entries
        // Format: "tensor_name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}
        size_t pos = 0;
        while (pos < header.size()) {
            // Find next key
            size_t key_start = header.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = header.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = header.substr(key_start + 1, key_end - key_start - 1);
            pos = key_end + 1;

            if (key == "__metadata__") {
                // Skip metadata
                size_t brace = header.find('{', pos);
                if (brace != std::string::npos) {
                    int depth = 1;
                    pos = brace + 1;
                    while (pos < header.size() && depth > 0) {
                        if (header[pos] == '{') depth++;
                        else if (header[pos] == '}') depth--;
                        pos++;
                    }
                }
                continue;
            }

            // Find data_offsets
            size_t offsets_pos = header.find("data_offsets", pos);
            if (offsets_pos == std::string::npos) break;
            size_t bracket = header.find('[', offsets_pos);
            if (bracket == std::string::npos) break;
            size_t comma = header.find(',', bracket);
            if (comma == std::string::npos) break;
            size_t end_bracket = header.find(']', comma);
            if (end_bracket == std::string::npos) break;

            uint64_t start = std::stoull(header.substr(bracket + 1, comma - bracket - 1));
            uint64_t end = std::stoull(header.substr(comma + 1, end_bracket - comma - 1));

            // Read tensor data
            size_t num_bytes = end - start;
            size_t num_floats = num_bytes / sizeof(float);

            std::vector<float> data(num_floats);
            ifs.seekg(data_offset_base + start);
            ifs.read(reinterpret_cast<char*>(data.data()), num_bytes);

            result[key] = std::move(data);
            pos = end_bracket + 1;
        }

        return result;
    }
};

// ============================================================================
// EnergyVad — 能量 VAD (降级方案, 保留原有逻辑)
// ============================================================================
class EnergyVad {
public:
    struct Config {
        float energy_threshold  = 0.01f;
        int   silence_ms        = 800;
        int   min_speech_ms     = 500;
        int   max_duration_s    = 30;
        float min_speech_energy = 0.008f;
    };

    EnergyVad() = default;
    explicit EnergyVad(const Config& cfg) : config_(cfg) {}

    void reset() {
        speech_detected_ = false;
        silence_samples_ = 0;
        total_energy_sum_ = 0;
        total_speech_samples_ = 0;
    }

    struct Result {
        bool speech_active = false;
        bool vad_triggered = false; // silence after speech → trigger ASR
    };

    Result process(const float* samples, int num_samples, int sample_rate = 16000) {
        Result r;

        // RMS energy
        double sum_sq = 0;
        for (int i = 0; i < num_samples; ++i) {
            sum_sq += (double)samples[i] * samples[i];
        }
        float rms = sqrtf((float)(sum_sq / std::max(1, num_samples)));

        if (rms > config_.energy_threshold) {
            speech_detected_ = true;
            silence_samples_ = 0;
            total_energy_sum_ += sum_sq;
            total_speech_samples_ += num_samples;
        } else {
            silence_samples_ += num_samples;
        }

        float silence_ms = (float)silence_samples_ * 1000.0f / sample_rate;
        float total_ms = (float)total_speech_samples_ * 1000.0f / sample_rate;

        r.speech_active = speech_detected_;
        r.vad_triggered = speech_detected_ &&
                          silence_ms >= config_.silence_ms &&
                          total_ms >= config_.min_speech_ms;

        if (r.vad_triggered) {
            float avg_energy = total_speech_samples_ > 0
                ? sqrtf((float)(total_energy_sum_ / total_speech_samples_)) : 0;
            if (avg_energy < config_.min_speech_energy) {
                r.vad_triggered = false;
            }
        }

        return r;
    }

private:
    Config config_;
    bool speech_detected_ = false;
    int silence_samples_ = 0;
    double total_energy_sum_ = 0;
    int total_speech_samples_ = 0;
};

} // namespace asr
} // namespace qwen_thor
