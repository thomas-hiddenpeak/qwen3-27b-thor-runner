// tts_speaker_encoder.cpp — ECAPA-TDNN Speaker Encoder (CPU inference)

#include "tts_speaker_encoder.h"
#include "../asr/audio_utils.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdio>

namespace qwen_thor {
namespace tts {

// ============================================================================
// BF16 → FP32 conversion
// ============================================================================
static float bf16_to_f32(uint16_t val) {
    uint32_t bits = (uint32_t)val << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static std::vector<float> bf16_to_f32_vec(const uint16_t* data, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(data[i]);
    return out;
}

// ============================================================================
// Conv1d with reflect padding (same output length)
// ============================================================================
void SpeakerEncoder::conv1d_forward(
    const std::vector<float>& input, int in_ch, int T,
    const Conv1dWeights& w,
    std::vector<float>& output)
{
    int out_ch = w.out_channels;
    int k = w.kernel_size;
    int d = w.dilation;
    int eff_k = d * (k - 1) + 1;           // effective kernel size
    int pad = (eff_k - 1) / 2;             // "same" padding (each side)

    // Build reflect-padded input: [in_ch, T + 2*pad]
    int T_padded = T + 2 * pad;
    std::vector<float> padded(in_ch * T_padded, 0.0f);
    for (int c = 0; c < in_ch; c++) {
        float* dst = padded.data() + c * T_padded;
        const float* src = input.data() + c * T;
        // Copy center
        memcpy(dst + pad, src, T * sizeof(float));
        // Reflect left padding
        for (int i = 0; i < pad; i++) {
            int idx = pad - i;  // reflect index
            if (idx >= T) idx = T - 1;
            dst[i] = src[idx];
        }
        // Reflect right padding
        for (int i = 0; i < pad; i++) {
            int idx = T - 2 - i;  // reflect index
            if (idx < 0) idx = 0;
            dst[pad + T + i] = src[idx];
        }
    }

    // Conv1d: output[oc, t] = sum_ic sum_ki weight[oc, ic, ki] * padded[ic, t + ki*d]
    output.resize(out_ch * T);
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < T; t++) {
            float sum = 0;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int ki = 0; ki < k; ki++) {
                    int idx = t + ki * d;
                    sum += w.weight[oc * in_ch * k + ic * k + ki] *
                           padded[ic * T_padded + idx];
                }
            }
            if (!w.bias.empty()) sum += w.bias[oc];
            output[oc * T + t] = sum;
        }
    }
}

// ============================================================================
// Activation functions
// ============================================================================
void SpeakerEncoder::relu_inplace(std::vector<float>& x) {
    for (auto& v : x) v = std::max(v, 0.0f);
}

void SpeakerEncoder::sigmoid_inplace(std::vector<float>& x) {
    for (auto& v : x) v = 1.0f / (1.0f + std::exp(-v));
}

// ============================================================================
// TTS-style Mel Spectrogram
// ============================================================================
void SpeakerEncoder::compute_mel(
    const float* audio, int num_samples,
    std::vector<float>& mel_out, int& num_frames)
{
    const int n_fft = 1024;
    const int hop = 256;
    const int win = 1024;
    const int n_mels = config_.mel_dim;
    const int sr = config_.sample_rate;
    const int n_freqs = n_fft / 2 + 1;

    // Lazy init filterbank and window
    if (mel_filterbank_.empty()) {
        mel_filterbank_ = audio::build_mel_filterbank(n_mels, n_fft, sr);
        hann_window_ = audio::build_hann_window(win);
    }

    // Reflect padding: pad_size = (n_fft - hop) / 2 = 384
    int pad_size = (n_fft - hop) / 2;
    int padded_len = num_samples + 2 * pad_size;
    std::vector<float> padded(padded_len);

    // Copy center
    memcpy(padded.data() + pad_size, audio, num_samples * sizeof(float));
    // Reflect left
    for (int i = 0; i < pad_size; i++) {
        int idx = pad_size - i;
        if (idx >= num_samples) idx = num_samples - 1;
        padded[i] = audio[idx];
    }
    // Reflect right
    for (int i = 0; i < pad_size; i++) {
        int idx = num_samples - 2 - i;
        if (idx < 0) idx = 0;
        padded[pad_size + num_samples + i] = audio[idx];
    }

    num_frames = (padded_len - n_fft) / hop + 1;

    // STFT → magnitude → mel → log compression
    std::vector<float> frame(n_fft);
    std::vector<float> fft_real(n_freqs), fft_imag(n_freqs);
    mel_out.resize(num_frames * n_mels);

    for (int t = 0; t < num_frames; t++) {
        // Apply window
        for (int i = 0; i < n_fft; i++) {
            frame[i] = padded[t * hop + i] * hann_window_[i];
        }

        // DFT
        for (int k = 0; k < n_freqs; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < n_fft; n++) {
                float angle = 2.0f * (float)M_PI * k * n / n_fft;
                re += frame[n] * std::cos(angle);
                im -= frame[n] * std::sin(angle);
            }
            fft_real[k] = re;
            fft_imag[k] = im;
        }

        // Magnitude spectrum (not power): sqrt(re^2 + im^2 + 1e-9)
        for (int k = 0; k < n_freqs; k++) {
            float mag = std::sqrt(fft_real[k] * fft_real[k] +
                                   fft_imag[k] * fft_imag[k] + 1e-9f);
            fft_real[k] = mag;  // reuse buffer
        }

        // Mel filterbank matmul + log compression
        for (int m = 0; m < n_mels; m++) {
            float sum = 0;
            for (int k = 0; k < n_freqs; k++) {
                sum += mel_filterbank_[m * n_freqs + k] * fft_real[k];
            }
            // dynamic_range_compression: log(clamp(x, 1e-5))
            mel_out[t * n_mels + m] = std::log(std::max(sum, 1e-5f));
        }
    }
}

// ============================================================================
// Load Weights
// ============================================================================
bool SpeakerEncoder::load_weights(
    const std::vector<std::pair<std::string, std::pair<const uint16_t*, size_t>>>& weights)
{
    auto find_weight = [&](const std::string& name) -> const std::pair<const uint16_t*, size_t>* {
        for (const auto& [n, v] : weights) {
            if (n == name) return &v;
        }
        return nullptr;
    };

    auto load_conv = [&](const std::string& prefix, int in_ch, int out_ch,
                         int kernel_size, int dilation) -> Conv1dWeights {
        Conv1dWeights w;
        w.in_channels = in_ch;
        w.out_channels = out_ch;
        w.kernel_size = kernel_size;
        w.dilation = dilation;

        auto* wt = find_weight(prefix + ".weight");
        if (wt) w.weight = bf16_to_f32_vec(wt->first, wt->second);

        auto* bt = find_weight(prefix + ".bias");
        if (bt) w.bias = bf16_to_f32_vec(bt->first, bt->second);

        return w;
    };

    const auto& c = config_;
    int ch0 = c.enc_channels[0];  // 512
    int scale = c.enc_res2net_scale;  // 8
    int sub_ch = ch0 / scale;  // 64

    // blocks[0]: initial TDNN
    block0_conv_ = load_conv("speaker_encoder.blocks.0.conv",
                             c.mel_dim, ch0, c.enc_kernel_sizes[0], c.enc_dilations[0]);

    // blocks[1-3]: SE-Res2Net
    se_res2net_blocks_.resize(3);
    for (int b = 0; b < 3; b++) {
        auto& blk = se_res2net_blocks_[b];
        std::string prefix = "speaker_encoder.blocks." + std::to_string(b + 1);
        int k = c.enc_kernel_sizes[b + 1];
        int d = c.enc_dilations[b + 1];

        blk.tdnn1 = load_conv(prefix + ".tdnn1.conv", ch0, ch0, 1, 1);
        blk.tdnn2 = load_conv(prefix + ".tdnn2.conv", ch0, ch0, 1, 1);

        // Res2Net sub-blocks (scale-1 = 7)
        blk.res2net_blocks.resize(scale - 1);
        for (int s = 0; s < scale - 1; s++) {
            blk.res2net_blocks[s] = load_conv(
                prefix + ".res2net_block.blocks." + std::to_string(s) + ".conv",
                sub_ch, sub_ch, k, d);
        }

        // SE block
        blk.se_conv1 = load_conv(prefix + ".se_block.conv1", ch0, c.enc_se_channels, 1, 1);
        blk.se_conv2 = load_conv(prefix + ".se_block.conv2", c.enc_se_channels, ch0, 1, 1);
    }

    // MFA
    int mfa_ch = c.enc_channels.back();  // 1536
    mfa_conv_ = load_conv("speaker_encoder.mfa.conv", mfa_ch, mfa_ch, 1, 1);

    // ASP
    asp_tdnn_conv_ = load_conv("speaker_encoder.asp.tdnn.conv", mfa_ch * 3, c.enc_attention_channels, 1, 1);
    asp_conv_ = load_conv("speaker_encoder.asp.conv", c.enc_attention_channels, mfa_ch, 1, 1);

    // FC
    fc_conv_ = load_conv("speaker_encoder.fc", mfa_ch * 2, c.enc_dim, 1, 1);

    // Validate critical weights loaded
    if (block0_conv_.weight.empty() || fc_conv_.weight.empty()) {
        fprintf(stderr, "[SpeakerEncoder] ERROR: missing critical weights\n");
        return false;
    }

    loaded_ = true;
    fprintf(stderr, "[SpeakerEncoder] Loaded: mel_dim=%d, enc_dim=%d, channels=[%d,%d,%d,%d,%d]\n",
            c.mel_dim, c.enc_dim,
            c.enc_channels[0], c.enc_channels[1], c.enc_channels[2],
            c.enc_channels[3], c.enc_channels[4]);
    return true;
}

// ============================================================================
// Extract Speaker Embedding
// ============================================================================
std::vector<float> SpeakerEncoder::extract(
    const float* audio, int num_samples, int sample_rate)
{
    if (!loaded_) {
        fprintf(stderr, "[SpeakerEncoder] ERROR: not loaded\n");
        return {};
    }

    // Resample to encoder sample rate if needed
    std::vector<float> resampled;
    const float* audio_24k = audio;
    int n_24k = num_samples;
    if (sample_rate != config_.sample_rate) {
        std::vector<float> in_vec(audio, audio + num_samples);
        audio::resample(in_vec, sample_rate, resampled, config_.sample_rate);
        audio_24k = resampled.data();
        n_24k = (int)resampled.size();
    }

    // 1. Compute mel spectrogram: [T, mel_dim]
    std::vector<float> mel;
    int T;
    compute_mel(audio_24k, n_24k, mel, T);

    // Transpose to [mel_dim, T] for Conv1d (channels-first)
    int mel_dim = config_.mel_dim;
    std::vector<float> hidden(mel_dim * T);
    for (int m = 0; m < mel_dim; m++) {
        for (int t = 0; t < T; t++) {
            hidden[m * T + t] = mel[t * mel_dim + m];
        }
    }

    int ch0 = config_.enc_channels[0];  // 512

    // 2. blocks[0]: TDNN (mel→512)
    std::vector<float> out0;
    conv1d_forward(hidden, mel_dim, T, block0_conv_, out0);
    relu_inplace(out0);

    // Collect hidden states for MFA (blocks 1-3 outputs)
    std::vector<std::vector<float>> hidden_list;
    hidden_list.push_back(out0);  // blocks[0] output

    std::vector<float> current = out0;

    // 3. blocks[1-3]: SE-Res2Net
    int scale = config_.enc_res2net_scale;
    int sub_ch = ch0 / scale;

    for (int b = 0; b < 3; b++) {
        const auto& blk = se_res2net_blocks_[b];
        std::vector<float> residual = current;

        // TDNN1
        std::vector<float> h1;
        conv1d_forward(current, ch0, T, blk.tdnn1, h1);
        relu_inplace(h1);

        // Res2Net: split into scale chunks → sequential processing
        std::vector<std::vector<float>> chunks(scale);
        for (int s = 0; s < scale; s++) {
            chunks[s].resize(sub_ch * T);
            for (int c = 0; c < sub_ch; c++) {
                memcpy(chunks[s].data() + c * T,
                       h1.data() + (s * sub_ch + c) * T,
                       T * sizeof(float));
            }
        }

        std::vector<std::vector<float>> res2_outputs(scale);
        res2_outputs[0] = chunks[0];  // chunk 0 passes through

        for (int s = 1; s < scale; s++) {
            std::vector<float> inp = chunks[s];
            if (s > 1) {
                // Add previous output
                for (size_t i = 0; i < inp.size(); i++) {
                    inp[i] += res2_outputs[s - 1][i];
                }
            }
            conv1d_forward(inp, sub_ch, T, blk.res2net_blocks[s - 1], res2_outputs[s]);
            relu_inplace(res2_outputs[s]);
        }

        // Concatenate Res2Net outputs → [ch0, T]
        std::vector<float> res2_cat(ch0 * T);
        for (int s = 0; s < scale; s++) {
            for (int c = 0; c < sub_ch; c++) {
                memcpy(res2_cat.data() + (s * sub_ch + c) * T,
                       res2_outputs[s].data() + c * T,
                       T * sizeof(float));
            }
        }

        // TDNN2
        std::vector<float> h2;
        conv1d_forward(res2_cat, ch0, T, blk.tdnn2, h2);
        relu_inplace(h2);

        // SE: squeeze-excitation
        // Mean across T dimension
        std::vector<float> mean_vec(ch0, 0.0f);
        for (int c = 0; c < ch0; c++) {
            float sum = 0;
            for (int t = 0; t < T; t++) sum += h2[c * T + t];
            mean_vec[c] = sum / T;
        }

        // SE conv1: [ch0, 1] → [se_ch, 1]
        std::vector<float> se1(config_.enc_se_channels);
        for (int oc = 0; oc < config_.enc_se_channels; oc++) {
            float sum = 0;
            for (int ic = 0; ic < ch0; ic++) {
                sum += blk.se_conv1.weight[oc * ch0 + ic] * mean_vec[ic];
            }
            if (!blk.se_conv1.bias.empty()) sum += blk.se_conv1.bias[oc];
            se1[oc] = std::max(sum, 0.0f);  // ReLU
        }

        // SE conv2: [se_ch, 1] → [ch0, 1]
        std::vector<float> se2(ch0);
        for (int oc = 0; oc < ch0; oc++) {
            float sum = 0;
            for (int ic = 0; ic < config_.enc_se_channels; ic++) {
                sum += blk.se_conv2.weight[oc * config_.enc_se_channels + ic] * se1[ic];
            }
            if (!blk.se_conv2.bias.empty()) sum += blk.se_conv2.bias[oc];
            se2[oc] = 1.0f / (1.0f + std::exp(-sum));  // Sigmoid
        }

        // Scale h2 by SE weights + residual
        current.resize(ch0 * T);
        for (int c = 0; c < ch0; c++) {
            for (int t = 0; t < T; t++) {
                current[c * T + t] = h2[c * T + t] * se2[c] + residual[c * T + t];
            }
        }

        hidden_list.push_back(current);
    }

    // 4. MFA: concatenate blocks[1:] outputs → [1536, T]
    int mfa_in_ch = ch0 * 3;  // 512 * 3 = 1536
    std::vector<float> mfa_input(mfa_in_ch * T);
    for (int b = 0; b < 3; b++) {
        for (int c = 0; c < ch0; c++) {
            memcpy(mfa_input.data() + (b * ch0 + c) * T,
                   hidden_list[b + 1].data() + c * T,
                   T * sizeof(float));
        }
    }

    std::vector<float> mfa_out;
    conv1d_forward(mfa_input, mfa_in_ch, T, mfa_conv_, mfa_out);
    relu_inplace(mfa_out);

    // 5. ASP: Attentive Statistical Pooling
    int asp_ch = mfa_in_ch;  // 1536

    // Compute mean and std across T
    std::vector<float> mean_vec(asp_ch, 0.0f);
    std::vector<float> std_vec(asp_ch, 0.0f);
    for (int c = 0; c < asp_ch; c++) {
        float sum = 0;
        for (int t = 0; t < T; t++) sum += mfa_out[c * T + t];
        mean_vec[c] = sum / T;

        float var_sum = 0;
        for (int t = 0; t < T; t++) {
            float diff = mfa_out[c * T + t] - mean_vec[c];
            var_sum += diff * diff;
        }
        std_vec[c] = std::sqrt(var_sum / T + 1e-12f);
    }

    // Expand mean and std to [asp_ch, T] and concatenate with x → [3*asp_ch, T]
    std::vector<float> asp_in(3 * asp_ch * T);
    for (int c = 0; c < asp_ch; c++) {
        memcpy(asp_in.data() + c * T, mfa_out.data() + c * T, T * sizeof(float));
        for (int t = 0; t < T; t++) {
            asp_in[(asp_ch + c) * T + t] = mean_vec[c];
            asp_in[(2 * asp_ch + c) * T + t] = std_vec[c];
        }
    }

    // ASP TDNN: [3*1536, T] → [128, T] + ReLU
    std::vector<float> asp_h;
    conv1d_forward(asp_in, 3 * asp_ch, T, asp_tdnn_conv_, asp_h);
    relu_inplace(asp_h);

    // Tanh
    for (auto& v : asp_h) v = std::tanh(v);

    // ASP conv: [128, T] → [1536, T]
    std::vector<float> attn;
    conv1d_forward(asp_h, config_.enc_attention_channels, T, asp_conv_, attn);

    // Softmax over T dimension for each channel
    for (int c = 0; c < asp_ch; c++) {
        float max_val = -1e30f;
        for (int t = 0; t < T; t++) max_val = std::max(max_val, attn[c * T + t]);
        float sum = 0;
        for (int t = 0; t < T; t++) {
            attn[c * T + t] = std::exp(attn[c * T + t] - max_val);
            sum += attn[c * T + t];
        }
        for (int t = 0; t < T; t++) attn[c * T + t] /= sum;
    }

    // Weighted mean and std
    std::vector<float> w_mean(asp_ch, 0.0f);
    for (int c = 0; c < asp_ch; c++) {
        float sum = 0;
        for (int t = 0; t < T; t++) {
            sum += attn[c * T + t] * mfa_out[c * T + t];
        }
        w_mean[c] = sum;
    }

    std::vector<float> w_std(asp_ch, 0.0f);
    for (int c = 0; c < asp_ch; c++) {
        float sum = 0;
        for (int t = 0; t < T; t++) {
            float diff = mfa_out[c * T + t] - w_mean[c];
            sum += attn[c * T + t] * diff * diff;
        }
        w_std[c] = std::sqrt(std::max(sum, 1e-12f));
    }

    // Concatenate [w_mean, w_std] → [3072]
    // Then unsqueeze to [3072, 1] for FC conv
    std::vector<float> pooled(2 * asp_ch);
    memcpy(pooled.data(), w_mean.data(), asp_ch * sizeof(float));
    memcpy(pooled.data() + asp_ch, w_std.data(), asp_ch * sizeof(float));

    // 6. FC: Conv1d([3072, 1], k=1) → [enc_dim, 1]
    int enc_dim = config_.enc_dim;
    std::vector<float> embedding(enc_dim);
    for (int oc = 0; oc < enc_dim; oc++) {
        float sum = 0;
        for (int ic = 0; ic < 2 * asp_ch; ic++) {
            sum += fc_conv_.weight[oc * 2 * asp_ch + ic] * pooled[ic];
        }
        if (!fc_conv_.bias.empty()) sum += fc_conv_.bias[oc];
        embedding[oc] = sum;
    }

    fprintf(stderr, "[SpeakerEncoder] Extracted %d-dim embedding from %d samples (%d mel frames)\n",
            enc_dim, num_samples, T);
    return embedding;
}

} // namespace tts
} // namespace qwen_thor
