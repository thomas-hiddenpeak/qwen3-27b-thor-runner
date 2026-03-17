// speaker_encoder.h — CAM++ Speaker Encoder (Phase 2)
//
// CAM++ 说话人编码器: 从音频特征提取 192-dim speaker embedding。
// CPU 推理 (模型仅 6.9M 参数, ~26 MB, CPU 足够快)。
//
// 架构 (FunASR CAMPPlus):
//   FCM: conv1(s=1) → BN → ReLU → layer1(s=(2,1)) → layer2(s=(2,1))
//        → conv2(s=(2,1)) → BN → ReLU → flatten [320, T]
//   TDNN: Conv1d(320→128, k=5, s=2, p=2) → BN → ReLU
//   3× CAMDenseTDNNBlock (12/24/16 layers, dilation 1/2/2)
//   3× TransitLayer (pre-norm: BN→ReLU→Conv1d)
//   out_nonlinear: BN(512) → ReLU
//   StatsPool → DenseLayer(1024→192, BN affine=False) → L2-normalize → 192-dim

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <fstream>
#include <cstring>
#include <cassert>

namespace qwen_thor {
namespace asr {

// ============================================================================
// CamPlusSpeakerEncoder — CAM++ 512-dim 说话人编码
// ============================================================================
class CamPlusSpeakerEncoder {
public:
    CamPlusSpeakerEncoder() = default;

    bool load(const std::string& safetensors_path) {
        tensors_ = load_safetensors(safetensors_path);
        if (tensors_.empty()) return false;
        loaded_ = true;
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // 从 Mel 特征 (80-dim × T frames) 提取 192-dim embedding
    // mel: [T, 80], row-major
    std::vector<float> extract(const float* mel_80xT, int T) {
        if (!loaded_ || T < 10) return {};

        // 1. Transpose: mel [T, 80] → [80, T]
        std::vector<float> input(80 * T);
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < 80; ++f)
                input[f * T + t] = mel_80xT[t * 80 + f];

        // 2. FCM (Frequency Convolutional Module)
        // Order: conv1 → bn1 → relu → layer1 → layer2 → conv2 → bn2 → relu
        // conv1: [1, 80, T] → [32, 80, T] (k=3, s=1, pad=1)
        auto x = conv2d(input, 1, 80, T, "head.conv1", 32, 3, 1, 1, 1, 1);
        x = batch_norm_2d(x, 32, 80, T, "head.bn1");
        relu_inplace(x);

        // layer1: 2× BasicResBlock (stride in freq only)
        int H = 80;
        // layer1[0]: stride=(2,1) → freq 80→40, time stays T
        x = res_block(x, 32, H, T, "head.layer1.0", 2);
        H = (H + 2*1 - 3) / 2 + 1;  // = 40
        // layer1[1]: stride=1
        x = res_block(x, 32, H, T, "head.layer1.1", 1);

        // layer2: 2× BasicResBlock
        // layer2[0]: stride=(2,1) → freq 40→20
        x = res_block(x, 32, H, T, "head.layer2.0", 2);
        H = (H + 2*1 - 3) / 2 + 1;  // = 20
        // layer2[1]: stride=1
        x = res_block(x, 32, H, T, "head.layer2.1", 1);

        // conv2: stride=(2,1) → freq 20→10, time stays T
        x = conv2d(x, 32, H, T, "head.conv2", 32, 3, 2, 1, 1, 1);
        H = (H + 2*1 - 3) / 2 + 1;  // = 10
        x = batch_norm_2d(x, 32, H, T, "head.bn2");
        relu_inplace(x);

        // Flatten: [32, 10, T] → [320, T]
        int feat_dim = 32 * H;  // = 320

        // 3. TDNN: Conv1d(320→128, k=5, s=2, p=2, d=1)
        auto tdnn_out = conv1d(x, feat_dim, T, "xvector.tdnn.linear", 128, 5, 2, 2, 1);
        int T2 = (T + 2*2 - 1*(5-1) - 1) / 2 + 1;  // = (T-1)/2 + 1
        // TDNN nonlinear: BN(128) + ReLU
        tdnn_out = batch_norm_1d_seq(tdnn_out, 128, T2, "xvector.tdnn.nonlinear.batchnorm");
        relu_inplace(tdnn_out);

        // 4. Three CAM DenseTDNN blocks + transit layers
        // Block 1: 12 layers, dilation=1
        auto [b1_out, b1_dim] = cam_dense_block(tdnn_out, 128, T2, "xvector.block1", 12, 1);
        // Transit1: BN(in) → ReLU → Conv1d(in→out) [pre-norm]
        auto t1 = transit_layer(b1_out, b1_dim, T2, "xvector.transit1", b1_dim / 2);

        // Block 2: 24 layers, dilation=2
        auto [b2_out, b2_dim] = cam_dense_block(t1, b1_dim / 2, T2, "xvector.block2", 24, 2);
        auto t2 = transit_layer(b2_out, b2_dim, T2, "xvector.transit2", b2_dim / 2);

        // Block 3: 16 layers, dilation=2
        auto [b3_out, b3_dim] = cam_dense_block(t2, b2_dim / 2, T2, "xvector.block3", 16, 2);
        auto t3 = transit_layer(b3_out, b3_dim, T2, "xvector.transit3", b3_dim / 2);

        // 5. Out nonlinear: BN(512) + ReLU (applied BEFORE stats pool)
        int embed_channels = b3_dim / 2;  // = 512
        t3 = batch_norm_1d_seq(t3, embed_channels, T2, "xvector.out_nonlinear.batchnorm");
        relu_inplace(t3);

        // 6. StatsPool: mean + std (unbiased) over time → [1024]
        std::vector<float> pooled(embed_channels * 2);
        for (int d = 0; d < embed_channels; ++d) {
            float sum = 0;
            for (int t = 0; t < T2; ++t) sum += t3[d * T2 + t];
            float mean = sum / T2;
            float var_sum = 0;
            for (int t = 0; t < T2; ++t) {
                float diff = t3[d * T2 + t] - mean;
                var_sum += diff * diff;
            }
            float std_val = sqrtf(var_sum / std::max(1, T2 - 1) + 1e-2f);  // unbiased, eps=1e-2
            pooled[d] = mean;
            pooled[embed_channels + d] = std_val;
        }

        // 7. Dense: Conv1d(1024→192, k=1) + BN(192, affine=False)
        const int emb_size = 192;
        auto emb = dense_layer_conv1d(pooled, embed_channels * 2, "xvector.dense.linear", emb_size);
        emb = batch_norm_1d(emb, emb_size, "xvector.dense.nonlinear.batchnorm", false);

        // 8. L2 normalize
        float norm = 0;
        for (float v : emb) norm += v * v;
        norm = sqrtf(norm + 1e-12f);
        for (float& v : emb) v /= norm;

        return emb;
    }

    // 获取 embedding 维度 (192)
    static constexpr int embedding_dim() { return 192; }

    // 余弦相似度
    static float cosine_similarity(const std::vector<float>& a,
                                   const std::vector<float>& b) {
        if (a.size() != b.size() || a.empty()) return 0;
        float dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
    }

private:
    using TensorMap = std::unordered_map<std::string, std::vector<float>>;
    TensorMap tensors_;
    bool loaded_ = false;

    const std::vector<float>& get_tensor(const std::string& name) const {
        static std::vector<float> empty;
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            fprintf(stderr, "[CamPlus] WARNING: tensor '%s' not found\n", name.c_str());
            return empty;
        }
        return it->second;
    }

    // ---- Basic ops ----
    static void relu_inplace(std::vector<float>& x) {
        for (float& v : x) v = std::max(0.0f, v);
    }

    // Conv2d: [Cin, H, W] → [Cout, H', W'] with separate H/W strides
    std::vector<float> conv2d(const std::vector<float>& input,
                               int Cin, int H, int W,
                               const std::string& prefix,
                               int Cout, int k, int stride_h, int stride_w,
                               int pad_h, int pad_w) {
        const auto& weight = get_tensor(prefix + ".weight");
        if (weight.empty()) return std::vector<float>(Cout * ((H+2*pad_h-k)/stride_h+1) * ((W+2*pad_w-k)/stride_w+1), 0);

        int H_out = (H + 2*pad_h - k) / stride_h + 1;
        int W_out = (W + 2*pad_w - k) / stride_w + 1;
        std::vector<float> output(Cout * H_out * W_out, 0);

        for (int co = 0; co < Cout; ++co) {
            for (int ho = 0; ho < H_out; ++ho) {
                for (int wo = 0; wo < W_out; ++wo) {
                    float sum = 0;
                    for (int ci = 0; ci < Cin; ++ci) {
                        for (int kh = 0; kh < k; ++kh) {
                            for (int kw = 0; kw < k; ++kw) {
                                int hi = ho * stride_h - pad_h + kh;
                                int wi = wo * stride_w - pad_w + kw;
                                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                                    float w = weight[co * Cin * k * k + ci * k * k + kh * k + kw];
                                    float x = input[ci * H * W + hi * W + wi];
                                    sum += w * x;
                                }
                            }
                        }
                    }
                    output[co * H_out * W_out + ho * W_out + wo] = sum;
                }
            }
        }
        return output;
    }

    // BatchNorm2d: normalize per channel with running stats
    std::vector<float> batch_norm_2d(const std::vector<float>& input,
                                      int C, int H, int W,
                                      const std::string& prefix) {
        const auto& gamma = get_tensor(prefix + ".weight");
        const auto& beta = get_tensor(prefix + ".bias");
        const auto& mean = get_tensor(prefix + ".running_mean");
        const auto& var = get_tensor(prefix + ".running_var");
        if (mean.empty()) return input;

        std::vector<float> output(C * H * W);
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
            float g = gamma.empty() ? 1.0f : gamma[c];
            float b = beta.empty() ? 0.0f : beta[c];
            for (int i = 0; i < H * W; ++i) {
                output[c * H * W + i] = g * (input[c * H * W + i] - mean[c]) * inv_std + b;
            }
        }
        return output;
    }

    // BatchNorm1d: [C] vector, with optional affine
    std::vector<float> batch_norm_1d(const std::vector<float>& input,
                                      int C,
                                      const std::string& prefix,
                                      bool affine = true) {
        const auto& mean = get_tensor(prefix + ".running_mean");
        const auto& var = get_tensor(prefix + ".running_var");
        if (mean.empty()) return input;

        std::vector<float> output(C);
        if (affine) {
            const auto& gamma = get_tensor(prefix + ".weight");
            const auto& beta = get_tensor(prefix + ".bias");
            for (int c = 0; c < C; ++c) {
                float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
                float g = gamma.empty() ? 1.0f : gamma[c];
                float b = beta.empty() ? 0.0f : beta[c];
                output[c] = g * (input[c] - mean[c]) * inv_std + b;
            }
        } else {
            for (int c = 0; c < C; ++c) {
                float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
                output[c] = (input[c] - mean[c]) * inv_std;
            }
        }
        return output;
    }

    // BatchNorm1d over [C, T] sequence
    std::vector<float> batch_norm_1d_seq(const std::vector<float>& input,
                                          int C, int T,
                                          const std::string& prefix) {
        const auto& gamma = get_tensor(prefix + ".weight");
        const auto& beta = get_tensor(prefix + ".bias");
        const auto& mean = get_tensor(prefix + ".running_mean");
        const auto& var = get_tensor(prefix + ".running_var");
        if (mean.empty()) return input;

        std::vector<float> output(C * T);
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
            float g = gamma.empty() ? 1.0f : gamma[c];
            float b = beta.empty() ? 0.0f : beta[c];
            for (int t = 0; t < T; ++t) {
                output[c * T + t] = g * (input[c * T + t] - mean[c]) * inv_std + b;
            }
        }
        return output;
    }

    // ResBlock (BasicBlock): stride only in freq (H), time (W) stays unchanged
    std::vector<float> res_block(const std::vector<float>& input,
                                  int C, int H, int W,
                                  const std::string& prefix,
                                  int stride) {
        // conv1: stride=(stride, 1) → downsample freq only
        int pad = 1;
        auto x = conv2d(input, C, H, W, prefix + ".conv1", C, 3, stride, 1, pad, pad);
        int H2 = (H + 2*pad - 3) / stride + 1;
        int W2 = W;  // time dimension unchanged
        x = batch_norm_2d(x, C, H2, W2, prefix + ".bn1");
        relu_inplace(x);

        // conv2: stride=1
        x = conv2d(x, C, H2, W2, prefix + ".conv2", C, 3, 1, 1, 1, 1);
        x = batch_norm_2d(x, C, H2, W2, prefix + ".bn2");

        // shortcut
        std::vector<float> shortcut;
        if (stride != 1) {
            // Downsample shortcut: Conv1x1 stride=(stride,1) + BN
            shortcut = conv2d(input, C, H, W, prefix + ".shortcut.0", C, 1, stride, 1, 0, 0);
            shortcut = batch_norm_2d(shortcut, C, H2, W2, prefix + ".shortcut.1");
        } else {
            shortcut = input;
        }

        // Add + ReLU
        for (size_t i = 0; i < x.size(); ++i) x[i] += shortcut[i];
        relu_inplace(x);
        return x;
    }

    // Conv1d with dilation support: [Cin, T] → [Cout, T']
    std::vector<float> conv1d(const std::vector<float>& input,
                               int Cin, int T,
                               const std::string& prefix,
                               int Cout, int k, int stride, int pad, int dilation) {
        const auto& weight = get_tensor(prefix + ".weight");
        auto bias_it = tensors_.find(prefix + ".bias");

        int T_out = (T + 2*pad - dilation*(k-1) - 1) / stride + 1;
        if (T_out <= 0 || weight.empty()) return std::vector<float>(Cout * std::max(1, T_out), 0);
        std::vector<float> output(Cout * T_out, 0);

        for (int co = 0; co < Cout; ++co) {
            float b = (bias_it != tensors_.end() && !bias_it->second.empty()) ? bias_it->second[co] : 0;
            for (int to = 0; to < T_out; ++to) {
                float sum = b;
                for (int ci = 0; ci < Cin; ++ci) {
                    for (int ki = 0; ki < k; ++ki) {
                        int ti = to * stride - pad + ki * dilation;
                        if (ti >= 0 && ti < T) {
                            sum += weight[co * Cin * k + ci * k + ki] * input[ci * T + ti];
                        }
                    }
                }
                output[co * T_out + to] = sum;
            }
        }
        return output;
    }

    // Segment-level average pooling (used by CAM layer)
    static std::vector<float> seg_pooling(const std::vector<float>& input,
                                           int C, int T, int seg_len = 100) {
        std::vector<float> output(C * T);
        int num_segs = (T + seg_len - 1) / seg_len;
        for (int c = 0; c < C; ++c) {
            for (int s = 0; s < num_segs; ++s) {
                int start = s * seg_len;
                int end = std::min(start + seg_len, T);
                float sum = 0;
                for (int t = start; t < end; ++t) sum += input[c * T + t];
                float avg = sum / (end - start);
                for (int t = start; t < end; ++t) output[c * T + t] = avg;
            }
        }
        return output;
    }

    // CAM (Context-Aware Masking) layer with segment pooling
    std::vector<float> cam_layer(const std::vector<float>& input,
                                  int bn_channels, int out_channels, int T,
                                  const std::string& prefix,
                                  int kernel_size, int dilation, int padding) {
        // linear_local: Conv1d(bn_channels→out_channels, k, d, pad)
        auto local_out = conv1d(input, bn_channels, T, prefix + ".linear_local",
                                out_channels, kernel_size, 1, padding, dilation);

        // Context: global mean + segment pooling → [bn_channels, T]
        auto seg_avg = seg_pooling(input, bn_channels, T, 100);
        std::vector<float> context(bn_channels * T);
        for (int c = 0; c < bn_channels; ++c) {
            float global_mean = 0;
            for (int t = 0; t < T; ++t) global_mean += input[c * T + t];
            global_mean /= T;
            for (int t = 0; t < T; ++t)
                context[c * T + t] = global_mean + seg_avg[c * T + t];
        }

        // linear1: Conv1d(bn_channels→bn_channels/2, k=1) applied per time step
        int mid_channels = bn_channels / 2;  // = 64
        const auto& w1 = get_tensor(prefix + ".linear1.weight");
        const auto& b1 = get_tensor(prefix + ".linear1.bias");
        std::vector<float> g1(mid_channels * T, 0);
        for (int o = 0; o < mid_channels; ++o) {
            float bias = b1.empty() ? 0 : b1[o];
            for (int t = 0; t < T; ++t) {
                float sum = bias;
                for (int i = 0; i < bn_channels; ++i)
                    sum += w1[o * bn_channels + i] * context[i * T + t];
                g1[o * T + t] = std::max(0.0f, sum);  // ReLU
            }
        }

        // linear2: Conv1d(bn_channels/2→out_channels, k=1) → sigmoid → mask
        const auto& w2 = get_tensor(prefix + ".linear2.weight");
        const auto& b2 = get_tensor(prefix + ".linear2.bias");
        std::vector<float> output(out_channels * T);
        for (int o = 0; o < out_channels; ++o) {
            float bias = b2.empty() ? 0 : b2[o];
            for (int t = 0; t < T; ++t) {
                float sum = bias;
                for (int i = 0; i < mid_channels; ++i)
                    sum += w2[o * mid_channels + i] * g1[i * T + t];
                float mask = 1.0f / (1.0f + expf(-sum));  // sigmoid
                output[o * T + t] = local_out[o * T + t] * mask;
            }
        }
        return output;
    }

    // CAM Dense TDNN Block with dilation
    std::pair<std::vector<float>, int> cam_dense_block(
            const std::vector<float>& input, int in_dim, int T,
            const std::string& prefix, int num_layers, int dilation) {
        const int growth = 32;
        const int bn_channels = 128;  // = bn_size(4) * growth(32)
        int kernel_size = 3;
        int padding = (kernel_size - 1) / 2 * dilation;

        std::vector<float> concat = input;
        int current_dim = in_dim;

        for (int l = 0; l < num_layers; ++l) {
            std::string lp = prefix + ".tdnnd" + std::to_string(l + 1);

            // nonlinear1: BN(current_dim) + ReLU
            auto normed = batch_norm_1d_seq(concat, current_dim, T,
                                             lp + ".nonlinear1.batchnorm");
            relu_inplace(normed);

            // linear1: Conv1d(current_dim→128, k=1, bias=False)
            auto h = conv1d(normed, current_dim, T, lp + ".linear1", bn_channels, 1, 1, 0, 1);

            // nonlinear2: BN(128) + ReLU
            h = batch_norm_1d_seq(h, bn_channels, T, lp + ".nonlinear2.batchnorm");
            relu_inplace(h);

            // CAM layer: Conv1d(128→32, k=3, dilation) + context attention
            h = cam_layer(h, bn_channels, growth, T, lp + ".cam_layer",
                          kernel_size, dilation, padding);

            // Concatenate: append [growth, T] to [current_dim, T]
            std::vector<float> new_concat((current_dim + growth) * T);
            std::copy(concat.begin(), concat.end(), new_concat.begin());
            std::copy(h.begin(), h.end(), new_concat.begin() + current_dim * T);
            concat = std::move(new_concat);
            current_dim += growth;
        }

        return {concat, current_dim};
    }

    // Transit layer: BN(in) → ReLU → Conv1d(in→out) [pre-norm]
    std::vector<float> transit_layer(const std::vector<float>& input,
                                      int in_dim, int T,
                                      const std::string& prefix,
                                      int out_dim) {
        // nonlinear: BN(in_dim) + ReLU  (applied before linear)
        auto x = batch_norm_1d_seq(input, in_dim, T, prefix + ".nonlinear.batchnorm");
        relu_inplace(x);
        // linear: Conv1d(in→out, k=1, bias=False)
        x = conv1d(x, in_dim, T, prefix + ".linear", out_dim, 1, 1, 0, 1);
        return x;
    }

    // Dense layer as Conv1d(k=1): Linear(in→out) for 1D vector
    std::vector<float> dense_layer_conv1d(const std::vector<float>& input,
                                           int in_dim,
                                           const std::string& prefix,
                                           int out_dim) {
        const auto& weight = get_tensor(prefix + ".weight");
        if (weight.empty()) return std::vector<float>(out_dim, 0);
        // weight shape is [out_dim, in_dim, 1] — same as [out_dim, in_dim]
        std::vector<float> output(out_dim, 0);
        for (int o = 0; o < out_dim; ++o) {
            for (int i = 0; i < in_dim; ++i) {
                output[o] += weight[o * in_dim + i] * input[i];
            }
        }
        return output;
    }

    // ---- Safetensors loader (same as VadEngine) ----
    static TensorMap load_safetensors(const std::string& path) {
        TensorMap result;
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return result;

        uint64_t header_size = 0;
        ifs.read(reinterpret_cast<char*>(&header_size), 8);
        if (header_size > 1000000) return result;

        std::string header(header_size, '\0');
        ifs.read(&header[0], header_size);

        size_t data_offset_base = 8 + header_size;
        size_t pos = 0;

        while (pos < header.size()) {
            size_t key_start = header.find('"', pos);
            if (key_start == std::string::npos) break;
            size_t key_end = header.find('"', key_start + 1);
            if (key_end == std::string::npos) break;
            std::string key = header.substr(key_start + 1, key_end - key_start - 1);
            pos = key_end + 1;

            if (key == "__metadata__") {
                size_t brace = header.find('{', pos);
                if (brace != std::string::npos) {
                    int depth = 1; pos = brace + 1;
                    while (pos < header.size() && depth > 0) {
                        if (header[pos] == '{') depth++;
                        else if (header[pos] == '}') depth--;
                        pos++;
                    }
                }
                continue;
            }

            size_t offsets_pos = header.find("data_offsets", pos);
            if (offsets_pos == std::string::npos) break;
            size_t bracket = header.find('[', offsets_pos);
            size_t comma = header.find(',', bracket);
            size_t end_bracket = header.find(']', comma);

            uint64_t start = std::stoull(header.substr(bracket + 1, comma - bracket - 1));
            uint64_t end = std::stoull(header.substr(comma + 1, end_bracket - comma - 1));

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
// SpeakerManager — 说话人注册/识别管理
// ============================================================================
class SpeakerManager {
public:
    struct MatchResult {
        std::string name;
        int   speaker_id  = -1;
        float similarity  = 0;
        bool  is_new      = false;
    };

    SpeakerManager() = default;

    // 注册说话人
    void register_speaker(const std::string& name, const std::vector<float>& embedding) {
        Speaker s;
        s.name = name;
        s.embedding = embedding;
        s.id = next_id_++;
        s.seen_count = 1;
        speakers_.push_back(s);
    }

    // 识别: 返回最匹配的说话人 (或注册新说话人)
    MatchResult identify(const std::vector<float>& embedding,
                         float threshold = 0.65f,
                         bool auto_register = true) {
        MatchResult best;
        best.similarity = -1;

        for (auto& s : speakers_) {
            float sim = CamPlusSpeakerEncoder::cosine_similarity(embedding, s.embedding);
            if (sim > best.similarity) {
                best.similarity = sim;
                best.name = s.name;
                best.speaker_id = s.id;
            }
        }

        if (best.similarity >= threshold) {
            // 更新 embedding (moving average)
            for (auto& s : speakers_) {
                if (s.id == best.speaker_id) {
                    update_embedding(s, embedding, 0.1f);
                    break;
                }
            }
            best.is_new = false;
            return best;
        }

        // 新说话人
        if (auto_register) {
            best.speaker_id = next_id_;
            best.name = "Speaker_" + std::to_string(next_id_);
            best.is_new = true;
            register_speaker(best.name, embedding);
        } else {
            best.speaker_id = -1;
            best.name = "Unknown";
            best.is_new = false;
        }
        return best;
    }

    // 列表
    int speaker_count() const { return (int)speakers_.size(); }

    // 获取所有说话人名称
    std::vector<std::string> speaker_names() const {
        std::vector<std::string> names;
        for (const auto& s : speakers_) names.push_back(s.name);
        return names;
    }

    // 按名称删除说话人
    bool remove_by_name(const std::string& name) {
        for (auto it = speakers_.begin(); it != speakers_.end(); ++it) {
            if (it->name == name) {
                speakers_.erase(it);
                return true;
            }
        }
        return false;
    }

    // 按名称获取 embedding
    std::vector<float> get_embedding(const std::string& name) const {
        for (const auto& s : speakers_) {
            if (s.name == name) return s.embedding;
        }
        return {};
    }

    // 按 ID 获取名称和 embedding
    std::pair<std::string, std::vector<float>> get_embedding_by_id(int id) const {
        for (const auto& s : speakers_) {
            if (s.id == id) return {s.name, s.embedding};
        }
        return {};
    }

    // 重置
    void clear() {
        speakers_.clear();
        next_id_ = 0;
    }

    // 二次聚类: 碎片吸收 + 已确立合并
    // 1) 将观测次数少的 "碎片" 说话人吸收到最近的 "确立" 说话人
    // 2) 如果 established_merge_threshold > 0, 合并相似度极高的已确立说话人对
    // 返回 old_id → new_id 映射
    std::unordered_map<int, int> merge_similar(float merge_threshold = 0.55f,
                                                int min_established = 5,
                                                float established_merge_threshold = -1.0f) {
        std::unordered_map<int, int> id_map;
        for (auto& s : speakers_) id_map[s.id] = s.id; // identity

        // 打印 pairwise 相似度矩阵 (诊断)
        if (speakers_.size() > 1) {
            fprintf(stderr, "[SpeakerMerge] pairwise similarity (fragment absorption, "
                    "threshold=%.2f, min_established=%d):\n",
                    merge_threshold, min_established);
            for (size_t i = 0; i < speakers_.size(); ++i) {
                const char* tag_i = speakers_[i].seen_count >= min_established ? "E" : "F";
                for (size_t j = i + 1; j < speakers_.size(); ++j) {
                    const char* tag_j = speakers_[j].seen_count >= min_established ? "E" : "F";
                    float sim = CamPlusSpeakerEncoder::cosine_similarity(
                        speakers_[i].embedding, speakers_[j].embedding);
                    fprintf(stderr, "  %s[%s]↔%s[%s]: %.3f (seen %d,%d)\n",
                        speakers_[i].name.c_str(), tag_i,
                        speakers_[j].name.c_str(), tag_j,
                        sim, speakers_[i].seen_count, speakers_[j].seen_count);
                }
            }
        }

        // 分离确立说话人与碎片说话人
        std::vector<size_t> established_idx, fragment_idx;
        for (size_t i = 0; i < speakers_.size(); ++i) {
            if (speakers_[i].seen_count >= min_established)
                established_idx.push_back(i);
            else
                fragment_idx.push_back(i);
        }
        fprintf(stderr, "[SpeakerMerge] %zu established, %zu fragments\n",
                established_idx.size(), fragment_idx.size());

        if (established_idx.empty()) return id_map; // 没有确立说话人, 无法吸收

        // 对每个碎片说话人, 找最近的确立说话人
        std::vector<size_t> to_remove;
        for (size_t fi : fragment_idx) {
            float best_sim = -1;
            size_t best_ei = 0;
            for (size_t ei : established_idx) {
                float sim = CamPlusSpeakerEncoder::cosine_similarity(
                    speakers_[fi].embedding, speakers_[ei].embedding);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_ei = ei;
                }
            }

            if (best_sim >= merge_threshold) {
                fprintf(stderr, "[SpeakerMerge] absorb %s (seen %d) → %s (sim %.3f)\n",
                        speakers_[fi].name.c_str(), speakers_[fi].seen_count,
                        speakers_[best_ei].name.c_str(), best_sim);
                // 更新 id_map
                int old_id = speakers_[fi].id;
                int new_id = speakers_[best_ei].id;
                for (auto& [k, v] : id_map) {
                    if (v == old_id) v = new_id;
                }
                // 加权平均 embedding
                auto& si = speakers_[best_ei];
                auto& sj = speakers_[fi];
                float wi = (float)si.seen_count;
                float wj = (float)sj.seen_count;
                float total = wi + wj;
                for (size_t k = 0; k < si.embedding.size() && k < sj.embedding.size(); ++k) {
                    si.embedding[k] = (wi * si.embedding[k] + wj * sj.embedding[k]) / total;
                }
                float norm = 0;
                for (float v : si.embedding) norm += v * v;
                norm = sqrtf(norm + 1e-12f);
                for (float& v : si.embedding) v /= norm;
                si.seen_count = (int)total;
                to_remove.push_back(fi);
            } else {
                fprintf(stderr, "[SpeakerMerge] keep %s (seen %d, best sim %.3f < %.2f)\n",
                        speakers_[fi].name.c_str(), speakers_[fi].seen_count,
                        best_sim, merge_threshold);
            }
        }

        // 删除被吸收的说话人 (从后往前)
        std::sort(to_remove.rbegin(), to_remove.rend());
        for (size_t idx : to_remove)
            speakers_.erase(speakers_.begin() + idx);

        // Pass 2: 已确立↔已确立合并 (处理高阈值 identify 导致的过度分裂)
        if (established_merge_threshold > 0) {
            bool merged_any = true;
            while (merged_any) {
                merged_any = false;
                float best_sim = -1;
                size_t best_i = 0, best_j = 0;
                for (size_t i = 0; i < speakers_.size(); ++i) {
                    if (speakers_[i].seen_count < min_established) continue;
                    for (size_t j = i + 1; j < speakers_.size(); ++j) {
                        if (speakers_[j].seen_count < min_established) continue;
                        float sim = CamPlusSpeakerEncoder::cosine_similarity(
                            speakers_[i].embedding, speakers_[j].embedding);
                        if (sim > best_sim) {
                            best_sim = sim;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                if (best_sim >= established_merge_threshold) {
                    // 合并 seen_count 较小的到较大的
                    size_t keep = best_i, absorb = best_j;
                    if (speakers_[keep].seen_count < speakers_[absorb].seen_count)
                        std::swap(keep, absorb);
                    fprintf(stderr, "[SpeakerMerge] merge established %s (seen %d) → %s (seen %d, sim %.3f)\n",
                            speakers_[absorb].name.c_str(), speakers_[absorb].seen_count,
                            speakers_[keep].name.c_str(), speakers_[keep].seen_count, best_sim);
                    int old_id = speakers_[absorb].id;
                    int new_id = speakers_[keep].id;
                    for (auto& [k, v] : id_map) {
                        if (v == old_id) v = new_id;
                    }
                    // 加权平均 embedding
                    auto& si = speakers_[keep];
                    auto& sj = speakers_[absorb];
                    float wi = (float)si.seen_count, wj = (float)sj.seen_count;
                    float total = wi + wj;
                    for (size_t k = 0; k < si.embedding.size() && k < sj.embedding.size(); ++k)
                        si.embedding[k] = (wi * si.embedding[k] + wj * sj.embedding[k]) / total;
                    float norm = 0;
                    for (float v : si.embedding) norm += v * v;
                    norm = sqrtf(norm + 1e-12f);
                    for (float& v : si.embedding) v /= norm;
                    si.seen_count = (int)total;
                    speakers_.erase(speakers_.begin() + absorb);
                    merged_any = true;
                }
            }
        }

        return id_map;
    }

private:
    struct Speaker {
        std::string name;
        std::vector<float> embedding;
        int id = 0;
        int seen_count = 0;
    };

    std::vector<Speaker> speakers_;
    int next_id_ = 0;

    void update_embedding(Speaker& s, const std::vector<float>& new_emb, float alpha) {
        for (size_t i = 0; i < s.embedding.size() && i < new_emb.size(); ++i) {
            s.embedding[i] = (1 - alpha) * s.embedding[i] + alpha * new_emb[i];
        }
        // Re-normalize
        float norm = 0;
        for (float v : s.embedding) norm += v * v;
        norm = sqrtf(norm + 1e-12f);
        for (float& v : s.embedding) v /= norm;
        s.seen_count++;
    }
};

} // namespace asr
} // namespace qwen_thor
