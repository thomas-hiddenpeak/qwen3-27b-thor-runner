// speaker_encoder_eres2netv2.h — ERes2NetV2 Speaker Encoder (CPU, BLAS-accelerated)
//
// ERes2NetV2 说话人编码器: 从 80-dim Mel 特征提取 192-dim speaker embedding。
// 比 CAM++ 更强的说话人区分能力 (3D-Speaker EER: 6.52% vs CAM++ 7.75%)。
//
// 架构 (3D-Speaker ERes2NetV2):
//   conv1(1→64, k=3, s=1, p=1) → BN → ReLU
//   layer1: 3× BasicBlock    (64→128, s=1)
//   layer2: 4× BasicBlock    (128→256, s=2)
//   layer3: 6× BasicBlockAFF (256→512, s=2)
//   layer4: 3× BasicBlockAFF (512→1024, s=2)
//   layer3_ds: Conv2d(512→1024, k=3, s=2, p=1)
//   fuse34: AFF(1024, r=4)
//   TSTP pool → Linear(20480→192) → L2-normalize → 192-dim
//
// 优化: conv2d 使用 im2col + cblas_sgemm (NVPL BLAS), ~50× faster than naive loops.

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
#include <cstdint>
#include <nvpl_blas_cblas.h>

namespace qwen_thor {
namespace asr {

class ERes2NetV2SpeakerEncoder {
public:
    ERes2NetV2SpeakerEncoder() = default;

    bool load(const std::string& safetensors_path) {
        tensors_ = load_safetensors(safetensors_path);
        if (tensors_.empty()) return false;
        // Verify it's an ERes2NetV2 model
        if (tensors_.find("layer3_ds.weight") == tensors_.end() ||
            tensors_.find("seg_1.weight") == tensors_.end()) {
            fprintf(stderr, "[ERes2NetV2] ERROR: not an ERes2NetV2 model\n");
            tensors_.clear();
            return false;
        }
        loaded_ = true;
        fprintf(stderr, "[ERes2NetV2] Loaded %zu tensors\n", tensors_.size());
        return true;
    }

    bool is_loaded() const { return loaded_; }

    // 从 Mel 特征 (80-dim × T frames) 提取 192-dim embedding
    // mel: [T, 80], row-major
    std::vector<float> extract(const float* mel_80xT, int T) {
        if (!loaded_ || T < 10) return {};

        // 0. CMN: subtract per-bin mean
        std::vector<float> mel_cmn(80 * T);
        float bin_mean[80] = {};
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < 80; ++f)
                bin_mean[f] += mel_80xT[t * 80 + f];
        float inv_T = 1.0f / T;
        for (int f = 0; f < 80; ++f) bin_mean[f] *= inv_T;
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < 80; ++f)
                mel_cmn[t * 80 + f] = mel_80xT[t * 80 + f] - bin_mean[f];

        // 1. Transpose: [T, 80] → [1, 80, T] (Cin=1, H=80, W=T for conv2d)
        std::vector<float> x(80 * T);
        for (int t = 0; t < T; ++t)
            for (int f = 0; f < 80; ++f)
                x[f * T + t] = mel_cmn[t * 80 + f];

        // 2. conv1 → bn1 → F.relu (standard ReLU, no cap)
        int H = 80, W = T;
        x = conv2d(x, 1, H, W, "conv1", 64, 3, 1, 1);
        x = batch_norm_2d(x, 64, H, W, "bn1");
        relu_inplace(x);

        // 3. layer1: 3× BasicBlock (planes=64, stride=1)
        int in_planes = 64;
        forward_layer(x, in_planes, H, W, 64, 3, 1, "layer1", false);

        // 4. layer2: 4× BasicBlock (planes=128, stride=2)
        forward_layer(x, in_planes, H, W, 128, 4, 2, "layer2", false);

        // Save layer3 input dimensions for layer3_ds later
        auto out2 = x;  // [256, H2, W2]
        int H2 = H, W2 = W;

        // 5. layer3: 6× BasicBlockAFF (planes=256, stride=2)
        forward_layer(x, in_planes, H, W, 256, 6, 2, "layer3", true);
        auto out3 = x;  // [512, H3, W3]
        int H3 = H, W3 = W;

        // 6. layer4: 3× BasicBlockAFF (planes=512, stride=2)
        forward_layer(x, in_planes, H, W, 512, 3, 2, "layer4", true);
        // out4: [1024, H4, W4]
        int H4 = H, W4 = W;

        // 7. layer3_ds: downsample layer3 output to match layer4 spatial dims
        auto out3_ds = conv2d(out3, 512, H3, W3, "layer3_ds", 1024, 3, 2, 1);
        // No BN after layer3_ds (just Conv2d, no bias)

        // 8. fuse34: AFF fusion of out4 and out3_ds
        x = aff_forward(x, out3_ds, 1024, H4, W4, "fuse34");

        // 9. TSTP: mean + std over time → [C*H*2]
        auto stats = tstp_pool(x, 1024, H4, W4);

        // 10. seg_1: Linear(20480→192)
        auto emb = linear(stats, (int)stats.size(), "seg_1", 192);

        // 11. L2 normalize
        float norm = 0;
        for (float v : emb) norm += v * v;
        norm = sqrtf(norm + 1e-12f);
        for (float& v : emb) v /= norm;

        return emb;
    }

    static constexpr int embedding_dim() { return 192; }

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

    // Config (ERes2NetV2 defaults)
    static constexpr int BASE_WIDTH = 26;
    static constexpr int SCALE = 2;
    static constexpr int EXPANSION = 2;
    static constexpr int M_CHANNELS = 64;

    const std::vector<float>& get_tensor(const std::string& name) const {
        static std::vector<float> empty;
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            fprintf(stderr, "[ERes2NetV2] WARNING: tensor '%s' not found\n", name.c_str());
            return empty;
        }
        return it->second;
    }

    bool has_tensor(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    // ========== Basic Ops ==========

    static void relu_inplace(std::vector<float>& x) {
        for (float& v : x) v = std::max(0.0f, v);
    }

    // HardTanh(0, 20) — used in Res2Net blocks
    static void hardtanh_inplace(std::vector<float>& x) {
        for (float& v : x) v = std::min(std::max(v, 0.0f), 20.0f);
    }

    static void silu_inplace(std::vector<float>& x) {
        for (float& v : x) v = v / (1.0f + expf(-v));
    }

    // ========== im2col helper ==========
    // Input [Cin, H, W] → col [Cin*k*k, H_out*W_out]
    static void im2col(const float* input, int Cin, int H, int W,
                       int k, int stride, int pad,
                       float* col, int H_out, int W_out) {
        int col_width = H_out * W_out;
        for (int c = 0; c < Cin; ++c) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int row = (c * k + kh) * k + kw;
                    float* col_row = col + row * col_width;
                    for (int ho = 0; ho < H_out; ++ho) {
                        int hi = ho * stride - pad + kh;
                        for (int wo = 0; wo < W_out; ++wo) {
                            int wi = wo * stride - pad + kw;
                            col_row[ho * W_out + wo] =
                                (hi >= 0 && hi < H && wi >= 0 && wi < W)
                                    ? input[c * H * W + hi * W + wi] : 0.0f;
                        }
                    }
                }
            }
        }
    }

    // Conv2d: [Cin, H, W] → [Cout, H', W'] using im2col + cblas_sgemm
    std::vector<float> conv2d(const std::vector<float>& input,
                              int Cin, int H, int W,
                              const std::string& prefix,
                              int Cout, int k, int stride, int pad) {
        const auto& weight = get_tensor(prefix + ".weight");
        int H_out = (H + 2 * pad - k) / stride + 1;
        int W_out = (W + 2 * pad - k) / stride + 1;
        if (weight.empty() || H_out <= 0 || W_out <= 0)
            return std::vector<float>(Cout * std::max(1, H_out) * std::max(1, W_out), 0);

        int N = H_out * W_out;
        int K = Cin * k * k;
        std::vector<float> output(Cout * N);

        if (k == 1 && stride == 1 && pad == 0) {
            // 1×1 conv, stride=1: direct GEMM, no im2col needed
            // weight [Cout, Cin] × input [Cin, H*W] = output [Cout, H*W]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Cout, N, Cin, 1.0f,
                        weight.data(), Cin,
                        input.data(), N,
                        0.0f, output.data(), N);
        } else {
            // General conv: im2col + GEMM
            std::vector<float> col(K * N);
            im2col(input.data(), Cin, H, W, k, stride, pad, col.data(), H_out, W_out);
            // weight [Cout, K] × col [K, N] = output [Cout, N]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Cout, N, K, 1.0f,
                        weight.data(), K,
                        col.data(), N,
                        0.0f, output.data(), N);
        }
        return output;
    }

    // Conv2d with bias: [Cin, H, W] → [Cout, H', W']
    std::vector<float> conv2d_bias(const std::vector<float>& input,
                                   int Cin, int H, int W,
                                   const std::string& prefix,
                                   int Cout, int k, int stride, int pad) {
        auto output = conv2d(input, Cin, H, W, prefix, Cout, k, stride, pad);
        auto it = tensors_.find(prefix + ".bias");
        if (it != tensors_.end() && !it->second.empty()) {
            int H_out = (H + 2 * pad - k) / stride + 1;
            int W_out = (W + 2 * pad - k) / stride + 1;
            int spatial = H_out * W_out;
            for (int co = 0; co < Cout; ++co) {
                float b = it->second[co];
                for (int i = 0; i < spatial; ++i)
                    output[co * spatial + i] += b;
            }
        }
        return output;
    }

    // BatchNorm2d: [C, H, W] → [C, H, W]
    std::vector<float> batch_norm_2d(const std::vector<float>& input,
                                     int C, int H, int W,
                                     const std::string& prefix) {
        const auto& mean = get_tensor(prefix + ".running_mean");
        const auto& var = get_tensor(prefix + ".running_var");
        if (mean.empty()) return input;

        const auto& gamma = get_tensor(prefix + ".weight");
        const auto& beta = get_tensor(prefix + ".bias");

        std::vector<float> output(C * H * W);
        int spatial = H * W;
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
            float g = gamma.empty() ? 1.0f : gamma[c];
            float b = beta.empty() ? 0.0f : beta[c];
            float scale = g * inv_std;
            float shift = b - g * mean[c] * inv_std;
            for (int i = 0; i < spatial; ++i)
                output[c * spatial + i] = input[c * spatial + i] * scale + shift;
        }
        return output;
    }

    // ========== AFF (Attention Feature Fusion) ==========
    // local_att: Conv2d(2C→C/r, k=1, bias) → BN → SiLU → Conv2d(C/r→C, k=1, bias) → BN
    // att = 1 + tanh(result)
    // out = x * att + y * (2 - att)
    std::vector<float> aff_forward(const std::vector<float>& x,
                                   const std::vector<float>& y,
                                   int C, int H, int W,
                                   const std::string& prefix) {
        int inter_C = C / 4;  // r=4
        int spatial = H * W;

        // Concatenate x and y along channel dim: [2C, H, W]
        std::vector<float> xa(2 * C * spatial);
        std::memcpy(xa.data(), x.data(), C * spatial * sizeof(float));
        std::memcpy(xa.data() + C * spatial, y.data(), C * spatial * sizeof(float));

        // local_att.0: Conv2d(2C→inter_C, k=1, bias)
        auto h = conv2d_bias(xa, 2 * C, H, W, prefix + ".local_att.0", inter_C, 1, 1, 0);
        // local_att.1: BN
        h = batch_norm_2d(h, inter_C, H, W, prefix + ".local_att.1");
        // local_att.2: SiLU (implicit in Sequential, index 2)
        silu_inplace(h);
        // local_att.3: Conv2d(inter_C→C, k=1, bias)
        h = conv2d_bias(h, inter_C, H, W, prefix + ".local_att.3", C, 1, 1, 0);
        // local_att.4: BN
        h = batch_norm_2d(h, C, H, W, prefix + ".local_att.4");

        // att = 1 + tanh(h)
        // out = x * att + y * (2 - att)
        std::vector<float> output(C * spatial);
        for (int i = 0; i < C * spatial; ++i) {
            float att = 1.0f + tanhf(h[i]);
            output[i] = x[i] * att + y[i] * (2.0f - att);
        }
        return output;
    }

    // ========== Res2Net Basic Block (without AFF) ==========
    // conv1(k=1, s=stride) → BN → ReLU → split → [conv3x3 → BN → ReLU per scale] → cat
    // → conv3(k=1) → BN + shortcut → ReLU
    std::vector<float> basic_block(const std::vector<float>& input,
                                   int in_planes, int planes,
                                   int H, int W, int stride,
                                   const std::string& prefix,
                                   bool use_aff,
                                   int& H_out, int& W_out) {
        int width = (int)std::floor(planes * (BASE_WIDTH / 64.0));
        int width_x_scale = width * SCALE;
        int out_planes = planes * EXPANSION;

        // conv1: in_planes → width*scale, k=1, s=stride
        auto x = conv2d(input, in_planes, H, W, prefix + ".conv1", width_x_scale, 1, stride, 0);
        H_out = (H - 1) / stride + 1;
        W_out = (W - 1) / stride + 1;
        x = batch_norm_2d(x, width_x_scale, H_out, W_out, prefix + ".bn1");
        hardtanh_inplace(x);

        // Split into SCALE chunks of width
        int spatial = H_out * W_out;

        // Process scales
        std::vector<float> out_cat;  // will accumulate [width * SCALE, H_out, W_out]
        std::vector<float> sp;       // running state for cumulative processing

        for (int s = 0; s < SCALE; ++s) {
            // Extract spx[s]: channels [s*width, (s+1)*width)
            std::vector<float> spx(width * spatial);
            std::memcpy(spx.data(), x.data() + s * width * spatial, width * spatial * sizeof(float));

            if (s == 0) {
                sp = spx;
            } else if (use_aff) {
                // AFF fusion: sp = fuse_models[s-1](sp, spx)
                sp = aff_forward(sp, spx, width, H_out, W_out,
                                 prefix + ".fuse_models." + std::to_string(s - 1));
            } else {
                // Simple addition: sp = sp + spx
                for (int i = 0; i < width * spatial; ++i)
                    sp[i] += spx[i];
            }

            // convs[s] + bns[s] + relu
            sp = conv2d(sp, width, H_out, W_out,
                        prefix + ".convs." + std::to_string(s), width, 3, 1, 1);
            sp = batch_norm_2d(sp, width, H_out, W_out,
                               prefix + ".bns." + std::to_string(s));
            hardtanh_inplace(sp);

            // Concatenate to output
            if (s == 0) {
                out_cat = sp;
            } else {
                out_cat.insert(out_cat.end(), sp.begin(), sp.end());
            }
        }

        // conv3: width*scale → out_planes, k=1
        auto out = conv2d(out_cat, width_x_scale, H_out, W_out,
                          prefix + ".conv3", out_planes, 1, 1, 0);
        out = batch_norm_2d(out, out_planes, H_out, W_out, prefix + ".bn3");

        // Shortcut
        std::vector<float> shortcut;
        if (stride != 1 || in_planes != out_planes) {
            shortcut = conv2d(input, in_planes, H, W,
                              prefix + ".shortcut.0", out_planes, 1, stride, 0);
            shortcut = batch_norm_2d(shortcut, out_planes, H_out, W_out,
                                     prefix + ".shortcut.1");
        } else {
            shortcut = input;
        }

        // Residual add + ReLU
        for (size_t i = 0; i < out.size(); ++i)
            out[i] += shortcut[i];
        hardtanh_inplace(out);

        return out;
    }

    // ========== Make Layer ==========
    void forward_layer(std::vector<float>& x, int& in_planes,
                       int& H, int& W,
                       int planes, int num_blocks, int stride,
                       const std::string& prefix, bool use_aff) {
        // First block uses the given stride, rest use stride=1
        for (int i = 0; i < num_blocks; ++i) {
            int s = (i == 0) ? stride : 1;
            int H_out, W_out;
            x = basic_block(x, in_planes, planes, H, W, s, 
                           prefix + "." + std::to_string(i),
                           use_aff, H_out, W_out);
            H = H_out;
            W = W_out;
            in_planes = planes * EXPANSION;
        }
    }

    // ========== TSTP Pool ==========
    // Input: [C, H, W] → mean + std over W (time dim) → flatten → [C*H*2]
    std::vector<float> tstp_pool(const std::vector<float>& input,
                                 int C, int H, int W) {
        int CH = C * H;
        std::vector<float> stats(CH * 2);

        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                int idx = c * H + h;
                const float* row = &input[c * H * W + h * W];

                // Mean
                float sum = 0;
                for (int w = 0; w < W; ++w) sum += row[w];
                float mean = sum / W;

                // Variance (unbiased, Bessel's correction)
                float var_sum = 0;
                for (int w = 0; w < W; ++w) {
                    float d = row[w] - mean;
                    var_sum += d * d;
                }
                float var = (W > 1) ? var_sum / (W - 1) : 0.0f;
                float std_val = sqrtf(var + 1e-8f);

                stats[idx] = mean;
                stats[CH + idx] = std_val;
            }
        }
        return stats;
    }

    // ========== Linear (cblas_sgemv) ==========
    std::vector<float> linear(const std::vector<float>& input,
                              int in_dim, const std::string& prefix,
                              int out_dim) {
        const auto& weight = get_tensor(prefix + ".weight");
        const auto& bias = get_tensor(prefix + ".bias");
        if (weight.empty()) return std::vector<float>(out_dim, 0);

        std::vector<float> output(out_dim);
        // weight [out_dim, in_dim] × input [in_dim] = output [out_dim]
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    out_dim, in_dim, 1.0f,
                    weight.data(), in_dim,
                    input.data(), 1,
                    0.0f, output.data(), 1);
        if (!bias.empty()) {
            for (int o = 0; o < out_dim; ++o)
                output[o] += bias[o];
        }
        return output;
    }

    // ========== Safetensors Loader ==========
    static TensorMap load_safetensors(const std::string& path) {
        TensorMap result;
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) return result;

        uint64_t header_size = 0;
        ifs.read(reinterpret_cast<char*>(&header_size), 8);
        if (header_size > 10000000) return result;

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

} // namespace asr
} // namespace qwen_thor
