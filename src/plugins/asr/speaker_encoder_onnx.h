// speaker_encoder_onnx.h — ONNX Runtime Speaker Encoder
//
// 通用 ONNX 格式说话人编码器。输入 [T, 80] Mel 特征, 输出 L2-normalized embedding。
// 支持任意 WeSpeaker 导出的 ONNX 模型: ResNet, ECAPA-TDNN, SimAMResNet 等。
// 使用 ONNX Runtime C API, CPU 执行。

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cuda_runtime.h>

// Forward declare ORT types to avoid header pollution
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtAllocator;

namespace qwen_thor {
namespace asr {

class OnnxSpeakerEncoder {
public:
    OnnxSpeakerEncoder();
    ~OnnxSpeakerEncoder();

    // Load ONNX model file
    bool load(const std::string& onnx_path);
    bool is_loaded() const { return loaded_; }

    // Extract embedding from CPU Mel features [T, 80] row-major
    std::vector<float> extract(const float* mel_80xT, int T);

    // Extract from GPU Mel features (copies to CPU first, then does ONNX inference)
    std::vector<float> extract_gpu(const float* d_mel, int T);

    // Batch extract from GPU Mel features
    struct BatchChunk { const float* d_mel; int T; };
    std::vector<std::vector<float>> extract_batch_gpu(
        const std::vector<BatchChunk>& chunks);

    // Dynamic embedding dim (detected from model output shape)
    int embedding_dim() const { return embed_dim_; }

    // Model name (for logging)
    const std::string& model_name() const { return model_name_; }

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
    OrtEnv* env_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtAllocator* allocator_ = nullptr;
    bool loaded_ = false;
    int embed_dim_ = 192;
    std::string model_name_;
    std::string input_name_;
    std::string output_name_;

    // CMN (cepstral mean normalization) + L2 normalization
    void apply_cmn(float* feats, int T, int D);
    void l2_normalize(std::vector<float>& emb);

    // Run ONNX inference on CPU mel [B, T, 80]
    std::vector<std::vector<float>> run_batch(const float* mel_data,
                                               const std::vector<int>& lengths,
                                               int max_T, int batch_size);
};

}  // namespace asr
}  // namespace qwen_thor
