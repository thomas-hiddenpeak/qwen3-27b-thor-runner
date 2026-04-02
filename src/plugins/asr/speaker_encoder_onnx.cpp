// speaker_encoder_onnx.cpp — ONNX Runtime Speaker Encoder 实现
//
// 使用 ONNX Runtime C API 实现通用说话人编码器。
// 支持任意 WeSpeaker 导出的 ONNX 模型 (ResNet, ECAPA-TDNN, SimAMResNet 等)。
// 推理在 CPU 上执行 (Jetson AGX Thor 无独立 GPU EP, 且 embedding 提取不是瓶颈)。

#include "speaker_encoder_onnx.h"
#include "onnxruntime_c_api.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace qwen_thor {
namespace asr {

static const OrtApi* g_ort = nullptr;

static void check_status(OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "[OnnxEncoder] ORT error: %s\n", msg);
        g_ort->ReleaseStatus(status);
    }
}

OnnxSpeakerEncoder::OnnxSpeakerEncoder() = default;

OnnxSpeakerEncoder::~OnnxSpeakerEncoder() {
    if (session_) g_ort->ReleaseSession(session_);
    if (session_options_) g_ort->ReleaseSessionOptions(session_options_);
    if (memory_info_) g_ort->ReleaseMemoryInfo(memory_info_);
    if (env_) g_ort->ReleaseEnv(env_);
}

bool OnnxSpeakerEncoder::load(const std::string& onnx_path) {
    // Initialize ORT API
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "[OnnxEncoder] Failed to get ORT API\n");
        return false;
    }

    // Create env
    OrtStatus* status = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "speaker_encoder", &env_);
    if (status) { check_status(status); return false; }

    // Create session options
    status = g_ort->CreateSessionOptions(&session_options_);
    if (status) { check_status(status); return false; }

    // Use 4 threads for intra-op parallelism (Jetson has 14 cores)
    g_ort->SetIntraOpNumThreads(session_options_, 4);
    g_ort->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL);

    // Create session
    status = g_ort->CreateSession(env_, onnx_path.c_str(), session_options_, &session_);
    if (status) {
        check_status(status);
        return false;
    }

    // Get allocator
    status = g_ort->GetAllocatorWithDefaultOptions(&allocator_);
    if (status) { check_status(status); return false; }

    // Get input name
    {
        char* name = nullptr;
        status = g_ort->SessionGetInputName(session_, 0, allocator_, &name);
        if (status) { check_status(status); return false; }
        input_name_ = name;
        g_ort->AllocatorFree(allocator_, name);
    }

    // Get output name
    {
        char* name = nullptr;
        status = g_ort->SessionGetOutputName(session_, 0, allocator_, &name);
        if (status) { check_status(status); return false; }
        output_name_ = name;
        g_ort->AllocatorFree(allocator_, name);
    }

    // Detect embedding dim by running a dummy forward (1, 300, 80)
    {
        std::vector<float> dummy(300 * 80, 0.0f);
        int64_t input_shape[3] = {1, 300, 80};
        OrtMemoryInfo* mem_info = nullptr;
        g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        OrtValue* input_tensor = nullptr;
        g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, dummy.data(), dummy.size() * sizeof(float),
            input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        OrtValue* output_tensor = nullptr;

        status = g_ort->Run(session_, nullptr, input_names, &input_tensor, 1,
                            output_names, 1, &output_tensor);
        if (status) {
            check_status(status);
            g_ort->ReleaseValue(input_tensor);
            g_ort->ReleaseMemoryInfo(mem_info);
            return false;
        }

        // Get output shape
        OrtTensorTypeAndShapeInfo* type_info = nullptr;
        g_ort->GetTensorTypeAndShape(output_tensor, &type_info);
        size_t dim_count = 0;
        g_ort->GetDimensionsCount(type_info, &dim_count);
        std::vector<int64_t> dims(dim_count);
        g_ort->GetDimensions(type_info, dims.data(), dim_count);
        g_ort->ReleaseTensorTypeAndShapeInfo(type_info);

        embed_dim_ = (dim_count >= 2) ? (int)dims[dim_count - 1] : 192;

        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(input_tensor);
        g_ort->ReleaseMemoryInfo(mem_info);
    }

    // Create memory info for inference
    status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_);
    if (status) { check_status(status); return false; }

    // Extract model name
    {
        size_t last_slash = onnx_path.rfind('/');
        size_t last_dot = onnx_path.rfind('.');
        if (last_slash != std::string::npos && last_dot != std::string::npos && last_dot > last_slash) {
            model_name_ = onnx_path.substr(last_slash + 1, last_dot - last_slash - 1);
        } else {
            model_name_ = onnx_path;
        }
    }

    loaded_ = true;
    fprintf(stderr, "[OnnxEncoder] Loaded: %s (input=%s, output=%s, embed_dim=%d)\n",
            model_name_.c_str(), input_name_.c_str(), output_name_.c_str(), embed_dim_);
    return true;
}

void OnnxSpeakerEncoder::apply_cmn(float* feats, int T, int D) {
    // Cepstral Mean Normalization: subtract per-feature mean
    std::vector<float> mean(D, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            mean[d] += feats[t * D + d];
    float inv_T = 1.0f / T;
    for (int d = 0; d < D; ++d)
        mean[d] *= inv_T;
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d)
            feats[t * D + d] -= mean[d];
}

void OnnxSpeakerEncoder::l2_normalize(std::vector<float>& emb) {
    float norm = 0.0f;
    for (float v : emb) norm += v * v;
    norm = 1.0f / (sqrtf(norm) + 1e-12f);
    for (float& v : emb) v *= norm;
}

std::vector<float> OnnxSpeakerEncoder::extract(const float* mel_80xT, int T) {
    if (!loaded_ || T < 10) return {};

    // Copy mel data and apply CMN (input: [T, 80] row-major)
    const int D = 80;
    std::vector<float> feats(T * D);
    std::memcpy(feats.data(), mel_80xT, T * D * sizeof(float));
    apply_cmn(feats.data(), T, D);

    // Prepare input tensor [1, T, 80]
    int64_t input_shape[3] = {1, (int64_t)T, (int64_t)D};
    OrtValue* input_tensor = nullptr;
    g_ort->CreateTensorWithDataAsOrtValue(
        memory_info_, feats.data(), feats.size() * sizeof(float),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    OrtValue* output_tensor = nullptr;

    OrtStatus* status = g_ort->Run(session_, nullptr, input_names, &input_tensor, 1,
                                    output_names, 1, &output_tensor);
    if (status) {
        check_status(status);
        g_ort->ReleaseValue(input_tensor);
        return {};
    }

    // Extract embedding
    float* output_data = nullptr;
    g_ort->GetTensorMutableData(output_tensor, (void**)&output_data);

    std::vector<float> embedding(output_data, output_data + embed_dim_);
    l2_normalize(embedding);

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    return embedding;
}

std::vector<float> OnnxSpeakerEncoder::extract_gpu(const float* d_mel, int T) {
    if (!loaded_ || T < 10) return {};

    // Copy GPU mel to CPU
    const int D = 80;
    std::vector<float> cpu_mel(T * D);
    cudaMemcpy(cpu_mel.data(), d_mel, T * D * sizeof(float), cudaMemcpyDeviceToHost);

    return extract(cpu_mel.data(), T);
}

std::vector<std::vector<float>> OnnxSpeakerEncoder::extract_batch_gpu(
    const std::vector<BatchChunk>& chunks)
{
    if (!loaded_ || chunks.empty()) return {};

    auto t0 = std::chrono::steady_clock::now();

    // Process one-by-one (ORT CPU, no batch dim benefit for variable-length inputs)
    // Could batch with padding, but for ~1000 3s chunks at CPU speed this is fine
    std::vector<std::vector<float>> results;
    results.reserve(chunks.size());

    for (auto& chunk : chunks) {
        results.push_back(extract_gpu(chunk.d_mel, chunk.T));
    }

    auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[OnnxEncoder] Batch extracted %zu chunks in %.1f ms (%.1f ms/chunk)\n",
            chunks.size(), elapsed, elapsed / chunks.size());

    return results;
}

std::vector<std::vector<float>> OnnxSpeakerEncoder::run_batch(
    const float* mel_data, const std::vector<int>& lengths, int max_T, int batch_size)
{
    // Not used in current implementation (reserved for padded batch inference)
    (void)mel_data; (void)lengths; (void)max_T; (void)batch_size;
    return {};
}

}  // namespace asr
}  // namespace qwen_thor
