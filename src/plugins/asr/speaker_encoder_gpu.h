// speaker_encoder_gpu.h — CAM++ GPU Speaker Encoder
//
// GPU 加速版本的 CAM++ 说话人编码器。
// 使用 cuBLAS SGEMM + 自定义 CUDA kernels 替代 CPU 循环实现。
// 接口与 CamPlusSpeakerEncoder 完全兼容。

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace qwen_thor {
namespace asr {

// ScratchPool: pre-allocated GPU scratch buffers, reused each DenseTDNN layer
struct ScratchPool {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* d = nullptr;
    float* e = nullptr;
    float* f = nullptr;
    float* concat[2] = {nullptr, nullptr};
    int which_concat = 0;
    size_t total_bytes = 0;

    bool alloc(int max_T, int max_spatial);
    void free();
    float* cur_concat() { return concat[which_concat]; }
    float* next_concat() { return concat[1 - which_concat]; }
    void swap_concat() { which_concat = 1 - which_concat; }
};

class GpuSpeakerEncoder {
public:
    GpuSpeakerEncoder();
    ~GpuSpeakerEncoder();

    // 加载 safetensors 权重到 GPU
    bool load(const std::string& safetensors_path);
    bool is_loaded() const { return loaded_; }

    // 从 Mel 特征提取 192-dim embedding (mel: [T, 80] row-major)
    std::vector<float> extract(const float* mel_80xT, int T);

    // 从 GPU Mel 特征提取 (跳过 CPU CMN+transpose+H2D, 全 GPU 路径)
    // d_mel: GPU 指针, [T, 80] row-major
    std::vector<float> extract_gpu(const float* d_mel, int T);

    // 批量 GPU 提取: 多个 chunk 使用多 CUDA stream 并行处理
    // 减少 kernel launch + sync 开销, 1331 chunks: 33s → ~3s
    struct BatchChunk { const float* d_mel; int T; };
    std::vector<std::vector<float>> extract_batch_gpu(
        const std::vector<BatchChunk>& chunks);

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

    // GPU weights
    std::unordered_map<std::string, float*> gpu_tensors_;
    std::unordered_map<std::string, int> tensor_sizes_;
    float* workspace_ = nullptr;
    size_t workspace_size_ = 0;
    cublasHandle_t cublas_ = nullptr;
    bool loaded_ = false;

    // Persistent resources (avoid per-call alloc/free)
    ScratchPool scratch_;
    cudaStream_t stream_ = nullptr;
    int scratch_max_T_ = 0;

    // Multi-stream batch resources
    static constexpr int BATCH_CONCURRENCY = 16;
    struct BatchResources {
        ScratchPool scratch[BATCH_CONCURRENCY];
        cudaStream_t streams[BATCH_CONCURRENCY] = {};
        cublasHandle_t cublas[BATCH_CONCURRENCY] = {};
        float* d_emb_buf = nullptr;  // [BATCH_CONCURRENCY * 192]
        int max_T = 0;
        bool initialized = false;
    };
    BatchResources batch_;
    bool ensure_batch(int max_T);

    // Core forward pass (no sync, writes 192-dim embedding to d_emb_out)
    void forward_one(const float* d_mel, int T, ScratchPool& sp,
                     cudaStream_t stream, cublasHandle_t cublas,
                     float* d_emb_out);

    // Ensure scratch is large enough for T frames
    bool ensure_scratch(int T);

    const float* get_gpu(const std::string& name) const;
    TensorMap load_safetensors(const std::string& path);

    // Forward declarations — ScratchPool defined above
    void gpu_res_block(const float* d_input, float* d_output,
                        int C, int H, int W,
                        const std::string& prefix, int stride,
                        float* scratch_a, float* scratch_b,
                        cudaStream_t stream);
    void gpu_cam_dense_block(ScratchPool& sp, int in_dim, int T,
                              const std::string& prefix,
                              int num_layers, int dilation,
                              cublasHandle_t cublas,
                              cudaStream_t stream);
    void gpu_cam_layer(ScratchPool& sp, int bn_ch, int out_ch,
                        int T, const std::string& prefix,
                        int k, int dilation, int padding,
                        cublasHandle_t cublas,
                        cudaStream_t stream);
    void gpu_transit(ScratchPool& sp, int in_dim, int T,
                      const std::string& prefix, int out_dim,
                      cublasHandle_t cublas,
                      cudaStream_t stream);
};

} // namespace asr
} // namespace qwen_thor
