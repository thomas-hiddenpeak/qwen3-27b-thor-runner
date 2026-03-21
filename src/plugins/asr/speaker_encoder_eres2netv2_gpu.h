// speaker_encoder_eres2netv2_gpu.h — ERes2NetV2 GPU Speaker Encoder
//
// GPU 加速版本的 ERes2NetV2 说话人编码器 (17.8M params, 192-dim)。
// 使用 cuBLAS SGEMM (im2col for conv2d) + 自定义 CUDA kernels。
// 支持多 CUDA stream 并行 batch 提取。
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

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace qwen_thor {
namespace asr {

// ERes2NetV2 GPU workspace: pre-allocated buffers for one forward pass
struct ERes2Scratch {
    // Main buffers for layer IO and intermediates
    float* buf[8] = {};
    // im2col workspace
    float* im2col = nullptr;
    // Concatenation buffer for Res2Net scale outputs
    float* scale_cat = nullptr;
    // AFF workspace
    float* aff_tmp = nullptr;
    // Dedicated buffer for layer3 output backup (used by layer3_ds + fuse34)
    float* out3 = nullptr;

    size_t total_bytes = 0;
    bool alloc(int max_T);
    void free();
};

class GpuERes2NetV2Encoder {
public:
    GpuERes2NetV2Encoder();
    ~GpuERes2NetV2Encoder();

    bool load(const std::string& safetensors_path);
    bool is_loaded() const { return loaded_; }

    // 从 GPU Mel 特征提取 192-dim embedding
    // d_mel: GPU 指针, [T, 80] row-major
    std::vector<float> extract_gpu(const float* d_mel, int T);

    // 批量 GPU 提取: 多个 chunk 使用多 CUDA stream 并行处理
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
    cublasHandle_t cublas_ = nullptr;
    bool loaded_ = false;

    // Persistent resources (default stream)
    ERes2Scratch scratch_;
    cudaStream_t stream_ = nullptr;
    int scratch_max_T_ = 0;

    // Multi-stream batch resources (limited to avoid OOM on unified memory)
    static constexpr int BATCH_CONCURRENCY = 2;
    struct BatchResources {
        ERes2Scratch scratch[BATCH_CONCURRENCY];
        cudaStream_t streams[BATCH_CONCURRENCY] = {};
        cublasHandle_t cublas[BATCH_CONCURRENCY] = {};
        float* d_emb_buf = nullptr;
        int max_T = 0;
        bool initialized = false;
    };
    BatchResources batch_;
    bool ensure_batch(int max_T);

    // ERes2NetV2 config
    static constexpr int BASE_WIDTH = 26;
    static constexpr int SCALE = 2;
    static constexpr int EXPANSION = 2;

    // Core forward pass (no sync, writes 192-dim to d_emb_out)
    void forward_one(const float* d_mel, int T, ERes2Scratch& sp,
                     cudaStream_t stream, cublasHandle_t cublas,
                     float* d_emb_out);

    bool ensure_scratch(int T);

    const float* get_gpu(const std::string& name) const;
    TensorMap load_safetensors(const std::string& path);

    // GPU building blocks
    void gpu_conv2d(const float* input, float* output, float* im2col_buf,
                    int Cin, int H, int W, int Cout, int k, int stride, int pad,
                    const std::string& weight_name,
                    cublasHandle_t cublas, cudaStream_t stream);

    void gpu_basic_block(const float* input, float* output,
                         ERes2Scratch& sp,
                         int in_planes, int planes, int H, int W,
                         int stride, const std::string& prefix,
                         bool use_aff,
                         int& H_out, int& W_out,
                         cublasHandle_t cublas, cudaStream_t stream);

    void gpu_aff(const float* x, const float* y, float* output,
                 ERes2Scratch& sp,
                 int C, int H, int W, const std::string& prefix,
                 cublasHandle_t cublas, cudaStream_t stream);
};

} // namespace asr
} // namespace qwen_thor
