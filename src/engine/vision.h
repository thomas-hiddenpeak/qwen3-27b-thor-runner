// vision.h — Qwen3.5 Vision Encoder (ViT + Merger)
//
// 27-layer Vision Transformer with:
//   - Conv3D patch embedding (equivalent to linear proj on flattened patches)
//   - Learned absolute position embeddings (bilinear interpolated)
//   - 2D rotary position encoding (half-rotation, row+col)
//   - Bidirectional attention (no causal mask), 16 heads, head_dim=72
//   - GELU(tanh) activated MLP
//   - 2×2 spatial merger → projects to text hidden_size (5120)
//
// Architecture:
//   Input:  [N_patches, C*T*P*P=1536] pixel patches (merge-friendly order)
//   Output: [N_output, 5120] visual features (N_output = N_patches / 4)

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <vector>
#include <string>

namespace qwen_thor {
namespace core {

// ============================================================================
// Vision Encoder Configuration (from config.json vision_config)
// ============================================================================
struct VisionConfig {
    int depth                   = 27;     // number of ViT blocks
    int hidden_size             = 1152;   // ViT hidden dim
    int num_heads               = 16;     // attention heads
    int head_dim                = 72;     // hidden_size / num_heads
    int intermediate_size       = 4304;   // MLP intermediate
    int in_channels             = 3;
    int patch_size              = 16;
    int temporal_patch_size     = 2;
    int spatial_merge_size      = 2;
    int out_hidden_size         = 5120;   // output dim (= text hidden_size)
    int num_position_embeddings = 2304;   // 48² learned embeddings
    int rotary_dim              = 36;     // head_dim / 2  (only half dims rotated)
    float layernorm_eps         = 1e-6f;
    float rope_theta            = 10000.0f;

    // Derived
    int patch_input_dim() const { return in_channels * temporal_patch_size * patch_size * patch_size; }  // 1536
    int merger_hidden() const { return hidden_size * spatial_merge_size * spatial_merge_size; }           // 4608
    int pos_grid_size() const;  // sqrt(num_position_embeddings) = 48

    // Preprocessing limits
    int min_pixels = 65536;      // 256²
    int max_pixels = 262144;     // 512²  — 1024 patches max, ~0.5s vision fwd on Thor
    int factor() const { return patch_size * spatial_merge_size; }  // 32
};

// ============================================================================
// Image Input / Processed Image
// ============================================================================
struct ImageInput {
    std::vector<uint8_t> pixels;  // RGB, row-major [height, width, 3]
    int width  = 0;
    int height = 0;
};

struct VideoInput {
    std::vector<std::vector<uint8_t>> frames;  // each frame: RGB row-major [H, W, 3]
    int width  = 0;    // uniform frame width
    int height = 0;    // uniform frame height
    float source_fps = 24.0f;   // original video FPS (for timestamp calc)
    float target_fps = 2.0f;    // target sample FPS (default: 2)
    int min_frames = 4;
    int max_frames = 768;
};

struct ProcessedImage {
    std::vector<uint16_t> pixel_values_bf16; // BF16 patches [N_patches, 1536] in merge order
    std::vector<int> positions_hw;           // precomputed RoPE positions [N_patches, 2] (row, col)
    int grid_t = 0;                   // temporal grid (1 for single image, >1 for video)
    int grid_h = 0;                   // height grid (patches, before merge)
    int grid_w = 0;                   // width grid (patches, before merge)
    std::vector<float> timestamps;    // per temporal-group timestamps (empty for images)
    bool is_video = false;            // true if from video input

    int num_patches() const { return grid_t * grid_h * grid_w; }
    int num_output_tokens() const { return grid_t * (grid_h / 2) * (grid_w / 2); }
    int tokens_per_frame() const { return (grid_h / 2) * (grid_w / 2); }
};

// ============================================================================
// Vision Encoder (ViT + Merger)
// ============================================================================
class VisionEncoder {
public:
    explicit VisionEncoder(const VisionConfig& config);
    ~VisionEncoder();

    // ---- Weight Binding ----
    void set_patch_embed_weights(__nv_bfloat16* proj_w, __nv_bfloat16* proj_b);
    void set_pos_embed_weight(__nv_bfloat16* w);
    void set_block_weights(int block_idx,
                           __nv_bfloat16* norm1_w, __nv_bfloat16* norm1_b,
                           __nv_bfloat16* qkv_w,   __nv_bfloat16* qkv_b,
                           __nv_bfloat16* proj_w,   __nv_bfloat16* proj_b,
                           __nv_bfloat16* norm2_w, __nv_bfloat16* norm2_b,
                           __nv_bfloat16* fc1_w,   __nv_bfloat16* fc1_b,
                           __nv_bfloat16* fc2_w,   __nv_bfloat16* fc2_b);
    void set_merger_weights(__nv_bfloat16* norm_w,  __nv_bfloat16* norm_b,
                            __nv_bfloat16* fc1_w,   __nv_bfloat16* fc1_b,
                            __nv_bfloat16* fc2_w,   __nv_bfloat16* fc2_b);

    // ---- CPU-side Image Preprocessing ----
    // Smart resize → normalize → patch extraction in merge-friendly order
    static ProcessedImage preprocess_image(const ImageInput& image,
                                           const VisionConfig& config);

    // ---- CPU-side Video Preprocessing ----
    // Frame sampling → smart resize (with temporal budget) → normalize
    // → multi-frame patch extraction with grid_t > 1
    static ProcessedImage preprocess_video(const VideoInput& video,
                                           const VisionConfig& config);

    // ---- Compute video grid dimensions (lightweight, no pixel processing) ----
    // Returns (grid_t, grid_h, grid_w) given frame count and resolution
    static std::tuple<int, int, int> compute_video_grid(
        int num_frames, int height, int width, const VisionConfig& config);

    // ---- GPU Forward Pass ----
    // Takes preprocessed patches, returns visual features [N_output, out_hidden_size]
    // The returned pointer is within workspace.
    // workspace must be at least workspace_bytes(num_patches) bytes.
    __nv_bfloat16* forward(const ProcessedImage& image,
                           __nv_bfloat16* workspace,
                           size_t workspace_bytes,
                           cudaStream_t stream);

    // Workspace size needed for a given number of patches
    size_t workspace_bytes(int num_patches) const;

    const VisionConfig& config() const { return config_; }

private:
    // ---- Internal forward helpers ----
    void patch_embed_forward(__nv_bfloat16* out, const __nv_bfloat16* patches,
                             int num_patches, cudaStream_t stream);
    void add_position_embedding(__nv_bfloat16* hidden, int grid_h, int grid_w,
                                int num_patches, cudaStream_t stream);
    void block_forward(int block_idx, __nv_bfloat16* hidden, __nv_bfloat16* workspace,
                       const float* rope_cos, const float* rope_sin,
                       int num_patches, cudaStream_t stream);
    void merger_forward(__nv_bfloat16* out, const __nv_bfloat16* hidden,
                        __nv_bfloat16* workspace, int num_patches, cudaStream_t stream);

    // cuBLAS handle (lazy init)
    cublasHandle_t get_cublas();

    VisionConfig config_;
    cublasHandle_t cublas_handle_ = nullptr;
    bool cublas_warmed_up_ = false;

    // ---- Weight pointers ----
    // Patch embedding: Conv3D [1152, 3, 2, 16, 16] reshaped as [1152, 1536]
    __nv_bfloat16* patch_proj_w_ = nullptr;   // [hidden_size, patch_input_dim] = [1152, 1536]
    __nv_bfloat16* patch_proj_b_ = nullptr;   // [hidden_size] = [1152]

    // Learned position embedding
    __nv_bfloat16* pos_embed_w_ = nullptr;    // [2304, 1152]

    // Per-block weights (27 blocks)
    struct BlockWeights {
        __nv_bfloat16* norm1_w = nullptr;   // [1152]
        __nv_bfloat16* norm1_b = nullptr;   // [1152]
        __nv_bfloat16* qkv_w   = nullptr;   // [3456, 1152]  (Q+K+V merged)
        __nv_bfloat16* qkv_b   = nullptr;   // [3456]
        __nv_bfloat16* proj_w  = nullptr;   // [1152, 1152]  (output projection)
        __nv_bfloat16* proj_b  = nullptr;   // [1152]
        __nv_bfloat16* norm2_w = nullptr;   // [1152]
        __nv_bfloat16* norm2_b = nullptr;   // [1152]
        __nv_bfloat16* fc1_w   = nullptr;   // [4304, 1152]
        __nv_bfloat16* fc1_b   = nullptr;   // [4304]
        __nv_bfloat16* fc2_w   = nullptr;   // [1152, 4304]
        __nv_bfloat16* fc2_b   = nullptr;   // [1152]
    };
    std::vector<BlockWeights> blocks_;

    // Merger weights
    __nv_bfloat16* merger_norm_w_  = nullptr;  // [1152]
    __nv_bfloat16* merger_norm_b_  = nullptr;  // [1152]
    __nv_bfloat16* merger_fc1_w_   = nullptr;  // [4608, 4608]
    __nv_bfloat16* merger_fc1_b_   = nullptr;  // [4608]
    __nv_bfloat16* merger_fc2_w_   = nullptr;  // [5120, 4608]
    __nv_bfloat16* merger_fc2_b_   = nullptr;  // [5120]
};

// ============================================================================
// Vision CUDA Kernel Launch Wrappers
// ============================================================================
namespace vision_ops {

// Standard LayerNorm (with bias): y = (x - mean) / sqrt(var + eps) * w + b
void invoke_layernorm(__nv_bfloat16* out, const __nv_bfloat16* x,
                      const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                      float eps, int num_tokens, int hidden_size,
                      cudaStream_t stream);

// Fused Add + LayerNorm: hidden[i] += addend[i]; out = LN(hidden)
// Saves 1 kernel launch + 1 memory round-trip vs separate add + LN
// out may safely alias addend (addend read in pass 1, out written in pass 3)
void invoke_fused_add_layernorm(__nv_bfloat16* out, __nv_bfloat16* hidden,
                                const __nv_bfloat16* addend,
                                const __nv_bfloat16* weight, const __nv_bfloat16* bias,
                                float eps, int num_tokens, int hidden_size,
                                cudaStream_t stream);

// GELU with tanh approximation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
void invoke_gelu_tanh(__nv_bfloat16* x, int n, cudaStream_t stream);

// Standard GELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))
void invoke_gelu(__nv_bfloat16* x, int n, cudaStream_t stream);

// Add bias: out[i] += bias[i % hidden_size]
void invoke_add_bias(__nv_bfloat16* x, const __nv_bfloat16* bias,
                     int num_tokens, int hidden_size, cudaStream_t stream);

// Add residual: out[i] = a[i] + b[i]
void invoke_add(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                int n, cudaStream_t stream);

// Bidirectional softmax (no causal mask): applied per-row
// scores: [num_heads, seq_len, seq_len], in-place
void invoke_softmax(float* scores, int num_heads, int seq_len, cudaStream_t stream);

// Softmax (FP32) then in-place cast to BF16 for P×V GEMM
// scores_bf16_out aliases scores_f32 (reinterpret cast, BF16 fits in first half)
void invoke_softmax_cast_bf16(float* scores_f32, __nv_bfloat16* scores_bf16_out,
                              int num_heads, int seq_len, cudaStream_t stream);

// FlashAttention for Vision (bidirectional, online softmax in registers)
// Q/K/V: [H, N, D] HND layout, output: [N, H, D] NHD layout
// Single kernel replaces: chunked QK GEMM + softmax + cast + PV GEMM + transpose
void invoke_vision_flash_attention(__nv_bfloat16* out,
                                    const __nv_bfloat16* Q,
                                    const __nv_bfloat16* K,
                                    const __nv_bfloat16* V,
                                    int N, int H, int D, float scale,
                                    cudaStream_t stream);

// Fused bias + GELU(tanh): x[i] = gelu_tanh(x[i] + bias[i % hs])
void invoke_add_bias_gelu_tanh(__nv_bfloat16* x, const __nv_bfloat16* bias,
                                int num_tokens, int hidden_size,
                                cudaStream_t stream);

// Fused bias + residual add: hidden[i] += x[i] + bias[i % hs]
void invoke_add_bias_residual(__nv_bfloat16* hidden, const __nv_bfloat16* x,
                               const __nv_bfloat16* bias,
                               int num_tokens, int hidden_size,
                               cudaStream_t stream);

// Fused add + bias + LayerNorm: hidden += (addend + addend_bias); out = LN(hidden)
void invoke_fused_add_bias_layernorm(__nv_bfloat16* out, __nv_bfloat16* hidden,
                                      const __nv_bfloat16* addend,
                                      const __nv_bfloat16* addend_bias,
                                      const __nv_bfloat16* weight,
                                      const __nv_bfloat16* bias,
                                      float eps, int num_tokens, int hidden_size,
                                      cudaStream_t stream);

// 2D Vision RoPE: apply rotary embedding to Q and K
// positions: [num_patches, 2] (row, col) — already on device
void invoke_vision_rope(__nv_bfloat16* q, __nv_bfloat16* k,
                        const float* rope_cos, const float* rope_sin,
                        int num_patches, int num_heads, int head_dim, int rotary_dim,
                        cudaStream_t stream);

// Compute RoPE cos/sin tables for 2D vision positions
// cos_out, sin_out: [num_patches, rotary_dim] (FP32)
void invoke_compute_vision_rope_table(float* cos_out, float* sin_out,
                                       const int* positions_hw,
                                       int num_patches, int rotary_dim,
                                       float theta, int grid_h, int grid_w,
                                       cudaStream_t stream);

// Bilinear interpolation of position embeddings
// embed_table: [num_pos, hidden_size] learned embeddings
// out: [num_patches, hidden_size]
void invoke_pos_embed_interp(__nv_bfloat16* out,
                              const __nv_bfloat16* embed_table,
                              int num_pos, int pos_grid_size,
                              int grid_h, int grid_w, int num_patches,
                              int hidden_size, int spatial_merge_size,
                              int patches_per_frame,
                              cudaStream_t stream);

// Replace image_pad token embeddings with vision features
// hidden: [total_tokens, hidden_size] — text embeddings
// vision_features: [num_vision_tokens, hidden_size]
// token_ids: [total_tokens] — original token IDs (on device)
// image_token_id: token ID for <|image_pad|> (248056)
// num_vision_tokens: total features available; kernel skips OOB pads (pass 0 to disable check)
void invoke_replace_image_tokens(__nv_bfloat16* hidden,
                                  const __nv_bfloat16* vision_features,
                                  const int* token_ids, int total_tokens,
                                  int image_token_id, int hidden_size,
                                  int vision_offset,
                                  int num_vision_tokens,
                                  cudaStream_t stream);

// FP32 → BF16 conversion
void invoke_f32_to_bf16(const float* src, __nv_bfloat16* dst, int n,
                         cudaStream_t stream);

// Replace video_pad token embeddings with vision features (same kernel, different token_id)
// Alias for invoke_replace_image_tokens with video_token_id=248057
inline void invoke_replace_video_tokens(__nv_bfloat16* hidden,
                                        const __nv_bfloat16* vision_features,
                                        const int* token_ids, int total_tokens,
                                        int hidden_size, int vision_offset,
                                        int num_vision_tokens,
                                        cudaStream_t stream) {
    invoke_replace_image_tokens(hidden, vision_features, token_ids,
                                total_tokens, 248057 /* video_pad_id */,
                                hidden_size, vision_offset, num_vision_tokens, stream);
}

} // namespace vision_ops

} // namespace core
} // namespace qwen_thor
