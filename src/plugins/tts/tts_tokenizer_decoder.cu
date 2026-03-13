// tts_tokenizer_decoder.cu — Speech Tokenizer Decoder Implementation
//
// Converts codec tokens [16, T] → PCM audio at 24kHz
// All computation in F32 (weights are stored in F32)

#include "tts_tokenizer_decoder.h"
#include "engine/safetensors.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace qwen_thor {
namespace tts {

// ============================================================================
// CUDA Kernels (F32)
// ============================================================================

// ---------- Codebook Embedding Lookup ----------
// For each code, look up embedding from pre-computed codebook
// codes: [T], codebook: [codebook_size, dim]
// output: [T, dim] (accumulated additively)
__global__ void codebook_lookup_add_kernel(
    float* __restrict__ output,           // [T, dim]
    const int* __restrict__ codes,        // [T]
    const float* __restrict__ codebook,   // [codebook_size, dim]
    int T, int dim)
{
    int t = blockIdx.x;
    int d = threadIdx.x;
    if (t >= T || d >= dim) return;

    int code = codes[t];
    output[t * dim + d] += codebook[code * dim + d];
}

// ---------- Matrix-Vector Product for Conv1d k=1 (pointwise) ----------
// Equivalent to MatMul: output[T, out_dim] = input[T, in_dim] @ weight.T[in_dim, out_dim]
// weight stored as [out_dim, in_dim]
// This is just a batched GEMV over T positions

// ---------- Causal Conv1d: output[out_c, t] = sum_{k,ic} weight[oc,ic,k] * input[ic, t-pad+k] + bias[oc] ----------
__global__ void causal_conv1d_kernel(
    float* __restrict__ output,           // [out_channels, T_out]
    const float* __restrict__ input,      // [in_channels, T_in]
    const float* __restrict__ weight,     // [out_channels, in_channels, kernel_size]
    const float* __restrict__ bias,       // [out_channels] or nullptr
    int in_channels, int out_channels, int kernel_size,
    int T_in, int T_out, int dilation, int stride)
{
    int oc = blockIdx.x;
    int t_out = blockIdx.y * blockDim.x + threadIdx.x;
    if (oc >= out_channels || t_out >= T_out) return;

    int eff_k = (kernel_size - 1) * dilation + 1;
    int pad = eff_k - stride;  // left padding for causal

    float sum = bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            int t_in = t_out * stride + k * dilation - pad;
            if (t_in >= 0 && t_in < T_in) {
                sum += weight[(oc * in_channels + ic) * kernel_size + k] *
                       input[ic * T_in + t_in];
            }
        }
    }
    output[oc * T_out + t_out] = sum;
}

// ---------- Depthwise Causal Conv1d (groups = channels) ----------
__global__ void depthwise_causal_conv1d_kernel(
    float* __restrict__ output,           // [channels, T_out]
    const float* __restrict__ input,      // [channels, T_in]
    const float* __restrict__ weight,     // [channels, 1, kernel_size]
    const float* __restrict__ bias,       // [channels] or nullptr
    int channels, int kernel_size, int T_in, int T_out)
{
    int ch = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (ch >= channels || t >= T_out) return;

    int pad = kernel_size - 1;  // causal: full left padding
    float sum = bias ? bias[ch] : 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t + k - pad;
        if (t_in >= 0 && t_in < T_in) {
            sum += weight[ch * kernel_size + k] * input[ch * T_in + t_in];
        }
    }
    output[ch * T_out + t] = sum;
}

// ---------- Causal Transposed Conv1d ----------
// ConvTranspose1d with right-side cropping (causal):
// Full output T = (T_in - 1) * stride + kernel, then crop right (kernel - stride)
// Result: T_out = T_in * stride
__global__ void causal_transconv1d_kernel(
    float* __restrict__ output,           // [out_channels, T_out]
    const float* __restrict__ input,      // [in_channels, T_in]
    const float* __restrict__ weight,     // [in_channels, out_channels, kernel_size]
    const float* __restrict__ bias,       // [out_channels] or nullptr
    int in_channels, int out_channels, int kernel_size, int stride,
    int T_in, int T_out)
{
    // Each thread computes one output element
    int oc = blockIdx.x;
    int t_out = blockIdx.y * blockDim.x + threadIdx.x;
    if (oc >= out_channels || t_out >= T_out) return;

    float sum = bias ? bias[oc] : 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int k = 0; k < kernel_size; k++) {
            // Full transconv: t_out_full = t_in * stride + k
            // With right-crop, t_out_full == t_out for the first T_out positions
            int t_in_x_stride = t_out - k;
            if (t_in_x_stride >= 0 && t_in_x_stride % stride == 0) {
                int t_in = t_in_x_stride / stride;
                if (t_in >= 0 && t_in < T_in) {
                    sum += weight[(ic * out_channels + oc) * kernel_size + k] *
                           input[ic * T_in + t_in];
                }
            }
        }
    }
    output[oc * T_out + t_out] = sum;
}

// ---------- SnakeBeta activation ----------
// x + (1/(exp(beta) + eps)) * sin²(x * exp(alpha))
__global__ void snake_beta_kernel(
    float* __restrict__ x,                // [channels, T] — in-place
    const float* __restrict__ alpha,      // [channels]
    const float* __restrict__ beta,       // [channels]
    int channels, int T)
{
    int ch = blockIdx.x;
    int t = blockIdx.y * blockDim.x + threadIdx.x;
    if (ch >= channels || t >= T) return;

    float a = expf(alpha[ch]);
    float b = expf(beta[ch]);
    float val = x[ch * T + t];
    float s = sinf(val * a);
    x[ch * T + t] = val + (1.0f / (b + 1e-9f)) * s * s;
}

// ---------- RMSNorm (F32) ----------
// output[i] = input[i] / rms * weight[i]
__global__ void rms_norm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int dim, float eps)
{
    // One block per token
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = input[blockIdx.x * dim + i];
        sum_sq += v * v;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        output[blockIdx.x * dim + i] = input[blockIdx.x * dim + i] * rms * weight[i];
    }
}

// ---------- LayerNorm (F32) ----------
__global__ void layer_norm_f32_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int dim, float eps)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int token = blockIdx.x;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += input[token * dim + i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean = sdata[0] / dim;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float d = input[token * dim + i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(sdata[0] / dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        float val = (input[token * dim + i] - mean) * inv_std;
        output[token * dim + i] = val * weight[i] + (bias ? bias[i] : 0.0f);
    }
}

// ---------- GELU activation ----------
__global__ void gelu_kernel(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// ---------- SiLU-Gate (SwiGLU): output = silu(gate) * up ----------
__global__ void silu_gate_kernel(
    float* __restrict__ output,           // [T, dim]
    const float* __restrict__ gate,       // [T, dim]
    const float* __restrict__ up,         // [T, dim]
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    output[i] = (g / (1.0f + expf(-g))) * up[i];
}

// ---------- Layer scale: x *= scale ----------
__global__ void layer_scale_kernel(
    float* __restrict__ x,                // [T, dim]
    const float* __restrict__ scale,      // [dim]
    int T, int dim)
{
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= dim) return;
    x[t * dim + d] *= scale[d];
}

// ---------- Residual add ----------
__global__ void add_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

// ---------- Add bias ----------
__global__ void add_bias_f32_kernel(
    float* __restrict__ x,                // [T, dim]
    const float* __restrict__ bias,       // [dim]
    int T, int dim)
{
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= dim) return;
    x[t * dim + d] += bias[d];
}

// ---------- Clamp ----------
__global__ void clamp_kernel(float* __restrict__ x, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = fminf(fmaxf(x[i], lo), hi);
}

// ---------- RoPE for pre-transformer ----------
// Q/K: [T, num_heads * head_dim] row-major, apply RoPE with theta=10000
// Standard half-rotation: rotate (x[2i], x[2i+1]) pairs
__global__ void rope_f32_kernel(
    float* __restrict__ q,                // [T, total_dim]
    float* __restrict__ k,                // [T, total_dim]
    int T, int total_dim, int head_dim, float theta)
{
    int t = blockIdx.x;
    int pair = blockIdx.y * blockDim.x + threadIdx.x;  // pair index within head
    int half_hd = head_dim / 2;
    int num_pairs = total_dim / 2;  // across all heads
    if (t >= T || pair >= num_pairs) return;

    // Which head and which pair within that head
    int head = pair / half_hd;
    int pair_in_head = pair % half_hd;

    float freq = 1.0f / powf(theta, (float)(2 * pair_in_head) / head_dim);
    float angle = (float)t * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    int idx0 = t * total_dim + head * head_dim + 2 * pair_in_head;
    int idx1 = idx0 + 1;

    // Apply to Q
    float q0 = q[idx0], q1 = q[idx1];
    q[idx0] = q0 * cos_a - q1 * sin_a;
    q[idx1] = q0 * sin_a + q1 * cos_a;

    // Apply to K
    float k0 = k[idx0], k1 = k[idx1];
    k[idx0] = k0 * cos_a - k1 * sin_a;
    k[idx1] = k0 * sin_a + k1 * cos_a;
}

// ---------- Softmax with causal + sliding window mask ----------
// scores: [num_heads, T, T], mask based on sliding_window
__global__ void masked_softmax_kernel(
    float* __restrict__ scores,           // [num_heads, T, T]
    int T, int sliding_window)
{
    int h = blockIdx.x;
    int row = blockIdx.y;
    if (row >= T) return;

    float* row_ptr = scores + (h * T + row) * T;

    // Apply mask: set positions to -inf where col > row (causal) or col < row - sliding_window + 1
    int min_col = (sliding_window > 0) ? max(0, row - sliding_window + 1) : 0;
    float max_val = -1e30f;
    for (int c = 0; c < T; c++) {
        if (c > row || c < min_col) {
            row_ptr[c] = -1e30f;
        }
        max_val = fmaxf(max_val, row_ptr[c]);
    }

    // Softmax
    float sum = 0.0f;
    for (int c = 0; c < T; c++) {
        row_ptr[c] = expf(row_ptr[c] - max_val);
        sum += row_ptr[c];
    }
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int c = 0; c < T; c++) {
        row_ptr[c] *= inv_sum;
    }
}

// ---------- Transpose [channels, T] ↔ [T, channels] ----------
__global__ void transpose_2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int rows, int cols)
{
    int r = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    output[c * rows + r] = input[r * cols + c];
}

// ---------- Set buffer to zero ----------
__global__ void zero_kernel(float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.0f;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

SpeechTokenizerDecoder::SpeechTokenizerDecoder() = default;

SpeechTokenizerDecoder::~SpeechTokenizerDecoder() {
    for (auto p : device_ptrs_) cudaFree(p);
    if (workspace_) cudaFree(workspace_);
    if (cublas_) cublasDestroy(cublas_);
}

// ============================================================================
// Weight Loading
// ============================================================================

bool SpeechTokenizerDecoder::load_weights(
    const std::string& tokenizer_dir,
    const TokenizerDecoderConfig& config)
{
    config_ = config;
    using namespace qwen_thor::io;

    std::string dir = tokenizer_dir;
    if (dir.back() != '/') dir += '/';

    std::string model_path = dir + "model.safetensors";
    SafetensorsLoader loader(model_path);

    auto load_f32 = [&](const std::string& name) -> float* {
        if (!loader.has_tensor(name)) {
            fprintf(stderr, "[TokenizerDecoder] WARNING: tensor '%s' not found\n", name.c_str());
            return nullptr;
        }
        auto tensor = loader.get_tensor(name);
        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, tensor->nbytes());
        cudaMemcpy(d_ptr, tensor->data(), tensor->nbytes(), cudaMemcpyHostToDevice);
        device_ptrs_.push_back(d_ptr);
        return reinterpret_cast<float*>(d_ptr);
    };

    // Helper to pre-compute codebook: embed = embed_sum / cluster_usage
    auto load_codebook = [&](const std::string& prefix) -> float* {
        auto embed_sum_tensor = loader.get_tensor(prefix + ".embedding_sum");
        auto usage_tensor = loader.get_tensor(prefix + ".cluster_usage");
        if (!embed_sum_tensor || !usage_tensor) return nullptr;

        int codebook_size = embed_sum_tensor->shape()[0];
        int dim = embed_sum_tensor->shape()[1];
        size_t nbytes = codebook_size * dim * sizeof(float);

        // Pre-compute on CPU
        const float* embed_sum = reinterpret_cast<const float*>(embed_sum_tensor->data());
        const float* usage = reinterpret_cast<const float*>(usage_tensor->data());
        std::vector<float> codebook(codebook_size * dim);
        for (int i = 0; i < codebook_size; i++) {
            float u = std::max(usage[i], 1e-5f);
            for (int j = 0; j < dim; j++) {
                codebook[i * dim + j] = embed_sum[i * dim + j] / u;
            }
        }

        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, nbytes);
        cudaMemcpy(d_ptr, codebook.data(), nbytes, cudaMemcpyHostToDevice);
        device_ptrs_.push_back(d_ptr);
        return reinterpret_cast<float*>(d_ptr);
    };

    fprintf(stderr, "[TokenizerDecoder] Loading weights from %s\n", model_path.c_str());

    // ===== RVQ Codebooks =====
    semantic_codebook_ = load_codebook("decoder.quantizer.rvq_first.vq.layers.0._codebook");
    if (!semantic_codebook_) {
        fprintf(stderr, "[TokenizerDecoder] ERROR: missing semantic codebook\n");
        return false;
    }

    // Acoustic codebooks: 15 layers, stored contiguously
    int num_acoustic = config_.num_quantizers - config_.num_semantic_quantizers;  // 15
    size_t cb_size = config_.codebook_size * (config_.codebook_dim / 2) * sizeof(float);
    cudaMalloc(&acoustic_codebooks_, num_acoustic * cb_size);
    device_ptrs_.push_back(acoustic_codebooks_);

    for (int i = 0; i < num_acoustic; i++) {
        std::string prefix = "decoder.quantizer.rvq_rest.vq.layers." + std::to_string(i) + "._codebook";
        float* cb = load_codebook(prefix);
        if (cb) {
            cudaMemcpy(acoustic_codebooks_ + i * config_.codebook_size * (config_.codebook_dim / 2),
                       cb, cb_size, cudaMemcpyDeviceToDevice);
            // The codebook was already added to device_ptrs_ by load_codebook
        }
    }

    // Output projections (Conv1d k=1, stored as [out, in, 1])
    semantic_output_proj_w_ = load_f32("decoder.quantizer.rvq_first.output_proj.weight");
    acoustic_output_proj_w_ = load_f32("decoder.quantizer.rvq_rest.output_proj.weight");

    // ===== Pre-conv =====
    pre_conv_w_ = load_f32("decoder.pre_conv.conv.weight");
    pre_conv_b_ = load_f32("decoder.pre_conv.conv.bias");

    // ===== Pre-transformer =====
    pt_input_proj_w_ = load_f32("decoder.pre_transformer.input_proj.weight");
    pt_input_proj_b_ = load_f32("decoder.pre_transformer.input_proj.bias");
    pt_output_proj_w_ = load_f32("decoder.pre_transformer.output_proj.weight");
    pt_output_proj_b_ = load_f32("decoder.pre_transformer.output_proj.bias");
    pt_norm_w_ = load_f32("decoder.pre_transformer.norm.weight");

    for (int i = 0; i < config_.num_hidden_layers; i++) {
        std::string p = "decoder.pre_transformer.layers." + std::to_string(i) + ".";
        auto& lw = pt_layers_[i];
        lw.input_layernorm_w = load_f32(p + "input_layernorm.weight");
        lw.q_proj_w = load_f32(p + "self_attn.q_proj.weight");
        lw.k_proj_w = load_f32(p + "self_attn.k_proj.weight");
        lw.v_proj_w = load_f32(p + "self_attn.v_proj.weight");
        lw.o_proj_w = load_f32(p + "self_attn.o_proj.weight");
        lw.attn_layer_scale = load_f32(p + "self_attn_layer_scale.scale");
        lw.post_attention_layernorm_w = load_f32(p + "post_attention_layernorm.weight");
        lw.gate_proj_w = load_f32(p + "mlp.gate_proj.weight");
        lw.up_proj_w = load_f32(p + "mlp.up_proj.weight");
        lw.down_proj_w = load_f32(p + "mlp.down_proj.weight");
        lw.mlp_layer_scale = load_f32(p + "mlp_layer_scale.scale");
    }

    // ===== Upsample =====
    for (int i = 0; i < 2; i++) {
        std::string p = "decoder.upsample." + std::to_string(i) + ".";
        upsample_[i].transconv_w = load_f32(p + "0.conv.weight");
        upsample_[i].transconv_b = load_f32(p + "0.conv.bias");
        upsample_[i].convnext.dwconv_w = load_f32(p + "1.dwconv.conv.weight");
        upsample_[i].convnext.dwconv_b = load_f32(p + "1.dwconv.conv.bias");
        upsample_[i].convnext.norm_w = load_f32(p + "1.norm.weight");
        upsample_[i].convnext.norm_b = load_f32(p + "1.norm.bias");
        upsample_[i].convnext.pwconv1_w = load_f32(p + "1.pwconv1.weight");
        upsample_[i].convnext.pwconv1_b = load_f32(p + "1.pwconv1.bias");
        upsample_[i].convnext.pwconv2_w = load_f32(p + "1.pwconv2.weight");
        upsample_[i].convnext.pwconv2_b = load_f32(p + "1.pwconv2.bias");
        upsample_[i].convnext.gamma = load_f32(p + "1.gamma");
    }

    // ===== BigVGAN Decoder =====
    initial_conv_w_ = load_f32("decoder.decoder.0.conv.weight");
    initial_conv_b_ = load_f32("decoder.decoder.0.conv.bias");

    // 4 stages (decoder.decoder.{1,2,3,4})
    for (int stage = 0; stage < 4; stage++) {
        int idx = stage + 1;
        std::string p = "decoder.decoder." + std::to_string(idx) + ".block.";
        auto& sw = decoder_stages_[stage];
        sw.snake_alpha = load_f32(p + "0.alpha");
        sw.snake_beta = load_f32(p + "0.beta");
        sw.transconv_w = load_f32(p + "1.conv.weight");
        sw.transconv_b = load_f32(p + "1.conv.bias");

        // 3 ResBlocks (indices 2, 3, 4)
        for (int r = 0; r < 3; r++) {
            std::string rp = p + std::to_string(r + 2) + ".";
            sw.res_blocks[r].act1_alpha = load_f32(rp + "act1.alpha");
            sw.res_blocks[r].act1_beta = load_f32(rp + "act1.beta");
            sw.res_blocks[r].conv1_w = load_f32(rp + "conv1.conv.weight");
            sw.res_blocks[r].conv1_b = load_f32(rp + "conv1.conv.bias");
            sw.res_blocks[r].act2_alpha = load_f32(rp + "act2.alpha");
            sw.res_blocks[r].act2_beta = load_f32(rp + "act2.beta");
            sw.res_blocks[r].conv2_w = load_f32(rp + "conv2.conv.weight");
            sw.res_blocks[r].conv2_b = load_f32(rp + "conv2.conv.bias");
        }
    }

    // Final SnakeBeta + Conv
    final_snake_alpha_ = load_f32("decoder.decoder.5.alpha");
    final_snake_beta_ = load_f32("decoder.decoder.5.beta");
    final_conv_w_ = load_f32("decoder.decoder.6.conv.weight");
    final_conv_b_ = load_f32("decoder.decoder.6.conv.bias");

    fprintf(stderr, "[TokenizerDecoder] Loaded %zu weight tensors\n", device_ptrs_.size());
    loaded_ = true;
    return true;
}

// ============================================================================
// Initialize
// ============================================================================

void SpeechTokenizerDecoder::initialize(cudaStream_t stream) {
    cublasCreate(&cublas_);

    // Workspace: allocate enough for the largest intermediate tensor
    // For chunk_size=300 + context=25 = 325 frames:
    // After stage 4 of BigVGAN: [96, 325*480] = 96 * 156000 * 4 bytes ≈ 60 MB
    // We need multiple buffers (double-buffering), so allocate generously
    int max_frames = config_.chunk_size + config_.left_context_size;
    // Largest intermediate: decoder_dim * max_frames * 4 (after upsample 2x2)
    size_t largest = (size_t)config_.decoder_dim * max_frames * 4 * sizeof(float);  // after upsample 2x2
    // Add extra for all intermediate buffers
    workspace_size_ = largest * 4;  // generous
    if (workspace_size_ < 256 * 1024 * 1024)
        workspace_size_ = 256 * 1024 * 1024;  // minimum 256 MB

    cudaMalloc(&workspace_, workspace_size_);
    fprintf(stderr, "[TokenizerDecoder] Workspace: %.1f MB\n", workspace_size_ / (1024.0f * 1024.0f));
}

// Debug: print tensor stats (min, max, mean, first few values)
// Disabled in release builds — each call adds ~1ms from cudaStreamSynchronize + D2H copy
#ifdef TTS_DEBUG_TENSORS
static void debug_tensor(const char* name, const float* d_ptr, int n, cudaStream_t s) {
    std::vector<float> h(std::min(n, 10));
    cudaMemcpyAsync(h.data(), d_ptr, h.size() * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    // Also get full stats
    std::vector<float> full(n);
    cudaMemcpy(full.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    float mn = full[0], mx = full[0], sum = 0;
    int nan_count = 0;
    for (int i = 0; i < n; i++) {
        if (std::isnan(full[i])) { nan_count++; continue; }
        if (full[i] < mn) mn = full[i];
        if (full[i] > mx) mx = full[i];
        sum += full[i];
    }
    fprintf(stderr, "[DEBUG] %s: n=%d min=%.6f max=%.6f mean=%.6f nans=%d first=[",
            name, n, mn, mx, sum/n, nan_count);
    for (int i = 0; i < (int)h.size(); i++) fprintf(stderr, "%.4f%s", h[i], i < (int)h.size()-1 ? "," : "");
    fprintf(stderr, "]\n");
}
#else
static inline void debug_tensor(const char*, const float*, int, cudaStream_t) {}
#endif

// ============================================================================
// RVQ Dequantization
// ============================================================================

void SpeechTokenizerDecoder::rvq_dequant(
    const int* d_codes, int T, float* d_output, cudaStream_t s)
{
    int vq_dim = config_.codebook_dim / 2;  // 256
    int cb_dim = config_.codebook_dim;      // 512

    // Temporary for accumulated vq embeddings
    float* sem_accum = workspace_;                       // [T, 256]
    float* aco_accum = sem_accum + T * vq_dim;           // [T, 256]
    float* sem_proj = aco_accum + T * vq_dim;            // [T, 512]
    float* aco_proj = sem_proj + T * cb_dim;             // [T, 512]

    // Zero accumulators
    int n_sem = T * vq_dim;
    int n_aco = T * vq_dim;
    zero_kernel<<<(n_sem + 255) / 256, 256, 0, s>>>(sem_accum, n_sem);
    zero_kernel<<<(n_aco + 255) / 256, 256, 0, s>>>(aco_accum, n_aco);

    // Semantic (1 quantizer): codes[0, :]
    codebook_lookup_add_kernel<<<T, vq_dim, 0, s>>>(
        sem_accum, d_codes, semantic_codebook_, T, vq_dim);

    // Acoustic (15 quantizers): codes[1..15, :]
    int num_acoustic = config_.num_quantizers - config_.num_semantic_quantizers;
    for (int q = 0; q < num_acoustic; q++) {
        float* cb = acoustic_codebooks_ + q * config_.codebook_size * vq_dim;
        codebook_lookup_add_kernel<<<T, vq_dim, 0, s>>>(
            aco_accum, d_codes + (q + 1) * T, cb, T, vq_dim);
    }

    // Apply output projections (Conv1d k=1 = matmul)
    // sem_proj [T, 512] = sem_accum [T, 256] @ semantic_output_proj_w_.T [256, 512]
    // Weight is stored as [512, 256, 1], treat as [512, 256]
    float alpha = 1.0f, beta_val = 0.0f;
    cublasSetStream(cublas_, s);
    // cuBLAS: C[T,512] = A[T,256] * B[256,512]
    // B is [512, 256] row-major = [256, 512] col-major
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                cb_dim, T, vq_dim,
                &alpha,
                semantic_output_proj_w_, vq_dim,
                sem_accum, vq_dim,
                &beta_val,
                sem_proj, cb_dim);

    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                cb_dim, T, vq_dim,
                &alpha,
                acoustic_output_proj_w_, vq_dim,
                aco_accum, vq_dim,
                &beta_val,
                aco_proj, cb_dim);

    // Add: output = sem_proj + aco_proj → [T, 512]
    // Then transpose to [512, T] for conv pipeline
    int total = T * cb_dim;
    add_kernel<<<(total + 255) / 256, 256, 0, s>>>(sem_proj, aco_proj, total);

    // Transpose [T, 512] → [512, T]
    dim3 tg(T, (cb_dim + 255) / 256);
    transpose_2d_kernel<<<tg, 256, 0, s>>>(d_output, sem_proj, T, cb_dim);
}

// ============================================================================
// Pre-conv: CausalConv1d(512→1024, k=3)
// ============================================================================

void SpeechTokenizerDecoder::run_pre_conv(
    float* input, int T, float* output, cudaStream_t s)
{
    int in_c = config_.codebook_dim;   // 512
    int out_c = config_.latent_dim;    // 1024
    int k = 3;
    dim3 grid(out_c, (T + 255) / 256);
    causal_conv1d_kernel<<<grid, 256, 0, s>>>(
        output, input, pre_conv_w_, pre_conv_b_,
        in_c, out_c, k, T, T, /*dilation=*/1, /*stride=*/1);
}

// ============================================================================
// Pre-transformer Forward
// ============================================================================

void SpeechTokenizerDecoder::transformer_layer_forward(
    const TokenizerTransformerLayerWeights& w,
    float* hidden, int T, float* workspace, cudaStream_t s)
{
    int h = config_.hidden_size;         // 512
    int num_heads = config_.num_attention_heads; // 16
    int head_dim = config_.head_dim;     // 64
    int inter = config_.intermediate_size; // 1024
    int sw = config_.sliding_window;     // 72
    int total_qkv_dim = num_heads * head_dim; // 1024

    // Workspace layout
    float* norm_out = workspace;                  // [T, h]
    float* q = norm_out + T * h;                  // [T, total_qkv_dim]
    float* k = q + T * total_qkv_dim;             // [T, total_qkv_dim]
    float* v = k + T * total_qkv_dim;             // [T, total_qkv_dim]
    float* attn_scores = v + T * total_qkv_dim;   // [num_heads, T, T]
    float* attn_out = attn_scores + (size_t)num_heads * T * T;  // [T, total_qkv_dim]
    float* o_proj_out = attn_out + T * total_qkv_dim; // [T, h]
    float* gate_out = o_proj_out + T * h;          // [T, inter]
    float* up_out = gate_out + T * inter;          // [T, inter]

    float alpha = 1.0f, beta_val = 0.0f;
    cublasSetStream(cublas_, s);

    // 1. LayerNorm
    int block_dim = std::min(h, 1024);
    rms_norm_f32_kernel<<<T, block_dim, block_dim * sizeof(float), s>>>(
        norm_out, hidden, w.input_layernorm_w, h, config_.rms_norm_eps);

    // 2. QKV projections: [T, h] → [T, total_qkv_dim]
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                total_qkv_dim, T, h, &alpha,
                w.q_proj_w, h, norm_out, h, &beta_val, q, total_qkv_dim);
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                total_qkv_dim, T, h, &alpha,
                w.k_proj_w, h, norm_out, h, &beta_val, k, total_qkv_dim);
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                total_qkv_dim, T, h, &alpha,
                w.v_proj_w, h, norm_out, h, &beta_val, v, total_qkv_dim);

    // 3. Apply RoPE (standard half-rotation with theta=10000)
    int num_pairs = total_qkv_dim / 2;
    dim3 rope_grid(T, (num_pairs + 255) / 256);
    rope_f32_kernel<<<rope_grid, 256, 0, s>>>(
        q, k, T, total_qkv_dim, head_dim, config_.rope_theta);

    // 4. Attention: scores = Q @ K.T / sqrt(head_dim)
    // Reshape Q, K, V: [T, num_heads, head_dim] → [num_heads, T, head_dim] (batch GEMM)
    // Use cublasSgemmStridedBatched
    float scale = 1.0f / sqrtf((float)head_dim);
    // Q is [T, num_heads * head_dim] in row-major = num_heads groups of [T, head_dim]
    // We need [num_heads, T, head_dim]. The data is already interleaved if viewed as
    // [T, num_heads, head_dim]. We need to transpose first two dims.
    // Let's transpose Q/K/V to [num_heads, T, head_dim]
    // Q[t, h, d] = q[t * total_qkv_dim + h * head_dim + d]
    // We want Q_t[h, t, d] = q[t * total_qkv_dim + h * head_dim + d]
    // This is a stride: for head h, stride between t positions = total_qkv_dim

    // cublasSgemmStridedBatched can handle this with appropriate strides
    // A = Q, shape [T, head_dim] per head, stride = total_qkv_dim (between T rows)
    // Actually, Q is T×total_qkv_dim row-major. For head h, elements at positions
    // Q[t][h*head_dim ... h*head_dim+head_dim-1]
    // In column-major for cuBLAS: Q col-major is [total_qkv_dim, T]
    // Head h data: starts at offset h*head_dim, stride total_qkv_dim per column

    // scores[h, i, j] = sum_d Q[i, h, d] * K[j, h, d] * scale
    // = (Q_h @ K_h^T) * scale where Q_h is [T, head_dim], K_h is [T, head_dim]

    // Using SgemmStridedBatched:
    // C[h] = Q_h @ K_h^T, C is [T, T], Q_h is [T, head_dim], K_h is [T, head_dim]
    // cuBLAS col-major: C = alpha * op(A) * op(B) + beta * C
    // A = K_h^T of shape [head_dim, T] → op(A) = K_h^T → CUBLAS_OP_T, A is [T, head_dim] col-major
    // B = Q_h of shape [T, head_dim] → op(B) = Q_h → CUBLAS_OP_N as [head_dim, T] col-major?
    // Hmm, this is getting confusing with strides. Let me just use a simple approach.

    // Simple approach: reorganize Q/K/V to [num_heads, T, head_dim] contiguously
    // Then use batched GEMM
    float* Q_reorg = up_out + T * inter;  // [num_heads, T, head_dim]
    float* K_reorg = Q_reorg + (size_t)num_heads * T * head_dim;
    float* V_reorg = K_reorg + (size_t)num_heads * T * head_dim;
    // Reorganize: Q_reorg[h, t, d] = q[t * total_qkv_dim + h * head_dim + d]
    // This is a transpose of dim 0 and 1 with dim 2 staying
    for (int h_idx = 0; h_idx < num_heads; h_idx++) {
        // Copy head h_idx data: for each t, copy head_dim elements
        // src stride: total_qkv_dim, dst stride: head_dim
        // Use cudaMemcpy2D
        cudaMemcpy2DAsync(
            Q_reorg + h_idx * T * head_dim, head_dim * sizeof(float),
            q + h_idx * head_dim, total_qkv_dim * sizeof(float),
            head_dim * sizeof(float), T,
            cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(
            K_reorg + h_idx * T * head_dim, head_dim * sizeof(float),
            k + h_idx * head_dim, total_qkv_dim * sizeof(float),
            head_dim * sizeof(float), T,
            cudaMemcpyDeviceToDevice, s);
        cudaMemcpy2DAsync(
            V_reorg + h_idx * T * head_dim, head_dim * sizeof(float),
            v + h_idx * head_dim, total_qkv_dim * sizeof(float),
            head_dim * sizeof(float), T,
            cudaMemcpyDeviceToDevice, s);
    }

    // Batched GEMM: scores[h] = Q_h @ K_h^T, [T, T] = [T, head_dim] @ [head_dim, T]
    // In col-major: C[T,T] = A^T[T,head_dim] * B[head_dim,T]... no
    // Col-major: Q_reorg is [head_dim, T] per head (since row-major [T, head_dim] = col-major [head_dim, T])
    // scores = Q @ K^T in row-major → col-major: scores = K @ Q^T
    // C[T,T] col = K[head_dim,T] transposed × Q[head_dim,T]
    // → C = alpha * op(A) * op(B) where A=K, B=Q
    // A=K col-major [head_dim, T], op=T → [T, head_dim]
    // B=Q col-major [head_dim, T], op=N → [head_dim, T]
    // C = [T, head_dim] × [head_dim, T]? That's [T,T]. Yes!
    // Wait: C[m,n] = alpha * op(A)[m,k] * op(B)[k,n]
    // m=T, n=T, k=head_dim
    // op(A) = A^T [T, head_dim], so A is [head_dim, T], op=T
    // op(B) = B [head_dim, T], so B is [head_dim, T], op=N
    // C = A^T[T,head_dim] * B[head_dim, T] = [T, T] ✓

    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_T, CUBLAS_OP_N,
        T, T, head_dim,
        &scale,
        K_reorg, head_dim, T * head_dim,   // A = K, [head_dim, T] col-major
        Q_reorg, head_dim, T * head_dim,   // B = Q, [head_dim, T] col-major
        &beta_val,
        attn_scores, T, T * T,             // C = scores, [T, T] col-major
        num_heads);

    // 5. Apply causal + sliding window mask + softmax
    dim3 sm_grid(num_heads, T);
    masked_softmax_kernel<<<sm_grid, 1, 0, s>>>(attn_scores, T, sw);

    // 6. Attention output: attn_out[h] = scores[h] @ V[h], [T, head_dim]
    // Col-major: C[head_dim, T] = V[head_dim, T] * scores[T, T]
    // C = alpha * V * scores
    float one = 1.0f;
    cublasSgemmStridedBatched(cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, T, T,
        &one,
        V_reorg, head_dim, T * head_dim,
        attn_scores, T, T * T,
        &beta_val,
        Q_reorg, head_dim, T * head_dim,  // reuse Q_reorg for output
        num_heads);

    // 7. Reorganize back to [T, total_qkv_dim] and apply o_proj
    for (int h_idx = 0; h_idx < num_heads; h_idx++) {
        cudaMemcpy2DAsync(
            attn_out + h_idx * head_dim, total_qkv_dim * sizeof(float),
            Q_reorg + h_idx * T * head_dim, head_dim * sizeof(float),
            head_dim * sizeof(float), T,
            cudaMemcpyDeviceToDevice, s);
    }

    // o_proj: [T, total_qkv_dim] → [T, h]
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                h, T, total_qkv_dim, &alpha,
                w.o_proj_w, total_qkv_dim, attn_out, total_qkv_dim,
                &beta_val, o_proj_out, h);

    // 8. Layer scale + residual
    dim3 ls_grid(T, (h + 255) / 256);
    layer_scale_kernel<<<ls_grid, 256, 0, s>>>(o_proj_out, w.attn_layer_scale, T, h);
    add_kernel<<<(T * h + 255) / 256, 256, 0, s>>>(hidden, o_proj_out, T * h);

    // 9. Post-attention norm
    rms_norm_f32_kernel<<<T, block_dim, block_dim * sizeof(float), s>>>(
        norm_out, hidden, w.post_attention_layernorm_w, h, config_.rms_norm_eps);

    // 10. SwiGLU MLP: gate = norm @ gate_proj, up = norm @ up_proj
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                inter, T, h, &alpha,
                w.gate_proj_w, h, norm_out, h, &beta_val, gate_out, inter);
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                inter, T, h, &alpha,
                w.up_proj_w, h, norm_out, h, &beta_val, up_out, inter);

    // SiLU-gate
    silu_gate_kernel<<<(T * inter + 255) / 256, 256, 0, s>>>(
        gate_out, gate_out, up_out, T * inter);

    // down_proj: [T, inter] → [T, h]
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                h, T, inter, &alpha,
                w.down_proj_w, inter, gate_out, inter, &beta_val, o_proj_out, h);

    // 11. MLP layer scale + residual
    layer_scale_kernel<<<ls_grid, 256, 0, s>>>(o_proj_out, w.mlp_layer_scale, T, h);
    add_kernel<<<(T * h + 255) / 256, 256, 0, s>>>(hidden, o_proj_out, T * h);
}

void SpeechTokenizerDecoder::run_pre_transformer(
    float* input, int T, float* output, cudaStream_t s)
{
    int latent = config_.latent_dim;   // 1024
    int h = config_.hidden_size;       // 512

    // input is [latent_dim, T] channel-first → need [T, latent_dim] for transformer
    // Use workspace for transformer hidden states
    float* work_base = workspace_ + T * config_.codebook_dim * 4; // skip RVQ area
    float* hidden_tl = work_base;                    // [T, latent_dim]
    float* hidden_h = hidden_tl + T * latent;        // [T, h]
    float* layer_ws = hidden_h + T * h;              // workspace for layers

    // Transpose input [latent, T] → [T, latent]
    dim3 tg(latent, (T + 255) / 256);
    transpose_2d_kernel<<<tg, 256, 0, s>>>(hidden_tl, input, latent, T);

    // Input projection: [T, latent] → [T, h]
    float alpha = 1.0f, beta_val = 0.0f;
    cublasSetStream(cublas_, s);
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                h, T, latent, &alpha,
                pt_input_proj_w_, latent, hidden_tl, latent,
                &beta_val, hidden_h, h);
    // Add bias
    dim3 bg(T, (h + 255) / 256);
    add_bias_f32_kernel<<<bg, 256, 0, s>>>(hidden_h, pt_input_proj_b_, T, h);

    // Run 8 transformer layers
    for (int i = 0; i < config_.num_hidden_layers; i++) {
        transformer_layer_forward(pt_layers_[i], hidden_h, T, layer_ws, s);
    }

    // Final RMSNorm
    int block_dim = std::min(h, 1024);
    rms_norm_f32_kernel<<<T, block_dim, block_dim * sizeof(float), s>>>(
        hidden_h, hidden_h, pt_norm_w_, h, config_.rms_norm_eps);

    // Output projection: [T, h] → [T, latent]
    cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                latent, T, h, &alpha,
                pt_output_proj_w_, h, hidden_h, h,
                &beta_val, hidden_tl, latent);
    add_bias_f32_kernel<<<dim3(T, (latent + 255) / 256), 256, 0, s>>>(
        hidden_tl, pt_output_proj_b_, T, latent);

    // Transpose back [T, latent] → [latent, T]
    dim3 tg2(T, (latent + 255) / 256);
    transpose_2d_kernel<<<tg2, 256, 0, s>>>(output, hidden_tl, T, latent);
}

// ============================================================================
// Upsample
// ============================================================================

void SpeechTokenizerDecoder::run_upsample(
    float* input, int T_in, float* output, int& T_out, cudaStream_t s)
{
    int dim = config_.latent_dim;  // 1024
    int T = T_in;

    // Stage 0: input → stage0_out, Stage 1: stage0_out → output
    float* stage_in = input;

    for (int stage = 0; stage < 2; stage++) {
        int factor = config_.upsampling_ratios[stage];  // 2
        int T_new = T * factor;
        size_t out_size = (size_t)dim * T_new * sizeof(float);

        // Allocate output: last stage writes to output, others to temp
        float* stage_out = nullptr;
        if (stage == 1) {
            stage_out = output;
        } else {
            cudaMalloc(&stage_out, out_size);
        }

        // ConvTranspose1d: [dim, T] → [dim, T_new], no padding (kernel=stride)
        dim3 grid(dim, (T_new + 255) / 256);
        causal_transconv1d_kernel<<<grid, 256, 0, s>>>(
            stage_out, stage_in, upsample_[stage].transconv_w,
            upsample_[stage].transconv_b,
            dim, dim, factor, factor, T, T_new);

        // ConvNeXt block: residual around the whole block
        auto& cn = upsample_[stage].convnext;

        // Allocate temp buffers for ConvNeXt
        float* dw_out = nullptr;    // [dim, T_new]
        float* dw_tl = nullptr;     // [T_new, dim]
        float* pw1_out = nullptr;   // [T_new, 4*dim]
        cudaMalloc(&dw_out, out_size);
        cudaMalloc(&dw_tl, out_size);
        cudaMalloc(&pw1_out, (size_t)4 * dim * T_new * sizeof(float));

        // Depthwise Conv1d (groups=dim, k=7)
        dim3 dw_grid(dim, (T_new + 255) / 256);
        depthwise_causal_conv1d_kernel<<<dw_grid, 256, 0, s>>>(
            dw_out, stage_out, cn.dwconv_w, cn.dwconv_b,
            dim, 7, T_new, T_new);

        // Transpose to [T, dim] for LayerNorm + pointwise
        dim3 tg(dim, (T_new + 255) / 256);
        transpose_2d_kernel<<<tg, 256, 0, s>>>(dw_tl, dw_out, dim, T_new);

        // LayerNorm
        int block_dim = std::min(dim, 1024);
        layer_norm_f32_kernel<<<T_new, block_dim, block_dim * sizeof(float), s>>>(
            dw_tl, dw_tl, cn.norm_w, cn.norm_b, dim, 1e-6f);

        // Pointwise1: [T, dim] → [T, 4*dim]
        float alpha = 1.0f, beta_val = 0.0f;
        cublasSetStream(cublas_, s);
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    4 * dim, T_new, dim, &alpha,
                    cn.pwconv1_w, dim, dw_tl, dim,
                    &beta_val, pw1_out, 4 * dim);
        dim3 bg1(T_new, (4 * dim + 255) / 256);
        add_bias_f32_kernel<<<bg1, 256, 0, s>>>(pw1_out, cn.pwconv1_b, T_new, 4 * dim);

        // GELU
        gelu_kernel<<<(T_new * 4 * dim + 255) / 256, 256, 0, s>>>(pw1_out, T_new * 4 * dim);

        // Pointwise2: [T, 4*dim] → [T, dim]
        cublasSgemm(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                    dim, T_new, 4 * dim, &alpha,
                    cn.pwconv2_w, 4 * dim, pw1_out, 4 * dim,
                    &beta_val, dw_tl, dim);
        add_bias_f32_kernel<<<dim3(T_new, (dim + 255) / 256), 256, 0, s>>>(
            dw_tl, cn.pwconv2_b, T_new, dim);

        // Layer scale (gamma)
        dim3 ls_grid(T_new, (dim + 255) / 256);
        layer_scale_kernel<<<ls_grid, 256, 0, s>>>(dw_tl, cn.gamma, T_new, dim);

        // Transpose back to [dim, T]
        dim3 tg2(T_new, (dim + 255) / 256);
        transpose_2d_kernel<<<tg2, 256, 0, s>>>(dw_out, dw_tl, T_new, dim);

        // Residual: stage_out += dw_out
        int total = dim * T_new;
        add_kernel<<<(total + 255) / 256, 256, 0, s>>>(stage_out, dw_out, total);

        // Free ConvNeXt temp buffers
        cudaFree(dw_out);
        cudaFree(dw_tl);
        cudaFree(pw1_out);

        // Free previous stage input if we allocated it
        if (stage > 0 && stage_in != input) {
            cudaFree(stage_in);
        }

        stage_in = stage_out;
        T = T_new;
    }

    T_out = T;
}

// ============================================================================
// BigVGAN Decoder
// ============================================================================

void SpeechTokenizerDecoder::run_bigvgan(
    float* input, int T_in, float* output, int& T_out, cudaStream_t s)
{
    int latent = config_.latent_dim;     // 1024
    int dec_dim = config_.decoder_dim;   // 1536
    int T = T_in;
    static const int dilations[3] = {1, 3, 9};

    // Use dynamic allocation per stage to avoid buffer overflow
    // Each stage upsamples T significantly, so buffers grow

    // Initial conv: [latent=1024, T] → [dec_dim=1536, T]
    float* conv_out = nullptr;
    cudaMalloc(&conv_out, (size_t)dec_dim * T * sizeof(float));
    dim3 grid(dec_dim, (T + 255) / 256);
    causal_conv1d_kernel<<<grid, 256, 0, s>>>(
        conv_out, input, initial_conv_w_, initial_conv_b_,
        latent, dec_dim, 7, T, T, 1, 1);

    float* buf_a = conv_out;
    int in_dim = dec_dim;

    // 4 decoder stages
    for (int stage = 0; stage < 4; stage++) {
        int upsample_rate = config_.upsample_rates[stage];  // [8, 5, 4, 3]
        int out_dim = in_dim / 2;  // 1536→768→384→192→96
        int T_new = T * upsample_rate;
        int kernel = 2 * upsample_rate;
        size_t out_size = (size_t)out_dim * T_new * sizeof(float);

        // SnakeBeta activation (in-place on buf_a)
        dim3 sg(in_dim, (T + 255) / 256);
        snake_beta_kernel<<<sg, 256, 0, s>>>(
            buf_a, decoder_stages_[stage].snake_alpha,
            decoder_stages_[stage].snake_beta, in_dim, T);

        // Allocate output buffer for transposed conv
        float* buf_b = nullptr;
        cudaMalloc(&buf_b, out_size);

        // ConvTranspose1d: [in_dim, T] → [out_dim, T_new] (right-crop = kernel - stride)
        dim3 tc_grid(out_dim, (T_new + 255) / 256);
        causal_transconv1d_kernel<<<tc_grid, 256, 0, s>>>(
            buf_b, buf_a, decoder_stages_[stage].transconv_w,
            decoder_stages_[stage].transconv_b,
            in_dim, out_dim, kernel, upsample_rate, T, T_new);

        // Free input buffer (no longer needed after transconv reads it)
        cudaFree(buf_a);

        // Allocate residual and temp buffers for ResBlocks
        float* res_buf = nullptr;
        float* temp_buf = nullptr;
        cudaMalloc(&res_buf, out_size);
        cudaMalloc(&temp_buf, out_size);

        // 3 ResBlocks with dilations [1, 3, 9]
        for (int r = 0; r < 3; r++) {
            auto& rb = decoder_stages_[stage].res_blocks[r];
            int dil = dilations[r];

            // Save residual
            cudaMemcpyAsync(res_buf, buf_b, out_size, cudaMemcpyDeviceToDevice, s);

            // SnakeBeta1 (in-place on buf_b)
            dim3 sb1(out_dim, (T_new + 255) / 256);
            snake_beta_kernel<<<sb1, 256, 0, s>>>(
                buf_b, rb.act1_alpha, rb.act1_beta, out_dim, T_new);

            // Conv1 (dilated): [out_dim, T_new] → [out_dim, T_new]
            dim3 c1_grid(out_dim, (T_new + 255) / 256);
            causal_conv1d_kernel<<<c1_grid, 256, 0, s>>>(
                temp_buf, buf_b, rb.conv1_w, rb.conv1_b,
                out_dim, out_dim, 7, T_new, T_new, dil, 1);

            // SnakeBeta2 (in-place on temp_buf)
            snake_beta_kernel<<<sb1, 256, 0, s>>>(
                temp_buf, rb.act2_alpha, rb.act2_beta, out_dim, T_new);

            // Conv2 (k=1): [out_dim, T_new] → [out_dim, T_new]
            dim3 c2_grid(out_dim, (T_new + 255) / 256);
            causal_conv1d_kernel<<<c2_grid, 256, 0, s>>>(
                buf_b, temp_buf, rb.conv2_w, rb.conv2_b,
                out_dim, out_dim, 1, T_new, T_new, 1, 1);

            // Residual add
            int total = out_dim * T_new;
            add_kernel<<<(total + 255) / 256, 256, 0, s>>>(buf_b, res_buf, total);
        }

        // Free ResBlock temp buffers
        cudaFree(res_buf);
        cudaFree(temp_buf);

        // Next stage: buf_b becomes buf_a
        buf_a = buf_b;
        in_dim = out_dim;
        T = T_new;
    }

    // Final: SnakeBeta(96) + Conv1d(96→1, k=7)
    int final_dim = in_dim;  // 96
    dim3 fs_grid(final_dim, (T + 255) / 256);
    snake_beta_kernel<<<fs_grid, 256, 0, s>>>(
        buf_a, final_snake_alpha_, final_snake_beta_, final_dim, T);

    // Final conv: [96, T] → [1, T]
    dim3 fc_grid(1, (T + 255) / 256);
    causal_conv1d_kernel<<<fc_grid, 256, 0, s>>>(
        output, buf_a, final_conv_w_, final_conv_b_,
        final_dim, 1, 7, T, T, 1, 1);

    cudaFree(buf_a);

    // Clamp to [-1, 1]
    clamp_kernel<<<(T + 255) / 256, 256, 0, s>>>(output, T, -1.0f, 1.0f);

    T_out = T;
}

// ============================================================================
// Chunked Decode (main API)
// ============================================================================

std::vector<float> SpeechTokenizerDecoder::chunked_decode(
    const int* d_codes, int T, cudaStream_t s)
{
    int chunk_size = config_.chunk_size;        // 300
    int left_ctx = config_.left_context_size;   // 25
    int upsample = config_.decode_upsample_rate; // 1920

    std::vector<float> all_pcm;
    all_pcm.reserve(T * upsample);

    int start = 0;
    while (start < T) {
        int end = std::min(start + chunk_size, T);
        int ctx = (start > left_ctx) ? left_ctx : start;
        int chunk_start = start - ctx;
        int chunk_len = end - chunk_start;

        // Allocate temp for chunk codes and outputs
        int* chunk_codes = nullptr;
        cudaMalloc(&chunk_codes, config_.num_quantizers * chunk_len * sizeof(int));

        // Copy chunk codes: for each quantizer, copy chunk_len ints
        for (int q = 0; q < config_.num_quantizers; q++) {
            cudaMemcpyAsync(chunk_codes + q * chunk_len,
                            d_codes + q * T + chunk_start,
                            chunk_len * sizeof(int),
                            cudaMemcpyDeviceToDevice, s);
        }

        // RVQ dequant → [512, chunk_len]
        float* latent = nullptr;
        size_t latent_size = config_.codebook_dim * chunk_len * sizeof(float);
        cudaMalloc(&latent, latent_size);

        rvq_dequant(chunk_codes, chunk_len, latent, s);

        if (start == 0) debug_tensor("rvq_dequant", latent, std::min(config_.codebook_dim * chunk_len, 1000), s);

        // Pre-conv → [1024, chunk_len]
        float* pre_conv_out = nullptr;
        cudaMalloc(&pre_conv_out, config_.latent_dim * chunk_len * sizeof(float));
        run_pre_conv(latent, chunk_len, pre_conv_out, s);

        if (start == 0) debug_tensor("pre_conv", pre_conv_out, std::min(config_.latent_dim * chunk_len, 1000), s);

        // Pre-transformer → [1024, chunk_len]
        float* pt_out = nullptr;
        cudaMalloc(&pt_out, config_.latent_dim * chunk_len * sizeof(float));
        run_pre_transformer(pre_conv_out, chunk_len, pt_out, s);

        if (start == 0) debug_tensor("pre_transformer", pt_out, std::min(config_.latent_dim * chunk_len, 1000), s);

        // Upsample → [1024, chunk_len * 4]
        int T_up;
        float* upsample_out = nullptr;
        int T_after_up = chunk_len * 4;
        cudaMalloc(&upsample_out, (size_t)config_.latent_dim * T_after_up * sizeof(float));
        run_upsample(pt_out, chunk_len, upsample_out, T_up, s);

        if (start == 0) debug_tensor("upsample", upsample_out, std::min(config_.latent_dim * T_up, 1000), s);

        // BigVGAN → [1, T_pcm]
        int T_pcm;
        int expected_pcm = chunk_len * upsample;
        float* pcm_out = nullptr;
        cudaMalloc(&pcm_out, expected_pcm * sizeof(float));
        run_bigvgan(upsample_out, T_up, pcm_out, T_pcm, s);

        if (start == 0) debug_tensor("bigvgan_pcm", pcm_out, std::min(T_pcm, 1000), s);

        // Copy PCM to CPU, skip context portion
        int ctx_samples = ctx * upsample;
        int valid_samples = T_pcm - ctx_samples;
        if (valid_samples > 0) {
            std::vector<float> chunk_pcm(valid_samples);
            cudaMemcpyAsync(chunk_pcm.data(), pcm_out + ctx_samples,
                            valid_samples * sizeof(float),
                            cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            all_pcm.insert(all_pcm.end(), chunk_pcm.begin(), chunk_pcm.end());
        }

        // Cleanup
        cudaFree(chunk_codes);
        cudaFree(latent);
        cudaFree(pre_conv_out);
        cudaFree(pt_out);
        cudaFree(upsample_out);
        cudaFree(pcm_out);

        start = end;
    }

    return all_pcm;
}

std::vector<float> SpeechTokenizerDecoder::decode(
    const int* codes_cpu, int num_groups, int num_frames, cudaStream_t s)
{
    if (!loaded_) {
        fprintf(stderr, "[TokenizerDecoder] ERROR: not loaded\n");
        return {};
    }

    // Upload codes to GPU: [num_groups, num_frames]
    int total = num_groups * num_frames;
    int* d_codes = nullptr;
    cudaMalloc(&d_codes, total * sizeof(int));
    cudaMemcpy(d_codes, codes_cpu, total * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<float> pcm;
    if (num_frames <= config_.chunk_size + config_.left_context_size) {
        // Small enough to decode in one shot
        float* latent = nullptr;
        cudaMalloc(&latent, config_.codebook_dim * num_frames * sizeof(float));
        rvq_dequant(d_codes, num_frames, latent, s);
        debug_tensor("rvq_dequant", latent, std::min(config_.codebook_dim * num_frames, 1000), s);

        float* pre_conv_out = nullptr;
        cudaMalloc(&pre_conv_out, config_.latent_dim * num_frames * sizeof(float));
        run_pre_conv(latent, num_frames, pre_conv_out, s);
        debug_tensor("pre_conv", pre_conv_out, std::min(config_.latent_dim * num_frames, 1000), s);

        float* pt_out = nullptr;
        cudaMalloc(&pt_out, config_.latent_dim * num_frames * sizeof(float));
        run_pre_transformer(pre_conv_out, num_frames, pt_out, s);
        debug_tensor("pre_transformer", pt_out, std::min(config_.latent_dim * num_frames, 1000), s);

        int T_up;
        int T_after_up = num_frames * 4;
        float* up_out = nullptr;
        cudaMalloc(&up_out, (size_t)config_.latent_dim * T_after_up * sizeof(float));
        run_upsample(pt_out, num_frames, up_out, T_up, s);
        debug_tensor("upsample", up_out, std::min(config_.latent_dim * T_up, 1000), s);

        int T_pcm;
        int expected_pcm = num_frames * config_.decode_upsample_rate;
        float* pcm_out = nullptr;
        cudaMalloc(&pcm_out, (size_t)expected_pcm * sizeof(float));
        run_bigvgan(up_out, T_up, pcm_out, T_pcm, s);
        debug_tensor("bigvgan_pcm", pcm_out, std::min(T_pcm, 1000), s);

        pcm.resize(T_pcm);
        cudaMemcpy(pcm.data(), pcm_out, T_pcm * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(latent);
        cudaFree(pre_conv_out);
        cudaFree(pt_out);
        cudaFree(up_out);
        cudaFree(pcm_out);
    } else {
        pcm = chunked_decode(d_codes, num_frames, s);
    }

    cudaFree(d_codes);
    return pcm;
}

// ============================================================================
// WAV Writer (16-bit PCM, mono)
// ============================================================================

bool SpeechTokenizerDecoder::write_wav(
    const std::string& path, const std::vector<float>& pcm, int sample_rate)
{
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[WAV] ERROR: cannot create %s\n", path.c_str());
        return false;
    }

    int num_samples = (int)pcm.size();
    int bytes_per_sample = 2;
    int data_size = num_samples * bytes_per_sample;
    int file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t chunk_size = file_size;
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, 2, 1, f);
    uint16_t num_channels = 1;
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, 4, 1, f);
    uint32_t byte_rate = sample_rate * bytes_per_sample;
    fwrite(&byte_rate, 4, 1, f);
    uint16_t block_align = bytes_per_sample;
    fwrite(&block_align, 2, 1, f);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, 2, 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    uint32_t ds = data_size;
    fwrite(&ds, 4, 1, f);

    // Convert float [-1, 1] → int16
    for (int i = 0; i < num_samples; i++) {
        float v = pcm[i];
        v = std::max(-1.0f, std::min(1.0f, v));
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return true;
}

} // namespace tts
} // namespace qwen_thor
