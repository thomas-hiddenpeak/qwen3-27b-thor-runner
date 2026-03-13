// tts_ops.h — TTS-specific CUDA operations
//
// Operations not available in audio_ops but needed for TTS:
//   - SiLU activation (text_projection)
//   - Add bias
//   - Top-K / Top-P sampling with temperature
//   - Repetition penalty
//   - Element-wise add (for dual-track embedding merge)
//   - Argmax (greedy decode fallback)

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace qwen_thor {
namespace tts {

// SiLU activation: output = x * sigmoid(x)
void invoke_silu(__nv_bfloat16* output, const __nv_bfloat16* input,
                 int num_elements, cudaStream_t stream = 0);

// Add bias to each row: output[i][j] += bias[j]
void invoke_add_bias(__nv_bfloat16* output, const __nv_bfloat16* bias,
                     int num_tokens, int hidden_size, cudaStream_t stream = 0);

// Element-wise add: output = a + b (in-place on a)
void invoke_add(__nv_bfloat16* a, const __nv_bfloat16* b,
                int num_elements, cudaStream_t stream = 0);

// Scale and add: output = a + scale * b
void invoke_scale_add(__nv_bfloat16* a, const __nv_bfloat16* b,
                      float scale, int num_elements, cudaStream_t stream = 0);

// Apply repetition penalty to logits (in-place)
// For each token_id in history, divide/multiply logit by penalty
void invoke_repetition_penalty(__nv_bfloat16* logits,
                               const int* token_ids, int num_tokens,
                               float penalty, cudaStream_t stream = 0);

// GPU Top-K + Top-P sampling with temperature
// Returns sampled token ID in result[0] (managed memory)
// logits: [vocab_size] BF16
// result: [1] int (managed memory, cudaMallocManaged)
void invoke_sample_top_k_top_p(__nv_bfloat16* logits,  // modified in-place (temperature scaling)
                               int vocab_size,
                               int top_k,
                               float top_p,
                               float temperature,
                               int* result,
                               unsigned long long seed,
                               cudaStream_t stream = 0);

// Greedy argmax: result[0] = argmax(logits)
void invoke_argmax(const __nv_bfloat16* logits, int vocab_size,
                   int* result, cudaStream_t stream = 0);

// Sum N embeddings into 1: output[j] = sum_{i=0}^{N-1} embeddings[i][j]
// embeddings: [N, hidden_size]
// output: [hidden_size]

// Suppress tokens: set logits[i] = -inf for i in [start, end) except keep_id
void invoke_suppress_tokens(__nv_bfloat16* logits, int start, int end,
                            int keep_id, cudaStream_t stream = 0);
void invoke_sum_embeddings(__nv_bfloat16* output,
                           const __nv_bfloat16* embeddings,
                           int num_embeddings, int hidden_size,
                           cudaStream_t stream = 0);

// GPU-resident Top-K + Top-P sampling for small vocab (≤4096)
// Performs temperature scaling, softmax, top-k, top-p, and sampling entirely on GPU.
// Result is written to device memory — no CPU sync needed.
// logits: [vocab_size] BF16 (NOT modified)
// result: [1] int (device memory)
void invoke_gpu_sample_top_k_top_p(const __nv_bfloat16* logits, int vocab_size,
                                    int top_k, float top_p, float temperature,
                                    int* result, unsigned long long seed,
                                    cudaStream_t stream = 0);

} // namespace tts
} // namespace qwen_thor
