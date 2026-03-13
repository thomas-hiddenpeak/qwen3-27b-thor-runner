// tts_ops.cu — TTS-specific CUDA kernels

#include "tts_ops.h"
#include <cuda_bf16.h>
#include <cstdio>
#include <cfloat>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>

namespace qwen_thor {
namespace tts {

// ============================================================
// SiLU: output = x * sigmoid(x)
// ============================================================
__global__ void silu_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float x = __bfloat162float(input[idx]);
        float sig = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2bfloat16(x * sig);
    }
}

void invoke_silu(__nv_bfloat16* output, const __nv_bfloat16* input,
                 int num_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(output, input, num_elements);
}

// ============================================================
// Add bias: output[i][j] += bias[j]
// ============================================================
__global__ void add_bias_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ bias,
    int num_tokens, int hidden_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * hidden_size) {
        int j = idx % hidden_size;
        float val = __bfloat162float(output[idx]) + __bfloat162float(bias[j]);
        output[idx] = __float2bfloat16(val);
    }
}

void invoke_add_bias(__nv_bfloat16* output, const __nv_bfloat16* bias,
                     int num_tokens, int hidden_size, cudaStream_t stream) {
    int total = num_tokens * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(output, bias, num_tokens, hidden_size);
}

// ============================================================
// Element-wise add: a[i] += b[i]
// ============================================================
__global__ void add_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = __bfloat162float(a[idx]) + __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(val);
    }
}

void invoke_add(__nv_bfloat16* a, const __nv_bfloat16* b,
                int num_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, num_elements);
}

// ============================================================
// Scale and add: a[i] += scale * b[i]
// ============================================================
__global__ void scale_add_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float scale, int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = __bfloat162float(a[idx]) + scale * __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(val);
    }
}

void invoke_scale_add(__nv_bfloat16* a, const __nv_bfloat16* b,
                      float scale, int num_elements, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    scale_add_kernel<<<blocks, threads, 0, stream>>>(a, b, scale, num_elements);
}

// ============================================================
// Repetition Penalty (in-place on logits)
// ============================================================
__global__ void repetition_penalty_kernel(
    __nv_bfloat16* __restrict__ logits,
    const int* __restrict__ token_ids,
    int num_tokens,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens) {
        int tid = token_ids[idx];
        float val = __bfloat162float(logits[tid]);
        // Following HuggingFace convention: divide if positive, multiply if negative
        if (val > 0.0f) {
            val /= penalty;
        } else {
            val *= penalty;
        }
        logits[tid] = __float2bfloat16(val);
    }
}

void invoke_repetition_penalty(__nv_bfloat16* logits,
                               const int* token_ids, int num_tokens,
                               float penalty, cudaStream_t stream) {
    if (num_tokens == 0 || penalty == 1.0f) return;
    int threads = 256;
    int blocks = (num_tokens + threads - 1) / threads;
    repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(logits, token_ids, num_tokens, penalty);
}

// ============================================================
// Top-K + Top-P Sampling with Temperature
// ============================================================
// Strategy: CPU-side implementation for simplicity and correctness
// (GPU sampling with top-k + top-p is complex; for TTS with vocab_size=3072,
// CPU is fast enough since it's one sample per ~28L transformer forward)

// GPU kernels for temperature scaling and softmax
__global__ void temperature_scale_kernel(
    __nv_bfloat16* __restrict__ logits,
    float inv_temperature,
    int vocab_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        float val = __bfloat162float(logits[idx]) * inv_temperature;
        logits[idx] = __float2bfloat16(val);
    }
}

// Argmax kernel: blockDim.x threads cooperate to find max
__global__ void argmax_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int vocab_size,
    int* __restrict__ result
) {
    __shared__ float s_max[32];
    __shared__ int s_idx[32];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __bfloat162float(logits[i]);
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        s_max[warp_id] = local_max;
        s_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (tid == 0) {
        float best_max = -FLT_MAX;
        int best_idx = 0;
        for (int i = 0; i < num_warps; i++) {
            if (s_max[i] > best_max) {
                best_max = s_max[i];
                best_idx = s_idx[i];
            }
        }
        result[0] = best_idx;
    }
}

void invoke_argmax(const __nv_bfloat16* logits, int vocab_size,
                   int* result, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(logits, vocab_size, result);
}

// Top-K + Top-P sampling: GPU temperature scale + CPU sort + sample
// For vocab_size=3072 (talker) or 2048 (code predictor), CPU sort is fast
void invoke_sample_top_k_top_p(__nv_bfloat16* logits, int vocab_size,
                               int top_k, float top_p, float temperature,
                               int* result, unsigned long long seed,
                               cudaStream_t stream) {
    // Step 1: Temperature scale on GPU
    if (temperature != 1.0f && temperature > 0.0f) {
        float inv_temp = 1.0f / temperature;
        int threads = 256;
        int blocks = (vocab_size + threads - 1) / threads;
        temperature_scale_kernel<<<blocks, threads, 0, stream>>>(logits, inv_temp, vocab_size);
    }

    // Step 2: Copy logits to CPU for sorting + sampling
    cudaStreamSynchronize(stream);

    std::vector<float> host_logits(vocab_size);
    std::vector<__nv_bfloat16> bf16_buf(vocab_size);
    cudaMemcpy(bf16_buf.data(), logits, vocab_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < vocab_size; i++) {
        host_logits[i] = __bfloat162float(bf16_buf[i]);
    }

    // Step 3: Find top-K indices
    struct TokenProb {
        int id;
        float logit;
    };
    std::vector<TokenProb> candidates(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        candidates[i] = {i, host_logits[i]};
    }

    // Partial sort for top-K
    int k = (top_k > 0 && top_k < vocab_size) ? top_k : vocab_size;
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
                      [](const TokenProb& a, const TokenProb& b) { return a.logit > b.logit; });

    // Step 4: Softmax over top-K
    float max_logit = candidates[0].logit;
    float sum_exp = 0.0f;
    for (int i = 0; i < k; i++) {
        candidates[i].logit = expf(candidates[i].logit - max_logit);
        sum_exp += candidates[i].logit;
    }
    for (int i = 0; i < k; i++) {
        candidates[i].logit /= sum_exp;
    }

    // Step 5: Top-P nucleus filtering
    float cum_prob = 0.0f;
    int nucleus_size = k;
    if (top_p < 1.0f) {
        for (int i = 0; i < k; i++) {
            cum_prob += candidates[i].logit;
            if (cum_prob >= top_p) {
                nucleus_size = i + 1;
                break;
            }
        }
        // Renormalize
        float renorm_sum = 0.0f;
        for (int i = 0; i < nucleus_size; i++) renorm_sum += candidates[i].logit;
        for (int i = 0; i < nucleus_size; i++) candidates[i].logit /= renorm_sum;
    }

    // Step 6: Random sample
    // Use a simple LCG from seed
    unsigned long long s = seed;
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    float r = (float)(s * 0x2545F4914F6CDD1DULL) / (float)UINT64_MAX;
    r = r < 0.0f ? -r : r;
    if (r >= 1.0f) r = 0.999f;

    cum_prob = 0.0f;
    int sampled = candidates[0].id;
    for (int i = 0; i < nucleus_size; i++) {
        cum_prob += candidates[i].logit;
        if (r < cum_prob) {
            sampled = candidates[i].id;
            break;
        }
    }

    result[0] = sampled;
}

// ============================================================
// Sum N embeddings into 1
// ============================================================
__global__ void sum_embeddings_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ embeddings,
    int num_embeddings, int hidden_size
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < hidden_size) {
        float sum = 0.0f;
        for (int i = 0; i < num_embeddings; i++) {
            sum += __bfloat162float(embeddings[i * hidden_size + j]);
        }
        output[j] = __float2bfloat16(sum);
    }
}

void invoke_sum_embeddings(__nv_bfloat16* output,
                           const __nv_bfloat16* embeddings,
                           int num_embeddings, int hidden_size,
                           cudaStream_t stream) {
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    sum_embeddings_kernel<<<blocks, threads, 0, stream>>>(
        output, embeddings, num_embeddings, hidden_size);
}

// ============================================================================
// Suppress tokens kernel
// ============================================================================

static __global__ void suppress_tokens_kernel(
    __nv_bfloat16* logits, int start, int end, int keep_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < end && i != keep_id) {
        logits[i] = __float2bfloat16(-1e9f);
    }
}

void invoke_suppress_tokens(__nv_bfloat16* logits, int start, int end,
                            int keep_id, cudaStream_t stream) {
    int n = end - start;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    suppress_tokens_kernel<<<blocks, threads, 0, stream>>>(
        logits, start, end, keep_id);
}

// ============================================================================
// GPU-Resident Top-K/Top-P Sampling for Small Vocab (≤4096)
//
// Performs temperature scaling, softmax, top-k selection, top-p filtering,
// and random sampling entirely on GPU — no CPU sync required.
// Result is written to device memory for direct use by subsequent kernels.
// ============================================================================

// ============================================================================
// GPU-Resident Top-K/Top-P Sampling for Small Vocab (≤4096)
//
// Parallel softmax + cooperative top-k via per-thread local maxima +
// single-thread selection from candidates. Improved RNG using SplitMix64.
// ============================================================================

__global__ void gpu_sample_top_k_top_p_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int vocab_size,
    int top_k,
    float top_p,
    float inv_temperature,
    int* __restrict__ result,
    unsigned long long seed)
{
    extern __shared__ float smem[];
    // Layout: smem[0..vocab_size-1] = probs, smem[vocab_size..] = candidates
    float* probs = smem;
    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    // Phase 1: Load logits → SMEM with temperature scaling, find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += bdim) {
        float v = __bfloat162float(logits[i]) * inv_temperature;
        probs[i] = v;
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float s_reduce[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) s_reduce[warp_id] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = s_reduce[0];
        for (int i = 1; i < (bdim + 31) / 32; i++) m = fmaxf(m, s_reduce[i]);
        s_reduce[0] = m;
    }
    __syncthreads();
    float block_max = s_reduce[0];

    // Phase 2: exp(x - max) and compute sum — use exp2f for speed
    float local_sum = 0.0f;
    const float log2e = 1.4426950408889634f;
    for (int i = tid; i < vocab_size; i += bdim) {
        float v = exp2f((probs[i] - block_max) * log2e);
        probs[i] = v;
        local_sum += v;
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if (lane_id == 0) s_reduce[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int i = 0; i < (bdim + 31) / 32; i++) s += s_reduce[i];
        s_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_reduce[0];

    // Phase 3: Normalize to probabilities
    for (int i = tid; i < vocab_size; i += bdim) {
        probs[i] *= inv_sum;
    }
    __syncthreads();

    // Phase 4: Cooperative top-k candidate gathering
    // Each thread finds its local top candidates, thread 0 merges
    int k = top_k;
    if (k <= 0 || k > vocab_size) k = vocab_size;
    if (k > 128) k = 128;

    // Each thread tracks its top-2 local maxima per pass
    // Thread 0 collects from all threads' ranges
    if (tid == 0) {
        // Use a simple but correct approach: partial selection sort
        // For k<=128, vocab<=4096: k*V/bdim iterations per thread
        // With 256 threads and V=3072, each thread handles 12 elements
        float top_probs[128];
        int top_ids[128];

        // Parallel-friendly: each iteration finds global max, marks it used
        for (int j = 0; j < k; j++) {
            float best = -1.0f;
            int best_id = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (probs[i] > best) {
                    best = probs[i];
                    best_id = i;
                }
            }
            top_probs[j] = best;
            top_ids[j] = best_id;
            probs[best_id] = -1.0f;  // mark used
        }

        // Renormalize top-k
        float renorm_sum = 0.0f;
        for (int i = 0; i < k; i++) renorm_sum += top_probs[i];
        float inv_renorm = (renorm_sum > 0.0f) ? (1.0f / renorm_sum) : 1.0f;
        for (int i = 0; i < k; i++) top_probs[i] *= inv_renorm;

        // Top-P nucleus filtering
        int nucleus = k;
        if (top_p < 1.0f && top_p > 0.0f) {
            float cum = 0.0f;
            for (int i = 0; i < k; i++) {
                cum += top_probs[i];
                if (cum >= top_p) { nucleus = i + 1; break; }
            }
            // Renormalize nucleus
            renorm_sum = 0.0f;
            for (int i = 0; i < nucleus; i++) renorm_sum += top_probs[i];
            if (renorm_sum > 0.0f) {
                inv_renorm = 1.0f / renorm_sum;
                for (int i = 0; i < nucleus; i++) top_probs[i] *= inv_renorm;
            }
        }

        // SplitMix64 RNG — better distribution than xorshift
        unsigned long long s = seed;
        s += 0x9E3779B97F4A7C15ULL;
        s = (s ^ (s >> 30)) * 0xBF58476D1CE4E5B9ULL;
        s = (s ^ (s >> 27)) * 0x94D049BB133111EBULL;
        s = s ^ (s >> 31);
        // Convert to [0, 1) using upper bits for better uniformity
        float r = (float)(s >> 40) / (float)(1ULL << 24);

        float cum = 0.0f;
        int sampled = top_ids[0];
        for (int i = 0; i < nucleus; i++) {
            cum += top_probs[i];
            if (r < cum) { sampled = top_ids[i]; break; }
        }
        result[0] = sampled;
    }
}

void invoke_gpu_sample_top_k_top_p(const __nv_bfloat16* logits, int vocab_size,
                                    int top_k, float top_p, float temperature,
                                    int* result, unsigned long long seed,
                                    cudaStream_t stream) {
    float inv_temp = (temperature > 0.0f && temperature != 1.0f)
                     ? (1.0f / temperature) : 1.0f;
    int threads = 256;
    int smem_bytes = vocab_size * sizeof(float);
    gpu_sample_top_k_top_p_kernel<<<1, threads, smem_bytes, stream>>>(
        logits, vocab_size, top_k, top_p, inv_temp, result, seed);
}

} // namespace tts
} // namespace qwen_thor
