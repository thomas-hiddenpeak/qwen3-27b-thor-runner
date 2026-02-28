#include "light_ops.h"
#include <cuda_bf16.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>

namespace qwen_thor {
namespace ops {

// ----------------------------------------------------------------------------
// Warp Reduce 辅助函数
// ----------------------------------------------------------------------------
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; // 假设最大 block size 为 1024 (32 warps)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // 读取 shared memory，只有第一个 warp 需要工作
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// ----------------------------------------------------------------------------
// RMSNorm Kernel (vectorized, register-cached x to avoid double-read)
// ----------------------------------------------------------------------------
// 每个 block 处理一个 token (一行)
// hidden_size 是 8 的倍数，使用 float4 向量化加载
// x 值在 Pass 1 读取后缓存在寄存器/shared memory, Pass 2 不重读
__global__ void rmsnorm_kernel(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight, float eps, int hidden_size) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const __nv_bfloat16* x_row = x + token_idx * hidden_size;
    __nv_bfloat16* out_row = out + token_idx * hidden_size;

    // 每线程处理 hidden_size/blockDim.x 个元素
    // 对于 hs=5120, 256 threads → 20 elements/thread → 可以存在寄存器中
    constexpr int MAX_REGS = 40;  // 最多缓存 40 个 float (hs≤10240, 256 threads)
    float r_cache[MAX_REGS];
    int elems_per_thread = hidden_size / blockDim.x;

    float sum_sq = 0.0f;
    
    // Pass 1: 读 x → 缓存到寄存器 + 计算平方和
    for (int e = 0; e < elems_per_thread; e++) {
        int i = tid + e * blockDim.x;
        float val = __bfloat162float(x_row[i]);
        r_cache[e] = val;
        sum_sq += val * val;
    }

    // Block 内归约求和
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float rms = sum_sq / (float)hidden_size;
        s_inv_rms = rsqrtf(rms + eps);
    }
    __syncthreads();

    // Pass 2: 用寄存器中的 x + 读 weight → 写 out (不重读 x)
    float inv_rms = s_inv_rms;
    for (int e = 0; e < elems_per_thread; e++) {
        int i = tid + e * blockDim.x;
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(r_cache[e] * inv_rms * (1.0f + w));
    }
}

void invoke_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight, float eps, int num_tokens, int hidden_size, cudaStream_t stream) {
    // 每个 token 分配一个 block，每个 block 256 个线程
    int threads = 256;
    int blocks = num_tokens;
    rmsnorm_kernel<<<blocks, threads, 0, stream>>>(out, x, weight, eps, hidden_size);
}

// ----------------------------------------------------------------------------
// RoPE Kernel
// ----------------------------------------------------------------------------
// 每个线程处理一个 head 的一对 (dim_i, dim_i+1)
__global__ void rope_kernel(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids, int num_tokens, int num_heads, int num_kv_heads, int head_dim, float base) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x * 2; // 每次处理 2 个维度

    if (dim_idx >= head_dim) return;

    int pos = pos_ids[token_idx];
    
    // 计算频率
    float inv_freq = powf(base, -((float)dim_idx / (float)head_dim));
    float freq = (float)pos * inv_freq;
    
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);

    // 处理 Q
    if (head_idx < num_heads) {
        int q_offset = token_idx * (num_heads * head_dim) + head_idx * head_dim + dim_idx;
        float q0 = __bfloat162float(q[q_offset]);
        float q1 = __bfloat162float(q[q_offset + 1]);
        
        q[q_offset]     = __float2bfloat16(q0 * cos_val - q1 * sin_val);
        q[q_offset + 1] = __float2bfloat16(q1 * cos_val + q0 * sin_val);
    }

    // 处理 K (MQA/GQA 情况下 num_kv_heads < num_heads)
    if (head_idx < num_kv_heads) {
        int k_offset = token_idx * (num_kv_heads * head_dim) + head_idx * head_dim + dim_idx;
        float k0 = __bfloat162float(k[k_offset]);
        float k1 = __bfloat162float(k[k_offset + 1]);
        
        k[k_offset]     = __float2bfloat16(k0 * cos_val - k1 * sin_val);
        k[k_offset + 1] = __float2bfloat16(k1 * cos_val + k0 * sin_val);
    }
}

void invoke_rope(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids, int num_tokens, int num_heads, int num_kv_heads, int head_dim, float base, cudaStream_t stream) {
    // block.x = token, block.y = head
    dim3 blocks(num_tokens, max(num_heads, num_kv_heads));
    // 每个线程处理 2 个维度
    dim3 threads(head_dim / 2); 
    
    rope_kernel<<<blocks, threads, 0, stream>>>(q, k, pos_ids, num_tokens, num_heads, num_kv_heads, head_dim, base);
}

// ----------------------------------------------------------------------------
// SwiGLU Kernel
// ----------------------------------------------------------------------------
// silu(x) = x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swiglu_kernel(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int total_elements) {
    int n8 = total_elements / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n8) {
        float4 g4 = reinterpret_cast<const float4*>(gate)[idx];
        float4 u4 = reinterpret_cast<const float4*>(up)[idx];
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&g4);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&u4);
        float4 o4;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 gf = __bfloat1622float2(g2[j]);
            float2 uf = __bfloat1622float2(u2[j]);
            float sg0 = silu(gf.x), sg1 = silu(gf.y);
            o2[j] = __floats2bfloat162_rn(sg0 * uf.x, sg1 * uf.y);
        }
        reinterpret_cast<float4*>(out)[idx] = o4;
    }
    // Scalar tail
    if (idx == 0) {
        for (int i = n8 * 8; i < total_elements; i++) {
            float g = __bfloat162float(gate[i]);
            float u = __bfloat162float(up[i]);
            out[i] = __float2bfloat16(silu(g) * u);
        }
    }
}

void invoke_swiglu(__nv_bfloat16* out, const __nv_bfloat16* gate, const __nv_bfloat16* up, int num_tokens, int intermediate_size, cudaStream_t stream) {
    int total_elements = num_tokens * intermediate_size;
    int threads = 256;
    int n8 = total_elements / 8;
    int blocks = (n8 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(out, gate, up, total_elements);
}

// ----------------------------------------------------------------------------
// Embedding Lookup
// ----------------------------------------------------------------------------
__global__ void embedding_lookup_kernel(__nv_bfloat16* out, const int* tokens, const __nv_bfloat16* embedding_table, int num_tokens, int hidden_size) {
    int token_idx = blockIdx.x;
    if (token_idx < num_tokens) {
        int token_id = tokens[token_idx];
        for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            out[token_idx * hidden_size + i] = embedding_table[token_id * hidden_size + i];
        }
    }
}

void invoke_embedding_lookup(__nv_bfloat16* out, const int* tokens, const __nv_bfloat16* embedding_table, int num_tokens, int hidden_size, cudaStream_t stream) {
    int threads = std::min(hidden_size, 1024);
    dim3 blocks(num_tokens);
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(out, tokens, embedding_table, num_tokens, hidden_size);
}

// ----------------------------------------------------------------------------
// element-wise add (vectorized float4 = 8×BF16 per thread)
// ----------------------------------------------------------------------------
__global__ void add_kernel(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int n) {
    int n8 = n / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n8) {
        float4 a4 = reinterpret_cast<const float4*>(a)[idx];
        float4 b4 = reinterpret_cast<const float4*>(b)[idx];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        float4 c4;
        __nv_bfloat162* c2 = reinterpret_cast<__nv_bfloat162*>(&c4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            c2[j] = __floats2bfloat162_rn(af.x + bf.x, af.y + bf.y);
        }
        reinterpret_cast<float4*>(out)[idx] = c4;
    }
    // Scalar tail for n % 8 != 0
    if (idx == 0) {
        for (int i = n8 * 8; i < n; i++)
            out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}
void invoke_add(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int n, cudaStream_t stream) {
    int threads = 256;
    int n8 = n / 8;
    int blocks = (n8 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    add_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

// ----------------------------------------------------------------------------
// Fused Add + RMSNorm: residual[i] += bias[i], then RMSNorm on residual
// Eliminates global memory round-trip between add and rmsnorm.
// Each block processes one token. Uses centered weight: out = x * rsqrt(var+eps) * (1+w)
// 线程布局: Grid(num_tokens), Block(256)
// ----------------------------------------------------------------------------
__global__ void fused_add_rmsnorm_kernel(
    __nv_bfloat16* norm_out,        // [T, hs] 输出 normalized result
    __nv_bfloat16* residual,        // [T, hs] in-place: residual += bias
    const __nv_bfloat16* bias,      // [T, hs] 要加到 residual 上的值
    const __nv_bfloat16* weight,    // [hs] RMSNorm weight
    float eps,
    int hidden_size)
{
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    __nv_bfloat16* res_row = residual + token_idx * hidden_size;
    __nv_bfloat16* out_row = norm_out + token_idx * hidden_size;
    const __nv_bfloat16* bias_row = bias + token_idx * hidden_size;

    // 寄存器缓存 (res + bias) 值, 避免 Pass 2 重读
    constexpr int MAX_REGS = 40;
    float r_cache[MAX_REGS];
    int elems_per_thread = hidden_size / blockDim.x;

    // Pass 1: Read res + bias → sum to register cache + compute sum_sq
    float sum_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int i = tid + e * blockDim.x;
        float r = __bfloat162float(res_row[i]) + __bfloat162float(bias_row[i]);
        r_cache[e] = r;
        sum_sq += r * r;
    }

    // Block reduce
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)hidden_size + eps);
    }
    __syncthreads();

    // Pass 2: Write res (updated) + normalized output from register cache
    float inv_rms = s_inv_rms;
    for (int e = 0; e < elems_per_thread; e++) {
        int i = tid + e * blockDim.x;
        float r = r_cache[e];
        res_row[i] = __float2bfloat16(r);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16(r * inv_rms * (1.0f + w));
    }
}

void invoke_fused_add_rmsnorm(
    __nv_bfloat16* norm_out, __nv_bfloat16* residual, const __nv_bfloat16* bias,
    const __nv_bfloat16* weight, float eps, int num_tokens, int hidden_size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_tokens;
    fused_add_rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
        norm_out, residual, bias, weight, eps, hidden_size);
}

// ----------------------------------------------------------------------------
// FP32 → BF16 转换（加载权重时使用）
// BF16 与 FP32 动态范围相同，无需裁剪
// ----------------------------------------------------------------------------
__global__ void f32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}
void invoke_f32_to_bf16(const float* src, __nv_bfloat16* dst, size_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

// ----------------------------------------------------------------------------
// Per-head RMSNorm (q_norm / k_norm)
// x: [num_tokens, num_heads, head_dim], weight: [head_dim]
// ----------------------------------------------------------------------------
__global__ void per_head_rmsnorm_kernel(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                         float eps, int num_heads, int head_dim, bool centered) {
    // blockIdx.x = token, blockIdx.y = head
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    const __nv_bfloat16* row = x + (token * num_heads + head) * head_dim;
    __nv_bfloat16*       dst = out + (token * num_heads + head) * head_dim;

    float sum_sq = 0.f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        sum_sq += v * v;
    }
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(row[i]);
        float w = __bfloat162float(weight[i]);
        float scale = centered ? (1.0f + w) : w;
        dst[i] = __float2bfloat16(v * inv_rms * scale);
    }
}
void invoke_per_head_rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* weight,
                              float eps, int num_tokens, int num_heads, int head_dim,
                              cudaStream_t stream, bool centered) {
    dim3 blocks(num_tokens, num_heads);
    int threads = std::min(head_dim, 256);
    per_head_rmsnorm_kernel<<<blocks, threads, 0, stream>>>(out, x, weight, eps, num_heads, head_dim, centered);
}

// ----------------------------------------------------------------------------
// Partial RoPE: 只旋转每个 head 前 rotary_dim 维
// q: [num_tokens, num_q_heads, head_dim]
// k: [num_tokens, num_kv_heads, head_dim]
// ----------------------------------------------------------------------------
__global__ void rope_partial_kernel(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids,
                                     int num_q_heads, int num_kv_heads,
                                     int head_dim, int rotary_dim, float base) {
    // blockIdx.x = token, blockIdx.y = head (max of q/kv heads)
    // Uses half-rotation pairing: pair (d, d + rotary_dim/2), matching HF rotate_half
    int token    = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_pair = threadIdx.x;  // indexes a pair: (dim_pair, dim_pair + half_rot)
    int half_rot = rotary_dim / 2;

    if (dim_pair >= half_rot) return;

    int pos = pos_ids[token];
    // inv_freq = base^(-(dim_pair*2) / rotary_dim)  — same frequency indexing as HF
    float inv_freq = powf(base, -(float)(dim_pair * 2) / (float)rotary_dim);
    float angle    = (float)pos * inv_freq;
    float cos_v    = cosf(angle);
    float sin_v    = sinf(angle);

    // Q — rotate_half style: pair (dim_pair, dim_pair + half_rot)
    if (head_idx < num_q_heads) {
        int off = token * (num_q_heads * head_dim) + head_idx * head_dim;
        float x0 = __bfloat162float(q[off + dim_pair]);
        float x1 = __bfloat162float(q[off + dim_pair + half_rot]);
        q[off + dim_pair]            = __float2bfloat16(x0 * cos_v - x1 * sin_v);
        q[off + dim_pair + half_rot] = __float2bfloat16(x1 * cos_v + x0 * sin_v);
    }
    // K
    if (head_idx < num_kv_heads) {
        int off = token * (num_kv_heads * head_dim) + head_idx * head_dim;
        float x0 = __bfloat162float(k[off + dim_pair]);
        float x1 = __bfloat162float(k[off + dim_pair + half_rot]);
        k[off + dim_pair]            = __float2bfloat16(x0 * cos_v - x1 * sin_v);
        k[off + dim_pair + half_rot] = __float2bfloat16(x1 * cos_v + x0 * sin_v);
    }
}
void invoke_rope_partial(__nv_bfloat16* q, __nv_bfloat16* k, const int* pos_ids,
                          int num_tokens, int num_q_heads, int num_kv_heads,
                          int head_dim, int rotary_dim,
                          float base, cudaStream_t stream) {
    dim3 blocks(num_tokens, std::max(num_q_heads, num_kv_heads));
    int threads = rotary_dim / 2;  // pairs in rotary_dim
    rope_partial_kernel<<<blocks, threads, 0, stream>>>(
        q, k, pos_ids, num_q_heads, num_kv_heads, head_dim, rotary_dim, base);
}

// ----------------------------------------------------------------------------
// Write KV Cache
// k/v: [num_tokens, num_kv_heads, head_dim]   (row-major)
// k/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
// ----------------------------------------------------------------------------
__global__ void write_kv_cache_kernel(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                                       const __nv_bfloat16* k, const __nv_bfloat16* v,
                                       const int* block_tables,
                                       const int* seq_positions,  // [batch_size] 每个序列的写入位置
                                       int num_tokens,
                                       int num_kv_heads, int head_dim,
                                       int block_size, int max_num_blocks_per_seq,
                                       int batch_size) {
    int token_idx = blockIdx.x;
    int kv_head   = blockIdx.y;
    int d         = threadIdx.x;
    if (d >= head_dim) return;

    // 确定此 token 属于哪个序列及其写入位置
    int seq_idx, pos;
    if (batch_size <= 1) {
        // Prefill 模式: 所有 token 属于序列 0
        seq_idx = 0;
        pos = seq_positions[0] + token_idx;
    } else {
        // Batched decode: token i 来自序列 i
        seq_idx = token_idx;
        pos = seq_positions[seq_idx];
    }

    int block_idx      = pos / block_size;
    int block_offset   = pos % block_size;
    int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + block_idx];

    int src  = token_idx * (num_kv_heads * head_dim) + kv_head * head_dim + d;
    int dst  = physical_block * (block_size * num_kv_heads * head_dim)
             + block_offset  * (num_kv_heads * head_dim)
             + kv_head       * head_dim + d;

    k_cache[dst] = k[src];
    v_cache[dst] = v[src];
}
void invoke_write_kv_cache(__nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                            const __nv_bfloat16* k, const __nv_bfloat16* v,
                            const int* block_tables,
                            int start_pos, int num_tokens,
                            int num_kv_heads, int head_dim,
                            int block_size, int max_num_blocks_per_seq,
                            cudaStream_t stream, int batch_size,
                            const int* seq_positions) {
    dim3 blocks(num_tokens, num_kv_heads);
    int  threads = std::min(head_dim, 256);

    // 向后兼容: 如果没有提供 seq_positions, 使用 start_pos 创建临时数组
    int* d_seq_pos = nullptr;
    bool need_free = false;
    if (seq_positions) {
        d_seq_pos = const_cast<int*>(seq_positions);
    } else {
        // 单序列模式: 分配一个 managed int
        cudaMallocManaged(&d_seq_pos, sizeof(int));
        *d_seq_pos = start_pos;
        need_free = true;
    }

    write_kv_cache_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache, k, v, block_tables,
        d_seq_pos, num_tokens, num_kv_heads, head_dim,
        block_size, max_num_blocks_per_seq,
        batch_size <= 0 ? 1 : batch_size);

    if (need_free) {
        cudaStreamSynchronize(stream);
        cudaFree(d_seq_pos);
    }
}

// ----------------------------------------------------------------------------
// Causal Conv1d（depthwise，kernel_size=4）
// x_io: [num_tokens, channels]  in-place
// conv_state: [channels, conv_k-1]  持久状态，in-place更新 (长度 conv_k-1=3)
// conv_w:     [channels, conv_k]
// 策略: 对于每个 token t，先用前面的状态补足历史窗口，然后计算卷积输出并更新状态
// ----------------------------------------------------------------------------

// ---- Prefill 优化: 全并行 conv1d (所有 token 同时处理) ----
// 前提: prefill 时所有 T 个 input 已知，可以直接并行计算
// 需要先复制 input 到 temp buffer (因为 x_io 是 in-place 被 SiLU 输出覆盖)
// Grid: (ceil(channels/256), num_tokens)
// Block: (256)
// 每线程处理一个 (t, ch) 对
__global__ void causal_conv1d_prefill_parallel_kernel(
    __nv_bfloat16* output,               // [num_tokens, token_stride]  输出 (SiLU(conv(x)))
    const __nv_bfloat16* input,           // [num_tokens, token_stride]  输入副本
    __nv_bfloat16* conv_state,            // [channels, conv_k-1]  持久状态
    const __nv_bfloat16* conv_w,          // [channels, conv_k]
    int num_tokens, int channels, int conv_k,
    int token_stride)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    int t  = blockIdx.y;
    if (ch >= channels || t >= num_tokens) return;

    int hist = conv_k - 1;  // 3

    // 加载 conv_w[ch] (4 个值)
    float w[4];
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        w[k] = (k < conv_k) ? __bfloat162float(conv_w[ch * conv_k + k]) : 0.f;
    }

    // 构建滑动窗口 [x[t-3], x[t-2], x[t-1], x[t]]
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < conv_k; k++) {
        int src_t = t - (hist - k);  // k=0: t-3, k=1: t-2, k=2: t-1, k=3: t
        float val;
        if (src_t < 0) {
            // 从 conv_state 获取历史值
            // conv_state[ch, src_t + hist] — src_t + hist = k 对应 state slot
            val = __bfloat162float(conv_state[ch * hist + (src_t + hist)]);
        } else {
            val = __bfloat162float(input[src_t * token_stride + ch]);
        }
        acc += val * w[k];
    }

    // SiLU activation
    float silu_out = acc / (1.f + expf(-acc));
    output[t * token_stride + ch] = __float2bfloat16(silu_out);
}

// 更新 conv_state: 写入最后 hist=3 个 input 值
__global__ void causal_conv1d_update_state_kernel(
    __nv_bfloat16* conv_state,           // [channels, conv_k-1]
    const __nv_bfloat16* input,          // [num_tokens, token_stride]
    int num_tokens, int channels, int conv_k,
    int token_stride)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int hist = conv_k - 1;
    // conv_state[ch, k] = input[num_tokens - hist + k, ch]
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        int src_t = num_tokens - hist + k;
        if (src_t >= 0) {
            conv_state[ch * hist + k] = input[src_t * token_stride + ch];
        }
        // 如果 src_t < 0, state 保持不变 (num_tokens < hist 的罕见情况)
    }
}

// ---- Decode: 寄存器优化版 ----
// 原始 kernel 每步访问 conv_state global memory 14 次
// 优化: conv_state 和 conv_w 预加载到寄存器
__global__ void causal_conv1d_kernel(__nv_bfloat16* x_io,
                                      __nv_bfloat16** conv_state_ptrs,  // [batch_size] 指针数组, 或 NULL 用 legacy
                                      __nv_bfloat16*  conv_state_single,// legacy 单序列 state
                                      const __nv_bfloat16* conv_w,
                                      int num_tokens, int channels, int conv_k,
                                      int token_stride, int batch_size,
                                      __nv_bfloat16* conv_state_checkpoint) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int hist = conv_k - 1;

    // 确定此 channel 使用哪个 conv state
    __nv_bfloat16* conv_state;
    if (batch_size > 1 && conv_state_ptrs) {
        // Batched decode: 此路径每 thread 处理 1 token, num_tokens 应 = batch_size
        // 保持原始逻辑 (不在此优化)
        for (int t = 0; t < num_tokens; t++) {
            conv_state = conv_state_ptrs[t];
            float cur = __bfloat162float(x_io[t * token_stride + ch]);
            float acc = 0.f;
            for (int k = 0; k < hist; k++) {
                float h_val  = __bfloat162float(conv_state[ch * hist + k]);
                float w_val  = __bfloat162float(conv_w[ch * conv_k + k]);
                acc += h_val * w_val;
            }
            acc += cur * __bfloat162float(conv_w[ch * conv_k + hist]);
            float silu_out = acc / (1.f + expf(-acc));
            x_io[t * token_stride + ch] = __float2bfloat16(silu_out);
            for (int k = 0; k < hist - 1; k++) {
                conv_state[ch * hist + k] = conv_state[ch * hist + k + 1];
            }
            if (hist > 0) {
                conv_state[ch * hist + (hist - 1)] = __float2bfloat16(cur);
            }
        }
        return;
    }

    // Single sequence path: 寄存器优化
    conv_state = conv_state_single;

    // 预加载 conv_w 到寄存器
    float w[4];
    #pragma unroll
    for (int k = 0; k < 4; k++)
        w[k] = (k < conv_k) ? __bfloat162float(conv_w[ch * conv_k + k]) : 0.f;

    // 预加载 conv_state 到寄存器滑动窗口
    float buf0 = __bfloat162float(conv_state[ch * hist + 0]);
    float buf1 = __bfloat162float(conv_state[ch * hist + 1]);
    float buf2 = __bfloat162float(conv_state[ch * hist + 2]);

    for (int t = 0; t < num_tokens; t++) {
        float cur = __bfloat162float(x_io[t * token_stride + ch]);
        float acc = buf0 * w[0] + buf1 * w[1] + buf2 * w[2] + cur * w[3];
        float silu_out = acc / (1.f + expf(-acc));
        x_io[t * token_stride + ch] = __float2bfloat16(silu_out);
        // 滑动窗口 (全在寄存器中)
        buf0 = buf1;
        buf1 = buf2;
        buf2 = cur;

        // MTP speculative checkpoint: save conv state after token[0] for T=2 verify rollback
        if (conv_state_checkpoint && t == 0 && num_tokens > 1) {
            conv_state_checkpoint[ch * hist + 0] = __float2bfloat16(buf0);
            conv_state_checkpoint[ch * hist + 1] = __float2bfloat16(buf1);
            conv_state_checkpoint[ch * hist + 2] = __float2bfloat16(buf2);
        }
    }

    // 写回 conv_state
    conv_state[ch * hist + 0] = __float2bfloat16(buf0);
    conv_state[ch * hist + 1] = __float2bfloat16(buf1);
    conv_state[ch * hist + 2] = __float2bfloat16(buf2);
}
void invoke_causal_conv1d(__nv_bfloat16* x_io, __nv_bfloat16* conv_state,
                           const __nv_bfloat16* conv_w,
                           int num_tokens, int channels, int conv_k,
                           cudaStream_t stream, int token_stride,
                           int batch_size, __nv_bfloat16** conv_state_ptrs,
                           __nv_bfloat16* conv_state_checkpoint) {
    if (token_stride <= 0) token_stride = channels;

    if (num_tokens > 2 && batch_size <= 1 && conv_state) {
        // Prefill 路径: 全并行 conv1d
        // 需要 input 副本 (因为 x_io 被 in-place 覆盖, 而后续 token 需要读前面的原始值)
        // 使用 workspace_conv_buf_ 静态分配
        static __nv_bfloat16* s_conv_input_copy = nullptr;
        static size_t s_conv_input_sz = 0;
        size_t need = (size_t)num_tokens * token_stride * sizeof(__nv_bfloat16);
        if (need > s_conv_input_sz) {
            if (s_conv_input_copy) cudaFree(s_conv_input_copy);
            cudaMalloc(&s_conv_input_copy, need);
            s_conv_input_sz = need;
        }
        // 复制 input
        cudaMemcpyAsync(s_conv_input_copy, x_io, need, cudaMemcpyDeviceToDevice, stream);

        // 并行 kernel: Grid (ceil(channels/256), num_tokens)
        int threads = 256;
        dim3 grid((channels + threads - 1) / threads, num_tokens);
        causal_conv1d_prefill_parallel_kernel<<<grid, threads, 0, stream>>>(
            x_io, s_conv_input_copy, conv_state, conv_w,
            num_tokens, channels, conv_k, token_stride);

        // 更新 conv_state (最后 hist=3 个 input)
        int blocks = (channels + threads - 1) / threads;
        causal_conv1d_update_state_kernel<<<blocks, threads, 0, stream>>>(
            conv_state, s_conv_input_copy,
            num_tokens, channels, conv_k, token_stride);
    } else {
        // Decode 路径 (T=1 或 batched decode): 寄存器优化串行 kernel
        int threads = 256;
        int blocks  = (channels + threads - 1) / threads;
        causal_conv1d_kernel<<<blocks, threads, 0, stream>>>(
            x_io, conv_state_ptrs, conv_state,
            conv_w, num_tokens, channels, conv_k, token_stride, batch_size,
            conv_state_checkpoint);
    }
}

// ----------------------------------------------------------------------------
// Gated DeltaNet 状态更新 (optimized: split value heads, register kS/y, precompute delta)
// Grid: nkh * nv_per_kh 个 blocks — 每个 block 处理一个 (key_head, value_head) 对
// 相比旧版本：
// 1. 3x 并行度 (48 blocks vs 16)，更好的 SM 利用率 (6.25% → 25%)
// 2. kS[j] 和 y[j] 使用寄存器而非 shared memory（减少 smem 读写）
// 3. delta[j] 预计算，避免 update loop 中 128 次冗余 v 和 kS 读取
// 4. k_hat/q_hat 预加载到 shared memory，避免重复 global 读取和 k_norm/q_norm 乘法
//
// q/k: [num_tokens, nkh, kd]    (half)
// v:   [num_tokens, nv, vd]     (half，已按 key head 分组)
// a_raw: [num_tokens, nv]       (half, raw projection output)
// dt_bias: [nv]                 (half, per-head bias for softplus)
// A_log: [nv]                   (float, log decay rate)
// beta_raw: [num_tokens, nv]    (half, raw projection, will be sigmoided inline)
// ssm_state: [nv, kd, vd]      (float, in-place, nv = nkh * nv_per_kh)
// y_out: [num_tokens, nv, vd]   (half)
//
// 每步流程（Gated DeltaNet rule）:
//   alpha = exp(-softplus(a_raw + dt_bias) * exp(A_log))   [inline]
//   beta  = sigmoid(beta_raw)                              [inline]
//   k_hat = k / ||k||_2 + eps
//   beta_k = beta * k_hat     [kd]
//   kS = S^T @ k_hat          [vd]   (register accumulation)
//   delta = v - alpha * kS    [vd]   (precomputed in register)
//   S += outer(beta_k, delta)
//   y = S_new^T @ q           [vd]   (register accumulation)
// ----------------------------------------------------------------------------

// ---- Prefill/Decode kernel: SSM state 缓存在 extended shared memory ----
// 128 threads, 1 per vd element. S[kd, vd_pad] 全量在 smem (66KB).
// 已验证: norm外提无增益(S循环占99%), kd并行化bank conflict(stride-32→4-way).
// Max 2 blocks/SM (66KB×2=132KB < 228KB), 256 threads → 8 warps/SM → 低占用率
// 但 smem 带宽是瓶颈(不是线程数), 所以 occupancy 无关紧要.
__global__ void __launch_bounds__(128, 2)
gated_delta_net_prefill_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias, const float* A_log,
    const __nv_bfloat16* beta_raw,
    float* ssm_state,
    __nv_bfloat16* y_out,
    int num_tokens, int kd, int nv_per_kh, int vd,
    int token_stride, int nkh_x_nvpkh,
    float* ssm_state_checkpoint)
{
    int h_v = blockIdx.x;
    int h_k = h_v / nv_per_kh;
    int nv  = nkh_x_nvpkh;
    int j   = threadIdx.x;  // vd index

    extern __shared__ float smem[];
    const int vd_pad = vd + 1;
    float* S_smem  = smem;
    float* k_hat_s = S_smem + kd * vd_pad;
    float* q_hat_s = k_hat_s + kd;
    float* s_norms = q_hat_s + kd;

    int ss_base = h_v * kd * vd;
    float q_scale = rsqrtf((float)kd);

    for (int i = 0; i < kd; i++)
        S_smem[i * vd_pad + j] = ssm_state[ss_base + i * vd + j];
    __syncthreads();

    for (int t = 0; t < num_tokens; t++) {
        int q_base = t * token_stride + h_k * kd;
        int k_base = t * token_stride + h_k * kd;

        if (threadIdx.x == 0) { s_norms[0] = 0.f; s_norms[1] = 0.f; }
        __syncthreads();
        float local_k_sq = 0.f, local_q_sq = 0.f;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            float kv = __bfloat162float(k[k_base + i]);
            local_k_sq += kv * kv;
            float qv = __bfloat162float(q[q_base + i]);
            local_q_sq += qv * qv;
        }
        local_k_sq = blockReduceSum(local_k_sq);
        if (threadIdx.x == 0) s_norms[0] = local_k_sq;
        __syncthreads();
        local_q_sq = blockReduceSum(local_q_sq);
        if (threadIdx.x == 0) s_norms[1] = local_q_sq;
        __syncthreads();

        float k_norm = rsqrtf(s_norms[0] + 1e-6f);
        float q_norm = rsqrtf(s_norms[1] + 1e-6f) * q_scale;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            k_hat_s[i] = __bfloat162float(k[k_base + i]) * k_norm;
            q_hat_s[i] = __bfloat162float(q[q_base + i]) * q_norm;
        }
        __syncthreads();

        float a_val = __bfloat162float(a_raw[t * nv + h_v]);
        float bias  = dt_bias ? __bfloat162float(dt_bias[h_v]) : 0.f;
        float a_l   = A_log   ? A_log[h_v]                    : 0.f;
        float ab    = a_val + bias;
        float dt_v  = (ab > 20.f) ? ab : log1pf(expf(ab));
        float alpha_v = expf(-dt_v * expf(a_l));
        float beta_v = 1.0f / (1.0f + expf(-__bfloat162float(beta_raw[t * nv + h_v])));

        int v_base = t * token_stride + h_v * vd;
        float kS_j = 0.f;
        for (int i = 0; i < kd; i++)
            kS_j += k_hat_s[i] * S_smem[i * vd_pad + j];

        float v_j = __bfloat162float(v[v_base + j]);
        float delta_j = v_j - alpha_v * kS_j;

        float y_j = 0.f;
        for (int i = 0; i < kd; i++) {
            float beta_k_i = beta_v * k_hat_s[i];
            float old_s = S_smem[i * vd_pad + j];
            float new_s = alpha_v * old_s + beta_k_i * delta_j;
            S_smem[i * vd_pad + j] = new_s;
            y_j += new_s * q_hat_s[i];
        }

        int y_base = (t * nv + h_v) * vd;
        y_out[y_base + j] = __float2bfloat16(y_j);
        __syncthreads();

        // MTP speculative checkpoint: save SSM state after token[0] for T=2 verify rollback
        if (ssm_state_checkpoint && t == 0 && num_tokens > 1) {
            for (int i = 0; i < kd; i++)
                ssm_state_checkpoint[ss_base + i * vd + j] = S_smem[i * vd_pad + j];
            __syncthreads();
        }
    }

    for (int i = 0; i < kd; i++)
        ssm_state[ss_base + i * vd + j] = S_smem[i * vd_pad + j];
}

// ---- Decode 版 (batch_size > 1): 原始 kernel ----
__global__ void gated_delta_net_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias, const float* A_log,
    const __nv_bfloat16* beta_raw,
    float** ssm_state_ptrs,     // [batch_size] 指针数组, 每个→ [nv, kd, vd]
    float*  ssm_state_single,   // legacy 单序列 state (batch_size<=1 时使用)
    __nv_bfloat16* y_out,
    int num_tokens, int kd, int nv_per_kh, int vd,
    int token_stride, int batch_size, int nkh_x_nvpkh)
{
    // blockIdx.x layout:
    //   batch_size <= 1: blockIdx.x = h_v (0..nkh*nv_per_kh-1), loop over num_tokens
    //   batch_size > 1:  blockIdx.x = batch_idx * nkh_x_nvpkh + h_v, process 1 token
    int h_v, batch_idx;
    float* ssm_state;
    int tokens_to_process;
    int token_start;

    if (batch_size <= 1) {
        h_v = blockIdx.x;
        batch_idx = 0;
        ssm_state = ssm_state_single;
        tokens_to_process = num_tokens;
        token_start = 0;
    } else {
        h_v = blockIdx.x % nkh_x_nvpkh;
        batch_idx = blockIdx.x / nkh_x_nvpkh;
        ssm_state = ssm_state_ptrs[batch_idx];
        tokens_to_process = 1;
        token_start = batch_idx;  // token batch_idx
    }

    int h_k = h_v / nv_per_kh;
    int nv  = nkh_x_nvpkh;

    extern __shared__ float smem[];
    float* k_hat_s = smem;
    float* q_hat_s = smem + kd;
    float* s_norms = smem + 2 * kd;

    float q_scale = rsqrtf((float)kd);
    int j = threadIdx.x;
    int ss_base = h_v * kd * vd;

    for (int ti = 0; ti < tokens_to_process; ti++) {
        int t = token_start + ti;
        int q_base = t * token_stride + h_k * kd;
        int k_base = t * token_stride + h_k * kd;

        if (threadIdx.x == 0) { s_norms[0] = 0.f; s_norms[1] = 0.f; }
        __syncthreads();
        float local_k_sq = 0.f;
        float local_q_sq = 0.f;
        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            float kv = __bfloat162float(k[k_base + i]);
            local_k_sq += kv * kv;
            float qv = __bfloat162float(q[q_base + i]);
            local_q_sq += qv * qv;
        }
        local_k_sq = blockReduceSum(local_k_sq);
        if (threadIdx.x == 0) s_norms[0] = local_k_sq;
        __syncthreads();
        local_q_sq = blockReduceSum(local_q_sq);
        if (threadIdx.x == 0) s_norms[1] = local_q_sq;
        __syncthreads();

        float k_norm = rsqrtf(s_norms[0] + 1e-6f);
        float q_norm = rsqrtf(s_norms[1] + 1e-6f) * q_scale;

        for (int i = threadIdx.x; i < kd; i += blockDim.x) {
            k_hat_s[i] = __bfloat162float(k[k_base + i]) * k_norm;
            q_hat_s[i] = __bfloat162float(q[q_base + i]) * q_norm;
        }
        __syncthreads();

        float a_val = __bfloat162float(a_raw[t * nv + h_v]);
        float bias  = dt_bias ? __bfloat162float(dt_bias[h_v]) : 0.f;
        float a_l   = A_log   ? A_log[h_v]                    : 0.f;
        float ab    = a_val + bias;
        float dt_v  = (ab > 20.f) ? ab : log1pf(expf(ab));
        float alpha_v = expf(-dt_v * expf(a_l));

        float beta_v = 1.0f / (1.0f + expf(-__bfloat162float(beta_raw[t * nv + h_v])));

        int v_base = t * token_stride + h_v * vd;

        float kS_j = 0.f;
        for (int i = 0; i < kd; i++) {
            kS_j += k_hat_s[i] * ssm_state[ss_base + i * vd + j];
        }

        float v_j = __bfloat162float(v[v_base + j]);
        float delta_j = v_j - alpha_v * kS_j;

        float y_j = 0.f;
        for (int i = 0; i < kd; i++) {
            float beta_k_i = beta_v * k_hat_s[i];
            float old_s = ssm_state[ss_base + i * vd + j];
            float new_s = alpha_v * old_s + beta_k_i * delta_j;
            ssm_state[ss_base + i * vd + j] = new_s;
            y_j += new_s * q_hat_s[i];
        }

        int y_base = (t * nv + h_v) * vd;
        y_out[y_base + j] = __float2bfloat16(y_j);
        __syncthreads();
    }
}

void invoke_gated_delta_net(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                             const __nv_bfloat16* a_raw, const __nv_bfloat16* dt_bias,
                             const float* A_log, const __nv_bfloat16* beta_raw,
                             float* ssm_state, __nv_bfloat16* y_out,
                             int num_tokens, int nkh, int kd, int nv_per_kh, int vd,
                             cudaStream_t stream, int token_stride,
                             int batch_size, float** ssm_state_ptrs,
                             float* ssm_state_checkpoint) {
    if (token_stride <= 0) token_stride = nkh * kd;
    int nkh_x_nvpkh = nkh * nv_per_kh;  // 48

    if (batch_size <= 1) {
        // ---- Prefill / single-decode: extended shared memory 缓存 SSM state ----
        // S[kd, vd] 全量放入 shared memory, 128 threads (1 per vd), 
        // sequential token loop. 已验证: norm 外提和 kd 并行化均无收益
        // (bank conflict + L2 pollution 抵消所有理论增益)
        int threads = std::min(vd, 128);
        int grid = nkh_x_nvpkh;  // 48 blocks
        const int vd_pad = vd + 1;
        size_t smem_bytes = (size_t)(kd * vd_pad + 2 * kd + 2) * sizeof(float);

        static bool smem_configured = false;
        if (!smem_configured) {
            cudaFuncSetAttribute(gated_delta_net_prefill_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem_bytes);
            smem_configured = true;
        }

        gated_delta_net_prefill_kernel<<<grid, threads, smem_bytes, stream>>>(
            q, k, v, a_raw, dt_bias, A_log, beta_raw,
            ssm_state, y_out,
            num_tokens, kd, nv_per_kh, vd, token_stride, nkh_x_nvpkh,
            ssm_state_checkpoint);
    } else {
        // ---- Batched decode 路径: 原始 kernel ----
        int threads = std::min(vd, 256);
        int grid = batch_size * nkh_x_nvpkh;
        size_t smem = (size_t)(2 * kd + 2) * sizeof(float);
        gated_delta_net_kernel<<<grid, threads, smem, stream>>>(
            q, k, v, a_raw, dt_bias, A_log, beta_raw,
            ssm_state_ptrs, ssm_state, y_out,
            num_tokens, kd, nv_per_kh, vd, token_stride,
            batch_size, nkh_x_nvpkh);
    }
}

// ----------------------------------------------------------------------------
// Sigmoid-Mul: out[i] = a[i] * sigmoid(b[i])
// ----------------------------------------------------------------------------
__global__ void sigmoid_mul_kernel(
    __nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b, int n)
{
    int n8 = n / 8;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n8) {
        float4 a4 = reinterpret_cast<const float4*>(a)[idx];
        float4 b4 = reinterpret_cast<const float4*>(b)[idx];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a4);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b4);
        float4 o4;
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o4);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 af = __bfloat1622float2(a2[j]);
            float2 bf = __bfloat1622float2(b2[j]);
            float sig0 = 1.0f / (1.0f + expf(-bf.x));
            float sig1 = 1.0f / (1.0f + expf(-bf.y));
            o2[j] = __floats2bfloat162_rn(af.x * sig0, af.y * sig1);
        }
        reinterpret_cast<float4*>(out)[idx] = o4;
    }
    // Scalar tail
    if (idx == 0) {
        for (int i = n8 * 8; i < n; i++) {
            float av = __bfloat162float(a[i]);
            float bv = __bfloat162float(b[i]);
            out[i] = __float2bfloat16(av / (1.0f + expf(-bv)));
        }
    }
}

void invoke_sigmoid_mul(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b,
                        int n, cudaStream_t stream) {
    int threads = 256;
    int n8 = n / 8;
    int blocks = (n8 + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    sigmoid_mul_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

// ----------------------------------------------------------------------------
// Deinterleave Q+Gate: [T, num_heads, 2*hd] → Q [T, num_heads, hd] + Gate [T, num_heads*hd]
// qg_in layout per token:  [h0_q0..q_{hd-1}, h0_g0..g_{hd-1}, h1_q0..q_{hd-1}, h1_g0..g_{hd-1}, ...]
// q_out layout per token:  [h0_q0..q_{hd-1}, h1_q0..q_{hd-1}, ...]
// gate_out layout per token: [h0_g0..g_{hd-1}, h1_g0..g_{hd-1}, ...]
// NOTE: q_out may alias qg_in (source index >= dest index, safe for forward copy)
// ----------------------------------------------------------------------------
__global__ void deinterleave_qgate_kernel(
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out, const __nv_bfloat16* qg_in,
    int num_tokens, int num_heads, int head_dim)
{
    // Each thread handles one element
    int total = num_tokens * num_heads * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int hd = head_dim;
    int elem_in_token = idx % (num_heads * hd);
    int t = idx / (num_heads * hd);
    int h = elem_in_token / hd;
    int d = elem_in_token % hd;

    int src_stride = num_heads * 2 * hd;  // per-token stride in qg_in
    int src_offset = t * src_stride + h * 2 * hd;

    q_out[idx]    = qg_in[src_offset + d];
    gate_out[idx] = qg_in[src_offset + hd + d];
}

void invoke_deinterleave_qgate(__nv_bfloat16* q_out, __nv_bfloat16* gate_out,
                                const __nv_bfloat16* qg_in,
                                int num_tokens, int num_heads, int head_dim,
                                cudaStream_t stream) {
    int total = num_tokens * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    deinterleave_qgate_kernel<<<blocks, threads, 0, stream>>>(
        q_out, gate_out, qg_in, num_tokens, num_heads, head_dim);
}

// ----------------------------------------------------------------------------
// Fused Deinterleave Q+Gate + Per-head RMSNorm on Q
// Input: qg_in [T, num_heads, 2*head_dim] (interleaved Q+Gate)
// Output: q_out [T, num_heads, head_dim] (normalized Q)
//         gate_out [T, num_heads*head_dim] (raw gate values)
// Each block processes one (token, head) pair.
// Uses centered weight: out = x * rsqrt(var+eps) * (1+w)
// 线程布局: Grid(T, num_heads), Block(min(head_dim, 256))
// ----------------------------------------------------------------------------
__global__ void fused_deinterleave_q_rmsnorm_kernel(
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out,
    const __nv_bfloat16* qg_in, const __nv_bfloat16* q_norm_weight,
    float eps, int num_heads, int head_dim)
{
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    int src_stride = num_heads * 2 * head_dim;
    int src_base   = token * src_stride + head * 2 * head_dim;
    int dst_base   = (token * num_heads + head) * head_dim;

    // Pass 1: Deinterleave + compute Q sum-of-squares
    float sum_sq = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float q_val = __bfloat162float(qg_in[src_base + d]);
        float g_val = __bfloat162float(qg_in[src_base + head_dim + d]);
        gate_out[dst_base + d] = __float2bfloat16(g_val);
        sum_sq += q_val * q_val;
    }

    // Block reduce for RMSNorm
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    }
    __syncthreads();

    // Pass 2: Normalize Q and write output
    float inv_rms = s_inv_rms;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float q_val = __bfloat162float(qg_in[src_base + d]);
        float w = __bfloat162float(q_norm_weight[d]);
        q_out[dst_base + d] = __float2bfloat16(q_val * inv_rms * (1.0f + w));
    }
}

void invoke_fused_deinterleave_q_rmsnorm(
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out,
    const __nv_bfloat16* qg_in, const __nv_bfloat16* q_norm_weight,
    float eps, int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream)
{
    dim3 blocks(num_tokens, num_heads);
    int threads = std::min(head_dim, 256);
    fused_deinterleave_q_rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
        q_out, gate_out, qg_in, q_norm_weight, eps, num_heads, head_dim);
}

// ============================================================================
// Fused Per-head RMSNorm + SiLU Gate (LinearAttn steps 7+8)
// ============================================================================
// y_ssm [T, num_heads, head_dim]: per-head RMSNorm with plain weight (not centered)
// z_out [T, num_heads * head_dim]: SiLU gate
// Result: y_ssm = rmsnorm(y_ssm) * silu(z_out)
// Grid: (num_tokens, num_heads), Block: min(head_dim, 256)
__global__ void fused_norm_silu_gate_kernel(
    __nv_bfloat16* y_ssm,
    const __nv_bfloat16* z_out,
    const __nv_bfloat16* weight,
    float eps, int num_heads, int head_dim)
{
    int token = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;
    int off   = (token * num_heads + head) * head_dim;

    // Step 1: Compute RMSNorm variance
    float sum_sq = 0.f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = __bfloat162float(y_ssm[off + i]);
        sum_sq += v * v;
    }
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_inv_rms;
    if (tid == 0) s_inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    __syncthreads();

    float inv_rms = s_inv_rms;

    // Step 2: Normalize + SiLU gate in one pass
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float y_val = __bfloat162float(y_ssm[off + i]);
        float w     = __bfloat162float(weight[i]);
        float normalized = y_val * inv_rms * w;  // plain weight (not centered)

        float z_val = __bfloat162float(z_out[off + i]);
        float silu_z = z_val / (1.f + expf(-z_val));

        y_ssm[off + i] = __float2bfloat16(normalized * silu_z);
    }
}

void invoke_fused_norm_silu_gate(
    __nv_bfloat16* y_ssm,
    const __nv_bfloat16* z_out,
    const __nv_bfloat16* weight,
    float eps, int num_tokens, int num_heads, int head_dim,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_heads);
    int threads = std::min(head_dim, 256);
    fused_norm_silu_gate_kernel<<<grid, threads, 0, stream>>>(
        y_ssm, z_out, weight, eps, num_heads, head_dim);
}

// ============================================================================
// GPU Argmax kernel
// ============================================================================
// Single-block reduction: 1 block of 1024 threads, each thread scans a strided
// chunk of the input, then we do shared-memory reduction to find the global max.
// For n=248320 and 1024 threads, each thread handles ceil(248320/1024)=243 elements.

__global__ void argmax_bf16_kernel(const __nv_bfloat16* __restrict__ logits,
                                    int* __restrict__ result_idx, int n)
{
    __shared__ float s_vals[1024];
    __shared__ int   s_idxs[1024];

    int tid = threadIdx.x;
    float local_max = -1e30f;
    int   local_idx = 0;

    // Strided scan
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    // Shared-memory tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result_idx = s_idxs[0];
    }
}

void invoke_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                   cudaStream_t stream) {
    // Single block of 1024 threads — sufficient for n up to ~250K
    argmax_bf16_kernel<<<1, 1024, 0, stream>>>(logits, result_idx, n);
}

// ============================================================================
// Batched GPU Argmax kernel
// ============================================================================
// Launches batch_size blocks, each block finds argmax of logits[b*n .. (b+1)*n-1].
// Eliminates 256× kernel launch overhead at B=256 (33ms → <0.5ms).

__global__ void batched_argmax_bf16_kernel(const __nv_bfloat16* __restrict__ logits,
                                            int* __restrict__ result_idx,
                                            int n, int batch_size)
{
    __shared__ float s_vals[1024];
    __shared__ int   s_idxs[1024];

    int b   = blockIdx.x;
    if (b >= batch_size) return;

    int tid = threadIdx.x;
    const __nv_bfloat16* my_logits = logits + (size_t)b * n;

    float local_max = -1e30f;
    int   local_idx = 0;

    // Strided scan
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(my_logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    s_vals[tid] = local_max;
    s_idxs[tid] = local_idx;
    __syncthreads();

    // Shared-memory tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result_idx[b] = s_idxs[0];
    }
}

void invoke_batched_argmax(const __nv_bfloat16* logits, int* result_idx, int n,
                           int batch_size, cudaStream_t stream) {
    if (batch_size == 1) {
        argmax_bf16_kernel<<<1, 1024, 0, stream>>>(logits, result_idx, n);
    } else {
        batched_argmax_bf16_kernel<<<batch_size, 1024, 0, stream>>>(
            logits, result_idx, n, batch_size);
    }
}

// ----------------------------------------------------------------------------
// Deinterleave merged GEMM output (vectorized float4, single kernel for 3-way split)
// Input: merged [T, N_total] RowMajor
// Output: out1 [T, s1], out2 [T, s2-s1], out3 [T, N_total-s2]
// All widths must be multiples of 8
// ----------------------------------------------------------------------------
__global__ void deinterleave_3way_kernel(
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    __nv_bfloat16* __restrict__ out3,
    const __nv_bfloat16* __restrict__ merged,
    int T, int N_total, int s1, int s2)
{
    // Each block processes one row (one token)
    int t = blockIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* src = merged + (size_t)t * N_total;
    int w1 = s1;
    int w2 = s2 - s1;
    int w3 = N_total - s2;

    // Copy w1 elements to out1
    __nv_bfloat16* dst1 = out1 + (size_t)t * w1;
    for (int i = threadIdx.x * 8; i < w1; i += blockDim.x * 8) {
        *reinterpret_cast<float4*>(dst1 + i) = *reinterpret_cast<const float4*>(src + i);
    }

    // Copy w2 elements to out2
    __nv_bfloat16* dst2 = out2 + (size_t)t * w2;
    const __nv_bfloat16* src2 = src + s1;
    for (int i = threadIdx.x * 8; i < w2; i += blockDim.x * 8) {
        *reinterpret_cast<float4*>(dst2 + i) = *reinterpret_cast<const float4*>(src2 + i);
    }

    // Copy w3 elements to out3
    __nv_bfloat16* dst3 = out3 + (size_t)t * w3;
    const __nv_bfloat16* src3 = src + s2;
    for (int i = threadIdx.x * 8; i < w3; i += blockDim.x * 8) {
        *reinterpret_cast<float4*>(dst3 + i) = *reinterpret_cast<const float4*>(src3 + i);
    }
}

void invoke_deinterleave_gemm_3way(
    __nv_bfloat16* out1, __nv_bfloat16* out2, __nv_bfloat16* out3,
    const __nv_bfloat16* merged, int num_tokens,
    int N_total, int split1, int split2, cudaStream_t stream)
{
    // Choose threads based on max width: each thread handles 8 elements
    int max_width = std::max({split1, split2 - split1, N_total - split2});
    int threads = std::min(256, (max_width / 8 + 31) / 32 * 32);
    if (threads < 32) threads = 32;
    deinterleave_3way_kernel<<<num_tokens, threads, 0, stream>>>(
        out1, out2, out3, merged, num_tokens, N_total, split1, split2);
}

// ----------------------------------------------------------------------------
// Fused SwiGLU for merged gate+up layout
// Input: merged_gateup [T, 2*is] RowMajor (gate at [0,is), up at [is,2*is) per row)
// Output: out [T, is] = silu(gate) * up
// Vectorized with float4 (8 BF16 per thread)
// ----------------------------------------------------------------------------
__global__ void swiglu_merged_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ merged,
    int T, int is)
{
    // Each block processes one token
    int t = blockIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* gate_row = merged + (size_t)t * 2 * is;
    const __nv_bfloat16* up_row   = gate_row + is;
    __nv_bfloat16* out_row = out + (size_t)t * is;

    for (int base = threadIdx.x * 8; base < is; base += blockDim.x * 8) {
        float4 g4 = *reinterpret_cast<const float4*>(gate_row + base);
        float4 u4 = *reinterpret_cast<const float4*>(up_row + base);
        float4 o4;

        __nv_bfloat162* g2 = reinterpret_cast<__nv_bfloat162*>(&g4);
        __nv_bfloat162* u2 = reinterpret_cast<__nv_bfloat162*>(&u4);
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&o4);

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float gv0 = __bfloat162float(g2[j].x);
            float gv1 = __bfloat162float(g2[j].y);
            float uv0 = __bfloat162float(u2[j].x);
            float uv1 = __bfloat162float(u2[j].y);
            float sg0 = gv0 / (1.f + expf(-gv0));
            float sg1 = gv1 / (1.f + expf(-gv1));
            o2[j] = __floats2bfloat162_rn(sg0 * uv0, sg1 * uv1);
        }

        *reinterpret_cast<float4*>(out_row + base) = o4;
    }
}

void invoke_swiglu_merged(__nv_bfloat16* out, const __nv_bfloat16* merged_gateup,
                           int num_tokens, int intermediate_size, cudaStream_t stream)
{
    // is / 8 iterations per thread, use enough threads to cover
    int threads = std::min(256, (intermediate_size / 8 + 31) / 32 * 32);
    if (threads < 32) threads = 32;
    swiglu_merged_kernel<<<num_tokens, threads, 0, stream>>>(
        out, merged_gateup, num_tokens, intermediate_size);
}

} // namespace ops
} // namespace qwen_thor
