// cache_manager.cpp — 统一缓存管理器实现
//
// Phase 1: 包装现有组件 (KVCacheManager, BlockSSDStore, CacheEngine, KVSwapper)
// 将 engine.cpp 中 ~500 行分散的缓存管理逻辑收拢到此

#include "cache_manager.h"
#include "layer.h"    // Qwen35Config definition
#include "allocator.h"
#include <iostream>
#include <cstdio>
#include <cstring>

namespace qwen_thor {
namespace cache {

// ===========================================================================
// 构造 & 析构
// ===========================================================================

CacheManager::CacheManager(const core::Qwen35Config& model_config,
                           const CacheConfig& cache_config,
                           const ModelCacheParams& mcp,
                           const CapacityReport& capacity,
                           cudaStream_t stream)
    : stream_(stream)
{
    // 保存模型参数
    int num_full_attn = model_config.num_full_attn_layers();
    num_layers_ = num_full_attn;
    num_kv_heads_ = model_config.num_key_value_heads;
    head_dim_ = model_config.head_dim;
    block_size_ = 16;
    total_gpu_blocks_ = capacity.gpu_kv_blocks;
    gpu_max_tokens_ = capacity.gpu_max_tokens;

    // 统计 linear attention 层数
    num_linear_layers_ = 0;
    for (int li = 0; li < model_config.num_hidden_layers; ++li)
        if (!model_config.is_full_attention(li)) ++num_linear_layers_;

    // SSM/Conv 尺寸
    ssm_size_per_layer_ = (size_t)model_config.linear_num_key_heads
                          * model_config.linear_key_head_dim
                          * model_config.lin_v_per_kh() * sizeof(__nv_bfloat16);
    int in_qkv_conv = model_config.lin_qk_dim() * 2 + model_config.lin_v_dim();
    conv_size_per_layer_ = (size_t)in_qkv_conv
                           * (model_config.linear_conv_kernel_dim - 1) * sizeof(__nv_bfloat16);

    // 1. GPU KV Block Pool
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    kv_manager_ = std::make_unique<ops::KVCacheManager>(
        capacity.gpu_kv_blocks, block_size_, model_config.num_key_value_heads,
        model_config.head_dim, core::DataType::FP16, allocator, num_full_attn);
    printf("[CacheManager] GPU KV pool: %d blocks, %d max tokens (%.1f GB)\n",
           total_gpu_blocks_, gpu_max_tokens_, cache_config.kv_cache_budget_gb);

    // 2. SSM/Conv Pool
    {
        size_t total_ssm = (size_t)MAX_CACHE_SSM_SLOTS * num_linear_layers_ * ssm_size_per_layer_;
        size_t total_conv = (size_t)MAX_CACHE_SSM_SLOTS * num_linear_layers_ * conv_size_per_layer_;
        cudaMalloc(&ssm_pool_base_, total_ssm);
        cudaMalloc(&conv_pool_base_, total_conv);

        pooled_ssm_states_.resize(MAX_CACHE_SSM_SLOTS);
        pooled_conv_states_.resize(MAX_CACHE_SSM_SLOTS);
        size_t ssm_slot_stride = (size_t)num_linear_layers_ * ssm_size_per_layer_;
        size_t conv_slot_stride = (size_t)num_linear_layers_ * conv_size_per_layer_;
        for (int s = 0; s < MAX_CACHE_SSM_SLOTS; ++s) {
            pooled_ssm_states_[s].resize(num_linear_layers_);
            pooled_conv_states_[s].resize(num_linear_layers_);
            for (int li = 0; li < num_linear_layers_; ++li) {
                pooled_ssm_states_[s][li] = (__nv_bfloat16*)((char*)ssm_pool_base_
                    + s * ssm_slot_stride + (size_t)li * ssm_size_per_layer_);
                pooled_conv_states_[s][li] = (__nv_bfloat16*)((char*)conv_pool_base_
                    + s * conv_slot_stride + (size_t)li * conv_size_per_layer_);
            }
        }
        cudaMemset(ssm_pool_base_, 0, total_ssm);
        cudaMemset(conv_pool_base_, 0, total_conv);

        free_ssm_slots_.reserve(MAX_CACHE_SSM_SLOTS);
        for (int s = MAX_CACHE_SSM_SLOTS - 1; s >= 0; --s)
            free_ssm_slots_.push_back(s);

        printf("[CacheManager] SSM/Conv pool: %d slots × %d layers, "
               "SSM=%.1f KB/layer, Conv=%.1f KB/layer, total=%.1f MB\n",
               MAX_CACHE_SSM_SLOTS, num_linear_layers_,
               ssm_size_per_layer_ / 1024.0, conv_size_per_layer_ / 1024.0,
               (total_ssm + total_conv) / 1048576.0);
    }

    // 3. Prefix Cache
    cache_engine_ = std::make_unique<CacheEngine>(cache_config, mcp);

    // 4. KV Swapper
    if (cache_config.enabled) {
        int block_bytes = block_size_ * num_kv_heads_ * head_dim_ * sizeof(__nv_bfloat16);
        std::string swap_dir = cache_config.cache_dir + "/kv_swap";
        kv_swapper_ = std::make_unique<KVSwapper>(swap_dir, block_bytes, num_layers_);
        printf("[CacheManager] KV Swapper at %s\n", swap_dir.c_str());
    }

    // 5. Block SSD Store + Streaming Attention Buffers
    {
        ssd_io_staging_size_ = 32ULL * 1024 * 1024;  // 32 MB
        posix_memalign(&ssd_io_staging_, 4096, ssd_io_staging_size_);

        std::string block_store_dir = cache_config.cache_dir + "/block_store";
        block_ssd_store_ = std::make_unique<BlockSSDStore>(
            block_store_dir, block_size_, num_kv_heads_, head_dim_,
            num_layers_, ssd_io_staging_, ssd_io_staging_size_);

        // Staging GPU KV cache
        size_t staging_k_bytes = ssd_io_staging_size_ / 2;
        int kv_dim_per_block = block_size_ * num_kv_heads_ * head_dim_;
        staging_num_blocks_ = (int)(staging_k_bytes / (kv_dim_per_block * sizeof(__nv_bfloat16)));
        if (staging_num_blocks_ < 1) staging_num_blocks_ = 1;

        cudaMalloc(&d_staging_k_, (size_t)staging_num_blocks_ * kv_dim_per_block * sizeof(__nv_bfloat16));
        cudaMalloc(&d_staging_v_, (size_t)staging_num_blocks_ * kv_dim_per_block * sizeof(__nv_bfloat16));

        // Partial attention buffers
        int max_partial_tokens = std::max(32, max_chunk_size_);
        int num_q_heads = model_config.num_attention_heads;
        cudaMalloc(&d_partial_out_,  (size_t)max_partial_tokens * num_q_heads * head_dim_ * sizeof(__nv_bfloat16));
        cudaMalloc(&d_partial_out2_, (size_t)max_partial_tokens * num_q_heads * head_dim_ * sizeof(__nv_bfloat16));
        cudaMalloc(&d_partial_m_,    (size_t)max_partial_tokens * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_l_,    (size_t)max_partial_tokens * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_m2_,   (size_t)max_partial_tokens * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_l2_,   (size_t)max_partial_tokens * num_q_heads * sizeof(float));
        cudaMalloc(&d_ssd_block_tables_,
                   (size_t)(total_gpu_blocks_ + staging_num_blocks_ + 256) * sizeof(int));
        cudaMalloc(&d_ssd_context_lens_, sizeof(int));

        printf("[CacheManager] Streaming attention: staging %d blocks\n", staging_num_blocks_);
    }
}

CacheManager::~CacheManager() {
    // Drain swapper before destroying
    if (kv_swapper_) {
        kv_swapper_->drain();
        kv_swapper_->print_stats();
    }

    // Free SSM/Conv pool
    if (ssm_pool_base_) cudaFree(ssm_pool_base_);
    if (conv_pool_base_) cudaFree(conv_pool_base_);

    // Free streaming attention buffers
    if (d_staging_k_) cudaFree(d_staging_k_);
    if (d_staging_v_) cudaFree(d_staging_v_);
    if (d_partial_out_) cudaFree(d_partial_out_);
    if (d_partial_out2_) cudaFree(d_partial_out2_);
    if (d_partial_m_) cudaFree(d_partial_m_);
    if (d_partial_l_) cudaFree(d_partial_l_);
    if (d_partial_m2_) cudaFree(d_partial_m2_);
    if (d_partial_l2_) cudaFree(d_partial_l2_);
    if (d_ssd_block_tables_) cudaFree(d_ssd_block_tables_);
    if (d_ssd_context_lens_) cudaFree(d_ssd_context_lens_);

    // Free SSD I/O staging
    if (ssd_io_staging_) free(ssd_io_staging_);
}

// ===========================================================================
// 请求生命周期
// ===========================================================================

bool CacheManager::allocate_request(uint64_t req_id, int initial_tokens,
                                    RequestCacheState& state) {
    // 1. 分配 SSM/Conv slot
    int slot = allocate_ssm_slot();
    if (slot < 0) {
        fprintf(stderr, "[CacheManager] No free SSM slots for req %lu\n", req_id);
        return false;
    }
    state.ssm_slot = slot;
    state.ssm_states.resize(num_linear_layers_);
    state.conv_states.resize(num_linear_layers_);
    for (int li = 0; li < num_linear_layers_; ++li) {
        state.ssm_states[li] = pooled_ssm_states_[slot][li];
        state.conv_states[li] = pooled_conv_states_[slot][li];
    }

    // 清零 SSM/Conv (避免残留数据)
    size_t ssm_slot_bytes = (size_t)num_linear_layers_ * ssm_size_per_layer_;
    size_t conv_slot_bytes = (size_t)num_linear_layers_ * conv_size_per_layer_;
    cudaMemsetAsync(
        (char*)ssm_pool_base_ + (size_t)slot * ssm_slot_bytes,
        0, ssm_slot_bytes, stream_);
    cudaMemsetAsync(
        (char*)conv_pool_base_ + (size_t)slot * conv_slot_bytes,
        0, conv_slot_bytes, stream_);

    // 2. 分配 KV blocks
    int blocks_needed = (initial_tokens + block_size_ - 1) / block_size_;
    if (blocks_needed > 0) {
        try {
            auto blocks = kv_manager_->allocate_blocks(blocks_needed);
            state.block_tracker.init(blocks);
        } catch (const std::runtime_error&) {
            // 回收 SSM slot
            free_ssm_slot(slot);
            state.ssm_slot = -1;
            fprintf(stderr, "[CacheManager] Cannot allocate %d blocks for req %lu "
                    "(free=%d)\n", blocks_needed, req_id, num_free_gpu_blocks());
            return false;
        }
    }

    state.context_len = initial_tokens;
    state.is_swapped = false;
    return true;
}

bool CacheManager::extend_blocks(uint64_t req_id, int total_tokens_needed,
                                 RequestCacheState& state) {
    int blocks_needed = (total_tokens_needed + block_size_ - 1) / block_size_;
    int current_blocks = state.num_total_blocks();
    int blocks_to_add = blocks_needed - current_blocks;

    if (blocks_to_add <= 0) {
        state.context_len = total_tokens_needed;
        return true;
    }

    try {
        auto new_blocks = kv_manager_->allocate_blocks(blocks_to_add);
        for (int b : new_blocks) {
            state.block_tracker.push_block(b);
        }
    } catch (const std::runtime_error&) {
        fprintf(stderr, "[CacheManager] Cannot extend %d blocks for req %lu "
                "(free=%d)\n", blocks_to_add, req_id, num_free_gpu_blocks());
        return false;
    }

    state.context_len = total_tokens_needed;
    return true;
}

void CacheManager::free_request(uint64_t req_id, RequestCacheState& state) {
    // 1. 释放 GPU KV blocks (跳过 SSD 的 -1)
    if (!state.is_swapped) {
        auto full_bt = state.full_block_table();
        std::vector<int> gpu_blocks;
        for (int b : full_bt)
            if (b >= 0) gpu_blocks.push_back(b);
        if (!gpu_blocks.empty())
            kv_manager_->free_blocks(gpu_blocks);
    }

    // 2. 清理 SSD block store 文件
    if (state.has_ssd_blocks() && block_ssd_store_) {
        block_ssd_store_->remove(req_id);
    }

    // 3. 清理 swap 文件
    if (state.is_swapped && kv_swapper_) {
        kv_swapper_->remove(req_id);
        state.is_swapped = false;
    }

    // 4. 回收 SSM/Conv slot
    if (state.ssm_slot >= 0) {
        free_ssm_slot(state.ssm_slot);
        state.ssm_slot = -1;
    }

    // 5. 清理状态
    state.ssm_states.clear();
    state.conv_states.clear();
    state.context_len = 0;
}

// ===========================================================================
// Block 级 GPU → SSD Eviction
// ===========================================================================

int CacheManager::evict_blocks_to_ssd(uint64_t req_id, int num_blocks_to_free,
                                      RequestCacheState& state) {
    if (!block_ssd_store_) return 0;

    // 找到可 evict 的 blocks (从头开始，FIFO)
    auto full_bt = state.full_block_table();
    std::vector<int> evict_logical;
    std::vector<int> phys_to_free;

    for (int i = 0; i < (int)full_bt.size() && (int)evict_logical.size() < num_blocks_to_free; ++i) {
        if (full_bt[i] >= 0) {  // GPU-resident
            evict_logical.push_back(i);
            phys_to_free.push_back(full_bt[i]);
        }
    }

    if (evict_logical.empty()) return 0;

    // Sync stream: 确保 KV 数据已写入
    cudaStreamSynchronize(stream_);

    // 写入 SSD
    block_ssd_store_->evict_blocks(req_id, evict_logical, full_bt, *kv_manager_);

    // 清除可能的 CUDA 错误
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[CacheManager] Warning: CUDA error after eviction: %s (cleared)\n",
                    cudaGetErrorString(err));
        }
    }

    // 释放 GPU blocks
    kv_manager_->free_blocks(phys_to_free);

    // 更新 tracker
    for (int idx : evict_logical) {
        state.block_tracker.mark_evicted(idx, 1);
    }

    return (int)evict_logical.size();
}

// ===========================================================================
// 请求级 Swap
// ===========================================================================

bool CacheManager::swap_out_request(uint64_t req_id, RequestCacheState& state) {
    if (!kv_swapper_) return false;
    if (state.is_swapped) return true;  // 已经换出

    auto full_bt = state.full_block_table();
    // 只能换出全 GPU-resident 的请求 (有 SSD blocks 的不支持整请求 swap)
    if (state.has_ssd_blocks()) {
        fprintf(stderr, "[CacheManager] Cannot swap_out req %lu: has SSD blocks\n", req_id);
        return false;
    }

    kv_swapper_->swap_out(
        req_id, *kv_manager_, full_bt, state.context_len,
        state.ssm_states.empty() ? nullptr : state.ssm_states.data(),
        num_linear_layers_, ssm_size_per_layer_,
        state.conv_states.empty() ? nullptr : state.conv_states.data(),
        conv_size_per_layer_, stream_);

    // 释放 GPU blocks
    kv_manager_->free_blocks(full_bt);

    // 回收 SSM slot
    if (state.ssm_slot >= 0) {
        free_ssm_slot(state.ssm_slot);
        state.ssm_slot = -1;
    }

    state.is_swapped = true;
    state.ssm_states.clear();
    state.conv_states.clear();
    return true;
}

bool CacheManager::swap_in_request(uint64_t req_id, RequestCacheState& state) {
    if (!kv_swapper_) return false;
    if (!state.is_swapped) return true;

    // 分配 SSM slot
    int slot = allocate_ssm_slot();
    if (slot < 0) {
        fprintf(stderr, "[CacheManager] swap_in failed: no SSM slots for req %lu\n", req_id);
        return false;
    }
    state.ssm_slot = slot;
    state.ssm_states.resize(num_linear_layers_);
    state.conv_states.resize(num_linear_layers_);
    for (int li = 0; li < num_linear_layers_; ++li) {
        state.ssm_states[li] = pooled_ssm_states_[slot][li];
        state.conv_states[li] = pooled_conv_states_[slot][li];
    }

    auto new_blocks = kv_swapper_->swap_in(
        req_id, *kv_manager_,
        state.ssm_states.data(), state.conv_states.data(),
        stream_);

    if (new_blocks.empty()) {
        free_ssm_slot(slot);
        state.ssm_slot = -1;
        state.ssm_states.clear();
        state.conv_states.clear();
        fprintf(stderr, "[CacheManager] swap_in failed for req %lu\n", req_id);
        return false;
    }

    state.block_tracker.init(new_blocks);
    state.is_swapped = false;
    return true;
}

int CacheManager::get_swapped_context_len(uint64_t req_id) const {
    if (!kv_swapper_) return 0;
    return kv_swapper_->get_swapped_context_len(req_id);
}

// ===========================================================================
// Prefix Cache
// ===========================================================================

int CacheManager::lookup_prefix(const int* tokens, int num_tokens) {
    if (!cache_engine_ || !cache_engine_->is_enabled()) return 0;
    auto result = cache_engine_->lookup_prefix(tokens, num_tokens);
    return result.matched_tokens;
}

int CacheManager::restore_prefix(const int* tokens, int num_tokens,
                                 RequestCacheState& state,
                                 __nv_bfloat16* workspace,
                                 int* d_block_table_buf) {
    if (!cache_engine_ || !cache_engine_->is_enabled()) return 0;

    std::vector<int> restored_blocks;
    int restored = cache_engine_->retrieve_prefix(
        tokens, num_tokens, *kv_manager_, restored_blocks,
        state.ssm_states.empty() ? nullptr : state.ssm_states.data(),
        state.conv_states.empty() ? nullptr : state.conv_states.data(),
        workspace, d_block_table_buf, stream_);

    if (restored > 0) {
        // 更新 block tracker
        state.block_tracker.init(restored_blocks);
        state.context_len = restored;
    }
    return restored;
}

void CacheManager::store_prefix(const int* tokens, int num_tokens,
                                const RequestCacheState& state,
                                __nv_bfloat16* workspace) {
    if (!cache_engine_ || !cache_engine_->is_enabled()) return;

    auto full_bt = state.full_block_table();
    cache_engine_->store_prefix(
        tokens, num_tokens, *kv_manager_,
        nullptr,  // d_block_table (will use host block table)
        (int)full_bt.size(),
        const_cast<__nv_bfloat16**>(state.ssm_states.data()),
        const_cast<__nv_bfloat16**>(state.conv_states.data()),
        workspace, stream_);
}

void CacheManager::record_prefix_stats(int prompt_tokens, int restored, int computed) {
    if (cache_engine_) {
        cache_engine_->record_request(prompt_tokens, restored, computed);
    }
}

// ===========================================================================
// Streaming Attention
// ===========================================================================

ops::StreamingAttnCtx CacheManager::build_streaming_ctx(
    uint64_t req_id,
    const RequestCacheState& state,
    int gpu_ctx_len)
{
    auto ssd_indices = state.ssd_logical_indices();
    auto gpu_bt = state.gpu_block_table();

    ops::StreamingAttnCtx sctx;
    sctx.total_ssd_blocks = (int)ssd_indices.size();
    sctx.ssd_tokens = sctx.total_ssd_blocks * block_size_;
    sctx.staging_capacity = staging_num_blocks_;
    sctx.staging_k = d_staging_k_;
    sctx.staging_v = d_staging_v_;
    sctx.d_staging_context_lens = d_ssd_context_lens_;
    sctx.partial_out = d_partial_out_;
    sctx.partial_m = d_partial_m_;
    sctx.partial_l = d_partial_l_;
    sctx.partial_out2 = d_partial_out2_;
    sctx.partial_m2 = d_partial_m2_;
    sctx.partial_l2 = d_partial_l2_;

    // GPU block table
    sctx.gpu_num_blocks = (int)gpu_bt.size();
    cudaMemcpyAsync(d_ssd_block_tables_, gpu_bt.data(),
                    gpu_bt.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);
    sctx.d_gpu_block_tables = d_ssd_block_tables_;

    // GPU context lens
    cudaMemcpyAsync(d_ssd_context_lens_, &gpu_ctx_len,
                    sizeof(int), cudaMemcpyHostToDevice, stream_);
    sctx.d_gpu_context_lens = d_ssd_context_lens_;

    // Staging block table: [0, 1, 2, ...]
    {
        std::vector<int> staging_bt(staging_num_blocks_);
        for (int i = 0; i < staging_num_blocks_; i++) staging_bt[i] = i;
        cudaMemcpyAsync(d_ssd_block_tables_ + gpu_bt.size(),
                        staging_bt.data(),
                        staging_num_blocks_ * sizeof(int),
                        cudaMemcpyHostToDevice, stream_);
        sctx.d_staging_block_tables = d_ssd_block_tables_ + gpu_bt.size();
    }

    // SSD load callback
    sctx.load_ssd_batch = [this, req_id, ssd_indices](
        int full_attn_idx, int batch_start, int batch_count) -> int {
        int end = std::min(batch_start + batch_count, (int)ssd_indices.size());
        std::vector<int> batch_indices(ssd_indices.begin() + batch_start,
                                       ssd_indices.begin() + end);
        return block_ssd_store_->load_blocks_for_layer(
            req_id, full_attn_idx, batch_indices,
            d_staging_k_, d_staging_v_);
    };

    return sctx;
}

int CacheManager::load_ssd_blocks_for_layer(uint64_t req_id, int layer_idx,
                                            const std::vector<int>& ssd_indices,
                                            int batch_start, int batch_count) {
    if (!block_ssd_store_) return 0;
    int end = std::min(batch_start + batch_count, (int)ssd_indices.size());
    std::vector<int> batch(ssd_indices.begin() + batch_start,
                           ssd_indices.begin() + end);
    return block_ssd_store_->load_blocks_for_layer(
        req_id, layer_idx, batch, d_staging_k_, d_staging_v_);
}

// ===========================================================================
// SSM/Conv Pool
// ===========================================================================

int CacheManager::allocate_ssm_slot() {
    if (free_ssm_slots_.empty()) return -1;
    int slot = free_ssm_slots_.back();
    free_ssm_slots_.pop_back();
    return slot;
}

void CacheManager::free_ssm_slot(int slot) {
    free_ssm_slots_.push_back(slot);
}

__nv_bfloat16* CacheManager::get_ssm_state(int slot, int layer) const {
    return pooled_ssm_states_[slot][layer];
}

__nv_bfloat16* CacheManager::get_conv_state(int slot, int layer) const {
    return pooled_conv_states_[slot][layer];
}

// ===========================================================================
// 查询
// ===========================================================================

int CacheManager::num_free_gpu_blocks() const {
    return kv_manager_->num_free_blocks();
}

void CacheManager::print_swap_stats() const {
    if (kv_swapper_) kv_swapper_->print_stats();
}

void CacheManager::drain_swapper() {
    if (kv_swapper_) kv_swapper_->drain();
}

} // namespace cache
} // namespace qwen_thor
