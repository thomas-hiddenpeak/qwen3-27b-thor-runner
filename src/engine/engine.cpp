#include "engine.h"
#include "light_ops.h"
#include "dense_gemm.h"
#include "streaming_attention.h"
#include "cache_engine.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <unistd.h>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <thread>

// ---------------------------------------------------------------------------
// GPU sync with timeout — prevents hardware watchdog (Tegra186 WDT, 2min)
// from hard-resetting the system if a CUDA kernel hangs.
// Uses cudaStreamQuery polling with 10ms sleep intervals (not spin-wait)
// so CPU stays responsive and systemd can feed the watchdog.
// ---------------------------------------------------------------------------
static bool sync_stream_with_timeout(cudaStream_t stream, int timeout_seconds,
                                      const char* context = nullptr) {
    auto start = std::chrono::steady_clock::now();
    for (;;) {
        cudaError_t err = cudaStreamQuery(stream);
        if (err == cudaSuccess) return true;
        if (err != cudaErrorNotReady) {
            fprintf(stderr, "[Engine] CUDA error during sync (%s): %s\n",
                    context ? context : "unknown",
                    cudaGetErrorString(err));
            cudaGetLastError(); // clear sticky error
            return false;
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= timeout_seconds) {
            fprintf(stderr, "[Engine] *** GPU SYNC TIMEOUT after %ds (%s) — possible kernel hang! ***\n",
                    timeout_seconds, context ? context : "unknown");
            fflush(stderr);
            return false;
        }
        // 10ms sleep: 让 CPU 空闲, systemd 可以喂狗 + 处理其他任务
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

namespace qwen_thor {
namespace core {

InferenceEngine::InferenceEngine(const Qwen35Config& config, const std::string& model_dir,
                                 const cache::CacheConfig& cache_config)
    : config_(config) {
    
    // cudaDeviceScheduleBlockingSync 已由 Backend 启动时的 check_memory_budget() 设置
    // (必须在 CUDA context 首次初始化前调用, 不能在此重复设置)
    
    cudaStreamCreate(&compute_stream_);

    // 1. 初始化模型
    std::cout << "Initializing Qwen3.5 Model..." << std::endl;
    model_ = std::make_unique<Qwen35Model>(config_);
    model_->load_weights(model_dir);

    log_tokenizer_ready_ = log_tokenizer_.load(model_dir);
    if (!log_tokenizer_ready_) {
        std::cerr << "[Engine] Warning: tokenizer load failed for log decoding from "
                  << model_dir << std::endl;
    }

    // 2. 构建 ModelCacheParams (KV + SSM 参数, 同时给 KV Manager 和 CacheEngine 使用)
    int num_full_attn_layers = config_.num_full_attn_layers();
    cache::ModelCacheParams mcp;
    mcp.num_full_attn_layers   = num_full_attn_layers;
    mcp.num_kv_heads           = config_.num_key_value_heads;
    mcp.head_dim               = config_.head_dim;
    mcp.block_size             = 16;
    mcp.num_linear_attn_layers = config_.num_hidden_layers - num_full_attn_layers;
    mcp.nkh                    = config_.linear_num_key_heads;
    mcp.kd                     = config_.linear_key_head_dim;
    mcp.v_per_kh               = config_.lin_v_per_kh();
    mcp.in_qkv                 = config_.lin_qk_dim() * 2 + config_.lin_v_dim();
    mcp.conv_k_minus_1         = config_.linear_conv_kernel_dim - 1;

    // 3. 容量规划: 从 kv_cache_budget_gb 计算实际 block 数
    auto capacity = cache::CapacityPlanner::plan(cache_config, mcp);
    cache::CapacityPlanner::print_report(capacity, cache_config);

    // 4. 初始化 KV Cache Manager (block 数由预算驱动, 不再硬编码)
    auto allocator = std::make_shared<UnifiedAllocator>();
    kv_manager_ = std::make_unique<ops::KVCacheManager>(
        capacity.gpu_kv_blocks, 16, config_.num_key_value_heads, config_.head_dim,
        DataType::FP16, allocator, num_full_attn_layers
    );
    gpu_max_tokens_ = capacity.gpu_max_tokens;
    printf("[Engine] Max tokens per request: %d (%.1fK) from KV budget %.1f GB\n",
           gpu_max_tokens_, gpu_max_tokens_ / 1024.0, cache_config.kv_cache_budget_gb);

    // 安全检查: gpu_max_tokens 不能超过 IPC 最大 prompt 长度
    if (gpu_max_tokens_ > ipc::MAX_PROMPT_LEN) {
        printf("[Engine] WARNING: gpu_max_tokens(%d) > MAX_PROMPT_LEN(%d), "
               "clamping. Recompile with larger MAX_PROMPT_LEN to use full budget.\n",
               gpu_max_tokens_, ipc::MAX_PROMPT_LEN);
        gpu_max_tokens_ = ipc::MAX_PROMPT_LEN;
    }

    // 5. 初始化 IPC 队列
    ipc_queue_      = std::make_unique<ipc::ShmRingBuffer<ipc::InferenceRequest,  8>>("/qwen_thor_ipc",  true);
    ipc_resp_queue_ = std::make_unique<ipc::ShmRingBuffer<ipc::InferenceResponse, 512>>("/qwen_thor_resp", true);

    // 5b. 预分配显存
    // max_tokens = 最大单 chunk 序列长度 (chunked prefill 时每块不超过此值)
    int max_tokens   = 8192;

    // d_hidden_states: max_tokens * hidden_size
    cudaMalloc(&d_hidden_states_, (size_t)max_tokens * config_.hidden_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos_ids_,       (size_t)max_tokens * sizeof(int));
    // d_block_tables: 支持全长度 prompt 的 block table (chunked prefill 需要)
    int max_kv_blks = (ipc::MAX_PROMPT_LEN + 15) / 16 + 1024;  // prompt + decode headroom
    cudaMalloc(&d_block_tables_,  (size_t)max_kv_blks * sizeof(int));
    cudaMalloc(&d_context_lens_,  sizeof(int));
    cudaMallocManaged(&d_argmax_result_, sizeof(int));  // GPU argmax 结果

    // Workspace: 单层最大激活量 (linear_attn 更大)
    // FullAttn  : 4*hs + 2*qp_dim + 2*kv_dim + 3*is  (qp_dim = 2*q_dim for Q+gate)
    // LinearAttn: hs + (2*qk+lin_v) + lin_v + nkh + qk + lin_v + hs + hs + 3*is + hs + nkh*2(float/half)
    const int hs     = config_.hidden_size;            // 5120
    const int is     = config_.intermediate_size;      // 17408
    const int q_dim  = config_.q_dim();                // 6144
    const int qp_dim = config_.q_proj_dim();           // 12288
    const int kv_dim = config_.kv_dim();               // 1024
    const int qk     = config_.lin_qk_dim();           // 2048
    const int lin_v  = config_.lin_v_dim();            // 6144
    const int nkh    = config_.linear_num_key_heads;   // 16
    const int in_qkv = 2 * qk + lin_v;                // 10240

    size_t ws_full   = (size_t)(4*hs + qp_dim + q_dim + 2*kv_dim + 3*is);
    // alpha_f32: nkh floats placed after the last half buffer; count as 2 halfs per float
    size_t ws_linear = (size_t)(hs + in_qkv + lin_v + nkh + qk + lin_v + hs + hs + 3*is + hs + nkh*2);
    size_t ws_per_tok = std::max(ws_full, ws_linear);
    cudaMalloc(&d_workspace_, ws_per_tok * max_tokens * sizeof(__nv_bfloat16));

    // 6. 计算并缓存 SSM/Conv 尺寸 (swap 和请求初始化复用)
    {
        num_linear_layers_ = 0;
        for (int li = 0; li < config_.num_hidden_layers; ++li)
            if (!config_.is_full_attention(li)) ++num_linear_layers_;
        ssm_size_per_layer_ = (size_t)config_.linear_num_key_heads
                              * config_.linear_key_head_dim
                              * config_.lin_v_per_kh() * sizeof(__nv_bfloat16);
        int in_qkv_conv = config_.lin_qk_dim() * 2 + config_.lin_v_dim();
        conv_size_per_layer_ = (size_t)in_qkv_conv
                               * (config_.linear_conv_kernel_dim - 1) * sizeof(__nv_bfloat16);

        // 预分配多槽位 SSM/Conv 状态池: 每个活跃请求独占一个 slot, 互不干扰
        // MAX_SSM_SLOTS 个独立 slot × num_linear_layers_ 层, 2 次连续 cudaMalloc
        // 预清零触发页面映射, 避免首次请求时的延迟页错引发 Jetson 看门狗
        size_t total_ssm_bytes = (size_t)MAX_SSM_SLOTS * num_linear_layers_ * ssm_size_per_layer_;
        size_t total_conv_bytes = (size_t)MAX_SSM_SLOTS * num_linear_layers_ * conv_size_per_layer_;
        cudaMalloc(&ssm_pool_base_,  total_ssm_bytes);
        cudaMalloc(&conv_pool_base_, total_conv_bytes);
        // 设置 [slot][layer] 指针视图
        pooled_ssm_states_.resize(MAX_SSM_SLOTS);
        pooled_conv_states_.resize(MAX_SSM_SLOTS);
        size_t ssm_slot_stride = (size_t)num_linear_layers_ * ssm_size_per_layer_;
        size_t conv_slot_stride = (size_t)num_linear_layers_ * conv_size_per_layer_;
        for (int s = 0; s < MAX_SSM_SLOTS; ++s) {
            pooled_ssm_states_[s].resize(num_linear_layers_);
            pooled_conv_states_[s].resize(num_linear_layers_);
            for (int li = 0; li < num_linear_layers_; ++li) {
                pooled_ssm_states_[s][li]  = (__nv_bfloat16*)((char*)ssm_pool_base_ + s * ssm_slot_stride + (size_t)li * ssm_size_per_layer_);
                pooled_conv_states_[s][li] = (__nv_bfloat16*)((char*)conv_pool_base_ + s * conv_slot_stride + (size_t)li * conv_size_per_layer_);
            }
        }
        // 预清零: 触发统一内存页面分配
        cudaMemset(ssm_pool_base_, 0, total_ssm_bytes);
        cudaMemset(conv_pool_base_, 0, total_conv_bytes);
        // 初始化 free slot 栈 (所有 slot 可用)
        free_ssm_slots_.reserve(MAX_SSM_SLOTS);
        for (int s = MAX_SSM_SLOTS - 1; s >= 0; --s)
            free_ssm_slots_.push_back(s);
        printf("[Engine] SSM/Conv state pool: %d slots × %d layers, SSM=%.1f KB/layer, Conv=%.1f KB/layer, total=%.1f MB\n",
               MAX_SSM_SLOTS, num_linear_layers_, ssm_size_per_layer_/1024.0, conv_size_per_layer_/1024.0,
               (total_ssm_bytes + total_conv_bytes) / 1048576.0);
    }

    // 7. 初始化 KV Cache Prefix Caching (SSD offload, 复用上面的 mcp)
    {
        cache_engine_ = std::make_unique<cache::CacheEngine>(cache_config, mcp);
    }

    // 8. 初始化 KV Swapper (请求级 KV 换出到 SSD)
    if (cache_config.enabled) {
        // block_bytes = block_size × num_kv_heads × head_dim × sizeof(bf16)
        int block_bytes = 16 * config_.num_key_value_heads * config_.head_dim * sizeof(__nv_bfloat16);
        std::string swap_dir = cache_config.cache_dir + "/kv_swap";
        kv_swapper_ = std::make_unique<cache::KVSwapper>(
            swap_dir, block_bytes, num_full_attn_layers);
        std::cout << "[Engine] KV Swapper initialized at " << swap_dir << std::endl;
    }

    // 9. 初始化 Block-level SSD Store + Streaming Attention Buffers
    //    用于当单请求 context 超过 gpu_max_tokens 时, 部分 blocks 驻留 SSD
    {
        // SSD I/O staging buffer (host, 4K aligned): 32 MB
        ssd_io_staging_size_ = 32ULL * 1024 * 1024;
        posix_memalign(&ssd_io_staging_, 4096, ssd_io_staging_size_);

        std::string block_store_dir = cache_config.cache_dir + "/block_store";
        block_ssd_store_ = std::make_unique<cache::BlockSSDStore>(
            block_store_dir, 16, config_.num_key_value_heads, config_.head_dim,
            num_full_attn_layers, ssd_io_staging_, ssd_io_staging_size_);

        // Staging GPU KV cache: 与 SSD I/O staging 匹配, 32 MB → ~512 blocks per layer
        // 每 block 每层 K or V = 16 × 4 × 256 × 2B = 32 KB
        // 32 MB / 64 KB (K+V) = 512 blocks per layer
        size_t staging_k_bytes = ssd_io_staging_size_ / 2;  // K 和 V 各一半 → 可容纳 512 blocks
        int kv_dim_per_block = 16 * config_.num_key_value_heads * config_.head_dim;
        staging_num_blocks_ = (int)(staging_k_bytes / (kv_dim_per_block * sizeof(__nv_bfloat16)));
        if (staging_num_blocks_ < 1) staging_num_blocks_ = 1;

        cudaMalloc(&d_staging_k_, (size_t)staging_num_blocks_ * kv_dim_per_block * sizeof(__nv_bfloat16));
        cudaMalloc(&d_staging_v_, (size_t)staging_num_blocks_ * kv_dim_per_block * sizeof(__nv_bfloat16));

        // Partial attention buffers (for decode: num_tokens=1 typically, max=32 for batched)
        int max_batch = 32;
        int num_q_heads = config_.num_attention_heads;
        cudaMalloc(&d_partial_out_,  (size_t)max_batch * num_q_heads * config_.head_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&d_partial_out2_, (size_t)max_batch * num_q_heads * config_.head_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&d_partial_m_,    (size_t)max_batch * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_l_,    (size_t)max_batch * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_m2_,   (size_t)max_batch * num_q_heads * sizeof(float));
        cudaMalloc(&d_partial_l2_,   (size_t)max_batch * num_q_heads * sizeof(float));
        cudaMalloc(&d_ssd_block_tables_, (size_t)(capacity.gpu_kv_blocks + staging_num_blocks_ + 256) * sizeof(int));
        cudaMalloc(&d_ssd_context_lens_, sizeof(int));

        printf("[Engine] Streaming attention: staging %d blocks, store at %s\n",
               staging_num_blocks_, block_store_dir.c_str());
    }

    // 10. MTP 投机解码初始化
    //     --mtp-enable: 强制开; --mtp-disable: 强制关; 默认 auto (模型有 MTP 权重则启用)
    {
        bool mtp_want = false;
        if (cache_config.mtp_mode == "on") {
            mtp_want = true;
            if (!model_->has_mtp()) {
                printf("[Engine] WARNING: --mtp-enable specified but model has no MTP weights. Ignoring.\n");
                mtp_want = false;
            }
        } else if (cache_config.mtp_mode == "off") {
            mtp_want = false;
            if (model_->has_mtp()) {
                printf("[Engine] MTP speculative decoding explicitly disabled via --mtp-disable\n");
            }
        } else {
            // auto: 模型有 MTP 权重则启用
            mtp_want = model_->has_mtp();
        }

        if (mtp_want) {
            int mtp_blocks = cache_config.mtp_kv_blocks;
            if (mtp_blocks < 1) mtp_blocks = 256;

            // a) MTP 自己的 KV Cache (1 full attention layer)
            auto mtp_allocator = std::make_shared<UnifiedAllocator>();
            mtp_kv_manager_ = std::make_unique<ops::KVCacheManager>(
                mtp_blocks /*blocks*/, 16 /*block_size*/, config_.num_key_value_heads, config_.head_dim,
                DataType::FP16, mtp_allocator, 1 /*num_layers*/
            );

            // b) SSM/Conv 状态 checkpoint buffers (T=2 verify rollback 用)
            ssm_elems_per_layer_ = (size_t)config_.linear_num_key_heads
                                   * config_.linear_key_head_dim
                                   * config_.lin_v_per_kh();
            conv_elems_per_layer_ = (size_t)(config_.lin_qk_dim() * 2 + config_.lin_v_dim())
                                    * (config_.linear_conv_kernel_dim - 1);
            cudaMalloc(&d_ssm_checkpoints_,  num_linear_layers_ * ssm_elems_per_layer_ * sizeof(__nv_bfloat16));
            cudaMalloc(&d_conv_checkpoints_, num_linear_layers_ * conv_elems_per_layer_ * sizeof(__nv_bfloat16));

            // c) MTP device buffers
            cudaMalloc(&d_mtp_block_tables_,  mtp_blocks * sizeof(int));
            cudaMalloc(&d_mtp_context_lens_,  sizeof(int));

            printf("[Engine] MTP speculative decoding ENABLED (mode=%s, drafts=%d, kv_blocks=%d = %d max tokens)\n",
                   cache_config.mtp_mode.c_str(), cache_config.mtp_num_drafts, mtp_blocks, mtp_blocks * 16);
            printf("[Engine]   SSM checkpoint %.1f MB, Conv checkpoint %.1f MB\n",
                   num_linear_layers_ * ssm_elems_per_layer_ * sizeof(__nv_bfloat16) / 1048576.0,
                   num_linear_layers_ * conv_elems_per_layer_ * sizeof(__nv_bfloat16) / 1048576.0);
            num_mtp_drafts_ = cache_config.mtp_num_drafts;
        } else {
            printf("[Engine] MTP speculative decoding DISABLED (mode=%s, model_has_mtp=%s)\n",
                   cache_config.mtp_mode.c_str(), model_->has_mtp() ? "yes" : "no");
        }
    }

    // 10. 初始化采样缓冲区 (CPU 侧)
    sampling_logits_.resize(config_.vocab_size);
    sampling_indices_.resize(config_.vocab_size);
    sampling_logits_bf16_.resize(config_.vocab_size);
    sampling_rng_.seed(std::random_device{}());

    // 11. Vision encoder workspace (如果模型有视觉编码器)
    if (model_->has_vision()) {
        auto& vcfg = model_->get_vision_config();
        // 预分配 workspace 支持 max_pixels 大小的图像
        // max_pixels / (patch_size² * spatial_merge_size²) = max output tokens
        int max_patches = vcfg.max_pixels / (vcfg.patch_size * vcfg.patch_size);
        vision_workspace_bytes_ = model_->get_vision_encoder()->workspace_bytes(max_patches);
        // BF16 pixel values are uploaded directly into workspace (no FP32 temp needed)
        cudaMalloc(&d_vision_workspace_, vision_workspace_bytes_);
        printf("[Engine] Vision workspace: %.1f MB (max %d patches from %d max_pixels)\n",
               vision_workspace_bytes_ / 1048576.0, max_patches, vcfg.max_pixels);
    }

    // 确保所有 CUDA 初始化完成 (页面映射、memset 等), 避免首请求时延迟缺页
    cudaDeviceSynchronize();
    printf("[Engine] Initialization complete, all CUDA pages settled.\n");
}

InferenceEngine::~InferenceEngine() {
    stop();
    if (kv_swapper_) {
        kv_swapper_->drain();        // 等待后台预取完成
        kv_swapper_->print_stats();
    }
    cudaFree(d_hidden_states_);
    cudaFree(d_pos_ids_);
    cudaFree(d_workspace_);
    cudaFree(d_block_tables_);
    cudaFree(d_context_lens_);
    cudaFree(d_argmax_result_);
    cudaFree(d_staging_k_);
    cudaFree(d_staging_v_);
    cudaFree(d_partial_out_);
    cudaFree(d_partial_out2_);
    cudaFree(d_partial_m_);
    cudaFree(d_partial_l_);
    cudaFree(d_partial_m2_);
    cudaFree(d_partial_l2_);
    cudaFree(d_ssd_block_tables_);
    cudaFree(d_ssd_context_lens_);
    cudaFree(d_ssm_checkpoints_);
    cudaFree(d_conv_checkpoints_);
    cudaFree(d_mtp_block_tables_);
    cudaFree(d_mtp_context_lens_);
    cudaFree(d_vision_workspace_);
    // 释放 SSM/Conv 状态池 (2次连续分配, 进程退出时释放)
    cudaFree(ssm_pool_base_);
    cudaFree(conv_pool_base_);
    if (ssd_io_staging_) free(ssd_io_staging_);
    cudaStreamDestroy(compute_stream_);
}

void InferenceEngine::start() {
    if (running_) return;
    running_ = true;
    worker_thread_ = std::thread(&InferenceEngine::inference_loop, this);
}

void InferenceEngine::stop() {
    if (!running_) return;
    running_ = false;
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void InferenceEngine::inference_loop() {
    std::cout << "Inference engine started." << std::endl;
    
    std::vector<RequestContext*> active_requests;

    while (running_) {
        // 1. 尝试从 IPC 队列获取新请求
        ipc::InferenceRequest req;
        if (ipc_queue_->pop(req)) {
            auto ctx = std::make_unique<RequestContext>();
            ctx->request_id = req.request_id;
            ctx->max_new_tokens = req.max_new_tokens;
            ctx->temperature = req.temperature;
            ctx->top_p       = req.top_p;
            ctx->top_k       = req.top_k;
            ctx->min_p       = req.min_p;
            ctx->repeat_penalty    = req.repeat_penalty;
            ctx->frequency_penalty = req.frequency_penalty;
            ctx->presence_penalty  = req.presence_penalty;
            ctx->seed              = req.seed;
            for (int i = 0; i < req.prompt_len; ++i) {
                ctx->prompt_tokens.push_back(req.prompt_tokens[i]);
            }
            ctx->context_len = req.prompt_len;

            // 多模态: 检索附加的图像数据
            {
                std::lock_guard<std::mutex> lock(pending_images_mutex_);
                auto it = pending_images_.find(req.request_id);
                if (it != pending_images_.end()) {
                    ctx->processed_images = std::move(it->second);
                    pending_images_.erase(it);
                    std::cout << "[Engine] Found " << ctx->processed_images.size()
                              << " vision items for request " << req.request_id << std::endl;
                } else {
                    std::cout << "[Engine] No vision items for request " << req.request_id
                              << " (pending_images size=" << pending_images_.size() << ")" << std::endl;
                }
            }

            // 检查: prompt 长度不能超过模型最大位置编码
            int total_tokens_needed = req.prompt_len + req.max_new_tokens;
            if (!block_ssd_store_ && total_tokens_needed > gpu_max_tokens_) {
                // 没有 SSD backing 时, 受 GPU KV 预算限制
                std::cerr << "[Engine] Request " << req.request_id
                          << " prompt(" << req.prompt_len << ") + max_new("
                          << req.max_new_tokens << ") = " << total_tokens_needed
                          << " exceeds gpu_max_tokens=" << gpu_max_tokens_
                          << ". Increase --kv-cache-gb or enable SSD. Dropping." << std::endl;
                ipc::InferenceResponse err_resp;
                err_resp.request_id = req.request_id;
                err_resp.token_id = 0;
                err_resp.is_finished = true;
                err_resp.error_code = -1;
                ipc_resp_queue_->push(err_resp);
                continue;
            }
            if (req.prompt_len > ipc::MAX_PROMPT_LEN) {
                std::cerr << "[Engine] Request " << req.request_id
                          << " prompt(" << req.prompt_len << ") > MAX_PROMPT_LEN("
                          << ipc::MAX_PROMPT_LEN << "). Dropping." << std::endl;
                ipc::InferenceResponse err_resp{};
                err_resp.request_id = req.request_id;
                err_resp.token_id = 0;
                err_resp.is_finished = true;
                err_resp.error_code = -1;
                ipc_resp_queue_->push(err_resp);
                continue;
            }
            
            // 分配初始的 KV Cache Block
            // 有 SSD backing 时: 只分配第一个 chunk 的 blocks (后续增量分配 + 溢出 evict 到 SSD)
            // 无 SSD backing 时: 分配全部 blocks
            int num_blocks_needed;
            if (block_ssd_store_ && total_tokens_needed > gpu_max_tokens_) {
                // SSD 模式: 只分配一个 chunk 的 blocks
                int first_chunk = std::min(req.prompt_len, max_chunk_size_);
                num_blocks_needed = (first_chunk + 15) / 16;
                std::cout << "[Engine] SSD mode: allocating " << num_blocks_needed
                          << " blocks for first chunk (total prompt=" << req.prompt_len << ")" << std::endl;
            } else {
                num_blocks_needed = (req.prompt_len + 15) / 16;
            }

            // 如果 free blocks 不够, 先尝试换出其它请求
            while (kv_manager_->num_free_blocks() < num_blocks_needed && kv_swapper_) {
                int freed = try_swap_out_victim(active_requests, num_blocks_needed);
                if (freed == 0) break;
            }

            std::vector<int> blocks;
            try {
                blocks = kv_manager_->allocate_blocks(num_blocks_needed);
            } catch (const std::runtime_error& e) {
                std::cerr << "[Engine] Cannot allocate " << num_blocks_needed
                          << " blocks (free=" << kv_manager_->num_free_blocks()
                          << "). Dropping request " << req.request_id << std::endl;
                continue;  // 丢弃请求
            }
            ctx->block_table = blocks;
            
            // 初始化 block tracker (用于 SSD 模式追踪 GPU vs SSD blocks)
            ctx->block_tracker.init(blocks);
            
            active_requests.push_back(ctx.get());
            // 从预分配池分配独立 SSM/Conv slot (无 cudaMalloc, 仅 memset 清零)
            // 每个活跃请求占一个独立 slot, 互不干扰
            // CRITICAL: cudaMemsetAsync 在 compute_stream_ 上, 保证与推理 kernel 序列化
            if (free_ssm_slots_.empty()) {
                // 无可用 slot → 尝试换出其它请求腾出 slot
                bool freed_slot = false;
                for (size_t i = 1; i < active_requests.size(); ++i) {
                    auto* r = active_requests[i];
                    if (!r->is_swapped && !r->is_finished && r->ssm_slot >= 0) {
                        std::cout << "[Engine] No free SSM slot, swapping out request "
                                  << r->request_id << " (slot " << r->ssm_slot << ")" << std::endl;
                        try_swap_out_victim(active_requests, 0);
                        freed_slot = !free_ssm_slots_.empty();
                        if (freed_slot) break;
                    }
                }
                if (!freed_slot) {
                    std::cerr << "[Engine] FATAL: No free SSM slots and cannot swap out. "
                              << "Dropping request " << req.request_id << std::endl;
                    // 释放已分配的 KV blocks
                    if (!ctx->block_table.empty()) {
                        kv_manager_->free_blocks(ctx->block_table);
                        ctx->block_table.clear();
                    }
                    active_requests.pop_back();
                    ipc::InferenceResponse err_resp{};
                    err_resp.request_id = req.request_id;
                    err_resp.token_id = 0;
                    err_resp.is_finished = true;
                    err_resp.error_code = -1;
                    ipc_resp_queue_->push(err_resp);
                    continue;
                }
            }
            {
                int slot = free_ssm_slots_.back();
                free_ssm_slots_.pop_back();
                ctx->ssm_slot = slot;
                ctx->ssm_states.resize(num_linear_layers_);
                ctx->conv_states.resize(num_linear_layers_);
                for (int li = 0; li < num_linear_layers_; ++li) {
                    ctx->ssm_states[li] = pooled_ssm_states_[slot][li];
                    ctx->conv_states[li] = pooled_conv_states_[slot][li];
                    cudaMemsetAsync(ctx->ssm_states[li], 0, ssm_size_per_layer_, compute_stream_);
                    cudaMemsetAsync(ctx->conv_states[li], 0, conv_size_per_layer_, compute_stream_);
                }
                std::cout << "[Engine] Assigned SSM slot " << slot << " to request "
                          << req.request_id << " (free slots: " << free_ssm_slots_.size() << ")" << std::endl;
            }
            all_requests_.push_back(std::move(ctx));
            
            std::cout << "Received request " << req.request_id << " with prompt len " << req.prompt_len << std::endl;
        }

        if (active_requests.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 2a. 检查取消请求 — 将已取消的 active request 标记为 finished
        {
            std::lock_guard<std::mutex> lock(cancel_mutex_);
            if (!cancel_set_.empty()) {
                for (auto* ctx : active_requests) {
                    if (cancel_set_.count(ctx->request_id)) {
                        ctx->is_finished = true;
                        // 发送 is_finished 响应
                        ipc::InferenceResponse resp{};
                        resp.request_id = ctx->request_id;
                        resp.token_id = 0;
                        resp.is_finished = true;
                        resp.error_code = -2;  // cancelled
                        ipc_resp_queue_->push(resp);
                        std::cout << "[Engine] Cancelled request " << ctx->request_id << std::endl;
                    }
                }
                cancel_set_.clear();
            }
        }

        // 跳过如果所有请求都被取消了
        if (std::all_of(active_requests.begin(), active_requests.end(),
                        [](RequestContext* ctx) { return ctx->is_finished; })) {
            // 直接进入清理阶段
        } else {
            // 2b. 调度执行
            step(active_requests);
        }

        // 3. 清理完成的请求，释放 KV blocks + SSM/Conv 状态
        // CRITICAL: 必须先等 compute_stream_ 上所有 kernel 完成,
        // 否则 SSM/Conv pool 的清零 (下一请求的 cudaMemsetAsync) 可能
        // 与正在运行的 kernel 竞态 → GPU hang.
        {
            bool has_finished = false;
            for (auto* ctx : active_requests)
                if (ctx->is_finished) { has_finished = true; break; }
            if (has_finished) {
                if (!sync_stream_with_timeout(compute_stream_, 90, "cleanup_pre_sync")) {
                    fprintf(stderr, "[Engine] FATAL: GPU hang detected during cleanup sync! "
                            "Marking all requests finished to prevent further damage.\\n");
                    fflush(stderr);
                    for (auto* ctx : active_requests) ctx->is_finished = true;
                }
            }
        }
        for (auto* ctx : active_requests) {
            if (ctx->is_finished) {
                fprintf(stderr, "[Engine] Cleanup: req=%lu freeing resources...\n", ctx->request_id);
                // 释放 GPU-resident KV Cache blocks (跳过已 evict 到 SSD 的 -1 entries)
                if (!ctx->block_table.empty() && !ctx->is_swapped) {
                    std::vector<int> gpu_blocks;
                    for (int b : ctx->block_table)
                        if (b >= 0) gpu_blocks.push_back(b);
                    if (!gpu_blocks.empty())
                        kv_manager_->free_blocks(gpu_blocks);
                    ctx->block_table.clear();
                }
                // 清理 SSD 上的 block store 文件
                if (ctx->uses_ssd_blocks && block_ssd_store_) {
                    block_ssd_store_->remove(ctx->request_id);
                    ctx->uses_ssd_blocks = false;
                }
                // 清理 SSD 上的 swap 文件 (如果被换出过)
                if (ctx->is_swapped && kv_swapper_) {
                    kv_swapper_->remove(ctx->request_id);
                    ctx->is_swapped = false;
                }
                // 释放 processed_images (ViT 后不再需要 bf16 pixel 数据)
                if (!ctx->processed_images.empty()) {
                    size_t img_bytes = 0;
                    for (auto& img : ctx->processed_images)
                        img_bytes += img.pixel_values_bf16.size() * sizeof(__nv_bfloat16);
                    ctx->processed_images.clear();
                    ctx->processed_images.shrink_to_fit();
                    fprintf(stderr, "[Engine] Cleanup: req=%lu freed %zu bytes of image data\n",
                            ctx->request_id, img_bytes);
                }
                // 池化: 不 cudaFree, 回收 slot 到 free list, 断开 ctx 指针
                if (ctx->ssm_slot >= 0) {
                    free_ssm_slots_.push_back(ctx->ssm_slot);
                    fprintf(stderr, "[Engine] Cleanup: req=%lu returned SSM slot %d (free slots: %zu)\n",
                            ctx->request_id, ctx->ssm_slot, free_ssm_slots_.size());
                    ctx->ssm_slot = -1;
                }
                ctx->ssm_states.clear();
                ctx->conv_states.clear();
                // 释放 MTP KV Cache blocks
                if (!ctx->mtp_block_table.empty() && mtp_kv_manager_) {
                    mtp_kv_manager_->free_blocks(ctx->mtp_block_table);
                    ctx->mtp_block_table.clear();
                }
                // 释放已完成请求持有的 prompt/generated tokens 内存
                ctx->prompt_tokens.clear();
                ctx->prompt_tokens.shrink_to_fit();
                ctx->generated_tokens.clear();
                ctx->generated_tokens.shrink_to_fit();
                // 非阻塞检查 GPU 异步错误 (不做 sync, 避免潜在 GPU hang)
                {
                    cudaError_t peek = cudaPeekAtLastError();
                    if (peek != cudaSuccess) {
                        fprintf(stderr, "[Engine] Cleanup: req=%lu GPU async error: %s\n",
                                ctx->request_id, cudaGetErrorString(peek));
                    }
                }
                fprintf(stderr, "[Engine] Cleanup: req=%lu fully done\n", ctx->request_id);
                fflush(stderr);
            }
        }

        active_requests.erase(
            std::remove_if(active_requests.begin(), active_requests.end(),
                [](RequestContext* ctx) { return ctx->is_finished; }),
            active_requests.end()
        );
        // 从 all_requests_ 中移除已完成的请求, 释放 RequestContext 内存
        all_requests_.erase(
            std::remove_if(all_requests_.begin(), all_requests_.end(),
                [](const std::unique_ptr<RequestContext>& p) { return p->is_finished; }),
            all_requests_.end()
        );

        // GPU 冷却间隙 + 全设备同步: 确保 GPU 完全 idle, 统一内存页面安定
        // 防止跨请求的 GPU 状态残留 (SMMU/IOMMU 页表、compute/copy engine 竞态)
        if (active_requests.empty()) {
            // 全设备同步 (带超时, 避免 WDT)
            sync_stream_with_timeout(compute_stream_, 60, "inter_request_sync");
            // 200ms idle 间隙: 让统一内存子系统完成页面迁移/回收
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}

void InferenceEngine::step() {
    // 供外部测试用的空实现
}

void InferenceEngine::step(std::vector<RequestContext*>& active_requests) {
    if (active_requests.empty()) return;

    // 简化版：只处理第一个请求
    auto* ctx = active_requests[0];

    // ---- 如果该请求被换出过, 先换入 ----
    if (ctx->is_swapped && kv_swapper_) {
        std::cout << "[Engine] Swapping in request " << ctx->request_id << std::endl;

        // 换入前检查: 需要足够的 free blocks, 否则先换出其它请求
        int ctx_len = kv_swapper_->get_swapped_context_len(ctx->request_id);
        int blocks_needed = (ctx_len + 15) / 16;
        while (kv_manager_->num_free_blocks() < blocks_needed && active_requests.size() > 1) {
            int freed = try_swap_out_victim(active_requests, blocks_needed);
            if (freed == 0) break;
        }

        // 从预分配池分配独立 SSM/Conv slot (swap-in)
        if (free_ssm_slots_.empty()) {
            // 尝试换出其它请求腾出 slot
            for (size_t i = 1; i < active_requests.size(); ++i) {
                auto* r = active_requests[i];
                if (!r->is_swapped && !r->is_finished && r->ssm_slot >= 0 && r != ctx) {
                    try_swap_out_victim(active_requests, 0);
                    if (!free_ssm_slots_.empty()) break;
                }
            }
        }
        if (free_ssm_slots_.empty()) {
            std::cerr << "[Engine] swap_in failed: no free SSM slots for request "
                      << ctx->request_id << std::endl;
            ctx->is_finished = true;
            return;
        }
        {
            int slot = free_ssm_slots_.back();
            free_ssm_slots_.pop_back();
            ctx->ssm_slot = slot;
            for (int li = 0; li < num_linear_layers_; ++li) {
                ctx->ssm_states[li] = pooled_ssm_states_[slot][li];
                ctx->conv_states[li] = pooled_conv_states_[slot][li];
            }
            std::cout << "[Engine] Swap-in: assigned SSM slot " << slot
                      << " to request " << ctx->request_id << std::endl;
        }
        auto new_blocks = kv_swapper_->swap_in(
            ctx->request_id, *kv_manager_,
            ctx->ssm_states.data(), ctx->conv_states.data(),
            compute_stream_);
        if (new_blocks.empty()) {
            std::cerr << "[Engine] swap_in failed for request " << ctx->request_id << std::endl;
            ctx->is_finished = true;
            return;
        }
        ctx->block_table = new_blocks;
        ctx->is_swapped = false;
    }
    
    if (ctx->generated_tokens.empty()) {
        // --------------------------------------------------------------------
        // Prefill 阶段
        // --------------------------------------------------------------------
        profiler_.request_start();
        int num_tokens = ctx->prompt_tokens.size();

        // ---- KV Cache Prefix Lookup ----
        int cached_tokens = 0;
        if (cache_engine_ && cache_engine_->is_enabled()) {
            auto lookup = cache_engine_->lookup_prefix(
                ctx->prompt_tokens.data(), num_tokens);
            if (lookup.matched_tokens > 0) {
                // 尝试从 SSD 恢复缓存的 KV + SSM/Conv
                cached_tokens = cache_engine_->retrieve_prefix(
                    ctx->prompt_tokens.data(), num_tokens,
                    *kv_manager_, ctx->block_table,
                    ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data(),
                    ctx->conv_states.empty() ? nullptr : ctx->conv_states.data(),
                    d_workspace_, compute_stream_);
                if (cached_tokens > 0) {
                    ctx->context_len = cached_tokens;
                    std::cout << "[Cache] Restored " << cached_tokens << "/" << num_tokens
                              << " tokens from SSD" << std::endl;
                }
            }
        }

        // 如果部分恢复, 只 prefill 剩余 token; 如果全部命中, 跳过 forward
        int prefill_start = cached_tokens;
        int prefill_tokens = num_tokens - cached_tokens;
        
        if (prefill_tokens > 0) {
            // ================================================================
            // Chunked Prefill: 当 prefill_tokens > max_chunk_size_ 时自动分块
            //
            // 原理:
            // - 将长 prompt 切成 chunk_size 大小的块, 依次 forward
            // - 第一块: 使用快速 GEMM self-attention (invoke_prefill_attention)
            // - 后续块: 使用 paged attention 读取所有之前块的 KV cache
            //           (force_paged_attn=true 强制 full attn 层用 paged attn)
            // - SSM 状态 (DeltaNet): 在块间自然传递 (in-place 更新)
            // - 只在最后一块完成后做 final_norm → lm_head → sampling
            //
            // 数学正确性:
            // - write_kv_cache: start_pos = context_lens[0] - num_tokens
            //   chunk c 写入位置 [chunk_start, chunk_end)
            // - paged_attention: start_pos = total_context - chunk_size
            //   token i 看到 [0, start_pos + i] 的全部 KV → 正确 causal masking
            // ================================================================

            int chunk_size = max_chunk_size_;
            int num_chunks = (prefill_tokens + chunk_size - 1) / chunk_size;

            if (num_chunks > 1) {
                std::cout << "[ChunkedPrefill] " << prefill_tokens << " tokens → "
                          << num_chunks << " chunks of ≤" << chunk_size << std::endl;
            }

            // SSD 模式追踪: evict 起始位置 (FIFO: 从头开始 evict)
            int ssd_evict_cursor = 0;  // 下一个要 evict 的 logical block index

            for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                int chunk_start = prefill_start + chunk_idx * chunk_size;
                int chunk_end   = std::min(chunk_start + chunk_size, num_tokens);
                int chunk_len   = chunk_end - chunk_start;
                bool is_first   = (chunk_idx == 0 && prefill_start == 0);

                // ========================================================
                // 增量 Block 分配: 确保当前 chunk 需要的 blocks 已分配
                // ========================================================
                int blocks_needed_so_far = (chunk_end + 15) / 16;
                int blocks_to_add = blocks_needed_so_far - (int)ctx->block_table.size();

                if (blocks_to_add > 0) {
                    // 检查 GPU 是否有足够空闲 blocks
                    int free_blocks = kv_manager_->num_free_blocks();
                    if (free_blocks < blocks_to_add && block_ssd_store_) {
                        // SSD 模式: evict 最旧的 blocks 腾出空间
                        int to_evict = blocks_to_add - free_blocks;
                        // 至少 evict 一个 batch 以减少频繁 eviction
                        to_evict = std::max(to_evict, std::min(256, (int)ctx->block_table.size() - ssd_evict_cursor));

                        if (to_evict > 0 && ssd_evict_cursor + to_evict <= (int)ctx->block_table.size()) {
                            sync_stream_with_timeout(compute_stream_, 60, "ssd_evict_prefill");  // 确保 KV 写入完成

                            std::vector<int> evict_logical;
                            std::vector<int> phys_to_free;
                            for (int i = ssd_evict_cursor; i < ssd_evict_cursor + to_evict; i++) {
                                if (ctx->block_table[i] >= 0) {
                                    evict_logical.push_back(i);
                                    phys_to_free.push_back(ctx->block_table[i]);
                                }
                            }

                            if (!evict_logical.empty()) {
                                block_ssd_store_->evict_blocks(ctx->request_id, evict_logical,
                                                                ctx->block_table, *kv_manager_);
                                kv_manager_->free_blocks(phys_to_free);

                                // 标记为已 evict
                                for (int idx : evict_logical) {
                                    ctx->block_table[idx] = -1;
                                }
                                ctx->block_tracker.mark_evicted(ssd_evict_cursor, to_evict);
                                ctx->uses_ssd_blocks = true;

                                std::cout << "[SSD-Evict] Evicted " << evict_logical.size()
                                          << " blocks [" << ssd_evict_cursor << ", "
                                          << ssd_evict_cursor + to_evict << ") to SSD" << std::endl;
                            }
                            ssd_evict_cursor += to_evict;
                        }
                    }

                    // 如果 free blocks 不够 (非 SSD 模式), 尝试换出其它请求
                    while (kv_manager_->num_free_blocks() < blocks_to_add && kv_swapper_) {
                        int freed = try_swap_out_victim(active_requests, blocks_to_add);
                        if (freed == 0) break;
                    }

                    try {
                        auto new_blocks = kv_manager_->allocate_blocks(blocks_to_add);
                        ctx->block_table.insert(ctx->block_table.end(), new_blocks.begin(), new_blocks.end());
                        // 同步 block tracker
                        for (int b : new_blocks) ctx->block_tracker.push_block(b);
                    } catch (const std::runtime_error& e) {
                        std::cerr << "[Engine] Cannot allocate " << blocks_to_add
                                  << " blocks for chunk " << chunk_idx
                                  << " (free=" << kv_manager_->num_free_blocks() << ")" << std::endl;
                        ctx->is_finished = true;
                        break;
                    }
                }

                // ========================================================
                // 上传 Block Table (每 chunk 重新上传, 因为可能有新分配/eviction)
                // ========================================================
                cudaMemcpyAsync(d_block_tables_, ctx->block_table.data(),
                                ctx->block_table.size() * sizeof(int),
                                cudaMemcpyHostToDevice, compute_stream_);

                // 1. Embedding: 当前 chunk 的 tokens
                profiler_.begin("embedding", compute_stream_);
                cudaMemcpyAsync(d_pos_ids_,
                                ctx->prompt_tokens.data() + chunk_start,
                                chunk_len * sizeof(int),
                                cudaMemcpyHostToDevice, compute_stream_);
                ops::invoke_embedding_lookup(
                    d_hidden_states_, d_pos_ids_, model_->get_embed_tokens(),
                    chunk_len, config_.hidden_size, compute_stream_
                );
                profiler_.end("embedding", compute_stream_);

                // 1b. Vision: 如果有图像且是第一个 chunk, 运行视觉编码器并替换 image_pad 嵌入
                if (chunk_idx == 0) {
                    fprintf(stderr, "[Engine] Chunk 0: req=%lu processed_images=%zu has_vision=%d workspace=%s\n",
                            ctx->request_id, ctx->processed_images.size(),
                            (int)model_->has_vision(), d_vision_workspace_ ? "yes" : "no");
                    fflush(stderr);
                    // Count image_pad tokens in this chunk for alignment verification
                    int pad_count_in_chunk = 0;
                    for (int ci = 0; ci < chunk_len; ci++) {
                        if (ctx->prompt_tokens[chunk_start + ci] == 248056 ||
                            ctx->prompt_tokens[chunk_start + ci] == 248057)
                            pad_count_in_chunk++;
                    }
                    int total_features = 0;
                    for (auto& img : ctx->processed_images) total_features += img.num_output_tokens();
                    fprintf(stderr, "[Engine] Chunk 0: pad_tokens=%d total_features=%d prompt_len=%d\n",
                            pad_count_in_chunk, total_features, (int)ctx->prompt_tokens.size());
                    fflush(stderr);
                }
                if (chunk_idx == 0 && !ctx->processed_images.empty() &&
                    model_->has_vision() && d_vision_workspace_) {
                    profiler_.begin("vision", compute_stream_);
                    auto* venc = model_->get_vision_encoder();
                    int vision_offset = 0;
                    int img_idx = 0;
                    for (auto& img : ctx->processed_images) {
                        fprintf(stderr, "[Engine] ViT run: req=%lu img=%d/%zu tokens=%d\n",
                                ctx->request_id, img_idx,
                                ctx->processed_images.size(), img.num_output_tokens());
                        fflush(stderr);
                        // 运行 ViT + Merger
                        __nv_bfloat16* features = venc->forward(
                            img, d_vision_workspace_, vision_workspace_bytes_,
                            compute_stream_);
                        if (features) {
                            // Sync stream with timeout to surface async CUDA errors
                            bool vsync_ok = sync_stream_with_timeout(compute_stream_, 60, "vision_forward");
                            cudaError_t vsync_err = vsync_ok ? cudaSuccess : cudaGetLastError();
                            if (!vsync_ok || vsync_err != cudaSuccess) {
                                fprintf(stderr, "[Engine] Vision sync error img=%d: %s\n",
                                        img_idx, vsync_ok ? cudaGetErrorString(vsync_err) : "TIMEOUT");
                                fflush(stderr);
                                // Skip token replacement — LLM will run without this image's features
                            } else {
                            // 根据 is_video 选择替换 image_pad (248056) 或 video_pad (248057)
                            int token_id = img.is_video ? 248057 : 248056;
                            int num_vision_tokens = img.num_output_tokens();
                            fprintf(stderr, "[Engine] replace_image_tokens: offset=%d num=%d token_id=%d\n",
                                    vision_offset, num_vision_tokens, token_id);
                            fflush(stderr);

                            core::vision_ops::invoke_replace_image_tokens(
                                d_hidden_states_, features, d_pos_ids_,
                                chunk_len, token_id,
                                config_.hidden_size, vision_offset,
                                num_vision_tokens,
                                compute_stream_);

                            vision_offset += num_vision_tokens;
                            } // end else (no CUDA error)
                        }     // end if (features)
                        img_idx++;
                    }
                    profiler_.end("vision", compute_stream_);
                }

                // 2. Position IDs: 绝对位置 [chunk_start, chunk_end)
                std::vector<int> pos_ids(chunk_len);
                for (int i = 0; i < chunk_len; ++i) pos_ids[i] = chunk_start + i;
                cudaMemcpyAsync(d_pos_ids_, pos_ids.data(),
                                chunk_len * sizeof(int),
                                cudaMemcpyHostToDevice, compute_stream_);

                // 3. Context lens: 累计到本 chunk 末尾
                int ctx_len_for_chunk = chunk_end;
                cudaMemcpyAsync(d_context_lens_, &ctx_len_for_chunk,
                                sizeof(int), cudaMemcpyHostToDevice, compute_stream_);

                // 4. Forward
                profiler_.begin("forward", compute_stream_);

                if (ctx->uses_ssd_blocks && !is_first) {
                    // ========================================================
                    // Streaming Forward: block_table 有 -1 (SSD) 条目
                    // 不能用标准 paged attention, 改为逐层迭代 + streaming attention
                    // ========================================================
                    auto ssd_indices = ctx->block_tracker.get_ssd_logical_indices();
                    auto gpu_bt = ctx->block_tracker.get_gpu_block_table();
                    int num_full_attn = config_.num_full_attn_layers();

                    // 构建 StreamingAttnCtx
                    ops::StreamingAttnCtx sctx;
                    sctx.total_ssd_blocks = (int)ssd_indices.size();
                    sctx.ssd_tokens = sctx.total_ssd_blocks * 16;
                    sctx.staging_capacity = staging_num_blocks_;
                    sctx.staging_k = d_staging_k_;
                    sctx.staging_v = d_staging_v_;
                    sctx.d_staging_block_tables = nullptr;  // 下面设置
                    sctx.d_staging_context_lens = d_ssd_context_lens_;
                    sctx.partial_out = d_partial_out_;
                    sctx.partial_m = d_partial_m_;
                    sctx.partial_l = d_partial_l_;
                    sctx.partial_out2 = d_partial_out2_;
                    sctx.partial_m2 = d_partial_m2_;
                    sctx.partial_l2 = d_partial_l2_;

                    // GPU block table: compact (不含 -1)
                    sctx.gpu_num_blocks = (int)gpu_bt.size();
                    cudaMemcpyAsync(d_ssd_block_tables_, gpu_bt.data(),
                                    gpu_bt.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);
                    sctx.d_gpu_block_tables = d_ssd_block_tables_;

                    // GPU context lens: GPU tokens + current chunk tokens (causal masking 需要)
                    int gpu_ctx_len = ctx_len_for_chunk - sctx.ssd_tokens;
                    cudaMemcpyAsync(d_ssd_context_lens_, &gpu_ctx_len,
                                    sizeof(int), cudaMemcpyHostToDevice, compute_stream_);
                    sctx.d_gpu_context_lens = d_ssd_context_lens_;

                    // Staging block table: [0, 1, 2, ...]
                    {
                        std::vector<int> staging_bt(staging_num_blocks_);
                        for (int i = 0; i < staging_num_blocks_; i++) staging_bt[i] = i;
                        cudaMemcpyAsync(d_ssd_block_tables_ + gpu_bt.size(),
                                        staging_bt.data(),
                                        staging_num_blocks_ * sizeof(int),
                                        cudaMemcpyHostToDevice, compute_stream_);
                        sctx.d_staging_block_tables = d_ssd_block_tables_ + gpu_bt.size();
                    }

                    // SSD load callback: engine 负责从 SSD 加载 blocks 到 staging
                    sctx.load_ssd_batch = [this, &ctx, &ssd_indices](
                        int full_attn_idx, int batch_start, int batch_count) -> int {
                        int end = std::min(batch_start + batch_count, (int)ssd_indices.size());
                        std::vector<int> batch_indices(ssd_indices.begin() + batch_start,
                                                       ssd_indices.begin() + end);
                        return block_ssd_store_->load_blocks_for_layer(
                            ctx->request_id, full_attn_idx, batch_indices,
                            d_staging_k_, d_staging_v_);
                    };

                    // 逐层迭代 (与 model_->forward_prefill 等价, 但传入 streaming_ctx)
                    int fa_idx = 0, lin_idx = 0;
                    for (int li = 0; li < config_.num_hidden_layers; ++li) {
                        if (config_.is_full_attention(li)) {
                            // 每层重新加载 GPU context lens (streaming attention 内部可能修改)
                            cudaMemcpyAsync(d_ssd_context_lens_, &gpu_ctx_len,
                                            sizeof(int), cudaMemcpyHostToDevice, compute_stream_);

                            model_->get_layer(li).get_full_attn()->forward(
                                d_hidden_states_, d_pos_ids_, *kv_manager_,
                                d_block_tables_, d_context_lens_,
                                (int)ctx->block_table.size(), ctx_len_for_chunk,
                                chunk_len, fa_idx, d_workspace_, compute_stream_,
                                1 /* batch_size */, true /* force_paged_attn */,
                                &sctx);
                            fa_idx++;
                        } else {
                            __nv_bfloat16** lin_ssm = ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data() + lin_idx;
                            __nv_bfloat16** lin_conv = ctx->conv_states.empty() ? nullptr : ctx->conv_states.data() + lin_idx;
                            model_->get_layer(li).get_linear_attn()->forward(
                                d_hidden_states_,
                                lin_ssm ? lin_ssm[0] : nullptr,
                                lin_conv ? lin_conv[0] : nullptr,
                                chunk_len, d_workspace_, compute_stream_);
                            lin_idx++;
                        }
                    }

                } else {
                    // 标准路径: 所有 blocks 在 GPU
                    model_->forward_prefill(
                        d_hidden_states_, d_pos_ids_, *kv_manager_,
                        d_block_tables_, d_context_lens_,
                        (int)ctx->block_table.size(), ctx_len_for_chunk,
                        chunk_len,
                        ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data(),
                        ctx->conv_states.empty() ? nullptr : ctx->conv_states.data(),
                        d_workspace_, compute_stream_,
                        !is_first  // force_paged_attn for non-first chunks
                    );
                }
                profiler_.end("forward", compute_stream_);

                ctx->context_len = ctx_len_for_chunk;

                if (num_chunks > 1) {
                    std::cout << "[ChunkedPrefill]   chunk " << chunk_idx
                              << "/" << num_chunks << ": tokens ["
                              << chunk_start << ", " << chunk_end
                              << "), ctx_len=" << ctx_len_for_chunk
                              << (is_first ? " (self-attn)" :
                                  (ctx->uses_ssd_blocks ? " (streaming-attn)" : " (paged-attn)"))
                              << std::endl;
                }
            }
        
            // ---- Cache Store: prefill 完成后缓存 KV + SSM/Conv 到 SSD ----
            if (cache_engine_ && cache_engine_->is_enabled()) {
                cache_engine_->store_prefix(
                    ctx->prompt_tokens.data(), num_tokens,
                    *kv_manager_, d_block_tables_,
                    (int)ctx->block_table.size(),
                    ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data(),
                    ctx->conv_states.empty() ? nullptr : ctx->conv_states.data(),
                    d_workspace_, compute_stream_);
            }
        } // end if (prefill_tokens > 0)

        if (prefill_tokens > 0) {
            // 6. 最后的 LayerNorm (在最后一个 chunk 的最后一个 token 上)
            // d_hidden_states_ 里只有最后一个 chunk 的数据, 取其最后一个 token
            int last_chunk_len = prefill_tokens % max_chunk_size_;
            if (last_chunk_len == 0) last_chunk_len = max_chunk_size_;
            last_chunk_len = std::min(last_chunk_len, prefill_tokens);
            profiler_.begin("final_norm", compute_stream_);
            __nv_bfloat16* norm_out = d_workspace_;
            ops::invoke_rmsnorm(norm_out,
                                d_hidden_states_ + (last_chunk_len - 1) * config_.hidden_size,
                                model_->get_norm_weight(), config_.rms_norm_eps,
                                1, config_.hidden_size, compute_stream_);
            profiler_.end("final_norm", compute_stream_);

            // 7. LM Head (计算最后一个 token 的 logits)
            profiler_.begin("lm_head", compute_stream_);
            int vocab_size = config_.vocab_size;
            __nv_bfloat16* logits = norm_out + config_.hidden_size;
            ops::invoke_dense_gemv(norm_out, model_->get_lm_head(), logits, vocab_size, config_.hidden_size, compute_stream_);
            profiler_.end("lm_head", compute_stream_);
            
            // 8. 采样
            profiler_.begin("sample", compute_stream_);
            int next_token = sample_token(logits, vocab_size,
                                          ctx->temperature, ctx->top_p, ctx->top_k,
                                          ctx->min_p,
                                          ctx->repeat_penalty, ctx->frequency_penalty,
                                          ctx->presence_penalty, ctx->seed,
                                          ctx->generated_tokens,
                                          compute_stream_);
            profiler_.end("sample", compute_stream_);
            
            ctx->generated_tokens.push_back(next_token);
            ctx->context_len++;
            
            // 推送响应 token 给 Python 前端
            {
                bool eos = (next_token == 248046 || next_token == 248044);
                ipc::InferenceResponse resp{};
                resp.request_id  = ctx->request_id;
                resp.token_id    = next_token;
                resp.is_finished = eos;
                resp.error_code  = 0;
                while (!ipc_resp_queue_->push(resp))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                if (eos) ctx->is_finished = true;
            }
            
            profiler_.request_prefill_done(num_tokens);
            std::cout << "Prefill tok=" << next_token
                      << " txt=\"" << token_to_log_text(next_token) << "\""
                      << " (" << num_tokens << " tokens)" << std::endl;
            profiler_.print_step_report(0);

            // ---- LMCache Monitor: 记录请求级指标 ----
            if (cache_engine_ && cache_engine_->is_enabled()) {
                cache_engine_->record_request(num_tokens, cached_tokens, prefill_tokens);
            }
        } else {
            // 完全缓存命中 (prefill_tokens == 0):
            // KV + SSM/Conv 已从 SSD 恢复, 跳过 forward
            // 用占位符标记 prefill 完成, 下一次 step() 会进入 decode 生成第一个 token
            ctx->generated_tokens.push_back(-1);  // 占位, decode 会使用 last prompt token
            profiler_.request_prefill_done(num_tokens);
            std::cout << "[Cache] Full hit! Skipped prefill (" << num_tokens << " tokens)" << std::endl;

            // ---- LMCache Monitor: 完全命中 ----
            if (cache_engine_ && cache_engine_->is_enabled()) {
                cache_engine_->record_request(num_tokens, num_tokens, 0);
            }
        }
        
    } else {
        // --------------------------------------------------------------------
        // Decode 阶段
        // --------------------------------------------------------------------
        if ((int)ctx->generated_tokens.size() >= ctx->max_new_tokens) {
            // 通知 Python 前端：本请求已完成
            ipc::InferenceResponse fin{};
            fin.request_id  = ctx->request_id;
            fin.token_id    = -1;
            fin.is_finished = true;
            fin.error_code  = 0;
            while (!ipc_resp_queue_->push(fin))
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            ctx->is_finished = true;
            profiler_.request_done();
            profiler_.print_request_summary();
            std::cout << "Request " << ctx->request_id << " finished (max_new_tokens)." << std::endl;
            return;
        }

        int last_token = ctx->generated_tokens.back();
        // 如果是 full cache hit 的占位 (-1), 使用最后一个 prompt token
        if (last_token == -1) {
            last_token = ctx->prompt_tokens.back();
        }

        const int hs = config_.hidden_size;
        const int vocab_size = config_.vocab_size;

        // ================================================================
        // 分支: MTP T=(N+1) 投机验证 vs 标准 T=1 decode
        // MTP 条件: 有 MTP 权重 + 有 draft tokens + 不在 SSD streaming 模式
        // ================================================================
        if (mtp_kv_manager_ && !ctx->draft_tokens.empty() && !ctx->uses_ssd_blocks) {
            // ============================================================
            // MTP Speculative Decode: T=(N+1) All-or-Nothing Verify
            // ============================================================
            const int N = num_mtp_drafts_;
            const int T = N + 1;  // e.g. N=3 → T=4: [last_token, D0, D1, D2]
            profiler_.begin("embedding", compute_stream_);

            // 1. Embed [last_token, D0, D1, D2] → d_hidden_states_ [T, hs]
            int tokens_T[4];
            tokens_T[0] = last_token;
            for (int i = 0; i < N; i++) tokens_T[i + 1] = ctx->draft_tokens[i];
            cudaMemcpyAsync(d_pos_ids_, tokens_T, T * sizeof(int), cudaMemcpyHostToDevice, compute_stream_);
            ops::invoke_embedding_lookup(
                d_hidden_states_, d_pos_ids_, model_->get_embed_tokens(),
                T, hs, compute_stream_);
            profiler_.end("embedding", compute_stream_);

            // 2. Pos IDs: [context_len - 1, ..., context_len + N - 1]
            int pos_ids_T[4];
            for (int i = 0; i < T; i++) pos_ids_T[i] = ctx->context_len - 1 + i;
            cudaMemcpyAsync(d_pos_ids_, pos_ids_T, T * sizeof(int), cudaMemcpyHostToDevice, compute_stream_);

            // 3. Block 分配: 确保 blocks 覆盖到 position context_len + N - 1
            int ctx_len_verify = ctx->context_len + N;  // 包含 T 个新 token
            {
                int blocks_needed = (ctx_len_verify + 15) / 16;
                while ((int)ctx->block_table.size() < blocks_needed) {
                    if (kv_manager_->num_free_blocks() < 1 && kv_swapper_) {
                        try_swap_out_victim(active_requests, 1);
                    }
                    try {
                        auto new_blks = kv_manager_->allocate_blocks(1);
                        ctx->block_table.push_back(new_blks[0]);
                        ctx->block_tracker.push_block(new_blks[0]);
                    } catch (const std::runtime_error& e) {
                        std::cerr << "Out of KV Cache memory during MTP verify!" << std::endl;
                        ctx->is_finished = true;
                        return;
                    }
                }
            }

            // 4. Upload block_tables, context_lens
            cudaMemcpyAsync(d_block_tables_, ctx->block_table.data(),
                            ctx->block_table.size() * sizeof(int),
                            cudaMemcpyHostToDevice, compute_stream_);
            cudaMemcpyAsync(d_context_lens_, &ctx_len_verify, sizeof(int),
                            cudaMemcpyHostToDevice, compute_stream_);

            // 5. Forward T=2: 逐层迭代, LinearAttn 层带 SSM/Conv checkpoint
            try {
                profiler_.begin("forward", compute_stream_);
                int fa_idx = 0, lin_idx = 0;
                for (int li = 0; li < config_.num_hidden_layers; ++li) {
                    if (config_.is_full_attention(li)) {
                        model_->get_layer(li).get_full_attn()->forward(
                            d_hidden_states_, d_pos_ids_, *kv_manager_,
                            d_block_tables_, d_context_lens_,
                            (int)ctx->block_table.size(), ctx_len_verify,
                            T /*num_tokens*/, fa_idx, d_workspace_, compute_stream_,
                            1 /*batch_size*/, true /*force_paged_attn*/);
                        fa_idx++;
                    } else {
                        __nv_bfloat16* ssm_ckpt = d_ssm_checkpoints_
                                          + lin_idx * ssm_elems_per_layer_;
                        __nv_bfloat16* conv_ckpt = d_conv_checkpoints_
                                                   + lin_idx * conv_elems_per_layer_;
                        model_->get_layer(li).get_linear_attn()->forward(
                            d_hidden_states_,
                            ctx->ssm_states[lin_idx],
                            ctx->conv_states[lin_idx],
                            T /*num_tokens*/, d_workspace_, compute_stream_,
                            1 /*batch_size*/,
                            nullptr, nullptr,
                            ssm_ckpt, conv_ckpt);
                        lin_idx++;
                    }
                    // Per-layer sync for unified memory stability
                    cudaStreamSynchronize(compute_stream_);
                }
                profiler_.end("forward", compute_stream_);
            } catch (const std::exception& e) {
                std::cerr << "MTP verify forward failed: " << e.what() << std::endl;
                ctx->is_finished = true;
                return;
            }

            // 6. Norm + LM head for all T tokens
            //    d_hidden_states_ = [T, hs] (last layer output, pre-norm)
            profiler_.begin("final_norm", compute_stream_);
            __nv_bfloat16* norm_out_T = d_workspace_;
            ops::invoke_rmsnorm(norm_out_T, d_hidden_states_, model_->get_norm_weight(),
                                config_.rms_norm_eps, T, hs, compute_stream_);
            profiler_.end("final_norm", compute_stream_);

            profiler_.begin("lm_head", compute_stream_);
            __nv_bfloat16* logits_T = norm_out_T + T * hs;
            ops::invoke_dense_gemm(norm_out_T, model_->get_lm_head(), logits_T,
                                   T, vocab_size, hs, compute_stream_);
            profiler_.end("lm_head", compute_stream_);

            // 7. Sample all T logit positions via argmax
            profiler_.begin("sample", compute_stream_);
            int verify[4];
            for (int i = 0; i < T; i++) {
                verify[i] = sample_argmax(logits_T + (size_t)i * vocab_size,
                                          vocab_size, compute_stream_);
            }
            profiler_.end("sample", compute_stream_);

            mtp_verify_count_++;

            // 8. All-or-nothing accept: check if all N drafts match
            bool all_match = true;
            for (int i = 0; i < N; i++) {
                if (verify[i] != ctx->draft_tokens[i]) {
                    all_match = false;
                    break;
                }
            }

            // Helper lambda: chain N MTP calls to generate N draft tokens
            auto generate_mtp_drafts = [&](__nv_bfloat16* main_hidden,
                                           int first_token, int start_pos) {
                ctx->draft_tokens.clear();
                __nv_bfloat16* h = main_hidden;
                int tok = first_token;
                int pos = start_pos;
                for (int di = 0; di < N; di++) {
                    int mtp_ctx_fwd = ctx->mtp_context_len + 1;
                    int mtp_blks = (mtp_ctx_fwd + 15) / 16;
                    while ((int)ctx->mtp_block_table.size() < mtp_blks) {
                        auto mb = mtp_kv_manager_->allocate_blocks(1);
                        ctx->mtp_block_table.push_back(mb[0]);
                    }
                    cudaMemcpyAsync(d_mtp_block_tables_,
                                    ctx->mtp_block_table.data(),
                                    ctx->mtp_block_table.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);
                    cudaMemcpyAsync(d_mtp_context_lens_, &mtp_ctx_fwd,
                                    sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);

                    __nv_bfloat16* mtp_hidden = nullptr;
                    __nv_bfloat16* mtp_logits = model_->mtp_forward(
                        h, tok, pos,
                        *mtp_kv_manager_,
                        d_mtp_block_tables_, d_mtp_context_lens_,
                        (int)ctx->mtp_block_table.size(), mtp_ctx_fwd,
                        d_workspace_, compute_stream_,
                        (di < N - 1) ? &mtp_hidden : nullptr);

                    int draft = sample_argmax(mtp_logits, vocab_size,
                                              compute_stream_);
                    ctx->draft_tokens.push_back(draft);
                    ctx->mtp_context_len++;

                    if (di < N - 1) {
                        h = mtp_hidden;
                        tok = draft;
                        pos++;
                    }
                }
            };

            if (all_match) {
                // ============================================================
                // ACCEPT: all N drafts correct, emit N+1 tokens
                // ============================================================
                mtp_accept_count_++;

                // Emit N draft tokens
                for (int i = 0; i < N; i++) {
                    int dtok = ctx->draft_tokens[i];
                    ctx->generated_tokens.push_back(dtok);
                    ctx->context_len++;
                    bool eos_d = (dtok == 248046 || dtok == 248044);
                    {
                        ipc::InferenceResponse resp{};
                        resp.request_id  = ctx->request_id;
                        resp.token_id    = dtok;
                        resp.is_finished = eos_d;
                        resp.error_code  = 0;
                        while (!ipc_resp_queue_->push(resp))
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    if (eos_d) {
                        ctx->is_finished = true;
                        ctx->draft_tokens.clear();
                        profiler_.request_done();
                        profiler_.print_request_summary();
                        std::cout << "Decode step " << ctx->generated_tokens.size()
                                  << ". tok=" << dtok
                                  << " (MTP accept, EOS at draft " << i << ")" << std::endl;
                        return;
                    }
                }

                // Emit next token (verify[N] = prediction after all drafts)
                int next_token = verify[N];
                ctx->generated_tokens.push_back(next_token);
                ctx->context_len++;
                bool eos_n = (next_token == 248046 || next_token == 248044);
                bool done_n = (int)ctx->generated_tokens.size() >= ctx->max_new_tokens || eos_n;
                {
                    ipc::InferenceResponse resp{};
                    resp.request_id  = ctx->request_id;
                    resp.token_id    = next_token;
                    resp.is_finished = done_n;
                    resp.error_code  = 0;
                    while (!ipc_resp_queue_->push(resp))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                if (done_n) {
                    ctx->is_finished = true;
                    ctx->draft_tokens.clear();
                    profiler_.request_done();
                    profiler_.print_request_summary();
                    std::cout << "Decode step " << ctx->generated_tokens.size()
                              << ". tok=" << next_token << " (MTP accept+done)" << std::endl;
                    return;
                }

                // Generate N new MTP drafts from hidden_states[N] (last accepted pos)
                __nv_bfloat16* h_for_mtp = d_hidden_states_ + N * hs;
                ctx->mtp_pos = ctx->context_len - 1;
                generate_mtp_drafts(h_for_mtp, next_token, ctx->mtp_pos);

                for (int i = 0; i < T; i++) profiler_.request_decode_step();
                int step_n = (int)ctx->generated_tokens.size();
                if (step_n % 10 == 0) profiler_.print_step_report(step_n);
                float accept_rate = mtp_verify_count_ > 0
                    ? (float)mtp_accept_count_ / mtp_verify_count_ * 100.f : 0.f;
                std::cout << "Decode step " << step_n
                          << ". tok=" << next_token
                          << " txt=\"" << token_to_log_text(next_token) << "\""
                          << " (MTP ACCEPT " << T << "tok, rate=" << accept_rate << "%)" << std::endl;

            } else {
                // ============================================================
                // REJECT: at least one draft wrong, rollback SSM/Conv, emit 1 token
                // ============================================================

                // Rollback SSM/Conv from checkpoint
                for (int li = 0; li < num_linear_layers_; ++li) {
                    __nv_bfloat16* ssm_ckpt = d_ssm_checkpoints_
                                      + li * ssm_elems_per_layer_;
                    __nv_bfloat16* conv_ckpt = d_conv_checkpoints_
                                               + li * conv_elems_per_layer_;
                    cudaMemcpyAsync(ctx->ssm_states[li], ssm_ckpt,
                                    ssm_elems_per_layer_ * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, compute_stream_);
                    cudaMemcpyAsync(ctx->conv_states[li], conv_ckpt,
                                    conv_elems_per_layer_ * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, compute_stream_);
                }

                // Emit corrected token (verify[0])
                int corrected = verify[0];
                ctx->generated_tokens.push_back(corrected);
                ctx->context_len++;
                // 注: context_len 只 +1, draft 的 KV 在下次 verify 时被覆写

                bool eos = (corrected == 248046 || corrected == 248044);
                bool done = (int)ctx->generated_tokens.size() >= ctx->max_new_tokens || eos;
                {
                    ipc::InferenceResponse resp{};
                    resp.request_id  = ctx->request_id;
                    resp.token_id    = corrected;
                    resp.is_finished = done;
                    resp.error_code  = 0;
                    while (!ipc_resp_queue_->push(resp))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
                if (done) {
                    ctx->is_finished = true;
                    ctx->draft_tokens.clear();
                    profiler_.request_done();
                    profiler_.print_request_summary();
                    std::cout << "Decode step " << ctx->generated_tokens.size()
                              << ". tok=" << corrected << " (MTP reject+done)" << std::endl;
                    return;
                }

                // Rollback MTP context: only 1st entry valid, rest stale
                ctx->mtp_context_len -= (N - 1);

                // Generate N new MTP drafts from hidden_states[0]
                __nv_bfloat16* h_for_mtp = d_hidden_states_;
                ctx->mtp_pos = ctx->context_len - 1;
                generate_mtp_drafts(h_for_mtp, corrected, ctx->mtp_pos);

                profiler_.request_decode_step();
                int step_n = (int)ctx->generated_tokens.size();
                if (step_n % 10 == 0) profiler_.print_step_report(step_n);
                float accept_rate = mtp_verify_count_ > 0
                    ? (float)mtp_accept_count_ / mtp_verify_count_ * 100.f : 0.f;
                std::cout << "Decode step " << step_n
                          << ". tok=" << corrected
                          << " txt=\"" << token_to_log_text(corrected) << "\""
                          << " (MTP REJECT 1tok, rate=" << accept_rate << "%)" << std::endl;
            }

        } else {
            // ============================================================
            // 标准 Decode (T=1) — 也作为 MTP bootstrap (首次 decode)
            // ============================================================
            int num_tokens = 1;
            int step_num = (int)ctx->generated_tokens.size() + 1;

            // 1. 准备输入数据 (只有 1 个 token)
            profiler_.begin("embedding", compute_stream_);
            cudaMemcpyAsync(d_pos_ids_, &last_token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_);
            ops::invoke_embedding_lookup(
                d_hidden_states_, d_pos_ids_, model_->get_embed_tokens(),
                num_tokens, hs, compute_stream_);
            profiler_.end("embedding", compute_stream_);

            // 2. 位置编码 (context_len - 1)
            int current_pos = ctx->context_len - 1;
            cudaMemcpyAsync(d_pos_ids_, &current_pos, sizeof(int), cudaMemcpyHostToDevice, compute_stream_);

            // 3. Block 分配
            if (ctx->context_len > (int)(ctx->block_table.size() * 16)) {
                // GPU blocks 不够: 先尝试 SSD eviction
                if (kv_manager_->num_free_blocks() < 1 && block_ssd_store_ && ctx->uses_ssd_blocks) {
                    int evict_start = 0;
                    for (int i = 0; i < (int)ctx->block_table.size(); i++) {
                        if (ctx->block_table[i] >= 0) { evict_start = i; break; }
                    }
                    int to_evict = std::min(64, (int)ctx->block_table.size() - evict_start);
                    if (to_evict > 0) {
                        sync_stream_with_timeout(compute_stream_, 60, "ssd_evict_decode");
                        std::vector<int> evict_logical, phys_to_free;
                        for (int i = evict_start; i < evict_start + to_evict; i++) {
                            if (ctx->block_table[i] >= 0) {
                                evict_logical.push_back(i);
                                phys_to_free.push_back(ctx->block_table[i]);
                            }
                        }
                        if (!evict_logical.empty()) {
                            block_ssd_store_->evict_blocks(ctx->request_id, evict_logical,
                                                            ctx->block_table, *kv_manager_);
                            kv_manager_->free_blocks(phys_to_free);
                            for (int idx : evict_logical) ctx->block_table[idx] = -1;
                            ctx->block_tracker.mark_evicted(evict_start, to_evict);
                        }
                    }
                }
                if (kv_manager_->num_free_blocks() < 1 && kv_swapper_) {
                    try_swap_out_victim(active_requests, 1);
                }
                try {
                    auto new_blocks = kv_manager_->allocate_blocks(1);
                    ctx->block_table.push_back(new_blocks[0]);
                    ctx->block_tracker.push_block(new_blocks[0]);
                } catch (const std::runtime_error& e) {
                    std::cerr << "Out of KV Cache memory! (even after swap attempt)" << std::endl;
                    ctx->is_finished = true;
                    return;
                }
            }

            cudaMemcpyAsync(d_block_tables_, ctx->block_table.data(),
                            ctx->block_table.size() * sizeof(int),
                            cudaMemcpyHostToDevice, compute_stream_);
            cudaMemcpyAsync(d_context_lens_, &ctx->context_len, sizeof(int),
                            cudaMemcpyHostToDevice, compute_stream_);

            // 4. Forward
            try {
                profiler_.begin("forward", compute_stream_);

                if (ctx->uses_ssd_blocks) {
                    // Streaming Decode: SSD blocks, 逐层迭代 + streaming attention
                    auto ssd_indices = ctx->block_tracker.get_ssd_logical_indices();
                    auto gpu_bt = ctx->block_tracker.get_gpu_block_table();

                    ops::StreamingAttnCtx sctx;
                    sctx.total_ssd_blocks = (int)ssd_indices.size();
                    sctx.ssd_tokens = sctx.total_ssd_blocks * 16;
                    sctx.staging_capacity = staging_num_blocks_;
                    sctx.staging_k = d_staging_k_;
                    sctx.staging_v = d_staging_v_;
                    sctx.d_staging_block_tables = nullptr;
                    sctx.d_staging_context_lens = d_ssd_context_lens_;
                    sctx.partial_out = d_partial_out_;
                    sctx.partial_m = d_partial_m_;
                    sctx.partial_l = d_partial_l_;
                    sctx.partial_out2 = d_partial_out2_;
                    sctx.partial_m2 = d_partial_m2_;
                    sctx.partial_l2 = d_partial_l2_;

                    sctx.gpu_num_blocks = (int)gpu_bt.size();
                    cudaMemcpyAsync(d_ssd_block_tables_, gpu_bt.data(),
                                    gpu_bt.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);
                    sctx.d_gpu_block_tables = d_ssd_block_tables_;

                    int gpu_ctx_len = ctx->context_len - sctx.ssd_tokens;
                    cudaMemcpyAsync(d_ssd_context_lens_, &gpu_ctx_len,
                                    sizeof(int), cudaMemcpyHostToDevice, compute_stream_);
                    sctx.d_gpu_context_lens = d_ssd_context_lens_;

                    {
                        std::vector<int> staging_bt(staging_num_blocks_);
                        for (int i = 0; i < staging_num_blocks_; i++) staging_bt[i] = i;
                        cudaMemcpyAsync(d_ssd_block_tables_ + gpu_bt.size(),
                                        staging_bt.data(),
                                        staging_num_blocks_ * sizeof(int),
                                        cudaMemcpyHostToDevice, compute_stream_);
                        sctx.d_staging_block_tables = d_ssd_block_tables_ + gpu_bt.size();
                    }

                    sctx.load_ssd_batch = [this, &ctx, &ssd_indices](
                        int full_attn_idx, int batch_start, int batch_count) -> int {
                        int end = std::min(batch_start + batch_count, (int)ssd_indices.size());
                        std::vector<int> batch_indices(ssd_indices.begin() + batch_start,
                                                       ssd_indices.begin() + end);
                        return block_ssd_store_->load_blocks_for_layer(
                            ctx->request_id, full_attn_idx, batch_indices,
                            d_staging_k_, d_staging_v_);
                    };

                    int fa_idx = 0, lin_idx = 0;
                    for (int li = 0; li < config_.num_hidden_layers; ++li) {
                        if (config_.is_full_attention(li)) {
                            cudaMemcpyAsync(d_ssd_context_lens_, &gpu_ctx_len,
                                            sizeof(int), cudaMemcpyHostToDevice, compute_stream_);
                            model_->get_layer(li).get_full_attn()->forward(
                                d_hidden_states_, d_pos_ids_, *kv_manager_,
                                d_block_tables_, d_context_lens_,
                                (int)ctx->block_table.size(), ctx->context_len,
                                num_tokens, fa_idx, d_workspace_, compute_stream_,
                                1, false, &sctx);
                            fa_idx++;
                        } else {
                            __nv_bfloat16** lin_ssm = ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data() + lin_idx;
                            __nv_bfloat16** lin_conv = ctx->conv_states.empty() ? nullptr : ctx->conv_states.data() + lin_idx;
                            model_->get_layer(li).get_linear_attn()->forward(
                                d_hidden_states_,
                                lin_ssm ? lin_ssm[0] : nullptr,
                                lin_conv ? lin_conv[0] : nullptr,
                                num_tokens, d_workspace_, compute_stream_);
                            lin_idx++;
                        }
                    }

                } else {
                    // 标准 decode 路径 (forward_decode 内含逐层 sync)
                    model_->forward_decode(
                        d_hidden_states_, d_pos_ids_, *kv_manager_,
                        d_block_tables_, d_context_lens_,
                        (int)ctx->block_table.size(), ctx->context_len,
                        num_tokens,
                        ctx->ssm_states.empty() ? nullptr : ctx->ssm_states.data(),
                        ctx->conv_states.empty() ? nullptr : ctx->conv_states.data(),
                        d_workspace_, compute_stream_
                    );
                }
                profiler_.end("forward", compute_stream_);
            } catch (const std::exception& e) {
                std::cerr << "Decode forward failed: " << e.what() << std::endl;
                ctx->is_finished = true;
                return;
            }

            // GPU 异步错误检查 (forward_decode 内已逐层 sync, 此处仅确认状态)
            {
                cudaError_t gpu_err = cudaGetLastError();
                if (gpu_err != cudaSuccess) {
                    fprintf(stderr, "[Engine] CUDA error after decode step %d: %s\n",
                            (int)ctx->generated_tokens.size(), cudaGetErrorString(gpu_err));
                    fflush(stderr);
                    ctx->is_finished = true;
                    ipc::InferenceResponse resp{};
                    resp.request_id = ctx->request_id;
                    resp.token_id = 0;
                    resp.is_finished = true;
                    resp.error_code = -3;  // CUDA error
                    while (!ipc_resp_queue_->push(resp))
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    return;
                }
            }

            // 5. Final Norm + LM Head + Sample (T=1)

            profiler_.begin("final_norm", compute_stream_);
            __nv_bfloat16* norm_out = d_workspace_;
            ops::invoke_rmsnorm(norm_out, d_hidden_states_, model_->get_norm_weight(),
                                config_.rms_norm_eps, 1, hs, compute_stream_);
            profiler_.end("final_norm", compute_stream_);

            profiler_.begin("lm_head", compute_stream_);
            __nv_bfloat16* logits = norm_out + hs;
            ops::invoke_dense_gemv(norm_out, model_->get_lm_head(), logits,
                                   vocab_size, hs, compute_stream_);
            profiler_.end("lm_head", compute_stream_);

            profiler_.begin("sample", compute_stream_);
            int next_token = sample_token(logits, vocab_size,
                                          ctx->temperature, ctx->top_p, ctx->top_k,
                                          ctx->min_p,
                                          ctx->repeat_penalty, ctx->frequency_penalty,
                                          ctx->presence_penalty, ctx->seed,
                                          ctx->generated_tokens,
                                          compute_stream_);
            profiler_.end("sample", compute_stream_);

            ctx->generated_tokens.push_back(next_token);
            ctx->context_len++;

            // 推送响应 token 给 Python 前端
            bool eos = (next_token == 248046 || next_token == 248044);
            bool done = (int)ctx->generated_tokens.size() >= ctx->max_new_tokens || eos;

            // N-gram 重复检测: 检测 4-gram 连续重复 >= 8 次 → 强制终止
            // 避免模型陷入无限重复循环浪费计算资源
            if (!done && (int)ctx->generated_tokens.size() >= 32) {
                const auto& gen = ctx->generated_tokens;
                int sz = (int)gen.size();
                // 检查最后 32 个 token 是否为 4-gram 重复
                bool is_repeating = true;
                for (int ri = 1; ri < 8 && is_repeating; ri++) {
                    for (int gi = 0; gi < 4; gi++) {
                        if (gen[sz - 1 - gi] != gen[sz - 1 - gi - 4 * ri]) {
                            is_repeating = false;
                            break;
                        }
                    }
                }
                if (is_repeating) {
                    done = true;
                    eos = true;
                    std::cout << "[Engine] Repetition detected at step "
                              << sz << ", force stopping." << std::endl;
                }
            }

            {
                ipc::InferenceResponse resp{};
                resp.request_id  = ctx->request_id;
                resp.token_id    = next_token;
                resp.is_finished = done;
                resp.error_code  = 0;
                while (!ipc_resp_queue_->push(resp))
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            if (done) {
                ctx->is_finished = true;
                profiler_.request_done();
                profiler_.print_request_summary();
            }

            // ---- MTP Bootstrap: 首次 decode 后链式产出 N 个 draft ----
            if (mtp_kv_manager_ && !ctx->is_finished && !ctx->uses_ssd_blocks
                && ctx->draft_tokens.empty()) {
                const int N = num_mtp_drafts_;
                __nv_bfloat16* h = d_hidden_states_;
                int tok = next_token;
                ctx->mtp_pos = ctx->context_len - 1;
                int pos = ctx->mtp_pos;

                for (int di = 0; di < N; di++) {
                    int mtp_ctx_fwd = ctx->mtp_context_len + 1;
                    int mtp_blks = (mtp_ctx_fwd + 15) / 16;
                    while ((int)ctx->mtp_block_table.size() < mtp_blks) {
                        auto mb = mtp_kv_manager_->allocate_blocks(1);
                        ctx->mtp_block_table.push_back(mb[0]);
                    }
                    cudaMemcpyAsync(d_mtp_block_tables_, ctx->mtp_block_table.data(),
                                    ctx->mtp_block_table.size() * sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);
                    cudaMemcpyAsync(d_mtp_context_lens_, &mtp_ctx_fwd, sizeof(int),
                                    cudaMemcpyHostToDevice, compute_stream_);

                    __nv_bfloat16* mtp_hidden = nullptr;
                    __nv_bfloat16* mtp_logits = model_->mtp_forward(
                        h, tok, pos,
                        *mtp_kv_manager_,
                        d_mtp_block_tables_, d_mtp_context_lens_,
                        (int)ctx->mtp_block_table.size(), mtp_ctx_fwd,
                        d_workspace_, compute_stream_,
                        (di < N - 1) ? &mtp_hidden : nullptr);

                    int draft = sample_argmax(mtp_logits, vocab_size, compute_stream_);
                    ctx->draft_tokens.push_back(draft);
                    ctx->mtp_context_len++;

                    if (di < N - 1) {
                        h = mtp_hidden;
                        tok = draft;
                        pos++;
                    }
                }
            }

            profiler_.request_decode_step();
            int step_n = (int)ctx->generated_tokens.size();
            if (step_n % 10 == 0) {
                profiler_.print_step_report(step_n);
            }
            std::cout << "Decode step " << step_n
                      << ". tok=" << next_token
                      << " txt=\"" << token_to_log_text(next_token) << "\""
                      << std::endl;
            fflush(stdout);
        }
    }
}

std::string InferenceEngine::token_to_log_text(int token_id) const {
    if (!log_tokenizer_ready_) return "";

    std::string piece = log_tokenizer_.decode(token_id);
    if (piece.empty()) return "";

    std::ostringstream oss;
    for (unsigned char ch : piece) {
        if (ch == '\n') oss << "\\n";
        else if (ch == '\r') oss << "\\r";
        else if (ch == '\t') oss << "\\t";
        else if (std::isprint(ch)) oss << static_cast<char>(ch);
        else {
            oss << "\\x" << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(ch) << std::dec;
        }
    }
    return oss.str();
}

int InferenceEngine::sample_argmax(__nv_bfloat16* logits, int vocab_size, cudaStream_t stream) {
    // GPU argmax: 单 block 1024 线程 reduce，结果写到 managed memory
    ops::invoke_argmax(logits, d_argmax_result_, vocab_size, stream);
    if (!sync_stream_with_timeout(stream, 60, "sample_argmax")) {
        fprintf(stderr, "[Engine] FATAL: GPU hang detected in sample_argmax, returning EOS\n");
        fflush(stderr);
        return 248046; // EOS token — force stop
    }
    return *d_argmax_result_;
}

// ---------------------------------------------------------------------------
// sample_token: temperature + top_k + top_p 采样
//   temperature <= 0 → 退化为 argmax (贪心解码)
//   top_k: 只考虑概率最高的 K 个 token
//   top_p: 在 top_k 结果中，按概率降序累加到 p 后截断
//
// CPU 侧实现 (Jetson 统一内存, sync 后 CPU 可直接读 GPU buffer)
// vocab_size=248320, partial_sort O(n) → ~0.5ms, 相对 decode 200ms 开销 <0.3%
// ---------------------------------------------------------------------------
int InferenceEngine::sample_token(__nv_bfloat16* logits, int vocab_size,
                                   float temperature, float top_p, int top_k,
                                   float min_p,
                                   float repeat_penalty, float frequency_penalty,
                                   float presence_penalty, int64_t seed,
                                   const std::vector<int>& generated_tokens,
                                   cudaStream_t stream) {
    // 贪心: temperature <= 0 或 top_k == 1
    if (temperature <= 0.0f || top_k == 1) {
        return sample_argmax(logits, vocab_size, stream);
    }

    // 确定性采样: seed >= 0 时重置 RNG
    if (seed >= 0) {
        sampling_rng_.seed(static_cast<uint64_t>(seed) + generated_tokens.size());
    }

    // 1. 等待 GPU 完成, cudaMemcpy logits 到 host staging buffer
    if (!sync_stream_with_timeout(stream, 60, "sample_token")) {
        fprintf(stderr, "[Engine] FATAL: GPU hang detected in sample_token, returning EOS\n");
        fflush(stderr);
        return 248046; // EOS token — force stop
    }
    cudaMemcpy(sampling_logits_bf16_.data(), logits,
               vocab_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // 2. BF16 → float (不做 temperature 缩放, penalty 在 raw logits 上操作)
    for (int i = 0; i < vocab_size; ++i) {
        sampling_logits_[i] = __bfloat162float(sampling_logits_bf16_[i]);
    }

    // 2.5 重复惩罚 / 频率惩罚 / 存在性惩罚 — 在 raw logits 上操作
    // (OpenAI 规范: penalty 在 temperature 缩放之前应用)
    bool has_penalty = (repeat_penalty != 1.0f) || (frequency_penalty != 0.0f) || (presence_penalty != 0.0f);
    if (has_penalty && !generated_tokens.empty()) {
        // 统计已生成 token 的出现次数
        std::unordered_map<int, int> token_counts;
        for (int tok : generated_tokens) {
            if (tok >= 0 && tok < vocab_size) token_counts[tok]++;
        }
        for (auto& [tok_id, count] : token_counts) {
            float& logit = sampling_logits_[tok_id];
            // Repeat penalty (Llama/Qwen style): logit /= penalty if >0, *penalty if <0
            if (repeat_penalty != 1.0f) {
                if (logit > 0.0f) logit /= repeat_penalty;
                else              logit *= repeat_penalty;
            }
            // Frequency penalty (OpenAI style): logit -= freq_penalty * count
            logit -= frequency_penalty * count;
            // Presence penalty (OpenAI style): logit -= pres_penalty * sign(count)
            logit -= presence_penalty;
        }
    }

    // 3. Temperature 缩放 (在 penalty 之后)
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; ++i) {
        sampling_logits_[i] *= inv_temp;
    }

    // 3. Clamp top_k
    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;

    // 4. 初始化 indices, partial_sort 取 top_k
    for (int i = 0; i < vocab_size; ++i) sampling_indices_[i] = i;
    int k = std::min(top_k, vocab_size);
    std::partial_sort(sampling_indices_.begin(),
                      sampling_indices_.begin() + k,
                      sampling_indices_.end(),
                      [this](int a, int b) {
                          return sampling_logits_[a] > sampling_logits_[b];
                      });

    // 5. 对 top_k 做 softmax
    float max_logit = sampling_logits_[sampling_indices_[0]];
    float sum = 0.0f;
    // 复用 sampling_logits_ 前 k 个位置存 prob (不影响, indices 已排好)
    float probs[256];  // top_k 通常 ≤ 64, 栈上分配
    float* prob_buf = (k <= 256) ? probs : new float[k];
    for (int i = 0; i < k; ++i) {
        prob_buf[i] = expf(sampling_logits_[sampling_indices_[i]] - max_logit);
        sum += prob_buf[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < k; ++i) prob_buf[i] *= inv_sum;

    // 6. Top-p (nucleus) 截断
    int cutoff = k;
    if (top_p < 1.0f && top_p > 0.0f) {
        float cumsum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cumsum += prob_buf[i];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        // 重新归一化
        sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) sum += prob_buf[i];
        inv_sum = 1.0f / sum;
        for (int i = 0; i < cutoff; ++i) prob_buf[i] *= inv_sum;
    }

    // 6.5 Min-p 过滤: 移除概率 < min_p * max_prob 的 token
    // 这有效防止高温度采样时选到极低概率的错误 token
    if (min_p > 0.0f && cutoff > 1) {
        float max_prob = prob_buf[0];  // 已按概率降序排列
        float threshold = min_p * max_prob;
        int new_cutoff = cutoff;
        for (int i = cutoff - 1; i >= 1; --i) {
            if (prob_buf[i] < threshold) new_cutoff = i;
            else break;
        }
        if (new_cutoff < cutoff && new_cutoff >= 1) {
            cutoff = new_cutoff;
            // 重新归一化
            sum = 0.0f;
            for (int i = 0; i < cutoff; ++i) sum += prob_buf[i];
            inv_sum = 1.0f / sum;
            for (int i = 0; i < cutoff; ++i) prob_buf[i] *= inv_sum;
        }
    }

    // 7. 随机采样
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(sampling_rng_);
    float cumsum = 0.0f;
    int selected = sampling_indices_[cutoff - 1];  // fallback
    for (int i = 0; i < cutoff; ++i) {
        cumsum += prob_buf[i];
        if (r < cumsum) {
            selected = sampling_indices_[i];
            break;
        }
    }

    if (prob_buf != probs) delete[] prob_buf;
    return selected;
}

// ---------------------------------------------------------------------------
// try_swap_out_victim: 当 KV blocks 不够时, 换出占用最多 block 的非当前请求
// 返回释放的 block 数
// ---------------------------------------------------------------------------
int InferenceEngine::try_swap_out_victim(
    std::vector<RequestContext*>& active_requests,
    int blocks_needed) {

    if (!kv_swapper_) return 0;

    // 选择 victim: 占用 blocks 最多且不是当前正在处理的请求 (active_requests[0])
    RequestContext* victim = nullptr;
    int max_blocks = 0;
    for (size_t i = 1; i < active_requests.size(); ++i) {
        auto* r = active_requests[i];
        if (r->is_swapped || r->is_finished) continue;
        if ((int)r->block_table.size() > max_blocks) {
            max_blocks = (int)r->block_table.size();
            victim = r;
        }
    }

    if (!victim || max_blocks == 0) {
        return 0;  // 没有可换出的请求
    }

    std::cout << "[Engine] Swapping out request " << victim->request_id
              << " (" << max_blocks << " blocks, ctx=" << victim->context_len << ")" << std::endl;

    kv_swapper_->swap_out(
        victim->request_id,
        *kv_manager_,
        victim->block_table,
        victim->context_len,
        victim->ssm_states.empty() ? nullptr : victim->ssm_states.data(),
        num_linear_layers_, ssm_size_per_layer_,
        victim->conv_states.empty() ? nullptr : victim->conv_states.data(),
        conv_size_per_layer_,
        compute_stream_);

    // 池化: 不 cudaFree SSM/Conv, 回收 slot 到 free list, 清空指针
    if (victim->ssm_slot >= 0) {
        free_ssm_slots_.push_back(victim->ssm_slot);
        std::cout << "[Engine] Swap-out: returned SSM slot " << victim->ssm_slot
                  << " (free slots: " << free_ssm_slots_.size() << ")" << std::endl;
        victim->ssm_slot = -1;
    }
    for (auto& p : victim->ssm_states)  p = nullptr;
    for (auto& p : victim->conv_states) p = nullptr;

    victim->block_table.clear();
    victim->is_swapped = true;

    return max_blocks;
}

} // namespace core
} // namespace qwen_thor