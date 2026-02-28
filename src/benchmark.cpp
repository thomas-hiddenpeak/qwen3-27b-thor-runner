// benchmark.cpp — 专用性能评估程序 (无 IPC 依赖)
//
// 用法:
//   ./benchmark [--warmup N] [--decode N] [--prompt-len N] [--nsys] [--csv]
//
// 说明:
//   1. 加载 Qwen3.5-27B 模型权重
//   2. 用固定 prompt (或指定长度) 做 prefill
//   3. 运行 decode 循环并采集每步精确计时
//   4. 支持 NVTX range 标记以配合 nsys/ncu
//   5. 输出表格 / CSV 格式结果
//
// nsys 用法:
//   nsys profile --trace=cuda,nvtx -o profile ./benchmark --decode 30 --nsys
//
// ncu 用法 (单步):
//   ncu --target-processes all --set full -o ncu_report ./benchmark --decode 3 --nsys

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nvtx3/nvToolsExt.h>

#include "engine/model.h"
#include "engine/engine.h"
#include "engine/layer.h"
#include "engine/allocator.h"
#include "engine/light_ops.h"
#include "engine/dense_gemm.h"
#include "engine/paged_attention.h"
#include "engine/cache_config.h"

using namespace qwen_thor;

// ============================================================================
// NVTX helpers
// ============================================================================
static bool g_nvtx_enabled = false;

static void nvtx_push(const char* name) {
    if (g_nvtx_enabled) nvtxRangePushA(name);
}
static void nvtx_pop() {
    if (g_nvtx_enabled) nvtxRangePop();
}

// ============================================================================
// CUDA Event Timer pair
// ============================================================================
struct EventTimer {
    cudaEvent_t start, stop;
    EventTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~EventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void record_start(cudaStream_t s) { cudaEventRecord(start, s); }
    void record_stop(cudaStream_t s)  { cudaEventRecord(stop, s); }
    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// ============================================================================
// 统计工具
// ============================================================================
struct Stats {
    std::vector<float> samples;

    void add(float v) { samples.push_back(v); }
    int   count()  const { return (int)samples.size(); }
    float sum()    const { return std::accumulate(samples.begin(), samples.end(), 0.0f); }
    float mean()   const { return count() > 0 ? sum() / count() : 0; }
    float median() const {
        if (samples.empty()) return 0;
        auto s = samples;
        std::sort(s.begin(), s.end());
        int n = (int)s.size();
        return (n % 2 == 0) ? (s[n/2 - 1] + s[n/2]) * 0.5f : s[n/2];
    }
    float min_val() const { return *std::min_element(samples.begin(), samples.end()); }
    float max_val() const { return *std::max_element(samples.begin(), samples.end()); }
    float percentile(float p) const {
        if (samples.empty()) return 0;
        auto s = samples;
        std::sort(s.begin(), s.end());
        int idx = (int)(s.size() * p);
        if (idx >= (int)s.size()) idx = (int)s.size() - 1;
        return s[idx];
    }
    float p50()  const { return percentile(0.50f); }
    float p95()  const { return percentile(0.95f); }
    float p99()  const { return percentile(0.99f); }
    float stddev() const {
        if (count() < 2) return 0;
        float m = mean();
        float acc = 0;
        for (auto v : samples) acc += (v - m) * (v - m);
        return sqrtf(acc / (count() - 1));
    }
};

// ============================================================================
// 命令行参数
// ============================================================================
struct BenchConfig {
    int warmup_steps   = 5;
    int decode_steps   = 50;
    int prompt_len     = 17;   // 默认 chat template "What is 2+2?"
    int batch_size     = 1;    // 并发请求数 (batched decode)
    double kv_cache_gb = 4.0;  // GPU KV Cache 预算 (GB)
    bool nsys_mode     = false;
    bool csv_output    = false;
    bool per_step      = false; // 是否打印每步详情
    std::string model_dir = "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup"     && i + 1 < argc) cfg.warmup_steps = std::atoi(argv[++i]);
        else if (arg == "--decode"     && i + 1 < argc) cfg.decode_steps  = std::atoi(argv[++i]);
        else if (arg == "--prompt-len" && i + 1 < argc) cfg.prompt_len    = std::atoi(argv[++i]);
        else if (arg == "--batch"      && i + 1 < argc) cfg.batch_size    = std::atoi(argv[++i]);
        else if (arg == "--model-dir"  && i + 1 < argc) cfg.model_dir     = argv[++i];
        else if (arg == "--kv-cache-gb" && i + 1 < argc) cfg.kv_cache_gb = std::atof(argv[++i]);
        else if (arg == "--nsys")      cfg.nsys_mode  = true;
        else if (arg == "--csv")       cfg.csv_output = true;
        else if (arg == "--per-step")  cfg.per_step   = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark [options]\n"
                      << "  --warmup N       Warmup decode steps (default: 5)\n"
                      << "  --decode N       Benchmark decode steps (default: 50)\n"
                      << "  --prompt-len N   Prompt token count (default: 17)\n"
                      << "  --batch N        Concurrent requests for batched decode (default: 1)\n"
                      << "  --kv-cache-gb N  GPU KV Cache budget in GB (default: 4.0)\n"
                      << "  --model-dir DIR  Model weights directory\n"
                      << "  --nsys           Enable NVTX markers for nsys/ncu\n"
                      << "  --csv            Output CSV format\n"
                      << "  --per-step       Print per-step timing details\n";
            exit(0);
        }
    }
    return cfg;
}

// ============================================================================
// 主程序
// ============================================================================
int run_benchmark(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);
    g_nvtx_enabled = cfg.nsys_mode;

    // 使用 blocking sync 而非 spin-wait, 避免 CPU 核心在等 GPU 时空转 100%
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    std::cout << "========================================\n"
              << "  Qwen3.5-27B Benchmark (SM110 Thor)\n"
              << "========================================\n"
              << "  Warmup steps : " << cfg.warmup_steps << "\n"
              << "  Decode steps : " << cfg.decode_steps  << "\n"
              << "  Prompt len   : " << cfg.prompt_len    << "\n"
              << "  Batch size   : " << cfg.batch_size    << "\n"
              << "  KV Cache GB  : " << cfg.kv_cache_gb   << "\n"
              << "  NVTX markers : " << (cfg.nsys_mode ? "ON" : "OFF") << "\n"
              << "  CSV output   : " << (cfg.csv_output ? "ON" : "OFF") << "\n"
              << "========================================\n\n";

    // ========================================================================
    // 1. 初始化模型
    // ========================================================================
    nvtx_push("model_init");
    core::Qwen35Config config;
    config.model_dir = cfg.model_dir;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[1/4] Loading model weights...\n";
    auto model = std::make_unique<core::Qwen35Model>(config);
    model->load_weights(cfg.model_dir);
    std::cout << "      Model loaded.\n";

    // L2 Cache Persistence: 设置 norm 权重常驻 L2 cache
    // 64 层 × 2 norms × 5120 × 2B = 1.25 MB (远小于 32 MB L2)
    // 每步被读 128+ 次, 命中 L2 可省去 DRAM 往返
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.persistingL2CacheMaxSize > 0) {
            size_t persist_size = std::min((size_t)4 * 1024 * 1024, (size_t)prop.persistingL2CacheMaxSize);
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_size);

            auto set_persist = [&](const void* ptr, size_t bytes) {
                if (!ptr || bytes == 0) return;
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
                attr.accessPolicyWindow.num_bytes  = bytes;
                attr.accessPolicyWindow.hitRatio   = 1.0f;
                attr.accessPolicyWindow.hitProp    = cudaAccessPropertyPersisting;
                attr.accessPolicyWindow.missProp   = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
            };

            size_t norm_bytes = config.hidden_size * sizeof(__nv_bfloat16);
            set_persist(model->get_norm_weight(), norm_bytes);
            std::cout << "      L2 Cache Persistence: enabled (" << persist_size / 1024 << " KB)\n";
        }
    }
    std::cout << "\n";

    // ========================================================================
    // 2. 分配推理资源
    // ========================================================================
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    const int batch_size = cfg.batch_size;

    // KV Cache — 使用容量规划系统计算 block 数
    int num_full_attn_layers = config.num_full_attn_layers();
    cache::ModelCacheParams mcp;
    mcp.num_full_attn_layers   = num_full_attn_layers;
    mcp.num_kv_heads           = config.num_key_value_heads;
    mcp.head_dim               = config.head_dim;
    mcp.block_size             = 16;
    mcp.num_linear_attn_layers = config.num_hidden_layers - num_full_attn_layers;
    mcp.nkh                    = config.linear_num_key_heads;
    mcp.kd                     = config.linear_key_head_dim;
    mcp.v_per_kh               = config.lin_v_per_kh();
    mcp.in_qkv                 = config.lin_qk_dim() * 2 + config.lin_v_dim();
    mcp.conv_k_minus_1         = config.linear_conv_kernel_dim - 1;

    cache::CacheConfig cache_cfg;
    cache_cfg.kv_cache_budget_gb = cfg.kv_cache_gb;
    auto capacity = cache::CapacityPlanner::plan(cache_cfg, mcp);
    cache::CapacityPlanner::print_report(capacity, cache_cfg);

    int num_kv_blocks = capacity.gpu_kv_blocks;

    // 容量检查: batch × (prompt + decode) 不能超过 GPU KV token 预算
    int total_kv_tokens_needed = cfg.batch_size * (cfg.prompt_len + cfg.decode_steps);
    int total_kv_blocks_needed = (total_kv_tokens_needed + 15) / 16;
    if (total_kv_blocks_needed > num_kv_blocks) {
        printf("\n  ⚠ WARNING: benchmark requires %d KV blocks (%d tokens)\n"
               "             but GPU budget only has %d blocks (%d tokens)\n"
               "             Reduce --batch, --prompt-len, --decode or increase --kv-cache-gb\n\n",
               total_kv_blocks_needed, total_kv_tokens_needed,
               num_kv_blocks, capacity.gpu_max_tokens);
    }

    auto kv_manager = std::make_unique<ops::KVCacheManager>(
        num_kv_blocks, 16, config.num_key_value_heads, config.head_dim,
        core::DataType::FP16, allocator, num_full_attn_layers);

    // 预分配 buffers — 大小自适应 prompt_len
    // prefill 需要 prompt_len token, batched decode 需要 batch_size token
    int max_tokens = std::max(cfg.prompt_len + 64, batch_size);
    int max_kv_blks_per_seq = ((cfg.prompt_len + cfg.decode_steps + 15) / 16) + 4;

    __nv_bfloat16* d_hidden_states = nullptr;
    int* d_pos_ids = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    int* d_argmax_result = nullptr;
    __nv_bfloat16* d_workspace = nullptr;

    cudaMalloc(&d_hidden_states, (size_t)max_tokens * config.hidden_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos_ids, (size_t)max_tokens * sizeof(int));
    cudaMalloc(&d_block_tables, (size_t)batch_size * max_kv_blks_per_seq * sizeof(int));
    cudaMalloc(&d_context_lens, (size_t)batch_size * sizeof(int));
    cudaMallocManaged(&d_argmax_result, (size_t)batch_size * sizeof(int));

    // Workspace
    const int hs = config.hidden_size;
    const int is = config.intermediate_size;
    const int qp_dim = config.q_proj_dim();
    const int kv_dim = config.kv_dim();
    const int qk = config.lin_qk_dim();
    const int lin_v = config.lin_v_dim();
    const int nkh = config.linear_num_key_heads;
    const int in_qkv = 2 * qk + lin_v;
    const int nv = config.linear_num_value_heads;

    size_t ws_full = (size_t)(4*hs + qp_dim + config.q_dim() + 2*kv_dim + 3*is);
    size_t ws_linear = (size_t)(hs + in_qkv + lin_v + nv + qk + lin_v + hs + hs + 3*is + hs + nkh*2);
    size_t ws_per_tok = std::max(ws_full, ws_linear);
    // workspace 需容纳 max_tokens 个 token 的激活 + 用于 LM head 的 logits [batch, vocab]
    size_t ws_total = ws_per_tok * max_tokens + (size_t)batch_size * config.vocab_size + (size_t)batch_size * hs;
    cudaMalloc(&d_workspace, ws_total * sizeof(__nv_bfloat16));

    // Per-request SSM/Conv 状态
    int num_lin = 0;
    for (int i = 0; i < config.num_hidden_layers; ++i)
        if (!config.is_full_attention(i)) ++num_lin;

    size_t ssm_sz = (size_t)nkh * config.linear_key_head_dim * config.lin_v_per_kh() * sizeof(float);
    size_t conv_sz = (size_t)in_qkv * (config.linear_conv_kernel_dim - 1) * sizeof(__nv_bfloat16);

    // Per-request context
    struct ReqCtx {
        std::vector<int> block_table;
        int context_len;
        std::vector<float*> ssm_states;
        std::vector<__nv_bfloat16*> conv_states;
        int last_token;
    };
    std::vector<ReqCtx> requests(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        requests[b].ssm_states.resize(num_lin);
        requests[b].conv_states.resize(num_lin);
        for (int li = 0; li < num_lin; ++li) {
            cudaMalloc(&requests[b].ssm_states[li], ssm_sz);
            cudaMemset(requests[b].ssm_states[li], 0, ssm_sz);
            cudaMalloc(&requests[b].conv_states[li], conv_sz);
            cudaMemset(requests[b].conv_states[li], 0, conv_sz);
        }
    }

    // Managed pointer arrays for SSM/Conv states (device-accessible)
    // Layout: [lin_idx * batch_size + batch_idx]
    float** d_ssm_ptrs = nullptr;
    __nv_bfloat16** d_conv_ptrs = nullptr;
    cudaMallocManaged(&d_ssm_ptrs, (size_t)num_lin * batch_size * sizeof(float*));
    cudaMallocManaged(&d_conv_ptrs, (size_t)num_lin * batch_size * sizeof(__nv_bfloat16*));
    for (int li = 0; li < num_lin; ++li) {
        for (int bi = 0; bi < batch_size; ++bi) {
            d_ssm_ptrs[li * batch_size + bi] = requests[bi].ssm_states[li];
            d_conv_ptrs[li * batch_size + bi] = requests[bi].conv_states[li];
        }
    }

    nvtx_pop(); // model_init

    // ========================================================================
    // 2.5. Prefill Warmup — 1 次预热 prefill (不计时), 确保 TLB/L2 有效
    // ========================================================================
    {
        std::cout << "[1.5/4] Prefill warmup..." << std::flush;
        int default_tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                                248046, 198, 248045, 74455, 198, 248068, 198};
        int prompt_len = cfg.prompt_len;
        std::vector<int> warm_tokens(prompt_len);
        for (int i = 0; i < prompt_len; ++i)
            warm_tokens[i] = (i < 17) ? default_tokens[i] : 1;

        // 使用 request 0 的状态做预热
        int num_blocks_needed = (prompt_len + 15) / 16;
        auto btable = kv_manager->allocate_blocks(num_blocks_needed);
        requests[0].block_table = btable;
        requests[0].context_len = prompt_len;

        cudaMemcpyAsync(d_pos_ids, warm_tokens.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        ops::invoke_embedding_lookup(d_hidden_states, d_pos_ids, model->get_embed_tokens(),
                                     prompt_len, hs, stream);
        std::vector<int> pos_ids(prompt_len);
        for (int i = 0; i < prompt_len; ++i) pos_ids[i] = i;
        cudaMemcpyAsync(d_pos_ids, pos_ids.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_block_tables, btable.data(),
                        btable.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        int cl = prompt_len;
        cudaMemcpyAsync(d_context_lens, &cl, sizeof(int), cudaMemcpyHostToDevice, stream);

        std::vector<float*> single_ssm(num_lin);
        std::vector<__nv_bfloat16*> single_conv(num_lin);
        for (int li = 0; li < num_lin; ++li) {
            single_ssm[li] = requests[0].ssm_states[li];
            single_conv[li] = requests[0].conv_states[li];
        }

        model->forward_prefill(d_hidden_states, d_pos_ids, *kv_manager,
                       d_block_tables, d_context_lens,
                       (int)btable.size(), cl, prompt_len,
                       single_ssm.data(), single_conv.data(),
                       d_workspace, stream);
        cudaStreamSynchronize(stream);

        // 释放 warmup blocks 并重置状态
        kv_manager->free_blocks(btable);
        for (int li = 0; li < num_lin; ++li) {
            cudaMemset(requests[0].ssm_states[li], 0, ssm_sz);
            cudaMemset(requests[0].conv_states[li], 0, conv_sz);
        }

        // Second warmup — 确保连续两次都能稳定
        btable = kv_manager->allocate_blocks(num_blocks_needed);
        requests[0].block_table = btable;
        cudaMemcpyAsync(d_pos_ids, warm_tokens.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        ops::invoke_embedding_lookup(d_hidden_states, d_pos_ids, model->get_embed_tokens(),
                                     prompt_len, hs, stream);
        cudaMemcpyAsync(d_pos_ids, pos_ids.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_block_tables, btable.data(),
                        btable.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_context_lens, &cl, sizeof(int), cudaMemcpyHostToDevice, stream);
        for (int li = 0; li < num_lin; ++li) {
            single_ssm[li] = requests[0].ssm_states[li];
            single_conv[li] = requests[0].conv_states[li];
        }
        model->forward_prefill(d_hidden_states, d_pos_ids, *kv_manager,
                       d_block_tables, d_context_lens,
                       (int)btable.size(), cl, prompt_len,
                       single_ssm.data(), single_conv.data(),
                       d_workspace, stream);
        cudaStreamSynchronize(stream);
        kv_manager->free_blocks(btable);
        for (int li = 0; li < num_lin; ++li) {
            cudaMemset(requests[0].ssm_states[li], 0, ssm_sz);
            cudaMemset(requests[0].conv_states[li], 0, conv_sz);
        }

        std::cout << " done.\n";
    }

    // ========================================================================
    // 3. Prefill — 每个请求独立 prefill (batch_size=1), 精确计时
    // ========================================================================
    std::cout << "[2/4] Running prefill (" << cfg.prompt_len << " tokens × "
              << batch_size << " requests)...\n";
    nvtx_push("prefill");

    // Chat template for "What is 2+2?"
    int default_tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                            248046, 198, 248045, 74455, 198, 248068, 198};
    int prompt_len = cfg.prompt_len;
    std::vector<int> prompt_tokens(prompt_len);
    for (int i = 0; i < prompt_len; ++i)
        prompt_tokens[i] = (i < 17) ? default_tokens[i] : 1;

    // Per-request TTFT 计时 (从 embedding 开始到 first token argmax 完成)
    Stats prefill_ttft_stats;       // TTFT (ms) 每个请求
    Stats prefill_forward_stats;    // forward 阶段 (ms)
    Stats prefill_embed_stats;      // embedding (ms)
    Stats prefill_lmhead_stats;     // final_norm + lm_head + argmax (ms)
    EventTimer pf_embed, pf_forward, pf_lmhead, pf_total;

    for (int b = 0; b < batch_size; ++b) {
        // 分配 KV blocks
        int num_blocks_needed = (prompt_len + 15) / 16;
        auto btable = kv_manager->allocate_blocks(num_blocks_needed);
        requests[b].block_table = btable;
        requests[b].context_len = prompt_len;

        // --- TTFT timer start ---
        pf_total.record_start(stream);

        // Embedding
        pf_embed.record_start(stream);
        cudaMemcpyAsync(d_pos_ids, prompt_tokens.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        ops::invoke_embedding_lookup(d_hidden_states, d_pos_ids, model->get_embed_tokens(),
                                     prompt_len, hs, stream);

        // Position IDs
        std::vector<int> pos_ids(prompt_len);
        for (int i = 0; i < prompt_len; ++i) pos_ids[i] = i;
        cudaMemcpyAsync(d_pos_ids, pos_ids.data(), prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        pf_embed.record_stop(stream);

        // Block tables & context lens (单请求)
        cudaMemcpyAsync(d_block_tables, btable.data(),
                        btable.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        int cl = requests[b].context_len;
        cudaMemcpyAsync(d_context_lens, &cl, sizeof(int), cudaMemcpyHostToDevice, stream);

        // 构建单请求状态指针 (batch_size=1 for prefill)
        std::vector<float*> single_ssm(num_lin);
        std::vector<__nv_bfloat16*> single_conv(num_lin);
        for (int li = 0; li < num_lin; ++li) {
            single_ssm[li] = requests[b].ssm_states[li];
            single_conv[li] = requests[b].conv_states[li];
        }

        // Forward (64 layers) — Prefill 专用路径 (per-layer sync)
        pf_forward.record_start(stream);
        model->forward_prefill(d_hidden_states, d_pos_ids, *kv_manager,
                       d_block_tables, d_context_lens,
                       (int)btable.size(), cl,
                       prompt_len,
                       single_ssm.data(),
                       single_conv.data(),
                       d_workspace, stream);
        pf_forward.record_stop(stream);

        // Final norm + lm_head + argmax (取最后一个 token)
        pf_lmhead.record_start(stream);
        __nv_bfloat16* norm_out_pf = d_workspace;
        __nv_bfloat16* logits_pf = norm_out_pf + hs;
        ops::invoke_rmsnorm(norm_out_pf,
                            d_hidden_states + (prompt_len - 1) * hs,
                            model->get_norm_weight(), config.rms_norm_eps, 1, hs, stream);
        ops::invoke_dense_gemv(norm_out_pf, model->get_lm_head(), logits_pf,
                               config.vocab_size, hs, stream);
        ops::invoke_argmax(logits_pf, d_argmax_result, config.vocab_size, stream);
        pf_lmhead.record_stop(stream);

        // --- TTFT timer stop ---
        pf_total.record_stop(stream);

        // Sync to read result
        cudaStreamSynchronize(stream);

        float ttft_ms  = pf_total.elapsed_ms();
        float emb_ms   = pf_embed.elapsed_ms();
        float fwd_ms   = pf_forward.elapsed_ms();
        float lmh_ms   = pf_lmhead.elapsed_ms();

        prefill_ttft_stats.add(ttft_ms);
        prefill_embed_stats.add(emb_ms);
        prefill_forward_stats.add(fwd_ms);
        prefill_lmhead_stats.add(lmh_ms);

        requests[b].last_token = *d_argmax_result;
        requests[b].context_len++;

        if (batch_size <= 8 || b == 0 || b == batch_size - 1) {
            printf("      Request %d: TTFT=%.1fms (embed=%.2f fwd=%.1f lmhead=%.2f) tok=%d\n",
                   b, ttft_ms, emb_ms, fwd_ms, lmh_ms, requests[b].last_token);
        }
    }

    float total_prefill_ms = prefill_ttft_stats.sum();
    float prefill_tok_per_sec = (float)prompt_len * 1000.0f / prefill_ttft_stats.median();
    std::cout << "      Prefill summary: TTFT median=" << std::fixed << std::setprecision(1)
              << prefill_ttft_stats.median() << "ms, "
              << "prefill tok/s=" << std::setprecision(0) << prefill_tok_per_sec
              << ", total=" << std::setprecision(1) << total_prefill_ms << "ms\n\n";
    nvtx_pop(); // prefill

    // 重建 d_ssm_ptrs / d_conv_ptrs (prefill 后状态已更新, 指针不变)

    // ========================================================================
    // 4. Decode benchmark — batched decode
    // ========================================================================
    int total_steps = cfg.warmup_steps + cfg.decode_steps;

    // Per-step timers
    Stats total_stats, forward_stats, embed_stats, norm_stats, lmhead_stats, sample_stats;
    EventTimer t_total, t_embed, t_forward, t_norm, t_lmhead, t_sample;

    // 用于组装 batch 输入的临时 host 缓冲区
    std::vector<int> h_token_ids(batch_size);
    std::vector<int> h_pos_ids(batch_size);
    std::vector<int> h_ctx_lens(batch_size);
    std::vector<int> h_block_tables_flat;

    // CUDA Graph: 固定 block_tables stride = max_kv_blks_per_seq
    // 确保 graph 捕获的 kernel 参数在步间不变
    const int fixed_max_blks = max_kv_blks_per_seq;
    h_block_tables_flat.resize(batch_size * fixed_max_blks, 0);

    // CUDA Graph 相关
    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_graph_exec = nullptr;
    bool graph_captured = false;

    // workspace 指针 (graph 内外共用, 地址固定)
    __nv_bfloat16* norm_out_graph = d_workspace + ws_per_tok * max_tokens;
    __nv_bfloat16* logits_graph = norm_out_graph + batch_size * hs;

    std::cout << "[3/4] Running decode (" << cfg.warmup_steps << " warmup + "
              << cfg.decode_steps << " measured, batch_size="
              << batch_size << ")...\n";

    for (int step = 0; step < total_steps; ++step) {
        bool is_warmup = (step < cfg.warmup_steps);
        const char* step_label = is_warmup ? "warmup" : "decode";

        char nvtx_name[64];
        snprintf(nvtx_name, sizeof(nvtx_name), "%s_step_%d", step_label, step);
        nvtx_push(nvtx_name);

        // -- 组装 batch 输入 --
        int max_ctx = 0;
        for (int b = 0; b < batch_size; ++b) {
            h_token_ids[b] = requests[b].last_token;
            h_pos_ids[b] = requests[b].context_len - 1;
            h_ctx_lens[b] = requests[b].context_len;
            max_ctx = std::max(max_ctx, requests[b].context_len);
        }

        // 组装 2D block_tables [batch_size, fixed_max_blks] (固定 stride)
        std::fill(h_block_tables_flat.begin(), h_block_tables_flat.end(), 0);
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < (int)requests[b].block_table.size(); ++j) {
                h_block_tables_flat[b * fixed_max_blks + j] = requests[b].block_table[j];
            }
        }

        // -- Total step timer --
        t_total.record_start(stream);

        // -- Embedding (batch_size tokens) — 在 graph 外执行 --
        nvtx_push("embedding");
        t_embed.record_start(stream);
        cudaMemcpyAsync(d_pos_ids, h_token_ids.data(), batch_size * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        ops::invoke_embedding_lookup(d_hidden_states, d_pos_ids,
                                     model->get_embed_tokens(), batch_size, hs, stream);
        cudaMemcpyAsync(d_pos_ids, h_pos_ids.data(), batch_size * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        t_embed.record_stop(stream);
        nvtx_pop();

        // -- Upload block tables and context lens --
        cudaMemcpyAsync(d_block_tables, h_block_tables_flat.data(),
                        batch_size * fixed_max_blks * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_context_lens, h_ctx_lens.data(), batch_size * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        if (graph_captured) {
            // ======== CUDA Graph 热路径: 单次 graph launch 替代 800+ kernel launches ========
            t_forward.record_start(stream);
            cudaGraphLaunch(decode_graph_exec, stream);
            t_forward.record_stop(stream);

            // 结果 timer (graph 内已包含 norm + lmhead + sample)
            t_norm.record_start(stream); t_norm.record_stop(stream);
            t_lmhead.record_start(stream); t_lmhead.record_stop(stream);
            t_sample.record_start(stream); t_sample.record_stop(stream);
        } else {
            // ======== 非 graph 路径: warmup 或首次执行 ========

            // -- Forward (64 layers, batched) — Decode 专用路径 (无 per-layer sync, CUDA Graph 兼容) --
            nvtx_push("forward");
            t_forward.record_start(stream);
            model->forward_decode(d_hidden_states, d_pos_ids, *kv_manager,
                           d_block_tables, d_context_lens,
                           fixed_max_blks, max_ctx,
                           batch_size,
                           d_ssm_ptrs, d_conv_ptrs,
                           d_workspace, stream);
            t_forward.record_stop(stream);
            nvtx_pop();

            // -- Final norm (batch_size tokens) --
            nvtx_push("final_norm");
            t_norm.record_start(stream);
            ops::invoke_rmsnorm(norm_out_graph, d_hidden_states, model->get_norm_weight(),
                                config.rms_norm_eps, batch_size, hs, stream);
            t_norm.record_stop(stream);
            nvtx_pop();

            // -- LM Head (batch_size tokens → logits) --
            nvtx_push("lm_head");
            t_lmhead.record_start(stream);
            if (batch_size == 1) {
                ops::invoke_dense_gemv(norm_out_graph, model->get_lm_head(), logits_graph,
                                       config.vocab_size, hs, stream);
            } else {
                ops::invoke_dense_gemm(norm_out_graph, model->get_lm_head(), logits_graph,
                                       batch_size, config.vocab_size, hs, stream);
            }
            t_lmhead.record_stop(stream);
            nvtx_pop();

            // -- Argmax (batched single launch) --
            nvtx_push("sample");
            t_sample.record_start(stream);
            ops::invoke_batched_argmax(logits_graph, d_argmax_result, config.vocab_size,
                                       batch_size, stream);
            t_sample.record_stop(stream);
            nvtx_pop();

            // -- 在最后一个 warmup step 之后, 捕获 CUDA Graph --
            if (is_warmup && step == cfg.warmup_steps - 1) {
                // 先同步确保上面的 warmup step 完成
                cudaStreamSynchronize(stream);

                // 在独立的 capture stream 上录制 graph
                cudaStream_t capture_stream;
                cudaStreamCreate(&capture_stream);

                cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

                // 录制: forward + norm + lmhead + argmax (Decode 路径)
                model->forward_decode(d_hidden_states, d_pos_ids, *kv_manager,
                               d_block_tables, d_context_lens,
                               fixed_max_blks, max_ctx,
                               batch_size,
                               d_ssm_ptrs, d_conv_ptrs,
                               d_workspace, capture_stream);

                ops::invoke_rmsnorm(norm_out_graph, d_hidden_states, model->get_norm_weight(),
                                    config.rms_norm_eps, batch_size, hs, capture_stream);

                if (batch_size == 1) {
                    ops::invoke_dense_gemv(norm_out_graph, model->get_lm_head(), logits_graph,
                                           config.vocab_size, hs, capture_stream);
                } else {
                    ops::invoke_dense_gemm(norm_out_graph, model->get_lm_head(), logits_graph,
                                           batch_size, config.vocab_size, hs, capture_stream);
                }

                ops::invoke_batched_argmax(logits_graph, d_argmax_result, config.vocab_size,
                                           batch_size, capture_stream);

                cudaStreamEndCapture(capture_stream, &decode_graph);
                cudaGraphInstantiate(&decode_graph_exec, decode_graph, 0);
                cudaStreamDestroy(capture_stream);
                graph_captured = true;

                std::cout << "      CUDA Graph captured (replaces ~"
                          << (batch_size == 1 ? 803 : 947)
                          << " kernel launches with 1 graph launch)\n";
            }
        }

        // -- Sync & collect --
        t_total.record_stop(stream);
        float ms_total   = t_total.elapsed_ms();
        float ms_embed   = t_embed.elapsed_ms();
        float ms_forward = t_forward.elapsed_ms();
        float ms_norm    = t_norm.elapsed_ms();
        float ms_lmhead  = t_lmhead.elapsed_ms();
        float ms_sample  = t_sample.elapsed_ms();

        // Update per-request state
        for (int b = 0; b < batch_size; ++b) {
            requests[b].last_token = d_argmax_result[b];
            requests[b].context_len++;
            // 检查是否需要分配新 KV blocks
            if (requests[b].context_len > (int)requests[b].block_table.size() * 16) {
                auto new_blks = kv_manager->allocate_blocks(1);
                if (!new_blks.empty()) requests[b].block_table.push_back(new_blks[0]);
            }
        }

        if (!is_warmup) {
            total_stats.add(ms_total);
            forward_stats.add(ms_forward);
            embed_stats.add(ms_embed);
            norm_stats.add(ms_norm);
            lmhead_stats.add(ms_lmhead);
            sample_stats.add(ms_sample);
        }

        if (cfg.per_step || is_warmup) {
            printf("  [%s %3d] total=%7.2f  fwd=%7.2f  embed=%5.2f  norm=%5.2f  "
                   "lm=%7.2f  sample=%5.2f  tok=%d\n",
                   step_label, step, ms_total, ms_forward, ms_embed, ms_norm,
                   ms_lmhead, ms_sample, requests[0].last_token);
        }

        nvtx_pop(); // step
    }

    // ========================================================================
    // 5. 结果报告 — 标准 LLM 推理指标
    // ========================================================================
    std::cout << "\n[4/4] Results\n";
    std::cout << "========================================\n";

    // 理论带宽计算 — 每步总权重读取量 (BF16)
    size_t la_params = (size_t)in_qkv * hs + (lin_v + 2*nv) * hs + hs * lin_v
                       + (size_t)is * hs + is * hs + hs * is;
    size_t fa_params = (size_t)(qp_dim + 2*kv_dim) * hs + hs * config.q_dim()
                       + (size_t)is * hs + is * hs + hs * is;
    size_t total_weight_bytes = (48 * la_params + 16 * fa_params
                                 + (size_t)config.vocab_size * hs) * 2;

    // --- Decode 指标 ---
    float itl_median = total_stats.median();   // ITL = Inter-Token Latency (ms)
    float itl_p95    = total_stats.p95();
    float itl_p99    = total_stats.p99();
    float decode_tps_median = (float)batch_size * 1000.0f / itl_median;    // Decode tok/s
    float decode_tps_mean   = (float)batch_size * 1000.0f / total_stats.mean();
    float bw_median = (float)total_weight_bytes / (itl_median / 1000.0f) / 1e9;
    float bw_mean   = (float)total_weight_bytes / (total_stats.mean() / 1000.0f) / 1e9;

    // --- Prefill 指标 ---
    float ttft_median = prefill_ttft_stats.median();  // TTFT (ms)
    float ttft_p95    = prefill_ttft_stats.p95();
    float ttft_p99    = prefill_ttft_stats.p99();
    float prefill_tps = (float)prompt_len * 1000.0f / ttft_median;  // Prefill tok/s

    if (cfg.csv_output) {
        printf("\n--- CSV ---\n");
        printf("batch_size,%d\n", batch_size);
        printf("prompt_len,%d\n", prompt_len);
        printf("decode_steps,%d\n", cfg.decode_steps);
        printf("weight_MB,%.1f\n", (float)total_weight_bytes / 1e6);
        // Prefill
        printf("ttft_median_ms,%.2f\n", ttft_median);
        printf("ttft_p95_ms,%.2f\n", ttft_p95);
        printf("ttft_p99_ms,%.2f\n", ttft_p99);
        printf("prefill_tok_per_sec,%.1f\n", prefill_tps);
        printf("prefill_embed_ms,%.2f\n", prefill_embed_stats.median());
        printf("prefill_forward_ms,%.2f\n", prefill_forward_stats.median());
        printf("prefill_lmhead_ms,%.2f\n", prefill_lmhead_stats.median());
        // Decode
        printf("itl_median_ms,%.2f\n", itl_median);
        printf("itl_p95_ms,%.2f\n", itl_p95);
        printf("itl_p99_ms,%.2f\n", itl_p99);
        printf("decode_tok_per_sec_median,%.2f\n", decode_tps_median);
        printf("decode_tok_per_sec_mean,%.2f\n", decode_tps_mean);
        printf("bw_median_GBs,%.1f\n", bw_median);
        printf("bw_mean_GBs,%.1f\n", bw_mean);
        // Decode phase breakdown
        printf("metric,mean,median,min,max,p95,stddev\n");
        printf("total_ms,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
               total_stats.mean(), total_stats.median(), total_stats.min_val(),
               total_stats.max_val(), total_stats.p95(), total_stats.stddev());
        printf("forward_ms,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
               forward_stats.mean(), forward_stats.median(), forward_stats.min_val(),
               forward_stats.max_val(), forward_stats.p95(), forward_stats.stddev());
    }

    // ---- 标准 LLM 推理指标概览 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║        Standard LLM Inference Metrics (batch=%d, prompt=%d)  ║\n", batch_size, prompt_len);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                ║\n");
    printf("║  ▸ TTFT (Time To First Token)                                  ║\n");
    printf("║      Median:  %8.1f ms                                      ║\n", ttft_median);
    printf("║      P95:     %8.1f ms                                      ║\n", ttft_p95);
    printf("║      P99:     %8.1f ms                                      ║\n", ttft_p99);
    printf("║                                                                ║\n");
    printf("║  ▸ Prefill Throughput                                          ║\n");
    printf("║      Prefill tok/s:   %7.0f  (%d tokens / %.1f ms)            ║\n",
           prefill_tps, prompt_len, ttft_median);
    printf("║      Breakdown:  embed=%.1fms  fwd=%.1fms  lmhead=%.1fms       ║\n",
           prefill_embed_stats.median(), prefill_forward_stats.median(),
           prefill_lmhead_stats.median());
    printf("║                                                                ║\n");
    printf("║  ▸ ITL (Inter-Token Latency) — Decode                         ║\n");
    printf("║      Median:  %8.2f ms                                      ║\n", itl_median);
    printf("║      P95:     %8.2f ms                                      ║\n", itl_p95);
    printf("║      P99:     %8.2f ms                                      ║\n", itl_p99);
    printf("║                                                                ║\n");
    printf("║  ▸ Decode Throughput                                           ║\n");
    printf("║      tok/s (median): %8.2f  (%d tok/step)                    ║\n",
           decode_tps_median, batch_size);
    printf("║      tok/s (mean):   %8.2f                                   ║\n", decode_tps_mean);
    printf("║                                                                ║\n");
    printf("║  ▸ Memory Bandwidth                                           ║\n");
    printf("║      Weight BW (median): %6.1f GB/s  (peak=273 GB/s)         ║\n", bw_median);
    printf("║      Weight BW (mean):   %6.1f GB/s                          ║\n", bw_mean);
    printf("║      Weight size:        %6.1f MB (BF16)                     ║\n",
           (float)total_weight_bytes / 1e6);
    printf("║                                                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Decode Phase Breakdown                                        ║\n");
    printf("║  ┌──────────┬────────┬────────┬────────┬───────┬──────┐        ║\n");
    printf("║  │ Phase    │Mean ms │Med  ms │Min  ms │Max ms │ P95  │        ║\n");
    printf("║  ├──────────┼────────┼────────┼────────┼───────┼──────┤        ║\n");
    printf("║  │ Total    │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           total_stats.mean(), total_stats.median(), total_stats.min_val(),
           total_stats.max_val(), total_stats.p95());
    printf("║  │ Forward  │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           forward_stats.mean(), forward_stats.median(), forward_stats.min_val(),
           forward_stats.max_val(), forward_stats.p95());
    printf("║  │ Embed    │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           embed_stats.mean(), embed_stats.median(), embed_stats.min_val(),
           embed_stats.max_val(), embed_stats.p95());
    printf("║  │ Norm     │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           norm_stats.mean(), norm_stats.median(), norm_stats.min_val(),
           norm_stats.max_val(), norm_stats.p95());
    printf("║  │ LM Head  │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           lmhead_stats.mean(), lmhead_stats.median(), lmhead_stats.min_val(),
           lmhead_stats.max_val(), lmhead_stats.p95());
    printf("║  │ Sample   │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
           sample_stats.mean(), sample_stats.median(), sample_stats.min_val(),
           sample_stats.max_val(), sample_stats.p95());
    printf("║  └──────────┴────────┴────────┴────────┴───────┴──────┘        ║\n");
    printf("║  Config: %d warmup + %d decode steps, CUDA Graph: %s           ║\n",
           cfg.warmup_steps, cfg.decode_steps, graph_captured ? "ON" : "OFF");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // Cleanup
    // ========================================================================
    if (decode_graph_exec) cudaGraphExecDestroy(decode_graph_exec);
    if (decode_graph) cudaGraphDestroy(decode_graph);
    for (auto& req : requests) {
        for (auto* p : req.ssm_states)  cudaFree(p);
        for (auto* p : req.conv_states) cudaFree(p);
    }
    cudaFree(d_ssm_ptrs);
    cudaFree(d_conv_ptrs);
    cudaFree(d_hidden_states);
    cudaFree(d_pos_ids);
    cudaFree(d_workspace);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
    cudaFree(d_argmax_result);
    cudaStreamDestroy(stream);

    return 0;
}
