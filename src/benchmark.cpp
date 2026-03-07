// benchmark.cpp — 专用性能评估程序 (无 IPC 依赖)
//
// 用法:
//   ./qwen3-27b-thor bench [--warmup N] [--decode N] [--prompt-len N] [--nsys] [--csv]
//
// 增强:
//   --batch 1,2,4,8            逗号分隔, 自动扫描所有 batch size
//   --prompt-len 32,128,256    逗号分隔, 扫描多个 prompt 长度
//   --iterations N             多轮迭代, 每轮独立测量 (跨次方差 + 95% CI)
//   --prefill-repeat N         Prefill 每个配置重复 N 次 (默认 3)
//   --json results.json        结构化 JSON 输出到文件
//   --no-graph                 禁用 CUDA Graph (用于精确每阶段计时)
//   --per-step                 打印每步详情
//   --nsys                     NVTX 标记
//   --csv                      CSV 输出
//
// nsys 用法:
//   nsys profile --trace=cuda,nvtx -o profile ./qwen3-27b-thor bench --decode 30 --nsys
//
// ncu 用法 (单步):
//   ncu --target-processes all --set full -o ncu_report ./qwen3-27b-thor bench --decode 3 --nsys

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

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
// 统计工具 — 增加 CI, CV%, trimmed mean
// ============================================================================
struct Stats {
    std::vector<float> samples;

    void add(float v) { samples.push_back(v); }
    void clear()      { samples.clear(); }
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

    // Coefficient of Variation (%) — 越小越稳定
    float cv_pct() const {
        float m = mean();
        return (m > 0 && count() >= 2) ? (stddev() / m * 100.0f) : 0;
    }

    // 95% Confidence Interval half-width (t-distribution approx for N>30: ~1.96)
    float ci95() const {
        if (count() < 2) return 0;
        // For small N, use t-value approximations
        float t_val = 1.96f; // N>=30
        if (count() < 30) {
            // Simple lookup for common small N
            static const float t_table[] = {
                0, 12.71f, 4.30f, 3.18f, 2.78f, 2.57f,  // N=1..5
                2.45f, 2.36f, 2.31f, 2.26f, 2.23f,       // N=6..10
                2.20f, 2.18f, 2.16f, 2.14f, 2.13f,       // N=11..15
                2.12f, 2.11f, 2.10f, 2.09f, 2.09f,       // N=16..20
                2.08f, 2.07f, 2.07f, 2.06f, 2.06f,       // N=21..25
                2.06f, 2.05f, 2.05f, 2.05f                // N=26..29
            };
            int idx = std::min(count(), 29);
            t_val = t_table[idx];
        }
        return t_val * stddev() / sqrtf((float)count());
    }

    // Trimmed mean (去掉最高/最低各 trim_pct%)
    float trimmed_mean(float trim_pct = 0.10f) const {
        if (count() < 4) return mean();
        auto s = samples;
        std::sort(s.begin(), s.end());
        int trim = std::max(1, (int)(s.size() * trim_pct));
        float acc = 0;
        int cnt = 0;
        for (int i = trim; i < (int)s.size() - trim; ++i) {
            acc += s[i];
            cnt++;
        }
        return cnt > 0 ? acc / cnt : mean();
    }
};

// ============================================================================
// 辅助: 解析逗号分隔的整数列表
// ============================================================================
static std::vector<int> parse_int_list(const char* str) {
    std::vector<int> result;
    std::istringstream iss(str);
    std::string token;
    while (std::getline(iss, token, ',')) {
        int val = std::atoi(token.c_str());
        if (val > 0) result.push_back(val);
    }
    return result;
}

// ============================================================================
// 命令行参数
// ============================================================================
struct BenchConfig {
    int warmup_steps    = 5;
    int decode_steps    = 50;
    std::vector<int> prompt_lens = {17};
    std::vector<int> batch_sizes = {1};
    int iterations      = 1;       // 跨次迭代 (每轮独立测量)
    int prefill_repeat  = 3;       // Prefill 重复次数
    double kv_cache_gb  = 4.0;
    bool nsys_mode      = false;
    bool csv_output     = false;
    bool per_step       = false;
    bool no_graph       = false;
    std::string model_dir = "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";
    std::string json_output;        // JSON 输出文件路径 (空则不输出)
};

BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    if (const char* env_model_dir = std::getenv("QWEN_MODEL_DIR"); env_model_dir && env_model_dir[0] != '\0') {
        cfg.model_dir = env_model_dir;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--warmup"        && i + 1 < argc) cfg.warmup_steps = std::atoi(argv[++i]);
        else if (arg == "--decode"        && i + 1 < argc) cfg.decode_steps  = std::atoi(argv[++i]);
        else if (arg == "--prompt-len"    && i + 1 < argc) cfg.prompt_lens   = parse_int_list(argv[++i]);
        else if (arg == "--batch"         && i + 1 < argc) cfg.batch_sizes   = parse_int_list(argv[++i]);
        else if (arg == "--iterations"    && i + 1 < argc) cfg.iterations    = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--prefill-repeat" && i + 1 < argc) cfg.prefill_repeat = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--model-dir"     && i + 1 < argc) cfg.model_dir     = argv[++i];
        else if (arg == "--kv-cache-gb"   && i + 1 < argc) cfg.kv_cache_gb   = std::atof(argv[++i]);
        else if (arg == "--json"          && i + 1 < argc) cfg.json_output   = argv[++i];
        else if (arg == "--nsys")      cfg.nsys_mode  = true;
        else if (arg == "--csv")       cfg.csv_output = true;
        else if (arg == "--per-step")  cfg.per_step   = true;
        else if (arg == "--no-graph")  cfg.no_graph   = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: qwen3-27b-thor bench [options]\n"
                      << "  --warmup N            Warmup decode steps (default: 5)\n"
                      << "  --decode N            Benchmark decode steps (default: 50)\n"
                      << "  --prompt-len N[,N..]  Prompt token count(s), comma-separated (default: 17)\n"
                      << "  --batch N[,N..]       Batch size(s), comma-separated (default: 1)\n"
                      << "  --iterations N        Independent iterations per config (default: 1)\n"
                      << "  --prefill-repeat N    Prefill repeats per iteration (default: 3)\n"
                      << "  --kv-cache-gb N       GPU KV Cache budget in GB (default: 4.0)\n"
                      << "  --model-dir DIR       Model weights directory\n"
                      << "  --json FILE           Output structured JSON results to file\n"
                      << "  --nsys                Enable NVTX markers for nsys/ncu\n"
                      << "  --csv                 Output CSV format\n"
                      << "  --per-step            Print per-step timing details\n"
                      << "  --no-graph            Disable CUDA Graph for accurate per-phase timing\n"
                      << "\nExamples:\n"
                      << "  bench --decode 50 --batch 1,2,4 --prompt-len 64,256 --iterations 3 --json results.json\n"
                      << "  bench --decode 30 --no-graph  (single config, backward-compatible)\n";
            exit(0);
        }
    }
    return cfg;
}

// ============================================================================
// 单次 benchmark 结果
// ============================================================================
struct BenchResult {
    int batch_size;
    int prompt_len;
    int iteration;
    int decode_steps;
    int warmup_steps;
    bool cuda_graph;

    // Prefill
    Stats prefill_ttft;
    Stats prefill_forward;
    Stats prefill_embed;
    Stats prefill_lmhead;

    // Decode
    Stats decode_total;
    Stats decode_forward;
    Stats decode_embed;
    Stats decode_norm;
    Stats decode_lmhead;
    Stats decode_sample;

    // Derived (computed after measurement)
    float weight_bytes = 0;
};

// ============================================================================
// 辅助: JSON 字符串转义
// ============================================================================
static std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '\n') result += "\\n";
        else result += c;
    }
    return result;
}

// ============================================================================
// JSON 输出: Stats → JSON object
// ============================================================================
static void json_write_stats(std::ostream& os, const Stats& s, const std::string& indent) {
    if (s.count() == 0) {
        os << indent << "\"count\": 0";
        return;
    }
    os << indent << "\"count\": " << s.count() << ",\n"
       << indent << "\"mean\": " << std::fixed << std::setprecision(3) << s.mean() << ",\n"
       << indent << "\"median\": " << s.median() << ",\n"
       << indent << "\"min\": " << s.min_val() << ",\n"
       << indent << "\"max\": " << s.max_val() << ",\n"
       << indent << "\"p95\": " << s.p95() << ",\n"
       << indent << "\"p99\": " << s.p99() << ",\n"
       << indent << "\"stddev\": " << s.stddev() << ",\n"
       << indent << "\"cv_pct\": " << s.cv_pct() << ",\n"
       << indent << "\"ci95\": " << s.ci95() << ",\n"
       << indent << "\"trimmed_mean_10pct\": " << s.trimmed_mean(0.10f);
}

// ============================================================================
// JSON 输出: 写入完整结果
// ============================================================================
static void write_json(const std::string& path, const std::vector<BenchResult>& results,
                        const BenchConfig& cfg, const core::Qwen35Config& config) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        fprintf(stderr, "[Error] Cannot open JSON output file: %s\n", path.c_str());
        return;
    }

    ofs << "{\n";
    ofs << "  \"benchmark\": \"qwen3.5-thor\",\n";
    ofs << "  \"model_dir\": \"" << json_escape(cfg.model_dir) << "\",\n";
    ofs << "  \"hidden_size\": " << config.hidden_size << ",\n";
    ofs << "  \"num_layers\": " << config.num_hidden_layers << ",\n";
    ofs << "  \"vocab_size\": " << config.vocab_size << ",\n";
    ofs << "  \"results\": [\n";

    for (size_t ri = 0; ri < results.size(); ++ri) {
        const auto& r = results[ri];
        ofs << "    {\n";
        ofs << "      \"batch_size\": " << r.batch_size << ",\n";
        ofs << "      \"prompt_len\": " << r.prompt_len << ",\n";
        ofs << "      \"iteration\": " << r.iteration << ",\n";
        ofs << "      \"decode_steps\": " << r.decode_steps << ",\n";
        ofs << "      \"warmup_steps\": " << r.warmup_steps << ",\n";
        ofs << "      \"cuda_graph\": " << (r.cuda_graph ? "true" : "false") << ",\n";
        ofs << "      \"weight_bytes\": " << std::fixed << std::setprecision(0) << r.weight_bytes << ",\n";

        // Derived metrics
        float itl = r.decode_total.median();
        float decode_tps = (itl > 0) ? (float)r.batch_size * 1000.0f / itl : 0;
        float bw = (itl > 0) ? r.weight_bytes / (itl / 1000.0f) / 1e9f : 0;
        float ttft = r.prefill_ttft.median();
        float prefill_tps = (ttft > 0) ? (float)r.prompt_len * 1000.0f / ttft : 0;

        ofs << "      \"itl_median_ms\": " << std::setprecision(3) << itl << ",\n";
        ofs << "      \"itl_ci95_ms\": " << r.decode_total.ci95() << ",\n";
        ofs << "      \"itl_cv_pct\": " << r.decode_total.cv_pct() << ",\n";
        ofs << "      \"decode_tok_per_sec\": " << std::setprecision(2) << decode_tps << ",\n";
        ofs << "      \"bw_GBs\": " << std::setprecision(1) << bw << ",\n";
        ofs << "      \"ttft_median_ms\": " << std::setprecision(3) << ttft << ",\n";
        ofs << "      \"prefill_tok_per_sec\": " << std::setprecision(1) << prefill_tps << ",\n";

        // Detailed stats
        auto write_section = [&](const char* name, const Stats& st) {
            ofs << "      \"" << name << "\": {\n";
            json_write_stats(ofs, st, "        ");
            ofs << "\n      }";
        };

        write_section("prefill_ttft", r.prefill_ttft);
        ofs << ",\n";
        write_section("prefill_forward", r.prefill_forward);
        ofs << ",\n";
        write_section("decode_total", r.decode_total);
        ofs << ",\n";
        write_section("decode_forward", r.decode_forward);
        ofs << "\n";

        ofs << "    }" << (ri + 1 < results.size() ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();
    printf("  JSON results written to: %s\n", path.c_str());
}

// ============================================================================
// Per-request context
// ============================================================================
struct ReqCtx {
    std::vector<int> block_table;
    int context_len;
    std::vector<__nv_bfloat16*> ssm_states;
    std::vector<__nv_bfloat16*> conv_states;
    int last_token;
};

// ============================================================================
// run_single_bench — 单次 (batch_size, prompt_len) 配置的完整测量
//
// 模型和 stream 由外部持有, 这里只分配/释放每次运行的临时资源
// ============================================================================
static BenchResult run_single_bench(
    const BenchConfig& cfg,
    const core::Qwen35Config& config,
    core::Qwen35Model* model,
    cudaStream_t stream,
    int batch_size,
    int prompt_len,
    int iteration,
    size_t total_weight_bytes)
{
    BenchResult result;
    result.batch_size   = batch_size;
    result.prompt_len   = prompt_len;
    result.iteration    = iteration;
    result.decode_steps = cfg.decode_steps;
    result.warmup_steps = cfg.warmup_steps;
    result.weight_bytes = (float)total_weight_bytes;
    result.cuda_graph   = !cfg.no_graph;

    const int hs = config.hidden_size;
    const int is = config.intermediate_size;
    const int qp_dim = config.q_proj_dim();
    const int kv_dim = config.kv_dim();
    const int qk = config.lin_qk_dim();
    const int lin_v = config.lin_v_dim();
    const int nkh = config.linear_num_key_heads;
    const int in_qkv = 2 * qk + lin_v;
    const int nv = config.linear_num_value_heads;

    int num_full_attn_layers = config.num_full_attn_layers();

    // Count linear layers
    int num_lin = 0;
    for (int i = 0; i < config.num_hidden_layers; ++i)
        if (!config.is_full_attention(i)) ++num_lin;

    // Allocate KV cache
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    cache::ModelCacheParams mcp;
    mcp.num_full_attn_layers   = num_full_attn_layers;
    mcp.num_kv_heads           = config.num_key_value_heads;
    mcp.head_dim               = config.head_dim;
    mcp.block_size             = 16;
    mcp.num_linear_attn_layers = config.num_hidden_layers - num_full_attn_layers;
    mcp.nkh                    = config.linear_num_key_heads;
    mcp.kd                     = config.linear_key_head_dim;
    mcp.v_per_kh               = config.lin_v_per_kh();
    mcp.in_qkv                 = in_qkv;
    mcp.conv_k_minus_1         = config.linear_conv_kernel_dim - 1;

    cache::CacheConfig cache_cfg;
    cache_cfg.kv_cache_budget_gb = cfg.kv_cache_gb;
    auto capacity = cache::CapacityPlanner::plan(cache_cfg, mcp);
    int num_kv_blocks = capacity.gpu_kv_blocks;

    // Capacity check
    int total_kv_tokens_needed = batch_size * (prompt_len + cfg.decode_steps);
    int total_kv_blocks_needed = (total_kv_tokens_needed + 15) / 16;
    if (total_kv_blocks_needed > num_kv_blocks) {
        printf("  ⚠ WARNING: need %d KV blocks but only have %d — results may be inaccurate\n",
               total_kv_blocks_needed, num_kv_blocks);
    }

    auto kv_manager = std::make_unique<ops::KVCacheManager>(
        num_kv_blocks, 16, config.num_key_value_heads, config.head_dim,
        core::DataType::FP16, allocator, num_full_attn_layers);

    // Allocate buffers
    int max_tokens = std::max(prompt_len + 64, batch_size);
    int max_kv_blks_per_seq = ((prompt_len + cfg.decode_steps + 15) / 16) + 4;

    __nv_bfloat16* d_hidden_states = nullptr;
    int* d_pos_ids = nullptr;
    int* d_block_tables = nullptr;
    int* d_context_lens = nullptr;
    int* d_argmax_result = nullptr;
    __nv_bfloat16* d_workspace = nullptr;

    cudaMalloc(&d_hidden_states, (size_t)max_tokens * hs * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos_ids, (size_t)max_tokens * sizeof(int));
    cudaMalloc(&d_block_tables, (size_t)batch_size * max_kv_blks_per_seq * sizeof(int));
    cudaMalloc(&d_context_lens, (size_t)batch_size * sizeof(int));
    cudaMallocManaged(&d_argmax_result, (size_t)batch_size * sizeof(int));

    // Workspace sizing
    size_t ws_full = (size_t)(4*hs + qp_dim + config.q_dim() + 2*kv_dim + 3*is);
    size_t ws_linear = (size_t)(hs + in_qkv + lin_v + nv + qk + lin_v + hs + hs + 3*is + hs + nkh*2);
    size_t ws_per_tok = std::max(ws_full, ws_linear);
    size_t ws_total = ws_per_tok * max_tokens + (size_t)batch_size * config.vocab_size + (size_t)batch_size * hs;
    cudaMalloc(&d_workspace, ws_total * sizeof(__nv_bfloat16));

    // SSM/Conv states per request
    size_t ssm_sz = (size_t)nkh * config.linear_key_head_dim * config.lin_v_per_kh() * sizeof(__nv_bfloat16);
    size_t conv_sz = (size_t)in_qkv * (config.linear_conv_kernel_dim - 1) * sizeof(__nv_bfloat16);

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

    // SSM/Conv pointer arrays
    __nv_bfloat16** d_ssm_ptrs = nullptr;
    __nv_bfloat16** d_conv_ptrs = nullptr;
    cudaMallocManaged(&d_ssm_ptrs, (size_t)num_lin * batch_size * sizeof(__nv_bfloat16*));
    cudaMallocManaged(&d_conv_ptrs, (size_t)num_lin * batch_size * sizeof(__nv_bfloat16*));
    for (int li = 0; li < num_lin; ++li) {
        for (int bi = 0; bi < batch_size; ++bi) {
            d_ssm_ptrs[li * batch_size + bi] = requests[bi].ssm_states[li];
            d_conv_ptrs[li * batch_size + bi] = requests[bi].conv_states[li];
        }
    }

    // ---- Prefill warmup (1x, no timing) ----
    {
        int default_tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                                248046, 198, 248045, 74455, 198, 248068, 198};
        std::vector<int> warm_tokens(prompt_len);
        for (int i = 0; i < prompt_len; ++i)
            warm_tokens[i] = (i < 17) ? default_tokens[i] : 1;

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

        std::vector<__nv_bfloat16*> single_ssm(num_lin);
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

        // Release warmup blocks & reset states
        kv_manager->free_blocks(btable);
        for (int li = 0; li < num_lin; ++li) {
            cudaMemset(requests[0].ssm_states[li], 0, ssm_sz);
            cudaMemset(requests[0].conv_states[li], 0, conv_sz);
        }
    }

    // ---- Prefill — measured (prefill_repeat × batch_size) ----
    {
        int default_tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                                248046, 198, 248045, 74455, 198, 248068, 198};
        std::vector<int> prompt_tokens(prompt_len);
        for (int i = 0; i < prompt_len; ++i)
            prompt_tokens[i] = (i < 17) ? default_tokens[i] : 1;

        EventTimer pf_embed, pf_forward, pf_lmhead, pf_total;

        for (int rep = 0; rep < cfg.prefill_repeat; ++rep) {
            for (int b = 0; b < batch_size; ++b) {
                // Reset SSM/Conv state for clean measurement
                for (int li = 0; li < num_lin; ++li) {
                    cudaMemset(requests[b].ssm_states[li], 0, ssm_sz);
                    cudaMemset(requests[b].conv_states[li], 0, conv_sz);
                }

                // Allocate KV blocks
                int num_blocks_needed = (prompt_len + 15) / 16;
                auto btable = kv_manager->allocate_blocks(num_blocks_needed);
                requests[b].block_table = btable;
                requests[b].context_len = prompt_len;

                pf_total.record_start(stream);

                // Embedding
                pf_embed.record_start(stream);
                cudaMemcpyAsync(d_pos_ids, prompt_tokens.data(), prompt_len * sizeof(int),
                                cudaMemcpyHostToDevice, stream);
                ops::invoke_embedding_lookup(d_hidden_states, d_pos_ids, model->get_embed_tokens(),
                                             prompt_len, hs, stream);
                std::vector<int> pos_ids(prompt_len);
                for (int i = 0; i < prompt_len; ++i) pos_ids[i] = i;
                cudaMemcpyAsync(d_pos_ids, pos_ids.data(), prompt_len * sizeof(int),
                                cudaMemcpyHostToDevice, stream);
                pf_embed.record_stop(stream);

                // Block tables & context lens
                cudaMemcpyAsync(d_block_tables, btable.data(),
                                btable.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
                int cl = prompt_len;
                cudaMemcpyAsync(d_context_lens, &cl, sizeof(int), cudaMemcpyHostToDevice, stream);

                // Single-request state pointers
                std::vector<__nv_bfloat16*> single_ssm(num_lin);
                std::vector<__nv_bfloat16*> single_conv(num_lin);
                for (int li = 0; li < num_lin; ++li) {
                    single_ssm[li] = requests[b].ssm_states[li];
                    single_conv[li] = requests[b].conv_states[li];
                }

                // Forward
                pf_forward.record_start(stream);
                model->forward_prefill(d_hidden_states, d_pos_ids, *kv_manager,
                               d_block_tables, d_context_lens,
                               (int)btable.size(), cl, prompt_len,
                               single_ssm.data(), single_conv.data(),
                               d_workspace, stream);
                pf_forward.record_stop(stream);

                // Final norm + lm_head + argmax
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

                pf_total.record_stop(stream);
                cudaStreamSynchronize(stream);

                float ttft_ms  = pf_total.elapsed_ms();
                float emb_ms   = pf_embed.elapsed_ms();
                float fwd_ms   = pf_forward.elapsed_ms();
                float lmh_ms   = pf_lmhead.elapsed_ms();

                result.prefill_ttft.add(ttft_ms);
                result.prefill_embed.add(emb_ms);
                result.prefill_forward.add(fwd_ms);
                result.prefill_lmhead.add(lmh_ms);

                requests[b].last_token = *d_argmax_result;
                requests[b].context_len++;

                // Release blocks for next repeat (last repeat keeps them for decode)
                if (rep < cfg.prefill_repeat - 1) {
                    kv_manager->free_blocks(btable);
                }
            }
        }
    }

    // ---- Decode benchmark ----
    int total_steps = cfg.warmup_steps + cfg.decode_steps;
    const int fixed_max_blks = max_kv_blks_per_seq;

    std::vector<int> h_token_ids(batch_size);
    std::vector<int> h_pos_ids(batch_size);
    std::vector<int> h_ctx_lens(batch_size);
    std::vector<int> h_block_tables_flat(batch_size * fixed_max_blks, 0);

    // CUDA Graph
    cudaGraph_t decode_graph = nullptr;
    cudaGraphExec_t decode_graph_exec = nullptr;
    bool graph_captured = false;

    __nv_bfloat16* norm_out_graph = d_workspace + ws_per_tok * max_tokens;
    __nv_bfloat16* logits_graph = norm_out_graph + batch_size * hs;

    EventTimer t_total, t_embed, t_forward, t_norm, t_lmhead, t_sample;

    for (int step = 0; step < total_steps; ++step) {
        bool is_warmup = (step < cfg.warmup_steps);

        char nvtx_name[64];
        snprintf(nvtx_name, sizeof(nvtx_name), "%s_step_%d", is_warmup ? "warmup" : "decode", step);
        nvtx_push(nvtx_name);

        // Assemble batch input
        int max_ctx = 0;
        for (int b = 0; b < batch_size; ++b) {
            h_token_ids[b] = requests[b].last_token;
            h_pos_ids[b] = requests[b].context_len - 1;
            h_ctx_lens[b] = requests[b].context_len;
            max_ctx = std::max(max_ctx, requests[b].context_len);
        }

        std::fill(h_block_tables_flat.begin(), h_block_tables_flat.end(), 0);
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < (int)requests[b].block_table.size(); ++j) {
                h_block_tables_flat[b * fixed_max_blks + j] = requests[b].block_table[j];
            }
        }

        t_total.record_start(stream);

        // Embedding
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

        // Upload block tables and context lens
        cudaMemcpyAsync(d_block_tables, h_block_tables_flat.data(),
                        batch_size * fixed_max_blks * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_context_lens, h_ctx_lens.data(), batch_size * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        if (graph_captured) {
            t_forward.record_start(stream);
            cudaGraphLaunch(decode_graph_exec, stream);
            t_forward.record_stop(stream);

            t_norm.record_start(stream); t_norm.record_stop(stream);
            t_lmhead.record_start(stream); t_lmhead.record_stop(stream);
            t_sample.record_start(stream); t_sample.record_stop(stream);
        } else {
            // Forward
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

            // Final norm
            nvtx_push("final_norm");
            t_norm.record_start(stream);
            ops::invoke_rmsnorm(norm_out_graph, d_hidden_states, model->get_norm_weight(),
                                config.rms_norm_eps, batch_size, hs, stream);
            t_norm.record_stop(stream);
            nvtx_pop();

            // LM Head
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

            // Argmax
            nvtx_push("sample");
            t_sample.record_start(stream);
            ops::invoke_batched_argmax(logits_graph, d_argmax_result, config.vocab_size,
                                       batch_size, stream);
            t_sample.record_stop(stream);
            nvtx_pop();

            // Capture CUDA Graph after last warmup step
            if (is_warmup && step == cfg.warmup_steps - 1 && !cfg.no_graph) {
                cudaStreamSynchronize(stream);

                cudaStream_t capture_stream;
                cudaStreamCreate(&capture_stream);
                cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

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
            }
        }

        t_total.record_stop(stream);
        float ms_total   = t_total.elapsed_ms();
        float ms_embed   = t_embed.elapsed_ms();
        float ms_forward = t_forward.elapsed_ms();
        float ms_norm    = t_norm.elapsed_ms();
        float ms_lmhead  = t_lmhead.elapsed_ms();
        float ms_sample  = t_sample.elapsed_ms();

        // Update request state
        for (int b = 0; b < batch_size; ++b) {
            requests[b].last_token = d_argmax_result[b];
            requests[b].context_len++;
            if (requests[b].context_len > (int)requests[b].block_table.size() * 16) {
                auto new_blks = kv_manager->allocate_blocks(1);
                if (!new_blks.empty()) requests[b].block_table.push_back(new_blks[0]);
            }
        }

        if (!is_warmup) {
            result.decode_total.add(ms_total);
            result.decode_forward.add(ms_forward);
            result.decode_embed.add(ms_embed);
            result.decode_norm.add(ms_norm);
            result.decode_lmhead.add(ms_lmhead);
            result.decode_sample.add(ms_sample);
        }

        if (cfg.per_step || is_warmup) {
            printf("    [%s %3d] total=%7.2f  fwd=%7.2f  embed=%5.2f  norm=%5.2f  "
                   "lm=%7.2f  sample=%5.2f  tok=%d\n",
                   is_warmup ? "warmup" : "decode", step, ms_total, ms_forward, ms_embed,
                   ms_norm, ms_lmhead, ms_sample, requests[0].last_token);
        }

        nvtx_pop(); // step
    }

    // ---- Cleanup ----
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

    return result;
}

// ============================================================================
// 打印单次结果 (紧凑格式)
// ============================================================================
static void print_single_result(const BenchResult& r) {
    float itl = r.decode_total.median();
    float itl_ci = r.decode_total.ci95();
    float itl_cv = r.decode_total.cv_pct();
    float decode_tps = (itl > 0) ? (float)r.batch_size * 1000.0f / itl : 0;
    float bw = (itl > 0) ? r.weight_bytes / (itl / 1000.0f) / 1e9f : 0;
    float ttft = r.prefill_ttft.median();
    float ttft_ci = r.prefill_ttft.ci95();
    float prefill_tps = (ttft > 0) ? (float)r.prompt_len * 1000.0f / ttft : 0;

    printf("    ┌────────────────────────────────────────────────────────────┐\n");
    printf("    │ B=%-3d  prompt=%-5d  iter=%d  graph=%s  steps=%d+%d        │\n",
           r.batch_size, r.prompt_len, r.iteration,
           r.cuda_graph ? "ON " : "OFF", r.warmup_steps, r.decode_steps);
    printf("    ├────────────────────────────────────────────────────────────┤\n");
    printf("    │ TTFT:  %7.1f ms ±%.1f  (N=%d, CV=%.1f%%)                 │\n",
           ttft, ttft_ci, r.prefill_ttft.count(), r.prefill_ttft.cv_pct());
    printf("    │        prefill tok/s: %.0f                                │\n", prefill_tps);
    printf("    │        embed=%.1fms  fwd=%.1fms  lmhead=%.1fms             │\n",
           r.prefill_embed.median(), r.prefill_forward.median(), r.prefill_lmhead.median());
    printf("    │ ITL:   %7.2f ms ±%.2f  (N=%d, CV=%.1f%%)                 │\n",
           itl, itl_ci, r.decode_total.count(), itl_cv);
    printf("    │        p95=%.2f  p99=%.2f  trimmed=%.2f                   │\n",
           r.decode_total.p95(), r.decode_total.p99(), r.decode_total.trimmed_mean());
    printf("    │ Decode tok/s: %.2f   BW: %.1f GB/s (peak=273)             │\n",
           decode_tps, bw);
    printf("    │ Phase:  fwd=%.2f  embed=%.2f  norm=%.2f  lm=%.2f  samp=%.2f│\n",
           r.decode_forward.median(), r.decode_embed.median(), r.decode_norm.median(),
           r.decode_lmhead.median(), r.decode_sample.median());
    printf("    └────────────────────────────────────────────────────────────┘\n");
}

// ============================================================================
// 打印扫描结果汇总表
// ============================================================================
static void print_sweep_summary(const std::vector<BenchResult>& results, size_t weight_bytes) {
    if (results.size() <= 1) return;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Sweep Summary                                                             ║\n");
    printf("╠═══════╦═══════╦═══════╦══════════╦════════╦════════╦═══════╦═══════╦════════╣\n");
    printf("║ Batch ║Prompt ║ Iter  ║ TTFT(ms) ║ITL(ms) ║tok/s   ║BW GB/s║CV(%%) ║±CI95ms║\n");
    printf("╠═══════╬═══════╬═══════╬══════════╬════════╬════════╬═══════╬═══════╬════════╣\n");

    for (const auto& r : results) {
        float itl = r.decode_total.median();
        float tps = (itl > 0) ? (float)r.batch_size * 1000.0f / itl : 0;
        float bw = (itl > 0) ? r.weight_bytes / (itl / 1000.0f) / 1e9f : 0;
        float ttft = r.prefill_ttft.median();

        printf("║ %5d ║ %5d ║ %5d ║ %8.1f ║%7.2f ║%7.2f ║%6.1f ║%5.1f ║ %5.2f ║\n",
               r.batch_size, r.prompt_len, r.iteration,
               ttft, itl, tps, bw, r.decode_total.cv_pct(), r.decode_total.ci95());
    }

    printf("╚═══════╩═══════╩═══════╩══════════╩════════╩════════╩═══════╩═══════╩════════╝\n");
}

// ============================================================================
// CSV 输出
// ============================================================================
static void print_csv(const std::vector<BenchResult>& results) {
    printf("\n--- CSV ---\n");
    printf("batch_size,prompt_len,iteration,ttft_median_ms,ttft_ci95_ms,ttft_cv_pct,"
           "prefill_tok_per_sec,itl_median_ms,itl_p95_ms,itl_p99_ms,itl_ci95_ms,itl_cv_pct,"
           "itl_trimmed_mean_ms,decode_tok_per_sec,bw_GBs,weight_MB,"
           "fwd_ms,embed_ms,norm_ms,lmhead_ms,sample_ms\n");

    for (const auto& r : results) {
        float itl = r.decode_total.median();
        float tps = (itl > 0) ? (float)r.batch_size * 1000.0f / itl : 0;
        float bw = (itl > 0) ? r.weight_bytes / (itl / 1000.0f) / 1e9f : 0;
        float ttft = r.prefill_ttft.median();
        float prefill_tps = (ttft > 0) ? (float)r.prompt_len * 1000.0f / ttft : 0;

        printf("%d,%d,%d,%.2f,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,"
               "%.2f,%.2f,%.2f,%.2f,%.2f\n",
               r.batch_size, r.prompt_len, r.iteration,
               ttft, r.prefill_ttft.ci95(), r.prefill_ttft.cv_pct(),
               prefill_tps,
               itl, r.decode_total.p95(), r.decode_total.p99(),
               r.decode_total.ci95(), r.decode_total.cv_pct(), r.decode_total.trimmed_mean(),
               tps, bw, r.weight_bytes / 1e6f,
               r.decode_forward.median(), r.decode_embed.median(),
               r.decode_norm.median(), r.decode_lmhead.median(), r.decode_sample.median());
    }
}

// ============================================================================
// 主程序
// ============================================================================
int run_benchmark(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);
    g_nvtx_enabled = cfg.nsys_mode;

    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    int total_configs = (int)cfg.batch_sizes.size() * (int)cfg.prompt_lens.size() * cfg.iterations;

    std::cout << "========================================\n"
              << "  Qwen3.5 Benchmark (SM110 Thor)\n"
              << "========================================\n"
              << "  Warmup steps  : " << cfg.warmup_steps << "\n"
              << "  Decode steps  : " << cfg.decode_steps  << "\n"
              << "  Prompt lens   : ";
    for (size_t i = 0; i < cfg.prompt_lens.size(); ++i)
        std::cout << (i ? "," : "") << cfg.prompt_lens[i];
    std::cout << "\n"
              << "  Batch sizes   : ";
    for (size_t i = 0; i < cfg.batch_sizes.size(); ++i)
        std::cout << (i ? "," : "") << cfg.batch_sizes[i];
    std::cout << "\n"
              << "  Iterations    : " << cfg.iterations << "\n"
              << "  Prefill repeat: " << cfg.prefill_repeat << "\n"
              << "  KV Cache GB   : " << cfg.kv_cache_gb   << "\n"
              << "  CUDA Graph    : " << (cfg.no_graph ? "OFF" : "ON") << "\n"
              << "  NVTX markers  : " << (cfg.nsys_mode ? "ON" : "OFF") << "\n"
              << "  Total configs : " << total_configs << "\n"
              << "  JSON output   : " << (cfg.json_output.empty() ? "(none)" : cfg.json_output) << "\n"
              << "========================================\n\n";

    // ========================================================================
    // 1. Load model (once)
    // ========================================================================
    nvtx_push("model_init");
    core::Qwen35Config config;
    config.model_dir = cfg.model_dir;
    config.load_from_model_dir(cfg.model_dir);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[1/3] Loading model weights...\n";
    auto model = std::make_unique<core::Qwen35Model>(config);
    model->load_weights(cfg.model_dir);
    std::cout << "      Model loaded. (hidden=" << config.hidden_size
              << " layers=" << config.num_hidden_layers
              << " vocab=" << config.vocab_size << ")\n";

    // L2 Cache Persistence
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
            std::cout << "      L2 Cache Persistence: " << persist_size / 1024 << " KB\n";
        }
    }
    nvtx_pop();

    // Compute total weight bytes
    const int hs = config.hidden_size;
    const int is_dim = config.intermediate_size;
    const int qp_dim = config.q_proj_dim();
    const int kv_dim = config.kv_dim();
    const int qk = config.lin_qk_dim();
    const int lin_v = config.lin_v_dim();
    const int nkh = config.linear_num_key_heads;
    const int in_qkv = 2 * qk + lin_v;
    const int nv = config.linear_num_value_heads;
    int num_full_attn_layers = config.num_full_attn_layers();
    int n_linear_layers = config.num_hidden_layers - num_full_attn_layers;

    size_t la_params = (size_t)in_qkv * hs + (lin_v + 2*nv) * hs + hs * lin_v
                       + (size_t)is_dim * hs + is_dim * hs + hs * is_dim;
    size_t fa_params = (size_t)(qp_dim + 2*kv_dim) * hs + hs * config.q_dim()
                       + (size_t)is_dim * hs + is_dim * hs + hs * is_dim;
    size_t total_weight_bytes = (n_linear_layers * la_params + num_full_attn_layers * fa_params
                                 + (size_t)config.vocab_size * hs) * 2;

    printf("      Weight size: %.1f MB (BF16)\n\n", (float)total_weight_bytes / 1e6);

    // ========================================================================
    // 2. Run sweep
    // ========================================================================
    std::vector<BenchResult> all_results;
    int config_idx = 0;

    for (int pl : cfg.prompt_lens) {
        for (int bs : cfg.batch_sizes) {
            for (int iter = 0; iter < cfg.iterations; ++iter) {
                config_idx++;
                printf("[2/3] Config %d/%d: B=%d, prompt=%d, iter=%d/%d\n",
                       config_idx, total_configs, bs, pl, iter + 1, cfg.iterations);

                auto result = run_single_bench(cfg, config, model.get(), stream,
                                               bs, pl, iter + 1, total_weight_bytes);

                print_single_result(result);
                all_results.push_back(std::move(result));
            }
        }
    }

    // ========================================================================
    // 3. Summary
    // ========================================================================
    std::cout << "\n[3/3] Results\n";

    if (all_results.size() == 1) {
        // Single config — detailed output (backward compatible)
        const auto& r = all_results[0];
        float itl = r.decode_total.median();
        float itl_p95 = r.decode_total.p95();
        float itl_p99 = r.decode_total.p99();
        float decode_tps = (float)r.batch_size * 1000.0f / itl;
        float decode_tps_mean = (float)r.batch_size * 1000.0f / r.decode_total.mean();
        float bw_median = r.weight_bytes / (itl / 1000.0f) / 1e9f;
        float bw_mean = r.weight_bytes / (r.decode_total.mean() / 1000.0f) / 1e9f;
        float ttft = r.prefill_ttft.median();
        float ttft_p95 = r.prefill_ttft.p95();
        float ttft_p99 = r.prefill_ttft.p99();
        float prefill_tps = (float)r.prompt_len * 1000.0f / ttft;

        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════╗\n");
        printf("║        Standard LLM Inference Metrics (batch=%d, prompt=%d)  ║\n", r.batch_size, r.prompt_len);
        printf("╠══════════════════════════════════════════════════════════════════╣\n");
        printf("║                                                                ║\n");
        printf("║  ▸ TTFT (Time To First Token)                                  ║\n");
        printf("║      Median:  %8.1f ms ±%.1f  (N=%d, CV=%.1f%%)               ║\n",
               ttft, r.prefill_ttft.ci95(), r.prefill_ttft.count(), r.prefill_ttft.cv_pct());
        printf("║      P95:     %8.1f ms                                      ║\n", ttft_p95);
        printf("║      P99:     %8.1f ms                                      ║\n", ttft_p99);
        printf("║                                                                ║\n");
        printf("║  ▸ Prefill Throughput                                          ║\n");
        printf("║      Prefill tok/s:   %7.0f  (%d tokens / %.1f ms)            ║\n",
               prefill_tps, r.prompt_len, ttft);
        printf("║      Breakdown:  embed=%.1fms  fwd=%.1fms  lmhead=%.1fms       ║\n",
               r.prefill_embed.median(), r.prefill_forward.median(),
               r.prefill_lmhead.median());
        printf("║                                                                ║\n");
        printf("║  ▸ ITL (Inter-Token Latency) — Decode                         ║\n");
        printf("║      Median:  %8.2f ms ±%.2f  (N=%d, CV=%.1f%%)              ║\n",
               itl, r.decode_total.ci95(), r.decode_total.count(), r.decode_total.cv_pct());
        printf("║      P95:     %8.2f ms                                      ║\n", itl_p95);
        printf("║      P99:     %8.2f ms                                      ║\n", itl_p99);
        printf("║      Trimmed: %8.2f ms (10%% trim)                           ║\n",
               r.decode_total.trimmed_mean());
        printf("║                                                                ║\n");
        printf("║  ▸ Decode Throughput                                           ║\n");
        printf("║      tok/s (median): %8.2f  (%d tok/step)                    ║\n",
               decode_tps, r.batch_size);
        printf("║      tok/s (mean):   %8.2f                                   ║\n", decode_tps_mean);
        printf("║                                                                ║\n");
        printf("║  ▸ Memory Bandwidth                                           ║\n");
        printf("║      Weight BW (median): %6.1f GB/s  (peak=273 GB/s)         ║\n", bw_median);
        printf("║      Weight BW (mean):   %6.1f GB/s                          ║\n", bw_mean);
        printf("║      Weight size:        %6.1f MB (BF16)                     ║\n",
               r.weight_bytes / 1e6f);
        printf("║                                                                ║\n");
        printf("╠══════════════════════════════════════════════════════════════════╣\n");
        printf("║  Decode Phase Breakdown                                        ║\n");
        printf("║  ┌──────────┬────────┬────────┬────────┬───────┬──────┐        ║\n");
        printf("║  │ Phase    │Mean ms │Med  ms │Min  ms │Max ms │ P95  │        ║\n");
        printf("║  ├──────────┼────────┼────────┼────────┼───────┼──────┤        ║\n");
        printf("║  │ Total    │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_total.mean(), r.decode_total.median(), r.decode_total.min_val(),
               r.decode_total.max_val(), r.decode_total.p95());
        printf("║  │ Forward  │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_forward.mean(), r.decode_forward.median(), r.decode_forward.min_val(),
               r.decode_forward.max_val(), r.decode_forward.p95());
        printf("║  │ Embed    │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_embed.mean(), r.decode_embed.median(), r.decode_embed.min_val(),
               r.decode_embed.max_val(), r.decode_embed.p95());
        printf("║  │ Norm     │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_norm.mean(), r.decode_norm.median(), r.decode_norm.min_val(),
               r.decode_norm.max_val(), r.decode_norm.p95());
        printf("║  │ LM Head  │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_lmhead.mean(), r.decode_lmhead.median(), r.decode_lmhead.min_val(),
               r.decode_lmhead.max_val(), r.decode_lmhead.p95());
        printf("║  │ Sample   │%7.2f │%7.2f │%7.2f │%6.1f │%5.1f │        ║\n",
               r.decode_sample.mean(), r.decode_sample.median(), r.decode_sample.min_val(),
               r.decode_sample.max_val(), r.decode_sample.p95());
        printf("║  └──────────┴────────┴────────┴────────┴───────┴──────┘        ║\n");
        printf("║  Config: %d warmup + %d decode steps, CUDA Graph: %s           ║\n",
               r.warmup_steps, r.decode_steps, r.cuda_graph ? "ON" : "OFF");
        printf("║  Prefill: %d repeats, N=%d total measurements                  ║\n",
               cfg.prefill_repeat, r.prefill_ttft.count());
        printf("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    // Sweep summary table (always, if >1 config)
    print_sweep_summary(all_results, total_weight_bytes);

    // CSV output
    if (cfg.csv_output) {
        print_csv(all_results);
    }

    // JSON output
    if (!cfg.json_output.empty()) {
        write_json(cfg.json_output, all_results, cfg, config);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    cudaStreamDestroy(stream);

    return 0;
}
