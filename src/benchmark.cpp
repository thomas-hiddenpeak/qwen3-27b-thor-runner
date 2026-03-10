// benchmark.cpp — Engine-based 性能基准测试
//
// 通过 InferenceBackend 走完整推理路径, 测量真实 TTFT/吞吐量:
//   - Chunked prefill (max_chunk_size)
//   - MTP 投机解码
//   - GPU 采样 (top-k/top-p/min-p/penalties)
//   - KV Cache 管理
//
// 用法:
//   ./qwen35-thor bench --config configs/qwen3.5-27b.conf --decode 50
//   ./qwen35-thor bench --config configs/qwen3.5-27b.conf --prompt-len 17,64,256 --iterations 3
//   ./qwen35-thor bench --model-dir <path> --decode 30 --json results.json
//
// 参数:
//   --decode N / --max-tokens N   每请求生成 token 数 (默认 50)
//   --prompt-len N[,N..]          逗号分隔 prompt 长度列表 (默认 17)
//   --iterations N                每配置独立迭代次数 (默认 1)
//   --warmup N                    预热请求数 (默认 1, 不计入统计)
//   --json FILE                   JSON 结构化输出
//   --temperature F               采样温度 (0=greedy, 默认 0)
//   --top-p F                     Top-p 采样阈值 (默认 0.95)
//   --top-k N                     Top-k 采样 (默认 20)
//   --seed N                      随机种子 (默认 42, 确定性)
//   --verbose                     显示 Engine 内部日志
//   --config FILE                 同 serve/chat 的配置文件
//   --model-dir DIR               模型目录 (可由 config 或 QWEN_MODEL_DIR 环境变量提供)
//   --kv-cache-gb F               KV Cache 预算 (GB)
//   --mtp-disable                 禁用 MTP 投机解码
//   --max-chunk-size N            Prefill 分块大小 (默认 2048)

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
#include <thread>

#include "engine/backend.h"

using namespace qwen_thor;
using Clock = std::chrono::steady_clock;

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

    float min_val() const { return samples.empty() ? 0 : *std::min_element(samples.begin(), samples.end()); }
    float max_val() const { return samples.empty() ? 0 : *std::max_element(samples.begin(), samples.end()); }

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

    float cv_pct() const {
        float m = mean();
        return (m > 0 && count() >= 2) ? (stddev() / m * 100.0f) : 0;
    }

    float ci95() const {
        if (count() < 2) return 0;
        float t_val = 1.96f;
        if (count() < 30) {
            static const float t_table[] = {
                0, 12.71f, 4.30f, 3.18f, 2.78f, 2.57f,
                2.45f, 2.36f, 2.31f, 2.26f, 2.23f,
                2.20f, 2.18f, 2.16f, 2.14f, 2.13f,
                2.12f, 2.11f, 2.10f, 2.09f, 2.09f,
                2.08f, 2.07f, 2.07f, 2.06f, 2.06f,
                2.06f, 2.05f, 2.05f, 2.05f
            };
            int idx = std::min(count(), 29);
            t_val = t_table[idx];
        }
        return t_val * stddev() / sqrtf((float)count());
    }

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
// Benchmark 配置
// ============================================================================
struct BenchConfig {
    // Engine 配置来源
    std::string config_file;
    std::string model_dir;
    double kv_cache_gb = 0;  // 0 = 不覆盖 config 文件
    std::string mtp_mode;    // empty = 不覆盖 config 文件的值
    int max_chunk_size = 0;  // 0 = 不覆盖

    // 测量参数
    std::vector<int> prompt_lens = {17};
    int max_tokens      = 50;
    int iterations      = 1;
    int warmup_requests = 1;
    std::string json_output;

    // 并发参数
    int concurrent          = 1;     // 并发请求数 (1=顺序, >1=同时提交)
    bool shared_prefix      = false; // 并发请求使用相同 prefix

    // 采样参数
    float temperature       = 0.0f;  // 0 = greedy
    float top_p             = 0.95f;
    int   top_k             = 20;
    float min_p             = 0.0f;
    float presence_penalty  = 0.0f;
    int64_t seed            = 42;

    // 日志
    bool verbose = false;
};

static BenchConfig parse_bench_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--config"     && i + 1 < argc) cfg.config_file  = argv[++i];
        else if (arg == "--model-dir"  && i + 1 < argc) cfg.model_dir    = argv[++i];
        else if (arg == "--kv-cache-gb" && i + 1 < argc) cfg.kv_cache_gb = std::atof(argv[++i]);
        else if (arg == "--decode"     && i + 1 < argc) cfg.max_tokens   = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--max-tokens" && i + 1 < argc) cfg.max_tokens   = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--prompt-len" && i + 1 < argc) cfg.prompt_lens  = parse_int_list(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) cfg.iterations   = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--warmup"     && i + 1 < argc) cfg.warmup_requests = std::max(0, std::atoi(argv[++i]));
        else if (arg == "--json"       && i + 1 < argc) cfg.json_output  = argv[++i];
        else if (arg == "--temperature" && i + 1 < argc) cfg.temperature = std::atof(argv[++i]);
        else if (arg == "--top-p"      && i + 1 < argc) cfg.top_p       = std::atof(argv[++i]);
        else if (arg == "--top-k"      && i + 1 < argc) cfg.top_k       = std::atoi(argv[++i]);
        else if (arg == "--min-p"      && i + 1 < argc) cfg.min_p       = std::atof(argv[++i]);
        else if (arg == "--seed"       && i + 1 < argc) cfg.seed        = std::atoll(argv[++i]);
        else if (arg == "--presence-penalty" && i + 1 < argc) cfg.presence_penalty = std::atof(argv[++i]);
        else if (arg == "--mtp-disable")   cfg.mtp_mode = "off";
        else if (arg == "--mtp-enable")    cfg.mtp_mode = "on";
        else if (arg == "--max-chunk-size" && i + 1 < argc) cfg.max_chunk_size = std::atoi(argv[++i]);
        else if (arg == "--concurrent" && i + 1 < argc) cfg.concurrent = std::max(1, std::atoi(argv[++i]));
        else if (arg == "--shared-prefix") cfg.shared_prefix = true;
        else if (arg == "--verbose")       cfg.verbose = true;
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: qwen35-thor bench [options]\n\n"
                   "Engine options:\n"
                   "  --config FILE           Engine config file (same as serve/chat)\n"
                   "  --model-dir DIR         Model weights directory\n"
                   "  --kv-cache-gb F         KV Cache budget in GB\n"
                   "  --max-chunk-size N      Prefill chunk size\n"
                   "  --mtp-disable           Disable MTP speculative decoding\n"
                   "  --mtp-enable            Enable MTP speculative decoding\n"
                   "\nBenchmark options:\n"
                   "  --decode N              Tokens to generate per request (default: 50)\n"
                   "  --max-tokens N          Same as --decode\n"
                   "  --prompt-len N[,N..]    Prompt lengths, comma-separated (default: 17)\n"
                   "  --iterations N          Iterations per prompt length (default: 1)\n"
                   "  --warmup N              Warmup requests (default: 1)\n"
                   "  --json FILE             JSON output file\n"
                   "  --verbose               Show engine internal logs\n"
                   "\nSampling options:\n"
                   "  --temperature F         Sampling temperature (0=greedy, default: 0)\n"
                   "  --top-p F               Top-p threshold (default: 0.95)\n"
                   "  --top-k N               Top-k (default: 20)\n"
                   "  --seed N                Random seed (default: 42)\n"
                   "\nConcurrency options:\n"
                   "  --concurrent N          Submit N requests simultaneously (default: 1)\n"
                   "  --shared-prefix         All concurrent requests share same prefix tokens\n"
                   "\nExamples:\n"
                   "  bench --config configs/qwen3.5-27b.conf --decode 50\n"
                   "  bench --config configs/qwen3.5-27b.conf --prompt-len 17,64,256 --iterations 3 --json results.json\n"
                   "  bench --config configs/qwen3.5-27b.conf --mtp-disable --decode 30\n"
                   "  bench --config configs/qwen3.5-27b.conf --concurrent 4 --decode 20\n"
                   "  bench --config configs/qwen3.5-27b.conf --concurrent 4 --shared-prefix --prompt-len 256 --decode 20\n");
            exit(0);
        }
    }
    return cfg;
}

// ============================================================================
// 构建 BackendConfig
// ============================================================================
static BackendConfig build_backend_config(const BenchConfig& cfg) {
    BackendConfig bcfg;

    // 1. 从 config 文件加载基础配置
    if (!cfg.config_file.empty()) {
        bcfg = BackendConfig::from_file(cfg.config_file);
    }

    // 2. CLI 覆盖
    if (!cfg.model_dir.empty())  bcfg.model_dir = cfg.model_dir;
    if (cfg.kv_cache_gb > 0)     bcfg.kv_cache_gb = cfg.kv_cache_gb;
    if (!cfg.mtp_mode.empty())   bcfg.mtp_mode = cfg.mtp_mode;
    if (cfg.max_chunk_size > 0)  bcfg.max_chunk_size = std::max(64, std::min(4096, cfg.max_chunk_size));

    // 3. Benchmark 模式: 关闭 SSD prefix cache (避免缓存干扰), 控制日志
    bcfg.cache_enabled = false;
    bcfg.verbose = cfg.verbose;

    // 4. 环境变量 fallback
    if (bcfg.model_dir.empty()) {
        const char* env = std::getenv("QWEN_MODEL_DIR");
        if (env && env[0]) bcfg.model_dir = env;
    }

    if (bcfg.model_dir.empty()) {
        fprintf(stderr, "[Error] --model-dir, --config, or QWEN_MODEL_DIR is required.\n");
        exit(1);
    }

    return bcfg;
}

// ============================================================================
// 单请求结果
// ============================================================================
struct RequestResult {
    int prompt_len;
    int total_tokens;     // 生成的 token 总数
    float ttft_ms;        // Time to First Token
    float generation_ms;  // 首 token 到末 token 的时间
    float total_ms;       // 提交到完成的总时间
    float gen_tps;        // generation throughput: (total_tokens-1) / generation_ms * 1000
    float overall_tps;    // total_tokens / total_ms * 1000
    float prefill_ms;     // Engine 内部 prefill 耗时 (CUDA event 计时)
    float prefill_tps;    // prefill throughput: prompt_len / prefill_ms * 1000
};

// ============================================================================
// 运行单个请求并测量
// ============================================================================
static RequestResult run_single_request(
    InferenceBackend& backend,
    const BenchConfig& cfg,
    int prompt_len,
    uint64_t request_id,
    bool wait_engine_idle = true)
{
    // 等待 Engine 的请求间 idle sleep (200ms) 完成, 避免计入 TTFT
    if (wait_engine_idle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    // 合成 prompt: chat 模板头 + 自然语言填充 (避免 EOS 提前终止)
    // <|im_start|>user\nWrite a quicksort...<|im_end|>\n<|im_start|>assistant\n<|think|>\n
    static const int default_tokens[] = {
        248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
        248046, 198, 248045, 74455, 198, 248068, 198
    };
    // 填充用常见自然语言 token (非特殊 token, 减少 EOS 触发)
    static const int padding_tokens[] = {
        279, 448, 350, 264, 311, 389, 369, 279, 498, 323,
        304, 448, 311, 389, 350, 264, 279, 323, 498, 304,
        279, 448, 279, 350, 389, 311, 264, 498, 304, 323
    };
    InferRequest req;
    req.request_id = request_id;
    req.prompt_tokens.resize(prompt_len);
    for (int i = 0; i < prompt_len; ++i) {
        if (i < 17)
            req.prompt_tokens[i] = default_tokens[i];
        else
            req.prompt_tokens[i] = padding_tokens[(i - 17) % 30];
    }
    req.max_new_tokens    = cfg.max_tokens;
    req.temperature       = cfg.temperature;
    req.top_p             = cfg.top_p;
    req.top_k             = cfg.top_k;
    req.min_p             = cfg.min_p;
    req.presence_penalty  = cfg.presence_penalty;
    req.seed              = cfg.seed;
    req.stream            = true;

    // 提交并计时
    auto t_submit = Clock::now();
    if (!backend.submit(req)) {
        fprintf(stderr, "[Bench] Failed to submit request %lu\n", (unsigned long)request_id);
        return {prompt_len, 0, 0, 0, 0, 0, 0};
    }

    int token_count = 0;
    float prefill_ms_from_engine = 0;
    auto t_first = t_submit;  // 将在首 token 时更新

    while (true) {
        InferResponse resp;
        if (backend.poll(resp)) {
            if (resp.request_id != request_id) continue;
            token_count++;
            if (token_count == 1) {
                t_first = Clock::now();
                prefill_ms_from_engine = resp.prefill_time_ms;
            }
            if (resp.is_finished || resp.error_code != 0) break;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    auto t_end = Clock::now();
    float ttft_ms  = std::chrono::duration<float, std::milli>(t_first - t_submit).count();
    float total_ms = std::chrono::duration<float, std::milli>(t_end - t_submit).count();
    float gen_ms   = std::chrono::duration<float, std::milli>(t_end - t_first).count();
    float gen_tps  = (token_count > 1 && gen_ms > 0) ? (token_count - 1) * 1000.0f / gen_ms : 0;
    float all_tps  = (total_ms > 0) ? token_count * 1000.0f / total_ms : 0;
    float pf_tps   = (prefill_ms_from_engine > 0) ? prompt_len * 1000.0f / prefill_ms_from_engine : 0;

    RequestResult result;
    result.prompt_len = prompt_len;
    result.total_tokens = token_count;
    result.ttft_ms = ttft_ms;
    result.generation_ms = gen_ms;
    result.total_ms = total_ms;
    result.gen_tps = gen_tps;
    result.overall_tps = all_tps;
    result.prefill_ms = prefill_ms_from_engine;
    result.prefill_tps = pf_tps;
    return result;
}

// ============================================================================
// 汇总结果 (按 prompt_len 聚合)
// ============================================================================
struct AggResult {
    int prompt_len;
    int iterations;
    int max_tokens;
    Stats ttft;
    Stats gen_tps;
    Stats total_tokens;
    Stats total_ms;
    Stats prefill_ms;
    Stats prefill_tps;
};

// ============================================================================
// JSON 输出
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

static void write_json(const std::string& path,
                        const std::vector<AggResult>& results,
                        const BenchConfig& cfg,
                        const BackendConfig& bcfg) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        fprintf(stderr, "[Error] Cannot open JSON output file: %s\n", path.c_str());
        return;
    }

    ofs << std::fixed;
    ofs << "{\n";
    ofs << "  \"benchmark\": \"qwen3.5-thor-engine\",\n";
    ofs << "  \"model_dir\": \"" << json_escape(bcfg.model_dir) << "\",\n";
    ofs << "  \"mtp_mode\": \"" << json_escape(bcfg.mtp_mode) << "\",\n";
    ofs << "  \"max_chunk_size\": " << bcfg.max_chunk_size << ",\n";
    ofs << "  \"max_tokens\": " << cfg.max_tokens << ",\n";
    ofs << "  \"temperature\": " << std::setprecision(2) << cfg.temperature << ",\n";
    ofs << "  \"seed\": " << cfg.seed << ",\n";
    ofs << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ofs << "    {\n";
        ofs << "      \"prompt_len\": " << r.prompt_len << ",\n";
        ofs << "      \"iterations\": " << r.iterations << ",\n";
        ofs << "      \"max_tokens\": " << r.max_tokens << ",\n";
        ofs << "      \"ttft_median_ms\": " << std::setprecision(1) << r.ttft.median() << ",\n";
        ofs << "      \"ttft_ci95_ms\": " << std::setprecision(2) << r.ttft.ci95() << ",\n";
        ofs << "      \"ttft_cv_pct\": " << std::setprecision(1) << r.ttft.cv_pct() << ",\n";
        ofs << "      \"prefill_median_ms\": " << std::setprecision(1) << r.prefill_ms.median() << ",\n";
        ofs << "      \"prefill_tps_median\": " << std::setprecision(0) << r.prefill_tps.median() << ",\n";
        ofs << "      \"gen_tps_median\": " << std::setprecision(2) << r.gen_tps.median() << ",\n";
        ofs << "      \"gen_tps_ci95\": " << std::setprecision(2) << r.gen_tps.ci95() << ",\n";
        ofs << "      \"total_tokens_median\": " << std::setprecision(0) << r.total_tokens.median() << ",\n";
        ofs << "      \"total_time_median_ms\": " << std::setprecision(1) << r.total_ms.median() << "\n";
        ofs << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
    ofs.close();
    printf("  JSON results written to: %s\n", path.c_str());
}

// ============================================================================
// 并发 Benchmark: 同时提交 N 个请求, 测量聚合吞吐量
// ============================================================================
struct ConcurrentResult {
    int concurrent;
    int prompt_len;
    int iterations;
    float wall_time_ms;          // 从首个请求提交到最后一个完成的壁钟时间
    float aggregate_tps;         // 总生成 token / wall_time
    int   total_tokens;          // 所有请求生成的 token 总数
    Stats per_req_ttft;          // 各请求的 TTFT 分布
    Stats per_req_tps;           // 各请求的吞吐量分布
    Stats per_req_tokens;        // 各请求的 token 数分布
    float batch_efficiency;      // aggregate_tps / (concurrent × single_req_tps)
};

static int run_concurrent_benchmark(int argc, char** argv) {
    BenchConfig cfg = parse_bench_args(argc, argv);

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Qwen3.5 Concurrent Engine Benchmark                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Concurrent reqs : %-4d                                        ║\n", cfg.concurrent);
    printf("║  Shared prefix   : %-4s                                        ║\n", cfg.shared_prefix ? "yes" : "no");
    printf("║  Prompt len      : %d                                          ║\n", cfg.prompt_lens[0]);
    printf("║  Max tokens      : %-4d                                        ║\n", cfg.max_tokens);
    printf("║  Iterations      : %-4d                                        ║\n", cfg.iterations);
    printf("║  Temperature     : %.2f%s                                      ║\n",
           cfg.temperature, cfg.temperature == 0 ? " (greedy)" : "");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    // 初始化 Backend (启用 ssd cache 若 shared_prefix, 否则关闭)
    printf("[1/4] Initializing engine...\n");
    BackendConfig bcfg = build_backend_config(cfg);
    if (cfg.shared_prefix) {
        bcfg.cache_enabled = true;
        bcfg.cache_dir = "/tmp/qwen_bench_concurrent_" + std::to_string(getpid());
    }
    InferenceBackend backend(bcfg);
    backend.start();
    printf("      Engine ready.\n\n");

    // 构建 prompt tokens (所有请求共享或各自不同)
    int prompt_len = cfg.prompt_lens[0];
    static const int chat_header[] = {
        248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
        248046, 198, 248045, 74455, 198, 248068, 198
    };
    // 填充用自然语言 token
    static const int padding_tokens[] = {
        279, 448, 350, 264, 311, 389, 369, 279, 498, 323,
        304, 448, 311, 389, 350, 264, 279, 323, 498, 304,
        279, 448, 279, 350, 389, 311, 264, 498, 304, 323
    };
    // 不同的 suffix tokens (使各请求生成不同内容, 避免 shared prefix 后完全相同)
    static const int suffix_variants[][10] = {
        {579, 748, 350, 264, 311, 389, 369, 279, 498, 323},
        {604, 848, 450, 364, 411, 489, 469, 379, 598, 423},
        {704, 948, 550, 464, 511, 589, 569, 479, 698, 523},
        {804, 1048, 650, 564, 611, 689, 669, 579, 798, 623},
        {904, 1148, 750, 664, 711, 789, 769, 679, 898, 723},
        {1004, 1248, 850, 764, 811, 889, 869, 779, 998, 823},
        {1104, 1348, 950, 864, 911, 989, 969, 879, 1098, 923},
        {1204, 1448, 1050, 964, 1011, 1089, 1069, 979, 1198, 1023},
    };

    auto build_prompt = [&](int req_idx) -> std::vector<int> {
        std::vector<int> tokens(prompt_len);
        for (int i = 0; i < prompt_len; ++i) {
            if (i < 17) {
                tokens[i] = chat_header[i];
            } else if (cfg.shared_prefix && i >= prompt_len - 10 && prompt_len > 27) {
                // 最后 10 个 token 使用不同 suffix (让各请求生成不同结果)
                int si = i - (prompt_len - 10);
                tokens[i] = suffix_variants[req_idx % 8][si];
            } else {
                tokens[i] = padding_tokens[(i - 17) % 30];
            }
        }
        return tokens;
    };

    // Warmup: 单请求预热
    printf("[2/4] Warmup...\n");
    {
        auto warmup_result = run_single_request(backend, cfg, prompt_len, 1, false);
        printf("      Warmup: TTFT=%.1fms, %d tokens, %.1f tok/s\n\n",
               warmup_result.ttft_ms, warmup_result.total_tokens, warmup_result.gen_tps);
    }

    // 先测量单请求基线 (用于计算 batch_efficiency)
    float baseline_tps = 0;
    {
        printf("[3/4] Baseline (single request)...\n");
        auto r = run_single_request(backend, cfg, prompt_len, 100, true);
        baseline_tps = r.gen_tps;
        printf("      Baseline: TTFT=%.1fms, %.1f tok/s, %d tokens\n\n",
               r.ttft_ms, r.gen_tps, r.total_tokens);
    }

    // 并发测量
    printf("[4/4] Concurrent measurement (N=%d, %d iterations)...\n\n",
           cfg.concurrent, cfg.iterations);

    std::vector<ConcurrentResult> all_results;
    uint64_t next_id = 200;

    for (int iter = 0; iter < cfg.iterations; ++iter) {
        // 等待引擎 idle
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        ConcurrentResult cr;
        cr.concurrent = cfg.concurrent;
        cr.prompt_len = prompt_len;
        cr.iterations = 1;

        // 同时提交 N 个请求
        auto t_start = Clock::now();
        for (int i = 0; i < cfg.concurrent; ++i) {
            InferRequest req;
            req.request_id = next_id++;
            req.prompt_tokens = build_prompt(i);
            req.max_new_tokens = cfg.max_tokens;
            req.temperature = cfg.temperature;
            req.top_p = cfg.top_p;
            req.top_k = cfg.top_k;
            req.min_p = cfg.min_p;
            req.presence_penalty = cfg.presence_penalty;
            req.seed = (cfg.seed >= 0) ? cfg.seed + i : -1;
            req.stream = true;
            // 重试提交 (队列可能暂时满, 等引擎消费)
            int retries = 0;
            while (!backend.submit(req)) {
                if (++retries > 200) {  // 20s 超时
                    fprintf(stderr, "  [Error] Failed to submit request %d after %d retries\n", i, retries);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // 收集所有响应
        struct PerReqState {
            int token_count = 0;
            Clock::time_point t_first;
            bool got_first = false;
            bool finished = false;
        };
        std::unordered_map<uint64_t, PerReqState> states;
        int finished = 0;

        while (finished < cfg.concurrent) {
            InferResponse resp;
            if (backend.poll(resp)) {
                auto& st = states[resp.request_id];
                st.token_count++;
                if (!st.got_first) {
                    st.t_first = Clock::now();
                    st.got_first = true;
                }
                if (resp.is_finished || resp.error_code != 0) {
                    if (!st.finished) {
                        st.finished = true;
                        finished++;
                    }
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        auto t_end = Clock::now();

        // 汇总本轮结果
        float wall_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        int total_tokens = 0;
        for (auto& [id, st] : states) {
            total_tokens += st.token_count;
            float ttft = std::chrono::duration<float, std::milli>(st.t_first - t_start).count();
            float req_ms = std::chrono::duration<float, std::milli>(t_end - st.t_first).count();
            float req_tps = (st.token_count > 1 && req_ms > 0)
                            ? (st.token_count - 1) * 1000.0f / req_ms : 0;
            cr.per_req_ttft.add(ttft);
            cr.per_req_tps.add(req_tps);
            cr.per_req_tokens.add((float)st.token_count);
        }
        cr.wall_time_ms = wall_ms;
        cr.total_tokens = total_tokens;
        cr.aggregate_tps = (wall_ms > 0) ? total_tokens * 1000.0f / wall_ms : 0;
        cr.batch_efficiency = (baseline_tps > 0)
                              ? cr.aggregate_tps / (cfg.concurrent * baseline_tps) : 0;

        printf("  [iter=%d] wall=%.0fms  tokens=%d  agg_tps=%.1f  "
               "per_req_tps=%.1f±%.1f  ttft=%.1f±%.1fms  efficiency=%.1f%%\n",
               iter + 1, wall_ms, total_tokens,
               cr.aggregate_tps,
               cr.per_req_tps.mean(), cr.per_req_tps.stddev(),
               cr.per_req_ttft.mean(), cr.per_req_ttft.stddev(),
               cr.batch_efficiency * 100);

        all_results.push_back(std::move(cr));
    }

    // 汇总
    Stats agg_tps_stats, agg_wall_stats, agg_eff_stats;
    for (auto& r : all_results) {
        agg_tps_stats.add(r.aggregate_tps);
        agg_wall_stats.add(r.wall_time_ms);
        agg_eff_stats.add(r.batch_efficiency * 100);
    }

    printf("\n╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Concurrent Benchmark Summary                                          ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Concurrent requests : %-4d    Shared prefix: %-3s                      ║\n",
           cfg.concurrent, cfg.shared_prefix ? "yes" : "no");
    printf("║  Prompt length       : %-6d  Max tokens   : %-4d                      ║\n",
           prompt_len, cfg.max_tokens);
    printf("║  Baseline (1 req)    : %6.1f tok/s                                     ║\n",
           baseline_tps);
    printf("╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Aggregate throughput: %6.1f tok/s ±%.1f (median: %.1f)                ║\n",
           agg_tps_stats.mean(), agg_tps_stats.ci95(), agg_tps_stats.median());
    printf("║  Wall time           : %6.0f ms ±%.0f                                  ║\n",
           agg_wall_stats.mean(), agg_wall_stats.ci95());
    printf("║  Batch efficiency    : %5.1f%%                                          ║\n",
           agg_eff_stats.mean());
    printf("║  Scaling factor      : %.2fx over single request                       ║\n",
           (baseline_tps > 0) ? agg_tps_stats.mean() / baseline_tps : 0.0f);
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // JSON 输出
    if (!cfg.json_output.empty()) {
        std::ofstream ofs(cfg.json_output);
        if (ofs.is_open()) {
            ofs << std::fixed;
            ofs << "{\n";
            ofs << "  \"benchmark\": \"concurrent-throughput\",\n";
            ofs << "  \"model_dir\": \"" << json_escape(bcfg.model_dir) << "\",\n";
            ofs << "  \"concurrent\": " << cfg.concurrent << ",\n";
            ofs << "  \"shared_prefix\": " << (cfg.shared_prefix ? "true" : "false") << ",\n";
            ofs << "  \"prompt_len\": " << prompt_len << ",\n";
            ofs << "  \"max_tokens\": " << cfg.max_tokens << ",\n";
            ofs << "  \"baseline_tps\": " << std::setprecision(2) << baseline_tps << ",\n";
            ofs << "  \"aggregate_tps_mean\": " << std::setprecision(2) << agg_tps_stats.mean() << ",\n";
            ofs << "  \"aggregate_tps_median\": " << std::setprecision(2) << agg_tps_stats.median() << ",\n";
            ofs << "  \"wall_time_mean_ms\": " << std::setprecision(1) << agg_wall_stats.mean() << ",\n";
            ofs << "  \"batch_efficiency_pct\": " << std::setprecision(1) << agg_eff_stats.mean() << ",\n";
            ofs << "  \"scaling_factor\": " << std::setprecision(2) << (baseline_tps > 0 ? agg_tps_stats.mean() / baseline_tps : 0) << ",\n";
            ofs << "  \"iterations\": " << cfg.iterations << "\n";
            ofs << "}\n";
            ofs.close();
            printf("  JSON results written to: %s\n", cfg.json_output.c_str());
        }
    }

    // 清理
    backend.stop();
    if (cfg.shared_prefix) {
        (void)system(("rm -rf /tmp/qwen_bench_concurrent_" + std::to_string(getpid())).c_str());
    }

    return 0;
}

// ============================================================================
// 主程序
// ============================================================================
int run_benchmark(int argc, char** argv) {
    BenchConfig cfg = parse_bench_args(argc, argv);

    // 并发模式: 路由到专用函数
    if (cfg.concurrent > 1) {
        return run_concurrent_benchmark(argc, argv);
    }

    int total_configs = (int)cfg.prompt_lens.size() * cfg.iterations;
    int total_requests = cfg.warmup_requests + total_configs;

    printf("========================================\n");
    printf("  Qwen3.5 Engine Benchmark\n");
    printf("========================================\n");
    printf("  Prompt lens   : ");
    for (size_t i = 0; i < cfg.prompt_lens.size(); ++i)
        printf("%s%d", i ? "," : "", cfg.prompt_lens[i]);
    printf("\n");
    printf("  Max tokens    : %d\n", cfg.max_tokens);
    printf("  Iterations    : %d\n", cfg.iterations);
    printf("  Warmup reqs   : %d\n", cfg.warmup_requests);
    printf("  Temperature   : %.2f%s\n", cfg.temperature, cfg.temperature == 0 ? " (greedy)" : "");
    printf("  Seed          : %lld\n", (long long)cfg.seed);
    printf("  Total requests: %d (%d warmup + %d measured)\n",
           total_requests, cfg.warmup_requests, total_configs);
    printf("  JSON output   : %s\n", cfg.json_output.empty() ? "(none)" : cfg.json_output.c_str());
    printf("========================================\n\n");

    // ========================================================================
    // 1. 初始化 Engine
    // ========================================================================
    printf("[1/3] Initializing engine...\n");
    BackendConfig bcfg = build_backend_config(cfg);
    InferenceBackend backend(bcfg);
    backend.start();
    printf("      Engine ready.\n\n");

    // ========================================================================
    // 2. Warmup
    // ========================================================================
    uint64_t next_id = 1;
    if (cfg.warmup_requests > 0) {
        printf("[2/3] Warmup (%d request%s)...\n",
               cfg.warmup_requests, cfg.warmup_requests > 1 ? "s" : "");
        for (int w = 0; w < cfg.warmup_requests; ++w) {
            int wpl = cfg.prompt_lens[0];
            auto r = run_single_request(backend, cfg, wpl, next_id++,
                                          w > 0 /* first warmup no wait */);
            printf("      Warmup %d: TTFT=%.1fms, %d tokens, %.1f tok/s\n",
                   w + 1, r.ttft_ms, r.total_tokens, r.gen_tps);
        }
        printf("\n");
    }

    // ========================================================================
    // 3. Measured runs
    // ========================================================================
    printf("[3/3] Measuring...\n\n");

    std::vector<AggResult> agg_results;

    for (int pl : cfg.prompt_lens) {
        AggResult agg;
        agg.prompt_len = pl;
        agg.iterations = cfg.iterations;
        agg.max_tokens = cfg.max_tokens;

        for (int iter = 0; iter < cfg.iterations; ++iter) {
            auto r = run_single_request(backend, cfg, pl, next_id++,
                                          true /* wait engine idle */);

            agg.ttft.add(r.ttft_ms);
            agg.gen_tps.add(r.gen_tps);
            agg.total_tokens.add((float)r.total_tokens);
            agg.total_ms.add(r.total_ms);
            agg.prefill_ms.add(r.prefill_ms);
            agg.prefill_tps.add(r.prefill_tps);

            printf("    [prompt=%d iter=%d] TTFT=%.1fms  prefill=%.1fms (%.0f tok/s)  gen=%.1f tok/s  "
                   "tokens=%d  total=%.0fms\n",
                   pl, iter + 1, r.ttft_ms, r.prefill_ms, r.prefill_tps,
                   r.gen_tps, r.total_tokens, r.total_ms);
        }

        agg_results.push_back(std::move(agg));
    }

    // ========================================================================
    // 4. Summary
    // ========================================================================
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Engine Benchmark Summary                                                  ║\n");
    printf("║  MTP: %-6s  chunk: %-5d  temp: %.1f  seed: %-6lld  max_tokens: %-4d     ║\n",
           bcfg.mtp_mode.c_str(), bcfg.max_chunk_size, cfg.temperature,
           (long long)cfg.seed, cfg.max_tokens);
    printf("╠═══════╦════════════════════╦═══════════════════════╦══════════════════╦════════╦════════╦═════╣\n");
    printf("║Prompt ║  TTFT (ms)         ║  Prefill              ║  Gen tok/s       ║ Tokens ║Time ms║Iters║\n");
    printf("║       ║  med    ±ci95  cv%%  ║  ms     tok/s         ║  med    ±ci95    ║  med   ║  med  ║     ║\n");
    printf("╠═══════╬════════════════════╬═══════════════════════╬══════════════════╬════════╬════════╬═════╣\n");

    for (const auto& r : agg_results) {
        printf("║ %5d ║ %6.1f ±%5.1f %4.1f%% ║ %6.1f  %6.0f tok/s  ║ %6.1f ±%5.2f    ║ %5.0f  ║%6.0f ║ %3d ║\n",
               r.prompt_len,
               r.ttft.median(), r.ttft.ci95(), r.ttft.cv_pct(),
               r.prefill_ms.median(), r.prefill_tps.median(),
               r.gen_tps.median(), r.gen_tps.ci95(),
               r.total_tokens.median(),
               r.total_ms.median(),
               r.iterations);
    }

    printf("╚═══════╩════════════════════╩═══════════════════════╩══════════════════╩════════╩════════╩═════╝\n");

    // 单配置时打印详细信息
    if (agg_results.size() == 1 && agg_results[0].ttft.count() >= 1) {
        const auto& r = agg_results[0];
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════╗\n");
        printf("║  Detailed Results (prompt=%d, max_tokens=%d)                ║\n",
               r.prompt_len, r.max_tokens);
        printf("╠══════════════════════════════════════════════════════════════════╣\n");
        printf("║                                                                ║\n");
        printf("║  ▸ TTFT (Time to First Token)                                  ║\n");
        printf("║      Median:  %8.1f ms ±%.1f  (N=%d, CV=%.1f%%)               ║\n",
               r.ttft.median(), r.ttft.ci95(), r.ttft.count(), r.ttft.cv_pct());
        if (r.ttft.count() > 1) {
            printf("║      P95:     %8.1f ms                                      ║\n", r.ttft.p95());
            printf("║      Min:     %8.1f ms  Max: %.1f ms                        ║\n",
                   r.ttft.min_val(), r.ttft.max_val());
        }
        printf("║                                                                ║\n");
        printf("║  ▸ Prefill (CUDA event timing from engine)                     ║\n");
        printf("║      Median:  %8.1f ms  → %6.0f tok/s  (N=%d, CV=%.1f%%)      ║\n",
               r.prefill_ms.median(), r.prefill_tps.median(),
               r.prefill_ms.count(), r.prefill_ms.cv_pct());
        if (r.prefill_ms.count() > 1) {
            printf("║      Min:     %8.1f ms  Max: %.1f ms                        ║\n",
                   r.prefill_ms.min_val(), r.prefill_ms.max_val());
        }
        printf("║                                                                ║\n");
        printf("║  ▸ Generation Throughput (first → last token)                  ║\n");
        printf("║      Median:  %8.1f tok/s ±%.2f (N=%d, CV=%.1f%%)             ║\n",
               r.gen_tps.median(), r.gen_tps.ci95(), r.gen_tps.count(), r.gen_tps.cv_pct());
        if (r.gen_tps.count() > 1) {
            printf("║      Min:     %8.1f tok/s  Max: %.1f tok/s                   ║\n",
                   r.gen_tps.min_val(), r.gen_tps.max_val());
        }
        printf("║                                                                ║\n");
        printf("║  ▸ Total                                                       ║\n");
        printf("║      Tokens:  %8.0f (median per request)                    ║\n",
               r.total_tokens.median());
        printf("║      Time:    %8.0f ms (median per request)                 ║\n",
               r.total_ms.median());
        printf("║                                                                ║\n");
        printf("╚══════════════════════════════════════════════════════════════════╝\n");
    }

    // JSON 输出
    if (!cfg.json_output.empty()) {
        write_json(cfg.json_output, agg_results, cfg, bcfg);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    backend.stop();

    return 0;
}
