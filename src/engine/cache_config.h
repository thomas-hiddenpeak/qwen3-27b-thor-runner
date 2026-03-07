// KV Cache Offload — Configuration (SSD-only for Jetson AGX Thor)
//
// Jetson AGX Thor 使用统一内存 (128 GB LPDDR5X), CPU 和 GPU 共享同一物理内存.
// 因此不存在"offload 到 CPU 内存"的概念 — 唯一有意义的 offload 目标是 SSD.
//
// CacheEngine 的作用是前缀缓存 (Prefix Caching):
//   - 将已计算的 KV + SSM/Conv 状态序列化到 SSD
//   - 后续请求如果共享相同前缀 (如 system prompt), 直接从 SSD 恢复
//   - 跳过 prefill 计算, 节省 GPU 时间
//
// Inspired by LMCache (https://github.com/LMCache/LMCache)
#pragma once

#include <string>
#include <cstddef>
#include <cuda_bf16.h>
#include <cstdint>
#include <iostream>
#include <fstream>

namespace qwen_thor {
namespace cache {

// ---------------------------------------------------------------------------
// CacheConfig: 控制 KV Cache Offload (Prefix Caching) 行为
// ---------------------------------------------------------------------------
struct CacheConfig {
    // ---- 总开关 ----
    bool enabled = false;    // 默认关闭, 需要显式启用

    // ---- GPU KV Cache 内存预算 ----
    // KV Cache 在统一内存中的预算 (GB)
    // 决定 paged attention 可分配的 block 数量
    // 公式: num_blocks = budget / (block_size × num_layers × 2 × num_kv_heads × head_dim × 2)
    // 默认 4.0 GB → 4096 blocks → 65536 tokens
    double kv_cache_budget_gb = 4.0;

    // ---- Chunk 粒度 ----
    // Token chunk 大小, 用于前缀哈希匹配
    // 必须是 block_size 的倍数 (block_size=16), 推荐 256
    int chunk_size = 256;

    // ---- SSD 后端 ----
    std::string cache_dir          = "/tmp/qwen_kv_cache";
    size_t      max_cache_bytes    = 20ULL * 1024 * 1024 * 1024;  // 20 GB

    // ---- 状态缓存 ----
    // 是否缓存 SSM/Conv 状态 (Qwen3.5 混合注意力架构必须开启)
    // 每个前缀快照额外 ~147 MB (48 层 DeltaNet SSM + Conv state)
    bool cache_ssm_state = true;

    // ---- 驱逐策略 ----
    std::string eviction_policy = "lru";

    // ---- MTP 投机解码 ----
    // 使用模型自带的 Multi-Token Prediction 模块实现投机解码
    // mtp_mode: "auto" (模型有 MTP 权重则启用), "on" (强制开), "off" (强制关)
    std::string mtp_mode = "auto";
    // MTP KV Cache blocks 数量 (每 block 16 tokens, 默认 256 = 4096 tokens)
    int mtp_kv_blocks = 256;
    // 每步生成的 draft token 数量 (1~8, 默认 3 → 最多 4 tokens/step)
    int mtp_num_drafts = 3;

    // ---- 从 CLI 参数解析 ----
    // 支持的参数:
    //   --kv-cache-gb N         GPU KV Cache 内存预算 (GB, 默认 4.0)
    //   --cache-enable          启用 SSD 前缀缓存
    //   --cache-dir PATH        SSD 缓存目录
    //   --cache-max-gb N        SSD 最大缓存容量 (GB)
    //   --cache-chunk-size N    chunk 大小 (tokens)
    //   --cache-no-ssm          不缓存 SSM/Conv 状态
    //   --cache-config FILE     从配置文件加载
    //   --mtp-enable            强制启用 MTP 投机解码
    //   --mtp-disable           强制禁用 MTP 投机解码
    //   --mtp-kv-blocks N       MTP KV Cache blocks (默认 256)
    //   --mtp-drafts N          每步 draft token 数 (1~8, 默认 3)
    static CacheConfig from_args(int argc, char** argv) {
        CacheConfig cfg;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cache-enable") {
                cfg.enabled = true;
            } else if (arg == "--kv-cache-gb" && i + 1 < argc) {
                cfg.kv_cache_budget_gb = std::stod(argv[++i]);
            } else if (arg == "--cache-dir" && i + 1 < argc) {
                cfg.cache_dir = argv[++i];
                cfg.enabled = true;  // 指定目录隐含启用
            } else if (arg == "--cache-max-gb" && i + 1 < argc) {
                double gb = std::stod(argv[++i]);
                cfg.max_cache_bytes = (size_t)(gb * 1024 * 1024 * 1024);
            } else if (arg == "--cache-chunk-size" && i + 1 < argc) {
                cfg.chunk_size = std::stoi(argv[++i]);
            } else if (arg == "--cache-no-ssm") {
                cfg.cache_ssm_state = false;
            } else if (arg == "--cache-config" && i + 1 < argc) {
                cfg = CacheConfig::from_file(argv[++i]);
            } else if (arg == "--mtp-enable") {
                cfg.mtp_mode = "on";
            } else if (arg == "--mtp-disable") {
                cfg.mtp_mode = "off";
            } else if (arg == "--mtp-kv-blocks" && i + 1 < argc) {
                cfg.mtp_kv_blocks = std::stoi(argv[++i]);
            } else if (arg == "--mtp-drafts" && i + 1 < argc) {
                cfg.mtp_num_drafts = std::max(1, std::min(8, std::stoi(argv[++i])));
            }
        }
        cfg.validate();
        return cfg;
    }

    // ---- 参数校验 ----
    void validate() const {
        if (!enabled) return;
        if (chunk_size <= 0 || (chunk_size % 16) != 0) {
            std::cerr << "[CacheConfig] ERROR: chunk_size=" << chunk_size
                      << " must be a positive multiple of block_size(16)" << std::endl;
            std::exit(1);
        }
        if (kv_cache_budget_gb < 0.1) {
            std::cerr << "[CacheConfig] ERROR: kv_cache_budget_gb=" << kv_cache_budget_gb
                      << " too small (minimum 0.1)" << std::endl;
            std::exit(1);
        }
    }

    // ---- 从简单 key=value 配置文件加载 ----
    // 格式 (每行一个, # 开头为注释):
    //   enabled=true
    //   cache_dir=/mnt/ssd/kv_cache
    //   max_cache_gb=20
    //   chunk_size=256
    //   cache_ssm_state=true
    //   eviction_policy=lru
    static CacheConfig from_file(const std::string& path) {
        CacheConfig cfg;
        std::ifstream ifs(path);
        if (!ifs) {
            std::cerr << "[CacheConfig] Cannot open config file: " << path << std::endl;
            return cfg;
        }

        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            while (!key.empty() && key.back() == ' ') key.pop_back();
            while (!val.empty() && val.front() == ' ') val.erase(val.begin());

            if (key == "kv_cache_budget_gb")  cfg.kv_cache_budget_gb = std::stod(val);
            else if (key == "enabled")              cfg.enabled = (val == "true" || val == "1");
            else if (key == "cache_dir")       cfg.cache_dir = val;
            else if (key == "max_cache_gb")    cfg.max_cache_bytes = (size_t)(std::stod(val) * 1024*1024*1024);
            else if (key == "chunk_size")      cfg.chunk_size = std::stoi(val);
            else if (key == "cache_ssm_state") cfg.cache_ssm_state = (val == "true" || val == "1");
            else if (key == "eviction_policy") cfg.eviction_policy = val;
            else if (key == "mtp_mode")        cfg.mtp_mode = val;
            else if (key == "mtp_kv_blocks")   cfg.mtp_kv_blocks = std::stoi(val);
            else if (key == "mtp_num_drafts" || key == "mtp_drafts")
                cfg.mtp_num_drafts = std::max(1, std::min(8, std::stoi(val)));
        }
        cfg.validate();
        return cfg;
    }

    void print() const {
        std::cerr << "  KV Cache Budget: " << kv_cache_budget_gb << " GB" << std::endl;
        std::cerr << "  Prefix Cache:    " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
        if (!enabled) return;
        std::cerr << "  Cache Dir:       " << cache_dir << std::endl;
        std::cerr << "  SSD Budget:      " << (max_cache_bytes / (1024ULL*1024*1024)) << " GB" << std::endl;
        std::cerr << "  Chunk Size:      " << chunk_size << " tokens" << std::endl;
        std::cerr << "  SSM Caching:     " << (cache_ssm_state ? "ON" : "OFF") << std::endl;
        std::cerr << "  Eviction:        " << eviction_policy << std::endl;
        std::cerr << "  MTP Spec Decode: " << mtp_mode << std::endl;
        std::cerr << "  MTP KV Blocks:   " << mtp_kv_blocks << std::endl;
    }
};

// ---------------------------------------------------------------------------
// ModelCacheParams: 模型特定的缓存参数, 由 Qwen35Config 推导
// ---------------------------------------------------------------------------
struct ModelCacheParams {
    // Full Attention (GQA) 参数
    int num_full_attn_layers = 16;
    int num_kv_heads         = 4;
    int head_dim             = 256;
    int block_size           = 16;

    // Linear Attention (Gated DeltaNet) 参数
    int num_linear_attn_layers = 48;
    int nkh           = 16;
    int kd            = 128;
    int v_per_kh      = 384;
    int in_qkv        = 10240;
    int conv_k_minus_1 = 3;

    // KV 大小每 token 每层: 2(K+V) * 4 heads * 256 dim * 2 bytes = 4 KB
    size_t kv_bytes_per_token_per_layer() const {
        return 2ULL * num_kv_heads * head_dim * sizeof(uint16_t);
    }
    // 跨所有 full attn 层: 4 KB * 16 = 64 KB per token
    size_t kv_bytes_per_token() const {
        return kv_bytes_per_token_per_layer() * num_full_attn_layers;
    }
    // SSM 状态每层: 16*128*384*2 = 1.5 MB (BF16)
    size_t ssm_bytes_per_layer() const {
        return (size_t)nkh * kd * v_per_kh * sizeof(__nv_bfloat16);
    }
    size_t ssm_bytes_total() const {
        return ssm_bytes_per_layer() * num_linear_attn_layers;
    }
    // Conv 状态每层: 10240*3*2 = 60 KB
    size_t conv_bytes_per_layer() const {
        return (size_t)in_qkv * conv_k_minus_1 * sizeof(uint16_t);
    }
    size_t conv_bytes_total() const {
        return conv_bytes_per_layer() * num_linear_attn_layers;
    }
    // 单个条目总大小
    size_t entry_bytes(int num_tokens, bool include_ssm) const {
        size_t kv = (size_t)num_tokens * kv_bytes_per_token();
        size_t state = include_ssm ? (ssm_bytes_total() + conv_bytes_total()) : 0;
        return kv + state;
    }

    void print() const;
};

// ---------------------------------------------------------------------------
// CapacityPlanner: 启动时容量规划与报告
//
// 职责:
//   1. 从 kv_cache_budget_gb 计算 GPU 可容纳的 KV block 数 / token 数
//   2. 从 max_cache_bytes 计算 SSD 可换出的 token 数 (请求级 KV swap)
//   3. 计算每请求固定开销 (SSM/Conv 状态) → 最大并发数
//   4. 打印启动诊断报告
//
// 重要区别:
//   GPU KV blocks = 同时活跃的 KV (decode 时必须全部在内存)
//   SSD swap     = 空闲请求的 KV 换出 (不参与当前 decode 的请求)
//   单个请求的最大上下文 = GPU KV tokens (全部 block 须同时在内存)
//   多请求总容量        = GPU + SSD (通过请求级换出实现)
// ---------------------------------------------------------------------------
struct CapacityReport {
    // GPU 热 KV (在统一内存中, 可被 GPU 直接访问)
    int    gpu_kv_blocks         = 0;   // 实际分配的 block 数
    int    gpu_max_tokens        = 0;   // = blocks × block_size (单请求最大上下文)
    double gpu_kv_memory_gb      = 0;   // 实际占用 (GB)

    // SSD 换出容量 (请求级 KV swap, 非活跃请求的 KV 暂存)
    int    ssd_swap_tokens       = 0;   // 可换出到 SSD 的 token 数
    double ssd_budget_gb         = 0;

    // 多请求总容量 (GPU 活跃 + SSD 换出)
    int    total_multi_req_tokens = 0;

    // 每请求固定开销
    int    num_linear_layers     = 0;   // 线性注意力层数 (用于报告)
    double ssm_per_request_mb    = 0;   // SSM 状态
    double conv_per_request_mb   = 0;   // Conv 状态
    double total_per_request_mb  = 0;   // SSM + Conv

    // 估算最大并发 (假设剩余内存用于 SSM/Conv)
    int    estimated_max_batch   = 0;
    double available_for_ssm_gb  = 0;   // 总内存 - 模型 - KV - 系统
};

class CapacityPlanner {
public:
    // 从 CacheConfig + ModelCacheParams 计算容量
    static CapacityReport plan(const CacheConfig& config,
                               const ModelCacheParams& params,
                               double total_memory_gb = 128.0,
                               double model_weights_gb = 51.2,
                               double system_reserved_gb = 5.0) {
        CapacityReport r;

        // -- GPU KV cache --
        // 每 block 占用: block_size × 2(K+V) × num_kv_heads × head_dim × 2(bf16) × num_layers
        size_t bytes_per_block = (size_t)params.block_size * 2 * params.num_kv_heads
                                 * params.head_dim * sizeof(uint16_t) * params.num_full_attn_layers;
        size_t budget_bytes = (size_t)(config.kv_cache_budget_gb * 1024 * 1024 * 1024);
        r.gpu_kv_blocks    = (int)(budget_bytes / bytes_per_block);
        r.gpu_max_tokens   = r.gpu_kv_blocks * params.block_size;
        r.gpu_kv_memory_gb = (double)(r.gpu_kv_blocks * bytes_per_block) / (1024.0*1024*1024);

        // -- SSD swap --
        if (config.enabled) {
            r.ssd_budget_gb    = (double)config.max_cache_bytes / (1024.0*1024*1024);
            size_t kv_per_token = params.kv_bytes_per_token();
            r.ssd_swap_tokens  = (int)(config.max_cache_bytes / kv_per_token);
        }

        r.total_multi_req_tokens = r.gpu_max_tokens + r.ssd_swap_tokens;

        // -- 每请求 SSM/Conv 开销 --
        r.num_linear_layers    = params.num_linear_attn_layers;
        r.ssm_per_request_mb   = params.ssm_bytes_total() / (1024.0*1024);
        r.conv_per_request_mb  = params.conv_bytes_total() / (1024.0*1024);
        r.total_per_request_mb = r.ssm_per_request_mb + r.conv_per_request_mb;

        // -- 最大并发估算 --
        r.available_for_ssm_gb = total_memory_gb - model_weights_gb
                                 - r.gpu_kv_memory_gb - system_reserved_gb;
        if (r.available_for_ssm_gb < 0) r.available_for_ssm_gb = 0;
        double ssm_per_req_gb = r.total_per_request_mb / 1024.0;
        r.estimated_max_batch = (ssm_per_req_gb > 0)
            ? (int)(r.available_for_ssm_gb / ssm_per_req_gb) : 0;

        return r;
    }

    // 打印格式化容量报告
    static void print_report(const CapacityReport& r, const CacheConfig& config) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║              Capacity Planning Report                      ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║                                                            ║\n");
        fprintf(stderr, "║  ── GPU KV Cache (Active, In-Memory) ──                    ║\n");
        fprintf(stderr, "║    Budget:           %6.1f GB                              ║\n", config.kv_cache_budget_gb);
        fprintf(stderr, "║    Blocks:           %6d  (block_size=16)                 ║\n", r.gpu_kv_blocks);
        fprintf(stderr, "║    Max Tokens/Req:   %6d  (%.1f K, 单请求上限)            ║\n",
               r.gpu_max_tokens, r.gpu_max_tokens / 1024.0);
        fprintf(stderr, "║    Actual Memory:    %6.2f GB                              ║\n", r.gpu_kv_memory_gb);
        fprintf(stderr, "║                                                            ║\n");
        if (config.enabled) {
            fprintf(stderr, "║  ── SSD Swap (Idle Request Offload) ──                     ║\n");
            fprintf(stderr, "║    Budget:           %6.1f GB                              ║\n", r.ssd_budget_gb);
            fprintf(stderr, "║    Swap Capacity:    %6d tokens (%.1f K)                  ║\n",
                   r.ssd_swap_tokens, r.ssd_swap_tokens / 1024.0);
            fprintf(stderr, "║    Directory:        %-36s   ║\n", config.cache_dir.c_str());
            fprintf(stderr, "║                                                            ║\n");
            fprintf(stderr, "║  ── Multi-Request Total ──                                 ║\n");
            fprintf(stderr, "║    GPU (active):     %6d tokens                          ║\n", r.gpu_max_tokens);
            fprintf(stderr, "║    SSD (swapped):  + %6d tokens                          ║\n", r.ssd_swap_tokens);
            fprintf(stderr, "║                     ────────                               ║\n");
            fprintf(stderr, "║    Total:            %6d tokens  (%.1f K)                ║\n",
                   r.total_multi_req_tokens, r.total_multi_req_tokens / 1024.0);
            fprintf(stderr, "║    (单请求最大上下文仍为 %d tokens)                        ║\n", r.gpu_max_tokens);
        } else {
            fprintf(stderr, "║    Max Tokens:       %6d  (%.1f K, GPU only)             ║\n",
                   r.gpu_max_tokens, r.gpu_max_tokens / 1024.0);
        }
        fprintf(stderr, "║                                                            ║\n");
        fprintf(stderr, "║  ── Per-Request Fixed Cost (SSM/Conv) ──                   ║\n");
        fprintf(stderr, "║    SSM State:        %6.1f MB  (%d layers, BF16)           ║\n", r.ssm_per_request_mb, r.num_linear_layers);
        fprintf(stderr, "║    Conv State:       %6.1f MB  (%d layers, BF16)           ║\n", r.conv_per_request_mb, r.num_linear_layers);
        fprintf(stderr, "║    Total/Request:    %6.1f MB                              ║\n", r.total_per_request_mb);
        fprintf(stderr, "║                                                            ║\n");
        fprintf(stderr, "║  ── Concurrency Estimate ──                                ║\n");
        fprintf(stderr, "║    Memory for SSM:   %6.1f GB available                    ║\n", r.available_for_ssm_gb);
        fprintf(stderr, "║    Max Batch Size:   %6d requests (同时 decode)            ║\n", r.estimated_max_batch);
        fprintf(stderr, "║                                                            ║\n");
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════╝\n\n");
    }
};

} // namespace cache
} // namespace qwen_thor
