// backend.cpp — 独立后端引擎实现
//
// InferenceBackend 封装 InferenceEngine, 提供线程安全的请求提交/结果轮询接口。
// 与传输层 (IPC/HTTP/TUI) 完全解耦。

#include "backend.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <malloc.h>   // malloc_trim

namespace qwen_thor {

static std::string default_model_dir_from_env() {
    const char* env_model_dir = std::getenv("QWEN_MODEL_DIR");
    if (env_model_dir && env_model_dir[0] != '\0') {
        return std::string(env_model_dir);
    }
    return "/home/rm01/models/dev/llm/Qwen/Qwen3.5-27B";
}

// ============================================================================
// BackendConfig
// ============================================================================

BackendConfig BackendConfig::from_args(int argc, char** argv) {
    BackendConfig cfg;
    cfg.model_dir = default_model_dir_from_env();
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            cfg.model_dir = argv[++i];
        } else if (arg == "--kv-cache-gb" && i + 1 < argc) {
            cfg.kv_cache_gb = std::stod(argv[++i]);
        } else if (arg == "--cache-enable") {
            cfg.cache_enabled = true;
        } else if (arg == "--cache-dir" && i + 1 < argc) {
            cfg.cache_dir = argv[++i];
            cfg.cache_enabled = true;
        } else if (arg == "--cache-max-gb" && i + 1 < argc) {
            cfg.cache_max_gb = std::stod(argv[++i]);
        } else if (arg == "--cache-chunk-size" && i + 1 < argc) {
            cfg.cache_chunk_size = std::stoi(argv[++i]);
        } else if (arg == "--cache-no-ssm") {
            cfg.cache_ssm_state = false;
        } else if (arg == "--mtp-enable") {
            cfg.mtp_mode = "on";
        } else if (arg == "--mtp-disable") {
            cfg.mtp_mode = "off";
        } else if (arg == "--mtp-kv-blocks" && i + 1 < argc) {
            cfg.mtp_kv_blocks = std::stoi(argv[++i]);
        } else if (arg == "--mtp-drafts" && i + 1 < argc) {
            cfg.mtp_num_drafts = std::max(1, std::min(8, std::stoi(argv[++i])));
        } else if (arg == "--config" && i + 1 < argc) {
            cfg = BackendConfig::from_file(argv[++i]);
        }
    }
    return cfg;
}

BackendConfig BackendConfig::from_file(const std::string& path) {
    BackendConfig cfg;
    cfg.model_dir = default_model_dir_from_env();
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[BackendConfig] Cannot open config file: " << path << std::endl;
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

        if      (key == "model_dir")          cfg.model_dir = val;
        else if (key == "kv_cache_budget_gb") cfg.kv_cache_gb = std::stod(val);
        else if (key == "enabled" || key == "cache_enabled")  cfg.cache_enabled = (val == "true" || val == "1");
        else if (key == "cache_dir")          cfg.cache_dir = val;
        else if (key == "max_cache_gb")       cfg.cache_max_gb = std::stod(val);
        else if (key == "chunk_size" || key == "cache_chunk_size")  cfg.cache_chunk_size = std::stoi(val);
        else if (key == "cache_ssm_state")    cfg.cache_ssm_state = (val == "true" || val == "1");
        else if (key == "eviction_policy")    cfg.eviction_policy = val;
        else if (key == "mtp_mode")           cfg.mtp_mode = val;
        else if (key == "mtp_kv_blocks")      cfg.mtp_kv_blocks = std::stoi(val);
        else if (key == "mtp_num_drafts" || key == "mtp_drafts")
            cfg.mtp_num_drafts = std::max(1, std::min(8, std::stoi(val)));
        // Serve keys are silently ignored here (parsed by ServeConfig::from_file)
    }
    return cfg;
}

cache::CacheConfig BackendConfig::to_cache_config() const {
    cache::CacheConfig cc;
    cc.kv_cache_budget_gb = kv_cache_gb;
    cc.enabled            = cache_enabled;
    cc.cache_dir          = cache_dir;
    cc.max_cache_bytes    = (size_t)(cache_max_gb * 1024 * 1024 * 1024);
    cc.chunk_size         = cache_chunk_size;
    cc.cache_ssm_state    = cache_ssm_state;
    cc.eviction_policy    = eviction_policy;
    cc.mtp_mode           = mtp_mode;
    cc.mtp_kv_blocks      = mtp_kv_blocks;
    cc.mtp_num_drafts     = mtp_num_drafts;
    cc.validate();
    return cc;
}

void BackendConfig::print() const {
    printf("\n");
    printf("┌─────────────────────────────────────────────┐\n");
    printf("│          Backend Configuration              │\n");
    printf("├─────────────────────────────────────────────┤\n");
    printf("│  Model Dir:     %-26s │\n", model_dir.c_str());
    printf("│  KV Cache:      %-6.1f GB                   │\n", kv_cache_gb);
    printf("│  Prefix Cache:  %-8s                    │\n", cache_enabled ? "ENABLED" : "DISABLED");
    if (cache_enabled) {
        printf("│  Cache Dir:     %-26s │\n", cache_dir.c_str());
        printf("│  SSD Budget:    %-6.1f GB                   │\n", cache_max_gb);
        printf("│  Chunk Size:    %-6d tokens               │\n", cache_chunk_size);
        printf("│  SSM Caching:   %-3s                        │\n", cache_ssm_state ? "ON" : "OFF");
    }
    printf("│  MTP Mode:      %-8s                    │\n", mtp_mode.c_str());
    printf("│  MTP Drafts:    %-6d                      │\n", mtp_num_drafts);
    printf("└─────────────────────────────────────────────┘\n\n");
}

// ============================================================================
// InferenceBackend
// ============================================================================

// ============================================================================
// InferenceBackend
// ============================================================================

// 启动前内存回收 + 预检
// 1. 回收 C heap fragmentation (malloc_trim)
// 2. 初始化 CUDA context → 驱动会回收已死进程的 managed memory
// 3. 设置 cudaDeviceScheduleBlockingSync (必须在 context 首次初始化时)
// 4. 估算总需求并与可用内存比较
static void check_memory_budget(const BackendConfig& config, const core::Qwen35Config& model_cfg) {
    namespace fs = std::filesystem;

    // ---- 0. 自动内存回收 ----
    // 设备标志必须在 CUDA context 初始化前调用
    // BlockingSync: cudaStreamSynchronize 时 CPU 让出而非 spin-wait
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // cudaFree(0) 强制初始化 CUDA context/driver
    // 这会触发 driver 回收被 kill -9 掉的旧进程的 managed memory
    cudaFree(0);
    cudaDeviceSynchronize();

    // 回收 C heap 碎片到 OS — 启动时安全 (推理循环中不安全, 见 WDT 说明)
    // malloc_trim(0); // DISABLED: glibc arena lock 可能阻塞 systemd watchdog 喂狗

    // ---- 1. 读取系统内存 ----
    // /proc/meminfo: MemTotal (物理总量), MemAvailable (当前可用)
    // cudaMemGetInfo: CUDA 视角的 free/total (Jetson 上 = 物理内存 - driver 保留)
    size_t mem_total_kb = 0, mem_available_kb = 0;
    {
        std::ifstream mi("/proc/meminfo");
        std::string line;
        while (std::getline(mi, line)) {
            if (line.rfind("MemTotal:", 0) == 0)
                sscanf(line.c_str(), "MemTotal: %zu kB", &mem_total_kb);
            else if (line.rfind("MemAvailable:", 0) == 0)
                sscanf(line.c_str(), "MemAvailable: %zu kB", &mem_available_kb);
        }
    }
    if (mem_total_kb == 0) {
        std::cerr << "[MemCheck] Warning: cannot read /proc/meminfo, skipping memory check\n";
        return;
    }

    // CUDA memory info (Jetson 统一内存: free+used = total ≈ MemTotal)
    size_t cuda_free = 0, cuda_total = 0;
    if (cudaMemGetInfo(&cuda_free, &cuda_total) != cudaSuccess) {
        cuda_free = cuda_total = 0;  // fallback: 仅使用 /proc/meminfo
    }

    double mem_total_gb      = mem_total_kb / 1048576.0;
    double mem_available_gb  = mem_available_kb / 1048576.0;
    double cuda_free_gb      = cuda_free / (1024.0 * 1024 * 1024);
    double cuda_total_gb     = cuda_total / (1024.0 * 1024 * 1024);

    // Jetson 上使用 CUDA free 更准确 (包含可回收的 managed memory)
    // 如果 CUDA free 可用且 > MemAvailable, 说明有可回收的 managed memory
    double effective_available_gb = mem_available_gb;
    if (cuda_free_gb > 0) {
        effective_available_gb = std::max(mem_available_gb, cuda_free_gb);
    }

    // ---- 2. 估算模型权重内存 (扫描 safetensors 文件大小) ----
    size_t weight_bytes = 0;
    if (fs::exists(config.model_dir) && fs::is_directory(config.model_dir)) {
        for (const auto& entry : fs::directory_iterator(config.model_dir)) {
            if (entry.path().extension() == ".safetensors")
                weight_bytes += entry.file_size();
        }
    }
    // Safetensors 文件大小 ≈ GPU 权重内存 (BF16 权重直接映射)
    // 少量 FP32 A_log 张量会额外多占 ~1% (可忽略)
    double weight_gb = weight_bytes / (1024.0 * 1024 * 1024);

    // ---- 3. KV Cache ----
    double kv_gb = config.kv_cache_gb;

    // ---- 4. Workspace (层间激活缓冲) ----
    // max_tokens=8192, ws_per_tok ≈ max(ws_full, ws_linear) × sizeof(bf16)
    const int hs = model_cfg.hidden_size;
    const int is = model_cfg.intermediate_size;
    const int qp_dim = model_cfg.q_proj_dim();
    const int q_dim  = model_cfg.q_dim();
    const int kv_dim = model_cfg.kv_dim();
    const int qk     = model_cfg.lin_qk_dim();
    const int lin_v  = model_cfg.lin_v_dim();
    const int nkh    = model_cfg.linear_num_key_heads;
    size_t ws_full   = (size_t)(4*hs + qp_dim + q_dim + 2*kv_dim + 3*is);
    size_t ws_linear = (size_t)(hs + 2*qk + lin_v + lin_v + nkh + qk + lin_v + hs + hs + 3*is + hs + nkh*2);
    size_t ws_per_tok = std::max(ws_full, ws_linear);
    double workspace_gb = (double)(ws_per_tok * 8192 * 2) / (1024.0 * 1024 * 1024);

    // d_hidden_states: max_tokens × hidden_size × 2
    workspace_gb += (double)(8192 * hs * 2) / (1024.0 * 1024 * 1024);

    // ---- 5. Vision encoder workspace ----
    double vision_gb = 0;
    // max_patches = max_pixels / patch_size²
    // 这里用默认 VisionConfig 估算 (与 engine.cpp 初始化逻辑一致)
    {
        core::VisionConfig vcfg;  // default: max_pixels=1048576, patch_size=14
        int max_patches = vcfg.max_pixels / (vcfg.patch_size * vcfg.patch_size);
        // workspace_bytes 需要精确计算, 这里用已知的经验公式估算
        // attn scores 是最大项: num_heads × N × N × 4 bytes
        // N=max_patches 时 scores = 16 × 5376 × 5376 × 4 = 1.8GB (远超实际)
        // 实际 max_patches=5376, workspace_bytes ≈ 1.2GB (由 vision.cu 计算)
        // 我们用保守估算: N² × 64 + N × 14000 (bytes)
        size_t N = max_patches;
        size_t vision_bytes = N * N * 64 + N * 14000 + (size_t)N * vcfg.patch_input_dim() * 4;
        vision_gb = (double)vision_bytes / (1024.0 * 1024 * 1024);
    }

    // ---- 6. SSM/Conv 状态 (单请求) ----
    int num_linear_layers = model_cfg.num_hidden_layers - model_cfg.num_full_attn_layers();
    size_t ssm_per_layer  = (size_t)nkh * model_cfg.linear_key_head_dim
                            * model_cfg.lin_v_per_kh() * sizeof(__nv_bfloat16);
    double ssm_gb = (double)(num_linear_layers * ssm_per_layer) / (1024.0 * 1024 * 1024);

    // ---- 7. OS/系统开销预留 ----
    double os_reserve_gb = 2.0;

    // ---- 合计 ----
    double total_gb = weight_gb + kv_gb + workspace_gb + vision_gb + ssm_gb + os_reserve_gb;

    // ---- 打印内存预算报告 ----
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                 Memory Budget Check                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  System Memory:    %6.1f GB total, %6.1f GB available      ║\n",
           mem_total_gb, mem_available_gb);
    if (cuda_total_gb > 0) {
        printf("║  CUDA Memory:     %6.1f GB total, %6.1f GB free          ║\n",
               cuda_total_gb, cuda_free_gb);
    }
    printf("║                                                            ║\n");
    printf("║  Estimated Requirements:                                   ║\n");
    printf("║    Model Weights:  %6.1f GB  (%d safetensors files)        ║\n",
           weight_gb, (int)std::count_if(
               fs::directory_iterator(config.model_dir),
               fs::directory_iterator{},
               [](const auto& e) { return e.path().extension() == ".safetensors"; }));
    printf("║    KV Cache:       %6.1f GB  (--kv-cache-gb)               ║\n", kv_gb);
    printf("║    Workspace:      %6.1f GB  (activations)                 ║\n", workspace_gb);
    printf("║    Vision:         %6.1f GB  (ViT encoder)                 ║\n", vision_gb);
    printf("║    SSM State:      %6.2f GB  (DeltaNet, 1 req)             ║\n", ssm_gb);
    printf("║    OS Reserve:     %6.1f GB                                ║\n", os_reserve_gb);
    printf("║    ─────────────────────────────────                       ║\n");
    printf("║    Total:          %6.1f GB                                ║\n", total_gb);
    printf("║                                                            ║\n");

    if (total_gb > effective_available_gb) {
        double deficit = total_gb - effective_available_gb;
        printf("║  ⚠  WARNING: Estimated %.1f GB exceeds available %.1f GB   ║\n",
               total_gb, effective_available_gb);
        printf("║     Deficit: %.1f GB — risk of OOM!                       ║\n", deficit);
        printf("║                                                            ║\n");
        printf("║  Suggestions:                                              ║\n");
        if (kv_gb > 4.0) {
            printf("║    • Reduce --kv-cache-gb (current: %.1f)                  ║\n", kv_gb);
        }
        printf("║    • Free system memory (kill other processes)             ║\n");
        printf("║    • Check: free -h                                        ║\n");
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");

        // 如果超过物理总内存 (无论如何都不可能启动), 直接终止
        if (total_gb > mem_total_gb) {
            fprintf(stderr, "\n[MemCheck] FATAL: Memory requirement (%.1f GB) exceeds "
                    "total physical memory (%.1f GB).\n"
                    "           Reduce --kv-cache-gb or use a smaller model.\n\n",
                    total_gb, mem_total_gb);
            exit(1);
        }
        fprintf(stderr, "[MemCheck] Proceeding with warning — OOM may occur during inference.\n\n");
    } else {
        double headroom = effective_available_gb - total_gb;
        printf("║  ✓  Memory OK: %.1f GB headroom (%.0f concurrent reqs)     ║\n",
               headroom, std::max(1.0, headroom / (ssm_gb > 0 ? ssm_gb : 0.15)));
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    }
}

InferenceBackend::InferenceBackend(const BackendConfig& config)
    : config_(config) {
    config_.print();
    auto cache_config = config_.to_cache_config();
    cache_config.print();

    // 启动前内存预检
    check_memory_budget(config_, model_config_);

    // 加载 tokenizer
    if (!tokenizer_.load(config_.model_dir)) {
        std::cerr << "[Backend] Warning: tokenizer load failed from " << config_.model_dir << std::endl;
    }

    engine_ = std::make_unique<core::InferenceEngine>(model_config_, config_.model_dir, cache_config);
}

InferenceBackend::~InferenceBackend() {
    stop();
}

void InferenceBackend::start() {
    if (running_.exchange(true)) return;  // 已在运行
    engine_->start();
    std::cout << "[Backend] Inference engine started." << std::endl;
}

void InferenceBackend::stop() {
    if (!running_.exchange(false)) return;
    engine_->stop();
    std::cout << "[Backend] Inference engine stopped." << std::endl;
}

bool InferenceBackend::submit(InferRequest& request) {
    // 构建 IPC InferenceRequest — 堆分配避免 1MB 栈分配
    auto ipc_req_ptr = std::make_unique<ipc::InferenceRequest>();
    auto& ipc_req = *ipc_req_ptr;
    memset(&ipc_req, 0, sizeof(ipc_req));  // 零初始化
    ipc_req.request_id     = request.request_id;
    ipc_req.prompt_len     = static_cast<int32_t>(request.prompt_tokens.size());
    ipc_req.max_new_tokens = request.max_new_tokens;
    ipc_req.temperature    = request.temperature;
    ipc_req.top_p          = request.top_p;
    ipc_req.top_k          = request.top_k;
    ipc_req.min_p          = request.min_p;
    ipc_req.repeat_penalty = request.repeat_penalty;
    ipc_req.frequency_penalty = request.frequency_penalty;
    ipc_req.presence_penalty  = request.presence_penalty;
    ipc_req.seed           = request.seed;
    ipc_req.stream         = request.stream;

    int copy_len = std::min(static_cast<int>(request.prompt_tokens.size()),
                            static_cast<int>(ipc::MAX_PROMPT_LEN));
    std::memcpy(ipc_req.prompt_tokens, request.prompt_tokens.data(), copy_len * sizeof(int32_t));

    // 多模态: 预处理图像和视频并附加到 engine
    {
        core::VisionConfig vcfg;  // default config
        std::vector<core::ProcessedImage> all_processed;

        // 处理图像 — move pixels 避免 36MB 复制
        for (auto& img : request.images) {
            core::ImageInput input;
            input.pixels = std::move(img.pixels);  // move, 不复制
            input.width  = img.width;
            input.height = img.height;
            all_processed.push_back(core::VisionEncoder::preprocess_image(input, vcfg));
            auto& p = all_processed.back();
            std::cout << "[Backend] Image preprocessed: grid=" << p.grid_t << "x"
                      << p.grid_h << "x" << p.grid_w
                      << " patches=" << p.num_patches()
                      << " output_tokens=" << p.num_output_tokens()
                      << " is_video=" << p.is_video << std::endl;
        }

        // 处理视频 (帧序列)
        std::cout << "[Backend] Videos: " << request.videos.size()
                  << ", Images: " << request.images.size() << std::endl;
        for (auto& vid : request.videos) {
            std::cout << "[Backend] Video: " << vid.frames.size() << " frames, "
                      << vid.width << "x" << vid.height
                      << ", fps=" << vid.source_fps << std::endl;
            core::VideoInput input;
            input.frames     = vid.frames;
            input.width      = vid.width;
            input.height     = vid.height;
            input.source_fps = vid.source_fps;
            all_processed.push_back(core::VisionEncoder::preprocess_video(input, vcfg));
            auto& p = all_processed.back();
            std::cout << "[Backend] Video preprocessed: grid=" << p.grid_t << "x"
                      << p.grid_h << "x" << p.grid_w
                      << " patches=" << p.num_patches()
                      << " output_tokens=" << p.num_output_tokens()
                      << " is_video=" << p.is_video
                      << " pixel_values_bf16=" << p.pixel_values_bf16.size() << std::endl;
        }

        if (!all_processed.empty()) {
            std::cout << "[Backend] Attaching " << all_processed.size()
                      << " vision items to request " << request.request_id << std::endl;
            std::cout.flush();
            engine_->attach_images(request.request_id, std::move(all_processed));
        }
    }

    // 通过 engine 的直接推送接口提交
    return engine_->push_request(ipc_req);  // copies into ring buffer, ipc_req_ptr freed on return
}

bool InferenceBackend::poll(InferResponse& response) {
    // 从 engine 的响应队列直接轮询
    ipc::InferenceResponse ipc_resp{};
    if (!engine_->pop_response(ipc_resp)) return false;

    response.request_id  = ipc_resp.request_id;
    response.token_id    = ipc_resp.token_id;
    response.is_finished = ipc_resp.is_finished;
    response.error_code  = ipc_resp.error_code;
    return true;
}

void InferenceBackend::cancel(uint64_t request_id) {
    // 直接委托给 engine, 在 inference_loop 中检查并取消
    engine_->cancel_request(request_id);
}

int InferenceBackend::active_request_count() const {
    return 0;  // TODO: expose from engine
}

} // namespace qwen_thor
