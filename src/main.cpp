// main.cpp — Qwen3.5-27B Thor 推理引擎统一入口
//
// 模式:
//   serve     — 启动 HTTP API 服务 (Ollama/OpenAI 兼容)
//   chat      — 启动 TUI 交互式对话
//   bench     — 运行性能基准测试
//   test      — 运行单元测试
//   version   — 打印版本信息
//
// 用法:
//   qwen3-27b-thor serve  [--config engine.conf] [--port 11434] [--host 0.0.0.0]
//   qwen3-27b-thor chat   [--config engine.conf] [--max-tokens 2048]
//   qwen3-27b-thor bench  [--warmup 5] [--decode 30] [--prompt-len 64]
//   qwen3-27b-thor test
//   qwen3-27b-thor --help

#include "engine/backend.h"
#include "serve/serve.h"
#include "tui/tui.h"
#include <iostream>
#include <cstring>
#include <csignal>
#include <thread>
#include <chrono>

static const char* VERSION = "0.2.0";
static const char* BUILD_DATE = __DATE__;
static const char* AUTHOR = "Thomas";

// ============================================================================
// Help 文本
// ============================================================================

static void print_usage() {
    printf("\n");
    printf("  Qwen3.5-27B Thor Inference Engine  v%s  (%s)\n", VERSION, BUILD_DATE);
    printf("  Author: %s\n", AUTHOR);
    printf("  NVIDIA Jetson AGX Thor • SM110a Blackwell • 128GB LPDDR5X • BF16\n\n");
    printf("  Usage:\n");
    printf("    qwen3-27b-thor <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    serve       Start HTTP API server (Ollama/OpenAI compatible)\n");
    printf("    chat        Start interactive TUI chat\n");
    printf("    bench       Run inference benchmarks\n");
    printf("    test        Run unit tests\n");
    printf("    version     Print version information\n\n");
    printf("  Engine Options (shared by serve/chat/bench):\n");
    printf("    --config <file>       Load unified configuration from file (engine + serve)\n");
    printf("    --model-dir <path>    Model weights directory\n");
    printf("    --kv-cache-gb <N>     GPU KV cache budget in GB (default: 4.0)\n");
    printf("    --cache-enable        Enable SSD prefix caching\n");
    printf("    --cache-dir <path>    SSD cache directory\n");
    printf("    --cache-max-gb <N>    Max SSD cache size in GB\n");
    printf("    --cache-chunk-size <N> Prefix cache chunk size (tokens)\n");
    printf("    --cache-no-ssm        Disable SSM/Conv state caching\n");
    printf("    --mtp-enable          Force enable MTP speculative decoding\n");
    printf("    --mtp-disable         Force disable MTP speculative decoding\n\n");
    printf("  Serve Options:\n");
    printf("    --host <addr>         Listen address (default: 0.0.0.0)\n");
    printf("    --ollama-port <N>     Ollama API port (default: 11434)\n");
    printf("    --openai-port <N>     OpenAI API port (default: 8080)\n");
    printf("    --port <N>            Alias for --ollama-port\n");
    printf("    --max-conns <N>       Max concurrent connections (default: 64)\n");
    printf("    --model-name <name>   Model display name (default: qwen3.5-27b)\n");
    printf("    --serve-config <file> Override serve config from separate file\n\n");
    printf("  Chat Options:\n");
    printf("    --max-tokens <N>      Max new tokens per response (default: 2048)\n");
    printf("    --temperature <F>     Sampling temperature (default: 1.0)\n");
    printf("    --top-p <F>           Nucleus sampling threshold (default: 0.95)\n");
    printf("    --no-stats            Disable performance statistics display\n\n");
    printf("  Bench Options:\n");
    printf("    --warmup <N>          Warmup decode steps (default: 5)\n");
    printf("    --decode <N>          Measured decode steps (default: 30)\n");
    printf("    --prompt-len <N>      Prompt length in tokens (default: 64)\n");
    printf("    --batch <N>           Batch size (default: 1)\n");
    printf("    --csv                 Output in CSV format\n");
    printf("    --nsys                Enable NVTX annotations for nsys profiling\n\n");
    printf("  Examples:\n");
    printf("    # Start API server with 8GB KV cache + SSD caching\n");
    printf("    qwen3-27b-thor serve --kv-cache-gb 8 --cache-enable\n\n");
    printf("    # Interactive chat with default settings\n");
    printf("    qwen3-27b-thor chat --kv-cache-gb 4\n\n");
    printf("    # Load from unified config file\n");
    printf("    qwen3-27b-thor serve --config configs/config.conf\n\n");
    printf("    # Run benchmarks\n");
    printf("    qwen3-27b-thor bench --decode 50 --prompt-len 512 --batch 4\n\n");
}

static void print_version() {
    printf("qwen3-27b-thor v%s (%s)\n", VERSION, BUILD_DATE);
    printf("  Author:  %s\n", AUTHOR);
    printf("  Device:  NVIDIA Jetson AGX Thor, 128GB LPDDR5X\n");
    printf("  Target:  SM110a Blackwell, 20 SMs, 5th-gen Tensor Cores\n");
    printf("  Model:   Qwen3.5-27B (64L, 48 DeltaNet + 16 GQA)\n");
    printf("  Precision: BF16\n");

    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("  CUDA:    Driver %d.%d, Runtime %d.%d\n",
           driver_version / 1000, (driver_version % 100) / 10,
           runtime_version / 1000, (runtime_version % 100) / 10);
}

// ============================================================================
// Signal handler
// ============================================================================

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int sig) {
    printf("\n[Signal %d] Shutting down...\n", sig);
    g_shutdown = true;
}

// ============================================================================
// 命令: serve
// ============================================================================

static int cmd_serve(int argc, char** argv) {
    using namespace qwen_thor;

    auto backend_config = BackendConfig::from_args(argc, argv);

    // --config 同时作为 engine 和 serve 的统一配置文件
    // --serve-config 可以覆盖 serve 相关配置
    serve::ServeConfig serve_config;
    std::string config_file;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc)
            config_file = argv[i + 1];
    }
    if (!config_file.empty()) {
        serve_config = serve::ServeConfig::from_file(config_file);
    }
    // --serve-config 覆盖
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--serve-config" && i + 1 < argc) {
            serve_config = serve::ServeConfig::from_file(argv[i + 1]);
        }
    }
    // CLI 参数最终覆盖
    serve_config = serve::ServeConfig::merge_args(serve_config, argc, argv);

    try {
        InferenceBackend backend(backend_config);
        serve::ServeApp app(serve_config, backend);

        // 信号处理
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        backend.start();

        // 在后台线程中监控 shutdown
        std::thread monitor([&]() {
            while (!g_shutdown) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            app.stop();
            backend.stop();
        });

        app.run();  // 阻塞

        if (monitor.joinable()) monitor.join();
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// ============================================================================
// 命令: chat
// ============================================================================

static int cmd_chat(int argc, char** argv) {
    using namespace qwen_thor;

    auto backend_config = BackendConfig::from_args(argc, argv);
    auto tui_config = tui::TuiConfig::from_args(argc, argv);

    try {
        InferenceBackend backend(backend_config);
        tui::ChatApp app(tui_config, backend);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// ============================================================================
// 命令: bench (调用原始 benchmark 逻辑)
// ============================================================================

// 前置声明 — benchmark.cpp 中定义
int run_benchmark(int argc, char** argv);

static int cmd_bench(int argc, char** argv) {
    return run_benchmark(argc, argv);
}

// ============================================================================
// 命令: test (调用原始测试逻辑)
// ============================================================================

// 前置声明 — tests.cpp 中定义
int run_tests(int argc, char** argv);

static int cmd_test(int argc, char** argv) {
    return run_tests(argc, argv);
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];

    if (cmd == "serve")   return cmd_serve(argc, argv);
    if (cmd == "chat")    return cmd_chat(argc, argv);
    if (cmd == "bench")   return cmd_bench(argc, argv);
    if (cmd == "test")    return cmd_test(argc, argv);
    if (cmd == "version" || cmd == "--version" || cmd == "-v") {
        print_version();
        return 0;
    }
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
        return 0;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    std::cerr << "Run 'qwen3-27b-thor --help' for usage.\n";
    return 1;
}
