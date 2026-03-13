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
//   qwen35-thor serve  [--config engine.conf] [--port 11434] [--host 0.0.0.0]
//   qwen35-thor chat   [--config engine.conf] [--max-tokens 2048]
//   qwen35-thor bench  [--warmup 5] [--decode 30] [--prompt-len 64]
//   qwen35-thor test
//   qwen35-thor --help

#include "engine/backend.h"
#include "serve/serve.h"
#include "tui/tui.h"
#include "plugins/asr/asr_plugin.h"
#include "plugins/asr/asr_engine.h"
#include "plugins/tts/tts_plugin.h"
#include "plugins/tts/tts_engine.h"

// SM110a hardware probe (Level 0)
namespace sm110a_probe { void run_sm110a_probes(); }
#include <iostream>
#include <cstring>
#include <csignal>
#include <thread>
#include <chrono>
#include <execinfo.h>
#include <unistd.h>

// Crash signal handler — writes directly to fd 2 (unbuffered)
static void crash_handler(int sig) {
    const char* msg = "\n[CRASH] Signal: ";
    ssize_t r_ __attribute__((unused));
    r_ = write(STDERR_FILENO, msg, strlen(msg));
    char num[16];
    int len = snprintf(num, sizeof(num), "%d\n", sig);
    r_ = write(STDERR_FILENO, num, len);

    // Print backtrace
    void* frames[32];
    int n = backtrace(frames, 32);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);

    // Re-raise to get default behavior (core dump)
    signal(sig, SIG_DFL);
    raise(sig);
}

static const char* VERSION = "2.0.0";
static const char* BUILD_DATE = __DATE__;
static const char* AUTHOR = "Thomas";

// ============================================================================
// Help 文本
// ============================================================================

static void print_usage() {
    printf("\n");
    printf("  Qwen3.5-Thor Inference Engine  v%s  (%s)\n", VERSION, BUILD_DATE);
    printf("  Author: %s\n", AUTHOR);
    printf("  NVIDIA Jetson AGX Thor • SM110a Blackwell • 128GB LPDDR5X • BF16\n\n");
    printf("  Usage:\n");
    printf("    qwen35-thor <command> [options]\n\n");
    printf("  Commands:\n");
    printf("    serve       Start HTTP API server (Ollama/OpenAI compatible)\n");
    printf("    chat        Start interactive TUI chat\n");
    printf("    bench       Run inference benchmarks\n");
    printf("    asr         Transcribe audio file (native Qwen3-ASR)\n");
    printf("    tts         Synthesize speech (native Qwen3-TTS)\n");
    printf("    test        Run unit tests\n");
    printf("    probe       SM110a hardware primitives micro-benchmark\n");
    printf("    version     Print version information\n\n");
    printf("  Engine Options (shared by serve/chat/bench):\n");
    printf("    --config <file>       Load unified configuration from file (REQUIRED*)\n");
    printf("    --model-dir <path>    Model weights directory (REQUIRED*)\n");
    printf("                          * Either --config or --model-dir must be specified,\n");
    printf("                            or set QWEN_MODEL_DIR environment variable.\n");
    printf("    --kv-cache-gb <N>     GPU KV cache budget in GB (default: 4.0)\n");
    printf("    --cache-enable        Enable SSD prefix caching\n");
    printf("    --cache-dir <path>    SSD cache directory\n");
    printf("    --cache-max-gb <N>    Max SSD cache size in GB\n");
    printf("    --cache-chunk-size <N> Prefix cache chunk size (tokens)\n");
    printf("    --cache-no-ssm        Disable SSM/Conv state caching\n");
    printf("    --max-chunk-size <N>  Prefill chunk size (64-4096, default: 2048)\n");
    printf("    --mtp-enable          Force enable MTP speculative decoding\n");
    printf("    --mtp-disable         Force disable MTP speculative decoding\n");
    printf("    --mtp-drafts <N>      Draft tokens per step (1-8, default: 1)\n\n");
    printf("  Serve Options:\n");
    printf("    --host <addr>         Listen address (default: 0.0.0.0)\n");
    printf("    --ollama-port <N>     Ollama API port (default: 11434)\n");
    printf("    --openai-port <N>     OpenAI API port (default: 8080)\n");
    printf("    --port <N>            Alias for --ollama-port\n");
    printf("    --max-conns <N>       Max concurrent connections (default: 64)\n");
    printf("    --model-name <name>   Model display name (default: qwen3.5-27b)\n");
    printf("    --serve-config <file> Override serve config from separate file\n\n");
    printf("  Plugin Options (ASR / TTS):\n");
    printf("    --asr-enabled         Enable ASR (speech-to-text) plugin\n");
    printf("    --asr-executable <p>  Path to ASR executable (e.g., whisper-cli)\n");
    printf("    --asr-model <path>    ASR model path\n");
    printf("    --asr-language <lang> Default ASR language (auto/zh/en/ja, default: auto)\n");
    printf("    --tts-enabled         Enable TTS (text-to-speech) plugin\n");
    printf("    --tts-executable <p>  Path to TTS executable (e.g., piper)\n");
    printf("    --tts-model <path>    TTS model path\n");
    printf("    --tts-voice <name>    Default TTS voice name\n");
    printf("    --tts-instruct <text> VoiceDesign instruct prompt\n\n");
    printf("  Chat Options:\n");
    printf("    --max-tokens <N>      Max new tokens per response (default: 2048)\n");
    printf("    --temperature <F>     Sampling temperature (default: 1.0)\n");
    printf("    --top-p <F>           Nucleus sampling threshold (default: 0.95)\n");
    printf("    --no-stats            Disable performance statistics display\n\n");
    printf("  Bench Options:\n");
    printf("    --warmup <N>          Warmup decode steps (default: 5)\n");
    printf("    --decode <N>          Measured decode steps (default: 50)\n");
    printf("    --prompt-len <N[,N]>  Prompt length(s), comma-separated (default: 17)\n");
    printf("    --batch <N[,N]>       Batch size(s), comma-separated (default: 1)\n");
    printf("    --iterations <N>      Independent iterations per config (default: 1)\n");
    printf("    --prefill-repeat <N>  Prefill repeats per iteration (default: 3)\n");
    printf("    --json <FILE>         Output structured JSON results\n");
    printf("    --csv                 Output in CSV format\n");
    printf("    --per-step            Print per-step timing details\n");
    printf("    --no-graph            Disable CUDA Graph for per-phase timing\n");
    printf("    --nsys                Enable NVTX annotations for nsys profiling\n\n");
    printf("  Examples:\n");
    printf("    # Start API server with config file\n");
    printf("    qwen35-thor serve --config configs/qwen3.5-27b.conf\n\n");
    printf("    # Start API server with model dir + options\n");
    printf("    qwen35-thor serve --model-dir /path/to/Qwen3.5-27B --kv-cache-gb 8\n\n");
    printf("    # Interactive chat\n");
    printf("    qwen35-thor chat --config configs/qwen3.5-9b.conf\n\n");
    printf("    # Run benchmarks\n");
    printf("    qwen35-thor bench --model-dir /path/to/Qwen3.5-27B --decode 50\n\n");
    printf("    # Parameter sweep with JSON output\n");
    printf("    qwen35-thor bench --model-dir /path/to/model --batch 1,2,4 --iterations 3 --json results.json\n\n");
    printf("    # SM110a hardware primitives probe\n");
    printf("    qwen35-thor probe\n\n");
}

static void print_version() {
    printf("qwen35-thor v%s (%s)\n", VERSION, BUILD_DATE);
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

    // 加载 ASR / TTS 插件配置 (从同一个 config 文件或 CLI 参数)
    plugins::AsrConfig asr_config;
    plugins::TtsConfig tts_config;
    if (!config_file.empty()) {
        asr_config = plugins::AsrConfig::from_file(config_file);
        tts_config = plugins::TtsConfig::from_file(config_file);
    }
    // CLI 覆盖: --asr-executable, --tts-executable 等
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--asr-enabled")       asr_config.enabled = true;
        else if (arg == "--asr-mode"       && i + 1 < argc) asr_config.mode       = argv[++i];
        else if (arg == "--asr-executable" && i + 1 < argc) asr_config.executable = argv[++i];
        else if (arg == "--asr-model"      && i + 1 < argc) asr_config.model_path = argv[++i];
        else if (arg == "--asr-language"   && i + 1 < argc) asr_config.language   = argv[++i];
        else if (arg == "--tts-enabled")       tts_config.enabled = true;
        else if (arg == "--tts-mode"       && i + 1 < argc) tts_config.mode       = argv[++i];
        else if (arg == "--tts-executable" && i + 1 < argc) tts_config.executable = argv[++i];
        else if (arg == "--tts-model"      && i + 1 < argc) tts_config.model_path = argv[++i];
        else if (arg == "--tts-voice"      && i + 1 < argc) tts_config.voice      = argv[++i];
        else if (arg == "--tts-language"   && i + 1 < argc) tts_config.language   = argv[++i];
        else if (arg == "--tts-instruct"   && i + 1 < argc) tts_config.instruct   = argv[++i];
    }

    auto asr_plugin = plugins::create_asr_plugin(asr_config);
    auto tts_plugin = plugins::create_tts_plugin(tts_config);
    if (asr_config.enabled) asr_config.print();
    if (tts_config.enabled) tts_config.print();

    try {
        InferenceBackend backend(backend_config);
        serve::ServeApp app(serve_config, backend,
                            std::move(asr_plugin), std::move(tts_plugin));

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
    backend_config.verbose = false;  // chat 模式抑制推理日志
    if (!backend_config.cache_enabled) {
        backend_config.cache_enabled = true;  // chat 模式默认启用 prefix cache
    }
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
// 命令: asr (原生 Qwen3-ASR 语音转录)
// ============================================================================

static int cmd_asr(int argc, char** argv) {
    std::string model_dir;
    std::string wav_path;
    float temperature = 0.0f;
    int max_tokens = 448;

    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model-dir" || arg == "--model") && i + 1 < argc)
            model_dir = argv[++i];
        else if (arg == "--temperature" && i + 1 < argc)
            temperature = std::stof(argv[++i]);
        else if (arg == "--max-tokens" && i + 1 < argc)
            max_tokens = std::stoi(argv[++i]);
        else if (arg[0] != '-')
            wav_path = arg;
    }

    if (model_dir.empty()) {
        // Try env
        const char* env = getenv("QWEN_ASR_MODEL_DIR");
        if (env) model_dir = env;
    }

    if (model_dir.empty() || wav_path.empty()) {
        fprintf(stderr, "Usage: qwen35-thor asr --model-dir <path/to/Qwen3-ASR-1.7B> <audio.wav>\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --model-dir <path>    ASR model directory (or set QWEN_ASR_MODEL_DIR)\n");
        fprintf(stderr, "  --temperature <F>     Sampling temperature (default: 0.0 = greedy)\n");
        fprintf(stderr, "  --max-tokens <N>      Max output tokens (default: 448)\n");
        return 1;
    }

    fprintf(stderr, "[ASR] Model: %s\n", model_dir.c_str());
    fprintf(stderr, "[ASR] Audio: %s\n", wav_path.c_str());

    qwen_thor::asr::ASREngine engine;
    engine.load_model(model_dir);

    if (!engine.is_loaded()) {
        fprintf(stderr, "[ASR] ERROR: failed to load model\n");
        return 1;
    }

    std::string text = engine.transcribe_file(wav_path, temperature, max_tokens);
    // Output transcription to stdout (clean, no prefix)
    printf("%s\n", text.c_str());
    return 0;
}

// ============================================================================
// 命令: tts (原生 Qwen3-TTS 语音合成)
// ============================================================================

static int cmd_tts(int argc, char** argv) {
    std::string model_dir;
    std::string text;
    std::string speaker = "serena";
    std::string language = "auto";
    std::string output_path;
    std::string instruct;
    int max_tokens = 4096;
    bool wav_mode = false;

    // Parse args
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model-dir" || arg == "--model") && i + 1 < argc)
            model_dir = argv[++i];
        else if (arg == "--speaker" && i + 1 < argc)
            speaker = argv[++i];
        else if (arg == "--language" && i + 1 < argc)
            language = argv[++i];
        else if (arg == "--max-tokens" && i + 1 < argc)
            max_tokens = std::stoi(argv[++i]);
        else if ((arg == "--text" || arg == "-t") && i + 1 < argc)
            text = argv[++i];
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            output_path = argv[++i];
            wav_mode = true;
        }
        else if (arg == "--instruct" && i + 1 < argc)
            instruct = argv[++i];
        else if (arg == "--wav")
            wav_mode = true;
        else if (arg[0] != '-' && text.empty())
            text = arg;
    }

    if (model_dir.empty()) {
        const char* env = getenv("QWEN_TTS_MODEL_DIR");
        if (env) model_dir = env;
    }

    if (model_dir.empty() || text.empty()) {
        fprintf(stderr, "Usage: qwen35-thor tts --model-dir <path/to/Qwen3-TTS> \"text to speak\"\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  --model-dir <path>    TTS model directory (or set QWEN_TTS_MODEL_DIR)\n");
        fprintf(stderr, "  --text, -t <text>     Text to synthesize\n");
        fprintf(stderr, "  --speaker <name>      Speaker name (default: serena)\n");
        fprintf(stderr, "  --language <lang>     Language hint (default: auto)\n");
        fprintf(stderr, "  --max-tokens <N>      Max codec tokens (default: 4096)\n");
        fprintf(stderr, "  --output, -o <path>   Output WAV file path\n");
        fprintf(stderr, "  --wav                 Enable WAV output (default: tmp/tts_output.wav)\n");
        fprintf(stderr, "  --instruct <text>     Voice description (VoiceDesign mode)\n");
        return 1;
    }

    // Default output path
    if (wav_mode && output_path.empty()) {
        output_path = "tmp/tts_output.wav";
    }

    fprintf(stderr, "[TTS] Model: %s\n", model_dir.c_str());
    fprintf(stderr, "[TTS] Text: %s\n", text.c_str());
    fprintf(stderr, "[TTS] Speaker: %s, Language: %s\n", speaker.c_str(), language.c_str());
    if (!instruct.empty())
        fprintf(stderr, "[TTS] Instruct: %s\n", instruct.c_str());

    qwen_thor::tts::TTSEngine engine;
    engine.load_model(model_dir);

    if (!engine.is_loaded()) {
        fprintf(stderr, "[TTS] ERROR: failed to load model\n");
        return 1;
    }

    if (wav_mode || !output_path.empty()) {
        // End-to-end: text → WAV
        bool ok = engine.synthesize_to_wav(text, output_path, speaker, language,
                                           instruct, max_tokens);
        return ok ? 0 : 1;
    }

    // Phase 1 fallback: text → codec tokens (JSON)
    auto codes = engine.synthesize(text, speaker, language, max_tokens);

    printf("{\n  \"num_steps\": %d,\n  \"num_groups\": %d,\n  \"codes\": [\n",
           (int)codes.size(), codes.empty() ? 0 : (int)codes[0].size());
    for (size_t i = 0; i < codes.size(); i++) {
        printf("    [");
        for (size_t j = 0; j < codes[i].size(); j++) {
            printf("%d%s", codes[i][j], j + 1 < codes[i].size() ? ", " : "");
        }
        printf("]%s\n", i + 1 < codes.size() ? "," : "");
    }
    printf("  ]\n}\n");

    return 0;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    // Make stdout line-buffered and stderr unbuffered
    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Install crash handlers
    signal(SIGSEGV, crash_handler);
    signal(SIGBUS, crash_handler);
    signal(SIGFPE, crash_handler);
    signal(SIGABRT, crash_handler);

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];

    int rc = 0;
    if (cmd == "serve")   rc = cmd_serve(argc, argv);
    else if (cmd == "chat")    rc = cmd_chat(argc, argv);
    else if (cmd == "bench")   rc = cmd_bench(argc, argv);
    else if (cmd == "asr")     rc = cmd_asr(argc, argv);
    else if (cmd == "tts")     rc = cmd_tts(argc, argv);
    else if (cmd == "test")    rc = cmd_test(argc, argv);
    else if (cmd == "probe")   { sm110a_probe::run_sm110a_probes(); }
    else if (cmd == "version" || cmd == "--version" || cmd == "-v") {
        print_version();
    }
    else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
    }
    else {
        std::cerr << "Unknown command: " << cmd << "\n";
        std::cerr << "Run 'qwen35-thor --help' for usage.\n";
        rc = 1;
    }

    // 释放所有 CUDA 资源 (包括 function-local static 缓冲区)
    // Jetson UMA: cudaDeviceReset 后物理页变为内核可回收 cache,
    // MemAvailable 会正确反映, 其他进程分配时内核自动回收
    cudaDeviceReset();

    return rc;
}
