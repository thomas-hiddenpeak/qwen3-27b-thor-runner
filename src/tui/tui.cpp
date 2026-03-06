// tui.cpp — TUI Chat 实现
//
// 终端交互式推理界面, 直接调用 InferenceBackend API。
// 流程: 用户输入 → chat template → tokenize → submit → poll tokens → decode → 打印
//
// 特性:
//   - Ctrl+C 中断当前生成 (不退出程序)
//   - Thinking 阶段显示动态进度 (token 计数)
//   - /think 命令切换 thinking 模式
//   - /nothink 强制关闭 thinking

#include "tui.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <thread>
#include <csignal>

namespace qwen_thor {
namespace tui {

// ============================================================================
// SIGINT 处理 — Ctrl+C 中断当前生成
// ============================================================================
std::atomic<bool> g_interrupted{false};

static struct sigaction s_old_sigint;

static void sigint_handler(int /*sig*/) {
    g_interrupted.store(true, std::memory_order_release);
}

static void install_sigint_handler() {
    struct sigaction sa{};
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // 不用 SA_RESTART, 让 getline 也能被中断
    sigaction(SIGINT, &sa, &s_old_sigint);
}

static void restore_sigint_handler() {
    sigaction(SIGINT, &s_old_sigint, nullptr);
}

// ============================================================================
// ANSI 颜色代码
// ============================================================================
namespace color {
    constexpr const char* RESET   = "\033[0m";
    constexpr const char* BOLD    = "\033[1m";
    constexpr const char* DIM     = "\033[2m";
    constexpr const char* RED     = "\033[31m";
    constexpr const char* GREEN   = "\033[32m";
    constexpr const char* YELLOW  = "\033[33m";
    constexpr const char* BLUE    = "\033[34m";
    constexpr const char* MAGENTA = "\033[35m";
    constexpr const char* CYAN    = "\033[36m";
    constexpr const char* WHITE   = "\033[37m";
    // 控制序列
    constexpr const char* CLEAR_LINE = "\033[2K\r";  // 清除当前行并回到行首
}

// ============================================================================
// TuiConfig
// ============================================================================

TuiConfig TuiConfig::from_args(int argc, char** argv) {
    TuiConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--max-tokens" && i + 1 < argc)
            cfg.max_new_tokens = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc)
            cfg.temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc)
            cfg.top_p = std::stof(argv[++i]);
        else if (arg == "--timeout" && i + 1 < argc)
            cfg.timeout_s = std::stoi(argv[++i]);
        else if (arg == "--no-stats")
            cfg.show_stats = false;
        else if (arg == "--no-think" || arg == "--nothink")
            cfg.enable_thinking = false;
        else if (arg == "--system" && i + 1 < argc)
            cfg.system_prompt = argv[++i];
    }
    return cfg;
}

// ============================================================================
// ChatApp
// ============================================================================

ChatApp::ChatApp(const TuiConfig& config, InferenceBackend& backend)
    : config_(config), backend_(backend) {}

void ChatApp::run() {
    print_banner();

    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        printf("%s[Error] Tokenizer not loaded. Cannot start chat.%s\n",
               color::RED, color::RESET);
        return;
    }

    backend_.start();
    install_sigint_handler();

    std::string line;
    while (true) {
        g_interrupted.store(false);
        print_prompt();
        if (!std::getline(std::cin, line)) {
            // EOF (Ctrl+D) 或 SIGINT 中断了 getline
            if (g_interrupted.load()) {
                // Ctrl+C 在等待输入时 — 清除并继续
                std::cin.clear();
                printf("\n");
                continue;
            }
            break;  // 真正的 EOF
        }

        // 去除首尾空白
        while (!line.empty() && line.front() == ' ') line.erase(line.begin());
        while (!line.empty() && line.back() == ' ')  line.pop_back();

        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit" || line == "/q") break;

        handle_input(line);
    }

    restore_sigint_handler();
    printf("\n%s%sGoodbye!%s\n", color::BOLD, color::CYAN, color::RESET);
    backend_.stop();
}

void ChatApp::print_banner() {
    printf("\n");
    printf("%s%s", color::BOLD, color::CYAN);
    printf("  ╔═══════════════════════════════════════════════════╗\n");
    printf("  ║           Qwen3.5-27B  •  Thor SM110             ║\n");
    printf("  ║         BF16 Inference  •  TUI Chat              ║\n");
    printf("  ╚═══════════════════════════════════════════════════╝\n");
    printf("%s", color::RESET);
    printf("\n");
    printf("  %sCommands:%s\n", color::DIM, color::RESET);
    printf("    %s/help%s     — Show help\n", color::YELLOW, color::RESET);
    printf("    %s/clear%s    — Clear history\n", color::YELLOW, color::RESET);
    printf("    %s/tokens N%s — Set max_new_tokens\n", color::YELLOW, color::RESET);
    printf("    %s/think%s    — Toggle thinking mode\n", color::YELLOW, color::RESET);
    printf("    %s/nothink%s  — Disable thinking mode\n", color::YELLOW, color::RESET);
    printf("    %s/stats%s    — Toggle performance stats\n", color::YELLOW, color::RESET);
    printf("    %s/exit%s     — Quit\n", color::YELLOW, color::RESET);
    printf("    %sCtrl+C%s    — Interrupt current generation\n", color::YELLOW, color::RESET);
    printf("\n");
    printf("  %sThinking mode: %s%s%s\n", color::DIM,
           config_.enable_thinking ? color::GREEN : color::YELLOW,
           config_.enable_thinking ? "ON" : "OFF", color::RESET);
    printf("\n");
}

void ChatApp::print_help() {
    printf("\n%sAvailable commands:%s\n", color::BOLD, color::RESET);
    printf("  %s/help%s       Show this help message\n", color::YELLOW, color::RESET);
    printf("  %s/clear%s      Clear conversation history\n", color::YELLOW, color::RESET);
    printf("  %s/tokens N%s   Set max new tokens (current: %d)\n",
           color::YELLOW, color::RESET, config_.max_new_tokens);
    printf("  %s/temp F%s     Set temperature (current: %.2f)\n",
           color::YELLOW, color::RESET, config_.temperature);
    printf("  %s/system S%s   Set system prompt\n", color::YELLOW, color::RESET);
    printf("  %s/think%s      Toggle thinking mode (current: %s)\n",
           color::YELLOW, color::RESET,
           config_.enable_thinking ? "ON" : "OFF");
    printf("  %s/nothink%s    Disable thinking mode\n", color::YELLOW, color::RESET);
    printf("  %s/stats%s      Toggle performance stats\n", color::YELLOW, color::RESET);
    printf("  %s/exit%s       Quit the chat\n", color::YELLOW, color::RESET);
    printf("  %sCtrl+C%s      Interrupt current generation\n", color::YELLOW, color::RESET);
    printf("\n");
}

void ChatApp::print_prompt() {
    printf("%s%sYou > %s", color::BOLD, color::GREEN, color::RESET);
    fflush(stdout);
}

void ChatApp::handle_input(const std::string& input) {
    if (input == "/help") {
        print_help();
    } else if (input == "/clear") {
        history_.clear();
        printf("%sConversation cleared.%s\n", color::DIM, color::RESET);
    } else if (input.size() > 8 && input.substr(0, 8) == "/tokens ") {
        config_.max_new_tokens = std::stoi(input.substr(8));
        printf("%smax_new_tokens = %d%s\n", color::DIM, config_.max_new_tokens, color::RESET);
    } else if (input.size() > 6 && input.substr(0, 6) == "/temp ") {
        config_.temperature = std::stof(input.substr(6));
        printf("%stemperature = %.2f%s\n", color::DIM, config_.temperature, color::RESET);
    } else if (input.size() > 8 && input.substr(0, 8) == "/system ") {
        config_.system_prompt = input.substr(8);
        printf("%ssystem prompt updated%s\n", color::DIM, color::RESET);
    } else if (input == "/think") {
        config_.enable_thinking = !config_.enable_thinking;
        printf("%sThinking mode: %s%s\n", color::DIM,
               config_.enable_thinking ? "ON" : "OFF", color::RESET);
    } else if (input == "/nothink") {
        config_.enable_thinking = false;
        printf("%sThinking mode: OFF%s\n", color::DIM, color::RESET);
    } else if (input == "/stats") {
        config_.show_stats = !config_.show_stats;
        printf("%sPerformance stats: %s%s\n", color::DIM,
               config_.show_stats ? "ON" : "OFF", color::RESET);
    } else if (input[0] == '/') {
        printf("%sUnknown command: %s (type /help)%s\n", color::RED, input.c_str(), color::RESET);
    } else {
        generate(input);
    }
}

void ChatApp::generate(const std::string& user_input) {
    const auto& tok = backend_.tokenizer();

    // 1. 构建消息列表 (包含历史对话)
    std::vector<std::pair<std::string, std::string>> messages;

    // System prompt
    if (!config_.system_prompt.empty()) {
        messages.emplace_back("system", config_.system_prompt);
    }

    // 历史对话
    for (const auto& [role, content] : history_) {
        messages.emplace_back(role, content);
    }

    // 当前用户输入
    messages.emplace_back("user", user_input);

    // 2. apply_chat_template → tokenize (传入 enable_thinking)
    auto prompt_tokens = tok.apply_chat_template(messages, true, config_.enable_thinking);

    if (prompt_tokens.empty()) {
        printf("%s[Error] Tokenization failed.%s\n", color::RED, color::RESET);
        return;
    }

    // 3. 构建推理请求
    InferRequest req;
    req.request_id     = next_id_++;
    req.prompt_tokens  = std::move(prompt_tokens);
    req.max_new_tokens = config_.max_new_tokens;
    req.temperature    = config_.temperature;
    req.top_p          = config_.top_p;
    req.top_k          = 20;              // Qwen3.5 官方推荐
    req.min_p          = 0.0f;            // Qwen3.5 官方推荐不使用
    req.presence_penalty = 1.5f;          // Qwen3.5 官方推荐
    req.stream         = true;

    // 4. 提交请求
    if (!backend_.submit(req)) {
        printf("%s[Error] Request queue full. Try again later.%s\n", color::RED, color::RESET);
        return;
    }

    // 5. 清除中断标志
    g_interrupted.store(false);

    // 6. 轮询 token 并流式打印
    auto start_time = std::chrono::steady_clock::now();
    int total_tokens = 0;    // 所有生成的 tokens (包括 thinking)
    int content_tokens = 0;  // 正式内容 tokens
    int think_tokens = 0;    // thinking 阶段 tokens
    bool first_content_token = true;
    std::chrono::steady_clock::time_point first_content_time;
    std::string assistant_response;

    // thinking 模式下先进入 thinking 阶段
    bool in_thinking = config_.enable_thinking;
    bool printed_header = false;

    auto timeout = std::chrono::seconds(config_.timeout_s);
    bool finished = false;
    bool interrupted = false;
    bool error = false;

    while (!finished) {
        // 检查用户中断 (Ctrl+C)
        if (g_interrupted.load(std::memory_order_acquire)) {
            interrupted = true;
            backend_.cancel(req.request_id);
            break;
        }

        InferResponse resp;
        if (backend_.poll(resp)) {
            if (resp.request_id != req.request_id) continue;

            if (resp.error_code != 0) {
                // 清除 thinking 进度行
                if (in_thinking && think_tokens > 0) {
                    printf("%s", color::CLEAR_LINE);
                }
                printf("\n%s[Error] Inference error (code=%d)%s\n",
                       color::RED, resp.error_code, color::RESET);
                error = true;
                break;
            }

            if (resp.is_finished) {
                finished = true;
                break;
            }

            int tid = resp.token_id;

            // EOS 停止 token
            if (tid == tok.eos_token_id() || tid == tok.eot_id() ||
                tid == tok.im_end_id()) {
                finished = true;
                break;
            }

            total_tokens++;

            // </think> 标记: 从 thinking 切换到 content
            if (tid == tok.think_end_id()) {
                if (in_thinking) {
                    // 清除 thinking 进度指示
                    printf("%s", color::CLEAR_LINE);
                    in_thinking = false;
                }
                continue;
            }

            // <think> 标记: 进入 thinking (通常不会出现, 但防御性处理)
            if (tid == tok.think_start_id()) {
                in_thinking = true;
                continue;
            }

            if (in_thinking) {
                // Thinking 阶段 — 不打印到终端, 但显示进度
                think_tokens++;
                // 每个 token 更新进度指示 (覆盖同一行)
                printf("%s  %s[thinking... %d tokens]%s",
                       color::CLEAR_LINE, color::DIM, think_tokens, color::RESET);
                fflush(stdout);
            } else {
                // Content 阶段 — 流式打印
                if (!printed_header) {
                    printf("\n%s%sAssistant > %s", color::BOLD, color::MAGENTA, color::RESET);
                    printed_header = true;
                }

                if (first_content_token) {
                    first_content_time = std::chrono::steady_clock::now();
                    first_content_token = false;
                }

                std::string piece = tok.decode(tid);
                printf("%s", piece.c_str());
                fflush(stdout);
                assistant_response += piece;
                content_tokens++;
            }
        } else {
            // 暂无 token, 短暂休眠避免 busy wait
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // 超时检查
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > timeout) {
            if (in_thinking && think_tokens > 0) {
                printf("%s", color::CLEAR_LINE);
            }
            printf("\n%s[Timeout] Generation exceeded %ds limit.%s\n",
                   color::YELLOW, config_.timeout_s, color::RESET);
            backend_.cancel(req.request_id);
            break;
        }
    }

    // 被 Ctrl+C 中断
    if (interrupted) {
        if (in_thinking && think_tokens > 0) {
            printf("%s", color::CLEAR_LINE);
        }
        printf("\n%s[Interrupted]%s\n", color::YELLOW, color::RESET);

        // 排空残余 tokens (engine 可能还在发, 需要等 is_finished)
        auto drain_start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - drain_start < std::chrono::seconds(2)) {
            InferResponse drain_resp;
            if (backend_.poll(drain_resp)) {
                if (drain_resp.request_id == req.request_id &&
                    (drain_resp.is_finished || drain_resp.error_code != 0))
                    break;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    // 清除 thinking 进度行 (如果还在 thinking 阶段结束)
    if (in_thinking && think_tokens > 0) {
        printf("%s", color::CLEAR_LINE);
    }

    printf("\n");

    // 7. 显示统计信息
    if (config_.show_stats && total_tokens > 0 && !error) {
        auto end_time = std::chrono::steady_clock::now();
        double total_s = std::chrono::duration<double>(end_time - start_time).count();
        double tok_per_s = total_tokens / total_s;

        printf("%s  [%d tokens (think:%d + content:%d), %.1f tok/s, %.1fs",
               color::DIM, total_tokens, think_tokens, content_tokens, tok_per_s, total_s);
        if (!first_content_token) {
            double ttft_ms = std::chrono::duration<double, std::milli>(
                first_content_time - start_time).count();
            printf(", TTFT %.0fms", ttft_ms);
        }
        if (interrupted) printf(", interrupted");
        printf("]%s\n", color::RESET);
    }
    printf("\n");

    // 8. 更新历史 (即使被中断, 部分回复也加入历史)
    history_.emplace_back("user", user_input);
    if (!assistant_response.empty()) {
        history_.emplace_back("assistant", assistant_response);
    }
}

} // namespace tui
} // namespace qwen_thor
