// tui.h — 终端交互式 Chat 界面
//
// 直接嵌入进程内的 TUI Chat, 通过 InferenceBackend API 与引擎交互。
// 使用 POSIX 终端控制实现彩色输出和行编辑。
// 支持 Ctrl+C 中断当前生成, SIGINT 安全处理。

#pragma once

#include "../engine/backend.h"
#include <string>
#include <vector>
#include <atomic>

namespace qwen_thor {
namespace tui {

// 全局中断标志 — SIGINT handler 设置, generate() loop 检查
extern std::atomic<bool> g_interrupted;

// ============================================================================
// TUI 配置
// ============================================================================
struct TuiConfig {
    int    max_new_tokens  = 2048;
    float  temperature     = 1.0f;
    float  top_p           = 0.95f;
    int    timeout_s       = 300;
    bool   show_stats      = true;     // 显示推理性能统计
    bool   enable_thinking = true;     // 启用 thinking 模式
    std::string system_prompt = "You are a helpful assistant.";

    static TuiConfig from_args(int argc, char** argv);
};

// ============================================================================
// ChatApp — TUI 交互式 Chat
// ============================================================================
class ChatApp {
public:
    ChatApp(const TuiConfig& config, InferenceBackend& backend);
    ~ChatApp() = default;

    // 启动交互式会话 (阻塞主线程)
    void run();

private:
    void print_banner();
    void print_help();
    void print_prompt();

    // 处理用户输入
    void handle_input(const std::string& input);

    // 发送推理请求并流式打印结果
    void generate(const std::string& prompt);

    TuiConfig config_;
    InferenceBackend& backend_;
    uint64_t next_id_ = 1;
    std::vector<std::pair<std::string, std::string>> history_;  // (role, content)
};

} // namespace tui
} // namespace qwen_thor
