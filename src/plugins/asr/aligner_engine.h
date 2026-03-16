// aligner_engine.h — Qwen3-ForcedAligner 字级时间戳
//
// 通过 Python 子进程使用 Qwen3-ForcedAligner-0.6B:
//   1. 启动 forced_aligner_server.py 长驻进程
//   2. 通过 stdin/stdout JSON 协议通信
//   3. 模型只加载一次, 避免重复初始化开销
//
// 协议:
//   C++ → Python: {"audio_pcm_path": "/tmp/fa_xxx.pcm", "sr": 16000, "text": "...", "language": "Chinese"}\n
//   Python → C++: {"ok": true, "words": [{"text": "字", "start_ms": 80, "end_ms": 160}, ...]}\n

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

namespace qwen_thor {
namespace asr {

// 对齐结果: 每个词的时间戳
struct AlignedWord {
    std::string word;
    int   start_ms    = -1;
    int   end_ms      = -1;
    float confidence  = 0;
};

// ============================================================================
// AlignerEngine — Qwen3-ForcedAligner subprocess wrapper
// ============================================================================
class AlignerEngine {
public:
    AlignerEngine() = default;

    ~AlignerEngine() {
        shutdown();
    }

    // 启动 Python 子进程并加载模型
    bool load_model(const std::string& model_dir,
                    const std::string& python_path = "",
                    const std::string& script_path = "",
                    const std::string& device = "cuda:0") {
        std::lock_guard<std::mutex> lock(mutex_);

        if (running_) shutdown_locked();

        model_dir_ = model_dir;

        // 验证模型文件存在
        {
            std::ifstream cf(model_dir + "/config.json");
            if (!cf.is_open()) {
                fprintf(stderr, "[AlignerEngine] config.json not found in %s\n", model_dir.c_str());
                return false;
            }
        }

        // 确定 Python 路径
        std::string py = python_path;
        if (py.empty()) {
            const char* candidates[] = {
                "/home/rm01/miniconda3/envs/vllm/bin/python3",
                "/usr/bin/python3",
                nullptr
            };
            for (auto* c = candidates; *c; ++c) {
                if (access(*c, X_OK) == 0) { py = *c; break; }
            }
        }
        if (py.empty()) {
            fprintf(stderr, "[AlignerEngine] No Python interpreter found\n");
            return false;
        }

        // 确定脚本路径
        std::string script = script_path;
        if (script.empty()) {
            const char* script_candidates[] = {
                "src/plugins/asr/forced_aligner_server.py",
                "../src/plugins/asr/forced_aligner_server.py",
                nullptr
            };
            for (auto* c = script_candidates; *c; ++c) {
                if (access(*c, R_OK) == 0) { script = *c; break; }
            }
        }
        if (script.empty() || access(script.c_str(), R_OK) != 0) {
            fprintf(stderr, "[AlignerEngine] Script not found: %s\n", script.c_str());
            return false;
        }

        // 创建管道
        int pipe_to_child[2];
        int pipe_from_child[2];
        if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) {
            fprintf(stderr, "[AlignerEngine] Failed to create pipes\n");
            return false;
        }

        pid_t pid = fork();
        if (pid < 0) {
            fprintf(stderr, "[AlignerEngine] Fork failed\n");
            close(pipe_to_child[0]); close(pipe_to_child[1]);
            close(pipe_from_child[0]); close(pipe_from_child[1]);
            return false;
        }

        if (pid == 0) {
            // Child process
            close(pipe_to_child[1]);
            close(pipe_from_child[0]);
            dup2(pipe_to_child[0], STDIN_FILENO);
            dup2(pipe_from_child[1], STDOUT_FILENO);
            close(pipe_to_child[0]);
            close(pipe_from_child[1]);

            execlp(py.c_str(), py.c_str(), script.c_str(),
                   "--model", model_dir.c_str(),
                   "--device", device.c_str(),
                   (char*)nullptr);
            _exit(127);
        }

        // Parent process
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);

        pid_ = pid;
        to_child_ = pipe_to_child[1];
        from_child_ = pipe_from_child[0];

        // 读取 ready 消息 (120s timeout: nagisa 初始化 + 模型加载较慢)
        std::string ready_line = read_line(120000);
        if (ready_line.empty() || ready_line.find("\"ready\"") == std::string::npos) {
            fprintf(stderr, "[AlignerEngine] Python subprocess failed to start\n");
            shutdown_locked();
            return false;
        }

        running_ = true;
        loaded_ = true;
        fprintf(stderr, "[AlignerEngine] ForcedAligner subprocess ready (pid=%d)\n", (int)pid_);
        return true;
    }

    bool is_loaded() const { return loaded_ && running_; }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_locked();
    }

    // 对齐: 给定 PCM 音频 + ASR 文本 → 每词时间戳
    std::vector<AlignedWord> align(const float* pcm, int num_samples,
                                    int sample_rate,
                                    const std::string& text,
                                    const std::string& language = "Chinese") {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!running_ || text.empty()) return {};

        // 写 PCM 到临时文件
        char tmppath[] = "/tmp/fa_pcm_XXXXXX";
        int tmpfd = mkstemp(tmppath);
        if (tmpfd < 0) return {};

        ssize_t nbytes = (ssize_t)num_samples * sizeof(float);
        ssize_t written = write(tmpfd, pcm, nbytes);
        close(tmpfd);

        if (written != nbytes) {
            unlink(tmppath);
            return {};
        }

        // 构造请求 JSON
        std::string escaped_text = escape_json_string(text);
        std::string request = "{\"audio_pcm_path\":\"" + std::string(tmppath) +
                              "\",\"sr\":" + std::to_string(sample_rate) +
                              ",\"text\":\"" + escaped_text +
                              "\",\"language\":\"" + language + "\"}\n";

        ssize_t wr = ::write(to_child_, request.c_str(), request.size());
        if (wr != (ssize_t)request.size()) {
            unlink(tmppath);
            fprintf(stderr, "[AlignerEngine] Write to subprocess failed\n");
            return {};
        }

        // 读取响应 (120s timeout: 长音频推理慢, GPU 共享时更慢)
        std::string response = read_line(120000);
        unlink(tmppath);

        if (response.empty()) {
            fprintf(stderr, "[AlignerEngine] Timeout reading alignment result\n");
            return {};
        }

        return parse_response(response);
    }

    // 从 ASR 文本提取词列表 (中文按字, 英文按空格, 去除标点)
    static std::vector<std::string> tokenize_for_align(const std::string& text) {
        std::vector<std::string> words;
        int i = 0;
        while (i < (int)text.size()) {
            unsigned char c = (unsigned char)text[i];
            if (c < 0x80) {
                if (c == ' ' || c == '\t' || c == '\n') { ++i; continue; }
                if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '\'')) { ++i; continue; }
                std::string word;
                while (i < (int)text.size() && (unsigned char)text[i] < 0x80 &&
                       text[i] != ' ' && text[i] != '\t' && text[i] != '\n') {
                    word += text[i++];
                }
                if (!word.empty()) words.push_back(word);
            } else {
                int len = 1;
                if (c >= 0xF0) len = 4;
                else if (c >= 0xE0) len = 3;
                else if (c >= 0xC0) len = 2;

                uint32_t codepoint = 0;
                if (len == 3)
                    codepoint = ((c & 0x0F) << 12) | (((unsigned char)text[i+1] & 0x3F) << 6) | ((unsigned char)text[i+2] & 0x3F);
                else if (len == 4)
                    codepoint = ((c & 0x07) << 18) | (((unsigned char)text[i+1] & 0x3F) << 12) | (((unsigned char)text[i+2] & 0x3F) << 6) | ((unsigned char)text[i+3] & 0x3F);
                else if (len == 2)
                    codepoint = ((c & 0x1F) << 6) | ((unsigned char)text[i+1] & 0x3F);

                bool is_cjk = (codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
                              (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
                              (codepoint >= 0x20000 && codepoint <= 0x2A6DF);
                bool is_letter = (codepoint >= 0x0080 && codepoint < 0x3000) ||
                                 (codepoint >= 0xAC00 && codepoint <= 0xD7AF);
                bool is_kana = (codepoint >= 0x3040 && codepoint <= 0x30FF) ||
                               (codepoint >= 0x31F0 && codepoint <= 0x31FF);

                if (is_cjk || is_letter || is_kana)
                    words.push_back(text.substr(i, len));

                i += len;
            }
        }
        return words;
    }

    // 确保时间戳单调递增 (LIS 后处理)
    static void fix_timestamps(std::vector<AlignedWord>& words) {
        if (words.size() <= 1) return;

        int n = (int)words.size();
        std::vector<int> starts(n);
        for (int i = 0; i < n; ++i) starts[i] = words[i].start_ms;

        std::vector<int> dp, pos;
        std::vector<int> parent(n, -1);
        for (int i = 0; i < n; ++i) {
            auto it = std::lower_bound(dp.begin(), dp.end(), starts[i]);
            int idx = (int)(it - dp.begin());
            if (it == dp.end()) {
                dp.push_back(starts[i]);
                pos.push_back(i);
            } else {
                *it = starts[i];
                pos[idx] = i;
            }
            parent[i] = idx > 0 ? pos[idx - 1] : -1;
        }

        std::vector<bool> in_lis(n, false);
        int cur = pos.back();
        while (cur >= 0) {
            in_lis[cur] = true;
            cur = parent[cur];
        }

        int prev_ms = 0, prev_idx = -1;
        for (int i = 0; i < n; ++i) {
            if (in_lis[i]) {
                if (prev_idx >= 0 && i - prev_idx > 1) {
                    int gap = words[i].start_ms - prev_ms;
                    int steps = i - prev_idx;
                    for (int j = prev_idx + 1; j < i; ++j)
                        words[j].start_ms = prev_ms + gap * (j - prev_idx) / steps;
                }
                prev_ms = words[i].start_ms;
                prev_idx = i;
            }
        }
        if (prev_idx >= 0 && prev_idx < n - 1) {
            for (int j = prev_idx + 1; j < n; ++j)
                words[j].start_ms = prev_ms + 80 * (j - prev_idx);
        }

        for (int i = 0; i < n - 1; ++i)
            words[i].end_ms = words[i + 1].start_ms;
    }

private:
    std::string model_dir_;
    bool loaded_ = false;
    bool running_ = false;
    pid_t pid_ = -1;
    int to_child_ = -1;
    int from_child_ = -1;
    std::mutex mutex_;

    void shutdown_locked() {
        if (to_child_ >= 0) {
            const char quit_cmd[] = "{\"cmd\":\"quit\"}\n";
            ssize_t r_ = ::write(to_child_, quit_cmd, sizeof(quit_cmd) - 1);
            (void)r_;
            close(to_child_);
            to_child_ = -1;
        }
        if (from_child_ >= 0) {
            close(from_child_);
            from_child_ = -1;
        }
        if (pid_ > 0) {
            int status;
            pid_t r = waitpid(pid_, &status, WNOHANG);
            if (r == 0) {
                kill(pid_, SIGTERM);
                usleep(100000);
                waitpid(pid_, &status, WNOHANG);
            }
            pid_ = -1;
        }
        running_ = false;
        loaded_ = false;
    }

    std::string read_line(int timeout_ms) {
        std::string line;
        char buf[1];
        int elapsed = 0;
        const int poll_interval = 10;

        while (elapsed < timeout_ms) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(from_child_, &fds);
            struct timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = poll_interval * 1000;

            int ret = select(from_child_ + 1, &fds, nullptr, nullptr, &tv);
            if (ret > 0 && FD_ISSET(from_child_, &fds)) {
                ssize_t n = read(from_child_, buf, 1);
                if (n <= 0) return "";
                if (buf[0] == '\n') return line;
                line += buf[0];
            } else if (ret < 0) {
                return "";
            }
            elapsed += poll_interval;
        }
        return "";
    }

    static std::string escape_json_string(const std::string& s) {
        std::string result;
        result.reserve(s.size() + 10);
        for (char c : s) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:   result += c; break;
            }
        }
        return result;
    }

    std::vector<AlignedWord> parse_response(const std::string& json) {
        std::vector<AlignedWord> result;

        if (json.find("\"ok\":false") != std::string::npos ||
            json.find("\"ok\": false") != std::string::npos) {
            fprintf(stderr, "[AlignerEngine] Alignment failed: %s\n", json.c_str());
            return result;
        }

        size_t words_pos = json.find("\"words\"");
        if (words_pos == std::string::npos) return result;

        size_t arr_start = json.find('[', words_pos);
        if (arr_start == std::string::npos) return result;

        size_t pos = arr_start + 1;
        while (pos < json.size()) {
            size_t obj_start = json.find('{', pos);
            if (obj_start == std::string::npos) break;
            size_t obj_end = json.find('}', obj_start);
            if (obj_end == std::string::npos) break;

            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);

            AlignedWord aw;

            size_t text_pos = obj.find("\"text\"");
            if (text_pos != std::string::npos) {
                size_t colon = obj.find(':', text_pos);
                size_t q1 = obj.find('"', colon + 1);
                size_t q2 = obj.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    aw.word = obj.substr(q1 + 1, q2 - q1 - 1);
            }

            size_t start_pos = obj.find("\"start_ms\"");
            if (start_pos != std::string::npos) {
                size_t colon = obj.find(':', start_pos);
                aw.start_ms = atoi(obj.c_str() + colon + 1);
            }

            size_t end_pos = obj.find("\"end_ms\"");
            if (end_pos != std::string::npos) {
                size_t colon = obj.find(':', end_pos);
                aw.end_ms = atoi(obj.c_str() + colon + 1);
            }

            if (!aw.word.empty()) {
                aw.confidence = (aw.end_ms > aw.start_ms) ? 0.9f : 0.3f;
                result.push_back(aw);
            }

            pos = obj_end + 1;
        }

        return result;
    }
};

} // namespace asr
} // namespace qwen_thor
