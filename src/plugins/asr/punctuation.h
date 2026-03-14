// punctuation.h — 标点恢复 (Phase 4)
//
// 为 ASR 无标点输出添加标点符号。两种模式:
//   1. 规则方案 (快速, 零依赖): 基于语句长度/停顿/常见模式
//   2. LLM 方案 (高精度): 调用已加载 LLM 添加标点
//
// 自动选择: 短文本 (≤30 字) 用规则, 长文本用 LLM (如可用)

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstring>
#include <algorithm>

namespace qwen_thor {
namespace asr {

// LLM 推理回调: 给定 system prompt + user text, 返回 LLM 输出
// 由调用方提供, 将 InferenceBackend 封装为此回调
using LlmInferFunc = std::function<std::string(const std::string& system_prompt,
                                               const std::string& user_text)>;

// ============================================================================
// PunctuationRestorer — 标点恢复
// ============================================================================
class PunctuationRestorer {
public:
    PunctuationRestorer() = default;

    // 规则方案 (无外部依赖)
    std::string restore_rules(const std::string& text) const {
        if (text.empty()) return text;

        std::string result;
        auto chars = split_utf8(text);
        int n = (int)chars.size();
        if (n == 0) return text;

        // 跟踪上一个标点位置
        int last_punc_pos = -1;

        for (int i = 0; i < n; ++i) {
            result += chars[i];

            // 已有标点则跳过
            if (is_punctuation(chars[i])) {
                last_punc_pos = i;
                continue;
            }

            // 句末处理
            if (i == n - 1) {
                // 检查是否为疑问句
                if (has_question_marker(chars, 0, n)) {
                    result += "\xef\xbc\x9f"; // ？
                } else if (has_exclamation_marker(chars, last_punc_pos + 1, n)) {
                    result += "\xef\xbc\x81"; // ！
                } else {
                    result += "\xe3\x80\x82"; // 。
                }
                break;
            }

            // 逗号插入: 连续 >15 字无标点, 在自然断点处
            int since_punc = i - last_punc_pos;
            if (since_punc >= 15 && is_natural_break(chars, i, n)) {
                result += "\xef\xbc\x8c"; // ，
                last_punc_pos = i;
            }
        }

        return result;
    }

    // LLM 方案 (高精度)
    std::string restore_llm(const std::string& text, LlmInferFunc infer_fn) const {
        if (text.empty() || !infer_fn) return text;

        const std::string system_prompt =
            "你是标点恢复助手。为以下语音转录文本添加标点符号。"
            "只添加标点，不修改任何文字内容。直接输出结果，不要任何解释。";

        std::string result = infer_fn(system_prompt, text);

        // 验证 LLM 输出: 去除标点后应与原文一致
        if (!validate_llm_output(text, result)) {
            // LLM 输出不可靠, 降级到规则
            return restore_rules(text);
        }
        return result;
    }

    // 自动选择: 短文本用规则, 长文本用 LLM
    std::string restore(const std::string& text,
                        LlmInferFunc infer_fn = nullptr,
                        bool prefer_rules = false) const {
        if (text.empty()) return text;
        if (has_existing_punctuation(text)) return text;

        // 优先规则 or 无 LLM 回调
        if (prefer_rules || !infer_fn) {
            return restore_rules(text);
        }

        // 短文本 (≤30 字) 用规则
        int char_count = count_utf8_chars(text);
        if (char_count <= 30) {
            return restore_rules(text);
        }

        // 长文本用 LLM
        return restore_llm(text, infer_fn);
    }

private:
    // ---- UTF-8 工具 ----
    static int utf8_char_len(unsigned char c) {
        if (c < 0x80) return 1;
        if (c < 0xE0) return 2;
        if (c < 0xF0) return 3;
        return 4;
    }

    static int count_utf8_chars(const std::string& s) {
        int count = 0;
        for (int i = 0; i < (int)s.size(); ) {
            i += utf8_char_len((unsigned char)s[i]);
            count++;
        }
        return count;
    }

    static std::vector<std::string> split_utf8(const std::string& s) {
        std::vector<std::string> chars;
        for (int i = 0; i < (int)s.size(); ) {
            int len = utf8_char_len((unsigned char)s[i]);
            chars.push_back(s.substr(i, len));
            i += len;
        }
        return chars;
    }

    // ---- 标点检测 ----
    static bool is_punctuation(const std::string& c) {
        // 中文标点
        static const char* zh_puncs[] = {
            "\xe3\x80\x82",     // 。
            "\xef\xbc\x8c",     // ，
            "\xef\xbc\x9f",     // ？
            "\xef\xbc\x81",     // ！
            "\xe3\x80\x81",     // 、
            "\xef\xbc\x9b",     // ；
            "\xef\xbc\x9a",     // ：
            "\xe2\x80\x9c",     // "
            "\xe2\x80\x9d",     // "
            "\xe2\x80\x98",     // '
            "\xe2\x80\x99",     // '
        };
        for (auto p : zh_puncs) {
            if (c == p) return true;
        }
        // ASCII 标点
        if (c.size() == 1) {
            char ch = c[0];
            return ch == '.' || ch == ',' || ch == '?' || ch == '!' ||
                   ch == ';' || ch == ':' || ch == '"' || ch == '\'';
        }
        return false;
    }

    // 检查文本是否已有标点
    static bool has_existing_punctuation(const std::string& text) {
        auto chars = split_utf8(text);
        int punc_count = 0;
        for (const auto& c : chars) {
            if (is_punctuation(c)) punc_count++;
        }
        // 如果标点占比 > 5%, 认为已有标点
        return punc_count > 0 && (float)punc_count / chars.size() > 0.05f;
    }

    // ---- 疑问词检测 ----
    static bool has_question_marker(const std::vector<std::string>& chars,
                                    int from, int to) {
        // 常见疑问词/语气词
        static const char* q_words[] = {
            "\xe5\x90\x97",     // 吗
            "\xe5\x91\xa2",     // 呢
            "\xe4\xb9\x88",     // 么 (什么)
            "\xe5\x93\xaa",     // 哪
            "\xe8\xb0\x81",     // 谁
            "\xe5\x87\xa0",     // 几
            "\xe5\xa4\x9a\xe5\xb0\x91", // 多少
            "\xe5\x93\xaa\xe9\x87\x8c", // 哪里
            "\xe5\x93\xaa\xe4\xb8\xaa", // 哪个
            "\xe5\x93\xaa\xe4\xba\x9b", // 哪些
            "\xe5\x93\xaa\xe5\x84\xbf", // 哪儿
        };
        static const char* q_prefixes[] = {
            "\xe4\xbb\x80\xe4\xb9\x88", // 什么
            "\xe6\x80\x8e\xe4\xb9\x88", // 怎么
            "\xe4\xb8\xba\xe4\xbb\x80\xe4\xb9\x88", // 为什么
            "\xe6\x98\xaf\xe5\x90\xa6",  // 是否
        };

        // 检查末尾疑问词
        for (auto q : q_words) {
            std::string qs(q);
            auto qc = split_utf8(qs);
            if ((int)qc.size() <= to - from) {
                bool match = true;
                for (int j = 0; j < (int)qc.size(); ++j) {
                    if (chars[to - (int)qc.size() + j] != qc[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) return true;
            }
        }

        // 检查句首疑问前缀
        std::string joined;
        for (int i = from; i < to; ++i) joined += chars[i];
        for (auto q : q_prefixes) {
            if (joined.find(q) != std::string::npos) return true;
        }

        return false;
    }

    // ---- 感叹词检测 ----
    static bool has_exclamation_marker(const std::vector<std::string>& chars,
                                       int from, int to) {
        static const char* exc_words[] = {
            "\xe5\xa4\xaa",     // 太
            "\xe7\x9c\x9f",     // 真
            "\xe5\x95\x8a",     // 啊
            "\xe5\x93\x87",     // 哇
            "\xe5\x93\xa6",     // 哦
        };
        for (int i = std::max(0, to - 3); i < to; ++i) {
            for (auto e : exc_words) {
                if (chars[i] == e) return true;
            }
        }
        return false;
    }

    // ---- 自然断点检测 (用于逗号插入) ----
    static bool is_natural_break(const std::vector<std::string>& chars,
                                  int pos, int n) {
        if (pos + 1 >= n) return false;

        // 1. 转折/连接词前: 但是/然后/所以/因为/如果/虽然/不过/以及/或者/而且
        static const char* conjunctions[][2] = {
            {"\xe4\xbd\x86", "\xe6\x98\xaf"},   // 但是
            {"\xe7\x84\xb6", "\xe5\x90\x8e"},   // 然后
            {"\xe6\x89\x80", "\xe4\xbb\xa5"},   // 所以
            {"\xe5\x9b\xa0", "\xe4\xb8\xba"},   // 因为
            {"\xe5\xa6\x82", "\xe6\x9e\x9c"},   // 如果
            {"\xe8\x99\xbd", "\xe7\x84\xb6"},   // 虽然
            {"\xe4\xb8\x8d", "\xe8\xbf\x87"},   // 不过
            {"\xe8\x80\x8c", "\xe4\xb8\x94"},   // 而且
            {"\xe6\x88\x96", "\xe8\x80\x85"},   // 或者
        };
        for (auto& conj : conjunctions) {
            if (pos + 2 < n && chars[pos + 1] == conj[0] && chars[pos + 2] == conj[1]) {
                return true;
            }
        }

        // 2. 在 "的" / "了" / "过" 后面做逗号
        static const char* particles[] = {
            "\xe7\x9a\x84",     // 的
            "\xe4\xba\x86",     // 了
            "\xe8\xbf\x87",     // 过
        };
        for (auto p : particles) {
            if (chars[pos] == p) return true;
        }

        return false;
    }

    // ---- LLM 输出验证 ----
    static bool validate_llm_output(const std::string& original, const std::string& restored) {
        // 去除标点后对比
        auto orig_chars = split_utf8(original);
        auto rest_chars = split_utf8(restored);

        std::string orig_clean, rest_clean;
        for (const auto& c : orig_chars) {
            if (!is_punctuation(c) && c != " ") orig_clean += c;
        }
        for (const auto& c : rest_chars) {
            if (!is_punctuation(c) && c != " ") rest_clean += c;
        }

        return orig_clean == rest_clean;
    }
};

} // namespace asr
} // namespace qwen_thor
