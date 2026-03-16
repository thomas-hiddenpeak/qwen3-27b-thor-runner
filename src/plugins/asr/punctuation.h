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
                if (is_filler(chars[i])) break;
                if (has_question_marker(chars, last_punc_pos + 1, n)) {
                    result += "\xef\xbc\x9f"; // ？
                } else if (has_exclamation_marker(chars, last_punc_pos + 1, n)) {
                    result += "\xef\xbc\x81"; // ！
                } else {
                    result += "\xe3\x80\x82"; // 。
                }
                break;
            }

            int since_punc = i - last_punc_pos;

            // --- "对不对/是不是/是吧/对吧/好吧" → ？ (反问/确认) ---
            if (since_punc >= 3 && i + 1 < n) {
                if (is_discourse_marker(chars, i, n)) {
                    result += "\xef\xbc\x9f"; // ？
                    last_punc_pos = i;
                    continue;
                }
            }

            // --- 语气词/填充词后断句 (高优先级) ---
            if (since_punc >= 1 && is_filler(chars[i]) &&
                i + 1 < n && !is_filler(chars[i + 1]) && !is_punctuation(chars[i + 1])) {
                // 防止拆开 "对不对/是不是"
                bool in_reduplicate = false;
                if (i + 2 < n && chars[i + 1] == "\xe4\xb8\x8d") { // "X不X"
                    if (chars[i + 2] == chars[i]) in_reduplicate = true;
                }
                if (!in_reduplicate) {
                    // 决定标点: 短句(嗯独立) → 。, 长句末尾有问句 → ？, 其他 → ，
                    std::string punc = select_clause_end(chars, last_punc_pos + 1, i + 1, since_punc);
                    result += punc;
                    last_punc_pos = i;
                    continue;
                }
            }

            // --- 连接词前断句 (中优先级) ---
            if (since_punc >= 6 && i + 2 < n && is_conjunction_ahead(chars, i + 1, n)) {
                std::string punc = select_clause_end(chars, last_punc_pos + 1, i + 1, since_punc);
                result += punc;
                last_punc_pos = i;
                continue;
            }

            // --- 虚词后断句 (低优先级, 需 ≥12 字距) ---
            if (since_punc >= 12 && is_safe_particle_break(chars, i, n)) {
                std::string punc = select_clause_end(chars, last_punc_pos + 1, i + 1, since_punc);
                result += punc;
                last_punc_pos = i;
                continue;
            }

            // --- 长句保底 (≥20 字无标点, 在任何自然点断) ---
            if (since_punc >= 20 && is_natural_break(chars, i, n)) {
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
        int len = to - from;
        if (len <= 0) return false;

        // 常见句尾疑问语气词 (只检查末尾 3 字)
        static const char* q_tail[] = {
            "\xe5\x90\x97",     // 吗
            "\xe5\x91\xa2",     // 呢
            "\xe8\xb0\x81",     // 谁
        };
        for (auto q : q_tail) {
            for (int j = std::max(from, to - 3); j < to; ++j) {
                if (chars[j] == q) return true;
            }
        }

        // "怎么/怎样" 类疑问前缀: 只在短句 (≤8字) 或出现在末尾 6 字内时触发
        static const char* q_near_end[] = {
            "\xe4\xbb\x80\xe4\xb9\x88", // 什么
            "\xe6\x80\x8e\xe4\xb9\x88", // 怎么
            "\xe6\x80\x8e\xe6\xa0\xb7", // 怎样
            "\xe4\xb8\xba\xe4\xbb\x80\xe4\xb9\x88", // 为什么
        };
        // 只在末尾 6 字范围内搜索疑问前缀, 避免 "什么什么东西..." 误判
        int search_from = (len <= 8) ? from : std::max(from, to - 6);
        std::string tail_text;
        for (int i = search_from; i < to; ++i) tail_text += chars[i];
        for (auto q : q_near_end) {
            if (tail_text.find(q) != std::string::npos) return true;
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

    // ---- 语气词/填充词检测 ----
    static bool is_filler(const std::string& c) {
        static const char* fillers[] = {
            "\xe5\x97\xaf",     // 嗯
            "\xe5\x95\x8a",     // 啊
            "\xe5\xaf\xb9",     // 对
            "\xe5\x93\xa6",     // 哦
            "\xe5\x93\x88",     // 哈
            "\xe5\x91\xa2",     // 呢
            "\xe5\x90\xa7",     // 吧
            "\xe5\x91\x80",     // 呀
            "\xe5\x93\x8e",     // 哎
            "\xe5\x96\x94",     // 喔
            "\xe5\x97\xaf",     // 嗯
        };
        for (auto f : fillers) {
            if (c == f) return true;
        }
        return false;
    }

    // ---- 话语标记检测 (pos 是末字位置) ----
    static bool is_discourse_marker(const std::vector<std::string>& chars, int pos, int n) {
        if (pos + 1 >= n) return false;
        // "对不对" at pos-2..pos
        if (pos >= 2) {
            if (chars[pos-2] == "\xe5\xaf\xb9" && chars[pos-1] == "\xe4\xb8\x8d" && chars[pos] == "\xe5\xaf\xb9") return true;
            // "是不是"
            if (chars[pos-2] == "\xe6\x98\xaf" && chars[pos-1] == "\xe4\xb8\x8d" && chars[pos] == "\xe6\x98\xaf") return true;
        }
        // "是吧/对吧/好吧" at pos-1..pos
        if (pos >= 1) {
            if (chars[pos] == "\xe5\x90\xa7") {
                if (chars[pos-1] == "\xe6\x98\xaf" || chars[pos-1] == "\xe5\xaf\xb9" || chars[pos-1] == "\xe5\xa5\xbd") return true;
            }
            // "对呀/是呀"
            if (chars[pos] == "\xe5\x91\x80") {
                if (chars[pos-1] == "\xe5\xaf\xb9" || chars[pos-1] == "\xe6\x98\xaf") return true;
            }
        }
        return false;
    }

    // ---- 连接词前瞻检测 ----
    static bool is_conjunction_ahead(const std::vector<std::string>& chars, int pos, int n) {
        if (pos + 1 >= n) return false;
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
            {"\xe5\xb0\xb1", "\xe6\x98\xaf"},   // 就是
            {"\xe4\xb9\x9f", "\xe5\xb0\xb1"},   // 也就
            {"\xe5\x8f\xaf", "\xe6\x98\xaf"},   // 可是
            {"\xe5\x8f\xaf", "\xe8\x83\xbd"},   // 可能
        };
        for (auto& conj : conjunctions) {
            if (chars[pos] == conj[0] && chars[pos + 1] == conj[1]) return true;
        }
        return false;
    }

    // ---- 安全虚词断点 (避免 "的话" 等固定搭配) ----
    static bool is_safe_particle_break(const std::vector<std::string>& chars, int pos, int n) {
        // "了" 后断, 但不拆 "了解/了不起"
        if (chars[pos] == "\xe4\xba\x86") {
            if (pos + 1 < n) {
                // "了解/了不" → 不断
                if (chars[pos+1] == "\xe8\xa7\xa3" || chars[pos+1] == "\xe4\xb8\x8d") return false;
            }
            return true;
        }
        // "的" 后断, 但不拆 "的话/的时候/的人/的事/的命"
        if (chars[pos] == "\xe7\x9a\x84") {
            if (pos + 1 < n) {
                if (chars[pos+1] == "\xe8\xaf\x9d") return false;  // 的话
                if (chars[pos+1] == "\xe4\xba\xba") return false;  // 的人
                if (chars[pos+1] == "\xe4\xba\x8b") return false;  // 的事
                if (chars[pos+1] == "\xe5\x91\xbd") return false;  // 的命
            }
            if (pos + 2 < n) {
                if (chars[pos+1] == "\xe6\x97\xb6" && chars[pos+2] == "\xe5\x80\x99") return false;  // 的时候
            }
            return true;
        }
        // "过" 后断, 但不拆 "过来/过去/过程"
        if (chars[pos] == "\xe8\xbf\x87") {
            if (pos + 1 < n) {
                if (chars[pos+1] == "\xe6\x9d\xa5" || chars[pos+1] == "\xe5\x8e\xbb" || chars[pos+1] == "\xe7\xa8\x8b") return false;
            }
            return true;
        }
        // "嘛" 后断
        if (chars[pos] == "\xe5\x98\x9b") return true;
        return false;
    }

    // ---- 子句末尾标点选择 ----
    // 根据子句内容选择 ？/。/，
    //   from: 子句起始 (含), to: 子句结束 (不含), since_punc: 距上一标点字数
    static std::string select_clause_end(const std::vector<std::string>& chars,
                                          int from, int to, int since_punc) {
        // 1. 子句是纯填充词 (嗯/哦/啊, 1-2字) → 。
        if (since_punc <= 2) {
            bool all_filler = true;
            for (int j = from; j < to; ++j) {
                if (!is_filler(chars[j])) { all_filler = false; break; }
            }
            if (all_filler) return "\xe3\x80\x82"; // 。
        }

        // 2. 子句含疑问词 → ？
        if (has_question_marker(chars, from, to)) {
            return "\xef\xbc\x9f"; // ？
        }

        // 3. 子句已够长(≥8字)且末字为了/的/吧/嘛/呢 → 。(完整句)
        if (since_punc >= 8) {
            // 末字 (to-1) 如果是语气词, 检查是否表示完整句
            const std::string& last = chars[to - 1];
            if (last == "\xe4\xba\x86" ||   // 了
                last == "\xe7\x9a\x84" ||   // 的
                last == "\xe5\x98\x9b" ||   // 嘛
                last == "\xe5\x91\xa2" ||   // 呢
                last == "\xe5\x90\xa7") {   // 吧
                return "\xe3\x80\x82"; // 。
            }
        }

        // 4. 默认: 逗号
        return "\xef\xbc\x8c"; // ，
    }

    // ---- 自然断点检测 (20字保底用, 宽松匹配) ----
    static bool is_natural_break(const std::vector<std::string>& chars,
                                  int pos, int n) {
        if (pos + 1 >= n) return false;

        // 连接词前
        if (is_conjunction_ahead(chars, pos + 1, n)) return true;
        // 语气词后
        if (is_filler(chars[pos])) return true;
        // 安全虚词断点
        if (is_safe_particle_break(chars, pos, n)) return true;

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
