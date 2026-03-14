// keyword_spotter.h — 关键词识别 (Phase 3)
//
// 基于 ASR 文本输出的流式/离线关键词匹配, 零额外模型。
// 支持:
//   - 精确文本子串匹配 (UTF-8)
//   - 模糊匹配 (编辑距离 ≤ 1)
//   - 流式 token 前缀匹配
//   - Aho-Corasick 多模式匹配 (关键词 > 20 个)
//
// 配置文件格式 (JSON):
//   { "keywords": [
//       {"text": "你好小助手", "threshold": 0.8, "action": "wake"},
//       ...
//   ]}

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <functional>

namespace qwen_thor {
namespace asr {

struct KeywordEntry {
    std::string text;            // 关键词文本
    std::vector<int> token_ids;  // 分词后 token IDs (可选)
    float threshold = 0.8f;      // 匹配置信度门限
    std::string action;          // 触发动作: "wake" / "stop" / "custom"
};

struct KeywordHit {
    std::string keyword;
    std::string action;
    int   char_offset = 0;       // 在文本中的 UTF-8 字符偏移
    float confidence  = 1.0f;
};

// ============================================================================
// KeywordSpotter — 多模式关键词匹配
// ============================================================================
class KeywordSpotter {
public:
    KeywordSpotter() = default;

    // 从 JSON 配置文件加载关键词 (见文件头格式)
    bool load_config(const std::string& json_path) {
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        return parse_json(content);
    }

    // 运行时添加关键词
    void add_keyword(const KeywordEntry& kw) {
        keywords_.push_back(kw);
        ac_dirty_ = true;
    }

    // 运行时移除关键词
    void remove_keyword(const std::string& text) {
        keywords_.erase(
            std::remove_if(keywords_.begin(), keywords_.end(),
                           [&](const KeywordEntry& k) { return k.text == text; }),
            keywords_.end());
        ac_dirty_ = true;
    }

    // 获取已注册的关键词列表
    const std::vector<KeywordEntry>& keywords() const { return keywords_; }

    // ASR 输出文本匹配 (整句)
    std::vector<KeywordHit> match(const std::string& asr_text) {
        if (keywords_.empty()) return {};

        // 关键词 ≤ 20: 简单子串搜索 + 模糊匹配
        if (keywords_.size() <= 20) {
            return match_simple(asr_text);
        }

        // 关键词 > 20: Aho-Corasick
        if (ac_dirty_) build_aho_corasick();
        return match_aho_corasick(asr_text);
    }

    // 流式 token 匹配: ASR decoder 每步输出后调用
    // 返回当前步骤触发的关键词命中 (如有)
    std::vector<KeywordHit> on_token(int token_id, const std::string& decoded_text) {
        stream_buffer_ += decoded_text;
        auto hits = match(stream_buffer_);

        // 对已触发的关键词去重 (避免重复触发)
        std::vector<KeywordHit> new_hits;
        for (auto& h : hits) {
            std::string key = h.keyword + "@" + std::to_string(h.char_offset);
            if (triggered_.find(key) == triggered_.end()) {
                triggered_[key] = true;
                new_hits.push_back(h);
            }
        }
        return new_hits;
    }

    // 重置流式状态 (新一轮 ASR 开始时调用)
    void reset_stream() {
        stream_buffer_.clear();
        triggered_.clear();
    }

private:
    std::vector<KeywordEntry> keywords_;
    std::string stream_buffer_;
    std::unordered_map<std::string, bool> triggered_;

    // ---- Aho-Corasick ----
    struct AcNode {
        std::unordered_map<char, int> children;
        int fail = 0;
        int keyword_idx = -1;  // 匹配的关键词索引, -1 = 无
    };
    std::vector<AcNode> ac_nodes_;
    bool ac_dirty_ = true;

    // ---- UTF-8 工具 ----
    static int utf8_char_len(unsigned char c) {
        if (c < 0x80) return 1;
        if (c < 0xE0) return 2;
        if (c < 0xF0) return 3;
        return 4;
    }

    // UTF-8 字符偏移 → 字节偏移
    static int utf8_byte_offset_to_char(const std::string& s, int byte_off) {
        int chars = 0;
        for (int i = 0; i < byte_off && i < (int)s.size(); ) {
            i += utf8_char_len((unsigned char)s[i]);
            chars++;
        }
        return chars;
    }

    // ---- 简单子串匹配 ----
    std::vector<KeywordHit> match_simple(const std::string& text) {
        std::vector<KeywordHit> hits;
        for (const auto& kw : keywords_) {
            // 精确子串匹配
            size_t pos = 0;
            while ((pos = text.find(kw.text, pos)) != std::string::npos) {
                KeywordHit hit;
                hit.keyword = kw.text;
                hit.action = kw.action;
                hit.char_offset = utf8_byte_offset_to_char(text, (int)pos);
                hit.confidence = 1.0f;
                hits.push_back(hit);
                pos += kw.text.size();
            }
            // 模糊匹配 (编辑距离 ≤ 1, 仅在精确匹配无结果时)
            if (hits.empty() && kw.threshold < 1.0f) {
                auto fuzzy = match_fuzzy(text, kw);
                hits.insert(hits.end(), fuzzy.begin(), fuzzy.end());
            }
        }
        return hits;
    }

    // ---- 模糊匹配 (滑动窗口 + 编辑距离) ----
    std::vector<KeywordHit> match_fuzzy(const std::string& text, const KeywordEntry& kw) {
        std::vector<KeywordHit> hits;
        if (kw.text.empty() || text.empty()) return hits;

        // 遍历文本中与关键词等长的 UTF-8 子串
        std::vector<int> text_char_starts;
        for (int i = 0; i < (int)text.size(); ) {
            text_char_starts.push_back(i);
            i += utf8_char_len((unsigned char)text[i]);
        }

        // 关键词字符数
        int kw_chars = 0;
        for (int i = 0; i < (int)kw.text.size(); ) {
            i += utf8_char_len((unsigned char)kw.text[i]);
            kw_chars++;
        }

        // 滑动窗口: 取 kw_chars-1 ~ kw_chars+1 长度的子串比较
        for (int wlen = std::max(1, kw_chars - 1); wlen <= kw_chars + 1; ++wlen) {
            for (int start = 0; start + wlen <= (int)text_char_starts.size(); ++start) {
                int byte_start = text_char_starts[start];
                int byte_end = (start + wlen < (int)text_char_starts.size())
                              ? text_char_starts[start + wlen]
                              : (int)text.size();
                std::string window = text.substr(byte_start, byte_end - byte_start);

                int dist = edit_distance_utf8(window, kw.text);
                if (dist <= 1 && dist > 0) {
                    float conf = 1.0f - (float)dist / (float)kw_chars;
                    if (conf >= kw.threshold) {
                        KeywordHit hit;
                        hit.keyword = kw.text;
                        hit.action = kw.action;
                        hit.char_offset = start;
                        hit.confidence = conf;
                        hits.push_back(hit);
                    }
                }
            }
        }
        return hits;
    }

    // ---- UTF-8 编辑距离 ----
    static int edit_distance_utf8(const std::string& a, const std::string& b) {
        // 提取 UTF-8 字符
        std::vector<std::string> ca, cb;
        for (int i = 0; i < (int)a.size(); ) {
            int len = utf8_char_len((unsigned char)a[i]);
            ca.push_back(a.substr(i, len));
            i += len;
        }
        for (int i = 0; i < (int)b.size(); ) {
            int len = utf8_char_len((unsigned char)b[i]);
            cb.push_back(b.substr(i, len));
            i += len;
        }
        int m = (int)ca.size(), n = (int)cb.size();
        // 早退: 如果长度差 > 1, 编辑距离必然 > 1
        if (std::abs(m - n) > 1) return std::abs(m - n);

        std::vector<int> prev(n + 1), curr(n + 1);
        for (int j = 0; j <= n; ++j) prev[j] = j;
        for (int i = 1; i <= m; ++i) {
            curr[0] = i;
            for (int j = 1; j <= n; ++j) {
                int cost = (ca[i - 1] == cb[j - 1]) ? 0 : 1;
                curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
            }
            std::swap(prev, curr);
        }
        return prev[n];
    }

    // ---- Aho-Corasick 构建 ----
    void build_aho_corasick() {
        ac_nodes_.clear();
        ac_nodes_.push_back(AcNode{}); // root

        // 插入所有关键词
        for (int ki = 0; ki < (int)keywords_.size(); ++ki) {
            int cur = 0;
            for (char c : keywords_[ki].text) {
                if (ac_nodes_[cur].children.find(c) == ac_nodes_[cur].children.end()) {
                    ac_nodes_[cur].children[c] = (int)ac_nodes_.size();
                    ac_nodes_.push_back(AcNode{});
                }
                cur = ac_nodes_[cur].children[c];
            }
            ac_nodes_[cur].keyword_idx = ki;
        }

        // BFS 构建 fail 指针
        std::queue<int> q;
        for (auto& [c, child] : ac_nodes_[0].children) {
            ac_nodes_[child].fail = 0;
            q.push(child);
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& [c, v] : ac_nodes_[u].children) {
                int f = ac_nodes_[u].fail;
                while (f && ac_nodes_[f].children.find(c) == ac_nodes_[f].children.end())
                    f = ac_nodes_[f].fail;
                ac_nodes_[v].fail = (ac_nodes_[f].children.count(c) && ac_nodes_[f].children[c] != v)
                                   ? ac_nodes_[f].children[c] : 0;
                q.push(v);
            }
        }
        ac_dirty_ = false;
    }

    // ---- Aho-Corasick 匹配 ----
    std::vector<KeywordHit> match_aho_corasick(const std::string& text) {
        std::vector<KeywordHit> hits;
        int cur = 0;
        for (int i = 0; i < (int)text.size(); ++i) {
            char c = text[i];
            while (cur && ac_nodes_[cur].children.find(c) == ac_nodes_[cur].children.end())
                cur = ac_nodes_[cur].fail;
            if (ac_nodes_[cur].children.count(c))
                cur = ac_nodes_[cur].children[c];

            // 检查当前节点及 fail 链上的所有匹配
            for (int t = cur; t; t = ac_nodes_[t].fail) {
                if (ac_nodes_[t].keyword_idx >= 0) {
                    int ki = ac_nodes_[t].keyword_idx;
                    int byte_start = i - (int)keywords_[ki].text.size() + 1;
                    if (byte_start >= 0) {
                        KeywordHit hit;
                        hit.keyword = keywords_[ki].text;
                        hit.action = keywords_[ki].action;
                        hit.char_offset = utf8_byte_offset_to_char(text, byte_start);
                        hit.confidence = 1.0f;
                        hits.push_back(hit);
                    }
                }
            }
        }
        return hits;
    }

    // ---- 最简 JSON 解析 (仅解析 keywords 数组) ----
    bool parse_json(const std::string& json) {
        keywords_.clear();
        ac_dirty_ = true;

        // 找 "keywords" 数组
        size_t pos = json.find("\"keywords\"");
        if (pos == std::string::npos) return false;
        pos = json.find('[', pos);
        if (pos == std::string::npos) return false;

        // 解析每个 keyword 对象
        while (true) {
            size_t obj_start = json.find('{', pos);
            if (obj_start == std::string::npos) break;
            size_t obj_end = json.find('}', obj_start);
            if (obj_end == std::string::npos) break;

            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
            KeywordEntry kw;
            kw.text = extract_json_string(obj, "text");
            kw.action = extract_json_string(obj, "action");
            std::string thresh_str = extract_json_string(obj, "threshold");
            if (!thresh_str.empty()) kw.threshold = std::stof(thresh_str);
            else {
                // 尝试提取非字符串数值
                size_t t = obj.find("\"threshold\"");
                if (t != std::string::npos) {
                    t = obj.find(':', t);
                    if (t != std::string::npos) {
                        kw.threshold = std::stof(obj.substr(t + 1));
                    }
                }
            }

            if (!kw.text.empty()) {
                keywords_.push_back(kw);
            }

            pos = obj_end + 1;
            // 检查是否到数组末端
            size_t next_bracket = json.find(']', pos);
            size_t next_obj = json.find('{', pos);
            if (next_bracket != std::string::npos &&
                (next_obj == std::string::npos || next_bracket < next_obj))
                break;
        }
        return !keywords_.empty();
    }

    static std::string extract_json_string(const std::string& obj, const std::string& key) {
        std::string search = "\"" + key + "\"";
        size_t pos = obj.find(search);
        if (pos == std::string::npos) return "";
        pos = obj.find(':', pos + search.size());
        if (pos == std::string::npos) return "";
        pos = obj.find('"', pos + 1);
        if (pos == std::string::npos) return "";
        size_t end = obj.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return obj.substr(pos + 1, end - pos - 1);
    }
};

} // namespace asr
} // namespace qwen_thor
