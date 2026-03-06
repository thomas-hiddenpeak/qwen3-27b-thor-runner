// tokenizer.cpp — GPT2-style Byte-Level BPE Tokenizer 实现
//
// Qwen3.5-27B 使用 HuggingFace tokenizers 的 BPE 模型:
//   - vocab.json: token_string → id  (248044 entries)
//   - merges.txt: 合并规则列表       (247587 entries)
//   - 26 个 added/special tokens     (248044–248069)
//
// Byte-Level 编码: 每个字节映射为一个可打印 Unicode 字符 (GPT2 byte_encoder)
// 这样 vocab 中的 token 都是可打印字符串, 不包含控制字符或 NUL

#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>

namespace qwen_thor {

// ============================================================================
// UTF-8 工具
// ============================================================================

// 返回 UTF-8 字符的字节长度
static inline int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;  // continuation byte (invalid as start)
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// 将 Unicode codepoint 编码为 UTF-8
static std::string codepoint_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// 拆分 UTF-8 字符串为单个字符列表
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        if (i + len > s.size()) len = 1;  // safety
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// ============================================================================
// JSON 解析工具 (极简, 仅用于 vocab.json)
// ============================================================================

// 解析 JSON 字符串中的转义序列
static std::string unescape_json_string(const char* start, size_t len) {
    std::string result;
    result.reserve(len);
    for (size_t i = 0; i < len; i++) {
        if (start[i] == '\\' && i + 1 < len) {
            char next = start[i + 1];
            switch (next) {
                case '"':  result += '"';  i++; break;
                case '\\': result += '\\'; i++; break;
                case '/':  result += '/';  i++; break;
                case 'n':  result += '\n'; i++; break;
                case 'r':  result += '\r'; i++; break;
                case 't':  result += '\t'; i++; break;
                case 'b':  result += '\b'; i++; break;
                case 'f':  result += '\f'; i++; break;
                case 'u': {
                    // \uXXXX
                    if (i + 5 < len) {
                        char hex[5] = {start[i+2], start[i+3], start[i+4], start[i+5], 0};
                        int cp = (int)strtol(hex, nullptr, 16);
                        // Handle surrogate pairs
                        if (cp >= 0xD800 && cp <= 0xDBFF && i + 11 < len &&
                            start[i+6] == '\\' && start[i+7] == 'u') {
                            char hex2[5] = {start[i+8], start[i+9], start[i+10], start[i+11], 0};
                            int cp2 = (int)strtol(hex2, nullptr, 16);
                            if (cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                                i += 6;  // skip second \uXXXX
                            }
                        }
                        result += codepoint_to_utf8(cp);
                        i += 5;
                    }
                    break;
                }
                default: result += start[i]; break;
            }
        } else {
            result += start[i];
        }
    }
    return result;
}

// ============================================================================
// Tokenizer::init_byte_mapping
// ============================================================================

void Tokenizer::init_byte_mapping() {
    // GPT2 byte_encoder: 将每个 byte (0-255) 映射为一个 Unicode codepoint
    //
    // 可打印字节 (33-126, 161-172, 174-255) 直接映射为同值 codepoint
    // 其余字节 (0-32, 127-160, 173) 依次映射为 256, 257, ...
    //
    // 映射后的 codepoint 都是可打印的 Unicode 字符

    std::vector<int> bs, cs;

    // 直接映射的字节范围
    for (int b = 33; b <= 126; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 161; b <= 172; b++) { bs.push_back(b); cs.push_back(b); }
    for (int b = 174; b <= 255; b++) { bs.push_back(b); cs.push_back(b); }

    // 其余字节映射到 256+
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool found = false;
        for (int x : bs) {
            if (x == b) { found = true; break; }
        }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // 构建映射表
    for (size_t i = 0; i < bs.size(); i++) {
        int byte_val = bs[i];
        int codepoint = cs[i];
        std::string utf8 = codepoint_to_utf8(codepoint);
        byte_to_unicode_[byte_val] = utf8;
        unicode_to_byte_[utf8] = static_cast<uint8_t>(byte_val);
    }
}

// ============================================================================
// Tokenizer::byte_encode / byte_decode
// ============================================================================

std::string Tokenizer::byte_encode(const std::string& utf8_text) const {
    std::string result;
    for (unsigned char b : utf8_text) {
        result += byte_to_unicode_[b];
    }
    return result;
}

std::string Tokenizer::byte_decode(const std::string& piece) const {
    std::string result;
    size_t i = 0;
    while (i < piece.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(piece[i]));
        if (i + len > piece.size()) break;
        std::string ch = piece.substr(i, len);
        auto it = unicode_to_byte_.find(ch);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        }
        i += len;
    }
    return result;
}

// ============================================================================
// Tokenizer::pre_tokenize — 简化版 GPT2 Regex Split
// ============================================================================
//
// 原始正则: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//           |\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// C++ <regex> 不支持 \p{L} 等 Unicode 属性, 用手工状态机实现:
//   1. ASCII 字母序列 (含英语缩写 's, 't, 're 等)
//   2. 非 ASCII UTF-8 字符序列 (CJK/Kana/Cyrillic 等)
//   3. 单个数字
//   4. 标点/符号 (含可选前导空格)
//   5. 换行符 (含可选前导空白)
//   6. 空白序列

static bool is_ascii_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // --- 换行符 (含前导空白) --- 
        if (c == '\n' || c == '\r') {
            std::string piece;
            while (i < n && (text[i] == '\n' || text[i] == '\r')) {
                piece += text[i++];
            }
            pieces.push_back(piece);
            continue;
        }

        // --- 空白 (空格/制表符) ---
        if (c == ' ' || c == '\t') {
            // 空格可能附着到下一个 "词" 上
            // GPT2 pattern: 空格 + 字母序列 合为一个 piece
            if (i + 1 < n) {
                unsigned char next = static_cast<unsigned char>(text[i + 1]);
                // 空格 + ASCII 字母
                if (is_ascii_letter(next)) {
                    std::string piece;
                    piece += text[i++];  // space
                    while (i < n && is_ascii_letter(text[i])) {
                        piece += text[i++];
                    }
                    // 检查缩写 ('s, 't, etc.)
                    if (i < n && text[i] == '\'') {
                        pieces.push_back(piece);
                        // 缩写作为独立 piece
                        std::string contr = "'";
                        size_t j = i + 1;
                        while (j < n && is_ascii_letter(text[j]) && j - i <= 3) {
                            contr += text[j++];
                        }
                        std::string lower_suffix = contr.substr(1);
                        for (auto& ch : lower_suffix) ch = tolower(ch);
                        if (lower_suffix == "s" || lower_suffix == "t" || lower_suffix == "re" ||
                            lower_suffix == "ve" || lower_suffix == "m" || lower_suffix == "ll" ||
                            lower_suffix == "d") {
                            pieces.push_back(contr);
                            i = j;
                        }
                    } else {
                        pieces.push_back(piece);
                    }
                    continue;
                }
                // 空格 + 非 ASCII (UTF-8 字母)
                if (next >= 0x80) {
                    std::string piece;
                    piece += text[i++];  // space
                    while (i < n && static_cast<unsigned char>(text[i]) >= 0x80) {
                        int len = utf8_char_len(static_cast<unsigned char>(text[i]));
                        if (i + len > n) break;
                        piece.append(text, i, len);
                        i += len;
                    }
                    pieces.push_back(piece);
                    continue;
                }
                // 空格 + 数字: 空格独立
                if (isdigit(next)) {
                    pieces.push_back(std::string(1, text[i++]));
                    continue;
                }
                // 空格 + 标点: 合为一个 piece
                if (!isspace(next)) {
                    std::string piece;
                    piece += text[i++];  // space
                    // 收集标点字符
                    while (i < n && !isspace(text[i]) && !is_ascii_letter(text[i]) &&
                           !isdigit(static_cast<unsigned char>(text[i])) &&
                           static_cast<unsigned char>(text[i]) < 0x80) {
                        piece += text[i++];
                    }
                    pieces.push_back(piece);
                    continue;
                }
            }
            // 尾部空白或连续空格
            std::string piece;
            while (i < n && (text[i] == ' ' || text[i] == '\t')) {
                piece += text[i++];
            }
            pieces.push_back(piece);
            continue;
        }

        // --- 数字 (单个) ---
        if (isdigit(c)) {
            pieces.push_back(std::string(1, text[i++]));
            continue;
        }

        // --- ASCII 字母序列 ---
        if (is_ascii_letter(c)) {
            std::string piece;
            while (i < n && is_ascii_letter(text[i])) {
                piece += text[i++];
            }
            // 检查缩写
            if (i < n && text[i] == '\'') {
                std::string contr = "'";
                size_t j = i + 1;
                while (j < n && is_ascii_letter(text[j]) && j - i <= 3) {
                    contr += text[j++];
                }
                std::string lower_suffix = contr.substr(1);
                for (auto& ch : lower_suffix) ch = tolower(ch);
                if (lower_suffix == "s" || lower_suffix == "t" || lower_suffix == "re" ||
                    lower_suffix == "ve" || lower_suffix == "m" || lower_suffix == "ll" ||
                    lower_suffix == "d") {
                    pieces.push_back(piece);
                    pieces.push_back(contr);
                    i = j;
                    continue;
                }
            }
            pieces.push_back(piece);
            continue;
        }

        // --- 非 ASCII UTF-8 字符序列 (CJK、Kana、Emoji 等) ---
        if (c >= 0x80) {
            std::string piece;
            while (i < n && static_cast<unsigned char>(text[i]) >= 0x80) {
                int len = utf8_char_len(static_cast<unsigned char>(text[i]));
                if (i + len > n) break;
                piece.append(text, i, len);
                i += len;
            }
            if (!piece.empty()) pieces.push_back(piece);
            continue;
        }

        // --- 其他 ASCII 字符 (标点/符号) ---
        pieces.push_back(std::string(1, text[i++]));
    }

    return pieces;
}

// ============================================================================
// Tokenizer::bpe — 对单个 byte-encoded piece 执行 BPE 合并
// ============================================================================

std::vector<int> Tokenizer::bpe(const std::string& piece) const {
    // 1. 拆分为 UTF-8 字符列表 (每个 byte-level unicode 字符 = 一个 BPE symbol)
    std::vector<std::string> symbols = utf8_chars(piece);

    if (symbols.empty()) return {};

    // 单字符直接查 vocab
    if (symbols.size() == 1) {
        auto it = piece_to_id_.find(symbols[0]);
        if (it != piece_to_id_.end()) return {it->second};
        return {};  // unknown
    }

    // 2. 迭代合并
    while (symbols.size() > 1) {
        // 找到优先级最高 (rank 最小) 的相邻对
        int best_rank = INT_MAX;
        int best_idx = -1;

        for (size_t j = 0; j + 1 < symbols.size(); j++) {
            std::string pair_key = symbols[j] + " " + symbols[j + 1];
            auto it = merges_.find(pair_key);
            if (it != merges_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_idx < 0) break;  // 无更多可合并的对

        // 合并 best_idx 和 best_idx+1
        std::string merged = symbols[best_idx] + symbols[best_idx + 1];
        symbols[best_idx] = merged;
        symbols.erase(symbols.begin() + best_idx + 1);
    }

    // 3. 查 vocab
    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = piece_to_id_.find(sym);
        if (it != piece_to_id_.end()) {
            ids.push_back(it->second);
        }
        // 未知 symbol 被静默丢弃 (byte-level BPE 理论上不会出现 OOV)
    }
    return ids;
}

// ============================================================================
// Tokenizer::encode
// ============================================================================

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!loaded_ || text.empty()) return {};

    std::vector<int> all_ids;

    // 1. 检查是否包含特殊 token (贪心匹配: 按长度降序)
    //    在特殊 token 处拆分文本, 特殊 token 直接用其 ID
    struct Segment {
        std::string text;
        bool is_special;
        int special_id;
    };
    std::vector<Segment> segments;

    size_t pos = 0;
    while (pos < text.size()) {
        bool found_special = false;
        for (const auto& st : special_token_list_) {
            if (pos + st.size() <= text.size() &&
                text.compare(pos, st.size(), st) == 0) {
                // 先保存特殊 token 前的普通文本
                // (上一段的 text 已在之前的迭代中处理)
                segments.push_back({st, true, special_tokens_.at(st)});
                pos += st.size();
                found_special = true;
                break;
            }
        }
        if (!found_special) {
            // 收集到下一个特殊 token 或文本末尾
            size_t start = pos;
            size_t next_special = std::string::npos;
            for (const auto& st : special_token_list_) {
                size_t f = text.find(st, pos);
                if (f != std::string::npos && f < next_special) {
                    next_special = f;
                }
            }
            size_t end = (next_special != std::string::npos) ? next_special : text.size();
            if (end > start) {
                segments.push_back({text.substr(start, end - start), false, -1});
            }
            pos = end;
        }
    }

    // 2. 对每个 segment 编码
    for (const auto& seg : segments) {
        if (seg.is_special) {
            all_ids.push_back(seg.special_id);
            continue;
        }

        // 预分词 → byte encode → BPE
        auto pieces = pre_tokenize(seg.text);
        for (const auto& piece : pieces) {
            std::string encoded = byte_encode(piece);
            auto ids = bpe(encoded);
            all_ids.insert(all_ids.end(), ids.begin(), ids.end());
        }
    }

    return all_ids;
}

// ============================================================================
// Tokenizer::decode
// ============================================================================

std::string Tokenizer::decode(int token_id) const {
    if (!loaded_) return "";
    if (token_id < 0 || token_id >= static_cast<int>(id_to_piece_.size())) return "";

    // 特殊 token 直接返回原文
    const std::string& piece = id_to_piece_[token_id];
    // 检查是否是特殊 token
    if (special_tokens_.count(piece) > 0) return piece;

    return byte_decode(piece);
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    for (int id : token_ids) {
        result += decode(id);
    }
    return result;
}

// ============================================================================
// Tokenizer::apply_chat_template
// ============================================================================
//
// Qwen3 ChatML 格式:
//   <|im_start|>system
//   You are a helpful assistant.<|im_end|>
//   <|im_start|>user
//   Hello!<|im_end|>
//   <|im_start|>assistant
//   ...

std::vector<int> Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt,
    bool enable_thinking) const
{
    std::string prompt;
    for (const auto& [role, content] : messages) {
        if (role == "tool") {
            // Qwen 官方格式: tool response 作为 user 消息, 包裹在 <tool_response> 标签中
            prompt += "<|im_start|>user\n<tool_response>\n" + content + "\n</tool_response><|im_end|>\n";
        } else {
            prompt += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) {
        if (enable_thinking) {
            // Qwen3.5 thinking 模式: assistant 后跟 <think>\n
            prompt += "<|im_start|>assistant\n<think>\n";
        } else {
            // 非 thinking 模式: 空 <think></think> 跳过思考
            prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }
    }
    return encode(prompt);
}

// ============================================================================
// Tokenizer::load
// ============================================================================

bool Tokenizer::load(const std::string& model_dir) {
    init_byte_mapping();

    std::string vocab_path  = model_dir + "/vocab.json";
    std::string merges_path = model_dir + "/merges.txt";

    if (!load_vocab(vocab_path)) {
        std::cerr << "[Tokenizer] Failed to load vocab: " << vocab_path << std::endl;
        return false;
    }
    if (!load_merges(merges_path)) {
        std::cerr << "[Tokenizer] Failed to load merges: " << merges_path << std::endl;
        return false;
    }

    // 注册特殊 tokens (Qwen3.5 added_tokens)
    // 从 vocab 中查找, 如果 vocab.json 不包含它们则手动赋值
    struct SpecialDef {
        const char* content;
        int fallback_id;
    };
    SpecialDef specials[] = {
        {"<|endoftext|>",         248044},
        {"<|im_start|>",          248045},
        {"<|im_end|>",            248046},
        {"<|object_ref_start|>",  248047},
        {"<|object_ref_end|>",    248048},
        {"<|box_start|>",         248049},
        {"<|box_end|>",           248050},
        {"<|quad_start|>",        248051},
        {"<|quad_end|>",          248052},
        {"<|vision_start|>",      248053},
        {"<|vision_end|>",        248054},
        {"<|vision_pad|>",        248055},
        {"<|image_pad|>",         248056},
        {"<|video_pad|>",         248057},
        {"<tool_call>",           248058},
        {"</tool_call>",          248059},
        {"<|fim_prefix|>",        248060},
        {"<|fim_middle|>",        248061},
        {"<|fim_suffix|>",        248062},
        {"<|fim_pad|>",           248063},
        {"<|repo_name|>",         248064},
        {"<|file_sep|>",          248065},
        {"<tool_response>",       248066},
        {"</tool_response>",      248067},
        {"<think>",               248068},
        {"</think>",              248069},
    };

    for (const auto& sp : specials) {
        auto it = piece_to_id_.find(sp.content);
        int id = (it != piece_to_id_.end()) ? it->second : sp.fallback_id;

        special_tokens_[sp.content] = id;
        special_token_list_.push_back(sp.content);

        // 确保 id_to_piece_ 能覆盖
        if (id >= static_cast<int>(id_to_piece_.size())) {
            id_to_piece_.resize(id + 1);
        }
        id_to_piece_[id] = sp.content;
        piece_to_id_[sp.content] = id;
    }

    // 按长度降序排列 special_token_list_ (贪心匹配用)
    std::sort(special_token_list_.begin(), special_token_list_.end(),
              [](const std::string& a, const std::string& b) {
                  return a.size() > b.size();
              });

    // 赋值特殊 ID
    eot_id_        = special_tokens_["<|endoftext|>"];
    im_start_id_   = special_tokens_["<|im_start|>"];
    im_end_id_     = special_tokens_["<|im_end|>"];
    eos_id_        = im_end_id_;  // Qwen3 uses <|im_end|> as EOS
    think_start_id_= special_tokens_["<think>"];
    think_end_id_  = special_tokens_["</think>"];
    tool_call_start_id_ = special_tokens_["<tool_call>"];
    tool_call_end_id_   = special_tokens_["</tool_call>"];

    loaded_ = true;
    fprintf(stderr, "[Tokenizer] Loaded: vocab=%d, merges=%d, specials=%d\n",
           static_cast<int>(piece_to_id_.size()),
           static_cast<int>(merges_.size()),
           static_cast<int>(special_tokens_.size()));
    return true;
}

// ============================================================================
// Tokenizer::load_vocab — 解析 vocab.json
// ============================================================================
//
// 格式: {"token_string": id, "token_string2": id2, ...}
// 文件 ~6.4 MB, 248044 条目

bool Tokenizer::load_vocab(const std::string& path) {
    // 读取整个文件
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    ifs.seekg(0, std::ios::end);
    size_t file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::string content(file_size, '\0');
    ifs.read(&content[0], file_size);

    // 找到最大 id 来预分配 id_to_piece_
    int max_id = 0;

    // 快速扫描式解析: 遍历 JSON 找 "key": value 对
    const char* p = content.c_str();
    const char* end = p + content.size();

    // 跳过开头的 {
    while (p < end && *p != '{') p++;
    if (p < end) p++;

    while (p < end) {
        // 跳过空白和逗号
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ','))
            p++;
        if (p >= end || *p == '}') break;

        // 读取 key (JSON 字符串)
        if (*p != '"') { p++; continue; }
        p++;  // skip opening "

        const char* key_start = p;
        // 找到 closing ", 正确处理转义 (包括 \\", \\\\" 等)
        // 规则: " 前面有偶数个 (含 0 个) 反斜杠 → 真正的 closing quote
        //        " 前面有奇数个反斜杠 → 被转义的 quote, 继续
        while (p < end) {
            if (*p == '"') {
                int bs_count = 0;
                const char* bp = p - 1;
                while (bp >= key_start && *bp == '\\') { bs_count++; bp--; }
                if (bs_count % 2 == 0) break;  // 偶数个 \ → 真 closing "
            }
            p++;
        }
        const char* key_end = p;
        size_t key_len = key_end - key_start;
        std::string key = unescape_json_string(key_start, key_len);
        if (p < end) p++;  // skip closing "

        // 跳过 :
        while (p < end && *p != ':') p++;
        if (p < end) p++;

        // 跳过空白
        while (p < end && (*p == ' ' || *p == '\t')) p++;

        // 读取 value (整数)
        int value = 0;
        bool negative = false;
        if (p < end && *p == '-') { negative = true; p++; }
        while (p < end && *p >= '0' && *p <= '9') {
            value = value * 10 + (*p - '0');
            p++;
        }
        if (negative) value = -value;

        piece_to_id_[key] = value;
        if (value > max_id) max_id = value;
    }

    // 构建 id → piece 反向映射
    id_to_piece_.resize(max_id + 1);
    for (const auto& [piece, id] : piece_to_id_) {
        if (id >= 0 && id < static_cast<int>(id_to_piece_.size())) {
            id_to_piece_[id] = piece;
        }
    }

    return !piece_to_id_.empty();
}

// ============================================================================
// Tokenizer::load_merges — 解析 merges.txt
// ============================================================================
//
// 格式:
//   #version: 0.2
//   Ġ Ġ
//   ĠĠ ĠĠ
//   i n
//   ...
// 每行: piece1<space>piece2   (空格分隔两个 BPE 片段)
// 行号 (从 0 开始, 跳过 #header) = merge rank

bool Tokenizer::load_merges(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return false;

    std::string line;
    int rank = 0;
    while (std::getline(ifs, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;

        // 找到第一个空格 (分隔 piece1 和 piece2)
        // 注意: piece 本身不含空格 (byte-level BPE 中空格被编码为 Ġ)
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;

        // merge key = "piece1 piece2" (与 BPE 合并时的查找 key 一致)
        merges_[line] = rank++;
    }

    return !merges_.empty();
}

} // namespace qwen_thor
