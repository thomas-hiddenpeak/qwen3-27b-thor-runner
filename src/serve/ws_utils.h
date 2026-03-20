// ws_utils.h — WebSocket / JSON 工具函数 (共享于 serve.cpp 和 voice_session.cpp)
//
// 所有函数定义为 inline, 可在多个 TU 中包含而不违反 ODR。

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <sys/socket.h>

namespace qwen_thor {
namespace serve {
namespace ws {

// ============================================================================
// WebSocket opcodes
// ============================================================================
static constexpr uint8_t OP_TEXT   = 0x01;
static constexpr uint8_t OP_BINARY = 0x02;
static constexpr uint8_t OP_CLOSE  = 0x08;
static constexpr uint8_t OP_PING   = 0x09;
static constexpr uint8_t OP_PONG   = 0x0A;

// ============================================================================
// JSON helpers
// ============================================================================
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

inline std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i+1]) {
                case '"':  out += '"';  i++; break;
                case '\\': out += '\\'; i++; break;
                case 'n':  out += '\n'; i++; break;
                case 't':  out += '\t'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case '/':  out += '/';  i++; break;
                case 'b':  out += '\b'; i++; break;
                case 'f':  out += '\f'; i++; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

inline std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    auto end = pos + 1;
    while (end < json.size()) {
        if (json[end] == '"') {
            size_t bs = 0;
            while (end - 1 - bs > pos && json[end - 1 - bs] == '\\') bs++;
            if (bs % 2 == 0) break;
        }
        end++;
    }
    return json_unescape(json.substr(pos + 1, end - pos - 1));
}

inline double json_get_number(const std::string& json, const std::string& key, double def = 0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    try { return std::stod(json.substr(pos)); } catch (...) { return def; }
}

inline bool json_get_bool(const std::string& json, const std::string& key, bool def = false) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    auto rest = json.substr(pos + 1, 10);
    if (rest.find("true") != std::string::npos) return true;
    if (rest.find("false") != std::string::npos) return false;
    return def;
}

inline int json_get_int(const std::string& json, const std::string& key, int def = 0) {
    return (int)json_get_number(json, key, def);
}

// ============================================================================
// 低级 I/O
// ============================================================================
inline bool send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::send(fd, p, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        p += n;
        remaining -= (size_t)n;
    }
    return true;
}

inline bool send_frame(int fd, uint8_t opcode, const uint8_t* data, size_t len) {
    uint8_t header[10];
    size_t hlen = 2;
    header[0] = 0x80 | opcode;
    if (len < 126) {
        header[1] = (uint8_t)len;
    } else if (len < 65536) {
        header[1] = 126;
        header[2] = (uint8_t)(len >> 8);
        header[3] = (uint8_t)(len & 0xFF);
        hlen = 4;
    } else {
        header[1] = 127;
        for (int i = 0; i < 8; i++)
            header[2 + i] = (uint8_t)(len >> ((7 - i) * 8));
        hlen = 10;
    }
    if (!send_all(fd, header, hlen)) return false;
    if (len > 0 && !send_all(fd, data, len)) return false;
    return true;
}

inline bool send_text(int fd, const std::string& text) {
    return send_frame(fd, OP_TEXT, (const uint8_t*)text.data(), text.size());
}

inline bool send_binary(int fd, const uint8_t* data, size_t len) {
    return send_frame(fd, OP_BINARY, data, len);
}

inline bool recv_frame(int fd, uint8_t& opcode, std::vector<uint8_t>& payload) {
    payload.clear();
    uint8_t hdr[2];
    if (recv(fd, hdr, 2, MSG_WAITALL) != 2) return false;

    opcode = hdr[0] & 0x0F;
    bool masked = (hdr[1] & 0x80) != 0;
    uint64_t plen = hdr[1] & 0x7F;

    if (plen == 126) {
        uint8_t ext[2];
        if (recv(fd, ext, 2, MSG_WAITALL) != 2) return false;
        plen = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (plen == 127) {
        uint8_t ext[8];
        if (recv(fd, ext, 8, MSG_WAITALL) != 8) return false;
        plen = 0;
        for (int i = 0; i < 8; i++) plen = (plen << 8) | ext[i];
    }

    if (plen > 64 * 1024 * 1024) return false;

    uint8_t mask_key[4] = {};
    if (masked) {
        if (recv(fd, mask_key, 4, MSG_WAITALL) != 4) return false;
    }

    payload.resize((size_t)plen);
    if (plen > 0) {
        size_t received = 0;
        while (received < plen) {
            ssize_t n = recv(fd, payload.data() + received, plen - received, 0);
            if (n <= 0) return false;
            received += n;
        }
        if (masked) {
            for (size_t i = 0; i < plen; i++)
                payload[i] ^= mask_key[i % 4];
        }
    }
    return true;
}

// ============================================================================
// Domain helpers
// ============================================================================

// 默认语音系统提示词
inline const char* default_voice_system_prompt() {
    return "你是一个语音助手，回答将通过语音播放。\n"
           "\n"
           "【规则】\n"
           "1. 不用 Markdown、特殊符号，数字用中文读法";
}

// 判断句末标点
inline bool is_sentence_end_punct(const std::string& ch) {
    return ch == "。" || ch == "！" || ch == "？" ||
           ch == "." || ch == "!" || ch == "?" || ch == "\n";
}

// 提取 [情感标注] → (clean_text, emotion)
inline std::pair<std::string, std::string> extract_tts_instruct(const std::string& text) {
    if (text.empty()) return {"", ""};

    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n')) start++;

    bool found = false;
    size_t tag_start = start;
    size_t tag_end = std::string::npos;

    // ASCII [ ]
    if (start < text.size() && text[start] == '[') {
        tag_end = text.find(']', start + 1);
        if (tag_end != std::string::npos && tag_end - start < 100) {
            found = true;
        }
    }
    // 中文【】
    else if (start + 2 < text.size() &&
             (unsigned char)text[start] == 0xE3 &&
             (unsigned char)text[start+1] == 0x80 &&
             (unsigned char)text[start+2] == 0x90) {
        for (size_t i = start + 3; i + 2 < text.size() && i < start + 100; i++) {
            if ((unsigned char)text[i] == 0xE3 &&
                (unsigned char)text[i+1] == 0x80 &&
                (unsigned char)text[i+2] == 0x91) {
                tag_end = i + 2;
                found = true;
                break;
            }
        }
        if (found) {
            std::string emotion = text.substr(start + 3, tag_end - 2 - (start + 3));
            std::string clean = text.substr(tag_end + 1);
            size_t cs = 0;
            while (cs < clean.size() && clean[cs] == ' ') cs++;
            if (cs > 0) clean = clean.substr(cs);
            if (emotion.empty()) return {clean, ""};
            return {clean, emotion};
        }
    }

    if (!found) return {text, ""};

    // ASCII [] case
    std::string emotion = text.substr(tag_start + 1, tag_end - tag_start - 1);
    std::string clean = text.substr(tag_end + 1);
    size_t cs = 0;
    while (cs < clean.size() && clean[cs] == ' ') cs++;
    if (cs > 0) clean = clean.substr(cs);
    if (emotion.empty()) return {clean, ""};
    return {clean, emotion};
}

// Base64 decode (for audio event in Voice protocol)
inline std::vector<uint8_t> base64_decode(const std::string& input) {
    static const uint8_t b64_table[256] = {
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,62,64,64,64,63,
        52,53,54,55,56,57,58,59,60,61,64,64,64, 0,64,64,
        64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,64,64,64,64,64,
        64,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,
        64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64
    };
    std::vector<uint8_t> out;
    out.reserve(input.size() * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (char c : input) {
        if (c == '\n' || c == '\r' || c == ' ') continue;
        if (c == '=') break;
        uint8_t val = b64_table[(uint8_t)c];
        if (val >= 64) continue;
        accum = (accum << 6) | val;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)((accum >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace ws
} // namespace serve
} // namespace qwen_thor
