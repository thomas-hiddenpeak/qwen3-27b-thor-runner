// serve.cpp — HTTP API 服务实现
//
// 轻量级 POSIX socket HTTP 服务, 无外部依赖。
// 支持 OpenAI / Ollama 兼容 API 端点。
// WebSocket 支持: RFC 6455 帧协议, /v1/voice 语音对话端点。

#include "serve.h"
#include "../engine/vision.h"
#include "../plugins/asr/audio_utils.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <thread>
#include <future>
#include <queue>
#include <random>
#include <cctype>
#include <map>
#include <set>
#include <filesystem>

// stb_image for decoding JPEG/PNG
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "../../third_party/stb/stb_image.h"

// malloc_trim removed — glibc global arena lock risks systemd WDT timeout
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>

namespace qwen_thor {
namespace serve {

// ============================================================================
// Simple JSON helpers (no external dependency)
// ============================================================================

static std::string json_escape(const std::string& s) {
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
                    // Escape control chars as \u00XX
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

// Forward declaration
static std::string json_unescape(const std::string& s);

// Minimal JSON value extraction (no full parser, just enough for API)
static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    // Scan for unescaped closing quote — count preceding backslashes
    auto end = pos + 1;
    while (end < json.size()) {
        if (json[end] == '"') {
            // Count consecutive backslashes before this quote
            size_t bs = 0;
            while (end - 1 - bs > pos && json[end - 1 - bs] == '\\') bs++;
            if (bs % 2 == 0) break;  // Even backslashes → quote is unescaped
        }
        end++;
    }
    return json_unescape(json.substr(pos + 1, end - pos - 1));
}

static double json_get_number(const std::string& json, const std::string& key, double def = 0) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;

    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;
    try { return std::stod(json.substr(pos)); } catch (...) { return def; }
}

static bool json_get_bool(const std::string& json, const std::string& key, bool def = false) {
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

static int json_get_int(const std::string& json, const std::string& key, int def = 0) {
    return (int)json_get_number(json, key, def);
}

static int clamp_max_output_tokens(int requested, int cap) {
    int safe_cap = std::max(1, cap);
    if (requested <= 0) return safe_cap;
    return std::min(requested, safe_cap);
}

// 默认 system prompt — 使用模型官方推荐, 双语确保中英文都能正常工作
static const char* DEFAULT_SYSTEM_PROMPT =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// 语音对话默认 system prompt — 引导 LLM 输出适合 TTS 的文本
// 不含情感标注 (只有 CustomVoice 需要, 通过 config 的 voice_system_prompt 配置)
static const char* DEFAULT_VOICE_SYSTEM_PROMPT =
    "你是一个语音助手，回答将通过语音播放。\n"
    "\n"
    "【规则】\n"
    "1. 不用 Markdown、特殊符号，数字用中文读法";

// 从 LLM 输出的文本中提取 [情感标注] 并返回 (clean_text, emotion)
// 例: "[温柔]你好啊" → ("你好啊", "温柔")
// 例: "你好啊" → ("你好啊", "")
static std::pair<std::string, std::string> extract_tts_instruct(const std::string& text) {
    if (text.empty()) return {"", ""};

    // 查找开头的 [xxx] 标注 (跳过前导空白)
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n')) start++;

    // 检查是否以 [ 开头 (UTF-8: '[' = 0x5B, 中文'[' = E3 80 90)
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
        // 搜索 】(E3 80 91)
        for (size_t i = start + 3; i + 2 < text.size() && i < start + 100; i++) {
            if ((unsigned char)text[i] == 0xE3 &&
                (unsigned char)text[i+1] == 0x80 &&
                (unsigned char)text[i+2] == 0x91) {
                tag_end = i + 2;  // 指向 】 的最后一个字节
                found = true;
                break;
            }
        }
        if (found) {
            // 提取标注内容 (去掉【】)
            std::string emotion = text.substr(start + 3, tag_end - 2 - (start + 3));
            std::string clean = text.substr(tag_end + 1);
            // 去掉 clean 前导空格
            size_t cs = 0;
            while (cs < clean.size() && clean[cs] == ' ') cs++;
            if (cs > 0) clean = clean.substr(cs);
            if (emotion.empty()) return {clean, ""};
            return {clean, emotion};
        }
    }

    if (!found) return {text, ""};

    // ASCII [] 情况: 提取标注内容
    std::string emotion = text.substr(tag_start + 1, tag_end - tag_start - 1);
    std::string clean = text.substr(tag_end + 1);
    // 去掉 clean 前导空格
    size_t cs = 0;
    while (cs < clean.size() && clean[cs] == ' ') cs++;
    if (cs > 0) clean = clean.substr(cs);
    if (emotion.empty()) return {clean, ""};
    return {clean, emotion};
}

// Unescape JSON string value (handles \", \\, \n, \t, \r, \/)
static std::string json_unescape(const std::string& s) {
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

// Extract raw JSON value between balanced braces starting at pos
// Returns the substring {....} and advances pos past it
static std::string extract_json_object(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '{') return "";
    size_t start = pos;
    int depth = 1;
    pos++;
    while (pos < s.size() && depth > 0) {
        if (s[pos] == '{') depth++;
        else if (s[pos] == '}') depth--;
        else if (s[pos] == '"') {
            pos++;
            while (pos < s.size()) {
                if (s[pos] == '"') {
                    int bs = 0;
                    size_t bp = pos;
                    while (bp > start && s[bp-1] == '\\') { bs++; bp--; }
                    if (bs % 2 == 0) break;
                }
                pos++;
            }
        }
        pos++;
    }
    return s.substr(start, pos - start);
}

static std::string extract_json_array(const std::string& s, size_t& pos) {
    if (pos >= s.size() || s[pos] != '[') return "";
    size_t start = pos;
    int depth = 1;
    pos++;
    while (pos < s.size() && depth > 0) {
        if (s[pos] == '[') depth++;
        else if (s[pos] == ']') depth--;
        else if (s[pos] == '"') {
            pos++;
            while (pos < s.size()) {
                if (s[pos] == '"') {
                    int bs = 0;
                    size_t bp = pos;
                    while (bp > start && s[bp - 1] == '\\') { bs++; bp--; }
                    if (bs % 2 == 0) break;
                }
                pos++;
            }
        }
        pos++;
    }
    return s.substr(start, pos - start);
}

static std::string extract_json_raw_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '{') {
        size_t p = pos;
        return extract_json_object(json, p);
    }
    if (json[pos] == '[') {
        size_t p = pos;
        return extract_json_array(json, p);
    }
    if (json[pos] == '"') {
        size_t p = pos + 1;
        while (p < json.size()) {
            if (json[p] == '"') {
                int bs = 0;
                size_t bp = p;
                while (bp > pos && json[bp - 1] == '\\') { bs++; bp--; }
                if (bs % 2 == 0) break;
            }
            p++;
        }
        if (p < json.size()) return json.substr(pos, p - pos + 1);
        return "";
    }

    size_t end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != ']') end++;
    return json.substr(pos, end - pos);
}

// Extract "tools" array from request body as raw JSON string
static std::string extract_tools_json(const std::string& body) {
    auto pos = body.find("\"tools\"");
    if (pos == std::string::npos) return "";
    auto arr_start = body.find('[', pos);
    if (arr_start == std::string::npos) return "";

    // Find matching ]
    int depth = 1;
    size_t end = arr_start + 1;
    while (end < body.size() && depth > 0) {
        if (body[end] == '[') depth++;
        else if (body[end] == ']') depth--;
        else if (body[end] == '"') {
            end++;
            while (end < body.size()) {
                if (body[end] == '"') {
                    int bs = 0;
                    size_t bp = end;
                    while (bp > arr_start && body[bp-1] == '\\') { bs++; bp--; }
                    if (bs % 2 == 0) break;
                }
                end++;
            }
        }
        end++;
    }
    return body.substr(arr_start, end - arr_start);
}

// Build Qwen-format tool system prompt from OpenAI tools JSON array
static std::string build_tool_system_prompt(const std::string& tools_json) {
    if (tools_json.empty()) return "";

    std::string tool_lines;
    size_t p = 1; // skip [
    while (p < tools_json.size()) {
        while (p < tools_json.size() && (tools_json[p] == ' ' || tools_json[p] == ','
               || tools_json[p] == '\n' || tools_json[p] == '\r' || tools_json[p] == '\t'))
            p++;
        if (p >= tools_json.size() || tools_json[p] == ']') break;
        if (tools_json[p] == '{') {
            std::string obj = extract_json_object(tools_json, p);
            tool_lines += "\n" + obj;
        } else {
            p++;
        }
    }

    return "\n\n# Tools\n\n"
           "You may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools></tools> XML tags:\n"
           "<tools>" + tool_lines + "\n</tools>\n\n"
           "For each function call, return a json object with function name and arguments "
           "within <tool_call></tool_call> XML tags:\n"
           "<tool_call>\n"
           "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
           "</tool_call>";
}

// ============================================================================
// Base64 Decoder + Image Loading (for multimodal API)
// ============================================================================

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

static std::vector<uint8_t> base64_decode(const std::string& input) {
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

// Decode image from base64 data URI or raw base64 string
// Returns ImageData with RGB pixels, or empty on failure
// In-place base64 decode: operates directly from data_uri offset, avoiding 15MB substr copy
static std::vector<uint8_t> base64_decode_from(const std::string& input, size_t offset) {
    std::vector<uint8_t> out;
    out.reserve((input.size() - offset) * 3 / 4);
    uint32_t accum = 0;
    int bits = 0;
    for (size_t i = offset; i < input.size(); ++i) {
        char c = input[i];
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

// ============================================================================
// Base64 encode (for WebSocket handshake)
// ============================================================================
static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = ((uint32_t)data[i]) << 16;
        if (i + 1 < len) n |= ((uint32_t)data[i+1]) << 8;
        if (i + 2 < len) n |= ((uint32_t)data[i+2]);
        out += b64_chars[(n >> 18) & 63];
        out += b64_chars[(n >> 12) & 63];
        out += (i + 1 < len) ? b64_chars[(n >> 6) & 63] : '=';
        out += (i + 2 < len) ? b64_chars[n & 63] : '=';
    }
    return out;
}

// ============================================================================
// Minimal SHA-1 (RFC 3174) — WebSocket 握手专用
// ============================================================================
static void sha1_compute(const uint8_t* data, size_t len, uint8_t hash[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
    // Pre-processing: pad message
    size_t msg_len = len + 1 + 8;  // original + 0x80 + 8 bytes length
    size_t padded_len = ((msg_len + 63) / 64) * 64;
    std::vector<uint8_t> msg(padded_len, 0);
    memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) msg[padded_len - 1 - i] = (uint8_t)(bit_len >> (i * 8));

    auto left_rotate = [](uint32_t v, int n) -> uint32_t { return (v << n) | (v >> (32 - n)); };

    for (size_t chunk = 0; chunk < padded_len; chunk += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)msg[chunk + i*4] << 24) | ((uint32_t)msg[chunk + i*4+1] << 16) |
                   ((uint32_t)msg[chunk + i*4+2] << 8) | (uint32_t)msg[chunk + i*4+3];
        for (int i = 16; i < 80; i++)
            w[i] = left_rotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;             k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;             k = 0xCA62C1D6; }
            uint32_t temp = left_rotate(a, 5) + f + e + k + w[i];
            e = d; d = c; c = left_rotate(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }
    auto store32 = [&](uint8_t* p, uint32_t v) { p[0]=v>>24; p[1]=v>>16; p[2]=v>>8; p[3]=v; };
    store32(hash, h0); store32(hash+4, h1); store32(hash+8, h2); store32(hash+12, h3); store32(hash+16, h4);
}

// ============================================================================
// WebSocket 帧编解码 (RFC 6455)
// ============================================================================

// WebSocket opcodes
static constexpr uint8_t WS_OP_TEXT   = 0x01;
static constexpr uint8_t WS_OP_BINARY = 0x02;
static constexpr uint8_t WS_OP_CLOSE  = 0x08;
static constexpr uint8_t WS_OP_PING   = 0x09;
static constexpr uint8_t WS_OP_PONG   = 0x0A;

// 完整写入 (处理部分写入和 EINTR)
static bool send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = ::send(fd, p, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;  // shouldn't happen on blocking socket
        p += n;
        remaining -= (size_t)n;
    }
    return true;
}

// 发送 WebSocket 帧 (服务端→客户端, 无 mask)
static bool ws_send_frame(int fd, uint8_t opcode, const uint8_t* data, size_t len) {
    uint8_t header[10];
    size_t hlen = 2;
    header[0] = 0x80 | opcode;  // FIN + opcode
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

static bool ws_send_text(int fd, const std::string& text) {
    return ws_send_frame(fd, WS_OP_TEXT, (const uint8_t*)text.data(), text.size());
}

static bool ws_send_binary(int fd, const uint8_t* data, size_t len) {
    return ws_send_frame(fd, WS_OP_BINARY, data, len);
}

// 接收 WebSocket 帧 (客户端→服务端, 带 mask)
// 返回 false = 连接关闭或错误
static bool ws_recv_frame(int fd, uint8_t& opcode, std::vector<uint8_t>& payload) {
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

    // 安全限制: 最大 64 MB
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

// WebSocket 握手: 验证并完成 HTTP 101 Upgrade
static bool ws_handshake(int fd, const HttpRequest& req) {
    auto it = req.headers.find("sec-websocket-key");
    if (it == req.headers.end()) return false;

    // Compute accept key: SHA-1(key + magic GUID), base64 encode
    static const char* WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string concat = it->second + WS_GUID;
    uint8_t hash[20];
    sha1_compute((const uint8_t*)concat.data(), concat.size(), hash);
    std::string accept = base64_encode(hash, 20);

    std::string response =
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Accept: " + accept + "\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    return ::send(fd, response.data(), response.size(), MSG_NOSIGNAL) > 0;
}

static ImageData decode_image_base64(const std::string& data_uri) {
    ImageData result;

    // Decode directly from offset, no substr copy
    size_t decode_offset = 0;
    auto comma_pos = data_uri.find(',');
    if (comma_pos != std::string::npos) {
        decode_offset = comma_pos + 1;
    }

    auto raw = base64_decode_from(data_uri, decode_offset);
    if (raw.empty()) return result;

    // Pre-flight: read image dimensions from header (no decode) to reject absurdly large images
    int info_w, info_h, info_c;
    if (stbi_info_from_memory(raw.data(), (int)raw.size(), &info_w, &info_h, &info_c)) {
        long long total_pixels = (long long)info_w * info_h;
        fprintf(stderr, "[Serve] Image header: %dx%d (%lld pixels, ~%.1f MB RGB)\n",
                info_w, info_h, total_pixels, total_pixels * 3.0 / 1048576.0);
        // Reject images that would need >256 MB for stbi decode (>85M pixels)
        if (total_pixels > 85000000LL) {
            fprintf(stderr, "[Serve] Image too large (%lld pixels), max 85M. Rejected.\n", total_pixels);
            return result;
        }
    }

    int w, h, channels;
    uint8_t* img = stbi_load_from_memory(raw.data(), (int)raw.size(), &w, &h, &channels, 3);
    // Release raw immediately after stbi load
    { std::vector<uint8_t>().swap(raw); }
    if (!img) {
        std::cerr << "[Serve] Failed to decode image: " << stbi_failure_reason() << std::endl;
        return result;
    }

    // Pre-downscale large images to avoid carrying 100+ MB through the entire pipeline.
    // The ViT smart_resize will shrink to max_pixels anyway; doing it here saves massive
    // memory (e.g., 7749x5812 = 129 MB RGB → 576x416 = 720 KB).
    const int MAX_PIXELS = 262144;  // matches VisionConfig.max_pixels
    if ((long long)w * h > MAX_PIXELS * 4) {
        // Compute smart_resize target dimensions
        int factor = 32;
        int h_bar = std::max(factor, (int)(std::round((double)h / factor) * factor));
        int w_bar = std::max(factor, (int)(std::round((double)w / factor) * factor));
        if ((long long)h_bar * w_bar > MAX_PIXELS) {
            double beta = std::sqrt((double)h * w / MAX_PIXELS);
            h_bar = std::max(factor, (int)(std::floor((double)h / beta / factor) * factor));
            w_bar = std::max(factor, (int)(std::floor((double)w / beta / factor) * factor));
        } else if ((long long)h_bar * w_bar < 256) {  // min_pixels
            double beta = std::sqrt(256.0 / ((double)h * w));
            h_bar = std::max(factor, (int)(std::ceil((double)h * beta / factor) * factor));
            w_bar = std::max(factor, (int)(std::ceil((double)w * beta / factor) * factor));
        }

        fprintf(stderr, "[Serve] Pre-downscale: %dx%d (%.1f MB) -> %dx%d (%.1f KB)\n",
                w, h, (double)w * h * 3 / 1048576.0, w_bar, h_bar,
                (double)w_bar * h_bar * 3 / 1024.0);

        // Bilinear downscale (uint8 RGB → uint8 RGB)
        std::vector<uint8_t> resized(w_bar * h_bar * 3);
        for (int y = 0; y < h_bar; y++) {
            float sy = (float)y * (h - 1) / std::max(h_bar - 1, 1);
            int y0 = (int)sy, y1 = std::min(y0 + 1, h - 1);
            float fy = sy - y0;
            for (int x = 0; x < w_bar; x++) {
                float sx = (float)x * (w - 1) / std::max(w_bar - 1, 1);
                int x0 = (int)sx, x1 = std::min(x0 + 1, w - 1);
                float fx = sx - x0;
                for (int c = 0; c < 3; c++) {
                    float val = (1 - fy) * ((1 - fx) * img[(y0 * w + x0) * 3 + c]
                                          + fx * img[(y0 * w + x1) * 3 + c])
                              + fy * ((1 - fx) * img[(y1 * w + x0) * 3 + c]
                                    + fx * img[(y1 * w + x1) * 3 + c]);
                    resized[(y * w_bar + x) * 3 + c] = (uint8_t)(val + 0.5f);
                }
            }
        }
        stbi_image_free(img);
        result.width  = w_bar;
        result.height = h_bar;
        result.pixels = std::move(resized);
    } else {
        result.width = w;
        result.height = h;
        result.pixels.assign(img, img + w * h * 3);
        stbi_image_free(img);
    }
    return result;
}

// Parse multimodal content array:
//   "content": [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:..."}},
//               {"type":"video","video":["base64_frame1",...],"fps":24}]
// Returns (text_content, images, videos)
struct MultimodalContent {
    std::string text;
    std::vector<ImageData> images;
    std::vector<VideoData> videos;
};

static MultimodalContent parse_multimodal_content(const std::string& content_json) {
    MultimodalContent result;

    auto extract_media_url = [](const std::string& obj,
                                const std::string& key,
                                const std::string& fallback_key = "url") -> std::string {
        // 兼容两种格式:
        //   1) "image_url": "data:..."
        //   2) "image_url": {"url":"data:..."}
        auto key_pos = obj.find("\"" + key + "\"");
        if (key_pos != std::string::npos) {
            size_t brace = obj.find('{', key_pos + key.size() + 2);
            size_t colon = obj.find(':', key_pos + key.size() + 2);
            if (brace != std::string::npos && (colon == std::string::npos || brace > colon)) {
                std::string nested = extract_json_object(obj, brace);
                std::string nested_url = json_get_string(nested, fallback_key);
                if (!nested_url.empty()) return nested_url;
            }
        }

        std::string direct = json_get_string(obj, key);
        if (!direct.empty()) return direct;
        return "";
    };

    // Check if content starts with [ (array) vs " (string)
    size_t start = 0;
    while (start < content_json.size() && (content_json[start] == ' ' || content_json[start] == '\n'))
        start++;

    if (start >= content_json.size() || content_json[start] != '[') {
        // Simple string content
        result.text = content_json;
        return result;
    }

    // Parse array of content parts
    size_t pos = start + 1;
    while (pos < content_json.size() && content_json[pos] != ']') {
        while (pos < content_json.size() && (content_json[pos] == ' ' || content_json[pos] == ','
               || content_json[pos] == '\n' || content_json[pos] == '\r' || content_json[pos] == '\t'))
            pos++;
        if (pos >= content_json.size() || content_json[pos] == ']') break;

        if (content_json[pos] == '{') {
            std::string obj = extract_json_object(content_json, pos);
            std::string type = json_get_string(obj, "type");

            if (type == "text" || type == "input_text") {
                std::string text = json_get_string(obj, "text");
                result.text += text;
            } else if (type == "image_url" || type == "input_image" || type == "image") {
                std::string url = extract_media_url(obj, "image_url");
                if (url.empty()) {
                    url = extract_media_url(obj, "image");
                }
                if (url.empty()) {
                    url = json_get_string(obj, "url");
                }
                if (!url.empty()) {
                    auto img = decode_image_base64(url);
                    if (img.width > 0) {
                        result.images.push_back(std::move(img));
                        result.text += "<|vision_start|><|image_pad|><|vision_end|>";
                    }
                }
            } else if (type == "video" || type == "input_video") {
                // Parse video as array of base64-encoded frames:
                //   {"type":"video", "video":["base64_frame1","base64_frame2",...], "fps":24}
                float fps = (float)json_get_number(obj, "fps", 24.0);
                auto vid_pos = obj.find("\"video\"");
                if (vid_pos == std::string::npos) {
                    vid_pos = obj.find("\"frames\"");
                }
                if (vid_pos != std::string::npos) {
                    auto vid_arr = obj.find('[', vid_pos);
                    if (vid_arr != std::string::npos) {
                        VideoData vd;
                        vd.source_fps = fps;
                        size_t vp = vid_arr + 1;
                        while (vp < obj.size() && obj[vp] != ']') {
                            while (vp < obj.size() && (obj[vp] == ' ' || obj[vp] == ','
                                   || obj[vp] == '\n' || obj[vp] == '\r' || obj[vp] == '\t'))
                                vp++;
                            if (vp >= obj.size() || obj[vp] == ']') break;
                            if (obj[vp] == '"') {
                                size_t start_q = vp + 1;
                                vp = start_q;
                                while (vp < obj.size() && obj[vp] != '"') {
                                    if (obj[vp] == '\\') vp++;
                                    vp++;
                                }
                                std::string b64_frame = obj.substr(start_q, vp - start_q);
                                vp++; // skip closing quote
                                auto frame_img = decode_image_base64(b64_frame);
                                if (frame_img.width > 0) {
                                    if (vd.width == 0) {
                                        vd.width = frame_img.width;
                                        vd.height = frame_img.height;
                                    }
                                    vd.frames.push_back(std::move(frame_img.pixels));
                                }
                            } else {
                                vp++;
                            }
                        }
                        if (!vd.frames.empty()) {
                            // Build video placeholder with timestamps
                            core::VisionConfig vcfg;
                            int num_frames = (int)vd.frames.size();
                            int target_frames = num_frames;
                            float target_fps = 2.0f;
                            if (vd.source_fps > 0)
                                target_frames = (int)(num_frames / vd.source_fps * target_fps);
                            target_frames = std::max(4, std::min(target_frames, 768));
                            target_frames = std::min(target_frames, num_frames);

                            // Compute grid for placeholder construction
                            auto [grid_t, grid_h, grid_w] = core::VisionEncoder::compute_video_grid(
                                target_frames, vd.height, vd.width, vcfg);

                            // Build selected frame indices for timestamps
                            std::vector<int> selected(target_frames);
                            for (int si = 0; si < target_frames; si++)
                                selected[si] = (int)std::round((double)si * (num_frames - 1) / std::max(1, target_frames - 1));

                            // Generate per-temporal-group placeholder text
                            for (int gt = 0; gt < grid_t; gt++) {
                                int f0 = gt * 2, f1 = gt * 2 + 1;
                                float t0 = (f0 < target_frames && vd.source_fps > 0) ?
                                            selected[f0] / vd.source_fps : 0;
                                float t1 = (f1 < target_frames && vd.source_fps > 0) ?
                                            selected[f1] / vd.source_fps : t0;
                                char buf[32];
                                snprintf(buf, sizeof(buf), "<%.1f seconds>", (t0 + t1) / 2.0f);
                                result.text += buf;
                                result.text += "<|vision_start|><|video_pad|><|vision_end|>";
                            }
                            result.videos.push_back(std::move(vd));
                        }
                    }
                }
            } else if (type == "video_url" || type == "input_video_url") {
                // Parse video_url: {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
                // Decode video file, extract frames with ffmpeg, build VideoData
                std::cerr << "[Serve] video_url type detected, obj.size()=" << obj.size() << std::endl;
                std::string url = extract_media_url(obj, "video_url");
                if (!url.empty()) {
                        std::cerr << "[Serve] url.size()=" << url.size() << std::endl;
                            // Strip data URI prefix to get raw base64
                            std::string b64_data;
                            auto comma = url.find(',');
                            if (comma != std::string::npos)
                                b64_data = url.substr(comma + 1);
                            else
                                b64_data = url;

                            std::cerr << "[Serve] b64_data.size()=" << b64_data.size() << std::endl;
                            auto video_bytes = base64_decode(b64_data);
                            std::cerr << "[Serve] video_bytes.size()=" << video_bytes.size() << std::endl;
                            if (!video_bytes.empty()) {
                                // Write to temp file
                                std::string tmp_video = "/tmp/qwen_video_" + std::to_string(getpid()) + ".mp4";
                                std::string tmp_dir = "/tmp/qwen_frames_" + std::to_string(getpid());
                                {
                                    FILE* f = fopen(tmp_video.c_str(), "wb");
                                    if (f) {
                                        fwrite(video_bytes.data(), 1, video_bytes.size(), f);
                                        fclose(f);
                                    }
                                }

                                // Get source fps with ffprobe
                                float source_fps = 30.0f;
                                {
                                    std::string cmd = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 " + tmp_video + " 2>/dev/null";
                                    FILE* pipe = popen(cmd.c_str(), "r");
                                    if (pipe) {
                                        char buf[128] = {};
                                        if (fgets(buf, sizeof(buf), pipe)) {
                                            int num = 0, den = 1;
                                            if (sscanf(buf, "%d/%d", &num, &den) == 2 && den > 0)
                                                source_fps = (float)num / den;
                                            else if (sscanf(buf, "%f", &source_fps) != 1)
                                                source_fps = 30.0f;
                                        }
                                        pclose(pipe);
                                    }
                                }

                                // Extract frames with ffmpeg at 2 fps
                                std::string mkdir_cmd = "mkdir -p " + tmp_dir;
                                if (system(mkdir_cmd.c_str()) != 0) {
                                    std::cerr << "[Serve] mkdir failed: " << tmp_dir << std::endl;
                                }
                                std::string ffmpeg_cmd = "ffmpeg -y -i " + tmp_video
                                    + " -vf fps=2 -frames:v 16 -q:v 2 " + tmp_dir + "/frame_%04d.jpg";
                                std::cerr << "[Serve] Running: " << ffmpeg_cmd << std::endl;
                                int ffret = system(ffmpeg_cmd.c_str());
                                std::cerr << "[Serve] ffmpeg returned: " << ffret << std::endl;

                                // Load extracted frames
                                VideoData vd;
                                vd.source_fps = source_fps;
                                for (int fi = 1; fi <= 16; fi++) {
                                    char fname[256];
                                    snprintf(fname, sizeof(fname), "%s/frame_%04d.jpg", tmp_dir.c_str(), fi);
                                    int fw = 0, fh = 0, fc = 0;
                                    uint8_t* fdata = stbi_load(fname, &fw, &fh, &fc, 3);
                                    if (!fdata) break;
                                    if (vd.width == 0) { vd.width = fw; vd.height = fh; }
                                    vd.frames.emplace_back(fdata, fdata + fw * fh * 3);
                                    stbi_image_free(fdata);
                                }

                                // Cleanup temp files
                                std::string cleanup = "rm -rf " + tmp_video + " " + tmp_dir;
                                if (system(cleanup.c_str()) != 0) {
                                    std::cerr << "[Serve] cleanup failed" << std::endl;
                                }

                                if (!vd.frames.empty()) {
                                    core::VisionConfig vcfg;
                                    int num_frames = (int)vd.frames.size();
                                    int target_frames = num_frames;
                                    float target_fps = 2.0f;

                                    auto [grid_t, grid_h, grid_w] = core::VisionEncoder::compute_video_grid(
                                        target_frames, vd.height, vd.width, vcfg);

                                    for (int gt = 0; gt < grid_t; gt++) {
                                        int f0 = gt * 2, f1 = gt * 2 + 1;
                                        float t0 = (f0 < target_frames && source_fps > 0) ?
                                                    (float)f0 / target_fps : 0;
                                        float t1 = (f1 < target_frames && source_fps > 0) ?
                                                    (float)f1 / target_fps : t0;
                                        char buf[32];
                                        snprintf(buf, sizeof(buf), "<%.1f seconds>", (t0 + t1) / 2.0f);
                                        result.text += buf;
                                        result.text += "<|vision_start|><|video_pad|><|vision_end|>";
                                    }
                                    std::cerr << "[Serve] video_url: " << vd.frames.size()
                                              << " frames extracted, " << vd.width << "x" << vd.height
                                              << " grid=" << grid_t << "x" << grid_h << "x" << grid_w
                                              << std::endl;
                                    result.videos.push_back(std::move(vd));
                                }
                            }
                }
            }
        } else {
            pos++;
        }
    }

    return result;
}

// Expand image placeholders in token sequence:
// Replace single <|image_pad|> with N_output copies based on image dimensions
// Multi-turn alignment: the LAST images.size() vision groups get expanded;
// earlier groups (history images not re-submitted) are collapsed to 0 pads.
// This ensures feature vectors align with the CORRECT positions in the prompt.
static void expand_image_placeholders(std::vector<int>& tokens,
                                       const std::vector<ImageData>& images,
                                       const Tokenizer& tokenizer) {
    // Token IDs for vision special tokens
    int vision_start_id = 248053;
    int image_pad_id    = 248056;
    int vision_end_id   = 248054;

    // Count total vision groups in the token sequence
    int total_vision_groups = 0;
    for (int t : tokens) if (t == vision_start_id) total_vision_groups++;

    // Number of leading groups to collapse (history images not in images[])
    int skip_groups = std::max(0, total_vision_groups - (int)images.size());

    core::VisionConfig vcfg;
    int group_idx = 0;  // increments each vision_start encountered
    int img_idx   = 0;  // increments only when a group is expanded

    std::vector<int> expanded;
    expanded.reserve(tokens.size() + images.size() * 256);

    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == vision_start_id) {
            expanded.push_back(vision_start_id);

            // Skip existing image_pad tokens and vision_end in source
            size_t j = i + 1;
            while (j < tokens.size() && tokens[j] == image_pad_id) j++;
            if (j < tokens.size() && tokens[j] == vision_end_id) j++;

            if (group_idx < skip_groups) {
                // Collapse: history group with no image data available
                // Emit 0 image_pads — just vision_start + vision_end
                std::cerr << "[Serve] Vision group " << group_idx
                          << " collapsed (no matching image in request)" << std::endl;
                expanded.push_back(vision_end_id);
            } else if (img_idx < (int)images.size()) {
                // Expand with the corresponding image
                // Pure arithmetic: compute n_tokens without any pixel data
                // smart_resize → grid_h/w → num_output_tokens
                auto& img = images[img_idx];
                int factor = vcfg.factor();  // 32
                int h_bar = std::max(factor, (int)(std::round((double)img.height / factor) * factor));
                int w_bar = std::max(factor, (int)(std::round((double)img.width  / factor) * factor));
                if ((long long)h_bar * w_bar > vcfg.max_pixels) {
                    double beta = std::sqrt((double)img.height * img.width / vcfg.max_pixels);
                    h_bar = std::max(factor, (int)(std::floor((double)img.height / beta / factor) * factor));
                    w_bar = std::max(factor, (int)(std::floor((double)img.width  / beta / factor) * factor));
                } else if ((long long)h_bar * w_bar < vcfg.min_pixels) {
                    double beta = std::sqrt((double)vcfg.min_pixels / ((double)img.height * img.width));
                    h_bar = std::max(factor, (int)(std::ceil((double)img.height * beta / factor) * factor));
                    w_bar = std::max(factor, (int)(std::ceil((double)img.width  * beta / factor) * factor));
                }
                int grid_h = h_bar / vcfg.patch_size;   // patch_size=16
                int grid_w = w_bar / vcfg.patch_size;
                int n_tokens = 1 * (grid_h / 2) * (grid_w / 2);  // grid_t=1, spatial_merge_size=2
                std::cerr << "[Serve] Vision group " << group_idx
                          << " expanded to " << n_tokens << " image_pad tokens" << std::endl;
                for (int k = 0; k < n_tokens; k++)
                    expanded.push_back(image_pad_id);
                expanded.push_back(vision_end_id);
                img_idx++;
            } else {
                // Defensive: no image left, collapse
                expanded.push_back(vision_end_id);
            }

            group_idx++;
            i = j - 1;  // -1 because loop will increment
        } else {
            expanded.push_back(tokens[i]);
        }
    }

    if (img_idx != (int)images.size()) {
        std::cerr << "[Serve] Warning: expand_image_placeholders consumed " << img_idx
                  << " of " << images.size() << " images" << std::endl;
    }

    // Count resulting image_pad tokens for verification
    int final_pad_count = 0;
    for (int t : expanded) if (t == image_pad_id) final_pad_count++;
    std::cerr << "[Serve] expand_image_placeholders: total_groups=" << total_vision_groups
              << " images=" << images.size() << " skip=" << skip_groups
              << " final_pad_count=" << final_pad_count << std::endl;
    std::cerr.flush();

    tokens = std::move(expanded);
}

// Expand video placeholders in token sequence:
// Each <|vision_start|><|video_pad|><|vision_end|> group (one per temporal group)
// gets expanded so the single <|video_pad|> becomes tokens_per_frame copies.
static void expand_video_placeholders(std::vector<int>& tokens,
                                       const std::vector<VideoData>& videos,
                                       const Tokenizer& tokenizer) {
    int vision_start_id = 248053;
    int video_pad_id    = 248057;
    int vision_end_id   = 248054;

    core::VisionConfig vcfg;

    // Pre-compute grid info for each video
    struct VideoGridInfo {
        int grid_t;
        int tokens_per_frame;  // (grid_h/2) * (grid_w/2)
        int groups_remaining;
    };
    std::vector<VideoGridInfo> infos;
    for (auto& vid : videos) {
        int num_frames = (int)vid.frames.size();
        float target_fps = 2.0f;
        int target_frames = num_frames;
        if (vid.source_fps > 0)
            target_frames = (int)(num_frames / vid.source_fps * target_fps);
        target_frames = std::max(4, std::min(target_frames, 768));
        target_frames = std::min(target_frames, num_frames);

        auto [gt, gh, gw] = core::VisionEncoder::compute_video_grid(
            target_frames, vid.height, vid.width, vcfg);
        int tpf = (gh / 2) * (gw / 2);
        infos.push_back({gt, tpf, gt});
    }

    int vid_idx = 0;
    std::vector<int> expanded;
    expanded.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == vision_start_id && vid_idx < (int)infos.size()) {
            // Check if next token is video_pad (not image_pad)
            size_t j = i + 1;
            if (j < tokens.size() && tokens[j] == video_pad_id) {
                expanded.push_back(vision_start_id);
                // Skip existing video_pad tokens
                while (j < tokens.size() && tokens[j] == video_pad_id) j++;
                // Skip vision_end
                if (j < tokens.size() && tokens[j] == vision_end_id) j++;

                // Insert correct count of video_pad tokens for this frame group
                int n = infos[vid_idx].tokens_per_frame;
                for (int k = 0; k < n; k++)
                    expanded.push_back(video_pad_id);
                expanded.push_back(vision_end_id);

                infos[vid_idx].groups_remaining--;
                if (infos[vid_idx].groups_remaining <= 0)
                    vid_idx++;

                i = j - 1;  // -1 because loop will increment
            } else {
                expanded.push_back(tokens[i]);
            }
        } else {
            expanded.push_back(tokens[i]);
        }
    }

    tokens = std::move(expanded);
}

// Parse "messages" array from JSON body into (role, content) pairs
// Handles:
//   - Regular messages: (role, content)
//   - Tool response messages (role "tool"): keeps role "tool", wraps content
//   - Assistant messages with tool_calls: appends <tool_call> blocks to content
static std::vector<std::pair<std::string, std::string>> parse_messages(
        const std::string& json,
        std::vector<ImageData>* out_images = nullptr,
        std::vector<VideoData>* out_videos = nullptr) {
    std::vector<std::pair<std::string, std::string>> messages;

    auto pos = json.find("\"messages\"");
    if (pos == std::string::npos) return messages;

    pos = json.find('[', pos);
    if (pos == std::string::npos) return messages;
    pos++;  // skip [

    while (pos < json.size()) {
        // Skip whitespace and commas
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
               json[pos] == '\n' || json[pos] == '\r' || json[pos] == ','))
            pos++;
        if (pos >= json.size() || json[pos] == ']') break;

        if (json[pos] == '{') {
            // Find matching }
            size_t obj_start = pos;
            int depth = 1;
            pos++;
            while (pos < json.size() && depth > 0) {
                if (json[pos] == '{') depth++;
                else if (json[pos] == '}') depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < json.size()) {
                        if (json[pos] == '"') {
                            int bs = 0;
                            size_t bp = pos;
                            while (bp > obj_start && json[bp - 1] == '\\') { bs++; bp--; }
                            if (bs % 2 == 0) break;
                        }
                        pos++;
                    }
                }
                pos++;
            }
            std::string obj = json.substr(obj_start, pos - obj_start);
            std::string role = json_get_string(obj, "role");
            std::string content;

            // Detect multimodal content (array vs string)
            {
                std::string csearch = "\"content\"";
                auto cpos = obj.find(csearch);
                if (cpos != std::string::npos) {
                    auto colon = obj.find(':', cpos + csearch.size());
                    if (colon != std::string::npos) {
                        size_t vp = colon + 1;
                        while (vp < obj.size() && (obj[vp] == ' ' || obj[vp] == '\t'
                               || obj[vp] == '\n' || obj[vp] == '\r'))
                            vp++;
                        if (vp < obj.size() && obj[vp] == '[') {
                            // Array content — extract the full array JSON
                            size_t arr_start = vp;
                            int d = 1;
                            size_t ap = vp + 1;
                            while (ap < obj.size() && d > 0) {
                                if (obj[ap] == '[') d++;
                                else if (obj[ap] == ']') d--;
                                else if (obj[ap] == '"') {
                                    ap++;
                                    while (ap < obj.size() && obj[ap] != '"') {
                                        if (obj[ap] == '\\') ap++;
                                        ap++;
                                    }
                                }
                                ap++;
                            }
                            std::string arr_json = obj.substr(arr_start, ap - arr_start);
                            auto mc = parse_multimodal_content(arr_json);
                            content = mc.text;
                            if (out_images) {
                                for (auto& img : mc.images)
                                    out_images->push_back(std::move(img));
                            }
                            if (out_videos) {
                                for (auto& vid : mc.videos)
                                    out_videos->push_back(std::move(vid));
                            }
                        } else {
                            // String content — use normal extraction
                            content = json_get_string(obj, "content");
                        }
                    }
                }
            }

            // Ollama-format images: "images": ["base64data", ...] per message
            if (out_images) {
                auto img_pos = obj.find("\"images\"");
                if (img_pos != std::string::npos) {
                    auto img_arr = obj.find('[', img_pos);
                    if (img_arr != std::string::npos) {
                        size_t ip = img_arr + 1;
                        while (ip < obj.size() && obj[ip] != ']') {
                            while (ip < obj.size() && (obj[ip] == ' ' || obj[ip] == ','
                                   || obj[ip] == '\n' || obj[ip] == '\r' || obj[ip] == '\t'))
                                ip++;
                            if (ip >= obj.size() || obj[ip] == ']') break;
                            if (obj[ip] == '"') {
                                size_t start_q = ip + 1;
                                ip = start_q;
                                while (ip < obj.size() && obj[ip] != '"') {
                                    if (obj[ip] == '\\') ip++;
                                    ip++;
                                }
                                std::string b64_str = obj.substr(start_q, ip - start_q);
                                ip++; // skip closing quote
                                auto img = decode_image_base64(b64_str);
                                if (img.width > 0) {
                                    out_images->push_back(std::move(img));
                                    // Insert vision placeholders into content
                                    content = "<|vision_start|><|image_pad|><|vision_end|>" + content;
                                }
                            } else {
                                ip++;
                            }
                        }
                    }
                }
            }

            if (role == "assistant") {
                // 检查是否包含 tool_calls 数组
                auto tc_pos = obj.find("\"tool_calls\"");
                if (tc_pos != std::string::npos) {
                    auto tc_arr = obj.find('[', tc_pos);
                    if (tc_arr != std::string::npos) {
                        // 遍历 tool_calls 数组中的每个对象
                        size_t tp = tc_arr + 1;
                        while (tp < obj.size() && obj[tp] != ']') {
                            while (tp < obj.size() && (obj[tp] == ' ' || obj[tp] == ','
                                   || obj[tp] == '\n' || obj[tp] == '\r' || obj[tp] == '\t'))
                                tp++;
                            if (tp >= obj.size() || obj[tp] == ']') break;
                            if (obj[tp] == '{') {
                                std::string tc_obj = extract_json_object(obj, tp);
                                // 找 "function" 子对象
                                auto func_pos = tc_obj.find("\"function\"");
                                if (func_pos != std::string::npos) {
                                    size_t fp = tc_obj.find('{', func_pos + 10);
                                    if (fp != std::string::npos) {
                                        std::string func = extract_json_object(tc_obj, fp);
                                        std::string name = json_get_string(func, "name");
                                        // arguments 是 JSON 字符串, 需要 unescape
                                        std::string args_raw = json_get_string(func, "arguments");
                                        std::string args = json_unescape(args_raw);
                                        content += "\n<tool_call>\n{\"name\": \"" + name
                                                 + "\", \"arguments\": " + args + "}\n</tool_call>";
                                    }
                                }
                            } else {
                                tp++;
                            }
                        }
                    }
                }
            }

            if (!role.empty()) {
                messages.emplace_back(role, content);
            }
        } else {
            pos++;
        }
    }

    return messages;
}

static std::string iso8601_now() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::string generate_id(const std::string& prefix = "chatcmpl") {
    // Generate chatcmpl-xxxxxxxxxxxx style ID with random hex suffix
    static std::mt19937_64 rng(std::random_device{}());
    static std::mutex rng_mu;
    static const char hex[] = "0123456789abcdef";
    char buf[13]; // 12 hex chars + null
    uint64_t val;
    { std::lock_guard<std::mutex> lk(rng_mu); val = rng(); }
    for (int i = 0; i < 12; ++i) {
        buf[i] = hex[(val >> (i * 4)) & 0xf];
    }
    buf[12] = '\0';
    return prefix + "-" + buf;
}

// ============================================================================
// ServeConfig
// ============================================================================

ServeConfig ServeConfig::from_args(int argc, char** argv) {
    ServeConfig cfg;
    return merge_args(cfg, argc, argv);
}

ServeConfig ServeConfig::merge_args(const ServeConfig& base, int argc, char** argv) {
    ServeConfig cfg = base;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc)        cfg.host = argv[++i];
        else if (arg == "--port" && i + 1 < argc)    cfg.ollama_port = std::stoi(argv[++i]);
        else if (arg == "--ollama-port" && i + 1 < argc) cfg.ollama_port = std::stoi(argv[++i]);
        else if (arg == "--openai-port" && i + 1 < argc) cfg.openai_port = std::stoi(argv[++i]);
        else if (arg == "--max-conns" && i + 1 < argc) cfg.max_conns = std::stoi(argv[++i]);
        else if (arg == "--model-name" && i + 1 < argc) cfg.model_name = argv[++i];
        else if (arg == "--timeout" && i + 1 < argc) cfg.timeout_s = std::stoi(argv[++i]);
        else if (arg == "--max-output-tokens" && i + 1 < argc) cfg.max_output_tokens_cap = std::stoi(argv[++i]);
        else if (arg == "--voice-max-turns" && i + 1 < argc) cfg.voice_max_turns = std::stoi(argv[++i]);
        else if (arg == "--voice-max-output-tokens" && i + 1 < argc) cfg.voice_max_output_tokens = std::stoi(argv[++i]);
        else if (arg == "--voice-system-prompt" && i + 1 < argc) cfg.voice_system_prompt = argv[++i];
    }
    return cfg;
}

ServeConfig ServeConfig::from_file(const std::string& path) {
    ServeConfig cfg;
    std::ifstream ifs(path);
    if (!ifs) return cfg;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        while (!val.empty() && val.front() == ' ') val.erase(val.begin());

        if      (key == "host")       cfg.host = val;
        else if (key == "port")       cfg.ollama_port = std::stoi(val);
        else if (key == "ollama_port") cfg.ollama_port = std::stoi(val);
        else if (key == "openai_port") cfg.openai_port = std::stoi(val);
        else if (key == "max_conns")  cfg.max_conns = std::stoi(val);
        else if (key == "model_name") cfg.model_name = val;
        else if (key == "timeout")    cfg.timeout_s = std::stoi(val);
        else if (key == "max_output_tokens") cfg.max_output_tokens_cap = std::stoi(val);
        else if (key == "voice_max_turns") cfg.voice_max_turns = std::stoi(val);
        else if (key == "voice_max_output_tokens") cfg.voice_max_output_tokens = std::stoi(val);
        else if (key == "voice_system_prompt") {
            // Support \n escape for multi-line prompts in config
            std::string parsed;
            for (size_t i = 0; i < val.size(); i++) {
                if (val[i] == '\\' && i + 1 < val.size() && val[i + 1] == 'n') {
                    parsed += '\n';
                    i++;
                } else {
                    parsed += val[i];
                }
            }
            cfg.voice_system_prompt = parsed;
        }
    }
    // 保存 config 文件中的初始值作为 reset 基准
    cfg.voice_system_prompt_default = cfg.voice_system_prompt;
    cfg.voice_max_turns_default = cfg.voice_max_turns;
    cfg.voice_max_output_tokens_default = cfg.voice_max_output_tokens;
    return cfg;
}

void ServeConfig::print() const {
    printf("┌─────────────────────────────────────────────┐\n");
    printf("│          Serve Configuration                │\n");
    printf("├─────────────────────────────────────────────┤\n");
    printf("│  Host:          %-26s │\n", host.c_str());
    printf("│  Ollama Port:   %-6d                       │\n", ollama_port);
    printf("│  OpenAI Port:   %-6d                       │\n", openai_port);
    printf("│  Max Conns:     %-6d                       │\n", max_conns);
    printf("│  Model Name:    %-26s │\n", model_name.c_str());
    printf("│  Timeout:       %-6d s                     │\n", timeout_s);
    printf("│  Max Output:    %-6d tok                   │\n", max_output_tokens_cap);
    printf("│  Voice Turns:   %-6d                       │\n", voice_max_turns);
    printf("│  Voice Tokens:  %-6d                       │\n", voice_max_output_tokens);
    if (!voice_system_prompt.empty())
        printf("│  Voice Prompt:  %.38s │\n", voice_system_prompt.c_str());
    printf("└─────────────────────────────────────────────┘\n");
}

// ============================================================================
// ServeApp
// ============================================================================

ServeApp::ServeApp(const ServeConfig& config, InferenceBackend& backend,
                   std::unique_ptr<plugins::AsrPlugin> asr,
                   std::unique_ptr<plugins::TtsPlugin> tts)
    : config_(config), backend_(backend),
      asr_plugin_(std::move(asr)), tts_plugin_(std::move(tts)) {
    if (asr_plugin_) {
        fprintf(stderr, "[Serve] ASR plugin loaded: %s (available=%s)\n",
                asr_plugin_->name().c_str(),
                asr_plugin_->is_available() ? "yes" : "no");
    }
    if (tts_plugin_) {
        fprintf(stderr, "[Serve] TTS plugin loaded: %s (available=%s)\n",
                tts_plugin_->name().c_str(),
                tts_plugin_->is_available() ? "yes" : "no");
    }

    // 加载 CAM++ 说话人编码器 (如果配置了 speaker_model 路径)
    // 从 AsrConfig 获取路径: 由 main.cpp 传入配置
    // 自动发现: 检查 ASR 模型同级目录下的 campplus 模型
    if (asr_plugin_ && asr_plugin_->is_available()) {
        // 通过环境变量或默认路径加载
        std::string speaker_model_path;
        const char* env_path = getenv("QWEN_SPEAKER_MODEL");
        if (env_path) {
            speaker_model_path = env_path;
        } else {
            // 默认路径: /home/rm01/models/dev/asr/campplus/campplus.safetensors
            std::string default_path = "/home/rm01/models/dev/asr/campplus/campplus.safetensors";
            std::ifstream test_file(default_path);
            if (test_file.good()) speaker_model_path = default_path;
        }

        if (!speaker_model_path.empty()) {
            speaker_encoder_ = std::make_unique<asr::GpuSpeakerEncoder>();
            if (speaker_encoder_->load(speaker_model_path)) {
                fprintf(stderr, "[Serve] Speaker encoder loaded: CAM++ (%s)\n",
                        speaker_model_path.c_str());

                // GPU Mel 特征提取 (cuFFT, Kaldi-compatible: Povey window, zero-pad 512, low_freq=20)
                asr::GpuMelConfig mel_cfg;
                // Kaldi/FunASR CAMPPlus default: Povey window, pad_to_power_of_two, low_freq=20
                gpu_mel_.init(mel_cfg);
                fprintf(stderr, "[Serve] GPU Mel extractor initialized (cuFFT)\n");

                // 加载 FSMN-VAD 引擎 (用于文件转写说话人分割)
                std::string vad_model_dir = "/home/rm01/models/dev/asr/fsmn_vad";
                const char* env_vad = getenv("QWEN_VAD_MODEL");
                if (env_vad) vad_model_dir = env_vad;
                if (vad_engine_.load(vad_model_dir)) {
                    fprintf(stderr, "[Serve] VAD engine loaded: FSMN (%s)\n",
                            vad_model_dir.c_str());

                    // GPU VAD: 加载 FSMN 权重到 GPU (batch GEMM + cuFFT 加速)
                    if (gpu_vad_engine_.load(vad_model_dir)) {
                        fprintf(stderr, "[Serve] GPU VAD engine loaded (cuFFT + batch GEMM)\n");
                    }
                } else {
                    fprintf(stderr, "[Serve] WARNING: Failed to load VAD engine from %s "
                            "(speaker diarization in file transcription will be disabled)\n",
                            vad_model_dir.c_str());
                }
            } else {
                fprintf(stderr, "[Serve] WARNING: Failed to load speaker encoder from %s\n",
                        speaker_model_path.c_str());
                speaker_encoder_.reset();
            }
        }
    }

    // 加载 ForcedAligner (Qwen3-ForcedAligner-0.6B)
    if (asr_plugin_ && asr_plugin_->is_available()) {
        std::string aligner_model = "/home/rm01/models/dev/asr/Qwen/Qwen3-ForcedAligner-0.6B";
        const char* env_aligner = getenv("QWEN_ALIGNER_MODEL");
        if (env_aligner) aligner_model = env_aligner;

        std::ifstream test_cfg(aligner_model + "/config.json");
        if (test_cfg.good()) {
            test_cfg.close();
            if (aligner_engine_.load_model(aligner_model)) {
                fprintf(stderr, "[Serve] ForcedAligner loaded: %s\n", aligner_model.c_str());
            } else {
                fprintf(stderr, "[Serve] WARNING: Failed to load ForcedAligner from %s\n",
                        aligner_model.c_str());
            }
        }
    }
}

ServeApp::~ServeApp() {
    stop();
}

void ServeApp::run() {
    // 忽略 SIGPIPE (客户端断开时不崩溃)
    signal(SIGPIPE, SIG_IGN);

    // 创建 Ollama 端口 socket
    ollama_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (ollama_fd_ < 0) {
        std::cerr << "[Serve] socket() failed for Ollama port: " << strerror(errno) << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(ollama_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in ollama_addr{};
    ollama_addr.sin_family = AF_INET;
    ollama_addr.sin_port   = htons(config_.ollama_port);
    inet_pton(AF_INET, config_.host.c_str(), &ollama_addr.sin_addr);

    if (bind(ollama_fd_, (struct sockaddr*)&ollama_addr, sizeof(ollama_addr)) < 0) {
        std::cerr << "[Serve] bind() failed for Ollama port " << config_.ollama_port
                  << ": " << strerror(errno) << " (continuing with OpenAI port only)" << std::endl;
        close(ollama_fd_);
        ollama_fd_ = -1;
        // Don't return — OpenAI port may still work
    }

    if (ollama_fd_ >= 0 && listen(ollama_fd_, config_.max_conns) < 0) {
        std::cerr << "[Serve] listen() failed for Ollama port: " << strerror(errno) << std::endl;
        close(ollama_fd_);
        ollama_fd_ = -1;
        // Don't return — OpenAI port may still work
    }

    // 创建 OpenAI 端口 socket
    openai_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (openai_fd_ < 0) {
        std::cerr << "[Serve] socket() failed for OpenAI port: " << strerror(errno) << std::endl;
        if (ollama_fd_ >= 0) { close(ollama_fd_); ollama_fd_ = -1; }
        return;
    }

    setsockopt(openai_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in openai_addr{};
    openai_addr.sin_family = AF_INET;
    openai_addr.sin_port   = htons(config_.openai_port);
    inet_pton(AF_INET, config_.host.c_str(), &openai_addr.sin_addr);

    if (bind(openai_fd_, (struct sockaddr*)&openai_addr, sizeof(openai_addr)) < 0) {
        std::cerr << "[Serve] bind() failed for OpenAI port " << config_.openai_port
                  << ": " << strerror(errno) << std::endl;
        close(openai_fd_);
        openai_fd_ = -1;
    }

    if (openai_fd_ >= 0 && listen(openai_fd_, config_.max_conns) < 0) {
        std::cerr << "[Serve] listen() failed for OpenAI port: " << strerror(errno) << std::endl;
        close(openai_fd_);
        openai_fd_ = -1;
    }

    // 两个端口都失败时才退出
    if (ollama_fd_ < 0 && openai_fd_ < 0) {
        std::cerr << "[Serve] Both Ollama and OpenAI ports failed to bind. Exiting." << std::endl;
        return;
    }

    running_ = true;
    config_.print();
    if (ollama_fd_ >= 0) {
        printf("\n[Serve] Ollama API on http://%s:%d\n", config_.host.c_str(), config_.ollama_port);
        printf("  POST /api/generate          — Ollama Generate API\n");
        printf("  POST /api/chat              — Ollama Chat API\n");
        printf("  POST /api/show              — Model information\n");
        printf("  GET  /api/tags              — List local models\n");
        printf("  GET  /api/ps                — List running models\n");
        printf("  GET  /api/version           — Version info\n");
        printf("  GET  /health                — Health check\n");
    } else {
        printf("\n[Serve] Ollama API: DISABLED (port %d unavailable)\n", config_.ollama_port);
    }
    if (openai_fd_ >= 0) {
        printf("\n[Serve] OpenAI API on http://%s:%d\n", config_.host.c_str(), config_.openai_port);
        printf("  POST /v1/responses          — OpenAI Responses API (minimal)\n");
        printf("  POST /v1/chat/completions   — OpenAI Chat API\n");
        printf("  POST /v1/completions        — OpenAI Completions API\n");
        printf("  GET  /v1/models             — Model list\n");
        printf("  GET  /v1/models/{model}     — Retrieve model\n");
        printf("  GET  /health                — Health check\n\n");
    } else {
        printf("\n[Serve] OpenAI API: DISABLED (port %d unavailable)\n\n", config_.openai_port);
    }

    // 启动响应分发线程 (从 backend 单消费者队列路由到 per-request 队列)
    resp_dispatcher_ = std::thread(&ServeApp::response_dispatch_loop, this);

    accept_loop();
}

void ServeApp::stop() {
    running_ = false;
    resp_cv_.notify_all();  // 唤醒等待中的 poll_request
    if (resp_dispatcher_.joinable()) resp_dispatcher_.join();
    if (ollama_fd_ >= 0) {
        close(ollama_fd_);
        ollama_fd_ = -1;
    }
    if (openai_fd_ >= 0) {
        close(openai_fd_);
        openai_fd_ = -1;
    }
    // Worker 线程已 detach, 等待它们完成
    while (active_workers_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void ServeApp::accept_loop() {
    while (running_) {
        struct pollfd pfds[2];
        pfds[0].fd = ollama_fd_;
        pfds[0].events = POLLIN;
        pfds[1].fd = openai_fd_;
        pfds[1].events = POLLIN;

        int ret = poll(pfds, 2, 1000);  // 1s 超时
        if (ret <= 0) continue;

        for (int i = 0; i < 2; i++) {
            if (!(pfds[i].revents & POLLIN)) continue;

            struct sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(pfds[i].fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) continue;

            fprintf(stderr, "[Serve] accept() fd=%d protocol=%s\n",
                    client_fd, i == 0 ? "ollama" : "openai");

            // 检查并发限制
            if (active_workers_.load() >= config_.max_conns) {
                const char* busy = "HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Length: 38\r\n\r\n"
                    "{\"error\":\"Too many connections\"}";
                send(client_fd, busy, strlen(busy), MSG_NOSIGNAL);
                close(client_fd);
                continue;
            }

            int protocol = i;  // 0 = Ollama, 1 = OpenAI
            active_workers_.fetch_add(1);
            fprintf(stderr, "[Serve] spawning thread for fd=%d...\n", client_fd);
            std::thread([this, client_fd, protocol]() {
                try {
                    handle_connection(client_fd, protocol);
                } catch (const std::exception& e) {
                    fprintf(stderr, "[Serve] EXCEPTION in connection handler fd=%d: %s\n", client_fd, e.what());
                    close(client_fd);
                } catch (...) {
                    fprintf(stderr, "[Serve] UNKNOWN EXCEPTION in connection handler fd=%d\n", client_fd);
                    close(client_fd);
                }
                active_workers_.fetch_sub(1);
            }).detach();
            fprintf(stderr, "[Serve] thread spawned for fd=%d\n", client_fd);
        }
    }
}

void ServeApp::handle_connection(int client_fd, int protocol) {
    // Print BEFORE body receive so we know this connection arrived even if crash follows
    fprintf(stderr, "[Serve] Incoming connection fd=%d\n", client_fd);
    fflush(stderr);
    auto req = parse_request(client_fd);
    fprintf(stderr, "[Serve] %s %s body=%zu bytes\n",
            req.method.c_str(), req.path.c_str(), req.body.size());
    fflush(stderr);

    // CORS preflight 两个端口都处理
    if (req.method == "OPTIONS") {
        handle_cors_preflight(req, client_fd);
        close(client_fd);
        return;
    }

    // WebSocket Upgrade 检测 (OpenAI 端口)
    {
        auto upgrade_it = req.headers.find("upgrade");
        if (upgrade_it != req.headers.end()) {
            std::string val = upgrade_it->second;
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "websocket") {
                if (protocol == 1 && req.path == "/v1/realtime") {
                    handle_websocket_realtime(client_fd, req);
                    close(client_fd);
                    return;
                }
                if (protocol == 1 && req.path == "/v1/voice") {
                    handle_websocket_voice(client_fd, req);
                    close(client_fd);
                    return;
                }
                // 不支持的 WebSocket 路径
                HttpResponse resp;
                resp.status_code = 404;
                resp.status_text = "Not Found";
                resp.body = "{\"error\":\"WebSocket endpoint not found\"}";
                send_response(client_fd, resp);
                close(client_fd);
                return;
            }
        }
    }

    // /health 两个端口都可用
    if (req.path == "/health" && req.method == "GET") {
        handle_health(req, client_fd);
        close(client_fd);
        return;
    }

    if (protocol == 0) {
        // Ollama 端口: 只接受 /api/* 路由
        if (req.path == "/api/tags" && req.method == "GET") {
            handle_ollama_tags(req, client_fd);
        } else if (req.path == "/api/show" && req.method == "POST") {
            handle_ollama_show(req, client_fd);
        } else if (req.path == "/api/ps" && req.method == "GET") {
            handle_ollama_ps(req, client_fd);
        } else if (req.path == "/api/version" && req.method == "GET") {
            handle_ollama_version(req, client_fd);
        } else if (req.path == "/api/generate" && req.method == "POST") {
            handle_ollama_generate(req, client_fd);
        } else if (req.path == "/api/chat" && req.method == "POST") {
            handle_ollama_chat(req, client_fd);
        } else {
            HttpResponse resp;
            resp.status_code = 404;
            resp.status_text = "Not Found";
            resp.body = "{\"error\":\"endpoint not found on Ollama port\"}";
            send_response(client_fd, resp);
        }
    } else {
        // OpenAI 端口: /v1/* API + 静态文件
        if (req.path == "/v1/models" && req.method == "GET") {
            handle_models(req, client_fd);
        } else if (req.path.rfind("/v1/models/", 0) == 0 && req.method == "GET") {
            // GET /v1/models/{model} — retrieve individual model
            handle_model_retrieve(req, client_fd);
        } else if (req.path == "/v1/responses" && req.method == "POST") {
            handle_openai_responses(req, client_fd);
        } else if (req.path == "/v1/chat/completions" && req.method == "POST") {
            handle_openai_chat(req, client_fd);
        } else if (req.path == "/v1/completions" && req.method == "POST") {
            handle_openai_completions(req, client_fd);
        } else if (req.path == "/v1/audio/transcriptions" && req.method == "POST") {
            handle_audio_transcriptions(req, client_fd);
        } else if (req.path == "/v1/audio/speech" && req.method == "POST") {
            handle_audio_speech(req, client_fd);
        } else if (req.path == "/v1/tts/info" && req.method == "GET") {
            handle_tts_info(req, client_fd);
        } else if (req.path == "/v1/voice_clone/register" && req.method == "POST") {
            handle_voice_clone_register(req, client_fd);
        } else if (req.path == "/v1/voice_clone/voices" && req.method == "GET") {
            handle_voice_clone_voices(req, client_fd);
        } else if (req.path == "/v1/voice_clone/delete" && req.method == "POST") {
            handle_voice_clone_delete(req, client_fd);
        } else if (req.path == "/v1/speakers/register" && req.method == "POST") {
            handle_speaker_register(req, client_fd);
        } else if (req.path == "/v1/speakers" && req.method == "GET") {
            handle_speaker_list(req, client_fd);
        } else if (req.path == "/v1/speakers/delete" && req.method == "POST") {
            handle_speaker_delete(req, client_fd);
        } else if (req.method == "GET" && req.path.rfind("/v1/recordings/", 0) == 0) {
            // GET /v1/recordings/{filename} — 下载录音文件
            std::string filename = req.path.substr(strlen("/v1/recordings/"));
            // 安全检查: 只允许文件名，不允许路径遍历
            if (filename.empty() || filename.find('/') != std::string::npos
                || filename.find("..") != std::string::npos) {
                HttpResponse resp;
                resp.status_code = 400;
                resp.status_text = "Bad Request";
                resp.body = "{\"error\":\"invalid filename\"}";
                send_response(client_fd, resp);
            } else {
                std::string filepath = "tmp/recordings/" + filename;
                std::ifstream f(filepath, std::ios::binary | std::ios::ate);
                if (!f) {
                    HttpResponse resp;
                    resp.status_code = 404;
                    resp.status_text = "Not Found";
                    resp.body = "{\"error\":\"file not found\"}";
                    send_response(client_fd, resp);
                } else {
                    auto size = f.tellg();
                    f.seekg(0);
                    std::string data((size_t)size, '\0');
                    f.read(data.data(), size);
                    f.close();
                    HttpResponse resp;
                    resp.status_code = 200;
                    resp.status_text = "OK";
                    resp.content_type = "audio/wav";
                    resp.body = std::move(data);
                    send_response(client_fd, resp);
                }
            }
        } else if (req.method == "GET" && (req.path == "/" || req.path.rfind("/examples/", 0) == 0)) {
            handle_static_file(req, client_fd);
        } else {
            HttpResponse resp;
            resp.status_code = 404;
            resp.status_text = "Not Found";
            resp.body = "{\"error\":{\"message\":\"endpoint not found on OpenAI port\",\"type\":\"invalid_request_error\"}}";
            send_response(client_fd, resp);
        }
    }

    close(client_fd);
}

HttpRequest ServeApp::parse_request(int client_fd) {
    HttpRequest req;
    req.client_fd = client_fd;

    fprintf(stderr, "[Serve] parse_request fd=%d: reading headers...\n", client_fd);

    // 设置 header 读取超时 (10秒), 防止空连接阻塞线程
    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 读取 HTTP 头部
    std::string raw;
    char buf[4096];
    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                fprintf(stderr, "[Serve] parse_request fd=%d: header read timeout (10s)\n", client_fd);
            }
            break;
        }
        raw.append(buf, n);
        if (raw.find("\r\n\r\n") != std::string::npos) break;
    }

    // 恢复无限超时 (body 读取和后续操作可能需要更久)
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    fprintf(stderr, "[Serve] parse_request fd=%d: headers received, raw=%zu bytes\n", client_fd, raw.size());

    // 解析请求行
    auto first_line_end = raw.find("\r\n");
    if (first_line_end != std::string::npos) {
        auto line = raw.substr(0, first_line_end);
        auto sp1 = line.find(' ');
        auto sp2 = line.find(' ', sp1 + 1);
        if (sp1 != std::string::npos && sp2 != std::string::npos) {
            req.method = line.substr(0, sp1);
            req.path   = line.substr(sp1 + 1, sp2 - sp1 - 1);
        }
    }

    // 解析 headers
    auto header_end = raw.find("\r\n\r\n");
    if (header_end != std::string::npos) {
        auto header_section = raw.substr(first_line_end + 2, header_end - first_line_end - 2);
        size_t pos = 0;
        while (pos < header_section.size()) {
            auto line_end = header_section.find("\r\n", pos);
            if (line_end == std::string::npos) line_end = header_section.size();
            auto line = header_section.substr(pos, line_end - pos);
            auto colon = line.find(':');
            if (colon != std::string::npos) {
                auto key = line.substr(0, colon);
                auto val = line.substr(colon + 1);
                while (!val.empty() && val.front() == ' ') val.erase(val.begin());
                // 转小写
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                req.headers[key] = val;
            }
            pos = line_end + 2;
        }

        // Body: 直接分配完整大小, 避免多次 realloc
        auto cl_it = req.headers.find("content-length");
        if (cl_it != req.headers.end()) {
            size_t content_len = std::stoull(cl_it->second);
            size_t body_in_raw = raw.size() - (header_end + 4);
            if (body_in_raw > content_len) body_in_raw = content_len;

            // 一次性分配完整 body 缓冲区
            req.body.resize(content_len);
            // 复制已在 header 读取中收到的 body 部分
            if (body_in_raw > 0) {
                memcpy(&req.body[0], raw.data() + header_end + 4, body_in_raw);
            }

            fprintf(stderr, "[Serve] parse_request fd=%d: body %zu/%zu bytes (Content-Length=%zu)\n",
                    client_fd, body_in_raw, content_len, content_len);

            // 直接 recv 到 body 缓冲区, 无中间复制
            size_t received = body_in_raw;
            while (received < content_len) {
                ssize_t n = recv(client_fd, &req.body[received], content_len - received, 0);
                if (n <= 0) break;
                received += n;
            }
            req.body.resize(received);  // 处理短读
            fprintf(stderr, "[Serve] parse_request fd=%d: body complete %zu bytes\n", client_fd, received);
        } else {
            // 无 Content-Length, 取 header 后的所有数据
            req.body = raw.substr(header_end + 4);
        }
    }

    return req;
}

void ServeApp::send_response(int client_fd, const HttpResponse& resp) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status_code << " " << resp.status_text << "\r\n";
    oss << "Content-Type: " << resp.content_type << "; charset=utf-8\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Content-Length: " << resp.body.size() << "\r\n";
    oss << "\r\n";
    oss << resp.body;

    auto str = oss.str();
    size_t sent = 0;
    while (sent < str.size()) {
        ssize_t n = send(client_fd, str.data() + sent, str.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

bool ServeApp::send_sse_event(int client_fd, const std::string& data) {
    std::string event = "data: " + data + "\n\n";
    return ::send(client_fd, event.c_str(), event.size(), MSG_NOSIGNAL) >= 0;
}

void ServeApp::send_sse_done(int client_fd) {
    std::string done = "data: [DONE]\n\n";
    send(client_fd, done.c_str(), done.size(), MSG_NOSIGNAL);
}

bool ServeApp::send_ndjson_chunk(int client_fd, const std::string& json_line) {
    std::string data = json_line + "\n";
    std::ostringstream oss;
    oss << std::hex << data.size() << "\r\n" << data << "\r\n";
    auto chunk = oss.str();
    return ::send(client_fd, chunk.c_str(), chunk.size(), MSG_NOSIGNAL) >= 0;
}

void ServeApp::send_chunked_end(int client_fd) {
    static const char* end_marker = "0\r\n\r\n";
    send(client_fd, end_marker, 5, MSG_NOSIGNAL);
}

void ServeApp::handle_cors_preflight(const HttpRequest& /*req*/, int client_fd) {
    std::string resp = "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization, X-Request-Id\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\n\r\n";
    send(client_fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
}

// ---------------------------------------------------------------------------
// 响应分发系统 — 解决并发 token 窃取问题
//
// backend_ 的响应队列是 SPSC (单生产者单消费者) 设计.
// response_dispatch_loop 作为唯一消费者, 将响应路由到 per-request 队列.
// 各 handler 线程通过 poll_request() 从自己的队列读取.
// ---------------------------------------------------------------------------

void ServeApp::response_dispatch_loop() {
    while (running_) {
        InferResponse resp;
        if (backend_.poll(resp)) {
            std::lock_guard<std::mutex> lock(resp_mutex_);
            auto it = resp_queues_.find(resp.request_id);
            if (it != resp_queues_.end()) {
                it->second.push_back(resp);
                resp_cv_.notify_all();
            }
            // 如果 request_id 没有注册的 queue (请求已超时), 丢弃
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void ServeApp::register_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(resp_mutex_);
    resp_queues_[request_id]; // 创建空 deque
}

void ServeApp::unregister_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(resp_mutex_);
    resp_queues_.erase(request_id);
}

bool ServeApp::poll_request(uint64_t request_id, InferResponse& resp, int timeout_ms) {
    std::unique_lock<std::mutex> lock(resp_mutex_);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        auto it = resp_queues_.find(request_id);
        if (it != resp_queues_.end() && !it->second.empty()) {
            resp = it->second.front();
            it->second.pop_front();
            return true;
        }
        if (!running_) return false;
        if (resp_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            return false;
        }
    }
}

int ServeApp::poll_tokens(uint64_t request_id,
                          const std::function<void(const std::string&)>& on_token,
                          int timeout_s,
                          bool start_in_thinking,
                          const std::vector<std::string>& stop_seqs,
                          const std::function<void(const std::string&)>& on_reasoning,
                          const std::function<void(const ToolCallInfo&)>& on_tool_call,
                          std::string* out_finish_reason,
                          std::atomic<bool>* abort_flag,
                          int* out_cached_tokens) {
    const auto& tok = backend_.tokenizer();
    int count = 0;
    // enable_thinking=true 时, prompt 已以 <think>\n 结尾,
    // 模型直接从思考模式开始, 不会再输出 <think> token
    bool in_thinking = start_in_thinking;
    auto start = std::chrono::steady_clock::now();

    // Stop sequence匹配状态: 缓冲最近输出, 延迟emission直到确认不含stop前缀
    std::string pending_output;  // 尚未发射的缓冲区
    size_t max_stop_len = 0;
    for (auto& s : stop_seqs) max_stop_len = std::max(max_stop_len, s.size());
    bool stopped_by_stop_seq = false;

    // Tool call 累积状态
    bool in_tool_call = false;
    std::string tool_call_accum;
    std::string finish_reason = "stop";

    // per-request 队列已在 submit 前注册 (避免竞态)

    while (true) {
        // 客户端已断开时立即退出
        if (abort_flag && abort_flag->load(std::memory_order_relaxed)) break;

        InferResponse resp;
        if (poll_request(request_id, resp, 100)) {
            // 首 token 携带 cached_tokens 信息
            if (out_cached_tokens && resp.cached_tokens > 0) {
                *out_cached_tokens = resp.cached_tokens;
            }
            if (resp.error_code != 0 || resp.is_finished) break;

            int tid = resp.token_id;
            if (tid == tok.eos_token_id() || tid == tok.eot_id() || tid == tok.im_end_id())
                break;

            // ---- Thinking 标记处理 ----
            if (tid == tok.think_start_id()) {
                in_thinking = true;
                continue;
            }
            if (tid == tok.think_end_id()) {
                in_thinking = false;
                continue;
            }
            if (in_thinking) {
                // 如果有 reasoning 回调, 发送 thinking 内容; 否则丢弃
                if (on_reasoning) {
                    std::string piece = tok.decode(tid);
                    on_reasoning(piece);
                }
                count++;
                continue;
            }

            // ---- Tool call 标记处理 ----
            if (tid == tok.tool_call_start_id()) {
                in_tool_call = true;
                tool_call_accum.clear();
                continue;
            }
            if (tid == tok.tool_call_end_id()) {
                in_tool_call = false;
                finish_reason = "tool_calls";

                // 解析累积的 JSON: {"name": "...", "arguments": {...}}
                std::string trimmed = tool_call_accum;
                while (!trimmed.empty() && (trimmed.front() == '\n' || trimmed.front() == ' '))
                    trimmed.erase(trimmed.begin());
                while (!trimmed.empty() && (trimmed.back() == '\n' || trimmed.back() == ' '))
                    trimmed.pop_back();

                ToolCallInfo tc;
                tc.id = generate_id("call");
                tc.name = json_get_string(trimmed, "name");

                // 提取 arguments (JSON 对象, 非字符串)
                auto args_pos = trimmed.find("\"arguments\"");
                if (args_pos != std::string::npos) {
                    auto colon = trimmed.find(':', args_pos + 11);
                    if (colon != std::string::npos) {
                        size_t sp = colon + 1;
                        while (sp < trimmed.size() && trimmed[sp] == ' ') sp++;
                        if (sp < trimmed.size() && trimmed[sp] == '{') {
                            size_t ep = sp;
                            std::string args_obj = extract_json_object(trimmed, ep);
                            tc.arguments = args_obj;
                        }
                    }
                }

                if (on_tool_call) {
                    on_tool_call(tc);
                }
                continue;
            }
            if (in_tool_call) {
                tool_call_accum += tok.decode(tid);
                count++;
                continue;
            }

            // ---- 常规内容 ----
            std::string piece = tok.decode(tid);

            // Stop sequence 检查 (缓冲方式: 延迟发射直到确认不含stop)
            if (!stop_seqs.empty()) {
                pending_output += piece;
                count++;

                // 在缓冲区中搜索stop序列
                bool stopped = false;
                for (auto& stop : stop_seqs) {
                    auto pos = pending_output.find(stop);
                    if (pos != std::string::npos) {
                        // 只发射 stop 之前的部分
                        if (pos > 0) {
                            on_token(pending_output.substr(0, pos));
                        }
                        stopped = true;
                        stopped_by_stop_seq = true;
                        break;
                    }
                }
                if (stopped) break;

                // 发射安全前缀: 不可能是stop开头的部分
                // 保留最后 max_stop_len-1 字符做匹配缓冲
                if (pending_output.size() > max_stop_len) {
                    size_t safe = pending_output.size() - max_stop_len;
                    on_token(pending_output.substr(0, safe));
                    pending_output = pending_output.substr(safe);
                }
                continue;  // 跳过下面的 on_token
            }

            on_token(piece);
            count++;
        }

        if (!running_) break;
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(timeout_s))
            break;
    }

    // Flush pending buffer for non-stop exits (EOS, timeout, max_tokens, shutdown)
    if (!stop_seqs.empty() && !stopped_by_stop_seq && !pending_output.empty()) {
        on_token(pending_output);
        pending_output.clear();
    }

    // 输出 finish_reason
    if (out_finish_reason) *out_finish_reason = finish_reason;

    // 取消引擎中的请求 (如果超时或提前结束, 引擎可能还在生成)
    backend_.cancel(request_id);
    // 注销 per-request 队列
    unregister_request(request_id);
    return count;
}

// ---- API Handlers ----

void ServeApp::handle_health(const HttpRequest& /*req*/, int client_fd) {
    HttpResponse resp;
    resp.body = "{\"status\":\"ok\",\"model\":\"" + config_.model_name + "\"}";
    send_response(client_fd, resp);
}

void ServeApp::handle_models(const HttpRequest& /*req*/, int client_fd) {
    auto now = std::time(nullptr);
    HttpResponse resp;
    resp.body = std::string("{\"object\":\"list\",\"data\":[{\"id\":\"") + config_.model_name +
                "\",\"object\":\"model\",\"created\":" + std::to_string(now) +
                ",\"owned_by\":\"local\""
                ",\"permission\":[]"
                ",\"capabilities\":{\"reasoning\":true,\"tool_calling\":true,\"multimodal\":true,\"vision\":true,\"video\":true}"
                ",\"input_modalities\":[\"text\",\"image\",\"video\"]"
                ",\"output_modalities\":[\"text\"]"
                ",\"modalities\":[\"text\",\"image\",\"video\"]"
                "}]}";
    send_response(client_fd, resp);
}

void ServeApp::handle_model_retrieve(const HttpRequest& req, int client_fd) {
    // Extract model ID from path: /v1/models/{model}
    std::string model_id = req.path.substr(11);  // skip "/v1/models/"
    // URL decode %xx
    std::string decoded;
    for (size_t i = 0; i < model_id.size(); i++) {
        if (model_id[i] == '%' && i + 2 < model_id.size()) {
            int hi = 0, lo = 0;
            char c1 = model_id[i+1], c2 = model_id[i+2];
            if (c1 >= '0' && c1 <= '9') hi = c1 - '0';
            else if (c1 >= 'a' && c1 <= 'f') hi = c1 - 'a' + 10;
            else if (c1 >= 'A' && c1 <= 'F') hi = c1 - 'A' + 10;
            if (c2 >= '0' && c2 <= '9') lo = c2 - '0';
            else if (c2 >= 'a' && c2 <= 'f') lo = c2 - 'a' + 10;
            else if (c2 >= 'A' && c2 <= 'F') lo = c2 - 'A' + 10;
            decoded += (char)((hi << 4) | lo);
            i += 2;
        } else {
            decoded += model_id[i];
        }
    }

    // Check if the requested model matches our loaded model
    bool model_ok =
        decoded == config_.model_name ||
        decoded == config_.model_name + ":latest";

    if (!model_ok) {
        HttpResponse resp;
        resp.status_code = 404;
        resp.status_text = "Not Found";
        resp.body = "{\"error\":{\"message\":\"The model '"+json_escape(decoded)+"' does not exist\","
                    "\"type\":\"invalid_request_error\",\"code\":\"model_not_found\"}}";
        send_response(client_fd, resp);
        return;
    }

    auto now = std::time(nullptr);
    HttpResponse resp;
    resp.body = "{\"id\":\"" + config_.model_name +
                "\",\"object\":\"model\",\"created\":" + std::to_string(now) +
                ",\"owned_by\":\"local\""
                ",\"permission\":[]"
                ",\"capabilities\":{\"reasoning\":true,\"tool_calling\":true,\"multimodal\":true,\"vision\":true,\"video\":true}"
                ",\"input_modalities\":[\"text\",\"image\",\"video\"]"
                ",\"output_modalities\":[\"text\"]"
                ",\"modalities\":[\"text\",\"image\",\"video\"]"
                "}";
    send_response(client_fd, resp);
}

void ServeApp::handle_ollama_tags(const HttpRequest& /*req*/, int client_fd) {
    std::string name = config_.model_name + ":latest";
    HttpResponse resp;
    resp.body = "{\"models\":[{"
        "\"name\":\"" + name + "\","
        "\"model\":\"" + name + "\","
        "\"modified_at\":\"2025-05-10T00:00:00Z\","
        "\"size\":54000000000,"
        "\"digest\":\"sha256:0000000000000000000000000000000000000000000000000000000000000000\","
        "\"details\":{"
            "\"parent_model\":\"\","
            "\"format\":\"safetensors\","
            "\"family\":\"qwen3\","
            "\"families\":[\"qwen3\"],"
            "\"parameter_size\":\"27B\","
            "\"quantization_level\":\"BF16\""
        "},"
        "\"capabilities\":[\"completion\",\"vision\",\"video\",\"tools\",\"thinking\"]"
    "}]}";
    send_response(client_fd, resp);
}

void ServeApp::handle_ollama_show(const HttpRequest& req, int client_fd) {
    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;
    std::string name = config_.model_name + ":latest";

    HttpResponse resp;
    resp.body = "{"
        "\"modelfile\":\"# Modelfile for " + config_.model_name + "\\n\","
        "\"parameters\":\"temperature 1.0\\ntop_p 0.95\\ntop_k 20\\n\","
        "\"template\":\"{{- range .Messages }}{{ .Role }}: {{ .Content }}\\n{{- end }}\","
        "\"details\":{"
            "\"parent_model\":\"\","
            "\"format\":\"safetensors\","
            "\"family\":\"qwen3\","
            "\"families\":[\"qwen3\"],"
            "\"parameter_size\":\"27B\","
            "\"quantization_level\":\"BF16\""
        "},"
        "\"model_info\":{"
            "\"general.architecture\":\"qwen3\","
            "\"general.parameter_count\":27000000000,"
            "\"general.file_type\":\"BF16\","
            "\"qwen3.vision\":true,"
            "\"qwen3.vision.image_size\":1024,"
            "\"qwen3.vision.patch_size\":16,"
            "\"qwen3.vision.num_layers\":27"
        "},"
        "\"capabilities\":[\"completion\",\"vision\",\"video\",\"tools\",\"thinking\"],"
        "\"modified_at\":\"2025-05-10T00:00:00Z\""
    "}";
    send_response(client_fd, resp);
}

void ServeApp::handle_ollama_ps(const HttpRequest& /*req*/, int client_fd) {
    std::string name = config_.model_name + ":latest";
    HttpResponse resp;
    resp.body = "{\"models\":[{"
        "\"name\":\"" + name + "\","
        "\"model\":\"" + name + "\","
        "\"size\":54000000000,"
        "\"digest\":\"sha256:0000000000000000000000000000000000000000000000000000000000000000\","
        "\"details\":{"
            "\"parent_model\":\"\","
            "\"format\":\"safetensors\","
            "\"family\":\"qwen3\","
            "\"families\":[\"qwen3\"],"
            "\"parameter_size\":\"27B\","
            "\"quantization_level\":\"BF16\""
        "},"
        "\"expires_at\":\"2099-12-31T23:59:59Z\","
        "\"size_vram\":54000000000"
    "}]}";
    send_response(client_fd, resp);
}

void ServeApp::handle_ollama_version(const HttpRequest& /*req*/, int client_fd) {
    HttpResponse resp;
    resp.body = "{\"version\":\"0.9.0\"}";
    send_response(client_fd, resp);
}

void ServeApp::handle_openai_chat(const HttpRequest& req, int client_fd) {
    std::cerr << "[Serve] handle_openai_chat: body=" << req.body.size() << std::endl;
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":{"message":"Tokenizer not loaded","type":"server_error"}})";
        send_response(client_fd, resp);
        return;
    }

    std::vector<ImageData> images;
    std::vector<VideoData> videos;
    fprintf(stderr, "[Serve] calling parse_messages (body=%zu)...\n", req.body.size());
    fflush(stderr);
    auto messages = parse_messages(req.body, &images, &videos);
    fprintf(stderr, "[Serve] parse_messages done: %zu msgs, %zu imgs, %zu vids\n",
            messages.size(), images.size(), videos.size());
    fflush(stderr);

    bool stream = json_get_bool(req.body, "stream", false);
    int max_tokens = json_get_int(req.body, "max_tokens", 4096);
    // 支持 max_completion_tokens (新标准命名)
    if (req.body.find("\"max_completion_tokens\"") != std::string::npos)
        max_tokens = json_get_int(req.body, "max_completion_tokens", max_tokens);
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);

    // ---- Thinking mode 控制 ----
    // 支持多种参数名: "think", "enable_thinking" (vLLM/SGLang chat_template_kwargs)
    // 以及 OpenAI 标准 "reasoning_effort" (none=禁用, low/medium/high=启用)
    bool enable_thinking = false;
    if (req.body.find("\"think\"") != std::string::npos)
        enable_thinking = json_get_bool(req.body, "think", true);
    else if (req.body.find("\"enable_thinking\"") != std::string::npos)
        enable_thinking = json_get_bool(req.body, "enable_thinking", true);
    if (req.body.find("\"reasoning_effort\"") != std::string::npos) {
        std::string effort = json_get_string(req.body, "reasoning_effort");
        if (effort == "none") enable_thinking = false;
    }

    // 根据 thinking 模式设定默认采样参数 (Qwen3.5 官方推荐)
    float def_temp = enable_thinking ? 1.0f : 0.7f;
    float def_top_p = enable_thinking ? 0.95f : 0.8f;
    float temperature = (float)json_get_number(req.body, "temperature", def_temp);
    float top_p = (float)json_get_number(req.body, "top_p", def_top_p);
    int top_k = json_get_int(req.body, "top_k", 20);

    // ---- 解析 stop 序列 ----
    std::vector<std::string> stop_seqs;
    {
        std::string single_stop = json_get_string(req.body, "stop");
        if (!single_stop.empty()) {
            stop_seqs.push_back(single_stop);
        } else {
            auto spos = req.body.find("\"stop\"");
            if (spos != std::string::npos) {
                auto arr_start = req.body.find('[', spos);
                auto arr_end = req.body.find(']', arr_start);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr = req.body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t p = 0;
                    while (p < arr.size()) {
                        auto q1 = arr.find('"', p);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        stop_seqs.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        p = q2 + 1;
                    }
                }
            }
        }
    }

    // ---- 解析 tools (Function Calling) ----
    std::string tools_json = extract_tools_json(req.body);
    bool has_tools = !tools_json.empty();
    if (has_tools) {
        // 将 OpenAI tools 定义注入到 system message 中 (Qwen3.5 官方格式)
        std::string tool_prompt = build_tool_system_prompt(tools_json);
        // 找到第一个 system message 并追加; 如果没有则创建一个
        bool found_system = false;
        for (auto& [role, content] : messages) {
            if (role == "system") {
                content += tool_prompt;
                found_system = true;
                break;
            }
        }
        if (!found_system) {
            messages.insert(messages.begin(),
                {"system", std::string(DEFAULT_SYSTEM_PROMPT) + tool_prompt});
        }
    }

    // stream_options.include_usage
    bool include_usage = false;
    if (req.body.find("\"stream_options\"") != std::string::npos) {
        auto so_pos = req.body.find("\"stream_options\"");
        auto so_obj_start = req.body.find('{', so_pos + 16);
        if (so_obj_start != std::string::npos) {
            size_t so_end = so_obj_start;
            std::string so_obj = extract_json_object(req.body, so_end);
            include_usage = json_get_bool(so_obj, "include_usage", false);
        }
    }

    int64_t seed = (int64_t)json_get_number(req.body, "seed", -1);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty  = (float)json_get_number(req.body, "presence_penalty", 1.5);
    float repeat_penalty    = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float min_p             = (float)json_get_number(req.body, "min_p", 0.0);
    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;
    std::string req_id = generate_id("chatcmpl");

    if (messages.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.status_text = "Bad Request";
        resp.body = R"({"error":{"message":"No messages provided","type":"invalid_request_error"}})";
        send_response(client_fd, resp);
        return;
    }

    // Tokenize via chat template
    // 确保有 system prompt — Qwen3.5 在无 system prompt 时行为显著下降
    {
        bool has_system = false;
        for (auto& [role, content] : messages) {
            if (role == "system") { has_system = true; break; }
        }
        if (!has_system) {
            messages.insert(messages.begin(),
                {"system", DEFAULT_SYSTEM_PROMPT});
        }
    }
    auto prompt_tokens = tok.apply_chat_template(messages, true, enable_thinking);

    // Expand image placeholders if multimodal content was found
    if (!images.empty()) {
        expand_image_placeholders(prompt_tokens, images, tok);
    }
    // Expand video placeholders if video content was found
    if (!videos.empty()) {
        expand_video_placeholders(prompt_tokens, videos, tok);
    }
    int prompt_count = (int)prompt_tokens.size();
    {
        int n_img_pad = 0, n_vid_pad = 0;
        for (int t : prompt_tokens) {
            if (t == 248056) n_img_pad++;
            else if (t == 248057) n_vid_pad++;
        }
        std::cerr << "[Serve] Final prompt: tokens=" << prompt_count
                  << " image_pad=" << n_img_pad << " video_pad=" << n_vid_pad
                  << " images_to_encode=" << images.size() << std::endl;
        std::cerr.flush();
    }

    // Submit inference request
    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens = max_tokens;
    infer_req.temperature    = temperature;
    infer_req.top_p          = top_p;
    infer_req.top_k          = top_k;
    infer_req.min_p          = min_p;
    infer_req.repeat_penalty = repeat_penalty;
    infer_req.frequency_penalty = frequency_penalty;
    infer_req.presence_penalty  = presence_penalty;
    infer_req.seed           = seed;
    infer_req.stream         = true;
    infer_req.images         = std::move(images);
    infer_req.videos         = std::move(videos);

    // 先注册队列,再提交请求 (避免引擎响应早于队列注册导致丢弃)
    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":{"message":"Request queue full","type":"server_error"}})";
        send_response(client_fd, resp);
        return;
    }

    if (stream) {
        // SSE streaming response
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/event-stream; charset=utf-8\r\n"
                             "Cache-Control: no-cache\r\n"
                             "Connection: keep-alive\r\n"
                             "Access-Control-Allow-Origin: *\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto now_t = (int64_t)std::time(nullptr);

        // Initial role chunk
        std::string role_chunk = "{\"id\":\"" + req_id + "\",\"object\":\"chat.completion.chunk\","
            "\"created\":" + std::to_string(now_t) + ",\"model\":\"" + json_escape(model) +
            "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
            ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
        send_sse_event(client_fd, role_chunk);

        // Stream tokens with reasoning + tool call support
        std::string finish_reason;
        int tool_call_idx = 0;
        std::atomic<bool> client_disconnected{false};
        int cached_tokens = 0;

        int comp_toks = poll_tokens(infer_req.request_id,
            // on_token (content)
            [&](const std::string& piece) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                if (!send_sse_event(client_fd, make_chat_chunk(model, piece, "", req_id, now_t))) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
            },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            // on_reasoning
            [&](const std::string& piece) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                if (!send_sse_event(client_fd, make_chat_reasoning_chunk(model, piece, req_id, now_t))) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
            },
            // on_tool_call
            [&](const ToolCallInfo& tc) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                if (!send_sse_event(client_fd, make_chat_tool_call_chunk(model, tc, tool_call_idx++, req_id, now_t))) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
            },
            &finish_reason,
            &client_disconnected,
            &cached_tokens
        );

        // Finish chunk
        std::string finish_chunk = "{\"id\":\"" + req_id + "\",\"object\":\"chat.completion.chunk\","
            "\"created\":" + std::to_string(now_t) + ",\"model\":\"" + json_escape(model) +
            "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
            ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + finish_reason + "\"}]}";
        send_sse_event(client_fd, finish_chunk);

        // Usage chunk (如果 stream_options.include_usage = true)
        if (include_usage) {
            int total = prompt_count + comp_toks;
            std::string usage_chunk = "{\"id\":\"" + req_id + "\",\"object\":\"chat.completion.chunk\","
                "\"created\":" + std::to_string(now_t) + ",\"model\":\"" + json_escape(model) +
                "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
                ",\"choices\":[],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
                ",\"completion_tokens\":" + std::to_string(comp_toks) +
                ",\"total_tokens\":" + std::to_string(total) +
                ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(cached_tokens) + "}}}";
            send_sse_event(client_fd, usage_chunk);
        }

        send_sse_done(client_fd);
    } else {
        // Non-streaming: collect all tokens, reasoning, and tool calls
        std::string content;
        std::string reasoning;
        std::vector<ToolCallInfo> tool_calls;
        std::string finish_reason;
        int cached_tokens = 0;

        int comp_toks = poll_tokens(infer_req.request_id,
            [&](const std::string& piece) { content += piece; },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            [&](const std::string& piece) { reasoning += piece; },
            [&](const ToolCallInfo& tc) { tool_calls.push_back(tc); },
            &finish_reason,
            nullptr,
            &cached_tokens
        );

        auto now_t = std::time(nullptr);
        int total = prompt_count + comp_toks;

        // 构建 message 对象
        std::string msg_body = "\"role\":\"assistant\"";

        // reasoning_content (thinking 内容, 独立于 content)
        if (!reasoning.empty()) {
            msg_body += ",\"reasoning_content\":\"" + json_escape(reasoning) + "\"";
        }

        // content 和 tool_calls
        if (!tool_calls.empty()) {
            msg_body += ",\"content\":null";
            msg_body += ",\"tool_calls\":[";
            for (size_t i = 0; i < tool_calls.size(); i++) {
                if (i > 0) msg_body += ",";
                msg_body += "{\"id\":\"" + tool_calls[i].id + "\",\"type\":\"function\","
                    "\"function\":{\"name\":\"" + json_escape(tool_calls[i].name) + "\","
                    "\"arguments\":\"" + json_escape(tool_calls[i].arguments) + "\"}}";
            }
            msg_body += "]";
        } else {
            msg_body += ",\"content\":\"" + json_escape(content) + "\"";
        }

        HttpResponse http_resp;
        http_resp.body = "{\"id\":\"" + req_id +
            "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(now_t) +
            ",\"model\":\"" + json_escape(model) +
            "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
            ",\"choices\":[{\"index\":0,\"message\":{" + msg_body + "},"
            "\"logprobs\":null,"
            "\"finish_reason\":\"" + finish_reason + "\"}],"
            "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
            ",\"completion_tokens\":" + std::to_string(comp_toks) +
            ",\"total_tokens\":" + std::to_string(total) +
            ",\"prompt_tokens_details\":{\"cached_tokens\":" + std::to_string(cached_tokens) + "}}}";
        send_response(client_fd, http_resp);
    }
}

void ServeApp::handle_openai_responses(const HttpRequest& req, int client_fd) {
    // 最小兼容层: 将 /v1/responses 请求转换为 /v1/chat/completions 处理。
    // 支持:
    //  1) input 为字符串
    //  2) input 为 messages 数组
    //  3) input 为 content parts 数组 (input_text/input_image/input_video)

    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;

    bool stream = json_get_bool(req.body, "stream", false);
    int max_tokens = json_get_int(req.body, "max_output_tokens", -1);
    if (max_tokens <= 0) max_tokens = json_get_int(req.body, "max_tokens", 4096);
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);

    std::string messages_json;
    std::string direct_messages = extract_json_raw_value(req.body, "messages");
    if (!direct_messages.empty() && direct_messages[0] == '[') {
        messages_json = direct_messages;
    } else {
        std::string input_raw = extract_json_raw_value(req.body, "input");
        if (input_raw.empty()) {
            HttpResponse resp;
            resp.status_code = 400;
            resp.status_text = "Bad Request";
            resp.body = R"({"error":{"message":"No input provided","type":"invalid_request_error"}})";
            send_response(client_fd, resp);
            return;
        }

        if (input_raw[0] == '"') {
            std::string input_text = json_get_string(req.body, "input");
            messages_json = "[{\"role\":\"user\",\"content\":\"" + json_escape(input_text) + "\"}]";
        } else if (input_raw[0] == '[') {
            if (input_raw.find("\"role\"") != std::string::npos) {
                messages_json = input_raw;
            } else {
                messages_json = "[{\"role\":\"user\",\"content\":" + input_raw + "}]";
            }
        } else if (input_raw[0] == '{') {
            if (input_raw.find("\"role\"") != std::string::npos) {
                messages_json = "[" + input_raw + "]";
            } else {
                messages_json = "[{\"role\":\"user\",\"content\":[" + input_raw + "]}]";
            }
        } else {
            HttpResponse resp;
            resp.status_code = 400;
            resp.status_text = "Bad Request";
            resp.body = R"({"error":{"message":"Unsupported input format","type":"invalid_request_error"}})";
            send_response(client_fd, resp);
            return;
        }
    }

    std::string chat_body = "{";
    chat_body += "\"model\":\"" + json_escape(model) + "\"";
    chat_body += ",\"stream\":" + std::string(stream ? "true" : "false");
    chat_body += ",\"max_tokens\":" + std::to_string(max_tokens);
    chat_body += ",\"messages\":" + messages_json;

    auto append_raw = [&](const std::string& src_key, const std::string& dst_key = "") {
        std::string raw = extract_json_raw_value(req.body, src_key);
        if (!raw.empty()) {
            chat_body += ",\"" + (dst_key.empty() ? src_key : dst_key) + "\":" + raw;
        }
    };

    append_raw("temperature");
    append_raw("top_p");
    append_raw("top_k");
    append_raw("seed");
    append_raw("frequency_penalty");
    append_raw("presence_penalty");
    append_raw("repeat_penalty");
    append_raw("min_p");
    append_raw("stop");
    append_raw("tools");
    append_raw("reasoning_effort");
    append_raw("think");
    append_raw("enable_thinking");
    chat_body += "}";

    HttpRequest chat_req = req;
    chat_req.body = std::move(chat_body);
    handle_openai_chat(chat_req, client_fd);
}

void ServeApp::handle_openai_completions(const HttpRequest& req, int client_fd) {
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":{"message":"Tokenizer not loaded","type":"server_error"}})";
        send_response(client_fd, resp);
        return;
    }

    std::string prompt = json_get_string(req.body, "prompt");
    bool stream = json_get_bool(req.body, "stream", false);
    int max_tokens = json_get_int(req.body, "max_tokens", 4096);
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);
    float temperature = (float)json_get_number(req.body, "temperature", 1.0);
    float top_p = (float)json_get_number(req.body, "top_p", 0.95);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float min_p = (float)json_get_number(req.body, "min_p", 0.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty = (float)json_get_number(req.body, "presence_penalty", 1.5);
    int64_t seed = (int64_t)json_get_number(req.body, "seed", -1);
    bool enable_thinking = false;  // raw completions 不使用 chat template
    // 解析 stop 序列 (支持单字符串和数组)
    std::vector<std::string> stop_seqs;
    {
        auto stop_pos = req.body.find("\"stop\"");
        if (stop_pos != std::string::npos) {
            auto arr_start = req.body.find('[', stop_pos);
            auto str_start = req.body.find('"', stop_pos + 6);
            // 判断是数组还是单字符串 (哪个先出现)
            if (arr_start != std::string::npos && (str_start == std::string::npos || arr_start < str_start)) {
                auto arr_end = req.body.find(']', arr_start);
                if (arr_end != std::string::npos) {
                    std::string arr = req.body.substr(arr_start + 1, arr_end - arr_start - 1);
                    size_t p = 0;
                    while (p < arr.size()) {
                        auto q1 = arr.find('"', p);
                        if (q1 == std::string::npos) break;
                        auto q2 = arr.find('"', q1 + 1);
                        if (q2 == std::string::npos) break;
                        stop_seqs.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                        p = q2 + 1;
                    }
                }
            } else {
                std::string single_stop = json_get_string(req.body, "stop");
                if (!single_stop.empty()) stop_seqs.push_back(single_stop);
            }
        }
    }
    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;
    std::string req_id = generate_id("cmpl");

    // Tokenize raw prompt
    auto prompt_tokens = tok.encode(prompt);
    int prompt_count = (int)prompt_tokens.size();

    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens    = max_tokens;
    infer_req.temperature       = temperature;
    infer_req.top_p             = top_p;
    infer_req.top_k             = top_k;
    infer_req.min_p             = min_p;
    infer_req.repeat_penalty    = repeat_penalty;
    infer_req.frequency_penalty = frequency_penalty;
    infer_req.presence_penalty  = presence_penalty;
    infer_req.seed              = seed;
    infer_req.stream            = true;

    // 先注册队列,再提交请求
    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":{"message":"Request queue full","type":"server_error"}})";
        send_response(client_fd, resp);
        return;
    }

    if (stream) {
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/event-stream; charset=utf-8\r\n"
                             "Cache-Control: no-cache\r\n"
                             "Connection: keep-alive\r\n"
                             "Access-Control-Allow-Origin: *\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto now_t = (int64_t)std::time(nullptr);
        std::atomic<bool> client_disconnected{false};
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            if (client_disconnected.load(std::memory_order_relaxed)) return;
            if (!send_sse_event(client_fd, make_completion_chunk(model, piece, "", req_id, now_t))) {
                client_disconnected.store(true, std::memory_order_relaxed);
                backend_.cancel(infer_req.request_id);
            }
        }, config_.timeout_s, enable_thinking, stop_seqs, {}, {}, nullptr, &client_disconnected);

        send_sse_event(client_fd, make_completion_chunk(model, "", "stop", req_id, now_t));
        send_sse_done(client_fd);
    } else {
        std::string text;
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            text += piece;
        }, config_.timeout_s, enable_thinking, stop_seqs);

        auto now_t = std::time(nullptr);
        int total = prompt_count + comp_toks;
        HttpResponse http_resp;
        http_resp.body = "{\"id\":\"" + req_id +
            "\",\"object\":\"text_completion\",\"created\":" + std::to_string(now_t) +
            ",\"model\":\"" + json_escape(model) +
            "\",\"choices\":[{\"text\":\"" + json_escape(text) + "\","
            "\"index\":0,\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_count) +
            ",\"completion_tokens\":" + std::to_string(comp_toks) +
            ",\"total_tokens\":" + std::to_string(total) +
            ",\"prompt_tokens_details\":{\"cached_tokens\":0}}}";
        send_response(client_fd, http_resp);
    }
}

void ServeApp::handle_ollama_generate(const HttpRequest& req, int client_fd) {
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":"Tokenizer not loaded"})";
        send_response(client_fd, resp);
        return;
    }

    std::string prompt = json_get_string(req.body, "prompt");
    std::string system = json_get_string(req.body, "system");
    bool stream = json_get_bool(req.body, "stream", true);  // Ollama 默认流式
    int max_tokens = json_get_int(req.body, "num_predict", 4096);
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);
    bool enable_thinking = json_get_bool(req.body, "think", false);
    float def_temp = enable_thinking ? 1.0f : 0.7f;
    float def_top_p = enable_thinking ? 0.95f : 0.8f;
    float temperature = (float)json_get_number(req.body, "temperature", def_temp);
    float top_p = (float)json_get_number(req.body, "top_p", def_top_p);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float min_p = (float)json_get_number(req.body, "min_p", 0.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty = (float)json_get_number(req.body, "presence_penalty", 1.5);
    int64_t seed = (int64_t)json_get_number(req.body, "seed", -1);
    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;
    std::vector<std::string> stop_seqs;

    // Parse options sub-object (Ollama standard)
    {
        auto opts_pos = req.body.find("\"options\"");
        if (opts_pos != std::string::npos) {
            auto obj_start = req.body.find('{', opts_pos + 9);
            if (obj_start != std::string::npos) {
                std::string opts_str = extract_json_object(req.body, obj_start);
                if (!opts_str.empty()) {
                    if (opts_str.find("\"num_predict\"") != std::string::npos) {
                        int np = json_get_int(opts_str, "num_predict", -1);
                        if (np > 0) max_tokens = np;
                    }
                    if (opts_str.find("\"temperature\"") != std::string::npos)
                        temperature = (float)json_get_number(opts_str, "temperature", temperature);
                    if (opts_str.find("\"top_p\"") != std::string::npos)
                        top_p = (float)json_get_number(opts_str, "top_p", top_p);
                    if (opts_str.find("\"top_k\"") != std::string::npos)
                        top_k = json_get_int(opts_str, "top_k", top_k);
                    if (opts_str.find("\"repeat_penalty\"") != std::string::npos)
                        repeat_penalty = (float)json_get_number(opts_str, "repeat_penalty", repeat_penalty);
                    if (opts_str.find("\"min_p\"") != std::string::npos)
                        min_p = (float)json_get_number(opts_str, "min_p", min_p);
                    if (opts_str.find("\"frequency_penalty\"") != std::string::npos)
                        frequency_penalty = (float)json_get_number(opts_str, "frequency_penalty", frequency_penalty);
                    if (opts_str.find("\"presence_penalty\"") != std::string::npos)
                        presence_penalty = (float)json_get_number(opts_str, "presence_penalty", presence_penalty);
                    if (opts_str.find("\"seed\"") != std::string::npos)
                        seed = (int64_t)json_get_number(opts_str, "seed", (double)seed);
                    // Ollama stop 序列
                    if (opts_str.find("\"stop\"") != std::string::npos) {
                        auto arr_pos = opts_str.find("\"stop\"");
                        auto arr_start = opts_str.find('[', arr_pos);
                        if (arr_start != std::string::npos) {
                            auto arr_end = opts_str.find(']', arr_start);
                            if (arr_end != std::string::npos) {
                                std::string arr = opts_str.substr(arr_start + 1, arr_end - arr_start - 1);
                                size_t p = 0;
                                while (p < arr.size()) {
                                    auto q1 = arr.find('"', p);
                                    if (q1 == std::string::npos) break;
                                    auto q2 = arr.find('"', q1 + 1);
                                    if (q2 == std::string::npos) break;
                                    stop_seqs.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                                    p = q2 + 1;
                                }
                            }
                        } else {
                            std::string s = json_get_string(opts_str, "stop");
                            if (!s.empty()) stop_seqs.push_back(s);
                        }
                    }
                }
            }
        }
    }
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);
    // 顶级 stop 参数
    {
        std::string single_stop = json_get_string(req.body, "stop");
        if (!single_stop.empty() && stop_seqs.empty()) stop_seqs.push_back(single_stop);
    }

    // Parse Ollama-format images: top-level "images": ["base64data", ...]
    std::vector<ImageData> images;
    {
        auto img_pos = req.body.find("\"images\"");
        if (img_pos != std::string::npos) {
            auto img_arr = req.body.find('[', img_pos);
            if (img_arr != std::string::npos) {
                size_t ip = img_arr + 1;
                while (ip < req.body.size() && req.body[ip] != ']') {
                    while (ip < req.body.size() && (req.body[ip] == ' ' || req.body[ip] == ','
                           || req.body[ip] == '\n' || req.body[ip] == '\r' || req.body[ip] == '\t'))
                        ip++;
                    if (ip >= req.body.size() || req.body[ip] == ']') break;
                    if (req.body[ip] == '"') {
                        size_t start_q = ip + 1;
                        ip = start_q;
                        while (ip < req.body.size() && req.body[ip] != '"') {
                            if (req.body[ip] == '\\') ip++;
                            ip++;
                        }
                        std::string b64_str = req.body.substr(start_q, ip - start_q);
                        ip++; // skip closing quote
                        auto img = decode_image_base64(b64_str);
                        if (img.width > 0) {
                            images.push_back(std::move(img));
                        }
                    } else {
                        ip++;
                    }
                }
            }
        }
    }

    // Parse Ollama-format videos: top-level "videos": [{"video":["base64_frame",...],"fps":24}, ...]
    std::vector<VideoData> videos;
    {
        auto vids_pos = req.body.find("\"videos\"");
        if (vids_pos != std::string::npos) {
            auto vids_arr = req.body.find('[', vids_pos);
            if (vids_arr != std::string::npos) {
                size_t vp = vids_arr + 1;
                while (vp < req.body.size() && req.body[vp] != ']') {
                    while (vp < req.body.size() && (req.body[vp] == ' ' || req.body[vp] == ','
                           || req.body[vp] == '\n' || req.body[vp] == '\r' || req.body[vp] == '\t'))
                        vp++;
                    if (vp >= req.body.size() || req.body[vp] == ']') break;

                    if (req.body[vp] == '{') {
                        std::string vobj = extract_json_object(req.body, vp);
                        VideoData vd;
                        vd.source_fps = (float)json_get_number(vobj, "fps", 24.0);

                        auto video_pos = vobj.find("\"video\"");
                        if (video_pos == std::string::npos) {
                            video_pos = vobj.find("\"frames\"");
                        }

                        if (video_pos != std::string::npos) {
                            auto frame_arr = vobj.find('[', video_pos);
                            if (frame_arr != std::string::npos) {
                                size_t fp = frame_arr + 1;
                                while (fp < vobj.size() && vobj[fp] != ']') {
                                    while (fp < vobj.size() && (vobj[fp] == ' ' || vobj[fp] == ','
                                           || vobj[fp] == '\n' || vobj[fp] == '\r' || vobj[fp] == '\t'))
                                        fp++;
                                    if (fp >= vobj.size() || vobj[fp] == ']') break;

                                    if (vobj[fp] == '"') {
                                        size_t start_q = fp + 1;
                                        fp = start_q;
                                        while (fp < vobj.size() && vobj[fp] != '"') {
                                            if (vobj[fp] == '\\') fp++;
                                            fp++;
                                        }
                                        std::string b64_frame = vobj.substr(start_q, fp - start_q);
                                        fp++; // skip closing quote

                                        auto frame_img = decode_image_base64(b64_frame);
                                        if (frame_img.width > 0) {
                                            if (vd.width == 0) {
                                                vd.width = frame_img.width;
                                                vd.height = frame_img.height;
                                            }
                                            vd.frames.push_back(std::move(frame_img.pixels));
                                        }
                                    } else {
                                        fp++;
                                    }
                                }
                            }
                        }

                        if (!vd.frames.empty()) {
                            videos.push_back(std::move(vd));
                        }
                    } else {
                        vp++;
                    }
                }
            }
        }
    }

    // Tokenize: if system provided, use chat template; otherwise raw encode
    std::vector<int> prompt_tokens;
    if (!system.empty() || !images.empty() || !videos.empty()) {
        // Use chat template for image support (need special tokens)
        std::vector<std::pair<std::string, std::string>> messages;
        if (!system.empty())
            messages.emplace_back("system", system);
        else
            messages.emplace_back("system", DEFAULT_SYSTEM_PROMPT);
        std::string user_content;
        for (size_t i = 0; i < images.size(); i++)
            user_content += "<|vision_start|><|image_pad|><|vision_end|>";
        for (auto& vid : videos) {
            core::VisionConfig vcfg;
            int num_frames = (int)vid.frames.size();
            int target_frames = num_frames;
            float target_fps = 2.0f;
            if (vid.source_fps > 0)
                target_frames = (int)(num_frames / vid.source_fps * target_fps);
            target_frames = std::max(4, std::min(target_frames, 768));
            target_frames = std::min(target_frames, num_frames);
            auto [grid_t, grid_h, grid_w] = core::VisionEncoder::compute_video_grid(
                target_frames, vid.height, vid.width, vcfg);

            for (int gt = 0; gt < grid_t; gt++) {
                int f0 = gt * 2, f1 = gt * 2 + 1;
                float t0 = (f0 < target_frames && vid.source_fps > 0) ? (float)f0 / target_fps : 0;
                float t1 = (f1 < target_frames && vid.source_fps > 0) ? (float)f1 / target_fps : t0;
                char buf[32];
                snprintf(buf, sizeof(buf), "<%.1f seconds>", (t0 + t1) / 2.0f);
                user_content += buf;
                user_content += "<|vision_start|><|video_pad|><|vision_end|>";
            }
        }
        user_content += prompt;
        messages.emplace_back("user", user_content);
        prompt_tokens = tok.apply_chat_template(messages, true, enable_thinking);
    } else {
        // 纯文本也走 chat template — 确保有 system prompt
        std::vector<std::pair<std::string, std::string>> messages;
        messages.emplace_back("system", DEFAULT_SYSTEM_PROMPT);
        messages.emplace_back("user", prompt);
        prompt_tokens = tok.apply_chat_template(messages, true, enable_thinking);
    }

    // Expand image placeholders
    if (!images.empty()) {
        expand_image_placeholders(prompt_tokens, images, tok);
    }
    if (!videos.empty()) {
        expand_video_placeholders(prompt_tokens, videos, tok);
    }

    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens    = max_tokens;
    infer_req.temperature       = temperature;
    infer_req.top_p             = top_p;
    infer_req.top_k             = top_k;
    infer_req.min_p             = min_p;
    infer_req.repeat_penalty    = repeat_penalty;
    infer_req.frequency_penalty = frequency_penalty;
    infer_req.presence_penalty  = presence_penalty;
    infer_req.seed              = seed;
    infer_req.stream            = true;
    infer_req.images            = std::move(images);
    infer_req.videos            = std::move(videos);

    // 先注册队列,再提交请求
    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":"Request queue full"})";
        send_response(client_fd, resp);
        return;
    }

    if (stream) {
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/x-ndjson; charset=utf-8\r\n"
                             "Transfer-Encoding: chunked\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto t0 = std::chrono::steady_clock::now();
        std::string finish_reason;
        std::atomic<bool> client_disconnected{false};
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            if (client_disconnected.load(std::memory_order_relaxed)) return;
            std::string line = "{\"model\":\"" + json_escape(model) +
                "\",\"created_at\":\"" + iso8601_now() +
                "\",\"response\":\"" + json_escape(piece) +
                "\",\"done\":false}";
            if (!send_ndjson_chunk(client_fd, line)) {
                client_disconnected.store(true, std::memory_order_relaxed);
                backend_.cancel(infer_req.request_id);
            }
        }, config_.timeout_s, enable_thinking, stop_seqs, {}, {}, &finish_reason, &client_disconnected);

        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        auto eval_ns = comp_toks > 0 ? dur_ns : 0;
        std::string done_line = "{\"model\":\"" + json_escape(model) +
            "\",\"created_at\":\"" + iso8601_now() +
            "\",\"response\":\"\",\"done\":true"
            ",\"total_duration\":" + std::to_string(dur_ns) +
            ",\"eval_count\":" + std::to_string(comp_toks) +
            ",\"eval_duration\":" + std::to_string(eval_ns) + "}";
        send_ndjson_chunk(client_fd, done_line);
        send_chunked_end(client_fd);
    } else {
        auto t0 = std::chrono::steady_clock::now();
        std::string response;
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            response += piece;
        }, config_.timeout_s, enable_thinking, stop_seqs);

        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        auto eval_ns = comp_toks > 0 ? dur_ns : 0;
        HttpResponse http_resp;
        http_resp.body = "{\"model\":\"" + json_escape(model) +
            "\",\"created_at\":\"" + iso8601_now() +
            "\",\"response\":\"" + json_escape(response) +
            "\",\"done\":true"
            ",\"total_duration\":" + std::to_string(dur_ns) +
            ",\"eval_count\":" + std::to_string(comp_toks) +
            ",\"eval_duration\":" + std::to_string(eval_ns) + "}";
        send_response(client_fd, http_resp);
    }
}

void ServeApp::handle_ollama_chat(const HttpRequest& req, int client_fd) {
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":"Tokenizer not loaded"})";
        send_response(client_fd, resp);
        return;
    }

    std::vector<ImageData> images;
    std::vector<VideoData> videos;
    auto messages = parse_messages(req.body, &images, &videos);

    bool stream = json_get_bool(req.body, "stream", true);
    int max_tokens = json_get_int(req.body, "num_predict", 4096);
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);
    bool enable_thinking = json_get_bool(req.body, "think", false);
    float def_temp = enable_thinking ? 1.0f : 0.7f;
    float def_top_p = enable_thinking ? 0.95f : 0.8f;
    float temperature = (float)json_get_number(req.body, "temperature", def_temp);
    float top_p = (float)json_get_number(req.body, "top_p", def_top_p);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float min_p = (float)json_get_number(req.body, "min_p", 0.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty  = (float)json_get_number(req.body, "presence_penalty", 1.5);
    int64_t seed = (int64_t)json_get_number(req.body, "seed", -1);

    // Ollama 标准: 采样参数也可以在 "options" 子对象中
    std::vector<std::string> stop_seqs;
    {
        auto opts_pos = req.body.find("\"options\"");
        if (opts_pos != std::string::npos) {
            auto obj_start = req.body.find('{', opts_pos + 9);
            if (obj_start != std::string::npos) {
                std::string opts = extract_json_object(req.body, obj_start);
                if (opts.find("\"num_predict\"") != std::string::npos)
                    max_tokens = json_get_int(opts, "num_predict", max_tokens);
                if (opts.find("\"temperature\"") != std::string::npos)
                    temperature = (float)json_get_number(opts, "temperature", temperature);
                if (opts.find("\"top_p\"") != std::string::npos)
                    top_p = (float)json_get_number(opts, "top_p", top_p);
                if (opts.find("\"top_k\"") != std::string::npos)
                    top_k = json_get_int(opts, "top_k", top_k);
                if (opts.find("\"repeat_penalty\"") != std::string::npos)
                    repeat_penalty = (float)json_get_number(opts, "repeat_penalty", repeat_penalty);
                if (opts.find("\"min_p\"") != std::string::npos)
                    min_p = (float)json_get_number(opts, "min_p", min_p);
                if (opts.find("\"frequency_penalty\"") != std::string::npos)
                    frequency_penalty = (float)json_get_number(opts, "frequency_penalty", frequency_penalty);
                if (opts.find("\"presence_penalty\"") != std::string::npos)
                    presence_penalty = (float)json_get_number(opts, "presence_penalty", presence_penalty);
                if (opts.find("\"seed\"") != std::string::npos)
                    seed = (int64_t)json_get_number(opts, "seed", (double)seed);
                // Ollama stop 序列
                if (opts.find("\"stop\"") != std::string::npos) {
                    auto arr_pos = opts.find("\"stop\"");
                    auto arr_start = opts.find('[', arr_pos);
                    if (arr_start != std::string::npos) {
                        auto arr_end = opts.find(']', arr_start);
                        if (arr_end != std::string::npos) {
                            std::string arr = opts.substr(arr_start + 1, arr_end - arr_start - 1);
                            size_t p = 0;
                            while (p < arr.size()) {
                                auto q1 = arr.find('"', p);
                                if (q1 == std::string::npos) break;
                                auto q2 = arr.find('"', q1 + 1);
                                if (q2 == std::string::npos) break;
                                stop_seqs.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                                p = q2 + 1;
                            }
                        }
                    } else {
                        // 单字符串 stop
                        std::string s = json_get_string(opts, "stop");
                        if (!s.empty()) stop_seqs.push_back(s);
                    }
                }
            }
        }
    }
    max_tokens = clamp_max_output_tokens(max_tokens, config_.max_output_tokens_cap);
    // 顶级 stop 参数
    {
        std::string single_stop = json_get_string(req.body, "stop");
        if (!single_stop.empty() && stop_seqs.empty()) stop_seqs.push_back(single_stop);
    }

    std::string model = json_get_string(req.body, "model");
    if (model.empty()) model = config_.model_name;

    // ---- 解析 tools (Function Calling, Ollama 格式) ----
    std::string tools_json = extract_tools_json(req.body);
    bool has_tools = !tools_json.empty();
    if (has_tools) {
        std::string tool_prompt = build_tool_system_prompt(tools_json);
        bool found_system = false;
        for (auto& [role, content] : messages) {
            if (role == "system") {
                content += tool_prompt;
                found_system = true;
                break;
            }
        }
        if (!found_system) {
            messages.insert(messages.begin(),
                {"system", std::string(DEFAULT_SYSTEM_PROMPT) + tool_prompt});
        }
    }

    if (messages.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.status_text = "Bad Request";
        resp.body = R"({"error":"No messages provided"})";
        send_response(client_fd, resp);
        return;
    }

    // 确保有 system prompt — Qwen3.5 在无 system prompt 时行为显著下降
    {
        bool has_system = false;
        for (auto& [role, content] : messages) {
            if (role == "system") { has_system = true; break; }
        }
        if (!has_system) {
            messages.insert(messages.begin(),
                {"system", DEFAULT_SYSTEM_PROMPT});
        }
    }

    auto prompt_tokens = tok.apply_chat_template(messages, true, enable_thinking);

    // Expand image placeholders if multimodal
    if (!images.empty()) {
        expand_image_placeholders(prompt_tokens, images, tok);
    }
    // Expand video placeholders if video content was found
    if (!videos.empty()) {
        expand_video_placeholders(prompt_tokens, videos, tok);
    }

    InferRequest infer_req;
    infer_req.request_id        = next_request_id();
    infer_req.prompt_tokens     = std::move(prompt_tokens);
    infer_req.max_new_tokens    = max_tokens;
    infer_req.temperature       = temperature;
    infer_req.top_p             = top_p;
    infer_req.top_k             = top_k;
    infer_req.min_p             = min_p;
    infer_req.repeat_penalty    = repeat_penalty;
    infer_req.frequency_penalty = frequency_penalty;
    infer_req.presence_penalty  = presence_penalty;
    infer_req.seed              = seed;
    infer_req.stream            = true;
    infer_req.images            = std::move(images);
    infer_req.videos            = std::move(videos);

    // 先注册队列,再提交请求
    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = R"({"error":"Request queue full"})";
        send_response(client_fd, resp);
        return;
    }

    if (stream) {
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/x-ndjson; charset=utf-8\r\n"
                             "Transfer-Encoding: chunked\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto t0 = std::chrono::steady_clock::now();
        std::string finish_reason;
        int tool_call_idx = 0;
        std::atomic<bool> client_disconnected{false};

        int comp_toks = poll_tokens(infer_req.request_id,
            // on_token (content)
            [&](const std::string& piece) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\","
                    "\"content\":\"" + json_escape(piece) + "\"},\"done\":false}";
                if (!send_ndjson_chunk(client_fd, line)) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
            },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            // on_reasoning (thinking content)
            [&](const std::string& piece) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\","
                    "\"content\":\"\",\"thinking\":\"" + json_escape(piece) + "\"},\"done\":false}";
                if (!send_ndjson_chunk(client_fd, line)) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
            },
            // on_tool_call
            [&](const ToolCallInfo& tc) {
                if (client_disconnected.load(std::memory_order_relaxed)) return;
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\",\"content\":\"\","
                    "\"tool_calls\":[{\"function\":{\"name\":\"" + json_escape(tc.name) +
                    "\",\"arguments\":" + tc.arguments + "}}]},"
                    "\"done\":false}";
                if (!send_ndjson_chunk(client_fd, line)) {
                    client_disconnected.store(true, std::memory_order_relaxed);
                    backend_.cancel(infer_req.request_id);
                }
                tool_call_idx++;
            },
            &finish_reason,
            &client_disconnected
        );

        auto elapsed = std::chrono::steady_clock::now() - t0;
        auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
        auto eval_ns = comp_toks > 0 ? dur_ns : 0;
        std::string done_line = "{\"model\":\"" + json_escape(model) +
            "\",\"created_at\":\"" + iso8601_now() +
            "\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true"
            ",\"done_reason\":\"" + finish_reason + "\""
            ",\"total_duration\":" + std::to_string(dur_ns) +
            ",\"eval_count\":" + std::to_string(comp_toks) +
            ",\"eval_duration\":" + std::to_string(eval_ns) + "}";
        send_ndjson_chunk(client_fd, done_line);
        send_chunked_end(client_fd);
    } else {
        // Non-streaming: collect all tokens, reasoning, and tool calls
        std::string content;
        std::string reasoning;
        std::vector<ToolCallInfo> tool_calls;
        std::string finish_reason;

        auto t0_ns = std::chrono::steady_clock::now();
        int comp_toks = poll_tokens(infer_req.request_id,
            [&](const std::string& piece) { content += piece; },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            [&](const std::string& piece) { reasoning += piece; },
            [&](const ToolCallInfo& tc) { tool_calls.push_back(tc); },
            &finish_reason
        );
        auto elapsed_ns = std::chrono::steady_clock::now() - t0_ns;
        auto dur_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed_ns).count();
        auto eval_ns = comp_toks > 0 ? dur_ns : 0;

        // 构建 message 对象
        std::string msg_body = "\"role\":\"assistant\"";
        if (!reasoning.empty()) {
            msg_body += ",\"thinking\":\"" + json_escape(reasoning) + "\"";
        }
        if (!tool_calls.empty()) {
            msg_body += ",\"content\":\"\"";
            msg_body += ",\"tool_calls\":[";
            for (size_t i = 0; i < tool_calls.size(); i++) {
                if (i > 0) msg_body += ",";
                msg_body += "{\"function\":{\"name\":\"" + json_escape(tool_calls[i].name) +
                    "\",\"arguments\":" + tool_calls[i].arguments + "}}";
            }
            msg_body += "]";
        } else {
            msg_body += ",\"content\":\"" + json_escape(content) + "\"";
        }

        HttpResponse http_resp;
        http_resp.body = "{\"model\":\"" + json_escape(model) +
            "\",\"created_at\":\"" + iso8601_now() +
            "\",\"message\":{" + msg_body + "},\"done\":true"
            ",\"done_reason\":\"" + finish_reason + "\""
            ",\"total_duration\":" + std::to_string(dur_ns) +
            ",\"eval_count\":" + std::to_string(comp_toks) +
            ",\"eval_duration\":" + std::to_string(eval_ns) + "}";
        send_response(client_fd, http_resp);
    }
}

std::string ServeApp::make_chat_chunk(const std::string& model, const std::string& content,
                                       const std::string& finish_reason, const std::string& id,
                                       int64_t created) {
    std::string fr = finish_reason.empty() ? "null" : "\"" + finish_reason + "\"";
    return "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
           "\"created\":" + std::to_string(created) +
           ",\"model\":\"" + json_escape(model) +
           "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
           ",\"choices\":[{\"index\":0,"
           "\"delta\":{\"content\":\"" + json_escape(content) + "\"},"
           "\"finish_reason\":" + fr + "}]}";
}

std::string ServeApp::make_chat_reasoning_chunk(const std::string& model, const std::string& reasoning,
                                                 const std::string& id, int64_t created) {
    return "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
           "\"created\":" + std::to_string(created) +
           ",\"model\":\"" + json_escape(model) +
           "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
           ",\"choices\":[{\"index\":0,"
           "\"delta\":{\"reasoning_content\":\"" + json_escape(reasoning) + "\"},"
           "\"finish_reason\":null}]}";
}

std::string ServeApp::make_chat_tool_call_chunk(const std::string& model, const ToolCallInfo& tc,
                                                 int index, const std::string& id, int64_t created) {
    return "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\","
           "\"created\":" + std::to_string(created) +
           ",\"model\":\"" + json_escape(model) +
           "\",\"system_fingerprint\":\"fp_qwen35_bf16\""
           ",\"choices\":[{\"index\":0,"
           "\"delta\":{\"tool_calls\":[{\"index\":" + std::to_string(index) +
           ",\"id\":\"" + tc.id + "\",\"type\":\"function\","
           "\"function\":{\"name\":\"" + json_escape(tc.name) + "\","
           "\"arguments\":\"" + json_escape(tc.arguments) + "\"}}]},"
           "\"finish_reason\":null}]}";
}

std::string ServeApp::make_completion_chunk(const std::string& model, const std::string& text,
                                             const std::string& finish_reason, const std::string& id,
                                             int64_t created) {
    std::string fr = finish_reason.empty() ? "null" : "\"" + finish_reason + "\"";
    return "{\"id\":\"" + id + "\",\"object\":\"text_completion\","
           "\"created\":" + std::to_string(created) +
           ",\"model\":\"" + json_escape(model) + "\",\"choices\":[{\"index\":0,"
           "\"text\":\"" + json_escape(text) + "\","
           "\"finish_reason\":" + fr + "}]}";
}

uint64_t ServeApp::next_request_id() {
    return req_id_counter_.fetch_add(1);
}

// ============================================================================
// Multipart/Form-Data 解析
// ============================================================================

ServeApp::MultipartForm ServeApp::parse_multipart(const HttpRequest& req) {
    MultipartForm form;

    // 从 Content-Type 中提取 boundary
    auto ct_it = req.headers.find("content-type");
    if (ct_it == req.headers.end()) return form;

    std::string ct = ct_it->second;
    auto bpos = ct.find("boundary=");
    if (bpos == std::string::npos) return form;
    std::string boundary = ct.substr(bpos + 9);
    // 去掉可能的引号
    if (!boundary.empty() && boundary.front() == '"') boundary.erase(boundary.begin());
    if (!boundary.empty() && boundary.back() == '"')  boundary.pop_back();

    std::string delim = "--" + boundary;
    std::string end_delim = delim + "--";

    const std::string& body = req.body;
    size_t pos = body.find(delim);
    if (pos == std::string::npos) return form;

    while (true) {
        // 找到当前 part 的起始
        pos = body.find(delim, pos);
        if (pos == std::string::npos) break;
        pos += delim.size();

        // 检查是否是结束标记
        if (body.substr(pos, 2) == "--") break;

        // 跳过 \r\n
        if (pos < body.size() && body[pos] == '\r') pos++;
        if (pos < body.size() && body[pos] == '\n') pos++;

        // 找到 part headers 的结束 (空行 \r\n\r\n)
        auto header_end = body.find("\r\n\r\n", pos);
        if (header_end == std::string::npos) break;

        std::string headers_str = body.substr(pos, header_end - pos);
        size_t data_start = header_end + 4;

        // 找到下一个 boundary
        auto next_boundary = body.find(delim, data_start);
        if (next_boundary == std::string::npos) break;

        // data 在 boundary 前 2 字节 (\r\n)
        size_t data_end = next_boundary;
        if (data_end >= 2 && body[data_end - 2] == '\r' && body[data_end - 1] == '\n') {
            data_end -= 2;
        }

        std::string data = body.substr(data_start, data_end - data_start);

        // 解析 Content-Disposition
        std::string field_name, filename, part_ct;
        // 逐行解析 headers
        std::istringstream hss(headers_str);
        std::string hline;
        while (std::getline(hss, hline)) {
            if (!hline.empty() && hline.back() == '\r') hline.pop_back();

            // Content-Disposition: form-data; name="file"; filename="audio.wav"
            std::string hline_lower = hline;
            std::transform(hline_lower.begin(), hline_lower.end(), hline_lower.begin(), ::tolower);

            if (hline_lower.find("content-disposition:") == 0) {
                auto npos = hline.find("name=\"");
                if (npos != std::string::npos) {
                    npos += 6;
                    auto nend = hline.find("\"", npos);
                    if (nend != std::string::npos) field_name = hline.substr(npos, nend - npos);
                }
                auto fpos = hline.find("filename=\"");
                if (fpos != std::string::npos) {
                    fpos += 10;
                    auto fend = hline.find("\"", fpos);
                    if (fend != std::string::npos) filename = hline.substr(fpos, fend - fpos);
                }
            } else if (hline_lower.find("content-type:") == 0) {
                part_ct = hline.substr(14);
                while (!part_ct.empty() && part_ct.front() == ' ') part_ct.erase(part_ct.begin());
            }
        }

        if (!filename.empty()) {
            // 文件字段
            MultipartFile mf;
            mf.field_name = field_name;
            mf.filename = filename;
            mf.content_type = part_ct;
            mf.data = std::move(data);
            form.files.push_back(std::move(mf));
        } else {
            // 文本字段
            form.fields[field_name] = data;
        }
    }

    return form;
}

// ============================================================================
// 发送二进制响应
// ============================================================================

void ServeApp::send_binary_response(int client_fd, int status_code,
                                     const std::string& content_type,
                                     const uint8_t* data, size_t size,
                                     const std::string& extra_headers) {
    std::string status_text = (status_code == 200) ? "OK" : "Error";
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
    oss << "Content-Type: " << content_type << "\r\n";
    oss << "Content-Length: " << size << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    if (!extra_headers.empty()) oss << extra_headers;
    oss << "\r\n";

    auto header_str = oss.str();
    send(client_fd, header_str.c_str(), header_str.size(), MSG_NOSIGNAL);

    // 分块发送大文件
    size_t sent = 0;
    while (sent < size) {
        size_t chunk = std::min(size - sent, (size_t)65536);
        ssize_t n = send(client_fd, data + sent, chunk, MSG_NOSIGNAL);
        if (n <= 0) break;
        sent += n;
    }
}

// ============================================================================
// POST /v1/audio/transcriptions — ASR 语音转文本
// ============================================================================

void ServeApp::handle_audio_transcriptions(const HttpRequest& req, int client_fd) {
    // 检查 ASR 插件是否已启用
    if (!asr_plugin_) {
        HttpResponse resp;
        resp.status_code = 501;
        resp.status_text = "Not Implemented";
        resp.body = "{\"error\":{\"message\":\"ASR plugin not configured\","
                    "\"type\":\"invalid_request_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    if (!asr_plugin_->is_available()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = "{\"error\":{\"message\":\"ASR executable not available\","
                    "\"type\":\"service_unavailable\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 解析 multipart/form-data
    auto form = parse_multipart(req);

    // 提取音频文件
    std::string audio_data;
    std::string audio_filename = "upload.wav";
    for (auto& f : form.files) {
        if (f.field_name == "file") {
            audio_data = std::move(f.data);
            audio_filename = f.filename;
            break;
        }
    }

    if (audio_data.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.status_text = "Bad Request";
        resp.body = "{\"error\":{\"message\":\"No audio file provided. "
                    "Use multipart/form-data with field name 'file'\","
                    "\"type\":\"invalid_request_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 提取其他参数
    std::string language = form.fields.count("language") ? form.fields["language"] : "auto";
    std::string response_format = form.fields.count("response_format") ?
                                  form.fields["response_format"] : "json";
    bool suppress_early_eos = form.fields.count("suppress_early_eos") &&
                              (form.fields["suppress_early_eos"] == "true" ||
                               form.fields["suppress_early_eos"] == "1");
    bool punctuate = form.fields.count("punctuate") &&
                     (form.fields["punctuate"] == "true" || form.fields["punctuate"] == "1");
    bool identify_spk = form.fields.count("speaker") &&
                        (form.fields["speaker"] == "true" || form.fields["speaker"] == "1");
    bool clean_oral = form.fields.count("clean") &&
                      (form.fields["clean"] == "true" || form.fields["clean"] == "1");
    // OpenAI-compatible: timestamp_granularities[]=word 或 timestamp_granularities=word
    bool want_word_timestamps = false;
    {
        auto it = form.fields.find("timestamp_granularities[]");
        if (it == form.fields.end()) it = form.fields.find("timestamp_granularities");
        if (it != form.fields.end() && it->second.find("word") != std::string::npos)
            want_word_timestamps = true;
    }

    fprintf(stderr, "[Serve] ASR: received %zu bytes audio (%s), language=%s%s%s%s%s\n",
            audio_data.size(), audio_filename.c_str(), language.c_str(),
            suppress_early_eos ? ", suppress_early_eos" : "",
            punctuate ? ", punctuate" : "",
            identify_spk ? ", speaker_id" : "",
            want_word_timestamps ? ", word_timestamps" : "");

    // ========================================================================
    // v4 管线: ASR-first + ForcedAligner + CAM++ speaker attribution
    //   Phase 1: 解析音频 + 整段 ASR → 高质量全文
    //   Phase 2: ForcedAligner → 字级时间戳
    //   Phase 3: Fine-grained VAD + CAM++ → 每 VAD 段说话人 ID
    //   Phase 4: 对齐时间戳 × 说话人标签 → 每字分配 speaker
    //   Phase 5: 按 speaker 连续段生成输出 segments
    // ========================================================================
    if (identify_spk && want_word_timestamps && aligner_engine_.is_loaded() &&
        speaker_encoder_ && vad_engine_.is_loaded()) {
        // Phase 1a: 解析音频到 PCM
        audio::AudioData wav;
        if (!audio::load_audio_from_memory(
                reinterpret_cast<const uint8_t*>(audio_data.data()),
                audio_data.size(), wav, audio_filename)) {
            HttpResponse resp;
            resp.status_code = 400;
            resp.body = "{\"error\":{\"message\":\"Failed to decode audio\"}}";
            send_response(client_fd, resp);
            close(client_fd);
            return;
        }

        float total_duration_s = (float)wav.samples.size() / wav.sample_rate;
        fprintf(stderr, "[Serve] v4 pipeline: audio %.1fs, %zu samples\n",
                total_duration_s, wav.samples.size());

        // 每次请求清空说话人注册 (避免跨请求的 embedding 累积干扰)
        {
            std::lock_guard<std::mutex> lock(speaker_mutex_);
            speaker_manager_.clear();
        }

        auto v4_t0 = std::chrono::steady_clock::now();
        auto phase_t0 = v4_t0;

        // Phase 1b: 整段 ASR (≤100s 分段, plain mode)
        std::string full_text;
        if (total_duration_s > 100.0f) {
            // VAD 分段 → 分组为 ≤100s chunk → 逐段转录
            // 优先 GPU VAD (batch FSMN + cuFFT, <1s), 回退 CPU VAD (~100s)
            struct VadRange { int start_ms, end_ms; };
            std::vector<VadRange> vad_ranges;

            if (gpu_vad_engine_.is_loaded()) {
                auto gpu_segs = gpu_vad_engine_.detect_all(
                    wav.samples.data(), (int)wav.samples.size(), 500, 15000);
                for (auto& gs : gpu_segs)
                    vad_ranges.push_back({gs.start_ms, gs.end_ms});
            } else {
                std::lock_guard<std::mutex> lock(vad_mutex_);
                auto& cfg = vad_engine_.mutable_config();
                int orig_silence = cfg.max_end_silence_time;
                int orig_segment = cfg.max_single_segment_time;
                cfg.max_end_silence_time = 500;
                cfg.max_single_segment_time = 15000;
                auto vad_segs_asr = vad_engine_.detect_all(wav.samples.data(), (int)wav.samples.size());
                cfg.max_end_silence_time = orig_silence;
                cfg.max_single_segment_time = orig_segment;
                for (auto& vs : vad_segs_asr)
                    vad_ranges.push_back({vs.start_ms, vs.end_ms});
            }

            // 分组为 ≤100s chunks
            struct AsrChunk { std::vector<size_t> seg_indices; int start_ms, end_ms; };
            std::vector<AsrChunk> asr_chunks;
            for (size_t i = 0; i < vad_ranges.size(); ++i) {
                auto& vs = vad_ranges[i];
                if (vs.end_ms - vs.start_ms < 200) continue;
                bool extend = !asr_chunks.empty() &&
                              (vs.end_ms - asr_chunks.back().start_ms) <= 100000;
                if (extend) {
                    asr_chunks.back().seg_indices.push_back(i);
                    asr_chunks.back().end_ms = vs.end_ms;
                } else {
                    AsrChunk c; c.seg_indices.push_back(i);
                    c.start_ms = vs.start_ms; c.end_ms = vs.end_ms;
                    asr_chunks.push_back(std::move(c));
                }
            }

            int chunk_idx = 0;
            for (auto& chunk : asr_chunks) {
                std::vector<float> chunk_pcm;
                const int silence_pad = wav.sample_rate / 4;
                for (size_t vi = 0; vi < chunk.seg_indices.size(); ++vi) {
                    auto& vr = vad_ranges[chunk.seg_indices[vi]];
                    // Slice PCM from original samples using timestamps
                    int s = (int)((int64_t)vr.start_ms * wav.sample_rate / 1000);
                    int e = (int)((int64_t)vr.end_ms * wav.sample_rate / 1000);
                    if (s < 0) s = 0;
                    if (e > (int)wav.samples.size()) e = (int)wav.samples.size();
                    if (vi > 0 && !chunk_pcm.empty())
                        chunk_pcm.resize(chunk_pcm.size() + silence_pad, 0.0f);
                    chunk_pcm.insert(chunk_pcm.end(),
                                     wav.samples.begin() + s, wav.samples.begin() + e);
                }
                if ((int)chunk_pcm.size() < wav.sample_rate / 5) continue;
                auto seg_result = asr_plugin_->transcribe_pcm(
                    chunk_pcm.data(), (int)chunk_pcm.size(), wav.sample_rate, language, true);
                if (seg_result.error_code == 0 && !seg_result.text.empty())
                    full_text += seg_result.text;
                chunk_idx++;
            }
        } else {
            auto result = asr_plugin_->transcribe_pcm(
                wav.samples.data(), (int)wav.samples.size(), wav.sample_rate, language, true);
            if (result.error_code == 0) full_text = result.text;
        }

        if (full_text.empty()) {
            HttpResponse resp;
            resp.status_code = 500;
            resp.body = "{\"error\":{\"message\":\"ASR transcription produced no text\"}}";
            send_response(client_fd, resp);
            close(client_fd);
            return;
        }

        fprintf(stderr, "[Serve] v4 Phase 1: ASR text = %zu chars (%.1fs)\n",
                full_text.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_t0).count());
        phase_t0 = std::chrono::steady_clock::now();

        // ================================================================
        // Phase 2 & 3 (P3优化): 并行执行 ForcedAligner || VAD+CAM++
        // Phase 2 使用 aligner_mutex_, Phase 3 使用 vad_mutex_/speaker_mutex_
        // 两者资源独立，可以真正并发
        // ================================================================
        struct SpkInterval {
            int start_ms, end_ms;
            int speaker_id;
            std::string speaker_name;
        };
        std::vector<asr::AlignedWord> aligned_words;
        std::vector<SpkInterval> spk_intervals;
        int phase3_speaker_count = 0;
        auto phase2_t0 = std::chrono::steady_clock::now();

        // Phase 2 在后台线程执行
        auto phase2_future = std::async(std::launch::async, [&]() -> std::vector<asr::AlignedWord> {
            std::vector<asr::AlignedWord> words;
            const int max_align_samples = wav.sample_rate * 180;
            if ((int)wav.samples.size() <= max_align_samples) {
                std::lock_guard<std::mutex> lock(aligner_mutex_);
                words = aligner_engine_.align(
                    wav.samples.data(), (int)wav.samples.size(),
                    wav.sample_rate, full_text, "Chinese");
            } else {
                // 分段对齐 (150s 段, 留 30s 余量)
                const int seg_samples = wav.sample_rate * 150;
                auto all_chars = asr::AlignerEngine::tokenize_for_align(full_text);
                int total_chars = (int)all_chars.size();
                int total_samples = (int)wav.samples.size();
                int num_segs = (total_samples + seg_samples - 1) / seg_samples;
                int chars_per_seg = (total_chars + num_segs - 1) / num_segs;
                int char_offset = 0;
                for (int si = 0; si < num_segs && char_offset < total_chars; ++si) {
                    int sample_start = si * seg_samples;
                    int sample_end = std::min(sample_start + seg_samples, total_samples);
                    int seg_chars = std::min(chars_per_seg, total_chars - char_offset);
                    std::string seg_text;
                    for (int ci = char_offset; ci < char_offset + seg_chars; ++ci)
                        seg_text += all_chars[ci];
                    int offset_ms = (int)((float)sample_start / wav.sample_rate * 1000);
                    std::vector<asr::AlignedWord> seg_aligned;
                    {
                        std::lock_guard<std::mutex> lock(aligner_mutex_);
                        seg_aligned = aligner_engine_.align(
                            wav.samples.data() + sample_start, sample_end - sample_start,
                            wav.sample_rate, seg_text, "Chinese");
                    }
                    for (auto& w : seg_aligned) {
                        w.start_ms += offset_ms;
                        w.end_ms += offset_ms;
                        words.push_back(w);
                    }
                    char_offset += seg_chars;
                }
            }
            return words;
        });

        // Phase 3 在主线程并发执行 (与 Phase 2 重叠)
        {
            auto phase3_t0 = std::chrono::steady_clock::now();

            // Step 3a: VAD — 优先使用 GPU, 回退 CPU
            struct VadResult { int start_ms; int end_ms; };
            std::vector<VadResult> vad_results;
            auto vad_t0 = std::chrono::steady_clock::now();

            if (gpu_vad_engine_.is_loaded()) {
                // GPU VAD: batch FSMN + cuFFT, 全部帧一次性推理
                auto gpu_segs = gpu_vad_engine_.detect_all(
                    wav.samples.data(), (int)wav.samples.size(), 300, 8000);
                for (auto& gs : gpu_segs)
                    vad_results.push_back({gs.start_ms, gs.end_ms});
            } else {
                // CPU VAD fallback
                std::lock_guard<std::mutex> lock(vad_mutex_);
                auto& cfg = vad_engine_.mutable_config();
                int orig_silence = cfg.max_end_silence_time;
                int orig_segment = cfg.max_single_segment_time;
                cfg.max_end_silence_time = 300;
                cfg.max_single_segment_time = 8000;
                auto vad_segments = vad_engine_.detect_all(wav.samples.data(), (int)wav.samples.size());
                cfg.max_end_silence_time = orig_silence;
                cfg.max_single_segment_time = orig_segment;
                for (auto& vs : vad_segments)
                    vad_results.push_back({vs.start_ms, vs.end_ms});
            }

            double vad_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - vad_t0).count();

            // Step 3b: Mel + CAM++ speaker embedding (chunk-level → spectral clustering)
            // 长 VAD 段含多说话人 → 整段 embedding 是混合信号;
            // 拆成 3s chunk 分别提 embedding, 再直接聚类 chunk (不做段级平均)
            auto mel_t0 = std::chrono::steady_clock::now();
            int mel_segments = 0;
            std::vector<std::vector<float>> seg_embeddings;  // chunk embeddings (重用变量名)

            struct ChunkInfo { int abs_start_ms, abs_end_ms; };
            std::vector<ChunkInfo> chunk_infos;
            float total_speech_sec = 0;  // 用于 duration heuristic

            const int CHUNK_FRAMES = 300;     // 3.0s @ 10ms/frame (best balance: gap=0.129, single-speaker likely)
            const int MIN_CHUNK_FRAMES = 150; // 1.5s minimum viable chunk

            // === Batch CAM++ extraction ===
            // Pass 1: Compute mel per segment, copy chunk mels to GPU batch buffer
            using BatchChunk = asr::GpuSpeakerEncoder::BatchChunk;
            std::vector<BatchChunk> batch_chunks;
            std::vector<ChunkInfo> batch_chunk_infos;

            // GPU batch mel buffer
            bool use_gpu_batch = gpu_mel_.is_initialized() && speaker_encoder_;
            float* d_batch_mels = nullptr;
            int batch_mel_capacity = (int)(total_duration_s * 100 + 10000) * 80;  // max frames × 80
            if (use_gpu_batch) {
                if (cudaMalloc(&d_batch_mels, (size_t)batch_mel_capacity * sizeof(float)) != cudaSuccess) {
                    fprintf(stderr, "[Serve] batch mel buffer alloc failed (%.1f MB), falling back to serial\n",
                            (size_t)batch_mel_capacity * sizeof(float) / (1024.0f * 1024.0f));
                    use_gpu_batch = false;
                }
            }
            int batch_mel_offset = 0;  // accumulated frames

            for (size_t vi = 0; vi < vad_results.size(); ++vi) {
                auto& vr = vad_results[vi];
                if (vr.end_ms - vr.start_ms < 200) continue;

                int64_t start_sample = (int64_t)vr.start_ms * wav.sample_rate / 1000;
                int64_t end_sample = (int64_t)vr.end_ms * wav.sample_rate / 1000;
                start_sample = std::max((int64_t)0, std::min(start_sample, (int64_t)wav.samples.size()));
                end_sample = std::max(start_sample, std::min(end_sample, (int64_t)wav.samples.size()));
                int seg_samples = (int)(end_sample - start_sample);
                if (seg_samples < 1600) continue;

                const float* seg_pcm = wav.samples.data() + start_sample;

                float rms = 0.0f;
                for (int i = 0; i < seg_samples; ++i) rms += seg_pcm[i] * seg_pcm[i];
                rms = std::sqrt(rms / seg_samples);
                if (rms < 0.005f) continue;

                // GPU mel path: compute on GPU, keep on GPU for speaker encoder
                int num_frames = 0;
                float* d_mel = nullptr;  // GPU pointer (only valid until next compute_gpu)
                std::vector<float> mel;  // CPU fallback
                if (gpu_mel_.is_initialized()) {
                    auto mel_result = gpu_mel_.compute_gpu(seg_pcm, seg_samples);
                    num_frames = mel_result.num_frames;
                    d_mel = mel_result.d_mel;
                    gpu_mel_.sync();  // ensure GPU mel complete before copy
                } else {
                    compute_mel_80(seg_pcm, seg_samples, wav.sample_rate, mel, num_frames);
                }
                if (num_frames < 10) continue;
                ++mel_segments;
                total_speech_sec += (vr.end_ms - vr.start_ms) / 1000.0f;

                // 将 mel 拆成 3s chunk; 短段 (<4.5s) 整段作为一个 chunk
                std::vector<std::pair<int,int>> chunk_ranges; // (frame_start, frame_end)
                if (num_frames <= CHUNK_FRAMES + MIN_CHUNK_FRAMES) {
                    chunk_ranges.push_back({0, num_frames});
                } else {
                    int pos = 0;
                    while (pos < num_frames) {
                        int end = pos + CHUNK_FRAMES;
                        if (num_frames - end < MIN_CHUNK_FRAMES) end = num_frames;
                        chunk_ranges.push_back({pos, std::min(end, num_frames)});
                        pos = end;
                        if (pos >= num_frames) break;
                    }
                }

                const float MS_PER_FRAME = 10.0f;  // hop=160 @ 16kHz
                for (auto& [f_start, f_end] : chunk_ranges) {
                    int chunk_frames = f_end - f_start;
                    if (chunk_frames < 10) continue;

                    int abs_start = vr.start_ms + (int)(f_start * MS_PER_FRAME);
                    int abs_end = vr.start_ms + (int)(f_end * MS_PER_FRAME);
                    abs_end = std::min(abs_end, vr.end_ms);

                    if (use_gpu_batch && d_mel &&
                        (batch_mel_offset + chunk_frames) * 80 <= batch_mel_capacity) {
                        // Copy chunk mel to persistent batch buffer (sync D2D, d_mel valid after gpu_mel_.sync)
                        cudaMemcpy(d_batch_mels + (size_t)batch_mel_offset * 80,
                                   d_mel + f_start * 80,
                                   (size_t)chunk_frames * 80 * sizeof(float),
                                   cudaMemcpyDeviceToDevice);
                        batch_chunks.push_back({d_batch_mels + (size_t)batch_mel_offset * 80, chunk_frames});
                        batch_chunk_infos.push_back({abs_start, abs_end});
                        batch_mel_offset += chunk_frames;
                    } else if (speaker_encoder_) {
                        // Serial fallback (CPU mel or buffer overflow)
                        std::vector<float> embedding;
                        {
                            std::lock_guard<std::mutex> spk_lock(speaker_mutex_);
                            if (d_mel) {
                                embedding = speaker_encoder_->extract_gpu(d_mel + f_start * 80, chunk_frames);
                            } else if (!mel.empty()) {
                                embedding = speaker_encoder_->extract(mel.data() + f_start * 80, chunk_frames);
                            }
                        }
                        if (!embedding.empty()) {
                            bool valid = true;
                            for (float v : embedding)
                                if (std::isnan(v) || std::isinf(v)) { valid = false; break; }
                            if (valid) {
                                seg_embeddings.push_back(std::move(embedding));
                                chunk_infos.push_back({abs_start, abs_end});
                            }
                        }
                    }
                }
            }

            // Pass 2: Batch extract all GPU chunks at once (multi-stream)
            if (use_gpu_batch && !batch_chunks.empty()) {
                std::lock_guard<std::mutex> spk_lock(speaker_mutex_);
                auto embeddings = speaker_encoder_->extract_batch_gpu(batch_chunks);
                for (int i = 0; i < (int)embeddings.size(); i++) {
                    if (!embeddings[i].empty()) {
                        seg_embeddings.push_back(std::move(embeddings[i]));
                        chunk_infos.push_back(batch_chunk_infos[i]);
                    }
                }
            }

            if (d_batch_mels) cudaFree(d_batch_mels);

            fprintf(stderr, "[Serve] Step 3b: %d VAD segs → %zu chunks (%.0fs speech) [batch=%zu]\n",
                    mel_segments, seg_embeddings.size(), total_speech_sec, batch_chunks.size());

            // Debug: Export embeddings for Python analysis
            {
                const char* dbg = getenv("QWEN_EXPORT_EMBEDDINGS");
                if (dbg && seg_embeddings.size() > 0) {
                    FILE* fp = fopen(dbg, "wb");
                    if (fp) {
                        int n = (int)seg_embeddings.size();
                        int d = (int)seg_embeddings[0].size();
                        fwrite(&n, 4, 1, fp);
                        fwrite(&d, 4, 1, fp);
                        for (auto& e : seg_embeddings)
                            fwrite(e.data(), 4, d, fp);
                        // Also write chunk times
                        for (auto& ci : chunk_infos) {
                            float cs = (float)ci.abs_start_ms / 1000.0f;
                            float ce2 = (float)ci.abs_end_ms / 1000.0f;
                            fwrite(&cs, 4, 1, fp);
                            fwrite(&ce2, 4, 1, fp);
                        }
                        fclose(fp);
                        fprintf(stderr, "[Serve] Exported %d embeddings (%d-d) to %s\n", n, d, dbg);
                    }
                }
            }

            // Phase 3b: Spectral Clustering (chunk-level)
            // 构建余弦相似度矩阵 → p-pruning → Laplacian → eigengap → k-means
            if (seg_embeddings.size() >= 2) {
                const int n_segs = (int)seg_embeddings.size();
                const int emb_dim = (int)seg_embeddings[0].size();

                // 1. 计算余弦相似度矩阵 N×N
                std::vector<float> sim_matrix(n_segs * n_segs, 0.0f);
                for (int i = 0; i < n_segs; ++i) {
                    sim_matrix[i * n_segs + i] = 1.0f;
                    for (int j = i + 1; j < n_segs; ++j) {
                        float dot = 0;
                        for (int k = 0; k < emb_dim; ++k)
                            dot += seg_embeddings[i][k] * seg_embeddings[j][k];
                        sim_matrix[i * n_segs + j] = dot;
                        sim_matrix[j * n_segs + i] = dot;
                    }
                }

                // 1b. Temporal proximity mixing
                // S_combined = (1-α)*S_cosine + α*exp(-|t_i-t_j|/τ)
                // 近距离的 chunk 更可能来自同一说话人 (对话轮替特性)
                // 参数扫描: α=0.65/τ=12s 在鲁棒性和精度之间最优 (chunk 71.6%→76.7%)
                {
                    constexpr float TEMPORAL_ALPHA = 0.65f;
                    constexpr float TEMPORAL_TAU = 12.0f; // seconds
                    constexpr float INV_TAU = 1.0f / TEMPORAL_TAU;
                    for (int i = 0; i < n_segs; ++i) {
                        float mid_i = (chunk_infos[i].abs_start_ms + chunk_infos[i].abs_end_ms) * 0.5e-3f;
                        for (int j = i + 1; j < n_segs; ++j) {
                            float mid_j = (chunk_infos[j].abs_start_ms + chunk_infos[j].abs_end_ms) * 0.5e-3f;
                            float t_prox = expf(-fabsf(mid_i - mid_j) * INV_TAU);
                            float cos_val = sim_matrix[i * n_segs + j];
                            float combined = (1.0f - TEMPORAL_ALPHA) * cos_val + TEMPORAL_ALPHA * t_prox;
                            sim_matrix[i * n_segs + j] = combined;
                            sim_matrix[j * n_segs + i] = combined;
                        }
                    }
                }

                // 1c. S-norm 已禁用 — 测试表明 S-norm 会降低准确率 (88.2%→88.0%)
                // 原因: embedding 维度已足够, S-norm 过度归一化反而模糊了 speaker 差异
                #if 0
                // S-norm 分数归一化 — 消除"通用"embedding 的偏差
                // 对每个 item i, 计算其与所有其他 item 的相似度均值 μ_i 和标准差 σ_i
                // 归一化: sim_norm(i,j) = (sim(i,j) - (μ_i+μ_j)/2) / ((σ_i+σ_j)/2 + ε)
                {
                    std::vector<float> mu(n_segs, 0), sigma(n_segs, 0);
                    for (int i = 0; i < n_segs; ++i) {
                        double sum = 0, sum2 = 0;
                        int cnt = 0;
                        for (int j = 0; j < n_segs; ++j) {
                            if (j == i) continue;
                            float s = sim_matrix[i * n_segs + j];
                            sum += s; sum2 += (double)s * s; ++cnt;
                        }
                        mu[i] = (float)(sum / cnt);
                        sigma[i] = sqrtf((float)(sum2 / cnt - (double)mu[i] * mu[i]) + 1e-12f);
                    }
                    for (int i = 0; i < n_segs; ++i) {
                        for (int j = i + 1; j < n_segs; ++j) {
                            float raw = sim_matrix[i * n_segs + j];
                            float mu_ij = (mu[i] + mu[j]) * 0.5f;
                            float sigma_ij = (sigma[i] + sigma[j]) * 0.5f + 1e-6f;
                            float normed = (raw - mu_ij) / sigma_ij;
                            sim_matrix[i * n_segs + j] = normed;
                            sim_matrix[j * n_segs + i] = normed;
                        }
                        sim_matrix[i * n_segs + i] = 1.0f; // diagonal stays 1
                    }
                }
                #endif  // S-norm disabled

                // 2. p-pruning: 每行只保留 top-p 个最大值, 其余设为 0
                int p = std::max(3, n_segs * 6 / 100);  // ~6% 最优 (temporal mixing 后重新扫描)
                p = std::min(p, n_segs - 1);
                for (int i = 0; i < n_segs; ++i) {
                    // 找第 p+1 大的值作为阈值
                    std::vector<float> row_vals;
                    for (int j = 0; j < n_segs; ++j) {
                        if (j != i) row_vals.push_back(sim_matrix[i * n_segs + j]);
                    }
                    std::sort(row_vals.rbegin(), row_vals.rend());
                    float threshold = (p < (int)row_vals.size()) ? row_vals[p] : -2.0f;
                    for (int j = 0; j < n_segs; ++j) {
                        if (j != i && sim_matrix[i * n_segs + j] < threshold)
                            sim_matrix[i * n_segs + j] = 0;
                    }
                }

                // 3. 对称化: W = (S + S^T) / 2 (sklearn 默认)
                for (int i = 0; i < n_segs; ++i) {
                    for (int j = i + 1; j < n_segs; ++j) {
                        float val = (sim_matrix[i * n_segs + j] + sim_matrix[j * n_segs + i]) * 0.5f;
                        val = std::max(0.0f, val);
                        sim_matrix[i * n_segs + j] = val;
                        sim_matrix[j * n_segs + i] = val;
                    }
                }

                // 检查孤立节点: 如果 row sum = 0 (p-pruning 断开), 连接到最相似邻居
                for (int i = 0; i < n_segs; ++i) {
                    float row_sum = 0;
                    for (int j = 0; j < n_segs; ++j)
                        if (j != i) row_sum += sim_matrix[i * n_segs + j];
                    if (row_sum < 1e-12f) {
                        // 找最相似的邻居 (在原始 embedding 空间)
                        float best = -2; int best_j = 0;
                        for (int j = 0; j < n_segs; ++j) {
                            if (j == i) continue;
                            float dot = 0;
                            for (int k = 0; k < emb_dim; ++k)
                                dot += seg_embeddings[i][k] * seg_embeddings[j][k];
                            if (dot > best) { best = dot; best_j = j; }
                        }
                        sim_matrix[i * n_segs + best_j] = std::max(0.01f, best);
                        sim_matrix[best_j * n_segs + i] = std::max(0.01f, best);
                    }
                }

                // 4. 计算 normalized Laplacian 的特征值 (用于 eigengap)
                // D = diag(row sums), L_sym = I - D^{-1/2} W D^{-1/2}
                std::vector<float> D(n_segs, 0.0f);
                for (int i = 0; i < n_segs; ++i) {
                    for (int j = 0; j < n_segs; ++j)
                        D[i] += sim_matrix[i * n_segs + j];
                }
                std::vector<float> D_inv_sqrt(n_segs);
                for (int i = 0; i < n_segs; ++i)
                    D_inv_sqrt[i] = (D[i] > 1e-12f) ? 1.0f / sqrtf(D[i]) : 0.0f;

                // L_sym = I - D^{-1/2} W D^{-1/2}
                // 对于特征值分解, 我们计算 D^{-1/2} W D^{-1/2} (特征值是 1-λ_L)
                std::vector<float> Lsym(n_segs * n_segs, 0.0f);
                for (int i = 0; i < n_segs; ++i)
                    for (int j = 0; j < n_segs; ++j)
                        Lsym[i * n_segs + j] = D_inv_sqrt[i] * sim_matrix[i * n_segs + j] * D_inv_sqrt[j];

                // 5. Power iteration 提取 top eigenvalues (最多 max_k=8)
                const int max_k = 8;
                int actual_max = std::min(max_k, n_segs);
                std::vector<std::vector<float>> eigenvectors(actual_max, std::vector<float>(n_segs, 0));
                std::vector<float> eigenvalues(actual_max, 0);

                // 简化版: 用 QR iteration 提取 top-k 特征向量
                // 这里用幂迭代 + deflation
                std::vector<float> Lwork = Lsym;  // 工作副本
                for (int k = 0; k < actual_max; ++k) {
                    // 随机初始化
                    std::vector<float> v(n_segs);
                    for (int i = 0; i < n_segs; ++i) v[i] = (float)(i + k * 7 + 1);
                    float vnorm = 0;
                    for (float x : v) vnorm += x * x;
                    vnorm = sqrtf(vnorm);
                    for (float& x : v) x /= vnorm;

                    // 幂迭代 (200 次)
                    for (int iter = 0; iter < 200; ++iter) {
                        std::vector<float> Av(n_segs, 0);
                        for (int i = 0; i < n_segs; ++i)
                            for (int j = 0; j < n_segs; ++j)
                                Av[i] += Lwork[i * n_segs + j] * v[j];
                        float norm = 0;
                        for (float x : Av) norm += x * x;
                        norm = sqrtf(norm + 1e-12f);
                        for (int i = 0; i < n_segs; ++i) v[i] = Av[i] / norm;
                    }

                    // 计算特征值 = v^T A v
                    float lambda = 0;
                    for (int i = 0; i < n_segs; ++i) {
                        float Av_i = 0;
                        for (int j = 0; j < n_segs; ++j)
                            Av_i += Lwork[i * n_segs + j] * v[j];
                        lambda += v[i] * Av_i;
                    }
                    eigenvalues[k] = lambda;
                    eigenvectors[k] = v;

                    // Deflation: A = A - lambda * v * v^T
                    for (int i = 0; i < n_segs; ++i)
                        for (int j = 0; j < n_segs; ++j)
                            Lwork[i * n_segs + j] -= lambda * v[i] * v[j];
                }

                // 6. NME (Normalized Maximum Eigengap) — FunASR 策略
                // gap[k] / (k+1) 归一化避免偏向小 k
                int optimal_k = 2;
                float max_nme = 0;
                for (int k = 0; k + 1 < actual_max; ++k) {
                    float gap = eigenvalues[k] - eigenvalues[k + 1];
                    if (eigenvalues[k] < 0.01f) continue;
                    float nme = gap / (k + 1);
                    if (nme > max_nme) {
                        max_nme = nme;
                        optimal_k = k + 1;
                    }
                }
                optimal_k = std::max(2, std::min(optimal_k, 8));
                
                // 长音频启发式: 说话人同性别时 embedding 差异小, eigengap 容易低估 k
                // > 5 min + > 50 段 → 至少 3 人; > 20 min + > 100 段 → 至少 4 人
                int min_k_heuristic = 2;
                if (total_speech_sec > 1200 && n_segs > 100) min_k_heuristic = 4;
                else if (total_speech_sec > 300 && n_segs > 50) min_k_heuristic = 3;
                if (optimal_k < min_k_heuristic) {
                    fprintf(stderr, "[Serve] Phase 3b: duration heuristic bumps k %d→%d (%.0fs, %d chunks)\n",
                            optimal_k, min_k_heuristic, total_speech_sec, n_segs);
                    optimal_k = min_k_heuristic;
                }

                fprintf(stderr, "[Serve] Phase 3b spectral: eigenvalues:");
                for (int k = 0; k < actual_max; ++k)
                    fprintf(stderr, " %.3f", eigenvalues[k]);
                fprintf(stderr, " → k=%d (nme=%.4f)\n", optimal_k, max_nme);

                // 日志: 所有 NME 候选
                fprintf(stderr, "[Serve] Phase 3b NME candidates:");
                for (int k = 0; k + 1 < actual_max; ++k) {
                    float gap = eigenvalues[k] - eigenvalues[k + 1];
                    float nme = (eigenvalues[k] > 0.01f) ? gap / (k + 1) : 0;
                    fprintf(stderr, " k=%d(%.4f)", k + 1, nme);
                }
                fprintf(stderr, "\n");

                // 7. 取前 optimal_k 个特征向量做行归一化, 然后 k-means
                // 构建 N × optimal_k 矩阵
                std::vector<float> features(n_segs * optimal_k);
                for (int i = 0; i < n_segs; ++i) {
                    float row_norm = 0;
                    for (int k = 0; k < optimal_k; ++k) {
                        features[i * optimal_k + k] = eigenvectors[k][i];
                        row_norm += eigenvectors[k][i] * eigenvectors[k][i];
                    }
                    row_norm = sqrtf(row_norm + 1e-12f);
                    for (int k = 0; k < optimal_k; ++k)
                        features[i * optimal_k + k] /= row_norm;
                }

                // K-means on spectral features with multiple restarts (n_init=10)
                std::vector<int> labels(n_segs, 0);
                float best_inertia = 1e30f;

                for (int restart = 0; restart < 10; ++restart) {
                    std::vector<std::vector<float>> cur_centroids(optimal_k, std::vector<float>(optimal_k, 0));
                    std::vector<int> cur_labels(n_segs, 0);

                    // k-means++ initialization with varied seed points
                    int seed_idx = restart * 137 % n_segs;
                    for (int j = 0; j < optimal_k; ++j)
                        cur_centroids[0][j] = features[seed_idx * optimal_k + j];
                    for (int c = 1; c < optimal_k; ++c) {
                        float best_d = -1;
                        int best_i = 0;
                        for (int i = 0; i < n_segs; ++i) {
                            float min_d = 1e30f;
                            for (int prev = 0; prev < c; ++prev) {
                                float d = 0;
                                for (int j = 0; j < optimal_k; ++j) {
                                    float diff = features[i * optimal_k + j] - cur_centroids[prev][j];
                                    d += diff * diff;
                                }
                                min_d = std::min(min_d, d);
                            }
                            if (min_d > best_d) {
                                best_d = min_d;
                                best_i = i;
                            }
                        }
                        for (int j = 0; j < optimal_k; ++j)
                            cur_centroids[c][j] = features[best_i * optimal_k + j];
                    }

                    for (int iter = 0; iter < 30; ++iter) {
                        // Assign
                        int changed = 0;
                        for (int i = 0; i < n_segs; ++i) {
                            float best_d = 1e30f;
                            int best_c = 0;
                            for (int c = 0; c < optimal_k; ++c) {
                                float d = 0;
                                for (int j = 0; j < optimal_k; ++j) {
                                    float diff = features[i * optimal_k + j] - cur_centroids[c][j];
                                    d += diff * diff;
                                }
                                if (d < best_d) {
                                    best_d = d;
                                    best_c = c;
                                }
                            }
                            if (best_c != cur_labels[i]) ++changed;
                            cur_labels[i] = best_c;
                        }

                        // Update centroids
                        for (int c = 0; c < optimal_k; ++c)
                            std::fill(cur_centroids[c].begin(), cur_centroids[c].end(), 0.0f);
                        std::vector<int> cnt(optimal_k, 0);
                        for (int i = 0; i < n_segs; ++i) {
                            cnt[cur_labels[i]]++;
                            for (int j = 0; j < optimal_k; ++j)
                                cur_centroids[cur_labels[i]][j] += features[i * optimal_k + j];
                        }
                        for (int c = 0; c < optimal_k; ++c)
                            if (cnt[c] > 0)
                                for (int j = 0; j < optimal_k; ++j)
                                    cur_centroids[c][j] /= cnt[c];

                        if (changed == 0) break;
                    }

                    // Compute inertia (sum of squared distances to centroids)
                    float inertia = 0;
                    for (int i = 0; i < n_segs; ++i) {
                        for (int j = 0; j < optimal_k; ++j) {
                            float diff = features[i * optimal_k + j] - cur_centroids[cur_labels[i]][j];
                            inertia += diff * diff;
                        }
                    }

                    if (inertia < best_inertia) {
                        best_inertia = inertia;
                        labels = cur_labels;
                    }
                }

                // 8. 后处理: 日志打印 cluster centroid 间相似度 (不做 merge, 信任 spectral clustering)
                // 计算 cluster embedding centroids (在原始 embedding 空间)
                std::vector<std::vector<float>> cluster_emb(optimal_k, std::vector<float>(emb_dim, 0));
                std::vector<int> cluster_cnt(optimal_k, 0);
                for (int i = 0; i < n_segs; ++i) {
                    cluster_cnt[labels[i]]++;
                    for (int j = 0; j < emb_dim; ++j)
                        cluster_emb[labels[i]][j] += seg_embeddings[i][j];
                }
                for (int c = 0; c < optimal_k; ++c) {
                    if (cluster_cnt[c] > 0) {
                        for (int j = 0; j < emb_dim; ++j)
                            cluster_emb[c][j] /= cluster_cnt[c];
                        float norm = 0;
                        for (float v : cluster_emb[c]) norm += v * v;
                        norm = sqrtf(norm + 1e-12f);
                        for (float& v : cluster_emb[c]) v /= norm;
                    }
                }

                // 日志: cluster centroid 间余弦相似度
                fprintf(stderr, "[Serve] Phase 3b: cluster centroid similarities:\n");
                for (int i = 0; i < optimal_k; ++i) {
                    for (int j = i + 1; j < optimal_k; ++j) {
                        float sim = 0;
                        for (int k = 0; k < emb_dim; ++k)
                            sim += cluster_emb[i][k] * cluster_emb[j][k];
                        fprintf(stderr, "  c%d-c%d: %.3f", i, j, sim);
                    }
                }
                fprintf(stderr, "\n");

                // 日志: embedding 相似度全局统计
                {
                    float min_sim = 2, max_sim = -2;
                    double sum_sim = 0;
                    int cnt = 0, nan_cnt = 0;
                    for (int i = 0; i < n_segs; ++i) {
                        for (int j = i + 1; j < n_segs; ++j) {
                            float dot = 0;
                            for (int k = 0; k < emb_dim; ++k)
                                dot += seg_embeddings[i][k] * seg_embeddings[j][k];
                            if (std::isnan(dot) || std::isinf(dot)) { ++nan_cnt; continue; }
                            if (dot < min_sim) min_sim = dot;
                            if (dot > max_sim) max_sim = dot;
                            sum_sim += dot;
                            ++cnt;
                        }
                    }
                    fprintf(stderr, "[Serve] Phase 3b: embedding sim stats: min=%.3f max=%.3f avg=%.3f (N=%d, nan_pairs=%d)\n",
                            min_sim, max_sim, cnt > 0 ? (float)(sum_sim / cnt) : 0.0f, n_segs, nan_cnt);
                }

                // 从 chunk labels 重建 spk_intervals (而非直接分配到已有段)
                // 每个 chunk → 一个时间区间, 然后合并相邻同说话人区间
                spk_intervals.clear();
                for (int i = 0; i < n_segs; ++i) {
                    SpkInterval si;
                    si.start_ms = chunk_infos[i].abs_start_ms;
                    si.end_ms = chunk_infos[i].abs_end_ms;
                    si.speaker_id = labels[i];
                    si.speaker_name = "Speaker_" + std::to_string(labels[i]);
                    spk_intervals.push_back(si);
                }

                // 按时间排序
                std::sort(spk_intervals.begin(), spk_intervals.end(),
                    [](const SpkInterval& a, const SpkInterval& b) { return a.start_ms < b.start_ms; });

                // 合并相邻同说话人区间 (gap ≤ 500ms)
                {
                    std::vector<SpkInterval> merged;
                    for (auto& si : spk_intervals) {
                        if (!merged.empty() && merged.back().speaker_id == si.speaker_id &&
                            si.start_ms - merged.back().end_ms <= 500) {
                            merged.back().end_ms = std::max(merged.back().end_ms, si.end_ms);
                        } else {
                            merged.push_back(si);
                        }
                    }
                    spk_intervals = std::move(merged);
                }

                // Phase 3b 分布日志
                {
                    std::map<int, int> dist;
                    for (auto& si : spk_intervals) dist[si.speaker_id]++;
                    fprintf(stderr, "[Serve] Phase 3b distribution (%zu intervals after merge):",
                            spk_intervals.size());
                    for (auto& [id, cnt] : dist)
                        fprintf(stderr, " spk%d=%d(%.0f%%)", id, cnt,
                                100.0f * cnt / (float)spk_intervals.size());
                    fprintf(stderr, "\n");
                }

                // 重编号 speaker_id + 按出现顺序排列
                std::map<int, int> renumber;
                int next_num = 0;
                for (auto& si : spk_intervals) {
                    if (renumber.find(si.speaker_id) == renumber.end()) {
                        renumber[si.speaker_id] = next_num++;
                    }
                    si.speaker_id = renumber[si.speaker_id];
                    si.speaker_name = "Speaker_" + std::to_string(si.speaker_id);
                }
                phase3_speaker_count = next_num;

                // 自动注册 diarization 说话人到全局 speaker_manager_
                if (!seg_embeddings.empty()) {
                    std::map<int, std::vector<float>> cluster_centroids;
                    std::map<int, int> cluster_counts;
                    int edim_reg = (int)seg_embeddings[0].size();
                    for (size_t ci = 0; ci < seg_embeddings.size(); ++ci) {
                        int final_id = renumber.count(labels[ci]) ? renumber[labels[ci]] : labels[ci];
                        if (cluster_centroids.find(final_id) == cluster_centroids.end()) {
                            cluster_centroids[final_id].assign(edim_reg, 0.0f);
                            cluster_counts[final_id] = 0;
                        }
                        cluster_counts[final_id]++;
                        for (int j = 0; j < edim_reg; ++j)
                            cluster_centroids[final_id][j] += seg_embeddings[ci][j];
                    }
                    for (auto& [id, c] : cluster_centroids) {
                        float cnt_f = (float)cluster_counts[id];
                        for (float& v : c) v /= cnt_f;
                        float norm = 0;
                        for (float v : c) norm += v * v;
                        norm = sqrtf(norm + 1e-12f);
                        for (float& v : c) v /= norm;
                        std::string name = "Speaker_" + std::to_string(id);
                        bool exists = false;
                        for (auto& gn : speaker_manager_.speaker_names()) {
                            if (gn == name) { exists = true; break; }
                        }
                        if (!exists) {
                            speaker_manager_.register_speaker(name, c);
                            fprintf(stderr, "[Serve] v4: auto-registered speaker '%s' to global manager\n", name.c_str());
                        }
                    }
                }
            }

            double mel_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mel_t0).count();
            double phase3_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - phase3_t0).count();

            fprintf(stderr, "[Serve] v4 Phase 3: %zu speaker intervals, %d unique speakers (%.1fs) "
                    "[VAD %.0fms, Mel+CAM++ %.0fms (%d segs → %zu chunks), %s VAD, %s Mel]\n",
                    spk_intervals.size(), phase3_speaker_count,
                    phase3_ms / 1000.0,
                    vad_ms, mel_ms, mel_segments, seg_embeddings.size(),
                    gpu_vad_engine_.is_loaded() ? "GPU" : "CPU",
                    gpu_mel_.is_initialized() ? "GPU" : "CPU");
        }

        // 等待 Phase 2 后台线程完成并获取结果
        aligned_words = phase2_future.get();
        fprintf(stderr, "[Serve] v4 Phase 2+3 parallel: %zu words aligned, "
            "%zu speaker intervals (%.1fs total)\n",
            aligned_words.size(), spk_intervals.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - phase2_t0).count());
        phase_t0 = std::chrono::steady_clock::now();

        // ================================================================
        // Phase 4: Word → Speaker 分配 (3 层策略)
        // ================================================================
        struct WordWithSpeaker {
            std::string word;
            int start_ms, end_ms;
            int speaker_id;
            std::string speaker_name;
        };
        std::vector<WordWithSpeaker> word_list;
        word_list.reserve(aligned_words.size());

        // 4a-pre: 零时长聚集重分布 — 修复 ForcedAligner chunk 边界对齐失败
        // 检测 ≥5 个连续 start_ms==end_ms 的词, 含前方大间隙词, 均匀重分布
        {
            size_t idx = 0;
            while (idx < aligned_words.size()) {
                if (aligned_words[idx].start_ms >= 0 &&
                    aligned_words[idx].start_ms == aligned_words[idx].end_ms) {
                    int cluster_ts = aligned_words[idx].start_ms;
                    size_t zero_start = idx;
                    while (idx < aligned_words.size() &&
                           aligned_words[idx].start_ms >= 0 &&
                           aligned_words[idx].start_ms == aligned_words[idx].end_ms &&
                           std::abs(aligned_words[idx].start_ms - cluster_ts) <= 200) {
                        ++idx;
                    }
                    size_t cluster_size = idx - zero_start;
                    if (cluster_size >= 5) {
                        // 向前扩展: 前一个词若与更前词间隙>5s, 说明被错放到 chunk 边界
                        size_t ext_start = zero_start;
                        while (ext_start > 0) {
                            if (aligned_words[ext_start - 1].start_ms < 0) break;
                            int gap_before = 0;
                            if (ext_start >= 2)
                                gap_before = aligned_words[ext_start - 1].start_ms -
                                             aligned_words[ext_start - 2].end_ms;
                            if (gap_before > 5000) { --ext_start; } else break;
                        }
                        size_t total = idx - ext_start;
                        // 左锚点: ext_start 前最后一个正常词的 end_ms
                        int bound_left = std::max(0, cluster_ts - (int)total * 200);
                        for (int j = (int)ext_start - 1; j >= 0; --j) {
                            if (aligned_words[j].start_ms >= 0 &&
                                aligned_words[j].end_ms > aligned_words[j].start_ms) {
                                bound_left = aligned_words[j].end_ms;
                                break;
                            }
                        }
                        int bound_right = cluster_ts;
                        int range = bound_right - bound_left;
                        if (range < (int)total * 60) {
                            bound_right = bound_left + (int)total * 150;
                        }
                        int step = std::max(60, (bound_right - bound_left) / (int)total);
                        for (size_t j = ext_start; j < idx; ++j) {
                            aligned_words[j].start_ms = bound_left + (int)(j - ext_start) * step;
                            aligned_words[j].end_ms = aligned_words[j].start_ms + std::min(step, 100);
                        }
                        fprintf(stderr, "[Serve] v4 4a-pre: redistributed %zu words @%dms → [%d-%d ms] step=%d\n",
                                total, cluster_ts, bound_left, aligned_words[idx-1].end_ms, step);
                    }
                } else { ++idx; }
            }
        }

        // 4a: 零时长词时间戳平滑 — 在相邻词之间分摊时间 (处理散落的个别零时长词)
        for (size_t i = 0; i < aligned_words.size(); ++i) {
            auto& aw = aligned_words[i];
            if (aw.start_ms >= 0 && aw.start_ms == aw.end_ms) {
                int left = (i > 0 && aligned_words[i-1].end_ms > 0) ? aligned_words[i-1].end_ms : aw.start_ms;
                int right = (i + 1 < aligned_words.size() && aligned_words[i+1].start_ms > 0)
                            ? aligned_words[i+1].start_ms : aw.start_ms;
                if (right > aw.start_ms) {
                    aw.end_ms = std::min(right, aw.start_ms + 80);
                } else if (aw.start_ms > left) {
                    aw.start_ms = std::max(left, aw.end_ms - 80);
                }
            }
        }

        // 4b: 主分配 — overlap > 0 或 midpoint 落在 interval 内
        for (auto& aw : aligned_words) {
            WordWithSpeaker wws;
            wws.word = aw.word;
            wws.start_ms = aw.start_ms;
            wws.end_ms = aw.end_ms;
            wws.speaker_id = -1;
            wws.speaker_name = "Unknown";
            if (aw.start_ms >= 0 && !spk_intervals.empty()) {
                int word_mid = (aw.start_ms + aw.end_ms) / 2;
                int best_overlap = 0;
                // 策略1: 最大重叠
                for (auto& si : spk_intervals) {
                    int os = std::max(aw.start_ms, si.start_ms);
                    int oe = std::min(aw.end_ms, si.end_ms);
                    int overlap = oe - os;
                    if (overlap > best_overlap) {
                        best_overlap = overlap;
                        wws.speaker_id = si.speaker_id;
                        wws.speaker_name = si.speaker_name;
                    }
                }
                // 策略2: midpoint 落在 interval 内
                if (wws.speaker_id < 0) {
                    for (auto& si : spk_intervals) {
                        if (word_mid >= si.start_ms && word_mid < si.end_ms) {
                            wws.speaker_id = si.speaker_id;
                            wws.speaker_name = si.speaker_name;
                            break;
                        }
                    }
                }
                // 策略3: nearest-neighbor — 找距离 word 最近的 interval
                if (wws.speaker_id < 0) {
                    int min_dist = INT_MAX;
                    for (auto& si : spk_intervals) {
                        int dist = 0;
                        if (word_mid < si.start_ms) dist = si.start_ms - word_mid;
                        else if (word_mid > si.end_ms) dist = word_mid - si.end_ms;
                        if (dist < min_dist) {
                            min_dist = dist;
                            wws.speaker_id = si.speaker_id;
                            wws.speaker_name = si.speaker_name;
                        }
                    }
                }
            }
            word_list.push_back(wws);
        }

        // 4c: 统计未分配 word (应为 0)
        {
            int unknown_count = 0;
            for (auto& w : word_list) if (w.speaker_id < 0) unknown_count++;
            if (unknown_count > 0)
                fprintf(stderr, "[Serve] v4 Phase 4: WARNING %d/%zu words still Unknown\n",
                        unknown_count, word_list.size());
        }

        // ================================================================
        // Phase 5: 构建 speaker segments
        // ================================================================
        struct V4Segment {
            int start_ms, end_ms;
            int speaker_id;
            std::string speaker_name;
            std::string text;
        };
        std::vector<V4Segment> v4_segments;

        // 5a: 连续同 speaker 合并 (gap ≤ 2s 容忍)
        for (auto& w : word_list) {
            bool extend = !v4_segments.empty() &&
                          w.speaker_id == v4_segments.back().speaker_id &&
                          w.speaker_id >= 0 &&
                          w.start_ms - v4_segments.back().end_ms <= 2000;
            if (extend) {
                v4_segments.back().end_ms = std::max(v4_segments.back().end_ms, w.end_ms);
                v4_segments.back().text += w.word;
            } else {
                V4Segment seg;
                seg.start_ms = w.start_ms;
                seg.end_ms = w.end_ms;
                seg.speaker_id = w.speaker_id;
                seg.speaker_name = w.speaker_name;
                seg.text = w.word;
                v4_segments.push_back(std::move(seg));
            }
        }

        // 5b: 短段吸收 — 极短段 (≤3字) 合并到前后的大段
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<V4Segment> merged;
            for (size_t i = 0; i < v4_segments.size(); ++i) {
                auto& seg = v4_segments[i];
                int char_count = 0;
                for (size_t j = 0; j < seg.text.size(); ) {
                    unsigned char c = seg.text[j];
                    if (c < 0x80) { ++j; } else if (c < 0xE0) { j += 2; } else if (c < 0xF0) { j += 3; } else { j += 4; }
                    ++char_count;
                }
                // 短段 (≤3字符且时长<2s) 尝试合并到同 speaker 邻段
                if (char_count <= 3 && (seg.end_ms - seg.start_ms) < 2000) {
                    // 优先合并到前一段 (同 speaker)
                    if (!merged.empty() &&
                        seg.start_ms - merged.back().end_ms <= 2000 &&
                        (seg.speaker_id == merged.back().speaker_id || seg.speaker_id < 0)) {
                        merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                        merged.back().text += seg.text;
                        continue;
                    }
                    // 否则合并到下一段 (同 speaker)
                    if (i + 1 < v4_segments.size() &&
                        v4_segments[i+1].start_ms - seg.end_ms <= 2000 &&
                        (seg.speaker_id == v4_segments[i+1].speaker_id || seg.speaker_id < 0)) {
                        v4_segments[i+1].start_ms = std::min(v4_segments[i+1].start_ms, seg.start_ms);
                        v4_segments[i+1].text = seg.text + v4_segments[i+1].text;
                        continue;
                    }
                }
                merged.push_back(seg);
            }
            v4_segments = std::move(merged);
        }

        // 5c: 相邻同 speaker 二次合并 (短段吸收可能引入新的连续同 speaker)
        {
            std::vector<V4Segment> merged;
            for (auto& seg : v4_segments) {
                if (!merged.empty() &&
                    seg.speaker_id == merged.back().speaker_id &&
                    seg.start_ms - merged.back().end_ms <= 3000) {
                    merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                    merged.back().text += seg.text;
                } else {
                    merged.push_back(seg);
                }
            }
            v4_segments = std::move(merged);
        }

        // ================================================================
        // Phase 6: 全文标点恢复 → 按 speaker + 句子边界重新分段
        // ================================================================
        // 旧方案: per-segment 独立加标点 → 跨段断句严重
        // 新方案: 全文拼接 → 一次标点恢复 → 按 speaker 和 。？！ 重新切段
        {
            // 6a: 拼接全文, 建立 char→word 映射
            std::string full_text;
            // word_char_map[i] = word_list index for the i-th UTF-8 char in full_text
            std::vector<int> word_char_map;
            for (size_t wi = 0; wi < word_list.size(); ++wi) {
                const auto& w = word_list[wi];
                for (size_t j = 0; j < w.word.size(); ) {
                    unsigned char c = (unsigned char)w.word[j];
                    int clen = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                    full_text += w.word.substr(j, clen);
                    word_char_map.push_back((int)wi);
                    j += clen;
                }
            }

            // 6b: 全文标点恢复
            std::string punctuated = punctuation_restorer_.restore(full_text);

            // 6c: 对齐 — 遍历标点后文本, 将原始字符映射回 word_list
            // 标点后的每个字符: 如果是原始字符则推进原始指针, 如果是新增标点则标记为标点
            struct CharInfo {
                std::string ch;
                int word_idx;       // -1 = inserted punctuation
                bool is_punc;
            };
            std::vector<CharInfo> char_infos;
            {
                // split both into UTF-8 chars
                auto split_u8 = [](const std::string& s) {
                    std::vector<std::string> out;
                    for (size_t i = 0; i < s.size(); ) {
                        unsigned char c = (unsigned char)s[i];
                        int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                        out.push_back(s.substr(i, len));
                        i += len;
                    }
                    return out;
                };
                auto orig_chars = split_u8(full_text);
                auto punc_chars = split_u8(punctuated);

                size_t oi = 0; // original char index
                for (size_t pi = 0; pi < punc_chars.size(); ++pi) {
                    if (oi < orig_chars.size() && punc_chars[pi] == orig_chars[oi]) {
                        // original char — map to word
                        char_infos.push_back({punc_chars[pi], word_char_map[oi], false});
                        ++oi;
                    } else {
                        // inserted punctuation — inherit from last original char
                        int inherit_wi = (char_infos.empty()) ? 0 :
                            (char_infos.back().word_idx >= 0) ? char_infos.back().word_idx : 0;
                        char_infos.push_back({punc_chars[pi], inherit_wi, true});
                    }
                }
            }

            // 6d: 重建 segments — 在 speaker 变化 或 句末标点(。？！) 处切段
            auto is_sentence_end = [](const std::string& c) {
                return c == "\xe3\x80\x82" || c == "\xef\xbc\x9f" || c == "\xef\xbc\x81"; // 。？！
            };

            v4_segments.clear();
            if (!char_infos.empty()) {
                // 找第一个有效 word_idx
                int first_wi = 0;
                for (auto& ci : char_infos) {
                    if (ci.word_idx >= 0) { first_wi = ci.word_idx; break; }
                }

                V4Segment cur;
                cur.speaker_id = word_list[first_wi].speaker_id;
                cur.speaker_name = word_list[first_wi].speaker_name;
                cur.start_ms = word_list[first_wi].start_ms;
                cur.end_ms = word_list[first_wi].end_ms;

                for (size_t i = 0; i < char_infos.size(); ++i) {
                    auto& ci = char_infos[i];
                    int wi = ci.word_idx >= 0 ? ci.word_idx : -1;

                    // 检查是否需要在此处切段
                    bool split_here = false;
                    // 句末标点检查: 原始标点或插入标点均可触发 (ASR 输出可能已有标点)
                    bool is_sent_end = is_sentence_end(ci.ch) && i + 1 < char_infos.size();
                    // 逗号也可用于长段切分
                    bool is_comma = (ci.ch == "\xef\xbc\x8c") && i + 1 < char_infos.size(); // ，

                    // 计算当前段的 UTF-8 字符数 (用于字符计数的切分条件)
                    int cur_char_count = 0;
                    {
                        for (size_t ci2 = 0; ci2 < cur.text.size(); ) {
                            unsigned char uc = (unsigned char)cur.text[ci2];
                            int cl = (uc < 0x80) ? 1 : (uc < 0xE0) ? 2 : (uc < 0xF0) ? 3 : 4;
                            ci2 += cl;
                            cur_char_count++;
                        }
                    }

                    if (is_sent_end || is_comma) {
                        // 句末标点 — 检查后续文字的 speaker
                        int next_wi = -1;
                        for (size_t j = i + 1; j < char_infos.size(); ++j) {
                            if (char_infos[j].word_idx >= 0) {
                                next_wi = char_infos[j].word_idx;
                                break;
                            }
                        }
                        if (is_sent_end && next_wi >= 0 && word_list[next_wi].speaker_id != cur.speaker_id) {
                            // speaker 变化 + 句末标点: 必须切
                            split_here = true;
                        }
                        // 同说话人长段: 段时长超 10s 或文本超 30 字时在句末标点处切分
                        if (!split_here && is_sent_end &&
                            ((cur.end_ms - cur.start_ms) > 10000 || cur_char_count > 30)) {
                            split_here = true;
                        }
                        // 较长段: 段时长超 15s 或文本超 40 字时, 在逗号处也可切分
                        if (!split_here && is_comma &&
                            ((cur.end_ms - cur.start_ms) > 15000 || cur_char_count > 40)) {
                            split_here = true;
                        }
                    }

                    // speaker 变化 (无标点处): 立即切段。
                    // 这里不再回退到历史标点/逗号处切分，否则会把旧 speaker 的文本/时间错误转移到新 speaker。
                    if (!split_here && !ci.is_punc && wi >= 0 &&
                        word_list[wi].speaker_id != cur.speaker_id) {
                        if (!cur.text.empty()) v4_segments.push_back(cur);
                        cur = V4Segment();
                        cur.speaker_id = word_list[wi].speaker_id;
                        cur.speaker_name = word_list[wi].speaker_name;
                        cur.start_ms = word_list[wi].start_ms;
                        cur.end_ms = word_list[wi].end_ms;
                        cur.text = ci.ch;
                        continue;
                    }

                    // 加入当前段
                    cur.text += ci.ch;
                    if (wi >= 0) {
                        cur.end_ms = std::max(cur.end_ms, word_list[wi].end_ms);
                    }

                    // 如果句末标点 + speaker 变化: 当前段结束
                    if (split_here) {
                        if (!cur.text.empty()) v4_segments.push_back(cur);
                        // 新段从下一个字符开始
                        int next_wi = -1;
                        for (size_t j = i + 1; j < char_infos.size(); ++j) {
                            if (char_infos[j].word_idx >= 0) {
                                next_wi = char_infos[j].word_idx;
                                break;
                            }
                        }
                        cur = V4Segment();
                        if (next_wi >= 0) {
                            cur.speaker_id = word_list[next_wi].speaker_id;
                            cur.speaker_name = word_list[next_wi].speaker_name;
                            cur.start_ms = word_list[next_wi].start_ms;
                            cur.end_ms = word_list[next_wi].end_ms;
                        }
                    }
                }
                // 最后一段
                if (!cur.text.empty()) v4_segments.push_back(cur);
            }

            // DEBUG: Phase 6d 段数统计
            {
                int same_spk_splits = 0, spk_change_splits = 0;
                int max_dur = 0;
                for (auto& seg : v4_segments) {
                    int dur = seg.end_ms - seg.start_ms;
                    if (dur > max_dur) max_dur = dur;
                }
                fprintf(stderr, "[Serve] v4 Phase 6d: %zu segments, max_dur=%dms\n",
                        v4_segments.size(), max_dur);
                // Show top 5 longest
                std::vector<std::pair<int,int>> dur_idx;
                for (int i = 0; i < (int)v4_segments.size(); ++i)
                    dur_idx.push_back({v4_segments[i].end_ms - v4_segments[i].start_ms, i});
                std::sort(dur_idx.rbegin(), dur_idx.rend());
                for (int k = 0; k < std::min(5, (int)dur_idx.size()); ++k) {
                    auto& seg = v4_segments[dur_idx[k].second];
                    fprintf(stderr, "  [%d] %dms, spk=%d, %zu chars\n",
                            dur_idx[k].second, dur_idx[k].first, seg.speaker_id, seg.text.size());
                }
            }

            // 6e: 后处理 — 段末标点修正
            for (auto& seg : v4_segments) {
                if (seg.text.empty()) continue;
                // 检查末尾 3 字节是否是句末标点
                bool ends_sent = false;
                if (seg.text.size() >= 3) {
                    std::string last3 = seg.text.substr(seg.text.size() - 3);
                    ends_sent = (last3 == "\xe3\x80\x82" || last3 == "\xef\xbc\x9f" || last3 == "\xef\xbc\x81");
                }
                if (!ends_sent) {
                    // 如果末尾是逗号, 替换为句号; 否则追加句号
                    if (seg.text.size() >= 3 && seg.text.substr(seg.text.size() - 3) == "\xef\xbc\x8c") {
                        seg.text.replace(seg.text.size() - 3, 3, "\xe3\x80\x82"); // ，→。
                    } else {
                        seg.text += "\xe3\x80\x82"; // 。
                    }
                }
            }
        }

        // ================================================================
        // Phase 6.5: 说话人时序平滑 (P1) — 消除噪声 speaker island
        // ================================================================
        // 规则: 若某 segment 时长 < 5s 且被相同 speaker 前后包围 → 合并到前后
        // 目的: 消除极短的说话人片段 (通常是噪声或识别错误)
        // 示例: [Speaker_0: 30s] → [Speaker_3: 1.5s] → [Speaker_0: 40s]
        //       变为: [Speaker_0: 71.5s]
        {
            std::vector<V4Segment> smoothed;
            int island_merged = 0;

            for (size_t i = 0; i < v4_segments.size(); ++i) {
                auto& seg = v4_segments[i];
                int dur_ms = seg.end_ms - seg.start_ms;

                // 检查: 是否为短 segment (<3s) 且前后被同一 speaker 包围?
                bool is_island = dur_ms < 3000 && seg.speaker_id >= 0 &&
                                 !smoothed.empty() && i + 1 < v4_segments.size();
                bool surrounded = is_island &&
                                  smoothed.back().speaker_id >= 0 &&
                                  smoothed.back().speaker_id == v4_segments[i+1].speaker_id &&
                                  smoothed.back().speaker_id != seg.speaker_id;

                if (surrounded) {
                    // 当前短段是噪声 island, 吸收到前一段；下一段会在后续 pass 中与前一段重新并上
                    smoothed.back().end_ms = std::max(smoothed.back().end_ms, seg.end_ms);
                    smoothed.back().text += seg.text;
                    ++island_merged;
                } else {
                    smoothed.push_back(seg);
                }
            }

            if (island_merged > 0) {
                fprintf(stderr, "[Serve] v4 Phase 6.5: smoothed %d speaker islands\n", island_merged);
            }

            // 吸收 island 后再做一次相邻同 speaker 合并，避免残留同 speaker 碎段。
            // 仅合并短碎段 — 若两段都 >5s 则保持句子级分段; 结果不得超 30s
            std::vector<V4Segment> merged;
            for (auto& seg : smoothed) {
                if (!merged.empty() &&
                    seg.speaker_id == merged.back().speaker_id &&
                    seg.start_ms - merged.back().end_ms <= 2000) {
                    int prev_dur = merged.back().end_ms - merged.back().start_ms;
                    int cur_dur = seg.end_ms - seg.start_ms;
                    int merged_dur = std::max(merged.back().end_ms, seg.end_ms) - merged.back().start_ms;
                    if ((prev_dur <= 5000 || cur_dur <= 5000) && merged_dur <= 30000) {
                        merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                        merged.back().text += seg.text;
                    } else {
                        merged.push_back(seg);
                    }
                } else {
                    merged.push_back(seg);
                }
            }
            v4_segments = std::move(merged);
        }

        // ================================================================
        // Phase 6.55: 口语规范化 (P2) — 去除冗余 oral 短语 (可选)
        // ================================================================
        // 规则: 若 clean=true, 去除常见口语冗余词
        // 示例冗余: "我想说", "你知道吗", "就是", "然后呢", "对吧", "这样的话"
        if (clean_oral) {
            // 常见口语冗余词库
            static const std::vector<std::string> oral_patterns = {
                "\xe6\x88\x91\xe6\x83\xb3\xe8\xaf\xb4",  // 我想说
                "\xe4\xbd\xa0\xe7\x9f\xa5\xe9\x81\x93\xe5\x90\x97",  // 你知道吗
                "\xef\xbc\x8c\xe5\x92\x8b",  // ，咋
                "\xef\xbc\x8c\xe5\xb0\xb1\xe6\x98\xaf",  // ，就是
                "\xe7\x84\xb6\xe5\x90\x8e\xe5\x91\xa2",  // 然后呢
                "\xe5\xaf\xb9\xe5\x90\xa7",  // 对吧
                "\xe8\xbf\x99\xe6\xa0\xb7\xe7\x9a\x84\xe8\xaf\x9d",  // 这样的话
                "\xef\xbc\x8c\xe6\x80\x8e\xe6\xa0\xb7",  // ，怎样
                "\xef\xbc\x8c\xe8\xae\xb2",  // ，讲
                "\xe7\xad\x89\xe7\xad\x89",  // 等等
                "\xe4\xb9\x9f\xe6\x98\xaf\xef\xbc\x8c",  // 也是，
                "\xef\xbc\x8c\xe6\x9c\x89\xe4\xb8\x00",  // ，有一
                "\xe8\xbf\x99\xe4\xb8\xaa\xe8\xaf\xb8",  // 这个诸
                "\xef\xbc\x8c\xe5\x9c\xa8\xe4\xba\x8e",  // ，在于
                "\xe7\xb1\xbb\xe4\xba\x8b",  // 类事
                "\xef\xbc\x8c\xe5\x8f\xab\xe4\xbb\x80\xe4\xb9\x88",  // ，叫什么
            };
            
            int oral_removed = 0;
            for (auto& seg : v4_segments) {
                std::string orig_len = std::to_string(seg.text.size());
                for (const auto& pattern : oral_patterns) {
                    size_t pos = 0;
                    while ((pos = seg.text.find(pattern, pos)) != std::string::npos) {
                        seg.text.erase(pos, pattern.size());
                        ++oral_removed;
                    }
                }
            }
            
            if (oral_removed > 0) {
                fprintf(stderr, "[Serve] v4 Phase 6.55: removed %d oral redundancies\n", oral_removed);
            }
        }

        // ================================================================
        // Phase 6.6: 空白区间填充 — 用 chunk 标签覆盖 segment 间的空隙
        // ================================================================
        // 评估显示 398s 未覆盖区间导致 ~11% 准确率损失
        // 对每个 segment 间的空隙, 用最近的 spk_interval 标签填充
        if (!spk_intervals.empty() && !v4_segments.empty()) {
            // 先按时间排序 v4_segments
            std::stable_sort(v4_segments.begin(), v4_segments.end(),
                [](const V4Segment& a, const V4Segment& b) { return a.start_ms < b.start_ms; });

            std::vector<V4Segment> filled;
            int gap_filled = 0;
            float gap_duration_ms = 0;

            // 填充第一个 segment 之前的空隙
            if (v4_segments[0].start_ms > 0) {
                int gap_start = 0;
                int gap_end = v4_segments[0].start_ms;
                if (gap_end - gap_start >= 200) {  // 至少 200ms 才值得填
                    int gap_mid = (gap_start + gap_end) / 2;
                    // 找最近的 spk_interval
                    int best_dist = INT_MAX;
                    int best_spk = v4_segments[0].speaker_id;
                    std::string best_name = v4_segments[0].speaker_name;
                    for (auto& si : spk_intervals) {
                        int center = (si.start_ms + si.end_ms) / 2;
                        int dist = std::abs(center - gap_mid);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_spk = si.speaker_id;
                            best_name = si.speaker_name;
                        }
                    }
                    V4Segment gap_seg;
                    gap_seg.start_ms = gap_start;
                    gap_seg.end_ms = gap_end;
                    gap_seg.speaker_id = best_spk;
                    gap_seg.speaker_name = best_name;
                    gap_seg.text = "";  // 无文字的空白段
                    filled.push_back(gap_seg);
                    ++gap_filled;
                    gap_duration_ms += gap_end - gap_start;
                }
            }

            for (size_t i = 0; i < v4_segments.size(); ++i) {
                filled.push_back(v4_segments[i]);

                // 检查与下一个 segment 之间的空隙
                if (i + 1 < v4_segments.size()) {
                    int gap_start = v4_segments[i].end_ms;
                    int gap_end = v4_segments[i + 1].start_ms;
                    if (gap_end - gap_start >= 200) {
                        int gap_mid = (gap_start + gap_end) / 2;
                        int best_dist = INT_MAX;
                        int best_spk = v4_segments[i].speaker_id;
                        std::string best_name = v4_segments[i].speaker_name;
                        for (auto& si : spk_intervals) {
                            int center = (si.start_ms + si.end_ms) / 2;
                            int dist = std::abs(center - gap_mid);
                            if (dist < best_dist) {
                                best_dist = dist;
                                best_spk = si.speaker_id;
                                best_name = si.speaker_name;
                            }
                        }
                        V4Segment gap_seg;
                        gap_seg.start_ms = gap_start;
                        gap_seg.end_ms = gap_end;
                        gap_seg.speaker_id = best_spk;
                        gap_seg.speaker_name = best_name;
                        gap_seg.text = "";
                        filled.push_back(gap_seg);
                        ++gap_filled;
                        gap_duration_ms += gap_end - gap_start;
                    }
                }
            }

            // 填充最后一个 segment 之后 (到音频结束)
            int audio_end_ms = (int)((float)wav.samples.size() / wav.sample_rate * 1000);
            if (v4_segments.back().end_ms < audio_end_ms - 200) {
                int gap_start = v4_segments.back().end_ms;
                int gap_end = audio_end_ms;
                int gap_mid = (gap_start + gap_end) / 2;
                int best_dist = INT_MAX;
                int best_spk = v4_segments.back().speaker_id;
                std::string best_name = v4_segments.back().speaker_name;
                for (auto& si : spk_intervals) {
                    int center = (si.start_ms + si.end_ms) / 2;
                    int dist = std::abs(center - gap_mid);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_spk = si.speaker_id;
                        best_name = si.speaker_name;
                    }
                }
                V4Segment gap_seg;
                gap_seg.start_ms = gap_start;
                gap_seg.end_ms = gap_end;
                gap_seg.speaker_id = best_spk;
                gap_seg.speaker_name = best_name;
                gap_seg.text = "";
                filled.push_back(gap_seg);
                ++gap_filled;
                gap_duration_ms += gap_end - gap_start;
            }

            if (gap_filled > 0) {
                fprintf(stderr, "[Serve] v4 Phase 6.6: filled %d gaps (%.1fs)\n",
                        gap_filled, gap_duration_ms / 1000.0f);
                v4_segments = std::move(filled);
            }
        }

        // 最终输出前按时间排序，避免 speaker 切段回退/吸收造成的非单调顺序影响前端展示与人工校对。
        std::stable_sort(v4_segments.begin(), v4_segments.end(),
                         [](const V4Segment& a, const V4Segment& b) {
                             if (a.start_ms != b.start_ms) return a.start_ms < b.start_ms;
                             if (a.end_ms != b.end_ms) return a.end_ms < b.end_ms;
                             return a.speaker_id < b.speaker_id;
                         });

        fprintf(stderr, "[Serve] v4 pipeline done: %zu segments, %zu words, total %.1fs\n",
                v4_segments.size(), word_list.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - v4_t0).count());

        // 构建响应
        HttpResponse resp;
        std::string out_text;
        for (auto& seg : v4_segments) out_text += seg.text;

        if (response_format == "text") {
            resp.content_type = "text/plain";
            for (auto& seg : v4_segments) {
                resp.body += "[" + seg.speaker_name + "] " + seg.text + "\n";
            }
        } else if (response_format == "verbose_json") {
            resp.body = "{\"task\":\"transcribe\",\"language\":\"" + json_escape(language) +
                        "\",\"duration\":" + std::to_string(total_duration_s) +
                        ",\"text\":\"" + json_escape(out_text) + "\"";
            // words
            resp.body += ",\"words\":[";
            for (size_t i = 0; i < word_list.size(); ++i) {
                auto& w = word_list[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"word\":\"" + json_escape(w.word) +
                             "\",\"start\":" + std::to_string(w.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(w.end_ms / 1000.0f) +
                             ",\"speaker\":\"" + json_escape(w.speaker_name) + "\"}";
            }
            resp.body += "]";
            // segments
            resp.body += ",\"segments\":[";
            for (size_t i = 0; i < v4_segments.size(); ++i) {
                auto& s = v4_segments[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"start\":" + std::to_string(s.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(s.end_ms / 1000.0f) +
                             ",\"text\":\"" + json_escape(s.text) +
                             "\",\"speaker\":\"" + json_escape(s.speaker_name) +
                             "\",\"speaker_id\":" + std::to_string(s.speaker_id) + "}";
            }
            resp.body += "]}";
        } else {
            resp.body = "{\"text\":\"" + json_escape(out_text) + "\"";
            resp.body += ",\"segments\":[";
            for (size_t i = 0; i < v4_segments.size(); ++i) {
                auto& s = v4_segments[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"start\":" + std::to_string(s.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(s.end_ms / 1000.0f) +
                             ",\"text\":\"" + json_escape(s.text) +
                             "\",\"speaker\":\"" + json_escape(s.speaker_name) + "\"}";
            }
            resp.body += "]}";
        }

        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // ========================================================================
    // 说话人分割模式 (v2): Speaker-first pipeline
    //   Phase 1: Fine-grained VAD → short segments for accurate speaker ID
    //   Phase 2: CAM++ speaker ID per segment
    //   Phase 3: Group consecutive same-speaker segments into speaker turns
    //   Phase 4: ASR each speaker turn with full context (up to ~100s)
    //   Phase 5: Build output segments
    // ========================================================================
    if (identify_spk && speaker_encoder_ && vad_engine_.is_loaded()) {
        // 解析音频到 PCM
        audio::AudioData wav;
        if (!audio::load_audio_from_memory(
                reinterpret_cast<const uint8_t*>(audio_data.data()),
                audio_data.size(), wav, audio_filename)) {
            HttpResponse resp;
            resp.status_code = 400;
            resp.body = "{\"error\":{\"message\":\"Failed to decode audio for diarization\"}}";
            send_response(client_fd, resp);
            close(client_fd);
            return;
        }

        float total_duration_s = (float)wav.samples.size() / wav.sample_rate;

        fprintf(stderr, "[Serve] Diarization v2: audio decoded to %.1fs, %zu samples\n",
                total_duration_s, wav.samples.size());

        // Phase 1: Fine-grained VAD (short segments for accurate speaker ID)
        std::vector<asr::VadSegment> vad_segments;
        if (gpu_vad_engine_.is_loaded()) {
            // GPU VAD: batch FSMN + cuFFT
            auto gpu_segs = gpu_vad_engine_.detect_all(
                wav.samples.data(), (int)wav.samples.size(), 300, 8000);
            for (auto& gs : gpu_segs) {
                asr::VadSegment vs;
                vs.start_ms = gs.start_ms;
                vs.end_ms = gs.end_ms;
                int64_t s0 = (int64_t)gs.start_ms * wav.sample_rate / 1000;
                int64_t s1 = (int64_t)gs.end_ms * wav.sample_rate / 1000;
                s0 = std::max((int64_t)0, std::min(s0, (int64_t)wav.samples.size()));
                s1 = std::max(s0, std::min(s1, (int64_t)wav.samples.size()));
                vs.pcm.assign(wav.samples.data() + s0, wav.samples.data() + s1);
                vad_segments.push_back(std::move(vs));
            }
        } else {
            std::lock_guard<std::mutex> lock(vad_mutex_);
            auto& cfg = vad_engine_.mutable_config();
            int orig_max_end_silence = cfg.max_end_silence_time;
            int orig_max_segment = cfg.max_single_segment_time;
            cfg.max_end_silence_time = 300;      // 300ms 静音切分 (细粒度说话人边界)
            cfg.max_single_segment_time = 8000;   // 8s per segment (CAM++ 最佳区间)
            vad_segments = vad_engine_.detect_all(wav.samples.data(), (int)wav.samples.size());
            cfg.max_end_silence_time = orig_max_end_silence;
            cfg.max_single_segment_time = orig_max_segment;
        }

        fprintf(stderr, "[Serve] Diarization v2: Phase 1 - %zu VAD segments from %.1fs audio\n",
                vad_segments.size(), total_duration_s);

        // Phase 2: Speaker ID per segment (skip silence & too-short segments)
        // Use a LOCAL SpeakerManager for diarization to avoid cross-request contamination
        asr::SpeakerManager diar_spk_mgr;

        struct SpkSegment {
            int start_ms, end_ms;
            int speaker_id;
            std::string speaker_name;
            float speaker_sim;
            size_t vad_index;  // index into vad_segments for PCM retrieval
        };
        std::vector<SpkSegment> spk_segments;

        for (size_t vi = 0; vi < vad_segments.size(); ++vi) {
            auto& vseg = vad_segments[vi];
            if (vseg.pcm.empty() || vseg.end_ms - vseg.start_ms < 200) continue;

            // Skip low-energy segments
            float rms = 0.0f;
            for (size_t si = 0; si < vseg.pcm.size(); si++)
                rms += vseg.pcm[si] * vseg.pcm[si];
            rms = std::sqrt(rms / vseg.pcm.size());
            if (rms < 0.005f) continue;

            SpkSegment ss;
            ss.start_ms = vseg.start_ms;
            ss.end_ms = vseg.end_ms;
            ss.vad_index = vi;

            // Extract embedding using GPU encoder
            std::vector<float> mel;
            int num_frames = 0;
            if (gpu_mel_.is_initialized()) {
                num_frames = gpu_mel_.compute(vseg.pcm.data(), (int)vseg.pcm.size(), mel);
            } else {
                compute_mel_80(vseg.pcm.data(), (int)vseg.pcm.size(), wav.sample_rate, mel, num_frames);
            }
            if (num_frames < 10) continue;

            std::vector<float> embedding;
            {
                std::lock_guard<std::mutex> spk_lock(speaker_mutex_);
                embedding = speaker_encoder_->extract(mel.data(), num_frames);
            }
            if (embedding.empty()) continue;

            auto spk = diar_spk_mgr.identify(embedding, 0.78f, true);
            ss.speaker_id = spk.speaker_id;
            ss.speaker_name = spk.speaker_id >= 0 ? spk.name : "Unknown";
            ss.speaker_sim = spk.similarity;

            fprintf(stderr, "[Serve] Phase 2: seg %zu [%d-%d ms] → %s (sim=%.3f, %s)\n",
                    vi, ss.start_ms, ss.end_ms, ss.speaker_name.c_str(),
                    ss.speaker_sim, spk.is_new ? "NEW" : "matched");

            spk_segments.push_back(ss);
        }

        fprintf(stderr, "[Serve] Diarization v2: Phase 2 - %zu speaker-labeled segments, %d unique speakers\n",
                spk_segments.size(), diar_spk_mgr.speaker_count());

        // Phase 3: Group consecutive same-speaker segments into turns (max ~100s per turn)
        struct SpeakerTurn {
            int start_ms, end_ms;
            int speaker_id;
            std::string speaker_name;
            float speaker_sim;
            std::string text;
            std::vector<size_t> vad_indices; // indices into vad_segments for PCM
        };
        std::vector<SpeakerTurn> turns;
        const int max_turn_ms = 100000; // 100s max per turn (encoder limit ~115s)

        for (auto& ss : spk_segments) {
            bool extend = !turns.empty()
                       && ss.speaker_id == turns.back().speaker_id
                       && ss.speaker_id >= 0
                       && ss.start_ms - turns.back().end_ms <= 1000
                       && (ss.end_ms - turns.back().start_ms) <= max_turn_ms;
            if (extend) {
                turns.back().end_ms = ss.end_ms;
                turns.back().vad_indices.push_back(ss.vad_index);
            } else {
                SpeakerTurn t;
                t.start_ms = ss.start_ms;
                t.end_ms = ss.end_ms;
                t.speaker_id = ss.speaker_id;
                t.speaker_name = ss.speaker_name;
                t.speaker_sim = ss.speaker_sim;
                t.vad_indices.push_back(ss.vad_index);
                turns.push_back(std::move(t));
            }
        }

        fprintf(stderr, "[Serve] Diarization v2: Phase 3 - %zu speaker turns\n", turns.size());

        // Phase 4: ASR each speaker turn (concatenate VAD PCMs for clean speech)
        for (size_t ti = 0; ti < turns.size(); ++ti) {
            auto& turn = turns[ti];

            // Concatenate only VAD segment PCMs (excludes silence/noise between segments)
            std::vector<float> turn_pcm;
            const int silence_pad = wav.sample_rate / 4; // 250ms silence between segments
            for (size_t vi = 0; vi < turn.vad_indices.size(); ++vi) {
                auto& vseg = vad_segments[turn.vad_indices[vi]];
                if (vi > 0 && !turn_pcm.empty()) {
                    // Insert brief silence between concatenated segments
                    turn_pcm.resize(turn_pcm.size() + silence_pad, 0.0f);
                }
                turn_pcm.insert(turn_pcm.end(), vseg.pcm.begin(), vseg.pcm.end());
            }

            float turn_dur_s = (float)turn_pcm.size() / wav.sample_rate;
            fprintf(stderr, "[Serve] Diarization v2: Phase 4 - turn %zu/%zu [%d-%d ms] "
                    "(speech %.1fs from %zu segs) speaker=%s\n",
                    ti + 1, turns.size(), turn.start_ms, turn.end_ms,
                    turn_dur_s, turn.vad_indices.size(), turn.speaker_name.c_str());

            if ((int)turn_pcm.size() < wav.sample_rate / 5) continue; // < 200ms, skip

            auto seg_result = asr_plugin_->transcribe_pcm(
                turn_pcm.data(), (int)turn_pcm.size(),
                wav.sample_rate, language, true);

            if (seg_result.error_code == 0 && !seg_result.text.empty()) {
                turn.text = seg_result.text;
            }
        }

        // Phase 5: Build output segments
        struct TransSegment {
            int start_ms;
            int end_ms;
            std::string text;
            std::string text_with_punc;
            std::string speaker;
            int speaker_id;
            float speaker_sim;
        };
        std::vector<TransSegment> segments;

        for (auto& turn : turns) {
            if (turn.text.empty()) continue;
            TransSegment ts;
            ts.start_ms = turn.start_ms;
            ts.end_ms = turn.end_ms;
            ts.text = turn.text;
            ts.speaker = turn.speaker_name;
            ts.speaker_id = turn.speaker_id;
            ts.speaker_sim = turn.speaker_sim;
            if (punctuate) {
                ts.text_with_punc = punctuation_restorer_.restore(ts.text);
            }
            segments.push_back(std::move(ts));
        }

        // Merge adjacent same-speaker segments (gap < 500ms)
        for (size_t i = 1; i < segments.size(); ) {
            auto& prev = segments[i - 1];
            auto& cur = segments[i];
            if (cur.speaker_id == prev.speaker_id && cur.speaker_id >= 0 &&
                cur.start_ms - prev.end_ms <= 500) {
                prev.end_ms = cur.end_ms;
                prev.text += cur.text;
                if (punctuate && !cur.text_with_punc.empty()) {
                    prev.text_with_punc += cur.text_with_punc;
                }
                segments.erase(segments.begin() + i);
            } else {
                ++i;
            }
        }

        fprintf(stderr, "[Serve] Diarization v2: %zu segments after merge\n", segments.size());

        // 4. 构建响应
        HttpResponse resp;
        if (segments.empty()) {
            // 回退: 无有效 VAD 段, 做整段 ASR
            auto fallback = asr_plugin_->transcribe_memory(
                reinterpret_cast<const uint8_t*>(audio_data.data()),
                audio_data.size(), language, audio_filename, suppress_early_eos);
            std::string fallback_text = fallback.text;
            if (punctuate && !fallback_text.empty()) {
                fallback_text = punctuation_restorer_.restore(fallback_text);
            }
            resp.body = "{\"text\":\"" + json_escape(fallback_text) + "\"}";
        } else if (response_format == "text") {
            resp.content_type = "text/plain";
            std::string body;
            for (auto& s : segments) {
                body += "[" + s.speaker + "] ";
                body += punctuate && !s.text_with_punc.empty() ? s.text_with_punc : s.text;
                body += "\n";
            }
            resp.body = body;
        } else if (response_format == "verbose_json") {
            // 完整 JSON: 包含 segments 数组 + 合并文本
            std::string full_text;
            for (auto& s : segments) {
                full_text += punctuate && !s.text_with_punc.empty() ? s.text_with_punc : s.text;
            }
            resp.body = "{\"task\":\"transcribe\",\"language\":\"" +
                        json_escape(language) +
                        "\",\"duration\":" + std::to_string(total_duration_s) +
                        ",\"text\":\"" + json_escape(full_text) + "\"";
            if (punctuate) {
                resp.body += ",\"text_with_punc\":\"" + json_escape(full_text) + "\"";
            }
            resp.body += ",\"segments\":[";
            for (size_t i = 0; i < segments.size(); ++i) {
                auto& s = segments[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"start\":" + std::to_string(s.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(s.end_ms / 1000.0f) +
                             ",\"text\":\"" + json_escape(
                                 punctuate && !s.text_with_punc.empty() ? s.text_with_punc : s.text
                             ) + "\",\"speaker\":\"" + json_escape(s.speaker) +
                             "\",\"speaker_id\":" + std::to_string(s.speaker_id) +
                             ",\"speaker_similarity\":" + std::to_string(s.speaker_sim) + "}";
            }
            resp.body += "]}";
        } else {
            // 默认 json: segments 数组
            std::string full_text;
            for (auto& s : segments) {
                full_text += punctuate && !s.text_with_punc.empty() ? s.text_with_punc : s.text;
            }
            resp.body = "{\"text\":\"" + json_escape(full_text) + "\"";
            resp.body += ",\"segments\":[";
            for (size_t i = 0; i < segments.size(); ++i) {
                auto& s = segments[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"start\":" + std::to_string(s.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(s.end_ms / 1000.0f) +
                             ",\"text\":\"" + json_escape(
                                 punctuate && !s.text_with_punc.empty() ? s.text_with_punc : s.text
                             ) + "\",\"speaker\":\"" + json_escape(s.speaker) + "\"}";
            }
            resp.body += "]}";
        }

        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // ========================================================================
    // 非分割模式: 整段 ASR + 可选整段 Speaker ID (原有逻辑)
    // 长音频 (>100s) 自动 VAD 分段转录
    // ========================================================================

    // 首先解析音频到 PCM，判断时长
    audio::AudioData wav_plain;
    bool parsed_pcm = audio::load_audio_from_memory(
        reinterpret_cast<const uint8_t*>(audio_data.data()),
        audio_data.size(), wav_plain, audio_filename);

    float audio_duration_s = parsed_pcm ? (float)wav_plain.samples.size() / wav_plain.sample_rate : 0;

    plugins::AsrResult result;

    // 长音频: VAD 分段 → 分组 → 逐段转录 → 拼接
    if (parsed_pcm && audio_duration_s > 100.0f && vad_engine_.is_loaded()) {
        fprintf(stderr, "[Serve] Plain mode: long audio %.1fs, using VAD chunked transcription\n",
                audio_duration_s);

        // VAD 分段
        std::vector<asr::VadSegment> vad_segs;
        {
            std::lock_guard<std::mutex> lock(vad_mutex_);
            auto& cfg = vad_engine_.mutable_config();
            int orig_silence = cfg.max_end_silence_time;
            int orig_segment = cfg.max_single_segment_time;
            cfg.max_end_silence_time = 500;       // 500ms 静音切分
            cfg.max_single_segment_time = 15000;   // 15s per segment
            vad_segs = vad_engine_.detect_all(wav_plain.samples.data(), (int)wav_plain.samples.size());
            cfg.max_end_silence_time = orig_silence;
            cfg.max_single_segment_time = orig_segment;
        }

        fprintf(stderr, "[Serve] Plain mode: %zu VAD segments\n", vad_segs.size());

        // 分组: 连续 VAD 段合并为 ≤100s 的 chunk
        struct Chunk {
            std::vector<size_t> seg_indices;
            int start_ms, end_ms;
        };
        std::vector<Chunk> chunks;
        const int max_chunk_ms = 100000;

        for (size_t i = 0; i < vad_segs.size(); ++i) {
            auto& vs = vad_segs[i];
            int dur = vs.end_ms - vs.start_ms;
            if (dur < 200) continue; // skip very short segments

            bool extend = !chunks.empty() &&
                          (vs.end_ms - chunks.back().start_ms) <= max_chunk_ms;
            if (extend) {
                chunks.back().seg_indices.push_back(i);
                chunks.back().end_ms = vs.end_ms;
            } else {
                Chunk c;
                c.seg_indices.push_back(i);
                c.start_ms = vs.start_ms;
                c.end_ms = vs.end_ms;
                chunks.push_back(std::move(c));
            }
        }

        fprintf(stderr, "[Serve] Plain mode: %zu chunks from %zu VAD segments\n",
                chunks.size(), vad_segs.size());

        // 逐 chunk 转录
        std::string full_text;
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            auto& chunk = chunks[ci];

            // 拼接 VAD 段 PCM (段间插入 250ms 静音)
            std::vector<float> chunk_pcm;
            const int silence_pad = wav_plain.sample_rate / 4;
            for (size_t vi = 0; vi < chunk.seg_indices.size(); ++vi) {
                auto& vseg = vad_segs[chunk.seg_indices[vi]];
                if (vi > 0 && !chunk_pcm.empty()) {
                    chunk_pcm.resize(chunk_pcm.size() + silence_pad, 0.0f);
                }
                chunk_pcm.insert(chunk_pcm.end(), vseg.pcm.begin(), vseg.pcm.end());
            }

            if ((int)chunk_pcm.size() < wav_plain.sample_rate / 5) continue;

            auto seg_result = asr_plugin_->transcribe_pcm(
                chunk_pcm.data(), (int)chunk_pcm.size(),
                wav_plain.sample_rate, language, true);

            if (seg_result.error_code == 0 && !seg_result.text.empty()) {
                full_text += seg_result.text;
            }
        }

        result.text = full_text;
        result.language = language;
        result.duration_s = audio_duration_s;
        if (result.text.empty()) {
            result.error_code = 3;
            result.error_message = "ASR transcription produced no text";
        }
    }
    // 短音频: 直接整段转录
    else {
        result = asr_plugin_->transcribe_memory(
            reinterpret_cast<const uint8_t*>(audio_data.data()),
            audio_data.size(), language, audio_filename, suppress_early_eos);
    }

    if (result.error_code != 0) {
        HttpResponse resp;
        resp.status_code = 500;
        resp.status_text = "Internal Server Error";
        resp.body = "{\"error\":{\"message\":\"" + json_escape(result.error_message) +
                    "\",\"type\":\"server_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 标点恢复
    if (punctuate && !result.text.empty()) {
        result.text_with_punc = punctuation_restorer_.restore(result.text);
    }

    // 强制对齐: 获取字级时间戳
    std::vector<asr::AlignedWord> aligned_words;
    if (want_word_timestamps && aligner_engine_.is_loaded() && !result.text.empty() && parsed_pcm) {
        // ForcedAligner 限制 ~180s, 超长音频按段对齐
        const int max_align_samples = wav_plain.sample_rate * 180;
        if ((int)wav_plain.samples.size() <= max_align_samples) {
            // 短音频: 整段对齐
            std::lock_guard<std::mutex> lock(aligner_mutex_);
            aligned_words = aligner_engine_.align(
                wav_plain.samples.data(), (int)wav_plain.samples.size(),
                wav_plain.sample_rate, result.text, "Chinese");
            fprintf(stderr, "[Serve] ForcedAligner: %zu words aligned from %.1fs audio\n",
                    aligned_words.size(), audio_duration_s);
        } else {
            // 长音频: 按 150s 段切分对齐 (留 30s 余量)
            const int seg_samples = wav_plain.sample_rate * 150;
            // 先按字分割文本, 然后按比例分配到音频段
            auto all_chars = asr::AlignerEngine::tokenize_for_align(result.text);
            int total_chars = (int)all_chars.size();
            int total_samples = (int)wav_plain.samples.size();
            int num_segs = (total_samples + seg_samples - 1) / seg_samples;
            int chars_per_seg = (total_chars + num_segs - 1) / num_segs;

            int char_offset = 0;
            for (int si = 0; si < num_segs && char_offset < total_chars; ++si) {
                int sample_start = si * seg_samples;
                int sample_end = std::min(sample_start + seg_samples, total_samples);
                int seg_chars = std::min(chars_per_seg, total_chars - char_offset);

                // 重建该段文本
                std::string seg_text;
                for (int ci = char_offset; ci < char_offset + seg_chars; ++ci)
                    seg_text += all_chars[ci];

                float offset_s = (float)sample_start / wav_plain.sample_rate;
                int offset_ms = (int)(offset_s * 1000);

                std::vector<asr::AlignedWord> seg_aligned;
                {
                    std::lock_guard<std::mutex> lock(aligner_mutex_);
                    seg_aligned = aligner_engine_.align(
                        wav_plain.samples.data() + sample_start,
                        sample_end - sample_start,
                        wav_plain.sample_rate, seg_text, "Chinese");
                }

                // 加上时间偏移
                for (auto& w : seg_aligned) {
                    w.start_ms += offset_ms;
                    w.end_ms += offset_ms;
                    aligned_words.push_back(w);
                }

                char_offset += seg_chars;
            }
            fprintf(stderr, "[Serve] ForcedAligner: %zu words aligned from %.1fs audio (%d segments)\n",
                    aligned_words.size(), audio_duration_s, num_segs);
        }

        // 填充 result.words
        result.words.resize(aligned_words.size());
        for (size_t i = 0; i < aligned_words.size(); ++i) {
            result.words[i].word = aligned_words[i].word;
            result.words[i].start_ms = aligned_words[i].start_ms;
            result.words[i].end_ms = aligned_words[i].end_ms;
            result.words[i].confidence = aligned_words[i].confidence;
        }
    }

    // 说话人识别 (对整段音频做识别, 无 VAD 分割)
    std::string speaker_name;
    int speaker_id = -1;
    float speaker_sim = 0;
    if (identify_spk && speaker_encoder_) {
        audio::AudioData wav;
        if (audio::load_audio_from_memory(
                reinterpret_cast<const uint8_t*>(audio_data.data()),
                audio_data.size(), wav, audio_filename)) {
            auto spk = identify_speaker(wav.samples.data(), (int)wav.samples.size(), wav.sample_rate, true);
            if (spk.speaker_id >= 0) {
                speaker_name = spk.name;
                speaker_id = spk.speaker_id;
                speaker_sim = spk.similarity;
            }
        }
    }

    // 构建响应
    HttpResponse resp;
    if (response_format == "text") {
        resp.content_type = "text/plain";
        resp.body = punctuate && !result.text_with_punc.empty() ?
                    result.text_with_punc : result.text;
    } else if (response_format == "verbose_json") {
        resp.body = "{\"task\":\"transcribe\",\"language\":\"" +
                    json_escape(result.language) +
                    "\",\"duration\":" + std::to_string(result.duration_s) +
                    ",\"text\":\"" + json_escape(result.text) + "\"";
        if (!result.text_with_punc.empty())
            resp.body += ",\"text_with_punc\":\"" + json_escape(result.text_with_punc) + "\"";
        if (speaker_id >= 0)
            resp.body += ",\"speaker\":\"" + json_escape(speaker_name) +
                         "\",\"speaker_id\":" + std::to_string(speaker_id) +
                         ",\"speaker_similarity\":" + std::to_string(speaker_sim);
        if (!result.words.empty()) {
            resp.body += ",\"words\":[";
            for (size_t i = 0; i < result.words.size(); ++i) {
                auto& w = result.words[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"word\":\"" + json_escape(w.word) +
                             "\",\"start\":" + std::to_string(w.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(w.end_ms / 1000.0f) + "}";
            }
            resp.body += "]";
        }
        resp.body += "}";
    } else {
        // 默认 json
        resp.body = "{\"text\":\"" + json_escape(
            punctuate && !result.text_with_punc.empty() ? result.text_with_punc : result.text
        ) + "\"";
        if (speaker_id >= 0)
            resp.body += ",\"speaker\":\"" + json_escape(speaker_name) + "\"";
        if (!result.words.empty()) {
            resp.body += ",\"words\":[";
            for (size_t i = 0; i < result.words.size(); ++i) {
                auto& w = result.words[i];
                if (i > 0) resp.body += ",";
                resp.body += "{\"word\":\"" + json_escape(w.word) +
                             "\",\"start\":" + std::to_string(w.start_ms / 1000.0f) +
                             ",\"end\":" + std::to_string(w.end_ms / 1000.0f) + "}";
            }
            resp.body += "]";
        }
        resp.body += "}";
    }

    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// POST /v1/audio/speech — TTS 文本转语音
// ============================================================================

void ServeApp::handle_audio_speech(const HttpRequest& req, int client_fd) {
    // 检查 TTS 插件是否已启用
    if (!tts_plugin_) {
        HttpResponse resp;
        resp.status_code = 501;
        resp.status_text = "Not Implemented";
        resp.body = "{\"error\":{\"message\":\"TTS plugin not configured\","
                    "\"type\":\"invalid_request_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    if (!tts_plugin_->is_available()) {
        HttpResponse resp;
        resp.status_code = 503;
        resp.status_text = "Service Unavailable";
        resp.body = "{\"error\":{\"message\":\"TTS executable not available\","
                    "\"type\":\"service_unavailable\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 解析 JSON body
    // {"model": "...", "input": "text to speak", "voice": "alloy", "speed": 1.0, "response_format": "wav"}
    std::string input_text, voice, format;
    float speed = 1.0f;

    // 简单 JSON 提取 (复用已有的 json_get_string / json_get_number)
    auto get_str = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        auto kpos = req.body.find(search);
        if (kpos == std::string::npos) return "";
        auto colon = req.body.find(':', kpos + search.size());
        if (colon == std::string::npos) return "";
        auto vstart = req.body.find('"', colon + 1);
        if (vstart == std::string::npos) return "";
        vstart++;
        std::string result;
        for (size_t i = vstart; i < req.body.size(); i++) {
            if (req.body[i] == '"' && (i == 0 || req.body[i-1] != '\\')) break;
            if (req.body[i] == '\\' && i + 1 < req.body.size()) {
                char next = req.body[i + 1];
                if (next == '"' || next == '\\') { result += next; i++; continue; }
                if (next == 'n') { result += '\n'; i++; continue; }
                if (next == 't') { result += '\t'; i++; continue; }
            }
            result += req.body[i];
        }
        return result;
    };

    auto get_num = [&](const std::string& key, float def) -> float {
        std::string search = "\"" + key + "\"";
        auto kpos = req.body.find(search);
        if (kpos == std::string::npos) return def;
        auto colon = req.body.find(':', kpos + search.size());
        if (colon == std::string::npos) return def;
        auto vstart = colon + 1;
        while (vstart < req.body.size() && req.body[vstart] == ' ') vstart++;
        try { return std::stof(req.body.substr(vstart)); }
        catch (...) { return def; }
    };

    input_text = get_str("input");
    voice = get_str("voice");
    format = get_str("response_format");
    speed = get_num("speed", 1.0f);
    std::string instruct = get_str("instruct");
    std::string language = get_str("language");

    if (input_text.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.status_text = "Bad Request";
        resp.body = "{\"error\":{\"message\":\"'input' field is required\","
                    "\"type\":\"invalid_request_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    fprintf(stderr, "[Serve] TTS: text=%zu chars, voice=%s, speed=%.1f, format=%s, instruct=%s, lang=%s\n",
            input_text.size(), voice.c_str(), speed, format.c_str(),
            instruct.empty() ? "(none)" : instruct.c_str(),
            language.empty() ? "(auto)" : language.c_str());

    // 调用 TTS 插件
    auto result = tts_plugin_->synthesize(input_text, voice, speed, format, instruct, language);

    if (result.error_code != 0) {
        HttpResponse resp;
        resp.status_code = 500;
        resp.status_text = "Internal Server Error";
        resp.body = "{\"error\":{\"message\":\"" + json_escape(result.error_message) +
                    "\",\"type\":\"server_error\"}}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // MIME type
    std::string content_type = "application/octet-stream";
    if (result.format == "wav")  content_type = "audio/wav";
    else if (result.format == "pcm")  content_type = "audio/pcm";

    // 返回二进制音频数据
    send_binary_response(client_fd, 200, content_type,
                         result.audio_data.data(), result.audio_data.size());
    close(client_fd);
}

// ============================================================================
// GET /v1/tts/info — 返回 TTS 模型信息 (模型类型, 可用音色, 采样率)
// ============================================================================

void ServeApp::handle_tts_info(const HttpRequest& /*req*/, int client_fd) {
    HttpResponse resp;

    if (!tts_plugin_ || !tts_plugin_->is_available()) {
        // TTS disabled, but still report ASR/speaker encoder status for standalone use
        resp.body = "{\"enabled\":false"
                    ",\"has_asr\":" + std::string((asr_plugin_ && asr_plugin_->is_available()) ? "true" : "false") +
                    ",\"has_speaker_encoder\":" + std::string(speaker_encoder_ ? "true" : "false") +
                    "}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    auto info = tts_plugin_->model_info();

    std::string voices_json = "[";
    for (size_t i = 0; i < info.available_voices.size(); i++) {
        if (i > 0) voices_json += ",";
        voices_json += "\"" + json_escape(info.available_voices[i]) + "\"";
    }
    voices_json += "]";

    std::string langs_json = "[";
    for (size_t i = 0; i < info.available_languages.size(); i++) {
        if (i > 0) langs_json += ",";
        langs_json += "\"" + json_escape(info.available_languages[i]) + "\"";
    }
    langs_json += "]";

    // speaker_dialects: {"eric":"sichuan_dialect","dylan":"beijing_dialect",...}
    std::string dialects_json = "{";
    bool first_d = true;
    for (const auto& [spk, dialect] : info.speaker_dialects) {
        if (!first_d) dialects_json += ",";
        first_d = false;
        dialects_json += "\"" + json_escape(spk) + "\":\"" + json_escape(dialect) + "\"";
    }
    dialects_json += "}";

    // clone_voices: registered voice clone names
    std::string clone_voices_json = "[";
    for (size_t i = 0; i < info.clone_voices.size(); i++) {
        if (i > 0) clone_voices_json += ",";
        clone_voices_json += "\"" + json_escape(info.clone_voices[i]) + "\"";
    }
    clone_voices_json += "]";

    resp.body = "{\"enabled\":true,"
                "\"model_type\":\"" + json_escape(info.model_type) + "\","
                "\"default_instruct\":\"" + json_escape(info.default_instruct) + "\","
                "\"sample_rate\":" + std::to_string(info.sample_rate) + ","
                "\"has_speaker_encoder\":" + (info.has_speaker_encoder ? "true" : "false") + ","
                "\"has_asr\":" + ((asr_plugin_ && asr_plugin_->is_available()) ? "true" : "false") + ","
                "\"available_voices\":" + voices_json + ","
                "\"available_languages\":" + langs_json + ","
                "\"clone_voices\":" + clone_voices_json + ","
                "\"speaker_dialects\":" + dialects_json + "}";
    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// POST /v1/voice_clone/register — 注册克隆音色 (multipart: file=audio, name=string)
// ============================================================================

void ServeApp::handle_voice_clone_register(const HttpRequest& req, int client_fd) {
    if (!tts_plugin_ || !tts_plugin_->is_available()) {
        HttpResponse resp;
        resp.status_code = 501;
        resp.body = "{\"error\":\"TTS plugin not configured\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    auto form = parse_multipart(req);

    std::string voice_name;
    auto it = form.fields.find("name");
    if (it != form.fields.end()) voice_name = it->second;

    if (voice_name.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"'name' field is required\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // Validate voice name (alphanumeric + underscore/hyphen)
    for (char c : voice_name) {
        if (!std::isalnum(c) && c != '_' && c != '-') {
            HttpResponse resp;
            resp.status_code = 400;
            resp.body = "{\"error\":\"voice name must contain only alphanumeric, underscore, or hyphen\"}";
            send_response(client_fd, resp);
            close(client_fd);
            return;
        }
    }

    // Find audio file
    const std::string* audio_data = nullptr;
    std::string audio_fn;
    for (const auto& f : form.files) {
        if (f.field_name == "file" || f.field_name == "audio") {
            audio_data = &f.data;
            audio_fn = f.filename;
            break;
        }
    }

    if (!audio_data || audio_data->empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"'file' or 'audio' field with audio data is required\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // Load WAV/PCM from uploaded data
    std::vector<float> pcm;
    int sample_rate = 0;

    // Try to parse as WAV
    if (audio_data->size() > 44 && audio_data->substr(0, 4) == "RIFF") {
        // WAV file: parse header
        const uint8_t* d = reinterpret_cast<const uint8_t*>(audio_data->data());
        uint16_t channels = *(uint16_t*)(d + 22);
        uint32_t sr = *(uint32_t*)(d + 24);
        uint16_t bits = *(uint16_t*)(d + 34);
        sample_rate = (int)sr;

        // Find data chunk
        size_t pos = 12;
        while (pos + 8 < audio_data->size()) {
            std::string chunk_id(audio_data->data() + pos, 4);
            uint32_t chunk_size = *(uint32_t*)(d + pos + 4);
            if (chunk_id == "data") {
                pos += 8;
                if (bits == 16) {
                    int num_samples = chunk_size / 2;
                    const int16_t* samples = reinterpret_cast<const int16_t*>(d + pos);
                    pcm.resize(num_samples / channels);
                    for (int i = 0; i < (int)pcm.size(); i++) {
                        pcm[i] = samples[i * channels] / 32768.0f;
                    }
                } else if (bits == 32) {
                    int num_samples = chunk_size / 4;
                    const float* samples = reinterpret_cast<const float*>(d + pos);
                    pcm.resize(num_samples / channels);
                    for (int i = 0; i < (int)pcm.size(); i++) {
                        pcm[i] = samples[i * channels];
                    }
                }
                break;
            }
            pos += 8 + chunk_size;
        }
    }

    // WAV 解析失败, 尝试 ffmpeg 转码
    if (pcm.empty() || sample_rate == 0) {
        audio::AudioData decoded;
        if (audio::load_audio_from_memory(
                reinterpret_cast<const uint8_t*>(audio_data->data()),
                audio_data->size(), decoded, audio_fn)) {
            pcm = std::move(decoded.samples);
            sample_rate = decoded.sample_rate;
        }
    }

    if (pcm.empty() || sample_rate == 0) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"Failed to parse audio file (supported: WAV/MP3/M4A/OGG/FLAC)\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    fprintf(stderr, "[Serve] Voice clone register: name=%s, samples=%zu, sr=%d\n",
            voice_name.c_str(), pcm.size(), sample_rate);

    bool ok = tts_plugin_->register_clone_voice(voice_name, pcm.data(), (int)pcm.size(), sample_rate);

    HttpResponse resp;
    if (ok) {
        resp.body = "{\"success\":true,\"voice\":\"" + json_escape(voice_name) + "\"}";
    } else {
        resp.status_code = 500;
        resp.body = "{\"error\":\"Failed to register voice (speaker encoder may not be available)\"}";
    }
    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// GET /v1/voice_clone/voices — 列出已注册的克隆音色
// ============================================================================

void ServeApp::handle_voice_clone_voices(const HttpRequest& /*req*/, int client_fd) {
    HttpResponse resp;
    if (!tts_plugin_ || !tts_plugin_->is_available()) {
        resp.body = "{\"voices\":[]}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    auto info = tts_plugin_->model_info();
    std::string json = "{\"voices\":[";
    for (size_t i = 0; i < info.clone_voices.size(); i++) {
        if (i > 0) json += ",";
        json += "\"" + json_escape(info.clone_voices[i]) + "\"";
    }
    json += "],\"has_speaker_encoder\":" +
            std::string(info.has_speaker_encoder ? "true" : "false") + "}";

    resp.body = json;
    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// POST /v1/voice_clone/delete — 删除已注册的克隆音色 {"name": "..."}
// ============================================================================

void ServeApp::handle_voice_clone_delete(const HttpRequest& req, int client_fd) {
    if (!tts_plugin_ || !tts_plugin_->is_available()) {
        HttpResponse resp;
        resp.status_code = 501;
        resp.body = "{\"error\":\"TTS plugin not configured\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // Parse name from JSON body
    std::string name;
    auto pos = req.body.find("\"name\"");
    if (pos != std::string::npos) {
        auto colon = req.body.find(':', pos);
        if (colon != std::string::npos) {
            auto q1 = req.body.find('"', colon + 1);
            if (q1 != std::string::npos) {
                auto q2 = req.body.find('"', q1 + 1);
                if (q2 != std::string::npos) {
                    name = req.body.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
    }

    if (name.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"'name' field is required\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    bool ok = tts_plugin_->delete_clone_voice(name);

    HttpResponse resp;
    if (ok) {
        resp.body = "{\"success\":true,\"deleted\":\"" + json_escape(name) + "\"}";
    } else {
        resp.status_code = 404;
        resp.body = "{\"error\":\"Voice '" + json_escape(name) + "' not found\"}";
    }
    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// Speaker Registration API — 说话人注册/识别/管理
// ============================================================================

// 80-dim Kaldi-style fbank 特征提取 (用于 CAM++ 说话人编码)
// FunASR CAM++ 期望 Kaldi fbank, NOT Whisper mel:
//   1. PCM × 32768 缩放 (FunASR forward_fbank convention)
//   2. pre-emphasis 0.97
//   3. Hann window, no center padding
//   4. power spectrum → mel filterbank (HTK, no Slaney normalization)
//   5. log(energy) (natural log, no Whisper global normalization)
void ServeApp::compute_mel_80(const float* samples, int num_samples, int sample_rate,
                               std::vector<float>& mel_out, int& num_frames) {
    const int n_fft = 400;       // 25ms window @ 16kHz
    const int fft_size = 512;    // Zero-pad to power of 2 (Kaldi default)
    const int hop = 160;         // 10ms @ 16kHz
    const int n_mels = 80;
    const int n_freqs = fft_size / 2 + 1;  // 257
    const float low_freq = 20.0f;   // Kaldi default
    const int target_sr = 16000;

    // Resample if needed
    std::vector<float> resampled_buf;
    const float* pcm = samples;
    int pcm_len = num_samples;
    if (sample_rate != target_sr) {
        std::vector<float> input_vec(samples, samples + num_samples);
        audio::resample(input_vec, sample_rate, resampled_buf, target_sr);
        pcm = resampled_buf.data();
        pcm_len = (int)resampled_buf.size();
    }

    // Scale by 32768 (FunASR convention: float [-1,1] → PCM16 range)
    std::vector<float> scaled(pcm_len);
    for (int i = 0; i < pcm_len; i++)
        scaled[i] = pcm[i] * 32768.0f;

    // Pre-emphasis (coefficient 0.97)
    for (int i = pcm_len - 1; i > 0; i--)
        scaled[i] -= 0.97f * scaled[i - 1];
    scaled[0] *= (1.0f - 0.97f);

    // Compute number of frames (Kaldi snip_edges=True: no center padding)
    num_frames = (pcm_len - n_fft) / hop + 1;
    if (num_frames <= 0) { mel_out.clear(); return; }

    // Build mel filterbank (HTK scale, low_freq=20Hz, using fft_size for bin resolution)
    // Cache filterbank across calls
    thread_local std::vector<float> mel_fb;
    thread_local std::vector<float> povey_win;
    thread_local bool fb_built = false;
    if (!fb_built) {
        // HTK mel filterbank with low_freq=20Hz
        mel_fb.resize((size_t)n_mels * n_freqs, 0.0f);
        auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

        float min_mel = hz_to_mel(low_freq);
        float max_mel = hz_to_mel((float)target_sr / 2.0f);
        std::vector<float> mel_points(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++)
            mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (n_mels + 1));

        for (int m = 0; m < n_mels; m++) {
            float left = mel_points[m] * fft_size / target_sr;
            float center = mel_points[m + 1] * fft_size / target_sr;
            float right = mel_points[m + 2] * fft_size / target_sr;
            for (int k = 0; k < n_freqs; k++) {
                float fk = (float)k;
                if (fk >= left && fk <= center)
                    mel_fb[m * n_freqs + k] = (fk - left) / (center - left);
                else if (fk > center && fk <= right)
                    mel_fb[m * n_freqs + k] = (right - fk) / (right - center);
            }
            // NO Slaney normalization (unlike Whisper filterbank)
        }

        // Povey window (Kaldi default: symmetric Hann^0.85)
        povey_win.resize(n_fft);
        for (int i = 0; i < n_fft; i++)
            povey_win[i] = std::pow(0.5f - 0.5f * std::cos(2.0f * (float)M_PI * i / (n_fft - 1)), 0.85f);

        fb_built = true;
    }

    // Precompute non-zero ranges for each mel bin
    std::vector<int> mel_start(n_mels, n_freqs);
    std::vector<int> mel_end(n_mels, 0);
    for (int m = 0; m < n_mels; m++) {
        for (int k = 0; k < n_freqs; k++) {
            if (mel_fb[m * n_freqs + k] != 0.0f) {
                if (k < mel_start[m]) mel_start[m] = k;
                if (k + 1 > mel_end[m]) mel_end[m] = k + 1;
            }
        }
    }

    // STFT → power spectrum → mel → log
    std::vector<float> mel_spec(n_mels * num_frames, 0.0f);
    std::vector<float> frame(fft_size, 0.0f);  // zero-padded to fft_size

    for (int t = 0; t < num_frames; t++) {
        // Window + zero-pad
        for (int i = 0; i < n_fft; i++)
            frame[i] = scaled[t * hop + i] * povey_win[i];
        for (int i = n_fft; i < fft_size; i++)
            frame[i] = 0.0f;

        // DFT (fft_size=512, cached twiddle factors)
        thread_local std::vector<float> tw_re, tw_im;
        thread_local bool tw_built = false;
        if (!tw_built) {
            tw_re.resize(n_freqs * fft_size);
            tw_im.resize(n_freqs * fft_size);
            for (int k = 0; k < n_freqs; k++) {
                for (int n = 0; n < fft_size; n++) {
                    double angle = -2.0 * M_PI * k * n / fft_size;
                    tw_re[k * fft_size + n] = (float)std::cos(angle);
                    tw_im[k * fft_size + n] = (float)std::sin(angle);
                }
            }
            tw_built = true;
        }

        // Power spectrum via DFT
        std::vector<float> power(n_freqs);
        for (int k = 0; k < n_freqs; k++) {
            float re = 0, im = 0;
            const float* tw_r = &tw_re[k * fft_size];
            const float* tw_i = &tw_im[k * fft_size];
            for (int n = 0; n < fft_size; n++) {
                re += frame[n] * tw_r[n];
                im += frame[n] * tw_i[n];
            }
            power[k] = re * re + im * im;
        }

        // Mel filterbank + log (natural log, no global normalization)
        for (int m = 0; m < n_mels; m++) {
            float sum = 0;
            for (int k = mel_start[m]; k < mel_end[m]; k++)
                sum += mel_fb[m * n_freqs + k] * power[k];
            mel_spec[m * num_frames + t] = std::log(std::max(sum, 1.175494e-38f));
        }
    }

    // Transpose [80, T] → [T, 80] for CAM++ extract()
    mel_out.resize(n_mels * num_frames);
    for (int t = 0; t < num_frames; t++)
        for (int f = 0; f < n_mels; f++)
            mel_out[t * n_mels + f] = mel_spec[f * num_frames + t];
}

// 说话人识别: 从 PCM 提取 embedding 并匹配
asr::SpeakerManager::MatchResult ServeApp::identify_speaker(
    const float* samples, int num_samples, int sample_rate, bool auto_register) {
    asr::SpeakerManager::MatchResult result;
    result.speaker_id = -1;
    result.name = "Unknown";

    if (!speaker_encoder_) return result;

    // 提取 80-dim Mel
    std::vector<float> mel;
    int num_frames = 0;
    compute_mel_80(samples, num_samples, sample_rate, mel, num_frames);

    if (num_frames < 10) return result;

    // CAM++ 提取 192-dim embedding
    std::lock_guard<std::mutex> lock(speaker_mutex_);
    auto embedding = speaker_encoder_->extract(mel.data(), num_frames);
    if (embedding.empty()) return result;

    // 与已注册说话人匹配
    result = speaker_manager_.identify(embedding, 0.65f, auto_register);

    return result;
}

// POST /v1/speakers/register — 注册说话人 (multipart: file=audio, name=string)
void ServeApp::handle_speaker_register(const HttpRequest& req, int client_fd) {
    if (!speaker_encoder_) {
        HttpResponse resp;
        resp.status_code = 501;
        resp.body = "{\"error\":\"Speaker encoder not loaded (CAM++ model not found)\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    auto form = parse_multipart(req);

    std::string audio_data;
    std::string audio_filename;
    for (auto& f : form.files) {
        if (f.field_name == "file") {
            audio_data = std::move(f.data);
            audio_filename = f.filename;
            break;
        }
    }

    std::string speaker_name = form.fields.count("name") ? form.fields["name"] : "";

    if (speaker_name.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"'name' field is required\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    if (audio_data.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"Audio file required. Use multipart/form-data with field 'file'\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 解析音频
    audio::AudioData wav;
    if (!audio::load_audio_from_memory(
            reinterpret_cast<const uint8_t*>(audio_data.data()),
            audio_data.size(), wav, audio_filename)) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"Failed to parse audio data\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // 提取 80-dim Mel 特征
    std::vector<float> mel;
    int num_frames = 0;
    compute_mel_80(wav.samples.data(), (int)wav.samples.size(), wav.sample_rate, mel, num_frames);

    if (num_frames < 10) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"Audio too short for speaker registration (min ~0.5s)\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    // CAM++ 提取 192-dim embedding
    std::lock_guard<std::mutex> lock(speaker_mutex_);
    auto embedding = speaker_encoder_->extract(mel.data(), num_frames);

    if (embedding.empty()) {
        HttpResponse resp;
        resp.status_code = 500;
        resp.body = "{\"error\":\"Failed to extract speaker embedding\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    speaker_manager_.register_speaker(speaker_name, embedding);

    fprintf(stderr, "[Speaker] Registered speaker '%s' (embedding_dim=%d, audio=%.1fs)\n",
            speaker_name.c_str(), (int)embedding.size(),
            (float)wav.samples.size() / wav.sample_rate);

    HttpResponse resp;
    resp.body = "{\"success\":true,\"name\":\"" + json_escape(speaker_name) +
                "\",\"embedding_dim\":" + std::to_string(embedding.size()) +
                ",\"total_speakers\":" + std::to_string(speaker_manager_.speaker_count()) + "}";
    send_response(client_fd, resp);
    close(client_fd);
}

// GET /v1/speakers — 列出已注册说话人
void ServeApp::handle_speaker_list(const HttpRequest& /*req*/, int client_fd) {
    std::lock_guard<std::mutex> lock(speaker_mutex_);

    auto names = speaker_manager_.speaker_names();

    HttpResponse resp;
    resp.body = "{\"speakers\":[";
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) resp.body += ",";
        resp.body += "\"" + json_escape(names[i]) + "\"";
    }
    resp.body += "],\"count\":" + std::to_string(names.size()) +
                 ",\"encoder\":\"" + (speaker_encoder_ ? "cam++" : "none") + "\"}";

    send_response(client_fd, resp);
    close(client_fd);
}

// POST /v1/speakers/delete — 删除说话人 {"name": "..."}
void ServeApp::handle_speaker_delete(const HttpRequest& req, int client_fd) {
    std::string name;
    auto pos = req.body.find("\"name\"");
    if (pos != std::string::npos) {
        auto colon = req.body.find(':', pos);
        if (colon != std::string::npos) {
            auto q1 = req.body.find('"', colon + 1);
            if (q1 != std::string::npos) {
                auto q2 = req.body.find('"', q1 + 1);
                if (q2 != std::string::npos) {
                    name = req.body.substr(q1 + 1, q2 - q1 - 1);
                }
            }
        }
    }

    if (name.empty()) {
        HttpResponse resp;
        resp.status_code = 400;
        resp.body = "{\"error\":\"'name' field is required\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    std::lock_guard<std::mutex> lock(speaker_mutex_);

    if (name == "all") {
        speaker_manager_.clear();
        HttpResponse resp;
        resp.body = "{\"success\":true,\"deleted\":\"all\"}";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    bool ok = speaker_manager_.remove_by_name(name);
    HttpResponse resp;
    if (ok) {
        resp.body = "{\"success\":true,\"deleted\":\"" + json_escape(name) + "\"}";
    } else {
        resp.status_code = 404;
        resp.body = "{\"error\":\"Speaker '" + json_escape(name) + "' not found\"}";
    }
    send_response(client_fd, resp);
    close(client_fd);
}

// ============================================================================
// 静态文件服务 (examples/ 目录)
// ============================================================================

void ServeApp::handle_static_file(const HttpRequest& req, int client_fd) {
    // Map "/" → "examples/index.html"
    // Map "/examples/..." → "examples/..."
    std::string file_path;
    if (req.path == "/") {
        file_path = "examples/index.html";
    } else {
        // Strip leading slash, sanitize
        file_path = req.path.substr(1);
    }

    // Security: reject path traversal
    if (file_path.find("..") != std::string::npos) {
        HttpResponse resp;
        resp.status_code = 403;
        resp.status_text = "Forbidden";
        resp.body = "Forbidden";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        HttpResponse resp;
        resp.status_code = 404;
        resp.status_text = "Not Found";
        resp.body = "File not found";
        send_response(client_fd, resp);
        close(client_fd);
        return;
    }

    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> content(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(content.data()), size);

    // Determine content type from extension
    std::string ct = "application/octet-stream";
    if (file_path.size() > 5 && file_path.substr(file_path.size() - 5) == ".html") ct = "text/html; charset=utf-8";
    else if (file_path.size() > 4 && file_path.substr(file_path.size() - 4) == ".css") ct = "text/css; charset=utf-8";
    else if (file_path.size() > 3 && file_path.substr(file_path.size() - 3) == ".js") ct = "application/javascript; charset=utf-8";
    else if (file_path.size() > 4 && file_path.substr(file_path.size() - 4) == ".svg") ct = "image/svg+xml";
    else if (file_path.size() > 4 && file_path.substr(file_path.size() - 4) == ".png") ct = "image/png";
    else if (file_path.size() > 5 && file_path.substr(file_path.size() - 5) == ".json") ct = "application/json";

    // HTML 文件禁止缓存，确保前端更新及时可见
    std::string extra_headers;
    if (ct.find("text/html") != std::string::npos) {
        extra_headers = "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                        "Pragma: no-cache\r\n";
    }
    send_binary_response(client_fd, 200, ct, content.data(), content.size(), extra_headers);
    close(client_fd);
}

// ============================================================================
// WebSocket /v1/voice — 语音对话 (ASR + LLM streaming + TTS)
// ============================================================================

void ServeApp::handle_websocket_voice(int client_fd, const HttpRequest& req) {
    // WebSocket 握手
    if (!ws_handshake(client_fd, req)) {
        fprintf(stderr, "[WS] Handshake failed fd=%d\n", client_fd);
        return;
    }
    fprintf(stderr, "[WS] Voice session started fd=%d\n", client_fd);

    // 设置 socket 超时，防止在死连接上阻塞
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 会话状态
    std::vector<std::pair<std::string, std::string>> chat_history;
    std::string voice = "serena";
    // VoiceDesign: 从 TTS config 初始化 base instruct, 保证音色描述始终生效
    std::string tts_instruct;
    if (tts_plugin_) {
        auto info = tts_plugin_->model_info();
        tts_instruct = info.default_instruct;
    }
    std::string tts_language;  // TTS 语言/方言 (空=auto)
    bool tts_enabled = true;

    // ---- 连接活跃性 + 生成控制 ----
    std::atomic<bool> conn_alive{true};
    std::atomic<bool> interrupted{false};
    std::atomic<bool> generating{false};
    std::mutex send_mutex;

    auto safe_send_text = [&](const std::string& text) -> bool {
        if (!conn_alive) return false;
        std::lock_guard<std::mutex> lock(send_mutex);
        if (!ws_send_text(client_fd, text)) {
            conn_alive = false;
            interrupted = true;
            return false;
        }
        return true;
    };
    auto safe_send_binary = [&](const uint8_t* data, size_t len) -> bool {
        if (!conn_alive) return false;
        std::lock_guard<std::mutex> lock(send_mutex);
        if (!ws_send_binary(client_fd, data, len)) {
            conn_alive = false;
            interrupted = true;
            return false;
        }
        return true;
    };

    // ---- 流式 ASR 状态 ----
    bool streaming_audio = false;
    std::vector<int16_t> pcm_buffer;
    int stream_sample_rate = 16000;
    constexpr float VAD_ENERGY_THRESHOLD = 0.01f;
    constexpr int VAD_SILENCE_MS = 800;
    constexpr int VAD_MIN_SPEECH_MS = 500;
    constexpr int VAD_MAX_DURATION_S = 30;
    constexpr float VAD_MIN_SPEECH_ENERGY = 0.008f;
    int silence_samples = 0;
    bool speech_detected = false;
    double total_energy_sum = 0;
    int total_speech_samples = 0;

    // 流式 ASR: 说话过程中定期识别, 发送中间结果 (参考 Qwen3-ASR 官方 streaming 方案)
    constexpr float STREAMING_ASR_CHUNK_S = 2.0f;  // 每 2 秒触发一次识别
    float streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
    size_t gen_audio_sample_count = 0;  // generating 期间的音频帧计数 (用于 audio.level 节流)

    // ASR→LLM 开关: 关闭时 ASR 仍运行但不触发 LLM 生成
    bool asr_to_llm = true;

    // ---- 服务端录音 (完整会话音频) ----
    std::vector<int16_t> recording_buffer;  // stream.start → stream.stop 全程录音
    int recording_sample_rate = 16000;
    std::chrono::steady_clock::time_point recording_start_time;

    // 保存录音到 WAV 文件
    auto save_recording_wav = [&]() -> std::string {
        if (recording_buffer.empty()) return "";
        // 创建 tmp/recordings/ 目录
        std::string dir = "tmp/recordings";
        std::filesystem::create_directories(dir);
        // 生成文件名: recording_YYYYMMDD_HHMMSS.wav
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        localtime_r(&t, &tm_buf);
        char fname[64];
        snprintf(fname, sizeof(fname), "recording_%04d%02d%02d_%02d%02d%02d.wav",
                 tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
                 tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);
        std::string path = dir + "/" + fname;
        // 写 WAV 文件
        int sr = recording_sample_rate;
        int data_bytes = (int)recording_buffer.size() * 2;
        std::ofstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "[WS] Failed to create recording file: %s\n", path.c_str());
            return "";
        }
        // WAV header (44 bytes)
        auto write16 = [&](uint16_t v) { f.write((char*)&v, 2); };
        auto write32 = [&](uint32_t v) { f.write((char*)&v, 4); };
        f.write("RIFF", 4);
        write32(36 + data_bytes);
        f.write("WAVE", 4);
        f.write("fmt ", 4);
        write32(16);         // chunk size
        write16(1);          // PCM
        write16(1);          // mono
        write32(sr);         // sample rate
        write32(sr * 2);     // byte rate
        write16(2);          // block align
        write16(16);         // bits per sample
        f.write("data", 4);
        write32(data_bytes);
        f.write((const char*)recording_buffer.data(), data_bytes);
        f.close();
        float dur_s = (float)recording_buffer.size() / sr;
        float size_mb = data_bytes / 1048576.0f;
        fprintf(stderr, "[WS] Recording saved: %s (%.1fs, %.1f MB, %d Hz)\n",
                path.c_str(), dur_s, size_mb, sr);
        return path;
    };

    // ---- Worker thread for ASR + LLM + TTS ----
    std::thread worker_thread;

    // 启动生成 (worker thread)
    auto start_generate = [&](std::string text, std::string voice_copy,
                               std::string instruct_copy, bool tts_copy,
                               std::string lang_copy) {
        if (generating || !conn_alive) return;
        if (worker_thread.joinable()) worker_thread.join();
        generating = true;
        interrupted = false;
        worker_thread = std::thread([&, text = std::move(text),
                                     voice_copy = std::move(voice_copy),
                                     instruct_copy = std::move(instruct_copy),
                                     lang_copy = std::move(lang_copy), tts_copy]() {
          try {
            ws_voice_generate(text, chat_history, voice_copy, instruct_copy, tts_copy,
                              safe_send_text, safe_send_binary, generating, interrupted, lang_copy);
          } catch (const std::exception& e) {
            fprintf(stderr, "[WS] EXCEPTION in generate worker: %s\n", e.what());
            safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating = false;
          } catch (...) {
            fprintf(stderr, "[WS] UNKNOWN EXCEPTION in generate worker\n");
            safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating = false;
          }
        });
    };

    // 启动语音输入 (ASR → LLM → TTS, 在 worker thread)
    auto start_voice_input = [&](std::vector<int16_t> audio, int sr,
                                  std::string voice_copy, std::string instruct_copy,
                                  bool tts_copy, std::string lang_copy) {
        if (generating || !conn_alive) return;
        if (worker_thread.joinable()) worker_thread.join();
        generating = true;
        interrupted = false;
        bool do_llm = asr_to_llm;  // capture current value
        worker_thread = std::thread([&, audio = std::move(audio), sr,
                                     voice_copy = std::move(voice_copy),
                                     instruct_copy = std::move(instruct_copy),
                                     lang_copy = std::move(lang_copy), tts_copy, do_llm]() {
          try {
            if (interrupted) { generating = false; return; }

            // ASR
            std::string asr_text;
            std::vector<float> float_pcm(audio.size());
            for (size_t i = 0; i < audio.size(); i++)
                float_pcm[i] = audio[i] / 32768.0f;

            if (asr_plugin_ && asr_plugin_->is_available()) {
                safe_send_text("{\"type\":\"status\",\"stage\":\"asr\"}");

                auto result = asr_plugin_->transcribe_pcm(
                    float_pcm.data(), (int)float_pcm.size(), sr, "auto", true);

                if (result.error_code == 0 && !result.text.empty()) {
                    int char_count = 0;
                    for (size_t i = 0; i < result.text.size(); ) {
                        unsigned char c = result.text[i];
                        int len = 1;
                        if (c >= 0xC0) len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
                        if (c > 0x20 && c != '.' && c != ',' && c != '!' && c != '?')
                            char_count++;
                        i += len;
                    }
                    if (char_count >= 2) {
                        asr_text = result.text;
                    } else {
                        fprintf(stderr, "[WS] ASR filtered (char_count=%d): \"%s\"\n",
                                char_count, result.text.c_str());
                    }
                } else {
                    fprintf(stderr, "[WS] ASR returned empty (error_code=%d, samples=%d)\n",
                            result.error_code, (int)audio.size());
                }
            }

            if (asr_text.empty()) {
                // 静默重置, 不向用户报错 (可能是噪声/回声触发了 VAD)
                fprintf(stderr, "[WS] No valid ASR result, silently resetting\n");
                generating = false;
                return;
            }

            // 说话人识别: 如有 CAM++ 编码器且有已注册说话人, 匹配身份
            std::string speaker_json;
            if (speaker_encoder_ && speaker_manager_.speaker_count() > 0) {
                auto spk = identify_speaker(float_pcm.data(), (int)float_pcm.size(), sr);
                if (spk.speaker_id >= 0 && spk.similarity >= 0.65f) {
                    speaker_json = ",\"speaker\":\"" + json_escape(spk.name) +
                                   "\",\"speaker_id\":" + std::to_string(spk.speaker_id) +
                                   ",\"speaker_similarity\":" + std::to_string(spk.similarity);
                    fprintf(stderr, "[WS] Speaker identified: %s (sim=%.3f)\n",
                            spk.name.c_str(), spk.similarity);
                }
            }

            safe_send_text("{\"type\":\"asr\",\"text\":\"" + json_escape(asr_text) + "\"" +
                           speaker_json + "}");
            if (interrupted) { generating = false; return; }

            // ASR→LLM 开关: 关闭时只做 ASR, 不触发 LLM 生成
            if (!do_llm) {
                safe_send_text("{\"type\":\"asr.done\"}");
                generating = false;
                return;
            }

            ws_voice_generate(asr_text, chat_history, voice_copy, instruct_copy, tts_copy,
                              safe_send_text, safe_send_binary, generating, interrupted, lang_copy);
          } catch (const std::exception& e) {
            fprintf(stderr, "[WS] EXCEPTION in voice worker: %s\n", e.what());
            safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating = false;
          } catch (...) {
            fprintf(stderr, "[WS] UNKNOWN EXCEPTION in voice worker\n");
            safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
            generating = false;
          }
        });
    };

    safe_send_text("{\"type\":\"session.created\"}");

    // ---- 主循环: poll-based, 保持响应性 ----
    auto last_activity = std::chrono::steady_clock::now();
    constexpr int WS_PING_INTERVAL_S = 15;  // 每 15 秒发一次 ping

    while (running_ && conn_alive) {
        struct pollfd pfd;
        pfd.fd = client_fd;
        pfd.events = POLLIN;
        int ret = ::poll(&pfd, 1, 100);

        if (ret < 0) break;
        if (ret == 0) {
            // 无数据: 检查是否需要发心跳 ping
            auto now = std::chrono::steady_clock::now();
            float idle_s = std::chrono::duration<float>(now - last_activity).count();
            if (idle_s >= WS_PING_INTERVAL_S) {
                std::lock_guard<std::mutex> lock(send_mutex);
                if (!ws_send_frame(client_fd, WS_OP_PING, nullptr, 0)) {
                    conn_alive = false;
                    break;
                }
                last_activity = now;
            }
            continue;
        }
        if (!(pfd.revents & POLLIN)) break;

        last_activity = std::chrono::steady_clock::now();

        uint8_t opcode;
        std::vector<uint8_t> payload;
        if (!ws_recv_frame(client_fd, opcode, payload)) {
            conn_alive = false;
            interrupted = true;
            break;
        }

        if (opcode == WS_OP_CLOSE) {
            std::lock_guard<std::mutex> lock(send_mutex);
            ws_send_frame(client_fd, WS_OP_CLOSE, nullptr, 0);
            conn_alive = false;
            interrupted = true;
            break;
        }
        if (opcode == WS_OP_PING) {
            std::lock_guard<std::mutex> lock(send_mutex);
            ws_send_frame(client_fd, WS_OP_PONG, payload.data(), payload.size());
            continue;
        }

        // ---- Binary frame: 流式 PCM 音频数据 ----
        // 始终接收音频 (即使 generating), 保持麦克风持续运行
        if (opcode == WS_OP_BINARY && streaming_audio) {
            size_t num_samples = payload.size() / 2;
            if (num_samples == 0) continue;

            const int16_t* samples = reinterpret_cast<const int16_t*>(payload.data());

            // 服务端录音: 始终累积全部音频 (不受 generating 影响)
            recording_buffer.insert(recording_buffer.end(), samples, samples + num_samples);

            // 计算当前帧能量 (用于 audio.level 显示)
            double energy_sum = 0;
            for (size_t i = 0; i < num_samples; i++) {
                float s = samples[i] / 32768.0f;
                energy_sum += s * s;
            }
            float rms = std::sqrt((float)(energy_sum / num_samples));

            // generating 期间: 只发 audio.level, 跳过 pcm_buffer 累积和 VAD
            // 防止 TTS 回声/环境噪声污染下一轮 ASR 输入
            if (generating) {
                // 发送音频电平 (每 100ms = 1600 samples @16kHz)
                size_t prev_count = gen_audio_sample_count;
                gen_audio_sample_count += num_samples;
                if (gen_audio_sample_count / 1600 > prev_count / 1600) {
                    char level_buf[64];
                    snprintf(level_buf, sizeof(level_buf),
                             "{\"type\":\"audio.level\",\"rms\":%.4f}", rms);
                    safe_send_text(level_buf);
                }
                continue;
            }
            gen_audio_sample_count = 0;  // generating 结束后重置

            size_t prev_size = pcm_buffer.size();
            pcm_buffer.insert(pcm_buffer.end(), samples, samples + num_samples);

            if (rms > VAD_ENERGY_THRESHOLD) {
                speech_detected = true;
                silence_samples = 0;
                total_speech_samples += (int)num_samples;
            } else {
                silence_samples += (int)num_samples;
            }
            total_energy_sum += energy_sum;

            if (pcm_buffer.size() / 1600 > prev_size / 1600) {
                char level_buf[64];
                snprintf(level_buf, sizeof(level_buf),
                         "{\"type\":\"audio.level\",\"rms\":%.4f}", rms);
                safe_send_text(level_buf);
            }

            float total_duration_s = (float)pcm_buffer.size() / stream_sample_rate;
            float silence_duration_ms = (float)silence_samples * 1000.0f / stream_sample_rate;

            // 流式 ASR: 说话过程中定期识别累积音频, 发送中间结果
            // generating 时跳过 (ASR 引擎 mutex 会阻塞事件循环)
            if (!generating && speech_detected && total_duration_s >= streaming_asr_next_s
                && asr_plugin_ && asr_plugin_->is_available()) {
                std::vector<float> float_pcm(pcm_buffer.size());
                for (size_t i = 0; i < pcm_buffer.size(); i++)
                    float_pcm[i] = pcm_buffer[i] / 32768.0f;

                auto partial = asr_plugin_->transcribe_pcm(
                    float_pcm.data(), (int)float_pcm.size(), stream_sample_rate, "auto", true);

                if (partial.error_code == 0 && !partial.text.empty()) {
                    fprintf(stderr, "[WS] Streaming ASR (%.1fs): \"%s\"\n",
                            total_duration_s, partial.text.substr(0, 80).c_str());
                    safe_send_text("{\"type\":\"asr.partial\",\"text\":\"" +
                                   json_escape(partial.text) + "\"}");
                }
                streaming_asr_next_s = total_duration_s + STREAMING_ASR_CHUNK_S;
            }

            bool vad_triggered = speech_detected &&
                                 silence_duration_ms >= VAD_SILENCE_MS &&
                                 total_duration_s >= (VAD_MIN_SPEECH_MS / 1000.0f);
            bool timeout = total_duration_s >= VAD_MAX_DURATION_S;

            if (vad_triggered || timeout) {
                // 不关闭 streaming_audio — 麦克风持续运行
                safe_send_text("{\"type\":\"stream.vad\"}");

                float avg_rms = (pcm_buffer.size() > 0)
                    ? std::sqrt((float)(total_energy_sum / pcm_buffer.size()))
                    : 0.0f;
                float speech_ratio = (float)total_speech_samples / std::max(1, (int)pcm_buffer.size());

                if (avg_rms < VAD_MIN_SPEECH_ENERGY) {
                    fprintf(stderr, "[WS] Rejected audio: avg_rms=%.4f speech_ratio=%.1f%% (too quiet)\n",
                            avg_rms, speech_ratio * 100);
                    // 重置 VAD 状态, 继续下一段
                    pcm_buffer.clear();
                    pcm_buffer.reserve(stream_sample_rate * 10);
                    silence_samples = 0;
                    speech_detected = false;
                    total_energy_sum = 0;
                    total_speech_samples = 0;
                    streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                    continue;
                }

                if (generating) {
                    // 上一轮还在生成, 丢弃当前段, 重置 VAD 等下一段
                    fprintf(stderr, "[WS] VAD during generation, dropping segment (%.1fs)\n",
                            (float)pcm_buffer.size() / stream_sample_rate);
                    pcm_buffer.clear();
                    pcm_buffer.reserve(stream_sample_rate * 10);
                    silence_samples = 0;
                    speech_detected = false;
                    total_energy_sum = 0;
                    total_speech_samples = 0;
                    streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                    continue;
                }

                // 去掉尾部静音
                int trim_samples = std::min(silence_samples, (int)pcm_buffer.size());
                if (trim_samples > stream_sample_rate / 10)
                    pcm_buffer.resize(pcm_buffer.size() - trim_samples + stream_sample_rate / 10);

                float audio_dur = (float)pcm_buffer.size() / stream_sample_rate;
                fprintf(stderr, "[WS] Stream VAD: %.1fs audio, %zu samples, avg_rms=%.4f speech=%.0f%%\n",
                        audio_dur, pcm_buffer.size(), avg_rms, speech_ratio * 100);

                // 启动 ASR + 生成 (worker thread)
                auto audio_copy = std::move(pcm_buffer);
                pcm_buffer.clear();
                pcm_buffer.reserve(stream_sample_rate * 10);
                silence_samples = 0;
                speech_detected = false;
                total_energy_sum = 0;
                total_speech_samples = 0;
                streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                start_voice_input(std::move(audio_copy), stream_sample_rate, voice, tts_instruct, tts_enabled, tts_language);
            }
            continue;
        }

        if (opcode != WS_OP_TEXT) continue;

        // 解析 JSON 事件
        std::string msg(payload.begin(), payload.end());
        std::string event_type = json_get_string(msg, "type");

        if (event_type == "config") {
            std::string v = json_get_string(msg, "voice");
            if (!v.empty()) voice = v;
            auto tts_pos = msg.find("\"tts\"");
            if (tts_pos != std::string::npos) {
                tts_enabled = json_get_bool(msg, "tts", true);
            }
            // VoiceDesign instruct (显式设置, 含空字符串=恢复默认)
            if (msg.find("\"tts_instruct\"") != std::string::npos) {
                std::string inst = json_get_string(msg, "tts_instruct");
                if (inst.empty() && tts_plugin_) {
                    tts_instruct = tts_plugin_->model_info().default_instruct;
                } else {
                    tts_instruct = inst;
                }
                fprintf(stderr, "[WS] tts_instruct updated: %s\n",
                        tts_instruct.empty() ? "(empty)" : tts_instruct.c_str());
            }
            // System prompt (per-session override)
            if (msg.find("\"system_prompt\"") != std::string::npos) {
                std::string sp = json_get_string(msg, "system_prompt");
                if (sp.empty()) {
                    // 恢复到 config 文件中的默认值
                    config_.voice_system_prompt = config_.voice_system_prompt_default;
                } else {
                    config_.voice_system_prompt = sp;
                }
                fprintf(stderr, "[WS] System prompt updated (%zu chars)\n",
                        config_.voice_system_prompt.size());
            }
            // Language/dialect
            std::string lang = json_get_string(msg, "tts_language");
            if (msg.find("\"tts_language\"") != std::string::npos) tts_language = lang;
            if (tts_plugin_ && msg.find("\"tts_temperature\"") != std::string::npos) {
                float tts_temp = (float)json_get_number(msg, "tts_temperature", 0.9);
                int tts_topk = json_get_int(msg, "tts_top_k", 50);
                float tts_topp = (float)json_get_number(msg, "tts_top_p", 1.0);
                float tts_rep = (float)json_get_number(msg, "tts_rep_penalty", 1.05);
                tts_plugin_->set_sampling(tts_temp, tts_topk, tts_topp, tts_rep);
            }
            // Voice chat limits (per-session override)
            if (msg.find("\"voice_max_turns\"") != std::string::npos) {
                int vmt = json_get_int(msg, "voice_max_turns", config_.voice_max_turns);
                if (vmt >= 1 && vmt <= 100) config_.voice_max_turns = vmt;
            }
            if (msg.find("\"voice_max_output_tokens\"") != std::string::npos) {
                int vmot = json_get_int(msg, "voice_max_output_tokens", config_.voice_max_output_tokens);
                if (vmot >= 10 && vmot <= 4096) config_.voice_max_output_tokens = vmot;
            }
            // ASR→LLM 开关
            if (msg.find("\"asr_to_llm\"") != std::string::npos) {
                asr_to_llm = json_get_bool(msg, "asr_to_llm", true);
            }
            fprintf(stderr, "[WS] config updated: voice=%s tts=%s lang=%s turns=%d tokens=%d asr_to_llm=%d fd=%d\n",
                    voice.empty() ? "(empty)" : voice.c_str(),
                    tts_enabled ? "on" : "off",
                    tts_language.empty() ? "auto" : tts_language.c_str(),
                    config_.voice_max_turns, config_.voice_max_output_tokens,
                    (int)asr_to_llm, client_fd);
            // 返回当前配置 (供 WebUI 初始化)
            {
                const std::string& sp = config_.voice_system_prompt.empty()
                    ? std::string(DEFAULT_VOICE_SYSTEM_PROMPT) : config_.voice_system_prompt;
                safe_send_text("{\"type\":\"config.updated\",\"system_prompt\":\"" + json_escape(sp) + "\""
                    ",\"voice_max_turns\":" + std::to_string(config_.voice_max_turns) +
                    ",\"voice_max_output_tokens\":" + std::to_string(config_.voice_max_output_tokens) + "}");
            }

        } else if (event_type == "chat") {
            if (generating) continue;
            std::string text = json_get_string(msg, "text");
            if (!text.empty()) {
                fprintf(stderr, "[WS] chat request: voice=%s tts=%s lang=%s chars=%zu fd=%d\n",
                        voice.empty() ? "(empty)" : voice.c_str(),
                        tts_enabled ? "on" : "off",
                        tts_language.empty() ? "auto" : tts_language.c_str(),
                        text.size(), client_fd);
                start_generate(std::move(text), voice, tts_instruct, tts_enabled, tts_language);
            }

        } else if (event_type == "stream.start") {
            // 始终允许开启流式录音 (即使 generating 中也接收音频)
            {
                streaming_audio = true;
                pcm_buffer.clear();
                pcm_buffer.reserve(stream_sample_rate * 10);
                silence_samples = 0;
                speech_detected = false;
                total_energy_sum = 0;
                total_speech_samples = 0;
                streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                stream_sample_rate = json_get_int(msg, "sample_rate", 16000);
                if (stream_sample_rate < 8000) stream_sample_rate = 8000;
                if (stream_sample_rate > 48000) stream_sample_rate = 48000;
                // 开始服务端录音
                recording_buffer.clear();
                recording_buffer.reserve(stream_sample_rate * 60);  // 预分配 1 分钟
                recording_sample_rate = stream_sample_rate;
                recording_start_time = std::chrono::steady_clock::now();
                safe_send_text("{\"type\":\"stream.started\"}");
                fprintf(stderr, "[WS] Audio stream started, rate=%d fd=%d\n",
                        stream_sample_rate, client_fd);
            }

        } else if (event_type == "stream.stop") {
            if (streaming_audio) {
                streaming_audio = false;

                float audio_dur = (float)pcm_buffer.size() / stream_sample_rate;
                if (pcm_buffer.size() >= (size_t)(stream_sample_rate * VAD_MIN_SPEECH_MS / 1000)) {
                    float avg_rms = std::sqrt((float)(total_energy_sum / std::max((size_t)1, pcm_buffer.size())));
                    if (avg_rms < VAD_MIN_SPEECH_ENERGY) {
                        fprintf(stderr, "[WS] Stream stopped: rejected (avg_rms=%.4f too quiet)\n", avg_rms);
                        safe_send_text("{\"type\":\"error\",\"message\":\"未检测到语音\"}");
                        pcm_buffer.clear();
                        silence_samples = 0;
                        speech_detected = false;
                        total_energy_sum = 0;
                        total_speech_samples = 0;
                        streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                        safe_send_text("{\"type\":\"stream.stopped\"}");
                        continue;
                    }

                    fprintf(stderr, "[WS] Stream stopped manually: %.1fs audio, avg_rms=%.4f\n", audio_dur, avg_rms);

                    // 启动 ASR + 生成 (worker thread)
                    auto audio_copy = std::move(pcm_buffer);
                    pcm_buffer.clear();
                    silence_samples = 0;
                    speech_detected = false;
                    total_energy_sum = 0;
                    total_speech_samples = 0;
                    streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                    start_voice_input(std::move(audio_copy), stream_sample_rate, voice, tts_instruct, tts_enabled, tts_language);
                } else {
                    fprintf(stderr, "[WS] Stream stopped, too short (%.1fs)\n", audio_dur);
                    safe_send_text("{\"type\":\"error\",\"message\":\"录音太短\"}");
                    pcm_buffer.clear();
                    silence_samples = 0;
                    speech_detected = false;
                    total_energy_sum = 0;
                    total_speech_samples = 0;
                    streaming_asr_next_s = STREAMING_ASR_CHUNK_S;
                }
            }
            // 保存服务端录音
            {
                std::string rec_path = save_recording_wav();
                if (!rec_path.empty()) {
                    safe_send_text("{\"type\":\"recording.saved\",\"path\":\"" +
                                   json_escape(rec_path) + "\"}");
                }
                recording_buffer.clear();
            }
            safe_send_text("{\"type\":\"stream.stopped\"}");

        } else if (event_type == "audio") {
            if (generating) continue;
            std::string audio_b64 = json_get_string(msg, "data");
            std::string format = json_get_string(msg, "format");
            if (audio_b64.empty()) continue;

            if (!asr_plugin_ || !asr_plugin_->is_available()) {
                safe_send_text("{\"type\":\"error\",\"message\":\"ASR not available\"}");
                continue;
            }

            // Base64 audio → worker thread for ASR + generate
            if (worker_thread.joinable()) worker_thread.join();
            generating = true;
            interrupted = false;
            worker_thread = std::thread([&, audio_b64 = std::move(audio_b64),
                                         voice_copy = voice,
                                         instruct_copy = tts_instruct,
                                         tts_copy = tts_enabled,
                                         lang_copy = tts_language]() {
              try {
                auto audio_bytes = base64_decode(audio_b64);
                if (audio_bytes.empty()) {
                    safe_send_text("{\"type\":\"error\",\"message\":\"Invalid audio data\"}");
                    generating = false;
                    return;
                }

                safe_send_text("{\"type\":\"status\",\"stage\":\"asr\"}");
                auto result = asr_plugin_->transcribe_memory(audio_bytes.data(),
                                                              audio_bytes.size(), "auto");

                if (result.error_code != 0 || result.text.empty()) {
                    safe_send_text("{\"type\":\"error\",\"message\":\"ASR failed: " +
                                   json_escape(result.error_message) + "\"}");
                    generating = false;
                    return;
                }

                safe_send_text("{\"type\":\"asr\",\"text\":\"" + json_escape(result.text) + "\"}");
                if (interrupted) { generating = false; return; }

                ws_voice_generate(result.text, chat_history, voice_copy, instruct_copy, tts_copy,
                                  safe_send_text, safe_send_binary, generating, interrupted, lang_copy);
              } catch (const std::exception& e) {
                fprintf(stderr, "[WS] EXCEPTION in audio worker: %s\n", e.what());
                safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
                generating = false;
              } catch (...) {
                fprintf(stderr, "[WS] UNKNOWN EXCEPTION in audio worker\n");
                safe_send_text("{\"type\":\"error\",\"message\":\"Internal error\"}");
                generating = false;
              }
            });

        } else if (event_type == "interrupt" || event_type == "tts.stop") {
            if (generating) {
                interrupted = true;
                fprintf(stderr, "[WS] Client interrupt fd=%d\n", client_fd);
            }

        } else if (event_type == "clear") {
            if (!generating) {
                chat_history.clear();
                safe_send_text("{\"type\":\"history.cleared\"}");
            }
        }
    }

    // 清理
    conn_alive = false;
    interrupted = true;
    // 连接断开时保存未保存的录音
    if (!recording_buffer.empty()) {
        std::string rec_path = save_recording_wav();
        if (!rec_path.empty()) {
            fprintf(stderr, "[WS] Recording saved on disconnect: %s\n", rec_path.c_str());
        }
        recording_buffer.clear();
    }
    if (worker_thread.joinable()) worker_thread.join();

    fprintf(stderr, "[WS] Voice session ended fd=%d\n", client_fd);
}

// 判断 UTF-8 字符是否是句末标点
static bool is_sentence_end_punct(const std::string& ch) {
    return ch == "。" || ch == "！" || ch == "？" ||
           ch == "." || ch == "!" || ch == "?" || ch == "\n";
}

void ServeApp::ws_voice_generate(const std::string& user_text,
                                  std::vector<std::pair<std::string, std::string>>& chat_history,
                                  const std::string& voice,
                                  const std::string& instruct,
                                  bool tts_enabled,
                                  const std::function<bool(const std::string&)>& send_text,
                                  const std::function<bool(const uint8_t*, size_t)>& send_binary,
                                  std::atomic<bool>& generating,
                                  std::atomic<bool>& interrupted,
                                  const std::string& language) {
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        send_text("{\"type\":\"error\",\"message\":\"Tokenizer not loaded\"}");
        generating = false;
        return;
    }

    chat_history.push_back({"user", user_text});

    // 保留最近 N 轮 (voice_max_turns, 每轮 2 条消息)
    const size_t max_messages = (size_t)config_.voice_max_turns * 2;
    while (chat_history.size() > max_messages) {
        chat_history.erase(chat_history.begin());
    }

    // 构建 messages
    std::vector<std::pair<std::string, std::string>> messages;
    const std::string& voice_prompt = config_.voice_system_prompt.empty()
        ? std::string(DEFAULT_VOICE_SYSTEM_PROMPT) : config_.voice_system_prompt;
    messages.push_back({"system", voice_prompt});
    for (auto& [role, content] : chat_history) {
        messages.push_back({role, content});
    }

    auto prompt_tokens = tok.apply_chat_template(messages, true, false);
    int prompt_count = (int)prompt_tokens.size();

    // Submit inference
    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens = config_.voice_max_output_tokens;
    infer_req.temperature    = 0.7f;
    infer_req.top_p          = 0.8f;
    infer_req.top_k          = 20;
    infer_req.presence_penalty = 1.5f;
    infer_req.frequency_penalty = 0.5f;
    infer_req.stream         = true;

    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        send_text("{\"type\":\"error\",\"message\":\"Request queue full\"}");
        generating = false;
        return;
    }

    bool do_stream_tts = tts_enabled && tts_plugin_ && tts_plugin_->is_available();

    send_text("{\"type\":\"llm.start\"}");

    // ---- TTS 生产者-消费者: LLM 和 TTS 并行执行 ----
    // 队列元素: (sentence_text, per_sentence_instruct)
    std::queue<std::pair<std::string, std::string>> tts_queue;
    std::mutex tts_mutex;
    std::condition_variable tts_cv;
    bool tts_done_flag = false;
    std::atomic<int> tts_segment_idx{0};

    // 通知客户端 TTS 输出格式
    if (do_stream_tts) {
        send_text("{\"type\":\"tts.stream_start\",\"sample_rate\":24000,\"format\":\"pcm16\"}");
    }

    // TTS 消费者线程: 每句独立 synthesize_streaming, 使用 per-sentence instruct
    auto* tts_raw = tts_plugin_.get();
    std::thread tts_thread;
    if (do_stream_tts) {
        tts_thread = std::thread([&, tts_raw]() {
            while (true) {
                std::string sentence;
                std::string sent_instruct;
                {
                    std::unique_lock<std::mutex> lock(tts_mutex);
                    tts_cv.wait(lock, [&]{ return !tts_queue.empty() || tts_done_flag; });
                    if (tts_queue.empty() && tts_done_flag) break;
                    if (tts_queue.empty()) continue;
                    auto& front = tts_queue.front();
                    sentence = std::move(front.first);
                    sent_instruct = std::move(front.second);
                    tts_queue.pop();
                }

                if (interrupted) break;

                // 合并 voice design base instruct 和 per-sentence emotion instruct
                // VoiceDesign: instruct="音色描述", sent_instruct="用X的语气说" → 合并
                // CustomVoice: instruct="" (空), sent_instruct="emotion" → 仅用 emotion
                std::string use_instruct;
                if (!sent_instruct.empty() && !instruct.empty()) {
                    use_instruct = instruct + "，" + sent_instruct;
                } else if (!sent_instruct.empty()) {
                    use_instruct = sent_instruct;
                } else {
                    use_instruct = instruct;
                }
                fprintf(stderr, "[TTS] Synthesize #%d [%s]: %.60s...\n",
                        tts_segment_idx.load() + 1,
                        use_instruct.empty() ? "default" : use_instruct.c_str(),
                        sentence.c_str());
                tts_raw->synthesize_streaming(sentence, voice, use_instruct,
                    [&](const float* data, int num_samples) -> bool {
                        if (interrupted) return false;
                        std::vector<int16_t> pcm16(num_samples);
                        for (int i = 0; i < num_samples; i++) {
                            float v = std::max(-1.0f, std::min(1.0f, data[i]));
                            pcm16[i] = (int16_t)(v * 32767.0f);
                        }
                        return send_binary(reinterpret_cast<const uint8_t*>(pcm16.data()),
                                           pcm16.size() * sizeof(int16_t));
                    }, 8, language);
                tts_segment_idx++;
            }
        });
    }

    // 推送句子到 TTS 队列 (自动解析 [情感标注] → TTS instruct, 后续句子继承前句情感)
    std::string last_emotion;  // 记住最近的情感标签, 防止多句输出时情绪跳变
    auto push_tts = [&](const std::string& sentence) {
        if (!do_stream_tts || sentence.empty()) return;
        auto [clean_text, emotion] = extract_tts_instruct(sentence);
        if (clean_text.empty()) return;
        // 有新情感则更新, 否则继承上一句
        if (!emotion.empty()) {
            last_emotion = emotion;
        }
        // Format emotion as proper TTS instruct (e.g., "温柔" → "用温柔的语气说")
        std::string formatted_instruct;
        if (!last_emotion.empty()) {
            formatted_instruct = "用" + last_emotion + "的语气说";
        }
        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_queue.push({std::move(clean_text), std::move(formatted_instruct)});
        }
        tts_cv.notify_one();
    };

    // ---- LLM 流式生成 ----
    std::string full_response;
    std::string pending_sentence;

    int comp_toks = poll_tokens(infer_req.request_id,
        [&](const std::string& piece) {
            full_response += piece;
            pending_sentence += piece;
            send_text("{\"type\":\"llm.delta\",\"delta\":\"" + json_escape(piece) + "\"}");

            if (!do_stream_tts) return;

            size_t pos = pending_sentence.size();
            if (pos == 0) return;

            size_t last_start = pos - 1;
            while (last_start > 0 && (pending_sentence[last_start] & 0xC0) == 0x80)
                last_start--;
            std::string last_ch = pending_sentence.substr(last_start);

            bool is_sentence_end = is_sentence_end_punct(last_ch) &&
                                   pending_sentence.size() >= 15;
            bool is_clause_break = (pending_sentence.size() > 100 &&
                                    (last_ch == "，" || last_ch == "," ||
                                     last_ch == "；" || last_ch == ";" ||
                                     last_ch == "：" || last_ch == ":"));

            if (is_sentence_end || is_clause_break) {
                std::string sentence = pending_sentence;
                while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
                    sentence.pop_back();
                if (!sentence.empty()) {
                    fprintf(stderr, "[LLM] Sentence split (%zu bytes): %.60s...\n",
                            sentence.size(), sentence.c_str());
                    push_tts(sentence);
                }
                pending_sentence.clear();
            }
        },
        config_.timeout_s,
        false, {}, {}, {}, nullptr, &interrupted, nullptr
    );

    // LLM 完成, 推送剩余文本
    if (!interrupted && !pending_sentence.empty()) {
        std::string sentence = pending_sentence;
        while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
            sentence.pop_back();
        if (!sentence.empty()) {
            fprintf(stderr, "[LLM] Final fragment (%zu bytes): %.60s...\n",
                    sentence.size(), sentence.c_str());
            push_tts(sentence);
        }
    }

    // 发送 LLM 完成
    if (!interrupted) {
        send_text("{\"type\":\"llm.done\",\"text\":\"" + json_escape(full_response) +
                     "\",\"prompt_tokens\":" + std::to_string(prompt_count) +
                     ",\"completion_tokens\":" + std::to_string(comp_toks) + "}");
    }

    chat_history.push_back({"assistant", full_response});

    // 通知 TTS 线程结束, 等待消费完所有排队句子
    if (do_stream_tts && tts_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_done_flag = true;
            tts_cv.notify_one();
        }
        tts_thread.join();

        if (!interrupted) {
            send_text("{\"type\":\"tts.done\",\"segments\":" +
                         std::to_string(tts_segment_idx.load()) + "}");
        }
    }

    generating = false;
}

// ============================================================================
// WebSocket /v1/realtime — 持续双向语音通道
//
// 协议:
//   Client→Server:
//     Binary:  PCM16LE 音频帧 (持续发送, 服务端做 VAD)
//     JSON:    {type:"session.update", voice:"...", sample_rate:N}
//     JSON:    {type:"text", text:"..."}     — 文本输入
//     JSON:    {type:"interrupt"}            — 中断当前生成
//
//   Server→Client:
//     JSON:    {type:"session.created"}
//     JSON:    {type:"input.speech_started"}
//     JSON:    {type:"input.speech_stopped"}
//     JSON:    {type:"input.transcription", text:"..."}
//     JSON:    {type:"response.started"}
//     JSON:    {type:"response.delta", delta:"..."}
//     JSON:    {type:"response.done", text:"..."}
//     JSON:    {type:"audio.started"}
//     Binary:  PCM16LE 24kHz mono 音频帧 (TTS, 分块发送)
//     JSON:    {type:"audio.done"}
//     JSON:    {type:"error", message:"..."}
// ============================================================================

void ServeApp::handle_websocket_realtime(int client_fd, const HttpRequest& req) {
    if (!ws_handshake(client_fd, req)) {
        fprintf(stderr, "[RT] Handshake failed fd=%d\n", client_fd);
        return;
    }
    fprintf(stderr, "[RT] Realtime session started fd=%d\n", client_fd);

    // ---- 会话配置 ----
    std::string voice = "serena";
    // VoiceDesign: 从 TTS config 初始化 base instruct
    std::string tts_instruct;
    if (tts_plugin_) {
        auto info = tts_plugin_->model_info();
        tts_instruct = info.default_instruct;
    }
    int client_sample_rate = 16000;
    std::vector<std::pair<std::string, std::string>> chat_history;

    // ---- 音频缓冲 + VAD ----
    std::vector<int16_t> pcm_buffer;
    constexpr float RT_VAD_ENERGY = 0.01f;
    constexpr int RT_VAD_SILENCE_MS = 600;           // 静音多久判定语音结束
    constexpr int RT_VAD_MIN_SPEECH_MS = 300;        // 最短语音
    constexpr int RT_VAD_MAX_S = 30;                 // 最长录音
    constexpr float RT_VAD_MIN_AVG_ENERGY = 0.008f;  // 整段最低平均 RMS
    int silence_samples = 0;
    bool speech_active = false;
    double total_energy_sum = 0;
    int total_samples_counted = 0;

    // 流式 ASR: 说话过程中定期识别, 发送中间结果
    constexpr float RT_STREAMING_ASR_CHUNK_S = 2.0f;
    float rt_streaming_asr_next_s = RT_STREAMING_ASR_CHUNK_S;

    // ---- 生成控制 ----
    std::atomic<bool> generating{false};
    std::atomic<bool> interrupted{false};
    std::atomic<bool> conn_alive{true};
    std::mutex send_mutex;

    // ASR→LLM 开关 (默认开启)
    bool rt_asr_to_llm = true;

    auto send_json = [&](const std::string& json) -> bool {
        if (!conn_alive) return false;
        std::lock_guard<std::mutex> lock(send_mutex);
        if (!ws_send_text(client_fd, json)) {
            conn_alive = false;
            interrupted = true;
            return false;
        }
        return true;
    };
    auto send_audio = [&](const uint8_t* data, size_t len) -> bool {
        if (!conn_alive) return false;
        std::lock_guard<std::mutex> lock(send_mutex);
        if (!ws_send_binary(client_fd, data, len)) {
            conn_alive = false;
            interrupted = true;
            return false;
        }
        return true;
    };

    // ---- 处理函数: ASR → LLM → TTS (在工作线程中运行) ----
    auto process_voice_input = [&](std::vector<int16_t> audio, int sr, bool do_llm) {
        if (interrupted) { generating = false; return; }

        // --- 转换 PCM ---
        std::vector<float> float_pcm(audio.size());
        for (size_t i = 0; i < audio.size(); i++)
            float_pcm[i] = audio[i] / 32768.0f;

        // --- ASR ---
        std::string asr_text;
        if (asr_plugin_ && asr_plugin_->is_available()) {
            auto result = asr_plugin_->transcribe_pcm(
                float_pcm.data(), (int)float_pcm.size(), sr, "auto", true);
            asr_text = result.text;
            // 去除前后空白
            while (!asr_text.empty() && (asr_text.front() == ' ' || asr_text.front() == '\n'))
                asr_text.erase(asr_text.begin());
            while (!asr_text.empty() && (asr_text.back() == ' ' || asr_text.back() == '\n'))
                asr_text.pop_back();
        }

        if (asr_text.empty()) {
            send_json("{\"type\":\"input.transcription\",\"text\":\"\"}");
            generating = false;
            return;
        }

        // 说话人识别
        std::string speaker_json;
        if (speaker_encoder_ && speaker_manager_.speaker_count() > 0) {
            auto spk = identify_speaker(float_pcm.data(), (int)float_pcm.size(), sr);
            if (spk.speaker_id >= 0 && spk.similarity >= 0.65f) {
                speaker_json = ",\"speaker\":\"" + json_escape(spk.name) +
                               "\",\"speaker_id\":" + std::to_string(spk.speaker_id) +
                               ",\"speaker_similarity\":" + std::to_string(spk.similarity);
            }
        }

        send_json("{\"type\":\"input.transcription\",\"text\":\"" + json_escape(asr_text) + "\"" +
                  speaker_json + "}");
        if (interrupted) { generating = false; return; }

        // ASR→LLM 开关: 关闭时只做 ASR, 不触发 LLM 生成
        if (!do_llm) {
            send_json("{\"type\":\"asr.done\"}");
            generating = false;
            return;
        }

        // --- LLM + TTS ---
        process_text_input(asr_text, chat_history, voice, tts_instruct,
                           send_json, send_audio, generating, interrupted);
    };

    auto process_text = [&](const std::string& text) {
        if (interrupted) { generating = false; return; }
        process_text_input(text, chat_history, voice, tts_instruct,
                           send_json, send_audio, generating, interrupted);
    };

    // 发送 session.created
    send_json("{\"type\":\"session.created\"}");

    // ---- 主循环: 使用 poll 超时避免阻塞 ----
    std::thread worker_thread;

    while (running_ && conn_alive) {
        // poll with 100ms timeout 以保持响应性
        struct pollfd pfd;
        pfd.fd = client_fd;
        pfd.events = POLLIN;
        int ret = ::poll(&pfd, 1, 100);

        if (ret < 0) break;  // error
        if (ret == 0) continue;  // timeout, 继续循环
        if (!(pfd.revents & POLLIN)) break;

        uint8_t opcode;
        std::vector<uint8_t> payload;
        if (!ws_recv_frame(client_fd, opcode, payload)) break;

        if (opcode == WS_OP_CLOSE) {
            std::lock_guard<std::mutex> lock(send_mutex);
            ws_send_frame(client_fd, WS_OP_CLOSE, nullptr, 0);
            break;
        }
        if (opcode == WS_OP_PING) {
            std::lock_guard<std::mutex> lock(send_mutex);
            ws_send_frame(client_fd, WS_OP_PONG, payload.data(), payload.size());
            continue;
        }

        // ---- Binary: 持续音频流 ----
        if (opcode == WS_OP_BINARY) {
            size_t num_samples = payload.size() / 2;
            if (num_samples == 0) continue;

            const int16_t* samples = reinterpret_cast<const int16_t*>(payload.data());

            // 计算 RMS 能量
            double energy_sum = 0;
            for (size_t i = 0; i < num_samples; i++) {
                float s = samples[i] / 32768.0f;
                energy_sum += s * s;
            }
            float rms = std::sqrt((float)(energy_sum / num_samples));

            // 正在生成时: 如果检测到用户说话 → 中断
            if (generating) {
                if (rms > RT_VAD_ENERGY * 3) {
                    interrupted = true;
                    // 等待工作线程结束
                    if (worker_thread.joinable()) worker_thread.join();
                    generating = false;
                    interrupted = false;
                    send_json("{\"type\":\"response.done\",\"text\":\"\",\"interrupted\":true}");
                    send_json("{\"type\":\"audio.done\"}");
                }
                continue;
            }

            // VAD 处理
            if (rms > RT_VAD_ENERGY) {
                if (!speech_active) {
                    speech_active = true;
                    pcm_buffer.clear();
                    total_energy_sum = 0;
                    total_samples_counted = 0;
                    silence_samples = 0;
                    rt_streaming_asr_next_s = RT_STREAMING_ASR_CHUNK_S;
                    send_json("{\"type\":\"input.speech_started\"}");
                }
                silence_samples = 0;
            } else if (speech_active) {
                silence_samples += (int)num_samples;
            }

            if (speech_active) {
                pcm_buffer.insert(pcm_buffer.end(), samples, samples + num_samples);
                total_energy_sum += energy_sum;
                total_samples_counted += (int)num_samples;

                // 流式 ASR: 定期识别累积音频, 发送中间结果
                float total_s = (float)pcm_buffer.size() / client_sample_rate;
                if (total_s >= rt_streaming_asr_next_s
                    && asr_plugin_ && asr_plugin_->is_available()) {
                    std::vector<float> float_pcm(pcm_buffer.size());
                    for (size_t i = 0; i < pcm_buffer.size(); i++)
                        float_pcm[i] = pcm_buffer[i] / 32768.0f;

                    auto partial = asr_plugin_->transcribe_pcm(
                        float_pcm.data(), (int)float_pcm.size(), client_sample_rate, "auto", true);

                    if (partial.error_code == 0 && !partial.text.empty()) {
                        fprintf(stderr, "[RT] Streaming ASR (%.1fs): \"%s\"\n",
                                total_s, partial.text.substr(0, 80).c_str());
                        send_json("{\"type\":\"input.transcription.partial\",\"text\":\"" +
                                   json_escape(partial.text) + "\"}");
                    }
                    rt_streaming_asr_next_s = total_s + RT_STREAMING_ASR_CHUNK_S;
                }

                // 检查静音超时 → 语音结束
                int silence_ms = silence_samples * 1000 / client_sample_rate;
                int total_ms = (int)pcm_buffer.size() * 1000 / client_sample_rate;

                bool speech_ended = (silence_ms >= RT_VAD_SILENCE_MS && total_ms > RT_VAD_MIN_SPEECH_MS);
                bool max_reached = (total_ms >= RT_VAD_MAX_S * 1000);

                if (speech_ended || max_reached) {
                    speech_active = false;
                    send_json("{\"type\":\"input.speech_stopped\"}");

                    // 检查平均能量
                    float avg_rms = total_samples_counted > 0
                        ? std::sqrt((float)(total_energy_sum / total_samples_counted)) : 0;

                    if (avg_rms < RT_VAD_MIN_AVG_ENERGY) {
                        // 太安静, 不触发 ASR
                        pcm_buffer.clear();
                        continue;
                    }

                    // 启动处理线程
                    if (worker_thread.joinable()) worker_thread.join();
                    generating = true;
                    interrupted = false;
                    auto audio_copy = std::move(pcm_buffer);
                    pcm_buffer.clear();
                    int sr = client_sample_rate;
                    worker_thread = std::thread([&, audio_copy, sr, do_llm = rt_asr_to_llm]() {
                      try {
                        process_voice_input(std::move(audio_copy), sr, do_llm);
                      } catch (const std::exception& e) {
                        fprintf(stderr, "[RT] EXCEPTION in voice worker: %s\n", e.what());
                        generating = false;
                      } catch (...) {
                        fprintf(stderr, "[RT] UNKNOWN EXCEPTION in voice worker\n");
                        generating = false;
                      }
                    });
                }
            }
            continue;
        }

        // ---- Text: JSON 控制消息 ----
        if (opcode == WS_OP_TEXT && !payload.empty()) {
            std::string text(payload.begin(), payload.end());

            // 简单 JSON 解析
            auto get_str = [&](const std::string& key) -> std::string {
                std::string search = "\"" + key + "\"";
                auto pos = text.find(search);
                if (pos == std::string::npos) return "";
                pos = text.find(':', pos + search.size());
                if (pos == std::string::npos) return "";
                pos = text.find('"', pos + 1);
                if (pos == std::string::npos) return "";
                auto end = text.find('"', pos + 1);
                if (end == std::string::npos) return "";
                return text.substr(pos + 1, end - pos - 1);
            };
            auto get_int = [&](const std::string& key) -> int {
                std::string search = "\"" + key + "\"";
                auto pos = text.find(search);
                if (pos == std::string::npos) return -1;
                pos = text.find(':', pos + search.size());
                if (pos == std::string::npos) return -1;
                pos++;
                while (pos < text.size() && text[pos] == ' ') pos++;
                return std::atoi(text.c_str() + pos);
            };

            std::string msg_type = get_str("type");

            if (msg_type == "session.update") {
                auto v = get_str("voice");
                if (!v.empty()) voice = v;
                auto inst = get_str("tts_instruct");
                if (!inst.empty()) tts_instruct = inst;
                int sr = get_int("sample_rate");
                if (sr > 0) client_sample_rate = sr;
                // ASR→LLM 开关
                if (text.find("\"asr_to_llm\"") != std::string::npos) {
                    rt_asr_to_llm = json_get_bool(text, "asr_to_llm", true);
                }
                fprintf(stderr, "[RT] Config: voice=%s instruct=%s sample_rate=%d asr_to_llm=%d\n",
                        voice.c_str(),
                        tts_instruct.empty() ? "(default)" : tts_instruct.c_str(),
                        client_sample_rate, (int)rt_asr_to_llm);
            }
            else if (msg_type == "text") {
                auto input_text = get_str("text");
                if (!input_text.empty() && !generating) {
                    if (worker_thread.joinable()) worker_thread.join();
                    generating = true;
                    interrupted = false;
                    worker_thread = std::thread([&, input_text]() {
                      try {
                        process_text(input_text);
                      } catch (const std::exception& e) {
                        fprintf(stderr, "[RT] EXCEPTION in text worker: %s\n", e.what());
                        generating = false;
                      } catch (...) {
                        fprintf(stderr, "[RT] UNKNOWN EXCEPTION in text worker\n");
                        generating = false;
                      }
                    });
                }
            }
            else if (msg_type == "interrupt") {
                if (generating) {
                    interrupted = true;
                }
            }
        }
    }

    // 清理
    conn_alive = false;
    interrupted = true;
    if (worker_thread.joinable()) worker_thread.join();

    fprintf(stderr, "[RT] Realtime session ended fd=%d\n", client_fd);
}

// ============================================================================
// process_text_input — LLM 生成 + TTS 合成 + 分块音频发送
// ============================================================================

void ServeApp::process_text_input(
    const std::string& user_text,
    std::vector<std::pair<std::string, std::string>>& chat_history,
    const std::string& voice,
    const std::string& instruct,
    const std::function<void(const std::string&)>& send_json,
    const std::function<void(const uint8_t*, size_t)>& send_audio,
    std::atomic<bool>& generating,
    std::atomic<bool>& interrupted)
{
    const auto& tok = backend_.tokenizer();
    if (!tok.is_loaded()) {
        send_json("{\"type\":\"error\",\"message\":\"Tokenizer not loaded\"}");
        generating = false;
        return;
    }

    chat_history.push_back({"user", user_text});
    const size_t max_messages = (size_t)config_.voice_max_turns * 2;
    while (chat_history.size() > max_messages) chat_history.erase(chat_history.begin());

    std::vector<std::pair<std::string, std::string>> messages;
    const std::string& voice_prompt = config_.voice_system_prompt.empty()
        ? std::string(DEFAULT_VOICE_SYSTEM_PROMPT) : config_.voice_system_prompt;
    messages.push_back({"system", voice_prompt});
    for (auto& [role, content] : chat_history) messages.push_back({role, content});

    auto prompt_tokens = tok.apply_chat_template(messages, true, false);

    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens = config_.voice_max_output_tokens;
    infer_req.temperature    = 0.7f;
    infer_req.top_p          = 0.8f;
    infer_req.top_k          = 20;
    infer_req.presence_penalty = 1.5f;
    infer_req.frequency_penalty = 0.5f;
    infer_req.stream         = true;

    register_request(infer_req.request_id);

    if (!backend_.submit(infer_req)) {
        unregister_request(infer_req.request_id);
        send_json("{\"type\":\"error\",\"message\":\"Request queue full\"}");
        generating = false;
        return;
    }

    bool do_tts = tts_plugin_ && tts_plugin_->is_available();

    send_json("{\"type\":\"response.started\"}");

    // ---- TTS 生产者-消费者 ----
    // 队列元素: (sentence_text, per_sentence_instruct)
    std::queue<std::pair<std::string, std::string>> tts_queue;
    std::mutex tts_mutex;
    std::condition_variable tts_cv;
    bool tts_done_flag = false;

    constexpr size_t AUDIO_CHUNK_SAMPLES = 4800;  // 200ms @ 24kHz

    int tts_sentence_count = 0;
    std::thread tts_thread;
    if (do_tts) {
        send_json("{\"type\":\"audio.started\"}");
        auto* tts_raw = tts_plugin_.get();
        tts_thread = std::thread([&, tts_raw]() {
            while (true) {
                if (interrupted) break;
                std::string sentence;
                std::string sent_instruct;
                {
                    std::unique_lock<std::mutex> lock(tts_mutex);
                    tts_cv.wait(lock, [&]{ return !tts_queue.empty() || tts_done_flag; });
                    if (tts_queue.empty() && tts_done_flag) break;
                    if (tts_queue.empty()) continue;
                    auto& front = tts_queue.front();
                    sentence = std::move(front.first);
                    sent_instruct = std::move(front.second);
                    tts_queue.pop();
                }
                if (interrupted) break;

                // 合并 base instruct 和 per-sentence emotion
                std::string use_instruct;
                if (!sent_instruct.empty() && !instruct.empty()) {
                    use_instruct = instruct + "，" + sent_instruct;
                } else if (!sent_instruct.empty()) {
                    use_instruct = sent_instruct;
                } else {
                    use_instruct = instruct;
                }
                tts_sentence_count++;
                fprintf(stderr, "[TTS] Synthesize #%d [%s]: %.60s...\n",
                        tts_sentence_count,
                        use_instruct.empty() ? "default" : use_instruct.c_str(),
                        sentence.c_str());
                tts_raw->synthesize_streaming(sentence, voice, use_instruct,
                    [&](const float* data, int num_samples) -> bool {
                        if (interrupted) return false;
                        std::vector<int16_t> pcm16(num_samples);
                        for (int i = 0; i < num_samples; i++) {
                            float v = std::max(-1.0f, std::min(1.0f, data[i]));
                            pcm16[i] = (int16_t)(v * 32767.0f);
                        }
                        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(pcm16.data());
                        size_t remaining = pcm16.size() * sizeof(int16_t);
                        const size_t chunk_bytes = AUDIO_CHUNK_SAMPLES * 2;
                        while (remaining > 0 && !interrupted) {
                            size_t send_size = std::min(remaining, chunk_bytes);
                            send_audio(ptr, send_size);
                            ptr += send_size;
                            remaining -= send_size;
                        }
                        return !interrupted;
                    }, 8);
            }
        });
    }

    std::string last_emotion;  // 记住最近的情感标签, 防止多句输出时情绪跳变
    auto push_tts = [&](const std::string& sentence) {
        if (!do_tts || sentence.empty()) return;
        // Strip markdown formatting (*, #, -, etc.) and whitespace
        std::string clean;
        clean.reserve(sentence.size());
        for (size_t i = 0; i < sentence.size(); i++) {
            char c = sentence[i];
            if (c == '*' || c == '#' || c == '`' || c == '$') continue;
            // Skip numbered list prefixes like "1. " "2. "
            if (c >= '0' && c <= '9' && i + 1 < sentence.size() && sentence[i+1] == '.') {
                i++;  // skip digit and dot
                if (i + 1 < sentence.size() && sentence[i+1] == ' ') i++;
                continue;
            }
            clean += c;
        }
        // Trim whitespace
        while (!clean.empty() && (clean.front() == ' ' || clean.front() == '\n'))
            clean.erase(clean.begin());
        while (!clean.empty() && (clean.back() == ' ' || clean.back() == '\n'))
            clean.pop_back();
        // Skip if too short (< 2 Chinese chars or < 6 bytes)
        if (clean.size() < 6) return;
        // 提取 [情感标注] → TTS instruct
        auto [text_part, emotion] = extract_tts_instruct(clean);
        if (text_part.size() < 6) return;
        // 有新情感则更新, 否则继承上一句
        if (!emotion.empty()) {
            last_emotion = emotion;
        }
        // Format emotion as proper TTS instruct (e.g., "温柔" → "用温柔的语气说")
        std::string formatted_instruct;
        if (!last_emotion.empty()) {
            formatted_instruct = "用" + last_emotion + "的语气说";
        }
        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_queue.push({std::move(text_part), std::move(formatted_instruct)});
            fprintf(stderr, "[TTS] push_tts: queue_size=%zu, text=%.40s...\n",
                    tts_queue.size(), clean.c_str());
        }
        tts_cv.notify_one();
    };

    // ---- LLM 流式生成 ----
    std::string full_response;
    std::string pending_sentence;

    poll_tokens(infer_req.request_id,
        [&](const std::string& piece) {
            if (interrupted) return;
            full_response += piece;
            pending_sentence += piece;

            send_json("{\"type\":\"response.delta\",\"delta\":\"" + json_escape(piece) + "\"}");

            if (!do_tts) return;

            // 句尾检测
            size_t pos = pending_sentence.size();
            if (pos == 0) return;
            size_t last_start = pos - 1;
            while (last_start > 0 && (pending_sentence[last_start] & 0xC0) == 0x80)
                last_start--;
            std::string last_ch = pending_sentence.substr(last_start);

            bool is_sent_end = is_sentence_end_punct(last_ch) && pending_sentence.size() >= 15;
            bool is_clause = (pending_sentence.size() > 100 &&
                             (last_ch == "，" || last_ch == "," ||
                              last_ch == "；" || last_ch == ";" ||
                              last_ch == "：" || last_ch == ":"));

            if (is_sent_end || is_clause) {
                std::string sentence = pending_sentence;
                while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
                    sentence.pop_back();
                if (!sentence.empty()) {
                    fprintf(stderr, "[LLM] Sentence split (%zu bytes): %.60s...\n",
                            sentence.size(), sentence.c_str());
                    push_tts(sentence);
                }
                pending_sentence.clear();
            }
        },
        config_.timeout_s,
        false, {}, {}, {}, nullptr, &interrupted, nullptr
    );

    // 推送剩余
    if (!pending_sentence.empty()) {
        std::string sentence = pending_sentence;
        while (!sentence.empty() && (sentence.back() == ' ' || sentence.back() == '\n'))
            sentence.pop_back();
        if (!sentence.empty()) {
            fprintf(stderr, "[LLM] Final fragment (%zu bytes): %.60s...\n",
                    sentence.size(), sentence.c_str());
            push_tts(sentence);
        }
    }

    // LLM 完成
    bool was_interrupted = interrupted.load();
    send_json("{\"type\":\"response.done\",\"text\":\"" + json_escape(full_response) +
              "\",\"interrupted\":" + (was_interrupted ? "true" : "false") + "}");
    chat_history.push_back({"assistant", full_response});

    // 等待 TTS 完成
    if (do_tts && tts_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(tts_mutex);
            tts_done_flag = true;
            tts_cv.notify_one();
        }
        tts_thread.join();
        send_json("{\"type\":\"audio.done\"}");
    }

    generating = false;
}

} // namespace serve
} // namespace qwen_thor
