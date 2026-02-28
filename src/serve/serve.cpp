// serve.cpp — HTTP API 服务实现
//
// 轻量级 POSIX socket HTTP 服务, 无外部依赖。
// 支持 OpenAI / Ollama 兼容 API 端点。

#include "serve.h"
#include "../engine/vision.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <thread>
#include <random>

// stb_image for decoding JPEG/PNG
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "../../third_party/stb/stb_image.h"

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
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// Minimal JSON value extraction (no full parser, just enough for API)
static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    auto end = pos + 1;
    while (end < json.size() && !(json[end] == '"' && json[end-1] != '\\')) end++;
    return json.substr(pos + 1, end - pos - 1);
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

    // Extract individual tool objects from the array and put each on its own line
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
static ImageData decode_image_base64(const std::string& data_uri) {
    ImageData result;
    std::string b64;

    // Strip "data:image/...;base64," prefix if present
    auto comma_pos = data_uri.find(',');
    if (comma_pos != std::string::npos) {
        b64 = data_uri.substr(comma_pos + 1);
    } else {
        b64 = data_uri;
    }

    auto raw = base64_decode(b64);
    if (raw.empty()) return result;

    int w, h, channels;
    uint8_t* img = stbi_load_from_memory(raw.data(), (int)raw.size(), &w, &h, &channels, 3);
    if (!img) {
        std::cerr << "[Serve] Failed to decode image: " << stbi_failure_reason() << std::endl;
        return result;
    }

    result.width = w;
    result.height = h;
    result.pixels.assign(img, img + w * h * 3);
    stbi_image_free(img);
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

            if (type == "text") {
                std::string text = json_get_string(obj, "text");
                result.text += text;
            } else if (type == "image_url") {
                // Extract nested image_url object
                auto iu_pos = obj.find("\"image_url\"");
                if (iu_pos != std::string::npos) {
                    size_t fp = obj.find('{', iu_pos + 11);
                    if (fp != std::string::npos) {
                        std::string iu_obj = extract_json_object(obj, fp);
                        std::string url = json_get_string(iu_obj, "url");
                        if (!url.empty()) {
                            auto img = decode_image_base64(url);
                            if (img.width > 0) {
                                result.images.push_back(std::move(img));
                                // Insert image placeholder text
                                // The actual tokens will be handled during tokenization
                                result.text += "<|vision_start|><|image_pad|><|vision_end|>";
                            }
                        }
                    }
                }
            } else if (type == "video") {
                // Parse video as array of base64-encoded frames:
                //   {"type":"video", "video":["base64_frame1","base64_frame2",...], "fps":24}
                float fps = (float)json_get_number(obj, "fps", 24.0);
                auto vid_pos = obj.find("\"video\"");
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
            } else if (type == "video_url") {
                // Parse video_url: {"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}
                // Decode video file, extract frames with ffmpeg, build VideoData
                std::cout << "[Serve] video_url type detected, obj.size()=" << obj.size() << std::endl;
                auto vu_pos = obj.find("\"video_url\"");
                if (vu_pos != std::string::npos) {
                    size_t fp = obj.find('{', vu_pos + 11);
                    if (fp != std::string::npos) {
                        std::string vu_obj = extract_json_object(obj, fp);
                        std::string url = json_get_string(vu_obj, "url");
                        std::cout << "[Serve] url.size()=" << url.size() << std::endl;
                        if (!url.empty()) {
                            // Strip data URI prefix to get raw base64
                            std::string b64_data;
                            auto comma = url.find(',');
                            if (comma != std::string::npos)
                                b64_data = url.substr(comma + 1);
                            else
                                b64_data = url;

                            std::cout << "[Serve] b64_data.size()=" << b64_data.size() << std::endl;
                            auto video_bytes = base64_decode(b64_data);
                            std::cout << "[Serve] video_bytes.size()=" << video_bytes.size() << std::endl;
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
                                system(mkdir_cmd.c_str());
                                std::string ffmpeg_cmd = "ffmpeg -y -i " + tmp_video
                                    + " -vf fps=2 -frames:v 16 -q:v 2 " + tmp_dir + "/frame_%04d.jpg";
                                std::cout << "[Serve] Running: " << ffmpeg_cmd << std::endl;
                                int ffret = system(ffmpeg_cmd.c_str());
                                std::cout << "[Serve] ffmpeg returned: " << ffret << std::endl;

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
                                system(cleanup.c_str());

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
                                    std::cout << "[Serve] video_url: " << vd.frames.size()
                                              << " frames extracted, " << vd.width << "x" << vd.height
                                              << " grid=" << grid_t << "x" << grid_h << "x" << grid_w
                                              << std::endl;
                                    result.videos.push_back(std::move(vd));
                                }
                            }
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
static void expand_image_placeholders(std::vector<int>& tokens,
                                       const std::vector<ImageData>& images,
                                       const Tokenizer& tokenizer) {
    // Token IDs for vision special tokens
    int vision_start_id = -1, image_pad_id = -1, vision_end_id = -1;
    // Look up from tokenizer's special tokens
    // These are standard Qwen3.5 token IDs
    vision_start_id = 248053;
    image_pad_id    = 248056;
    vision_end_id   = 248054;

    core::VisionConfig vcfg;
    int img_idx = 0;

    std::vector<int> expanded;
    expanded.reserve(tokens.size() + images.size() * 256);

    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == vision_start_id && img_idx < (int)images.size()) {
            expanded.push_back(vision_start_id);

            // Find the matching vision_end and count existing image_pad tokens
            // Then replace with the correct number
            size_t j = i + 1;
            while (j < tokens.size() && tokens[j] == image_pad_id) j++;
            // Skip past vision_end
            if (j < tokens.size() && tokens[j] == vision_end_id) j++;

            // Calculate actual number of output tokens for this image
            auto& img = images[img_idx];
            core::ImageInput input;
            input.width = img.width;
            input.height = img.height;
            input.pixels = img.pixels; // We need full pixels for preprocess
            auto processed = core::VisionEncoder::preprocess_image(input, vcfg);
            int n_tokens = processed.num_output_tokens();

            // Insert n_tokens copies of image_pad
            for (int k = 0; k < n_tokens; k++) {
                expanded.push_back(image_pad_id);
            }
            expanded.push_back(vision_end_id);

            i = j - 1;  // -1 because loop will increment
            img_idx++;
        } else {
            expanded.push_back(tokens[i]);
        }
    }

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
    static const char hex[] = "0123456789abcdef";
    char buf[13]; // 12 hex chars + null
    uint64_t val = rng();
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
    }
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
    printf("└─────────────────────────────────────────────┘\n");
}

// ============================================================================
// ServeApp
// ============================================================================

ServeApp::ServeApp(const ServeConfig& config, InferenceBackend& backend)
    : config_(config), backend_(backend) {}

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
                  << ": " << strerror(errno) << std::endl;
        close(ollama_fd_);
        ollama_fd_ = -1;
        return;
    }

    if (listen(ollama_fd_, config_.max_conns) < 0) {
        std::cerr << "[Serve] listen() failed for Ollama port: " << strerror(errno) << std::endl;
        close(ollama_fd_);
        ollama_fd_ = -1;
        return;
    }

    // 创建 OpenAI 端口 socket
    openai_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (openai_fd_ < 0) {
        std::cerr << "[Serve] socket() failed for OpenAI port: " << strerror(errno) << std::endl;
        close(ollama_fd_);
        ollama_fd_ = -1;
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
        close(ollama_fd_);
        ollama_fd_ = -1;
        return;
    }

    if (listen(openai_fd_, config_.max_conns) < 0) {
        std::cerr << "[Serve] listen() failed for OpenAI port: " << strerror(errno) << std::endl;
        close(openai_fd_);
        openai_fd_ = -1;
        close(ollama_fd_);
        ollama_fd_ = -1;
        return;
    }

    running_ = true;
    config_.print();
    printf("\n[Serve] Ollama API on http://%s:%d\n", config_.host.c_str(), config_.ollama_port);
    printf("  POST /api/generate          — Ollama Generate API\n");
    printf("  POST /api/chat              — Ollama Chat API\n");
    printf("  POST /api/show              — Model information\n");
    printf("  GET  /api/tags              — List local models\n");
    printf("  GET  /api/ps                — List running models\n");
    printf("  GET  /api/version           — Version info\n");
    printf("  GET  /health                — Health check\n");
    printf("\n[Serve] OpenAI API on http://%s:%d\n", config_.host.c_str(), config_.openai_port);
    printf("  POST /v1/chat/completions   — OpenAI Chat API\n");
    printf("  POST /v1/completions        — OpenAI Completions API\n");
    printf("  GET  /v1/models             — Model list\n");
    printf("  GET  /v1/models/{model}     — Retrieve model\n");
    printf("  GET  /health                — Health check\n\n");

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
            std::thread([this, client_fd, protocol]() {
                handle_connection(client_fd, protocol);
                active_workers_.fetch_sub(1);
            }).detach();
        }
    }
}

void ServeApp::handle_connection(int client_fd, int protocol) {
    auto req = parse_request(client_fd);
    std::cout << "[Serve] " << req.method << " " << req.path
              << " body=" << req.body.size() << " bytes" << std::endl;

    // CORS preflight 两个端口都处理
    if (req.method == "OPTIONS") {
        handle_cors_preflight(req, client_fd);
        close(client_fd);
        return;
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
        // OpenAI 端口: 只接受 /v1/* 路由
        if (req.path == "/v1/models" && req.method == "GET") {
            handle_models(req, client_fd);
        } else if (req.path.rfind("/v1/models/", 0) == 0 && req.method == "GET") {
            // GET /v1/models/{model} — retrieve individual model
            handle_model_retrieve(req, client_fd);
        } else if (req.path == "/v1/chat/completions" && req.method == "POST") {
            handle_openai_chat(req, client_fd);
        } else if (req.path == "/v1/completions" && req.method == "POST") {
            handle_openai_completions(req, client_fd);
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

    // 读取 HTTP 头部
    std::string raw;
    char buf[4096];
    while (true) {
        ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        raw.append(buf, n);
        if (raw.find("\r\n\r\n") != std::string::npos) break;
    }

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

        // Body
        req.body = raw.substr(header_end + 4);

        // 如果 Content-Length 指示更长的 body, 继续读取
        auto cl_it = req.headers.find("content-length");
        if (cl_it != req.headers.end()) {
            size_t content_len = std::stoull(cl_it->second);
            while (req.body.size() < content_len) {
                ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
                if (n <= 0) break;
                req.body.append(buf, n);
            }
        }
    }

    return req;
}

void ServeApp::send_response(int client_fd, const HttpResponse& resp) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status_code << " " << resp.status_text << "\r\n";
    oss << "Content-Type: " << resp.content_type << "\r\n";
    oss << "Access-Control-Allow-Origin: *\r\n";
    oss << "Content-Length: " << resp.body.size() << "\r\n";
    oss << "\r\n";
    oss << resp.body;

    auto str = oss.str();
    send(client_fd, str.c_str(), str.size(), 0);
}

void ServeApp::send_sse_event(int client_fd, const std::string& data) {
    std::string event = "data: " + data + "\n\n";
    send(client_fd, event.c_str(), event.size(), MSG_NOSIGNAL);
}

void ServeApp::send_sse_done(int client_fd) {
    std::string done = "data: [DONE]\n\n";
    send(client_fd, done.c_str(), done.size(), MSG_NOSIGNAL);
}

void ServeApp::send_ndjson_chunk(int client_fd, const std::string& json_line) {
    std::string data = json_line + "\n";
    std::ostringstream oss;
    oss << std::hex << data.size() << "\r\n" << data << "\r\n";
    auto chunk = oss.str();
    send(client_fd, chunk.c_str(), chunk.size(), MSG_NOSIGNAL);
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
                          std::string* out_finish_reason) {
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
        InferResponse resp;
        if (poll_request(request_id, resp, 100)) {
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
    resp.body = "{\"object\":\"list\",\"data\":[{\"id\":\"" + config_.model_name +
                "\",\"object\":\"model\",\"created\":" + std::to_string(now) +
                ",\"owned_by\":\"local\""
                ",\"capabilities\":{\"reasoning\":true,\"tool_calling\":true,\"multimodal\":true}"
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
    if (decoded != config_.model_name && decoded != config_.model_name + ":latest") {
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
                ",\"capabilities\":{\"reasoning\":true,\"tool_calling\":true,\"multimodal\":true}"
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
        "\"capabilities\":[\"completion\",\"vision\",\"tools\",\"thinking\"]"
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
        "\"capabilities\":[\"completion\",\"vision\",\"tools\",\"thinking\"],"
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
    std::cout << "[Serve] handle_openai_chat: body=" << req.body.size() << std::endl;
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
    std::cout << "[Serve] calling parse_messages..." << std::endl;
    auto messages = parse_messages(req.body, &images, &videos);
    std::cout << "[Serve] parse_messages done: " << messages.size() << " msgs, "
              << images.size() << " imgs, " << videos.size() << " vids" << std::endl;
    bool stream = json_get_bool(req.body, "stream", false);
    int max_tokens = json_get_int(req.body, "max_tokens", 4096);
    // 支持 max_completion_tokens (新标准命名)
    if (req.body.find("\"max_completion_tokens\"") != std::string::npos)
        max_tokens = json_get_int(req.body, "max_completion_tokens", max_tokens);

    // ---- Thinking mode 控制 ----
    // 支持多种参数名: "think", "enable_thinking" (vLLM/SGLang chat_template_kwargs)
    // 以及 OpenAI 标准 "reasoning_effort" (none=禁用, low/medium/high=启用)
    bool enable_thinking = true;
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
                {"system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tool_prompt});
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
    float presence_penalty  = (float)json_get_number(req.body, "presence_penalty", 0.0);
    float repeat_penalty    = (float)json_get_number(req.body, "repeat_penalty", 1.0);  // 非标准但广泛支持
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

    // Submit inference request
    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens = max_tokens;
    infer_req.temperature    = temperature;
    infer_req.top_p          = top_p;
    infer_req.top_k          = top_k;
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
                             "Content-Type: text/event-stream\r\n"
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

        int comp_toks = poll_tokens(infer_req.request_id,
            // on_token (content)
            [&](const std::string& piece) {
                send_sse_event(client_fd, make_chat_chunk(model, piece, "", req_id, now_t));
            },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            // on_reasoning
            [&](const std::string& piece) {
                send_sse_event(client_fd, make_chat_reasoning_chunk(model, piece, req_id, now_t));
            },
            // on_tool_call
            [&](const ToolCallInfo& tc) {
                send_sse_event(client_fd, make_chat_tool_call_chunk(model, tc, tool_call_idx++, req_id, now_t));
            },
            &finish_reason
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
                ",\"total_tokens\":" + std::to_string(total) + "}}";
            send_sse_event(client_fd, usage_chunk);
        }

        send_sse_done(client_fd);
    } else {
        // Non-streaming: collect all tokens, reasoning, and tool calls
        std::string content;
        std::string reasoning;
        std::vector<ToolCallInfo> tool_calls;
        std::string finish_reason;

        int comp_toks = poll_tokens(infer_req.request_id,
            [&](const std::string& piece) { content += piece; },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            [&](const std::string& piece) { reasoning += piece; },
            [&](const ToolCallInfo& tc) { tool_calls.push_back(tc); },
            &finish_reason
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
            ",\"total_tokens\":" + std::to_string(total) + "}}";
        send_response(client_fd, http_resp);
    }
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
    float temperature = (float)json_get_number(req.body, "temperature", 1.0);
    float top_p = (float)json_get_number(req.body, "top_p", 0.95);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty = (float)json_get_number(req.body, "presence_penalty", 0.0);
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
                             "Content-Type: text/event-stream\r\n"
                             "Cache-Control: no-cache\r\n"
                             "Connection: keep-alive\r\n"
                             "Access-Control-Allow-Origin: *\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto now_t = (int64_t)std::time(nullptr);
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            send_sse_event(client_fd, make_completion_chunk(model, piece, "", req_id, now_t));
        }, config_.timeout_s, enable_thinking, stop_seqs);

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
            ",\"total_tokens\":" + std::to_string(total) + "}}";
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
    bool enable_thinking = json_get_bool(req.body, "think", true);
    float def_temp = enable_thinking ? 1.0f : 0.7f;
    float def_top_p = enable_thinking ? 0.95f : 0.8f;
    float temperature = (float)json_get_number(req.body, "temperature", def_temp);
    float top_p = (float)json_get_number(req.body, "top_p", def_top_p);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty = (float)json_get_number(req.body, "presence_penalty", 0.0);
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

    // Tokenize: if system provided, use chat template; otherwise raw encode
    std::vector<int> prompt_tokens;
    if (!system.empty() || !images.empty()) {
        // Use chat template for image support (need special tokens)
        std::vector<std::pair<std::string, std::string>> messages;
        if (!system.empty())
            messages.emplace_back("system", system);
        std::string user_content;
        for (size_t i = 0; i < images.size(); i++)
            user_content += "<|vision_start|><|image_pad|><|vision_end|>";
        user_content += prompt;
        messages.emplace_back("user", user_content);
        prompt_tokens = tok.apply_chat_template(messages, true, enable_thinking);
    } else {
        prompt_tokens = tok.encode(prompt);
    }

    // Expand image placeholders
    if (!images.empty()) {
        expand_image_placeholders(prompt_tokens, images, tok);
    }

    InferRequest infer_req;
    infer_req.request_id     = next_request_id();
    infer_req.prompt_tokens  = std::move(prompt_tokens);
    infer_req.max_new_tokens    = max_tokens;
    infer_req.temperature       = temperature;
    infer_req.top_p             = top_p;
    infer_req.top_k             = top_k;
    infer_req.repeat_penalty    = repeat_penalty;
    infer_req.frequency_penalty = frequency_penalty;
    infer_req.presence_penalty  = presence_penalty;
    infer_req.seed              = seed;
    infer_req.stream            = true;
    infer_req.images            = std::move(images);

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
                             "Content-Type: application/x-ndjson\r\n"
                             "Transfer-Encoding: chunked\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto t0 = std::chrono::steady_clock::now();
        std::string finish_reason;
        int comp_toks = poll_tokens(infer_req.request_id, [&](const std::string& piece) {
            std::string line = "{\"model\":\"" + json_escape(model) +
                "\",\"created_at\":\"" + iso8601_now() +
                "\",\"response\":\"" + json_escape(piece) +
                "\",\"done\":false}";
            send_ndjson_chunk(client_fd, line);
        }, config_.timeout_s, enable_thinking, stop_seqs, {}, {}, &finish_reason);

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
    bool enable_thinking = json_get_bool(req.body, "think", true);
    float def_temp = enable_thinking ? 1.0f : 0.7f;
    float def_top_p = enable_thinking ? 0.95f : 0.8f;
    float temperature = (float)json_get_number(req.body, "temperature", def_temp);
    float top_p = (float)json_get_number(req.body, "top_p", def_top_p);
    int top_k = json_get_int(req.body, "top_k", 20);
    float repeat_penalty = (float)json_get_number(req.body, "repeat_penalty", 1.0);
    float frequency_penalty = (float)json_get_number(req.body, "frequency_penalty", 0.0);
    float presence_penalty  = (float)json_get_number(req.body, "presence_penalty", 0.0);
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
                {"system", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tool_prompt});
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
                             "Content-Type: application/x-ndjson\r\n"
                             "Transfer-Encoding: chunked\r\n\r\n";
        send(client_fd, header.c_str(), header.size(), MSG_NOSIGNAL);

        auto t0 = std::chrono::steady_clock::now();
        std::string finish_reason;
        int tool_call_idx = 0;

        int comp_toks = poll_tokens(infer_req.request_id,
            // on_token (content)
            [&](const std::string& piece) {
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\","
                    "\"content\":\"" + json_escape(piece) + "\"},\"done\":false}";
                send_ndjson_chunk(client_fd, line);
            },
            config_.timeout_s,
            enable_thinking,
            stop_seqs,
            // on_reasoning (thinking content)
            [&](const std::string& piece) {
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\","
                    "\"content\":\"\",\"thinking\":\"" + json_escape(piece) + "\"},\"done\":false}";
                send_ndjson_chunk(client_fd, line);
            },
            // on_tool_call
            [&](const ToolCallInfo& tc) {
                std::string line = "{\"model\":\"" + json_escape(model) +
                    "\",\"created_at\":\"" + iso8601_now() +
                    "\",\"message\":{\"role\":\"assistant\",\"content\":\"\","
                    "\"tool_calls\":[{\"function\":{\"name\":\"" + json_escape(tc.name) +
                    "\",\"arguments\":" + tc.arguments + "}}]},"
                    "\"done\":false}";
                send_ndjson_chunk(client_fd, line);
                tool_call_idx++;
            },
            &finish_reason
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

} // namespace serve
} // namespace qwen_thor
