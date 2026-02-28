// tokenizer.h — GPT2-style Byte-Level BPE Tokenizer for Qwen3.5-27B
//
// 从 model_dir 加载 vocab.json + merges.txt, 实现:
//   - encode(text) → token IDs
//   - decode(token_id) → text
//   - apply_chat_template() → Qwen3 ChatML 格式
//
// 编码流程: text → NFC → regex split → byte-level encode → BPE → vocab lookup
// 解码流程: token_id → vocab lookup → byte-level decode → UTF-8 text

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace qwen_thor {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer() = default;

    // 从 model_dir 加载 vocab.json + merges.txt
    // 返回 true 表示成功
    bool load(const std::string& model_dir);

    // 是否已加载
    bool is_loaded() const { return loaded_; }

    // ======================================================================
    // 编码: text → token IDs
    // ======================================================================
    std::vector<int> encode(const std::string& text) const;

    // ======================================================================
    // 解码: token ID(s) → text
    // ======================================================================
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& token_ids) const;

    // ======================================================================
    // Chat Template (Qwen3 ChatML)
    // ======================================================================
    // 构建 ChatML 格式的 prompt 并 tokenize
    // messages: [(role, content), ...]
    //   role = "system" | "user" | "assistant"
    // add_generation_prompt: 是否在末尾添加 "<|im_start|>assistant\n"
    std::vector<int> apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = true) const;

    // ======================================================================
    // 特殊 Token ID
    // ======================================================================
    int eos_token_id()   const { return eos_id_; }
    int im_start_id()    const { return im_start_id_; }
    int im_end_id()      const { return im_end_id_; }
    int eot_id()         const { return eot_id_; }
    int think_start_id() const { return think_start_id_; }
    int think_end_id()   const { return think_end_id_; }
    int tool_call_start_id() const { return tool_call_start_id_; }
    int tool_call_end_id()   const { return tool_call_end_id_; }
    int vocab_size()     const { return static_cast<int>(id_to_piece_.size()); }

private:
    // ======================================================================
    // 内部方法
    // ======================================================================

    // 初始化 GPT2 byte ↔ unicode 映射表
    void init_byte_mapping();

    // 将 UTF-8 文本的每个字节映射为 byte-level unicode 字符
    std::string byte_encode(const std::string& utf8_text) const;

    // 将 byte-level unicode 字符序列还原为 UTF-8 字节
    std::string byte_decode(const std::string& piece) const;

    // 预分词: 将文本按 GPT2 模式拆分为 "词" 片段
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // 对单个 byte-encoded 片段执行 BPE 合并
    std::vector<int> bpe(const std::string& piece) const;

    // 解析 vocab.json → piece_to_id_ + id_to_piece_
    bool load_vocab(const std::string& path);

    // 解析 merges.txt → merges_
    bool load_merges(const std::string& path);

    // ======================================================================
    // 数据
    // ======================================================================

    bool loaded_ = false;

    // 词表: piece ↔ id
    std::unordered_map<std::string, int> piece_to_id_;
    std::vector<std::string> id_to_piece_;

    // 特殊/added tokens: content → id (优先于 BPE)
    std::unordered_map<std::string, int> special_tokens_;
    std::vector<std::string> special_token_list_;  // 按长度降序排列, 用于贪心匹配

    // BPE 合并规则: "a b" → rank (越小优先级越高)
    std::unordered_map<std::string, int> merges_;

    // GPT2 byte-level 映射
    std::string byte_to_unicode_[256];                      // byte → UTF-8 string
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;  // UTF-8 string → byte

    // 特殊 Token IDs
    int eos_id_        = -1;   // <|im_end|> (Qwen 的 EOS)
    int im_start_id_   = -1;   // <|im_start|>
    int im_end_id_     = -1;   // <|im_end|>
    int eot_id_        = -1;   // <|endoftext|>
    int think_start_id_= -1;   // <think>
    int think_end_id_  = -1;   // </think>
    int tool_call_start_id_ = -1; // <tool_call>
    int tool_call_end_id_   = -1; // </tool_call>
};

} // namespace qwen_thor
