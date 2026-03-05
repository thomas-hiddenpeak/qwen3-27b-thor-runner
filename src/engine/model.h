#pragma once

#include "layer.h"
#include "vision.h"
#include "safetensors.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace qwen_thor {
namespace core {

class Qwen35Model {
public:
    Qwen35Model(const Qwen35Config& config);
    ~Qwen35Model();

    // 从目录加载所有的 safetensors 文件，并进行零拷贝映射
    void load_weights(const std::string& model_dir);

    // ======================================================================
    // P/D 分离: 独立的 Prefill 和 Decode 前向传播路径
    // ======================================================================

    // Prefill: 单请求, T > 1 tokens, 有 per-layer cudaStreamSynchronize
    //   (统一内存 + CUTLASS TMA 在 queue depth 过深时性能退化 3×, 必须 per-layer sync)
    // ssm_states[lin_idx] — 每个 linear_attn 层一个 SSM 状态
    // conv_states[lin_idx] — 每个 linear_attn 层一个 Conv 状态
    // force_paged_attn: chunked prefill 非首块时为 true, 强制 full attn 层使用 paged attention
    void forward_prefill(
        __nv_bfloat16* hidden_states,    // [num_tokens, hs]
        const int* pos_ids,              // [num_tokens]
        const ops::KVCacheManager& kv_manager,
        const int* block_tables,         // [max_num_blocks_per_seq]
        const int* context_lens,         // [1]
        int max_num_blocks_per_seq,
        int max_context_len,
        int num_tokens,                  // T > 1
        __nv_bfloat16** ssm_states,             // [num_lin_layers]
        __nv_bfloat16** conv_states,     // [num_lin_layers]
        __nv_bfloat16* workspace,
        cudaStream_t stream = 0,
        bool force_paged_attn = false    // chunked prefill: force paged attn for non-first chunks
    );

    // Batched Decode: batch_size 个请求各 1 token, 无 per-layer sync, CUDA Graph 兼容
    // ssm_states[lin_idx * batch_size + batch_idx] — SSM 状态指针
    // conv_states[lin_idx * batch_size + batch_idx] — Conv 状态指针
    void forward_decode(
        __nv_bfloat16* hidden_states,    // [batch_size, hs]
        const int* pos_ids,              // [batch_size]
        const ops::KVCacheManager& kv_manager,
        const int* block_tables,         // [batch_size, max_num_blocks_per_seq]
        const int* context_lens,         // [batch_size]
        int max_num_blocks_per_seq,
        int max_context_len,
        int batch_size,                  // num_tokens = batch_size
        __nv_bfloat16** ssm_states,             // [num_lin_layers * batch_size]
        __nv_bfloat16** conv_states,     // [num_lin_layers * batch_size]
        __nv_bfloat16* workspace,
        cudaStream_t stream = 0
    );

    // 获取配置
    const Qwen35Config& get_config() const { return config_; }

    // 获取词表 embedding 和 lm_head
    __nv_bfloat16* get_embed_tokens() const { return embed_tokens_w_; }
    __nv_bfloat16* get_norm_weight() const { return norm_w_; }
    __nv_bfloat16* get_lm_head() const { return lm_head_w_; }

    // 层访问 (streaming attention 时 engine 直接迭代)
    int num_layers() const { return (int)layers_.size(); }
    Qwen35Layer& get_layer(int idx) { return layers_[idx]; }

    // MTP (Multi-Token Prediction) 投机解码
    bool has_mtp() const { return has_mtp_; }

    // MTP forward: 根据主模型隐藏状态 + token embedding 预测下一个 token 的 logits
    // 返回指向 workspace 中 logits 的指针 [vocab_size]
    // main_hidden: [1, hidden_size] — 主模型最后一层输出 (未 norm)
    // input_token_id: token to embed (host value)
    // pos_id: 绝对位置 (用于 RoPE, host value)
    // mtp_kv_manager: MTP 层自己的 KV cache (1 层)
    __nv_bfloat16* mtp_forward(
        const __nv_bfloat16* main_hidden,
        int input_token_id,
        int pos_id,
        ops::KVCacheManager& mtp_kv_manager,
        const int* d_block_tables,
        const int* d_context_lens,
        int max_num_blocks_per_seq,
        int max_context_len,
        __nv_bfloat16* workspace,
        cudaStream_t stream
    );

private:
    Qwen35Config config_;
    std::vector<Qwen35Layer> layers_;
    std::vector<std::unique_ptr<io::SafetensorsLoader>> loaders_;
    std::vector<void*> device_weights_; // 存储分配的显存指针
    
    // 词表 embedding 和最后的 lm_head
    __nv_bfloat16* embed_tokens_w_ = nullptr;
    __nv_bfloat16* norm_w_ = nullptr;
    __nv_bfloat16* lm_head_w_ = nullptr;

    // MTP (Multi-Token Prediction) module
    bool has_mtp_ = false;
    __nv_bfloat16* mtp_pre_norm_h_w_ = nullptr;  // RMSNorm [hs] for main hidden
    __nv_bfloat16* mtp_pre_norm_e_w_ = nullptr;  // RMSNorm [hs] for embedding
    __nv_bfloat16* mtp_fc_w_ = nullptr;           // Linear [hs, 2*hs] fusion
    __nv_bfloat16* mtp_norm_w_ = nullptr;          // RMSNorm [hs] before lm_head
    std::unique_ptr<Qwen35FullAttnLayer> mtp_layer_;  // 1 transformer layer

    // Vision Encoder (ViT + Merger)
    bool has_vision_ = false;
    std::unique_ptr<VisionEncoder> vision_encoder_;

public:
    // Vision accessor
    bool has_vision() const { return has_vision_; }
    VisionEncoder* get_vision_encoder() { return vision_encoder_.get(); }
    const VisionConfig& get_vision_config() const {
        static VisionConfig default_cfg;
        return vision_encoder_ ? vision_encoder_->config() : default_cfg;
    }
};

} // namespace core
} // namespace qwen_thor