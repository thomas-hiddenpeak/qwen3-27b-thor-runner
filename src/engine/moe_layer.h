#pragma once

#include "tensor.h"
#include "moe_router.h"
#include "grouped_gemm.h"
#include <memory>
#include <vector>

namespace qwen_thor {
namespace ops {

// 基础的线性层 (Dense Linear)，用于共享专家和普通的 QKV 投影
class Linear {
public:
    Linear(int in_features, int out_features, bool bias = false);
    virtual ~Linear() = default;

    virtual void forward(
        const core::Tensor& input,
        const core::Tensor& weight,
        const core::Tensor* bias_tensor, // 可选
        core::Tensor& output,
        void* stream = nullptr
    ) = 0;
};

// Qwen3.5 特有的 MoE 层执行图
// 包含：1 个共享专家 (Shared Expert) + N 个路由专家 (Routed Experts)
class QwenMoELayer {
public:
    QwenMoELayer(
        int hidden_size,
        int intermediate_size,
        int shared_expert_intermediate_size,
        int num_experts,
        int top_k
    );
    ~QwenMoELayer();

    // 执行 MoE 层的前向传播
    // 核心逻辑：利用 CUDA Streams 实现共享专家与路由专家的并发执行
    void forward(
        const core::Tensor& hidden_states,
        // 路由相关权重
        const core::Tensor& gate_weight,
        // 路由专家权重 (通常包含 gate_proj, up_proj, down_proj)
        const std::vector<core::Tensor>& expert_weights,
        // 共享专家权重
        const core::Tensor& shared_expert_gate_proj,
        const core::Tensor& shared_expert_up_proj,
        const core::Tensor& shared_expert_down_proj,
        const core::Tensor& shared_expert_gate_weight, // 共享专家的门控权重 (用于缩放)
        // 输出
        core::Tensor& output,
        void* main_stream // 主 CUDA Stream
    );

private:
    int hidden_size_;
    int intermediate_size_;
    int shared_expert_intermediate_size_;
    int num_experts_;
    int top_k_;

    // 算子实例
    std::unique_ptr<MoERouter> router_;
    std::unique_ptr<GroupedGEMM> grouped_gemm_;
    std::unique_ptr<Linear> shared_expert_linear_;

    // 内部缓冲区 (用于存储中间结果)
    std::unique_ptr<core::Tensor> router_indices_;
    std::unique_ptr<core::Tensor> router_weights_;
    std::unique_ptr<core::Tensor> routed_experts_output_;
    std::unique_ptr<core::Tensor> shared_expert_output_;

    // CUDA Streams 用于并发执行
    void* shared_expert_stream_; // 专用于共享专家的流
};

} // namespace ops
} // namespace qwen_thor
