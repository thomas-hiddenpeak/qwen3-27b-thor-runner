#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

namespace qwen_thor {
namespace ops {

// Qwen3.5 MoE 路由算子接口
// 负责计算每个 Token 应该被分配给哪些专家，并计算对应的权重
class MoERouter {
public:
    // 构造函数
    // num_experts: 总路由专家数 (例如 512)
    // top_k: 每个 Token 激活的路由专家数 (例如 10)
    // hidden_size: 隐藏层维度
    MoERouter(int num_experts, int top_k, int hidden_size);
    virtual ~MoERouter() = default;

    // 执行路由计算
    // 输入:
    //   hidden_states: [batch_size * seq_len, hidden_size]
    //   gate_weights: 门控网络权重，必须保持 8-bit 或更高精度以防路由坍塌
    // 输出:
    //   expert_indices: [batch_size * seq_len, top_k] (每个 Token 选中的专家索引)
    //   expert_weights: [batch_size * seq_len, top_k] (每个 Token 对应专家的归一化权重)
    virtual void forward(
        const core::Tensor& hidden_states,
        const core::Tensor& gate_weights,
        core::Tensor& expert_indices,
        core::Tensor& expert_weights,
        void* stream = nullptr // cudaStream_t
    ) = 0;

protected:
    int num_experts_;
    int top_k_;
    int hidden_size_;
};

// 针对 Jetson Thor (SM110) 优化的 CUDA 路由实现
class MoERouterCUDA : public MoERouter {
public:
    MoERouterCUDA(int num_experts, int top_k, int hidden_size);
    ~MoERouterCUDA() override;

    void forward(
        const core::Tensor& hidden_states,
        const core::Tensor& gate_weights,
        core::Tensor& expert_indices,
        core::Tensor& expert_weights,
        void* stream = nullptr
    ) override;
};

} // namespace ops
} // namespace qwen_thor
