#pragma once

#include "tensor.h"
#include <memory>
#include <vector>

namespace qwen_thor {
namespace ops {

// Grouped GEMM 算子接口
// 专为 MoE 设计，能够根据 Router 的输出，将不同 Token 动态分发给不同的专家权重进行矩阵乘法
// 核心优化目标：利用 Blackwell 的 TMA 异步预取专家权重，并使用 WGMMA 进行计算
class GroupedGEMM {
public:
    GroupedGEMM() = default;
    virtual ~GroupedGEMM() = default;

    // 执行 Grouped GEMM 计算
    // 输入:
    //   hidden_states: [batch_size * seq_len, hidden_size]
    //   expert_indices: [batch_size * seq_len, top_k] (来自 Router)
    //   expert_weights: [batch_size * seq_len, top_k] (来自 Router)
    //   expert_matrices: 包含所有专家权重的张量集合 (通常在统一内存中)
    // 输出:
    //   output: [batch_size * seq_len, hidden_size] (聚合后的结果)
    virtual void forward(
        const core::Tensor& hidden_states,
        const core::Tensor& expert_indices,
        const core::Tensor& expert_weights,
        const std::vector<core::Tensor>& expert_matrices,
        core::Tensor& output,
        void* stream = nullptr // cudaStream_t
    ) = 0;
};

// 针对 Jetson Thor (SM110) 优化的 CUTLASS Grouped GEMM 实现
class GroupedGEMMCUTLASS : public GroupedGEMM {
public:
    GroupedGEMMCUTLASS();
    ~GroupedGEMMCUTLASS() override;

    void forward(
        const core::Tensor& hidden_states,
        const core::Tensor& expert_indices,
        const core::Tensor& expert_weights,
        const std::vector<core::Tensor>& expert_matrices,
        core::Tensor& output,
        void* stream = nullptr
    ) override;

private:
    // 内部辅助结构，用于管理 CUTLASS 3.x/4.x 的 TMA 描述符和 WGMMA 配置
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ops
} // namespace qwen_thor
