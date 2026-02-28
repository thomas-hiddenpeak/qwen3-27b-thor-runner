#include "moe_layer.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace qwen_thor {
namespace ops {

QwenMoELayer::QwenMoELayer(
    int hidden_size,
    int intermediate_size,
    int shared_expert_intermediate_size,
    int num_experts,
    int top_k
) : hidden_size_(hidden_size),
    intermediate_size_(intermediate_size),
    shared_expert_intermediate_size_(shared_expert_intermediate_size),
    num_experts_(num_experts),
    top_k_(top_k) {
    
    // 初始化算子 (这里假设有具体的 CUDA 实现类)
    // router_ = std::make_unique<MoERouterCUDA>(num_experts_, top_k_, hidden_size_);
    // grouped_gemm_ = std::make_unique<GroupedGEMMCUTLASS>();
    // shared_expert_linear_ = std::make_unique<LinearCUDA>(...);

    // 创建用于并发执行的 CUDA Stream
    cudaError_t err = cudaStreamCreate((cudaStream_t*)&shared_expert_stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream for shared expert.");
    }
}

QwenMoELayer::~QwenMoELayer() {
    if (shared_expert_stream_) {
        cudaStreamDestroy((cudaStream_t)shared_expert_stream_);
    }
}

void QwenMoELayer::forward(
    const core::Tensor& hidden_states,
    const core::Tensor& gate_weight,
    const std::vector<core::Tensor>& expert_weights,
    const core::Tensor& shared_expert_gate_proj,
    const core::Tensor& shared_expert_up_proj,
    const core::Tensor& shared_expert_down_proj,
    const core::Tensor& shared_expert_gate_weight,
    core::Tensor& output,
    void* main_stream
) {
    cudaStream_t stream_main = (cudaStream_t)main_stream;
    cudaStream_t stream_shared = (cudaStream_t)shared_expert_stream_;

    // -----------------------------------------------------------------
    // 分支 1: 路由专家 (Routed Experts) - 在 main_stream 上执行
    // -----------------------------------------------------------------
    // 1.1 执行 Router 计算 Top-K 索引和权重
    // router_->forward(hidden_states, gate_weight, *router_indices_, *router_weights_, stream_main);

    // 1.2 执行 Grouped GEMM (包含 SwiGLU 激活)
    // grouped_gemm_->forward(hidden_states, *router_indices_, *router_weights_, expert_weights, *routed_experts_output_, stream_main);

    // -----------------------------------------------------------------
    // 分支 2: 共享专家 (Shared Expert) - 在 shared_expert_stream 上并发执行
    // -----------------------------------------------------------------
    // 2.1 共享专家的 Gate & Up 投影
    // shared_expert_linear_->forward(hidden_states, shared_expert_gate_proj, nullptr, temp_gate, stream_shared);
    // shared_expert_linear_->forward(hidden_states, shared_expert_up_proj, nullptr, temp_up, stream_shared);
    
    // 2.2 SwiGLU 激活 (temp_gate * sigmoid(temp_gate) * temp_up)
    // launch_swiglu_kernel(..., stream_shared);

    // 2.3 共享专家的 Down 投影
    // shared_expert_linear_->forward(temp_swiglu, shared_expert_down_proj, nullptr, *shared_expert_output_, stream_shared);

    // 2.4 共享专家的门控缩放 (Shared Expert Gate)
    // launch_elementwise_mul_kernel(*shared_expert_output_, shared_expert_gate_weight, stream_shared);

    // -----------------------------------------------------------------
    // 同步与聚合 (Reduce)
    // -----------------------------------------------------------------
    // 确保 shared_expert_stream 的计算完成，然后再在 main_stream 上进行相加
    cudaEvent_t shared_done_event;
    cudaEventCreate(&shared_done_event);
    cudaEventRecord(shared_done_event, stream_shared);
    
    // 让 main_stream 等待 shared_done_event
    cudaStreamWaitEvent(stream_main, shared_done_event, 0);

    // 最终聚合: output = routed_experts_output_ + shared_expert_output_
    // launch_add_kernel(*routed_experts_output_, *shared_expert_output_, output, stream_main);

    cudaEventDestroy(shared_done_event);

    // std::cout << "QwenMoELayer forward executed (simulated)." << std::endl;
}

} // namespace ops
} // namespace qwen_thor
