#include "model.h"
#include "safetensors.h"
#include "light_ops.h"
#include "dense_gemm.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <vector>

namespace fs = std::filesystem;

namespace qwen_thor {
namespace core {

Qwen35Model::Qwen35Model(const Qwen35Config& config) : config_(config) {
    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        layers_.emplace_back(config_, i);
    }
}

Qwen35Model::~Qwen35Model() {
    for (void* ptr : device_weights_) {
        cudaFree(ptr);
    }
}

void Qwen35Model::load_weights(const std::string& model_dir) {
    std::unordered_map<std::string, __nv_bfloat16*>  tensor_map;
    std::unordered_map<std::string, float*> f32_map;   // for A_log (stays FP32)
    std::unordered_map<std::string, void*> raw_map;    // U8/F8_E4M3 NVFP4 tensors
    std::unordered_map<std::string, std::vector<int64_t>> raw_shape_map;
    std::unordered_map<std::string, float> scalar_f32_map;  // NVFP4 global_scale (CPU)
    bool is_nvfp4 = false;

    if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
        throw std::runtime_error("Model directory does not exist: " + model_dir);
    }

    // Create a stream for async dtype conversions
    cudaStream_t conv_stream;
    cudaStreamCreate(&conv_stream);

    // 1. 遍历目录，将所有 safetensors 权重拷贝到 VRAM
    int file_count = 0;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() != ".safetensors") continue;
        ++file_count;
        std::cerr << "Loading shard " << file_count << ": "
                  << entry.path().filename().string() << std::endl;

        auto loader = std::make_unique<io::SafetensorsLoader>(entry.path().string());
        for (const auto& name : loader->get_tensor_names()) {
            auto tensor = loader->get_tensor(name);
            if (!tensor) continue;

            size_t num_elements = 1;
            for (auto dim : tensor->shape()) num_elements *= dim;
            size_t size_bytes = num_elements * core::get_dtype_size(tensor->dtype());

            void* d_ptr = nullptr;
            if (cudaMalloc(&d_ptr, size_bytes) != cudaSuccess)
                throw std::runtime_error("cudaMalloc failed for " + name);
            if (cudaMemcpy(d_ptr, tensor->data(), size_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("cudaMemcpy failed for " + name);

            device_weights_.push_back(d_ptr);

            auto dtype = tensor->dtype();
            if (dtype == core::DataType::U8 || dtype == core::DataType::FP8_E4M3) {
                // NVFP4 quantization tensor (weight_packed or weight_scale)
                raw_map[name] = d_ptr;
                raw_shape_map[name] = std::vector<int64_t>(tensor->shape().begin(), tensor->shape().end());
                is_nvfp4 = true;
            } else if (dtype == core::DataType::FP32 &&
                       name.find("_global_scale") != std::string::npos) {
                // NVFP4 per-projection scalar: read CPU value
                float val = *static_cast<const float*>(tensor->data());
                scalar_f32_map[name] = val;
            } else if (dtype == core::DataType::BF16) {
                // Check for A_log stored as BF16 (NVFP4 model variant)
                bool is_a_log = (name.size() >= 5 &&
                                 name.substr(name.size() - 5) == "A_log");
                if (is_a_log) {
                    // BF16 → FP32 conversion on CPU (only 48 elements)
                    float* d_f32 = nullptr;
                    size_t f32_bytes = num_elements * sizeof(float);
                    if (cudaMalloc(reinterpret_cast<void**>(&d_f32), f32_bytes) != cudaSuccess)
                        throw std::runtime_error("cudaMalloc failed (f32 buf) for " + name);
                    std::vector<float> f32_buf(num_elements);
                    const uint16_t* bf16_src = static_cast<const uint16_t*>(tensor->data());
                    for (size_t j = 0; j < num_elements; j++) {
                        uint32_t bits = static_cast<uint32_t>(bf16_src[j]) << 16;
                        std::memcpy(&f32_buf[j], &bits, sizeof(float));
                    }
                    cudaMemcpy(d_f32, f32_buf.data(), f32_bytes, cudaMemcpyHostToDevice);
                    device_weights_.push_back(d_f32);
                    f32_map[name] = d_f32;
                } else {
                    tensor_map[name] = static_cast<__nv_bfloat16*>(d_ptr);
                }
            } else if (dtype == core::DataType::FP32) {
                // Check if this is an A_log tensor (keep as FP32)
                bool is_a_log = (name.size() >= 5 &&
                                 name.substr(name.size() - 5) == "A_log");
                if (is_a_log) {
                    f32_map[name] = static_cast<float*>(d_ptr);
                } else {
                    // Convert FP32 → FP16 into a new half buffer
                    __nv_bfloat16* d_fp16 = nullptr;
                    size_t fp16_bytes = num_elements * sizeof(__nv_bfloat16);
                    if (cudaMalloc(reinterpret_cast<void**>(&d_fp16), fp16_bytes) != cudaSuccess)
                        throw std::runtime_error("cudaMalloc failed (fp16 buf) for " + name);
                    device_weights_.push_back(d_fp16);
                    ops::invoke_f32_to_bf16(static_cast<float*>(d_ptr), d_fp16, num_elements, conv_stream);
                    tensor_map[name] = d_fp16;
                }
            } else {
                // FP16 or INT8 – keep as-is
                tensor_map[name] = static_cast<__nv_bfloat16*>(d_ptr);
            }
        }
        // 统一内存: 立即释放 mmap, 避免与 cudaMalloc 同时占用双份物理内存
        // (loader 析构 → munmap → OS 回收物理页)
        loader.reset();
    }
    // Wait for all dtype conversions to complete
    cudaStreamSynchronize(conv_stream);
    cudaStreamDestroy(conv_stream);

    std::cerr << "Loaded " << (tensor_map.size() + f32_map.size() + raw_map.size() + scalar_f32_map.size())
              << " tensors (" << file_count << " shards) into VRAM."
              << (is_nvfp4 ? " [NVFP4 quantized model detected]" : "") << std::endl;

    // 2. 绑定权重 — 根据层类型分别绑定
    auto get_ptr = [&](const std::string& key) -> __nv_bfloat16* {
        auto it = tensor_map.find(key);
        if (it != tensor_map.end()) return it->second;
        std::cerr << "Warning: tensor not found: " << key << std::endl;
        return nullptr;
    };
    auto get_f32_ptr = [&](const std::string& key) -> float* {
        auto it = f32_map.find(key);
        if (it != f32_map.end()) return it->second;
        std::cerr << "Warning: f32 tensor not found: " << key << std::endl;
        return nullptr;
    };

    std::cerr << "Weight binding complete." << std::endl;

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        std::string p = "model.language_model.layers." + std::to_string(i) + ".";

        __nv_bfloat16* in_n = get_ptr(p + "input_layernorm.weight");
        __nv_bfloat16* pa_n = get_ptr(p + "post_attention_layernorm.weight");

        if (is_nvfp4) {
            // NVFP4: MLP projections are quantized for all layers
            if (config_.is_full_attention(i)) {
                // Self-attn projections also quantized; only norms are BF16
                layers_[i].get_full_attn()->set_weights(
                    nullptr, nullptr, nullptr, nullptr,
                    get_ptr(p + "self_attn.q_norm.weight"),
                    get_ptr(p + "self_attn.k_norm.weight"),
                    nullptr, nullptr, nullptr,
                    in_n, pa_n);
            } else {
                // Linear attn projections are BF16; MLP is quantized
                layers_[i].get_linear_attn()->set_weights(
                    get_ptr(p + "linear_attn.in_proj_qkv.weight"),
                    get_ptr(p + "linear_attn.in_proj_z.weight"),
                    get_ptr(p + "linear_attn.in_proj_a.weight"),
                    get_ptr(p + "linear_attn.in_proj_b.weight"),
                    get_ptr(p + "linear_attn.out_proj.weight"),
                    get_ptr(p + "linear_attn.conv1d.weight"),
                    get_f32_ptr(p + "linear_attn.A_log"),
                    get_ptr(p + "linear_attn.dt_bias"),
                    get_ptr(p + "linear_attn.norm.weight"),
                    nullptr, nullptr, nullptr,
                    in_n, pa_n);
            }
        } else {
            // Original BF16 path
            __nv_bfloat16* gate = nullptr;
            __nv_bfloat16* up   = nullptr;
            __nv_bfloat16* down = nullptr;

            // MoE: bind router + packed experts + shared expert
            if (config_.is_moe) {
                auto find_ptr = [&](const std::string& key) -> __nv_bfloat16* {
                    auto it = tensor_map.find(key);
                    return it != tensor_map.end() ? it->second : nullptr;
                };
                MoEWeights moe;
                moe.router_w = get_ptr(p + "mlp.gate.weight");
                // Packed 3D expert tensors (no .weight suffix in safetensors)
                moe.experts_gate_up_w = find_ptr(p + "mlp.experts.gate_up_proj");
                moe.experts_down_w    = find_ptr(p + "mlp.experts.down_proj");
                if (!moe.experts_gate_up_w)
                    std::cerr << "Warning: tensor not found: " << p << "mlp.experts.gate_up_proj" << std::endl;
                if (!moe.experts_down_w)
                    std::cerr << "Warning: tensor not found: " << p << "mlp.experts.down_proj" << std::endl;
                moe.shared_gate_w        = get_ptr(p + "mlp.shared_expert.gate_proj.weight");
                moe.shared_up_w          = get_ptr(p + "mlp.shared_expert.up_proj.weight");
                moe.shared_down_w        = get_ptr(p + "mlp.shared_expert.down_proj.weight");
                moe.shared_expert_gate_w = get_ptr(p + "mlp.shared_expert_gate.weight");

                if (config_.is_full_attention(i)) {
                    layers_[i].get_full_attn()->set_moe_weights(moe);
                } else {
                    layers_[i].get_linear_attn()->set_moe_weights(moe);
                }
            } else {
                gate = get_ptr(p + "mlp.gate_proj.weight");
                up   = get_ptr(p + "mlp.up_proj.weight");
                down = get_ptr(p + "mlp.down_proj.weight");
            }

            if (config_.is_full_attention(i)) {
                layers_[i].get_full_attn()->set_weights(
                    get_ptr(p + "self_attn.q_proj.weight"),
                    get_ptr(p + "self_attn.k_proj.weight"),
                    get_ptr(p + "self_attn.v_proj.weight"),
                    get_ptr(p + "self_attn.o_proj.weight"),
                    get_ptr(p + "self_attn.q_norm.weight"),
                    get_ptr(p + "self_attn.k_norm.weight"),
                    gate, up, down, in_n, pa_n);
            } else {
                layers_[i].get_linear_attn()->set_weights(
                    get_ptr(p + "linear_attn.in_proj_qkv.weight"),
                    get_ptr(p + "linear_attn.in_proj_z.weight"),
                    get_ptr(p + "linear_attn.in_proj_a.weight"),
                    get_ptr(p + "linear_attn.in_proj_b.weight"),
                    get_ptr(p + "linear_attn.out_proj.weight"),
                    get_ptr(p + "linear_attn.conv1d.weight"),
                    get_f32_ptr(p + "linear_attn.A_log"),
                    get_ptr(p + "linear_attn.dt_bias"),
                    get_ptr(p + "linear_attn.norm.weight"),
                    gate, up, down, in_n, pa_n);
            }
        }
    }

    // 2c. NVFP4 quantized weight binding
    if (is_nvfp4) {
        auto make_qw = [&](const std::string& prefix) -> QuantizedWeight {
            QuantizedWeight qw;
            std::string pk = prefix + ".weight_packed";
            std::string sk = prefix + ".weight_scale";
            auto pit = raw_map.find(pk);
            auto sit = raw_map.find(sk);
            if (pit == raw_map.end() || sit == raw_map.end()) return qw;
            qw.packed = static_cast<uint8_t*>(pit->second);
            qw.scale = static_cast<uint8_t*>(sit->second);
            auto gsit = scalar_f32_map.find(prefix + ".weight_global_scale");
            auto isit = scalar_f32_map.find(prefix + ".input_global_scale");
            if (gsit != scalar_f32_map.end()) qw.global_scale = gsit->second;
            if (isit != scalar_f32_map.end()) qw.input_scale = isit->second;
            auto shit = raw_shape_map.find(pk);
            if (shit != raw_shape_map.end() && shit->second.size() == 2) {
                qw.N = static_cast<int>(shit->second[0]);
                qw.K = static_cast<int>(shit->second[1]) * 2;  // packed K/2 → logical K
            }
            return qw;
        };

        // Build ptr→idx map for releasing individual expert weights after packing
        std::unordered_map<void*, size_t> fp4_ptr_idx;
        for (size_t j = 0; j < device_weights_.size(); j++) {
            if (device_weights_[j]) fp4_ptr_idx[device_weights_[j]] = j;
        }
        auto release_raw = [&](void* ptr) {
            if (!ptr) return;
            auto it = fp4_ptr_idx.find(ptr);
            if (it != fp4_ptr_idx.end()) {
                cudaFree(ptr);
                device_weights_[it->second] = nullptr;
                fp4_ptr_idx.erase(it);
            }
        };

        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            std::string p = "model.language_model.layers." + std::to_string(i) + ".";

            if (config_.is_moe) {
                // ============================================================
                // NVFP4 MoE: pack individual expert FP4 weights into contiguous
                // buffers and bind shared expert FP4 weights
                // ============================================================
                const int E      = config_.num_experts;
                const int moe_is = config_.moe_intermediate_size;
                const int hs     = config_.hidden_size;

                // -- Expert gate+up: each expert has gate[moe_is, hs] + up[moe_is, hs]
                // Pack into contiguous: [E * 2*moe_is, hs/2] packed, [E * 2*moe_is, hs/16] scale
                const int gu_N = 2 * moe_is;       // per-expert output dim
                const int gu_K = hs;                // input dim
                const int gu_K_half = gu_K / 2;
                const int gu_K_groups = gu_K / 16;
                size_t gu_packed_total = (size_t)E * gu_N * gu_K_half;
                size_t gu_scale_total  = (size_t)E * gu_N * gu_K_groups;

                uint8_t* gu_packed_buf = nullptr;
                uint8_t* gu_scale_buf  = nullptr;
                cudaMalloc(&gu_packed_buf, gu_packed_total);
                cudaMalloc(&gu_scale_buf,  gu_scale_total);
                device_weights_.push_back(gu_packed_buf);
                device_weights_.push_back(gu_scale_buf);

                float gu_global_scale = 1.0f;
                bool gu_gs_set = false;

                // Per-expert inv_global_scale arrays (device + host staging)
                std::vector<float> h_gu_inv_gs(E, 1.0f);

                for (int e = 0; e < E; ++e) {
                    std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                    auto gate_qw = make_qw(ep + "gate_proj");
                    auto up_qw   = make_qw(ep + "up_proj");

                    if (gate_qw.valid()) {
                        h_gu_inv_gs[e] = 1.0f / gate_qw.global_scale;
                    }

                    // Destination offset for this expert: rows [e*gu_N .. e*gu_N + gu_N)
                    size_t row_off = (size_t)e * gu_N;
                    // gate rows [0..moe_is), up rows [moe_is..2*moe_is)
                    if (gate_qw.valid()) {
                        cudaMemcpy(gu_packed_buf + (row_off) * gu_K_half,
                                   gate_qw.packed, (size_t)moe_is * gu_K_half,
                                   cudaMemcpyDeviceToDevice);
                        cudaMemcpy(gu_scale_buf + (row_off) * gu_K_groups,
                                   gate_qw.scale, (size_t)moe_is * gu_K_groups,
                                   cudaMemcpyDeviceToDevice);
                        if (!gu_gs_set) { gu_global_scale = gate_qw.global_scale; gu_gs_set = true; }
                        release_raw(gate_qw.packed);
                        release_raw(gate_qw.scale);
                    }
                    if (up_qw.valid()) {
                        cudaMemcpy(gu_packed_buf + (row_off + moe_is) * gu_K_half,
                                   up_qw.packed, (size_t)moe_is * gu_K_half,
                                   cudaMemcpyDeviceToDevice);
                        cudaMemcpy(gu_scale_buf + (row_off + moe_is) * gu_K_groups,
                                   up_qw.scale, (size_t)moe_is * gu_K_groups,
                                   cudaMemcpyDeviceToDevice);
                        release_raw(up_qw.packed);
                        release_raw(up_qw.scale);
                    }
                }

                // Upload per-expert inv_gs to device
                float* d_gu_inv_gs = nullptr;
                cudaMalloc(&d_gu_inv_gs, E * sizeof(float));
                cudaMemcpy(d_gu_inv_gs, h_gu_inv_gs.data(), E * sizeof(float), cudaMemcpyHostToDevice);
                device_weights_.push_back(d_gu_inv_gs);

                // -- Expert down: each expert has down[hs, moe_is]
                // Pack into contiguous: [E * hs, moe_is/2] packed, [E * hs, moe_is/16] scale
                const int dn_N = hs;
                const int dn_K = moe_is;
                const int dn_K_half = dn_K / 2;
                const int dn_K_groups = dn_K / 16;
                size_t dn_packed_total = (size_t)E * dn_N * dn_K_half;
                size_t dn_scale_total  = (size_t)E * dn_N * dn_K_groups;

                uint8_t* dn_packed_buf = nullptr;
                uint8_t* dn_scale_buf  = nullptr;
                cudaMalloc(&dn_packed_buf, dn_packed_total);
                cudaMalloc(&dn_scale_buf,  dn_scale_total);
                device_weights_.push_back(dn_packed_buf);
                device_weights_.push_back(dn_scale_buf);

                float dn_global_scale = 1.0f;
                bool dn_gs_set = false;

                std::vector<float> h_dn_inv_gs(E, 1.0f);

                for (int e = 0; e < E; ++e) {
                    std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
                    auto down_qw = make_qw(ep + "down_proj");
                    size_t row_off = (size_t)e * dn_N;
                    if (down_qw.valid()) {
                        h_dn_inv_gs[e] = 1.0f / down_qw.global_scale;
                        cudaMemcpy(dn_packed_buf + row_off * dn_K_half,
                                   down_qw.packed, (size_t)dn_N * dn_K_half,
                                   cudaMemcpyDeviceToDevice);
                        cudaMemcpy(dn_scale_buf + row_off * dn_K_groups,
                                   down_qw.scale, (size_t)dn_N * dn_K_groups,
                                   cudaMemcpyDeviceToDevice);
                        if (!dn_gs_set) { dn_global_scale = down_qw.global_scale; dn_gs_set = true; }
                        release_raw(down_qw.packed);
                        release_raw(down_qw.scale);
                    }
                }

                float* d_dn_inv_gs = nullptr;
                cudaMalloc(&d_dn_inv_gs, E * sizeof(float));
                cudaMemcpy(d_dn_inv_gs, h_dn_inv_gs.data(), E * sizeof(float), cudaMemcpyHostToDevice);
                device_weights_.push_back(d_dn_inv_gs);

                // -- Build MoEWeights with FP4 fields
                MoEWeights moe;
                moe.router_w = get_ptr(p + "mlp.gate.weight");
                moe.shared_expert_gate_w = get_ptr(p + "mlp.shared_expert_gate.weight");

                // FP4 expert weights
                moe.fp4_experts_gate_up_packed = gu_packed_buf;
                moe.fp4_experts_gate_up_scale  = gu_scale_buf;
                moe.fp4_experts_gate_up_inv_gs = d_gu_inv_gs;
                moe.fp4_experts_gate_up_N      = gu_N;
                moe.fp4_experts_gate_up_K      = gu_K;
                moe.fp4_experts_down_packed    = dn_packed_buf;
                moe.fp4_experts_down_scale     = dn_scale_buf;
                moe.fp4_experts_down_inv_gs    = d_dn_inv_gs;
                moe.fp4_experts_down_N         = dn_N;
                moe.fp4_experts_down_K         = dn_K;

                // FP4 shared expert weights
                moe.shared_gate_qw = make_qw(p + "mlp.shared_expert.gate_proj");
                moe.shared_up_qw   = make_qw(p + "mlp.shared_expert.up_proj");
                moe.shared_down_qw = make_qw(p + "mlp.shared_expert.down_proj");

                if (config_.is_full_attention(i)) {
                    layers_[i].get_full_attn()->set_moe_weights(moe);
                } else {
                    layers_[i].get_linear_attn()->set_moe_weights(moe);
                }

                // Full attention: bind quantized self-attn projections
                if (config_.is_full_attention(i)) {
                    auto q_qw = make_qw(p + "self_attn.q_proj");
                    auto k_qw = make_qw(p + "self_attn.k_proj");
                    auto v_qw = make_qw(p + "self_attn.v_proj");
                    auto o_qw = make_qw(p + "self_attn.o_proj");
                    layers_[i].get_full_attn()->set_quantized_attn(q_qw, k_qw, v_qw, o_qw);
                } else {
                    // LinearAttn 投影 FP4
                    auto qkv_qw = make_qw(p + "linear_attn.in_proj_qkv");
                    auto z_qw   = make_qw(p + "linear_attn.in_proj_z");
                    auto out_qw = make_qw(p + "linear_attn.out_proj");
                    if (qkv_qw.valid()) {
                        layers_[i].get_linear_attn()->set_quantized_attn(qkv_qw, z_qw, out_qw);
                    }
                }

                if (i == 0) {
                    std::cerr << "[Model] NVFP4 MoE layer 0: " << E << " experts packed ("
                              << "gate_up " << gu_packed_total / 1048576 << "MB + "
                              << "down " << dn_packed_total / 1048576 << "MB), "
                              << "shared expert FP4=" << (moe.shared_gate_qw.valid() ? "yes" : "no")
                              << std::endl;
                }
            } else {
                // Non-MoE NVFP4: bind dense MLP quantized weights
                auto gate_qw = make_qw(p + "mlp.gate_proj");
                auto up_qw   = make_qw(p + "mlp.up_proj");
                auto down_qw = make_qw(p + "mlp.down_proj");
                if (config_.is_full_attention(i)) {
                    auto q_qw = make_qw(p + "self_attn.q_proj");
                    auto k_qw = make_qw(p + "self_attn.k_proj");
                    auto v_qw = make_qw(p + "self_attn.v_proj");
                    auto o_qw = make_qw(p + "self_attn.o_proj");
                    layers_[i].get_full_attn()->set_quantized_attn(q_qw, k_qw, v_qw, o_qw);
                    layers_[i].get_full_attn()->set_quantized_mlp(gate_qw, up_qw, down_qw);
                } else {
                    layers_[i].get_linear_attn()->set_quantized_mlp(gate_qw, up_qw, down_qw);
                    // LinearAttn 投影 FP4 (Kbenkhaled 模型有, Sehyo 模型无)
                    auto qkv_qw = make_qw(p + "linear_attn.in_proj_qkv");
                    auto z_qw   = make_qw(p + "linear_attn.in_proj_z");
                    auto out_qw = make_qw(p + "linear_attn.out_proj");
                    if (qkv_qw.valid()) {
                        layers_[i].get_linear_attn()->set_quantized_attn(qkv_qw, z_qw, out_qw);
                    }
                }
            }
        }
        std::cerr << "[Model] NVFP4 quantized weights bound ("
                  << raw_map.size() / 2 << " quantized projections)" << std::endl;
    }

    // 2b. 合并投影权重 — T=1 Decode GEMV 优化 (128 kernel launches/step saved)
    //     FullAttn: Q+K+V → 单个 [qp_dim+kv_dim*2, hs] 合并权重
    //     LinearAttn: Z+A+B → 单个 [lin_v+nv*2, hs] 合并权重
    //     合并后释放原始分离分配, 个别指针重定向到合并缓冲区子区域
    {
        const int qp_dim = config_.q_proj_dim();   // 12288
        const int kv_dim = config_.kv_dim();       // 1024
        const int lin_v  = config_.lin_v_dim();    // 6144
        const int nv     = config_.linear_num_value_heads;  // 48
        const int hs     = config_.hidden_size;    // 5120

        // Build ptr→index map for O(1) lookup when freeing originals
        std::unordered_map<void*, size_t> ptr_idx;
        for (size_t j = 0; j < device_weights_.size(); j++) {
            if (device_weights_[j]) ptr_idx[device_weights_[j]] = j;
        }
        auto release_weight = [&](void* ptr) {
            auto it = ptr_idx.find(ptr);
            if (it != ptr_idx.end()) {
                cudaFree(ptr);
                device_weights_[it->second] = nullptr;
                ptr_idx.erase(it);
            }
        };

        size_t merged_total = 0;
        for (int i = 0; i < config_.num_hidden_layers; i++) {
            std::string p = "model.language_model.layers." + std::to_string(i) + ".";

            if (config_.is_full_attention(i)) {
                // Merge Q[qp_dim,hs] + K[kv_dim,hs] + V[kv_dim,hs] → [qp_dim+kv_dim*2, hs]
                auto qit = tensor_map.find(p + "self_attn.q_proj.weight");
                auto kit = tensor_map.find(p + "self_attn.k_proj.weight");
                auto vit = tensor_map.find(p + "self_attn.v_proj.weight");
                if (qit == tensor_map.end() || kit == tensor_map.end() || vit == tensor_map.end()) {
                    // NVFP4: self-attn projections are quantized — merge FP4 QKV instead
                    auto* fa = layers_[i].get_full_attn();
                    if (fa->is_quantized()) {
                        auto& qq = fa->get_q_qw();
                        auto& kq = fa->get_k_qw();
                        auto& vq = fa->get_v_qw();
                        if (qq.valid() && kq.valid() && vq.valid()) {
                            int merged_N = qq.N + kq.N + vq.N;  // 14336
                            int K = qq.K;
                            // Concat packed [N, K/2]
                            size_t pk_bytes = (size_t)merged_N * (K / 2);
                            uint8_t* merged_packed = nullptr;
                            cudaMalloc(&merged_packed, pk_bytes);
                            cudaMemcpy(merged_packed,
                                       qq.packed, (size_t)qq.N * (K / 2), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(merged_packed + (size_t)qq.N * (K / 2),
                                       kq.packed, (size_t)kq.N * (K / 2), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(merged_packed + (size_t)(qq.N + kq.N) * (K / 2),
                                       vq.packed, (size_t)vq.N * (K / 2), cudaMemcpyDeviceToDevice);
                            // Concat scale [N, K/16]
                            size_t sc_bytes = (size_t)merged_N * (K / 16);
                            uint8_t* merged_scale = nullptr;
                            cudaMalloc(&merged_scale, sc_bytes);
                            cudaMemcpy(merged_scale,
                                       qq.scale, (size_t)qq.N * (K / 16), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(merged_scale + (size_t)qq.N * (K / 16),
                                       kq.scale, (size_t)kq.N * (K / 16), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(merged_scale + (size_t)(qq.N + kq.N) * (K / 16),
                                       vq.scale, (size_t)vq.N * (K / 16), cudaMemcpyDeviceToDevice);

                            core::QuantizedWeight merged_qw;
                            merged_qw.packed = merged_packed;
                            merged_qw.scale = merged_scale;
                            merged_qw.global_scale = qq.global_scale;  // same for Q/K/V
                            merged_qw.N = merged_N;
                            merged_qw.K = K;
                            fa->set_merged_fp4_qkv(merged_qw);

                            // Release original separate buffers
                            cudaFree(qq.packed); cudaFree(qq.scale);
                            cudaFree(kq.packed); cudaFree(kq.scale);
                            cudaFree(vq.packed); cudaFree(vq.scale);
                            kq.packed = nullptr; kq.scale = nullptr;
                            vq.packed = nullptr; vq.scale = nullptr;
                            qq.packed = merged_packed;
                            qq.scale = merged_scale;
                            qq.N = merged_N;  // q_qw_ now aliases merged

                            // Track new allocations
                            device_weights_.push_back(merged_packed);
                            device_weights_.push_back(merged_scale);
                            merged_total += pk_bytes + sc_bytes;
                        }
                    }
                    continue;
                }
                __nv_bfloat16* q_w = qit->second;
                __nv_bfloat16* k_w = kit->second;
                __nv_bfloat16* v_w = vit->second;

                int merged_N = qp_dim + kv_dim * 2;  // 14336
                size_t bytes = (size_t)merged_N * hs * sizeof(__nv_bfloat16);
                __nv_bfloat16* merged = nullptr;
                cudaMalloc(&merged, bytes);
                cudaMemcpy(merged, q_w,
                           (size_t)qp_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)qp_dim * hs, k_w,
                           (size_t)kv_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)(qp_dim + kv_dim) * hs, v_w,
                           (size_t)kv_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                device_weights_.push_back(merged);
                layers_[i].get_full_attn()->set_merged_qkv(merged);
                release_weight(q_w);
                release_weight(k_w);
                release_weight(v_w);
                merged_total += bytes;
            } else {
                // Super-merge QKV[in_qkv,hs] + Z[lin_v,hs] + A[nv,hs] + B[nv,hs]
                // → single [in_qkv+lin_v+nv*2, hs] = [16480, 5120]
                __nv_bfloat16* qkv_w = tensor_map[p + "linear_attn.in_proj_qkv.weight"];
                __nv_bfloat16* z_w = tensor_map[p + "linear_attn.in_proj_z.weight"];
                __nv_bfloat16* a_w = tensor_map[p + "linear_attn.in_proj_a.weight"];
                __nv_bfloat16* b_w = tensor_map[p + "linear_attn.in_proj_b.weight"];
                if (!qkv_w || !z_w) {
                    // FP4 attn model: merge FP4 QKV+Z if both available
                    auto* la = layers_[i].get_linear_attn();
                    auto& qkv_qw = la->get_qkv_qw();
                    auto& z_qw = la->get_z_qw();
                    if (qkv_qw.valid() && z_qw.valid()) {
                        int merged_N = qkv_qw.N + z_qw.N;  // 10240 + 6144 = 16384
                        int K = qkv_qw.K;
                        // Concat packed [N, K/2]
                        size_t pk_bytes = (size_t)merged_N * (K / 2);
                        uint8_t* mp = nullptr;
                        cudaMalloc(&mp, pk_bytes);
                        cudaMemcpy(mp, qkv_qw.packed, (size_t)qkv_qw.N * (K / 2), cudaMemcpyDeviceToDevice);
                        cudaMemcpy(mp + (size_t)qkv_qw.N * (K / 2), z_qw.packed, (size_t)z_qw.N * (K / 2), cudaMemcpyDeviceToDevice);
                        // Concat scale [N, K/16]
                        size_t sc_bytes = (size_t)merged_N * (K / 16);
                        uint8_t* ms = nullptr;
                        cudaMalloc(&ms, sc_bytes);
                        cudaMemcpy(ms, qkv_qw.scale, (size_t)qkv_qw.N * (K / 16), cudaMemcpyDeviceToDevice);
                        cudaMemcpy(ms + (size_t)qkv_qw.N * (K / 16), z_qw.scale, (size_t)z_qw.N * (K / 16), cudaMemcpyDeviceToDevice);

                        core::QuantizedWeight mqw;
                        mqw.packed = mp; mqw.scale = ms;
                        mqw.global_scale = qkv_qw.global_scale;
                        mqw.N = merged_N; mqw.K = K;
                        la->set_merged_fp4_qkv_z(mqw);

                        // Release originals, redirect qkv_qw to merged sub-region
                        cudaFree(qkv_qw.packed); cudaFree(qkv_qw.scale);
                        cudaFree(z_qw.packed); cudaFree(z_qw.scale);
                        z_qw.packed = nullptr; z_qw.scale = nullptr;
                        qkv_qw.packed = mp; qkv_qw.scale = ms;
                        qkv_qw.N = merged_N;
                        device_weights_.push_back(mp);
                        device_weights_.push_back(ms);
                        merged_total += pk_bytes + sc_bytes;
                    }
                    continue;
                }

                int in_qkv_dim = config_.lin_qk_dim() * 2 + lin_v;  // 10240
                int merged_N = in_qkv_dim + lin_v + nv * 2;  // 16480
                size_t bytes = (size_t)merged_N * hs * sizeof(__nv_bfloat16);
                __nv_bfloat16* merged = nullptr;
                cudaMalloc(&merged, bytes);
                cudaMemcpy(merged, qkv_w,
                           (size_t)in_qkv_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)in_qkv_dim * hs, z_w,
                           (size_t)lin_v * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)(in_qkv_dim + lin_v) * hs, a_w,
                           (size_t)nv * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)(in_qkv_dim + lin_v + nv) * hs, b_w,
                           (size_t)nv * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

                device_weights_.push_back(merged);
                layers_[i].get_linear_attn()->set_merged_all_proj(merged);
                release_weight(qkv_w);
                release_weight(z_w);
                release_weight(a_w);
                release_weight(b_w);
                merged_total += bytes;
            }
        }
        std::cerr << "      Merged projections: " << (merged_total >> 20)
                  << " MB (QKV×" << config_.num_full_attn_layers()
                  << " + QKVZAB×" << (config_.num_hidden_layers - config_.num_full_attn_layers())
                  << ", net zero)" << std::endl;

        // Level 3: Merge Gate+Up projections for T>1 (all 64 layers)
        // gate_proj[is, hs] + up_proj[is, hs] → [2*is, hs] = [34816, 5120]
        // Saves 64 cuBLAS launches per T>1 step (MTP verify T=4)
        size_t gate_up_total = 0;
        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            // MoE layers use packed expert weights — no dense gate/up to merge
            if (config_.is_moe) continue;

            std::string p = "model.language_model.layers." + std::to_string(i) + ".";
            __nv_bfloat16* gate_w = tensor_map[p + "mlp.gate_proj.weight"];
            __nv_bfloat16* up_w   = tensor_map[p + "mlp.up_proj.weight"];
            if (!gate_w || !up_w) {
                // NVFP4: merge FP4 gate+up packed+scale
                core::QuantizedWeight* gq = nullptr;
                core::QuantizedWeight* uq = nullptr;
                if (config_.is_full_attention(i)) {
                    auto* fa = layers_[i].get_full_attn();
                    gq = &fa->get_gate_qw();
                    uq = &fa->get_up_qw();
                } else {
                    auto* la = layers_[i].get_linear_attn();
                    gq = &la->get_gate_qw();
                    uq = &la->get_up_qw();
                }
                if (gq->valid() && uq->valid()) {
                    int merged_N = gq->N + uq->N;  // 2*is
                    int K = gq->K;
                    // Concat packed [N, K/2]
                    size_t pk_bytes = (size_t)merged_N * (K / 2);
                    uint8_t* merged_packed = nullptr;
                    cudaMalloc(&merged_packed, pk_bytes);
                    cudaMemcpy(merged_packed,
                               gq->packed, (size_t)gq->N * (K / 2), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(merged_packed + (size_t)gq->N * (K / 2),
                               uq->packed, (size_t)uq->N * (K / 2), cudaMemcpyDeviceToDevice);
                    // Concat scale [N, K/16]
                    size_t sc_bytes = (size_t)merged_N * (K / 16);
                    uint8_t* merged_scale = nullptr;
                    cudaMalloc(&merged_scale, sc_bytes);
                    cudaMemcpy(merged_scale,
                               gq->scale, (size_t)gq->N * (K / 16), cudaMemcpyDeviceToDevice);
                    cudaMemcpy(merged_scale + (size_t)gq->N * (K / 16),
                               uq->scale, (size_t)uq->N * (K / 16), cudaMemcpyDeviceToDevice);

                    core::QuantizedWeight merged_qw;
                    merged_qw.packed = merged_packed;
                    merged_qw.scale = merged_scale;
                    merged_qw.global_scale = gq->global_scale;
                    merged_qw.N = merged_N;
                    merged_qw.K = K;

                    if (config_.is_full_attention(i)) {
                        layers_[i].get_full_attn()->set_merged_fp4_gate_up(merged_qw);
                    } else {
                        layers_[i].get_linear_attn()->set_merged_fp4_gate_up(merged_qw);
                    }

                    // Release original separate buffers
                    cudaFree(gq->packed); cudaFree(gq->scale);
                    cudaFree(uq->packed); cudaFree(uq->scale);
                    uq->packed = nullptr; uq->scale = nullptr;
                    gq->packed = merged_packed;
                    gq->scale = merged_scale;
                    gq->N = merged_N;

                    device_weights_.push_back(merged_packed);
                    device_weights_.push_back(merged_scale);
                    gate_up_total += pk_bytes + sc_bytes;
                }
                continue;
            }

            int is = config_.intermediate_size;
            int hs = config_.hidden_size;
            size_t bytes = (size_t)2 * is * hs * sizeof(__nv_bfloat16);
            __nv_bfloat16* merged = nullptr;
            cudaMalloc(&merged, bytes);
            cudaMemcpy(merged, gate_w,
                       (size_t)is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
            cudaMemcpy(merged + (size_t)is * hs, up_w,
                       (size_t)is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);

            device_weights_.push_back(merged);
            if (config_.is_full_attention(i)) {
                layers_[i].get_full_attn()->set_merged_gate_up(merged);
            } else {
                layers_[i].get_linear_attn()->set_merged_gate_up(merged);
            }
            release_weight(gate_w);
            release_weight(up_w);
            gate_up_total += bytes;
        }
        std::cerr << "      Merged Gate+Up: " << (gate_up_total >> 20)
                  << " MB (" << config_.num_hidden_layers << " layers, net zero)" << std::endl;
    }

    // 3. 全局権重
    embed_tokens_w_ = get_ptr("model.language_model.embed_tokens.weight");
    norm_w_         = get_ptr("model.language_model.norm.weight");
    lm_head_w_      = get_ptr("lm_head.weight");

    // tie_word_embeddings: lm_head 共享 embed_tokens 权重 (4B 模型)
    if (!lm_head_w_ && config_.tie_word_embeddings && embed_tokens_w_) {
        lm_head_w_ = embed_tokens_w_;
        std::cerr << "  [Weight] tie_word_embeddings: lm_head shares embed_tokens" << std::endl;
    }

    if (!embed_tokens_w_ || !norm_w_ || !lm_head_w_) {
        throw std::runtime_error("Missing essential global weights "
            "(embed_tokens / norm / lm_head)");
    }

    // 4. MTP (Multi-Token Prediction) 权重
    //    mtp.pre_fc_norm_hidden.weight, mtp.pre_fc_norm_embedding.weight,
    //    mtp.fc.weight, mtp.norm.weight, mtp.layers.0.self_attn.*, mtp.layers.0.mlp.*
    {
        auto mtp_get = [&](const std::string& key) -> __nv_bfloat16* {
            auto it = tensor_map.find(key);
            return it != tensor_map.end() ? it->second : nullptr;
        };

        mtp_pre_norm_h_w_ = mtp_get("mtp.pre_fc_norm_hidden.weight");
        mtp_pre_norm_e_w_ = mtp_get("mtp.pre_fc_norm_embedding.weight");
        mtp_fc_w_          = mtp_get("mtp.fc.weight");
        mtp_norm_w_        = mtp_get("mtp.norm.weight");

        // MTP transformer layer weights
        std::string mp = "mtp.layers.0.";
        __nv_bfloat16* mtp_q  = mtp_get(mp + "self_attn.q_proj.weight");
        __nv_bfloat16* mtp_k  = mtp_get(mp + "self_attn.k_proj.weight");
        __nv_bfloat16* mtp_v  = mtp_get(mp + "self_attn.v_proj.weight");
        __nv_bfloat16* mtp_o  = mtp_get(mp + "self_attn.o_proj.weight");
        __nv_bfloat16* mtp_qn = mtp_get(mp + "self_attn.q_norm.weight");
        __nv_bfloat16* mtp_kn = mtp_get(mp + "self_attn.k_norm.weight");
        // Dense MLP: only used for non-MoE models
        __nv_bfloat16* mtp_gp = config_.is_moe ? nullptr : mtp_get(mp + "mlp.gate_proj.weight");
        __nv_bfloat16* mtp_up = config_.is_moe ? nullptr : mtp_get(mp + "mlp.up_proj.weight");
        __nv_bfloat16* mtp_dp = config_.is_moe ? nullptr : mtp_get(mp + "mlp.down_proj.weight");
        __nv_bfloat16* mtp_in = mtp_get(mp + "input_layernorm.weight");
        __nv_bfloat16* mtp_pn = mtp_get(mp + "post_attention_layernorm.weight");

        if (mtp_pre_norm_h_w_ && mtp_fc_w_ && mtp_q) {
            has_mtp_ = true;
            mtp_layer_ = std::make_unique<Qwen35FullAttnLayer>(config_, 0 /* dummy layer_idx */);
            mtp_layer_->set_weights(mtp_q, mtp_k, mtp_v, mtp_o, mtp_qn, mtp_kn,
                                     mtp_gp, mtp_up, mtp_dp, mtp_in, mtp_pn);

            // Build ptr→index map for releasing merged originals
            std::unordered_map<void*, size_t> mtp_ptr_idx;
            for (size_t j = 0; j < device_weights_.size(); j++) {
                if (device_weights_[j]) mtp_ptr_idx[device_weights_[j]] = j;
            }
            auto mtp_release = [&](void* ptr) {
                auto it = mtp_ptr_idx.find(ptr);
                if (it != mtp_ptr_idx.end()) {
                    cudaFree(ptr);
                    device_weights_[it->second] = nullptr;
                    mtp_ptr_idx.erase(it);
                }
            };

            // Merge MTP QKV weights: Q[qp_dim,hs] + K[kv_dim,hs] + V[kv_dim,hs] → [merged_N, hs]
            {
                const int qp_dim = config_.q_proj_dim();
                const int kv_dim = config_.kv_dim();
                const int hs     = config_.hidden_size;
                int merged_N = qp_dim + kv_dim * 2;
                size_t bytes = (size_t)merged_N * hs * sizeof(__nv_bfloat16);
                __nv_bfloat16* merged = nullptr;
                cudaMalloc(&merged, bytes);
                cudaMemcpy(merged, mtp_q,
                           (size_t)qp_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)qp_dim * hs, mtp_k,
                           (size_t)kv_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)(qp_dim + kv_dim) * hs, mtp_v,
                           (size_t)kv_dim * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                device_weights_.push_back(merged);
                mtp_layer_->set_merged_qkv(merged);
                mtp_release(mtp_q);
                mtp_release(mtp_k);
                mtp_release(mtp_v);
            }

            // MoE MTP: pack individual expert weights + bind shared expert
            if (config_.is_moe) {
                const int E      = config_.num_experts;
                const int moe_is = config_.moe_intermediate_size;
                const int hs     = config_.hidden_size;

                // Pack individual experts into contiguous gate_up [E, 2*moe_is, hs] and down [E, hs, moe_is]
                size_t gu_elems = (size_t)E * 2 * moe_is * hs;
                size_t dn_elems = (size_t)E * hs * moe_is;
                __nv_bfloat16* packed_gu = nullptr;
                __nv_bfloat16* packed_dn = nullptr;
                cudaMalloc(&packed_gu, gu_elems * sizeof(__nv_bfloat16));
                cudaMalloc(&packed_dn, dn_elems * sizeof(__nv_bfloat16));
                device_weights_.push_back(packed_gu);
                device_weights_.push_back(packed_dn);

                for (int e = 0; e < E; ++e) {
                    std::string ep = mp + "mlp.experts." + std::to_string(e) + ".";
                    auto* gw = mtp_get(ep + "gate_proj.weight");
                    auto* uw = mtp_get(ep + "up_proj.weight");
                    auto* dw = mtp_get(ep + "down_proj.weight");
                    size_t gu_off = (size_t)e * 2 * moe_is * hs;
                    if (gw) cudaMemcpy(packed_gu + gu_off, gw,
                                       (size_t)moe_is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                    if (uw) cudaMemcpy(packed_gu + gu_off + (size_t)moe_is * hs, uw,
                                       (size_t)moe_is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                    size_t dn_off = (size_t)e * hs * moe_is;
                    if (dw) cudaMemcpy(packed_dn + dn_off, dw,
                                       (size_t)hs * moe_is * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                }

                MoEWeights moe;
                moe.router_w             = mtp_get(mp + "mlp.gate.weight");
                moe.experts_gate_up_w    = packed_gu;
                moe.experts_down_w       = packed_dn;
                moe.shared_gate_w        = mtp_get(mp + "mlp.shared_expert.gate_proj.weight");
                moe.shared_up_w          = mtp_get(mp + "mlp.shared_expert.up_proj.weight");
                moe.shared_down_w        = mtp_get(mp + "mlp.shared_expert.down_proj.weight");
                moe.shared_expert_gate_w = mtp_get(mp + "mlp.shared_expert_gate.weight");
                mtp_layer_->set_moe_weights(moe);

                std::cerr << "[Model] MTP module loaded (MoE: " << E << " experts packed, "
                          << "QKV merged, speculative decoding enabled)" << std::endl;
            } else {
                // Dense MTP: Merge Gate+Up weights: [2*is, hs]
                const int is = config_.intermediate_size;
                const int hs = config_.hidden_size;
                size_t bytes = (size_t)2 * is * hs * sizeof(__nv_bfloat16);
                __nv_bfloat16* merged = nullptr;
                cudaMalloc(&merged, bytes);
                cudaMemcpy(merged, mtp_gp,
                           (size_t)is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                cudaMemcpy(merged + (size_t)is * hs, mtp_up,
                           (size_t)is * hs * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
                device_weights_.push_back(merged);
                mtp_layer_->set_merged_gate_up(merged);
                mtp_release(mtp_gp);
                mtp_release(mtp_up);

                std::cerr << "[Model] MTP module loaded (1 transformer layer, "
                          << "QKV+GateUp merged, speculative decoding enabled)" << std::endl;
            }
        } else {
            std::cerr << "[Model] No MTP weights found, speculative decoding disabled" << std::endl;
        }
    }

    // 5. Vision Encoder 权重 (ViT + Merger)
    //    model.visual.patch_embed.proj.weight/bias
    //    model.visual.pos_embed.weight
    //    model.visual.blocks.{0-26}.{norm1,attn.qkv,attn.proj,norm2,mlp.linear_fc1,mlp.linear_fc2}.weight/bias
    //    model.visual.merger.{norm,linear_fc1,linear_fc2}.weight/bias
    {
        auto vis_get = [&](const std::string& key) -> __nv_bfloat16* {
            auto it = tensor_map.find(key);
            return it != tensor_map.end() ? it->second : nullptr;
        };

        __nv_bfloat16* patch_w = vis_get("model.visual.patch_embed.proj.weight");
        __nv_bfloat16* patch_b = vis_get("model.visual.patch_embed.proj.bias");
        __nv_bfloat16* pos_w   = vis_get("model.visual.pos_embed.weight");

        if (patch_w && patch_b && pos_w) {
            VisionConfig vcfg;  // uses defaults matching Qwen3.5
            vcfg.out_hidden_size = config_.hidden_size;  // match text model
            vision_encoder_ = std::make_unique<VisionEncoder>(vcfg);
            vision_encoder_->set_patch_embed_weights(patch_w, patch_b);
            vision_encoder_->set_pos_embed_weight(pos_w);

            // Load 27 ViT block weights
            bool all_blocks_ok = true;
            for (int i = 0; i < vcfg.depth; i++) {
                std::string vp = "model.visual.blocks." + std::to_string(i) + ".";
                __nv_bfloat16* n1w = vis_get(vp + "norm1.weight");
                __nv_bfloat16* n1b = vis_get(vp + "norm1.bias");
                __nv_bfloat16* qw  = vis_get(vp + "attn.qkv.weight");
                __nv_bfloat16* qb  = vis_get(vp + "attn.qkv.bias");
                __nv_bfloat16* pw  = vis_get(vp + "attn.proj.weight");
                __nv_bfloat16* pb  = vis_get(vp + "attn.proj.bias");
                __nv_bfloat16* n2w = vis_get(vp + "norm2.weight");
                __nv_bfloat16* n2b = vis_get(vp + "norm2.bias");
                __nv_bfloat16* f1w = vis_get(vp + "mlp.linear_fc1.weight");
                __nv_bfloat16* f1b = vis_get(vp + "mlp.linear_fc1.bias");
                __nv_bfloat16* f2w = vis_get(vp + "mlp.linear_fc2.weight");
                __nv_bfloat16* f2b = vis_get(vp + "mlp.linear_fc2.bias");

                if (!n1w || !qw || !pw || !n2w || !f1w || !f2w) {
                    std::cerr << "[Model] Missing vision block " << i << " weights" << std::endl;
                    all_blocks_ok = false;
                    break;
                }
                vision_encoder_->set_block_weights(i, n1w, n1b, qw, qb, pw, pb,
                                                     n2w, n2b, f1w, f1b, f2w, f2b);
            }

            // Load merger weights
            __nv_bfloat16* mn_w = vis_get("model.visual.merger.norm.weight");
            __nv_bfloat16* mn_b = vis_get("model.visual.merger.norm.bias");
            __nv_bfloat16* mf1w = vis_get("model.visual.merger.linear_fc1.weight");
            __nv_bfloat16* mf1b = vis_get("model.visual.merger.linear_fc1.bias");
            __nv_bfloat16* mf2w = vis_get("model.visual.merger.linear_fc2.weight");
            __nv_bfloat16* mf2b = vis_get("model.visual.merger.linear_fc2.bias");

            if (all_blocks_ok && mn_w && mf1w && mf2w) {
                vision_encoder_->set_merger_weights(mn_w, mn_b, mf1w, mf1b, mf2w, mf2b);
                has_vision_ = true;
                std::cerr << "[Model] Vision encoder loaded (27-layer ViT + merger, "
                          << "~461M params)" << std::endl;
            } else {
                std::cerr << "[Model] Missing merger weights, vision disabled" << std::endl;
                vision_encoder_.reset();
            }
        } else {
            std::cerr << "[Model] No vision weights found, multimodal disabled" << std::endl;
        }
    }
}


// ============================================================================
// MTP Forward: 使用主模型隐藏状态 + token embedding 预测下一个 token
//
// 架构:
//   concat(RMSNorm(h), RMSNorm(embed(tok))) → fc [hs, 2hs] → FullAttnLayer → RMSNorm → lm_head
//
// Workspace layout (T=1, all at workspace pointer):
//   [0..hs)           norm_e       (RMSNorm of embedding)
//   [hs..2hs)         norm_h       (RMSNorm of main hidden, concat = [0..2hs) = [embed,hidden])
//   [2hs..3hs)        projected    (fc output, also hidden_states for attn layer)
//   [3hs..4hs)        raw_embed    (embedding lookup output)
//   [4hs..4hs+attn_ws)  attn_ws    (FullAttnLayer workspace for T=1)
//   after attn_ws:    normed [hs] + logits [vocab] + d_ids [2 ints]
// ============================================================================
__nv_bfloat16* Qwen35Model::mtp_forward(
    const __nv_bfloat16* main_hidden,
    int input_token_id,
    int pos_id,
    ops::KVCacheManager& mtp_kv_manager,
    const int* d_block_tables,
    const int* d_context_lens,
    int max_num_blocks_per_seq,
    int max_context_len,
    __nv_bfloat16* workspace,
    cudaStream_t stream,
    __nv_bfloat16** out_hidden,
    const int* d_input_token_id,
    perf::PerfProfiler* profiler)
{
    if (!has_mtp_ || !mtp_layer_) return nullptr;

    const int hs = config_.hidden_size;       // 5120
    const int vocab = config_.vocab_size;      // 248320

    // Workspace pointers
    // Layout: norm_e[hs] | norm_h[hs] | projected[hs] | raw_embed[hs] | attn_ws[full_attn_ws] | normed[hs] | logits[vocab] | d_ids[2 ints]
    // concat = [norm_e, norm_h] = [embed_norm, hidden_norm] (SGLang/HF 标准顺序)
    const int attn_ws_elems = config_.full_attn_workspace_elems_t1();  // 93184
    __nv_bfloat16* norm_e    = workspace;                     // [5120] — embed norm FIRST
    __nv_bfloat16* norm_h    = norm_e + hs;                   // [5120] — hidden norm SECOND
    __nv_bfloat16* projected = norm_h + hs;                   // [5120]
    __nv_bfloat16* raw_embed = projected + hs;                // [5120]
    __nv_bfloat16* attn_ws   = raw_embed + hs;
    __nv_bfloat16* normed    = attn_ws + attn_ws_elems;
    __nv_bfloat16* logits    = normed + hs;
    int* d_ids               = reinterpret_cast<int*>(logits + vocab);  // 2 ints at the very end

    // 1. RMSNorm(main_hidden) → norm_h
    if (profiler) profiler->begin("mtp_prep", stream);
    ops::invoke_rmsnorm(norm_h, main_hidden, mtp_pre_norm_h_w_,
                        config_.rms_norm_eps, 1, hs, stream);

    // 2. Embedding lookup → raw_embed, then RMSNorm → norm_e
    if (d_input_token_id) {
        // GPU-resident path: token ID already on device
        ops::invoke_embedding_lookup(raw_embed, d_input_token_id, embed_tokens_w_, 1, hs, stream);
    } else {
        // CPU path: H2D copy then lookup
        cudaMemcpyAsync(d_ids, &input_token_id, sizeof(int), cudaMemcpyHostToDevice, stream);
        ops::invoke_embedding_lookup(raw_embed, d_ids, embed_tokens_w_, 1, hs, stream);
    }
    ops::invoke_rmsnorm(norm_e, raw_embed, mtp_pre_norm_e_w_,
                        config_.rms_norm_eps, 1, hs, stream);

    // 3. FC projection: concat(norm_e, norm_h) = [10240] → projected = [5120]
    //    fc.weight is [5120, 10240], GEMV: projected = fc_w × [embed_norm, hidden_norm]
    ops::invoke_dense_gemv(norm_e, mtp_fc_w_, projected, hs, 2 * hs, stream);
    if (profiler) profiler->end("mtp_prep", stream);

    // 4. Full attention transformer layer
    //    projected serves as hidden_states (modified in-place with residual)
    if (profiler) profiler->begin("mtp_attn", stream);
    cudaMemcpyAsync(d_ids, &pos_id, sizeof(int), cudaMemcpyHostToDevice, stream);
    mtp_layer_->forward(
        projected,      // hidden_states [1, hs], in-place
        d_ids,          // pos_ids [1]
        mtp_kv_manager,
        d_block_tables,
        d_context_lens,
        max_num_blocks_per_seq, max_context_len,
        1,              // num_tokens
        0,              // full_attn_idx (MTP has 1 layer, always 0)
        attn_ws,        // workspace
        stream
    );
    if (profiler) profiler->end("mtp_attn", stream);

    // 5. Final RMSNorm (mtp.norm, centered weight)
    if (profiler) profiler->begin("mtp_lmhead", stream);
    ops::invoke_rmsnorm(normed, projected, mtp_norm_w_,
                        config_.rms_norm_eps, 1, hs, stream);

    // 6. LM head (shared with main model): GEMV [vocab, hs] × [hs] → [vocab]
    ops::invoke_dense_gemv(normed, lm_head_w_, logits, vocab, hs, stream);
    if (profiler) profiler->end("mtp_lmhead", stream);

    // Output MTP hidden state for chaining (projected = post-attention transformer output)
    if (out_hidden) *out_hidden = projected;

    return logits;
}

// ============================================================================
// Prefill 前向传播: 単請求, T>1 tokens, per-layer sync (統一メモリ必須)
// ============================================================================
// ============================================================================
// Prefill 前向传播
//
// 与 decode 同理，必须逐层 cudaStreamSynchronize:
//   SM110 统一内存 + 大量 kernel 深排队 (64层 × ~15 kernels/层 ≈ 960 kernels)
//   会导致 SMMU/驱动层面的数据损坏或资源耗尽。
//   prefill 中每层 kernel 数量比 decode 更多 (GEMM 尺寸更大)，
//   Prefill 一般只对性能影响约 1-3% (因单次 prefill GPU 计算时间远长于 decode)。
// ============================================================================
void Qwen35Model::forward_prefill(
    __nv_bfloat16* hidden_states,
    const int* pos_ids,
    const ops::KVCacheManager& kv_manager,
    const int* block_tables,
    const int* context_lens,
    int max_num_blocks_per_seq,
    int max_context_len,
    int num_tokens,
    __nv_bfloat16** ssm_states,
    __nv_bfloat16** conv_states,
    __nv_bfloat16* workspace,
    cudaStream_t stream,
    bool force_paged_attn)
{
    int lin_idx = 0;
    int fa_idx  = 0;

    for (int i = 0; i < config_.num_hidden_layers; ++i) {

        if (config_.is_full_attention(i)) {
            layers_[i].get_full_attn()->forward(
                hidden_states, pos_ids, kv_manager,
                block_tables, context_lens,
                max_num_blocks_per_seq, max_context_len,
                num_tokens, fa_idx, workspace, stream,
                1 /* batch_size=1 */, force_paged_attn);
            ++fa_idx;
        } else {
            __nv_bfloat16** lin_ssm = ssm_states ? ssm_states + lin_idx : nullptr;
            __nv_bfloat16** lin_conv = conv_states ? conv_states + lin_idx : nullptr;
            layers_[i].get_linear_attn()->forward(
                hidden_states,
                lin_ssm ? lin_ssm[0] : nullptr,
                lin_conv ? lin_conv[0] : nullptr,
                num_tokens, workspace, stream,
                1 /* batch_size=1 */,
                lin_ssm,
                lin_conv);
            ++lin_idx;
        }

        // 逐层 stream sync — 防止深排队引发 SM110 统一内存数据损坏
        cudaStreamSynchronize(stream);
    }
}

// ============================================================================
// Decode 前向传播: batch_size 个请求各 1 token
//
// 必须逐层 cudaStreamSynchronize:
//   SM110 统一内存 + 大量 kernel 深排队 (64层 × ~15 kernels/层 ≈ 960 kernels)
//   会导致 SMMU/驱动层面的资源耗尽，引发不可恢复的 GPU hard-reset。
//   逐层 sync 开销 ≈ 64 × 10μs = 0.64ms/step (vs ~237ms/step GPU 计算)，可忽略。
//   已验证: 无 sync → req27/step408 崩溃; 有 sync → 50 请求稳定通过。
// ============================================================================
void Qwen35Model::forward_decode(
    __nv_bfloat16* hidden_states,
    const int* pos_ids,
    const ops::KVCacheManager& kv_manager,
    const int* block_tables,
    const int* context_lens,
    int max_num_blocks_per_seq,
    int max_context_len,
    int batch_size,
    __nv_bfloat16** ssm_states,
    __nv_bfloat16** conv_states,
    __nv_bfloat16* workspace,
    cudaStream_t stream)
{
    if (batch_size <= 0) batch_size = 1;
    int num_tokens = batch_size;  // decode: 每请求 1 token, 共 batch_size 个 token
    int lin_idx = 0;
    int fa_idx  = 0;

    // === 可选层级计时 (环境变量 QWEN_LAYER_TIMING=1 启用) ===
    static bool timing_enabled = !!getenv("QWEN_LAYER_TIMING");
    static int timing_count = 0;
    cudaEvent_t ev_start, ev_end;
    float fa_total_ms = 0, la_total_ms = 0;
    if (timing_enabled) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_end);
    }

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        if (timing_enabled) {
            cudaEventRecord(ev_start, stream);
        }

        if (config_.is_full_attention(i)) {
            layers_[i].get_full_attn()->forward(
                hidden_states, pos_ids, kv_manager,
                block_tables, context_lens,
                max_num_blocks_per_seq, max_context_len,
                num_tokens, fa_idx, workspace, stream,
                batch_size);
            ++fa_idx;
        } else {
            __nv_bfloat16** lin_ssm = ssm_states ? ssm_states + lin_idx * batch_size : nullptr;
            __nv_bfloat16** lin_conv = conv_states ? conv_states + lin_idx * batch_size : nullptr;
            layers_[i].get_linear_attn()->forward(
                hidden_states,
                lin_ssm ? lin_ssm[0] : nullptr,
                lin_conv ? lin_conv[0] : nullptr,
                num_tokens, workspace, stream,
                batch_size,
                lin_ssm,
                lin_conv);
            ++lin_idx;
        }

        // 逐层 stream sync — 防止深排队引发 SM110 统一内存 hard-reset
        cudaStreamSynchronize(stream);

        if (timing_enabled) {
            cudaEventRecord(ev_end, stream);
            cudaEventSynchronize(ev_end);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev_start, ev_end);
            if (config_.is_full_attention(i))
                fa_total_ms += ms;
            else
                la_total_ms += ms;
        }
    }

    if (timing_enabled) {
        ++timing_count;
        if (timing_count % 20 == 0) {
            int n_fa = 0, n_la = 0;
            for (int i = 0; i < config_.num_hidden_layers; i++)
                config_.is_full_attention(i) ? n_fa++ : n_la++;
            fprintf(stderr, "[LayerTiming] step=%d: FullAttn(%d)=%.2fms(%.2f/layer), "
                            "LinearAttn(%d)=%.2fms(%.2f/layer), Total=%.2fms\n",
                    timing_count, n_fa, fa_total_ms, n_fa ? fa_total_ms/n_fa : 0.0f,
                    n_la, la_total_ms, n_la ? la_total_ms/n_la : 0.0f,
                    fa_total_ms + la_total_ms);
            fflush(stderr);
        }
    }

    if (timing_enabled) {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }
}

} // namespace core
} // namespace qwen_thor
