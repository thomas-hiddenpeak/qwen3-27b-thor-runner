#include "model.h"
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
        loaders_.push_back(std::move(loader));
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
            __nv_bfloat16* gate = get_ptr(p + "mlp.gate_proj.weight");
            __nv_bfloat16* up   = get_ptr(p + "mlp.up_proj.weight");
            __nv_bfloat16* down = get_ptr(p + "mlp.down_proj.weight");

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

        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            std::string p = "model.language_model.layers." + std::to_string(i) + ".";
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
                    // NVFP4: self-attn projections are quantized, skip QKV merge
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
                  << " MB (QKV×16 + QKVZAB×48, net zero)" << std::endl;

        // Level 3: Merge Gate+Up projections for T>1 (all 64 layers)
        // gate_proj[is, hs] + up_proj[is, hs] → [2*is, hs] = [34816, 5120]
        // Saves 64 cuBLAS launches per T>1 step (MTP verify T=4)
        size_t gate_up_total = 0;
        for (int i = 0; i < config_.num_hidden_layers; ++i) {
            std::string p = "model.language_model.layers." + std::to_string(i) + ".";
            __nv_bfloat16* gate_w = tensor_map[p + "mlp.gate_proj.weight"];
            __nv_bfloat16* up_w   = tensor_map[p + "mlp.up_proj.weight"];
            if (!gate_w || !up_w) continue;

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
                  << " MB (64 layers, net zero)" << std::endl;
    }

    // 3. 全局权重
    embed_tokens_w_ = get_ptr("model.language_model.embed_tokens.weight");
    norm_w_         = get_ptr("model.language_model.norm.weight");
    lm_head_w_      = get_ptr("lm_head.weight");

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
        __nv_bfloat16* mtp_gp = mtp_get(mp + "mlp.gate_proj.weight");
        __nv_bfloat16* mtp_up = mtp_get(mp + "mlp.up_proj.weight");
        __nv_bfloat16* mtp_dp = mtp_get(mp + "mlp.down_proj.weight");
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

            // Merge MTP Gate+Up weights: [2*is, hs]
            {
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
            }

            std::cerr << "[Model] MTP module loaded (1 transformer layer, "
                      << "QKV+GateUp merged, speculative decoding enabled)" << std::endl;
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

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
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
    }
}

} // namespace core
} // namespace qwen_thor
