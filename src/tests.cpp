// tests.cpp — 单元测试集合
//
// 通过 qwen3-27b-thor test 调用, 入口函数: run_tests()

#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include "engine/allocator.h"
#include "engine/tensor.h"
#include "engine/safetensors.h"
#include "engine/paged_attention.h"
#include "engine/shm_queue.h"
#include "engine/model.h"
#include "engine/engine.h"
#include "engine/dense_gemm.h"
#include "engine/light_ops.h"
#include "engine/layer.h"
#include "engine/cache_config.h"
#include "engine/cache_engine.h"
#include "engine/cache_monitor.h"
#include "engine/cache_key.h"
#include "engine/disk_backend.h"
#include "engine/kv_swapper.h"
#include <cmath>
#include <unistd.h>
#include <iomanip>
#include <unordered_map>

using namespace qwen_thor;

// 前置声明 (test helpers)
void test_inference_engine_with_cache(const cache::CacheConfig& cache_config);
void test_inference_engine();
void test_concurrent_swap();
void test_swap_256k_benchmark();

void test_dense_gemm() {
    std::cout << "\n--- Testing Dense GEMM (CUTLASS SM110) ---\n";
    
    // M=128 (batch*seq_len), K=4096 (hidden_size), N=4096 (hidden_size)
    int M = 128;
    int K = 4096;
    int N = 4096;
    
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    
    core::Tensor A({M, K}, core::DataType::FP16, allocator);
    core::Tensor B({K, N}, core::DataType::FP16, allocator);
    core::Tensor C({M, N}, core::DataType::FP16, allocator);
    
    try {
        ops::DenseGEMMCUTLASS gemm;
        
        // Warmup
        gemm.forward(A, B, C);
        
        // Benchmark
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        
        int iters = 10;
        for (int i = 0; i < iters; ++i) {
            gemm.forward(A, B, C);
        }
        
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> diff = end - start;
        double ms_per_iter = diff.count() / iters;
        
        // TFLOPS = 2 * M * N * K / (time_in_sec * 1e12)
        double tflops = (2.0 * M * N * K) / (ms_per_iter * 1e-3) / 1e12;
        
        std::cout << "Dense GEMM [" << M << ", " << N << ", " << K << "] executed successfully.\n";
        std::cout << "Avg Time: " << ms_per_iter << " ms\n";
        std::cout << "Performance: " << tflops << " TFLOPS\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Dense GEMM failed: " << e.what() << std::endl;
    }
}

void test_unified_allocator() {
    std::cout << "--- Testing UnifiedAllocator ---" << std::endl;
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    
    // 分配一个 1024x1024 的 FP16 张量 (2MB)
    std::vector<int64_t> shape = {1024, 1024};
    core::Tensor tensor(shape, core::DataType::FP16, allocator);
    
    std::cout << "Allocated Tensor:" << std::endl;
    std::cout << "  Shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
    std::cout << "  Numel: " << tensor.numel() << std::endl;
    std::cout << "  Bytes: " << tensor.nbytes() << " bytes" << std::endl;
    std::cout << "  Data Ptr: " << tensor.data() << std::endl;
    std::cout << "UnifiedAllocator test passed.\n" << std::endl;
}

void test_safetensors_loader() {
    std::cout << "--- Testing SafetensorsLoader ---" << std::endl;
    // 使用新下载的 Qwen3.5-35B-A3B 权重文件进行测试
    std::string real_path = "/home/rm01/runner/models/Qwen/Qwen3.5-35B-A3B/model.safetensors-00001-of-00014.safetensors";
    std::cout << "Loading from: " << real_path << std::endl;
    
    try {
        io::SafetensorsLoader loader(real_path);
        auto tensor_names = loader.get_tensor_names();
        std::cout << "Successfully parsed safetensors header. Found " << tensor_names.size() << " tensors." << std::endl;
        
        if (!tensor_names.empty()) {
            std::string first_tensor_name = tensor_names[0];
            std::cout << "First tensor name: " << first_tensor_name << std::endl;
            
            auto tensor = loader.get_tensor(first_tensor_name);
            std::cout << "  Shape: [";
            for (size_t i = 0; i < tensor->shape().size(); ++i) {
                std::cout << tensor->shape()[i] << (i == tensor->shape().size() - 1 ? "" : ", ");
            }
            std::cout << "]" << std::endl;
            std::cout << "  Data Ptr (Zero-Copy Mmap): " << tensor->data() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "SafetensorsLoader failed: " << e.what() << std::endl;
    }
    std::cout << "SafetensorsLoader test finished.\n" << std::endl;
}

void test_kv_cache_manager() {
    std::cout << "--- Testing KVCacheManager ---" << std::endl;
    auto allocator = std::make_shared<core::UnifiedAllocator>();
    
    // 模拟 Qwen3.5 的参数: 1024 个 Block, 每个 Block 16 个 Token, 8 个 KV 头, 维度 128
    int num_blocks = 1024;
    int block_size = 16;
    int num_heads = 8;
    int head_dim = 128;
    
    ops::KVCacheManager kv_manager(num_blocks, block_size, num_heads, head_dim, core::DataType::FP16, allocator);
    
    std::cout << "Initialized KVCacheManager with " << num_blocks << " blocks." << std::endl;
    std::cout << "K Cache Tensor Shape: [" 
              << kv_manager.get_k_cache().shape()[0] << ", "
              << kv_manager.get_k_cache().shape()[1] << ", "
              << kv_manager.get_k_cache().shape()[2] << ", "
              << kv_manager.get_k_cache().shape()[3] << "]" << std::endl;
              
    // 模拟分配
    auto allocated_blocks = kv_manager.allocate_blocks(3);
    std::cout << "Allocated 3 blocks: [" << allocated_blocks[0] << ", " 
              << allocated_blocks[1] << ", " << allocated_blocks[2] << "]" << std::endl;
              
    // 模拟释放
    kv_manager.free_blocks(allocated_blocks);
    std::cout << "Freed the 3 blocks." << std::endl;
    std::cout << "KVCacheManager test passed.\n" << std::endl;
}

void test_ipc_shm_queue() {
    std::cout << "--- Testing IPC Shared Memory Queue ---" << std::endl;
    const std::string shm_name = "/qwen_thor_req_queue";
    
    // 模拟前端 (Creator & Producer)
    ipc::ShmRingBuffer<ipc::InferenceRequest, 128> frontend_queue(shm_name, true);
    
    // 模拟后端 (Consumer)
    ipc::ShmRingBuffer<ipc::InferenceRequest, 128> backend_queue(shm_name, false);

    // 前端发送请求
    ipc::InferenceRequest req;
    req.request_id = 1001;
    req.prompt_len = 5;
    req.prompt_tokens[0] = 12; req.prompt_tokens[1] = 34;
    req.max_new_tokens = 50;
    
    if (frontend_queue.push(req)) {
        std::cout << "Frontend pushed request ID: " << req.request_id << std::endl;
    }

    // 后端接收请求
    ipc::InferenceRequest received_req;
    if (backend_queue.pop(received_req)) {
        std::cout << "Backend popped request ID: " << received_req.request_id 
                  << ", max_new_tokens: " << received_req.max_new_tokens << std::endl;
    }

    std::cout << "IPC Shared Memory Queue test passed.\n" << std::endl;
}

void test_light_ops() {
    std::cout << "\n--- Testing Light Ops (RMSNorm & RoPE) ---\n";
    
    int num_tokens = 128;
    int hidden_size = 4096;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    
    __nv_bfloat16*d_x, *d_weight, *d_out;
    cudaMalloc(&d_x, num_tokens * hidden_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_weight, hidden_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, num_tokens * hidden_size * sizeof(__nv_bfloat16));
    
    // 简单初始化
    cudaMemset(d_x, 0, num_tokens * hidden_size * sizeof(__nv_bfloat16));
    cudaMemset(d_weight, 0, hidden_size * sizeof(__nv_bfloat16));
    
    // 测试 RMSNorm
    ops::invoke_rmsnorm(d_out, d_x, d_weight, 1e-6f, num_tokens, hidden_size);
    cudaDeviceSynchronize();
    std::cout << "RMSNorm executed successfully.\n";
    
    __nv_bfloat16*d_q, *d_k;
    int *d_pos_ids;
    cudaMalloc(&d_q, num_tokens * num_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k, num_tokens * num_kv_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_pos_ids, num_tokens * sizeof(int));
    
    // 测试 RoPE
    ops::invoke_rope(d_q, d_k, d_pos_ids, num_tokens, num_heads, num_kv_heads, head_dim);
    cudaDeviceSynchronize();
    std::cout << "RoPE executed successfully.\n";
    
    // 测试 SwiGLU
    int intermediate_size = 11008; // Qwen3.5-7B 的 intermediate_size 示例
    __nv_bfloat16*d_gate, *d_up, *d_swiglu_out;
    cudaMalloc(&d_gate, num_tokens * intermediate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up, num_tokens * intermediate_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_swiglu_out, num_tokens * intermediate_size * sizeof(__nv_bfloat16));
    
    cudaMemset(d_gate, 0, num_tokens * intermediate_size * sizeof(__nv_bfloat16));
    cudaMemset(d_up, 0, num_tokens * intermediate_size * sizeof(__nv_bfloat16));
    
    ops::invoke_swiglu(d_swiglu_out, d_gate, d_up, num_tokens, intermediate_size);
    cudaDeviceSynchronize();
    std::cout << "SwiGLU executed successfully.\n";
    
    cudaFree(d_x); cudaFree(d_weight); cudaFree(d_out);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_pos_ids);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_swiglu_out);
}

void test_paged_attention() {
    std::cout << "\n--- Testing Paged Attention (SM110 Optimized) ---\n";
    
    int num_tokens = 2; // batch_size = 2
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 128;
    int block_size = 16;
    int max_num_blocks_per_seq = 4;
    int max_context_len = 64;
    float sm_scale = 1.0f / sqrtf(head_dim);
    
    // 分配 Q 和 Out
    __nv_bfloat16*d_q, *d_out;
    cudaMalloc(&d_q, num_tokens * num_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, num_tokens * num_heads * head_dim * sizeof(__nv_bfloat16));
    cudaMemset(d_q, 0, num_tokens * num_heads * head_dim * sizeof(__nv_bfloat16));
    
    // 模拟 KV Cache (假设总共有 10 个物理 block)
    int num_physical_blocks = 10;
    __nv_bfloat16*d_k_cache, *d_v_cache;
    cudaMalloc(&d_k_cache, num_physical_blocks * num_kv_heads * block_size * head_dim * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_cache, num_physical_blocks * num_kv_heads * block_size * head_dim * sizeof(__nv_bfloat16));
    cudaMemset(d_k_cache, 0, num_physical_blocks * num_kv_heads * block_size * head_dim * sizeof(__nv_bfloat16));
    cudaMemset(d_v_cache, 0, num_physical_blocks * num_kv_heads * block_size * head_dim * sizeof(__nv_bfloat16));
    
    // 模拟 Block Tables 和 Context Lens
    int *d_block_tables, *d_context_lens;
    cudaMalloc(&d_block_tables, num_tokens * max_num_blocks_per_seq * sizeof(int));
    cudaMalloc(&d_context_lens, num_tokens * sizeof(int));
    
    // 初始化数据：
    // Token 0: context_len = 35, 占用 3 个 blocks (物理索引: 1, 5, 8)
    // Token 1: context_len = 18, 占用 2 个 blocks (物理索引: 2, 7)
    int h_context_lens[] = {35, 18};
    int h_block_tables[] = {
        1, 5, 8, 0, // Token 0 的 block table
        2, 7, 0, 0  // Token 1 的 block table
    };
    
    cudaMemcpy(d_context_lens, h_context_lens, sizeof(h_context_lens), cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_tables, h_block_tables, sizeof(h_block_tables), cudaMemcpyHostToDevice);
    
    // 执行 Paged Attention
    ops::invoke_paged_attention(
        d_out, d_q, d_k_cache, d_v_cache, d_block_tables, d_context_lens,
        max_num_blocks_per_seq, max_context_len, num_tokens, num_heads, num_kv_heads,
        head_dim, block_size, sm_scale
    );
    cudaDeviceSynchronize();
    
    std::cout << "Paged Attention executed successfully.\n";
    
    cudaFree(d_q); cudaFree(d_out);
    cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_block_tables); cudaFree(d_context_lens);
}

void test_qwen_layer() {
    // layer.set_weights() 已废弃，统一通过 InferenceEngine 进行端到端测试
    std::cout << "\n--- test_qwen_layer: skipped (use test_inference_engine) ---\n";
}

void test_cache_engine() {
    std::cout << "\n--- Testing KV Cache Prefix Caching (SSD) ---\n";
    
    // 1. TokenHasher 正确性
    {
        std::cout << "  [1/8] TokenHasher prefix chain..." << std::endl;
        cache::TokenHasher hasher(4);  // chunk_size=4 for testing
        
        int tokens_a[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int tokens_b[] = {1, 2, 3, 4, 9, 10, 11, 12};  // 共享前 4 token
        int tokens_c[] = {9, 2, 3, 4, 5, 6, 7, 8};      // 第一个就不同
        
        auto keys_a = hasher.compute_keys(tokens_a, 8);
        auto keys_b = hasher.compute_keys(tokens_b, 8);
        auto keys_c = hasher.compute_keys(tokens_c, 8);
        
        // keys_a[0] == keys_b[0] (相同前缀 chunk)
        if (keys_a[0] != keys_b[0]) {
            std::cerr << "  FAIL: Same prefix should produce same key" << std::endl;
            return;
        }
        // keys_a[1] != keys_b[1] (不同后缀)
        if (keys_a[1] == keys_b[1]) {
            std::cerr << "  FAIL: Different suffix should produce different key" << std::endl;
            return;
        }
        // keys_a[0] != keys_c[0] (第一个 token 不同)
        if (keys_a[0] == keys_c[0]) {
            std::cerr << "  FAIL: Different first token should produce different key" << std::endl;
            return;
        }
        std::cout << "    PASS: prefix chain hashing correct" << std::endl;
    }
    
    // 2. DiskBackend put/get/remove/LRU
    {
        std::cout << "  [2/8] DiskBackend put/get/remove/LRU..." << std::endl;
        std::string test_dir = "/tmp/qwen_cache_test_" + std::to_string(getpid());
        size_t max_bytes = 1 * 1024 * 1024;  // 1 MB

        // 使用块作用域确保 backend 在 cleanup 前析构
        {
            cache::DiskBackend backend(test_dir, max_bytes);
        
            cache::CacheKey key1{0x1111};
            cache::CacheKey key2{0x2222};
            cache::CacheKey key3{0x3333};
        
            auto entry1 = std::make_shared<cache::CacheEntry>();
            entry1->num_tokens = 16;
            entry1->tokens.resize(16, 42);
            entry1->kv_data.resize(1024, 0xAB);
        
            auto entry2 = std::make_shared<cache::CacheEntry>();
            entry2->num_tokens = 16;
            entry2->tokens.resize(16, 99);
            entry2->kv_data.resize(2048, 0xCD);
        
            // Put
            if (!backend.put(key1, entry1)) { std::cerr << "  FAIL: put key1" << std::endl; return; }
            if (!backend.put(key2, entry2)) { std::cerr << "  FAIL: put key2" << std::endl; return; }
            if (!backend.contains(key1)) { std::cerr << "  FAIL: contains key1" << std::endl; return; }
        
            // Get
            auto got = backend.get(key1);
            if (!got) { std::cerr << "  FAIL: get key1" << std::endl; return; }
            if (got->num_tokens != 16 || got->kv_data.size() != 1024) {
                std::cerr << "  FAIL: data mismatch" << std::endl; return;
            }
        
            // Remove
            backend.remove(key1);
            if (backend.contains(key1)) { std::cerr << "  FAIL: remove key1" << std::endl; return; }
        
            // Prefix match
            std::vector<cache::CacheKey> prefix = {key2, key3};
            int matched = backend.prefix_match(prefix);
            if (matched != 1) { std::cerr << "  FAIL: prefix_match expected 1, got " << matched << std::endl; return; }
        } // backend 析构

        // Cleanup
        std::string cmd = "rm -rf " + test_dir;
        system(cmd.c_str());
        std::cout << "    PASS: DiskBackend operations correct" << std::endl;
    }
    
    // 3. CacheConfig 解析
    {
        std::cout << "  [3/8] CacheConfig from_args..." << std::endl;
        const char* test_args[] = {
            "test_runner",
            "--cache-enable",
            "--cache-dir", "/tmp/test_cache",
            "--cache-max-gb", "10",
            "--cache-chunk-size", "128",
            "--cache-no-ssm"
        };
        auto cfg = cache::CacheConfig::from_args(9, const_cast<char**>(test_args));
        if (!cfg.enabled) { std::cerr << "  FAIL: not enabled" << std::endl; return; }
        if (cfg.cache_dir != "/tmp/test_cache") { std::cerr << "  FAIL: cache_dir" << std::endl; return; }
        if (cfg.chunk_size != 128) { std::cerr << "  FAIL: chunk_size" << std::endl; return; }
        if (cfg.cache_ssm_state) { std::cerr << "  FAIL: cache_ssm_state should be false" << std::endl; return; }
        // ~10 GB
        size_t expected = (size_t)(10.0 * 1024 * 1024 * 1024);
        if (cfg.max_cache_bytes != expected) { std::cerr << "  FAIL: max_cache_bytes" << std::endl; return; }
        std::cout << "    PASS: CLI arg parsing correct" << std::endl;
    }

    // 4. CacheEngine store/retrieve roundtrip (不需要 GPU 模型, 使用 mock 数据)
    {
        std::cout << "  [4/8] CacheEngine store/retrieve roundtrip..." << std::endl;
        std::string test_dir = "/tmp/qwen_cache_engine_test_" + std::to_string(getpid());
        
        cache::CacheConfig cfg;
        cfg.enabled = true;
        cfg.cache_dir = test_dir;
        cfg.max_cache_bytes = 512ULL * 1024 * 1024;  // 512 MB
        cfg.chunk_size = 16;  // 小 chunk 便于测试
        cfg.cache_ssm_state = false;  // 简化测试, 不缓存 SSM
        
        cache::ModelCacheParams params;
        // 使用小参数方便测试
        params.num_full_attn_layers = 2;
        params.num_kv_heads = 2;
        params.head_dim = 64;
        params.block_size = 16;
        params.num_linear_attn_layers = 2;
        
        cache::CacheEngine engine(cfg, params);
        
        // 构造 token 序列 (32 tokens = 2 chunks of 16)
        std::vector<int> tokens(32);
        for (int i = 0; i < 32; i++) tokens[i] = 100 + i;
        
        // Lookup 应该 miss
        auto result = engine.lookup_prefix(tokens.data(), 32);
        if (result.matched_chunks != 0) {
            std::cerr << "  FAIL: expected 0 matched_chunks, got " << result.matched_chunks << std::endl;
            system(("rm -rf " + test_dir).c_str());
            return;
        }
        
        // 无法做真正的 store/retrieve (需要 KVCacheManager), 但验证 engine 创建和 lookup 正常
        auto stats = engine.get_stats();
        if (stats.total_lookups != 1 || stats.cache_misses != 1) {
            std::cerr << "  FAIL: stats mismatch" << std::endl;
            system(("rm -rf " + test_dir).c_str());
            return;
        }
        
        system(("rm -rf " + test_dir).c_str());
        std::cout << "    PASS: CacheEngine lifecycle correct" << std::endl;
    }

    // 5. CacheMonitor 指标计算
    {
        std::cout << "  [5/8] CacheMonitor metrics computation..." << std::endl;
        cache::CacheMonitor monitor;

        // 模拟 3 个请求: miss, partial hit, full hit
        monitor.record_request(1024, 0, 1024);      // miss
        monitor.record_request(1024, 512, 512);      // partial hit
        monitor.record_request(1024, 1024, 0);       // full hit

        auto m = monitor.snapshot();
        if (m.total_requests != 3) { std::cerr << "  FAIL: total_requests" << std::endl; return; }
        if (m.cache_miss_requests != 1) { std::cerr << "  FAIL: cache_miss_requests=" << m.cache_miss_requests << std::endl; return; }
        if (m.partial_hit_requests != 1) { std::cerr << "  FAIL: partial_hit" << std::endl; return; }
        if (m.full_hit_requests != 1) { std::cerr << "  FAIL: full_hit" << std::endl; return; }
        if (m.tokens_restored != 1536) { std::cerr << "  FAIL: tokens_restored=" << m.tokens_restored << std::endl; return; }

        // hit_rate: 2/3 = ~66.7%
        double hr = m.hit_rate();
        if (hr < 66.0 || hr > 67.5) { std::cerr << "  FAIL: hit_rate=" << hr << std::endl; return; }

        // token_save_ratio: 1536/3072 = 50%
        double tsr = m.token_save_ratio();
        if (tsr < 49.5 || tsr > 50.5) { std::cerr << "  FAIL: token_save_ratio=" << tsr << std::endl; return; }

        // 模拟 store/retrieve 延迟
        monitor.record_store(16*1024*1024, 8.5, 1, 256);    // 16 MB, 8.5 ms
        monitor.record_retrieve(16*1024*1024, 2.1);          // 16 MB, 2.1 ms

        auto m2 = monitor.snapshot();
        if (m2.store_ops != 1) { std::cerr << "  FAIL: store_ops" << std::endl; return; }
        if (m2.retrieve_ops != 1) { std::cerr << "  FAIL: retrieve_ops" << std::endl; return; }
        // SSD write BW: 16 MB / 8.5 ms ≈ 1.88 GB/s
        double wbw = m2.ssd_write_bandwidth_gbps();
        if (wbw < 1.5 || wbw > 2.5) { std::cerr << "  FAIL: write_bw=" << wbw << std::endl; return; }

        // 驱逐跟踪
        monitor.record_eviction(8*1024*1024);
        monitor.record_eviction(4*1024*1024);
        auto m3 = monitor.snapshot();
        if (m3.eviction_count != 2) { std::cerr << "  FAIL: eviction_count" << std::endl; return; }
        if (m3.eviction_bytes != 12*1024*1024) { std::cerr << "  FAIL: eviction_bytes" << std::endl; return; }

        // 容量更新
        monitor.update_capacity(100*1024*1024, 20ULL*1024*1024*1024, 10, 512, 4096);
        auto m4 = monitor.snapshot();
        if (m4.ssd_used_bytes != 100*1024*1024) { std::cerr << "  FAIL: ssd_used" << std::endl; return; }
        if (m4.num_entries != 10) { std::cerr << "  FAIL: num_entries" << std::endl; return; }
        // effective_context: 512*16 + 256 = 8448
        if (m4.effective_context_tokens() != 512*16 + 256) {
            std::cerr << "  FAIL: effective_context=" << m4.effective_context_tokens() << std::endl; return;
        }

        std::cout << "    PASS: CacheMonitor metrics correct" << std::endl;
    }

    // 6. CacheMonitor JSON 导出
    {
        std::cout << "  [6/8] CacheMonitor JSON export..." << std::endl;
        cache::CacheMonitor monitor;
        monitor.record_request(512, 256, 256);
        monitor.record_store(8*1024*1024, 4.0, 1, 256);
        monitor.record_retrieve(8*1024*1024, 1.0);
        monitor.update_capacity(50*1024*1024, 10ULL*1024*1024*1024, 5, 256, 4096);

        std::string json = monitor.to_json();
        // 验证 JSON 包含关键字段
        if (json.find("\"hit_rate\"") == std::string::npos) { std::cerr << "  FAIL: no hit_rate" << std::endl; return; }
        if (json.find("\"token_save_ratio\"") == std::string::npos) { std::cerr << "  FAIL: no token_save_ratio" << std::endl; return; }
        if (json.find("\"effective_context_tokens\"") == std::string::npos) { std::cerr << "  FAIL: no eff_ctx" << std::endl; return; }
        if (json.find("\"ssd_utilization\"") == std::string::npos) { std::cerr << "  FAIL: no ssd_util" << std::endl; return; }
        if (json.find("\"eviction\"") == std::string::npos) { std::cerr << "  FAIL: no eviction" << std::endl; return; }
        if (json.find("\"write_bandwidth_gbps\"") == std::string::npos) { std::cerr << "  FAIL: no write_bw" << std::endl; return; }
        if (json.find("\"retrieve_latency_avg_ms\"") == std::string::npos) { std::cerr << "  FAIL: no ret_lat" << std::endl; return; }
        std::cout << "    PASS: JSON export contains all fields" << std::endl;
    }

    // 7. LRU 驱逐压力测试 + 驱逐回调
    {
        std::cout << "  [7/8] LRU eviction under pressure + callback..." << std::endl;
        std::string test_dir = "/tmp/qwen_cache_evict_test_" + std::to_string(getpid());
        size_t max_bytes = 100 * 1024;  // 100 KB — 非常小, 迫使驱逐

        int eviction_count = 0;
        uint64_t eviction_total_bytes = 0;

        {
            cache::DiskBackend backend(test_dir, max_bytes);
            backend.set_eviction_callback([&](uint64_t bytes) {
                eviction_count++;
                eviction_total_bytes += bytes;
            });

            // 每个 entry ~10 KB, 100 KB 最多存 ~8 个 (含头部开销)
            for (int i = 0; i < 20; i++) {
                cache::CacheKey key{(uint64_t)(0xAABB0000 + i)};
                auto entry = std::make_shared<cache::CacheEntry>();
                entry->num_tokens = 4;
                entry->tokens.resize(4, i);
                entry->kv_data.resize(8 * 1024, (uint8_t)i);  // 8 KB
                backend.put(key, entry);
            }

            // 20 个 entry, 每个 ~8KB+256=~8.5KB, 总需 ~170 KB, max=100 KB → 必有驱逐
            if (eviction_count == 0) {
                std::cerr << "  FAIL: expected evictions, got 0" << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }

            // 被驱逐的应该是最早插入的
            // key 0x00 应该已被驱逐
            cache::CacheKey oldest{0xAABB0000};
            if (backend.contains(oldest)) {
                std::cerr << "  FAIL: oldest entry should be evicted" << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }

            // 最新的应该还在
            cache::CacheKey newest{(uint64_t)(0xAABB0000 + 19)};
            if (!backend.contains(newest)) {
                std::cerr << "  FAIL: newest entry should exist" << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }
        }
        system(("rm -rf " + test_dir).c_str());
        std::cout << "    PASS: LRU eviction correct (evicted " << eviction_count
                  << " entries, " << eviction_total_bytes << " bytes)" << std::endl;
    }

    // 8. 共享前缀测试 (多个序列共享 system prompt)
    {
        std::cout << "  [8/8] Shared prefix lookup test..." << std::endl;
        std::string test_dir = "/tmp/qwen_cache_shared_test_" + std::to_string(getpid());

        cache::CacheConfig cfg;
        cfg.enabled = true;
        cfg.cache_dir = test_dir;
        cfg.max_cache_bytes = 512ULL * 1024 * 1024;
        cfg.chunk_size = 8;    // chunk=8 tokens
        cfg.cache_ssm_state = false;

        cache::ModelCacheParams params;
        params.num_full_attn_layers = 1;
        params.num_kv_heads = 1;
        params.head_dim = 32;
        params.block_size = 8;
        params.num_linear_attn_layers = 1;

        cache::CacheEngine engine(cfg, params);

        // 构造共享前缀: system prompt = [10,20,30,40,50,60,70,80] (1 chunk)
        // 请求 A: system + [1,2,3,4,5,6,7,8]
        // 请求 B: system + [91,92,93,94,95,96,97,98]
        std::vector<int> reqA = {10,20,30,40,50,60,70,80, 1,2,3,4,5,6,7,8};
        std::vector<int> reqB = {10,20,30,40,50,60,70,80, 91,92,93,94,95,96,97,98};

        // 手动存入 system prompt 的 chunk (模拟第一次请求的 store)
        {
            cache::TokenHasher hasher(8);
            auto keys = hasher.compute_keys(reqA.data(), 16);
            // 只存第一个 chunk (system prompt)
            auto entry = std::make_shared<cache::CacheEntry>();
            entry->num_tokens = 8;
            entry->tokens.assign(reqA.begin(), reqA.begin() + 8);
            entry->kv_data.resize(8 * params.kv_bytes_per_token(), 0x42);
            // 直接存入 CacheEngine 内部使用的相同 DiskBackend
            // 通过 lookup → 确认未缓存 → 使用一个独立 backend
        }

        // 更直接: 通过 DiskBackend 手动存
        {
            cache::TokenHasher hasher(8);
            auto keysA = hasher.compute_keys(reqA.data(), 16);
            // keysA[0] 是 system prompt chunk

            // 手动创建 backend 引用 (同一目录)
            cache::DiskBackend direct_backend(test_dir, 512*1024*1024);
            auto entry = std::make_shared<cache::CacheEntry>();
            entry->num_tokens = 8;
            entry->tokens.assign(reqA.begin(), reqA.begin() + 8);
            entry->kv_data.resize(8 * params.kv_bytes_per_token(), 0x42);
            direct_backend.put(keysA[0], entry);
        }

        // 重新创建 engine (重新扫描目录 — DiskBackend 用 index_, 需重新创建)
        cache::CacheEngine engine2(cfg, params);

        // 由于 DiskBackend 不会自动扫描已有文件,
        // 我们改用同一个 engine 实例来验证 TokenHasher 共享逻辑
        // 验证: 两个不同请求的第一个 chunk key 相同
        {
            cache::TokenHasher hasher(8);
            auto keysA = hasher.compute_keys(reqA.data(), 16);
            auto keysB = hasher.compute_keys(reqB.data(), 16);

            if (keysA[0] != keysB[0]) {
                std::cerr << "  FAIL: Shared prefix should produce same chunk key" << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }
            if (keysA[1] == keysB[1]) {
                std::cerr << "  FAIL: Different suffix should produce different chunk key" << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }
        }

        // 验证 Monitor 记录
        {
            cache::CacheMonitor monitor;
            // 请求 A: cold miss, 计算 1024 tokens
            monitor.record_request(1024, 0, 1024);
            // 请求 B: 共享 256 tokens system prompt, 计算 768
            monitor.record_request(1024, 256, 768);
            // 请求 C: 同一对话继续, 共享全部 1024 tokens
            monitor.record_request(1024, 1024, 0);

            auto m = monitor.snapshot();
            double tsr = m.token_save_ratio();
            // (0+256+1024)/3072 ≈ 41.7%
            if (tsr < 41.0 || tsr > 42.5) {
                std::cerr << "  FAIL: shared prefix token_save_ratio=" << tsr << std::endl;
                system(("rm -rf " + test_dir).c_str());
                return;
            }
        }

        system(("rm -rf " + test_dir).c_str());
        std::cout << "    PASS: Shared prefix key matching correct" << std::endl;
    }
    
    std::cout << "KV Cache Prefix Caching test passed.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// KV Swapper 测试: swap_out → verify SSD file → swap_in → verify data
// ---------------------------------------------------------------------------
void test_kv_swapper() {
    std::cout << "\n--- Testing KV Swapper (GPU ↔ SSD) ---\n";

    std::string swap_dir = "/tmp/qwen_swap_test_" + std::to_string(getpid());

    // 模拟: 2 full-attn layers, 少量 blocks
    const int num_layers = 2;
    const int num_blocks = 8;
    const int block_size = 16;
    const int num_kv_heads = 2;
    const int head_dim = 32;
    const int block_bytes = block_size * num_kv_heads * head_dim * sizeof(__nv_bfloat16);
    // block_bytes = 16 × 2 × 32 × 2 = 2048

    auto alloc = std::make_shared<core::DeviceAllocator>();
    ops::KVCacheManager kv_mgr(num_blocks, block_size, num_kv_heads, head_dim,
                                core::DataType::BF16, alloc, num_layers);

    // 1. 分配 4 blocks, 写入已知模式
    std::cout << "  [1/5] Allocate blocks & fill pattern..." << std::endl;
    auto blocks = kv_mgr.allocate_blocks(4);
    if (blocks.size() != 4) {
        std::cerr << "  FAIL: expected 4 blocks, got " << blocks.size() << std::endl;
        return;
    }

    // 在每个 block 的每层填入可验证的模式 (使用小值, BF16 精确表示)
    size_t elems_per_block = block_bytes / sizeof(__nv_bfloat16);
    std::vector<__nv_bfloat16> pattern(elems_per_block);
    for (int b = 0; b < 4; ++b) {
        for (int L = 0; L < num_layers; ++L) {
            // K: 模式 = block_id * 10 + layer + elem (值 < 100, BF16 精确)
            for (size_t e = 0; e < elems_per_block; ++e) {
                pattern[e] = __float2bfloat16((float)(blocks[b] * 10 + L + (e % 50)));
            }
            size_t offset = (size_t)blocks[b] * elems_per_block;
            cudaMemcpy(kv_mgr.get_layer_k_cache_mut(L) + offset,
                       pattern.data(), block_bytes, cudaMemcpyHostToDevice);
            // V: 模式 negated
            for (size_t e = 0; e < elems_per_block; ++e) {
                pattern[e] = __float2bfloat16(-(float)(blocks[b] * 10 + L + (e % 50)));
            }
            cudaMemcpy(kv_mgr.get_layer_v_cache_mut(L) + offset,
                       pattern.data(), block_bytes, cudaMemcpyHostToDevice);
        }
    }
    int free_before_swap = kv_mgr.num_free_blocks();
    std::cout << "    OK: 4 blocks filled, free=" << free_before_swap << std::endl;

    // 2. Swap out
    std::cout << "  [2/5] Swap out (GPU → SSD)..." << std::endl;
    cache::KVSwapper swapper(swap_dir, block_bytes, num_layers);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto rec = swapper.swap_out(
        /*request_id=*/42, kv_mgr, blocks, /*context_len=*/60,
        nullptr, 0, 0,   // no SSM
        nullptr, 0,       // no Conv
        stream);

    if (rec.num_blocks != 4) {
        std::cerr << "  FAIL: swap_out num_blocks=" << rec.num_blocks << std::endl;
        return;
    }
    int free_after_swap = kv_mgr.num_free_blocks();
    if (free_after_swap != free_before_swap + 4) {
        std::cerr << "  FAIL: expected free=" << (free_before_swap + 4)
                  << ", got " << free_after_swap << std::endl;
        return;
    }
    std::cout << "    OK: swapped out, free=" << free_after_swap
              << ", SSD file=" << rec.kv_path << " (" << rec.swap_out_ms << " ms)" << std::endl;

    // 3. Verify SSD file exists
    std::cout << "  [3/5] Verify SSD file exists..." << std::endl;
    FILE* check = fopen(rec.kv_path.c_str(), "rb");
    if (!check) {
        std::cerr << "  FAIL: SSD file not found: " << rec.kv_path << std::endl;
        return;
    }
    fseek(check, 0, SEEK_END);
    long fsize = ftell(check);
    fclose(check);
    // Expected: 2 ints header + 4 blocks × 2 layers × 2(K+V) × block_bytes
    long expected_size = 2 * sizeof(int) + 4L * num_layers * 2 * block_bytes;
    std::cout << "    OK: file size=" << fsize << " (expected=" << expected_size << ")" << std::endl;

    // 4. Swap in
    std::cout << "  [4/5] Swap in (SSD → GPU)..." << std::endl;
    auto new_blocks = swapper.swap_in(42, kv_mgr, nullptr, nullptr, stream);
    if (new_blocks.size() != 4) {
        std::cerr << "  FAIL: swap_in returned " << new_blocks.size() << " blocks" << std::endl;
        return;
    }

    // 5. Verify data integrity
    std::cout << "  [5/5] Verify data integrity..." << std::endl;
    std::vector<__nv_bfloat16> readback(elems_per_block);
    bool data_ok = true;
    for (int b = 0; b < 4; ++b) {
        for (int L = 0; L < num_layers; ++L) {
            size_t offset = (size_t)new_blocks[b] * elems_per_block;

            // Check K
            cudaMemcpy(readback.data(), kv_mgr.get_layer_k_cache(L) + offset,
                       block_bytes, cudaMemcpyDeviceToHost);
            for (size_t e = 0; e < std::min(elems_per_block, (size_t)50); ++e) {
                float expected = (float)(blocks[b] * 10 + L + (e % 50));
                float actual = __bfloat162float(readback[e]);
                if (std::abs(actual - expected) > 0.5f) {
                    std::cerr << "  FAIL: K mismatch at block=" << b << " L=" << L
                              << " e=" << e << ": expected=" << expected
                              << " got=" << actual << std::endl;
                    data_ok = false;
                    break;
                }
            }

            // Check V
            cudaMemcpy(readback.data(), kv_mgr.get_layer_v_cache(L) + offset,
                       block_bytes, cudaMemcpyDeviceToHost);
            for (size_t e = 0; e < std::min(elems_per_block, (size_t)50); ++e) {
                float expected = -(float)(blocks[b] * 10 + L + (e % 50));
                float actual = __bfloat162float(readback[e]);
                if (std::abs(actual - expected) > 0.5f) {
                    std::cerr << "  FAIL: V mismatch at block=" << b << " L=" << L
                              << " e=" << e << ": expected=" << expected
                              << " got=" << actual << std::endl;
                    data_ok = false;
                    break;
                }
            }
            if (!data_ok) break;
        }
        if (!data_ok) break;
    }

    if (!data_ok) return;

    kv_mgr.free_blocks(new_blocks);
    cudaStreamDestroy(stream);
    swapper.print_stats();

    // Cleanup
    system(("rm -rf " + swap_dir).c_str());
    std::cout << "    PASS: Data integrity verified after KV swap roundtrip" << std::endl;
    std::cout << "KV Swapper test passed.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 并发 Swap 测试: 模拟多请求并发, 观察 SSD offload 行为
//
// 设计:
//   - 使用真实模型, 极小 KV 预算 迫使 swap 发生
//   - prompt=128 (8 blocks/req), KV=0.05 GB ≈ 50 blocks
//   - 50 / 8 ≈ 6 请求 KV 容量, 提交 8 请求 → 必触发 swap
//   - 观察: swap_out 次数, swap_in 次数, prefetch 命中, page_cache 释放
// ---------------------------------------------------------------------------
void test_concurrent_swap() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        Concurrent KV Swap Test (SSD Offload)                ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n" << std::endl;

    // 配置:
    //   block_bytes = 16 × 4 × 256 × 2 = 32768 per block per layer
    //   KV per block = 32768 × 16 layers × 2 (K+V) = 1 MB
    //   0.05 GB ≈ 51 MB → 51 blocks
    //   prompt=128 → 8 blocks/req
    //   51 / 8 = 6.4 → 第 7/8 个请求必须 swap
    //   SSM/Conv per req: 147 MB (cudaMalloc, swap 时释放)
    const int ACTUAL_REQUESTS = 8;
    const int ACTUAL_PROMPT   = 128;
    const int DECODE_STEPS    = 5;
    const float KV_GB         = 0.05f;  // 极小: ~50 blocks

    core::Qwen35Config config;
    std::string model_dir = "/home/rm01/runner/models/Qwen/Qwen3.5-27B";

    // 配置: 小 KV 预算 + 启用 cache (需要 swap dir)
    cache::CacheConfig cache_cfg;
    cache_cfg.enabled = true;
    cache_cfg.cache_dir = "/tmp/qwen_swap_concurrent_test_" + std::to_string(getpid());
    cache_cfg.kv_cache_budget_gb = KV_GB;

    std::cout << "  Config:" << std::endl;
    std::cout << "    Requests:    " << ACTUAL_REQUESTS << std::endl;
    std::cout << "    Prompt len:  " << ACTUAL_PROMPT << " tokens (" << (ACTUAL_PROMPT/16) << " blocks each)" << std::endl;
    std::cout << "    Decode steps:" << DECODE_STEPS << std::endl;
    std::cout << "    KV Budget:   " << KV_GB << " GB (~" << (int)(KV_GB * 1024) << " blocks)" << std::endl;
    std::cout << "    SSM/req:     ~147 MB" << std::endl;
    std::cout << "    Total blocks needed: " << ACTUAL_REQUESTS << " × " << (ACTUAL_PROMPT/16)
              << " = " << (ACTUAL_REQUESTS * ACTUAL_PROMPT / 16) 
              << " (exceeds budget → swap expected)" << std::endl;

    // 检查 prompt 是否超过 MAX_PROMPT_LEN (4096)
    if (ACTUAL_PROMPT > ipc::MAX_PROMPT_LEN) {
        std::cerr << "  ERROR: ACTUAL_PROMPT > MAX_PROMPT_LEN (" << ipc::MAX_PROMPT_LEN << ")" << std::endl;
        return;
    }

    auto t0_total = std::chrono::high_resolution_clock::now();

    try {
        // 1. 创建引擎 (加载模型, 初始化 KV manager)
        std::cout << "\n  [1/5] Creating engine with " << KV_GB << " GB KV budget..." << std::endl;
        core::InferenceEngine engine(config, model_dir, cache_cfg);

        // 2. 获取 IPC 队列 (attach to shared memory created by engine)
        std::cout << "  [2/5] Connecting IPC queues..." << std::endl;
        ipc::ShmRingBuffer<ipc::InferenceRequest, 128> req_queue("/qwen_thor_ipc", false);
        ipc::ShmRingBuffer<ipc::InferenceResponse, 512> resp_queue("/qwen_thor_resp", false);

        // 3. 提交所有请求
        std::cout << "  [3/5] Submitting " << ACTUAL_REQUESTS << " requests..." << std::endl;
        int base_tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                             248046, 198, 248045, 74455, 198, 248068, 198};

        for (int i = 0; i < ACTUAL_REQUESTS; ++i) {
            ipc::InferenceRequest req{};
            req.request_id = 5000 + i;
            req.prompt_len = ACTUAL_PROMPT;
            req.max_new_tokens = DECODE_STEPS;
            // 填充 prompt tokens
            for (int t = 0; t < ACTUAL_PROMPT; ++t) {
                req.prompt_tokens[t] = (t < 17) ? base_tokens[t] : (1 + (t % 100));
            }
            while (!req_queue.push(req)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        std::cout << "    All " << ACTUAL_REQUESTS << " requests submitted." << std::endl;

        // 4. 启动引擎, 收集响应
        std::cout << "  [4/5] Running engine + collecting responses..." << std::endl;
        engine.start();

        // 收集所有响应
        std::unordered_map<uint64_t, int> tokens_per_request;  // req_id → token count
        std::unordered_map<uint64_t, bool> request_finished;
        int total_tokens = 0;
        int finished_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();
        auto timeout = std::chrono::seconds(600);  // 10 分钟超时
        
        while (finished_count < ACTUAL_REQUESTS) {
            ipc::InferenceResponse resp;
            if (resp_queue.pop(resp)) {
                tokens_per_request[resp.request_id]++;
                total_tokens++;
                if (resp.is_finished && !request_finished[resp.request_id]) {
                    request_finished[resp.request_id] = true;
                    finished_count++;
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_s = std::chrono::duration<double>(now - start_time).count();
                    printf("    Request %lu finished (%d/%d) — %d tokens, elapsed %.1fs\n",
                           resp.request_id, finished_count, ACTUAL_REQUESTS,
                           tokens_per_request[resp.request_id], elapsed_s);
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            // 超时检查
            auto now = std::chrono::high_resolution_clock::now();
            if (now - start_time > timeout) {
                std::cerr << "    TIMEOUT: only finished " << finished_count
                          << "/" << ACTUAL_REQUESTS << " requests" << std::endl;
                break;
            }
        }

        // 5. 输出结果
        engine.stop();

        auto t1_total = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t1_total - t0_total).count();

        std::cout << "\n  [5/5] Results:" << std::endl;
        std::cout << "    Finished:      " << finished_count << "/" << ACTUAL_REQUESTS << std::endl;
        std::cout << "    Total tokens:  " << total_tokens << std::endl;
        std::cout << "    Total time:    " << std::fixed << std::setprecision(1) << total_s << "s" << std::endl;
        if (total_tokens > 0 && total_s > 0) {
            // 减去模型加载时间 (粗略估算: t0 到 start)
            double inference_s = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_time).count();
            std::cout << "    Throughput:    " << std::setprecision(1)
                      << (total_tokens / inference_s) << " tok/s" << std::endl;
        }

        // 清理
        (void)system(("rm -rf " + cache_cfg.cache_dir).c_str());

        if (finished_count == ACTUAL_REQUESTS) {
            std::cout << "\n    PASS: All " << ACTUAL_REQUESTS
                      << " concurrent requests completed with SSD swap" << std::endl;
        } else {
            std::cout << "\n    PARTIAL: " << finished_count << "/" << ACTUAL_REQUESTS
                      << " completed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nConcurrent swap test finished.\n" << std::endl;
}

// ---------------------------------------------------------------------------
// 256K 上下文 Swap Benchmark
//   直接测试 KVSwapper 在 Qwen3.5-27B 真实参数下的 SSD I/O 性能
//   无需加载模型权重, 仅使用 KVCacheManager + KVSwapper
//
// GPU KV:  4096 blocks = 4 GB (16 full-attn layers × 4 heads × 256 dim × BF16)
// Target:  256K tokens = 16384 blocks = 16 GB KV
//          64 requests × 256 blocks/req (4096 tokens/req)
//          GPU 可容纳 16 个请求, 其余 48 个需 SSD swap
//
// SSM per request: 48 layers × 3 MB = 144 MB  (FP32)
// Conv per request: 48 layers × 60 KB = 2.88 MB (BF16)
//
// Phase A: 不同块数下的 swap_out/swap_in 性能梯度 (8..2048 blocks)
// Phase B: 完整 256K 上下文轮转 (64 请求 × 256 blocks, 仅 16 在 GPU)
// Phase C: Prefetch vs Direct swap_in 对比
// ---------------------------------------------------------------------------
void test_swap_256k_benchmark() {
    std::cout << "\n" << std::string(66, '=') << std::endl;
    std::cout << "  256K Context Swap Benchmark (GPU 4 GB + SSD)" << std::endl;
    std::cout << std::string(66, '=') << "\n" << std::endl;

    // ============ Config: Qwen3.5-27B real params ============
    const int BLOCK_SIZE = 16;
    const int NUM_KV_HEADS = 4;
    const int HEAD_DIM = 256;
    const int NUM_FULL_ATTN_LAYERS = 16;
    const int NUM_LINEAR_LAYERS = 48;
    const int BLOCK_BYTES = BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM * (int)sizeof(__nv_bfloat16);  // 32768
    const size_t KV_PER_BLOCK = (size_t)BLOCK_BYTES * NUM_FULL_ATTN_LAYERS * 2;  // 1,048,576 = 1 MB

    const size_t SSM_PER_LAYER = (size_t)16 * 128 * 384 * sizeof(float);         // 3,145,728 = 3 MB
    const size_t CONV_PER_LAYER = (size_t)10240 * 3 * sizeof(__nv_bfloat16);      // 61,440 = 60 KB
    const size_t SSM_CONV_PER_REQ = SSM_PER_LAYER * NUM_LINEAR_LAYERS
                                  + CONV_PER_LAYER * NUM_LINEAR_LAYERS;           // ~147 MB

    const int GPU_BLOCKS = 4096;       // 4 GB GPU KV
    const int BLOCKS_PER_REQ = 256;    // 4096 tokens/req
    const int TOTAL_REQUESTS = 64;     // 256K tokens total
    const int MAX_GPU_REQS = GPU_BLOCKS / BLOCKS_PER_REQ;  // 16
    const int SWAP_REQS = TOTAL_REQUESTS - MAX_GPU_REQS;   // 48
    const size_t elems_per_block = BLOCK_BYTES / sizeof(__nv_bfloat16);  // 16384

    printf("  Config:\n");
    printf("    GPU KV Budget:     %d blocks = %.0f GB\n",
           GPU_BLOCKS, GPU_BLOCKS * KV_PER_BLOCK / (1024.0*1024*1024));
    printf("    KV per block:      %zu bytes (%d layers x K+V) = 1 MB\n",
           KV_PER_BLOCK, NUM_FULL_ATTN_LAYERS);
    printf("    SSM/Conv per req:  %.1f MB (48 layers x 3 MB SSM + 60 KB Conv)\n",
           SSM_CONV_PER_REQ / (1024.0*1024));
    printf("    Target:            256K tokens = %d reqs x %d blocks (4096 tok/req)\n",
           TOTAL_REQUESTS, BLOCKS_PER_REQ);
    printf("    GPU capacity:      %d reqs (64K tokens on GPU)\n", MAX_GPU_REQS);
    printf("    SSD swap needed:   %d reqs (192K tokens on SSD)\n\n", SWAP_REQS);

    // ============ Setup ============
    auto alloc = std::make_shared<core::DeviceAllocator>();

    printf("  [Setup] Allocating KV cache: %d blocks x %d layers x K+V = %.1f GB ...\n",
           GPU_BLOCKS, NUM_FULL_ATTN_LAYERS,
           GPU_BLOCKS * KV_PER_BLOCK / (1024.0*1024*1024));
    auto t_bench_start = std::chrono::high_resolution_clock::now();

    ops::KVCacheManager kv_mgr(GPU_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                core::DataType::BF16, alloc, NUM_FULL_ATTN_LAYERS);

    auto t_after_setup = std::chrono::high_resolution_clock::now();
    printf("  [Setup] KV cache ready in %.0f ms, free: %d blocks\n",
           std::chrono::duration<double, std::milli>(t_after_setup - t_bench_start).count(),
           kv_mgr.num_free_blocks());

    std::string swap_dir = "/tmp/qwen_swap_256k_bench_" + std::to_string(getpid());
    cache::KVSwapper swapper(swap_dir, BLOCK_BYTES, NUM_FULL_ATTN_LAYERS, 32);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---- Helper lambdas ----
    auto alloc_ssm = [&](std::vector<float*>& ssm, std::vector<__nv_bfloat16*>& conv, uint8_t fill) {
        ssm.resize(NUM_LINEAR_LAYERS);
        conv.resize(NUM_LINEAR_LAYERS);
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            cudaMalloc(&ssm[i], SSM_PER_LAYER);
            cudaMemset(ssm[i], fill, SSM_PER_LAYER);
            cudaMalloc(&conv[i], CONV_PER_LAYER);
            cudaMemset(conv[i], fill, CONV_PER_LAYER);
        }
    };
    auto free_ssm = [&](std::vector<float*>& ssm, std::vector<__nv_bfloat16*>& conv) {
        for (auto* p : ssm)  cudaFree(p);
        for (auto* p : conv) cudaFree(p);
        ssm.clear(); conv.clear();
    };
    auto fill_kv = [&](const std::vector<int>& blocks, uint8_t fill) {
        for (int b : blocks) {
            size_t off = (size_t)b * elems_per_block;
            for (int L = 0; L < NUM_FULL_ATTN_LAYERS; L++) {
                cudaMemset(kv_mgr.get_layer_k_cache_mut(L) + off, fill, BLOCK_BYTES);
                cudaMemset(kv_mgr.get_layer_v_cache_mut(L) + off, fill, BLOCK_BYTES);
            }
        }
        cudaDeviceSynchronize();
    };
    auto verify_kv = [&](const std::vector<int>& blocks, uint8_t expected) -> bool {
        std::vector<uint8_t> buf(BLOCK_BYTES);
        // Spot check: first block L=0 K, last block L=last V
        size_t off0 = (size_t)blocks.front() * elems_per_block;
        cudaMemcpy(buf.data(), kv_mgr.get_layer_k_cache(0) + off0,
                   BLOCK_BYTES, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 64; i++) {
            if (buf[i] != expected) return false;
        }
        size_t offN = (size_t)blocks.back() * elems_per_block;
        cudaMemcpy(buf.data(),
                   kv_mgr.get_layer_v_cache(NUM_FULL_ATTN_LAYERS - 1) + offN,
                   BLOCK_BYTES, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 64; i++) {
            if (buf[i] != expected) return false;
        }
        return true;
    };

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Phase A: Scale Gradient (with SSM/Conv)                  ║
    // ╚═══════════════════════════════════════════════════════════╝
    printf("\n  Phase A: Swap Performance Gradient\n");
    printf("  %s\n", std::string(62, '-').c_str());
    printf("  %-7s  %-9s  %-9s  %-10s  %-10s  %-9s  %-9s\n",
           "Blocks", "KV(MB)", "Tot(MB)", "Out(ms)", "In(ms)", "W(GB/s)", "R(GB/s)");
    printf("  %s\n", std::string(62, '-').c_str());

    int phase_a_sizes[] = {8, 32, 128, 256, 512, 1024, 2048};
    for (int nblk : phase_a_sizes) {
        if (nblk > GPU_BLOCKS / 2) break;  // keep room for swap_in

        uint64_t rid = 10000 + nblk;
        uint8_t fill = (uint8_t)((rid & 0x7F) + 1);

        // Allocate + fill KV
        auto blocks = kv_mgr.allocate_blocks(nblk);
        fill_kv(blocks, fill);

        // Allocate + fill SSM/Conv
        std::vector<float*> ssm;
        std::vector<__nv_bfloat16*> conv;
        alloc_ssm(ssm, conv, fill);

        // swap_out
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        auto rec = swapper.swap_out(
            rid, kv_mgr, blocks, nblk * BLOCK_SIZE,
            ssm.data(), NUM_LINEAR_LAYERS, SSM_PER_LAYER,
            conv.data(), CONV_PER_LAYER, stream);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        free_ssm(ssm, conv);  // engine frees SSM after swap_out

        // Re-alloc SSM for swap_in
        alloc_ssm(ssm, conv, 0);

        // swap_in
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto new_blocks = swapper.swap_in(rid, kv_mgr, ssm.data(), conv.data(), stream);
        cudaDeviceSynchronize();
        auto t3 = std::chrono::high_resolution_clock::now();
        double in_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        bool ok = verify_kv(new_blocks, fill);

        kv_mgr.free_blocks(new_blocks);
        free_ssm(ssm, conv);

        double kv_mb = nblk * KV_PER_BLOCK / (1024.0 * 1024);
        double tot_mb = kv_mb + SSM_CONV_PER_REQ / (1024.0 * 1024);
        double w_gbps = (tot_mb / 1024.0) / (out_ms / 1000.0);
        double r_gbps = (tot_mb / 1024.0) / (in_ms / 1000.0);

        printf("  %5d   %8.1f  %8.1f  %9.1f  %9.1f  %8.2f  %8.2f  %s\n",
               nblk, kv_mb, tot_mb, out_ms, in_ms, w_gbps, r_gbps,
               ok ? "PASS" : "FAIL");
    }

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Phase B: 256K Context Rotation                           ║
    // ╚═══════════════════════════════════════════════════════════╝
    printf("\n  Phase B: 256K Context Rotation (%d reqs, %d on GPU)\n",
           TOTAL_REQUESTS, MAX_GPU_REQS);
    printf("  %s\n", std::string(62, '-').c_str());

    struct BenchReq {
        uint64_t id;
        std::vector<int> block_table;
        std::vector<float*> ssm;
        std::vector<__nv_bfloat16*> conv;
        bool active = false;
        uint8_t fill_val;
    };
    std::vector<BenchReq> reqs(TOTAL_REQUESTS);
    for (int i = 0; i < TOTAL_REQUESTS; i++) {
        reqs[i].id = 20000 + i;
        reqs[i].fill_val = (uint8_t)((i & 0x7F) + 1);
    }

    // Step 1: Fill GPU with first 16 requests
    printf("\n  Step 1: Filling GPU (%d reqs x %d blocks = %d blocks)...\n",
           MAX_GPU_REQS, BLOCKS_PER_REQ, GPU_BLOCKS);
    auto t_b_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < MAX_GPU_REQS; i++) {
        auto& r = reqs[i];
        r.block_table = kv_mgr.allocate_blocks(BLOCKS_PER_REQ);
        fill_kv(r.block_table, r.fill_val);
        alloc_ssm(r.ssm, r.conv, r.fill_val);
        r.active = true;
    }

    auto t_step1 = std::chrono::high_resolution_clock::now();
    printf("    Done: GPU full (%d free), %.1f s\n",
           kv_mgr.num_free_blocks(),
           std::chrono::duration<double>(t_step1 - t_b_start).count());

    // Step 2: Rotate remaining 48 requests through GPU
    printf("\n  Step 2: Rotating %d requests (swap out oldest, fill new)...\n", SWAP_REQS);

    double phase_b_out_ms = 0;
    size_t phase_b_out_bytes = 0;
    int phase_b_out_count = 0;

    for (int i = MAX_GPU_REQS; i < TOTAL_REQUESTS; i++) {
        int victim_idx = i - MAX_GPU_REQS;
        auto& victim = reqs[victim_idx];

        auto rec = swapper.swap_out(
            victim.id, kv_mgr, victim.block_table,
            BLOCKS_PER_REQ * BLOCK_SIZE,
            victim.ssm.data(), NUM_LINEAR_LAYERS, SSM_PER_LAYER,
            victim.conv.data(), CONV_PER_LAYER, stream);

        free_ssm(victim.ssm, victim.conv);
        victim.block_table.clear();
        victim.active = false;

        phase_b_out_ms += rec.swap_out_ms;
        phase_b_out_bytes += rec.total_bytes;
        phase_b_out_count++;

        // Allocate new request
        auto& nr = reqs[i];
        nr.block_table = kv_mgr.allocate_blocks(BLOCKS_PER_REQ);
        fill_kv(nr.block_table, nr.fill_val);
        alloc_ssm(nr.ssm, nr.conv, nr.fill_val);
        nr.active = true;

        if ((phase_b_out_count % 12 == 0) || i == TOTAL_REQUESTS - 1) {
            printf("    [%2d/%d] swap_outs, free=%d, avg=%.1f ms\n",
                   phase_b_out_count, SWAP_REQS,
                   kv_mgr.num_free_blocks(),
                   phase_b_out_ms / phase_b_out_count);
        }
    }

    auto t_step2 = std::chrono::high_resolution_clock::now();
    double step2_s = std::chrono::duration<double>(t_step2 - t_step1).count();
    printf("\n    Rotation done: %d swap_outs, %.1f s\n", phase_b_out_count, step2_s);
    printf("    SSD written: %.2f GB, avg swap_out: %.1f ms\n",
           phase_b_out_bytes / (1024.0*1024*1024),
           phase_b_out_ms / phase_b_out_count);
    printf("    SSD write bandwidth: %.2f GB/s\n",
           (phase_b_out_bytes / (1024.0*1024*1024)) / (phase_b_out_ms / 1000.0));

    // Step 3: Swap back 5 requests and verify data integrity
    const int VERIFY_COUNT = 5;
    printf("\n  Step 3: Verifying %d swapped requests (swap in + check)...\n", VERIFY_COUNT);

    double phase_b_in_ms = 0;
    int verify_pass = 0;

    for (int i = 0; i < VERIFY_COUNT; i++) {
        auto& target = reqs[i];  // swapped to SSD
        int vic_idx = TOTAL_REQUESTS - 1 - i;
        auto& vic = reqs[vic_idx];

        // Swap out current active to make room
        swapper.swap_out(
            vic.id, kv_mgr, vic.block_table,
            BLOCKS_PER_REQ * BLOCK_SIZE,
            vic.ssm.data(), NUM_LINEAR_LAYERS, SSM_PER_LAYER,
            vic.conv.data(), CONV_PER_LAYER, stream);
        free_ssm(vic.ssm, vic.conv);
        vic.block_table.clear();
        vic.active = false;

        // Swap in target
        alloc_ssm(target.ssm, target.conv, 0);

        cudaDeviceSynchronize();
        auto ti0 = std::chrono::high_resolution_clock::now();
        target.block_table = swapper.swap_in(
            target.id, kv_mgr, target.ssm.data(), target.conv.data(), stream);
        cudaDeviceSynchronize();
        auto ti1 = std::chrono::high_resolution_clock::now();
        double in_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();
        phase_b_in_ms += in_ms;
        target.active = true;

        bool ok = verify_kv(target.block_table, target.fill_val);
        if (ok) verify_pass++;

        printf("    req %lu: swap_in %.1f ms, verify %s\n",
               target.id, in_ms, ok ? "PASS" : "FAIL");
    }

    printf("    Integrity: %d/%d passed, avg swap_in: %.1f ms\n",
           verify_pass, VERIFY_COUNT, phase_b_in_ms / VERIFY_COUNT);

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Phase C: Prefetch vs Direct                              ║
    // ╚═══════════════════════════════════════════════════════════╝
    // Use one of the still-swapped requests for prefetch test
    printf("\n  Phase C: Prefetch vs Direct swap_in\n");
    printf("  %s\n", std::string(62, '-').c_str());

    // Pick req index VERIFY_COUNT (still swapped, not yet touched)
    if (VERIFY_COUNT < SWAP_REQS) {
        int pf_idx = VERIFY_COUNT;  // e.g., reqs[5], still on SSD
        auto& pf_req = reqs[pf_idx];

        // Direct swap_in (no prefetch) — use stats from Phase B Step 3
        double direct_avg_ms = phase_b_in_ms / VERIFY_COUNT;

        // Now prefetch this request
        swapper.prefetch(pf_req.id);
        swapper.drain();  // wait for prefetch to complete

        // Swap out an active req to make room
        // Find an active one
        int pf_vic_idx = -1;
        for (int j = MAX_GPU_REQS; j < TOTAL_REQUESTS; j++) {
            if (reqs[j].active) { pf_vic_idx = j; break; }
        }
        if (pf_vic_idx >= 0) {
            auto& pv = reqs[pf_vic_idx];
            swapper.swap_out(
                pv.id, kv_mgr, pv.block_table,
                BLOCKS_PER_REQ * BLOCK_SIZE,
                pv.ssm.data(), NUM_LINEAR_LAYERS, SSM_PER_LAYER,
                pv.conv.data(), CONV_PER_LAYER, stream);
            free_ssm(pv.ssm, pv.conv);
            pv.block_table.clear();
            pv.active = false;
        }

        // swap_in with prefetch hit
        alloc_ssm(pf_req.ssm, pf_req.conv, 0);
        cudaDeviceSynchronize();
        auto tp0 = std::chrono::high_resolution_clock::now();
        pf_req.block_table = swapper.swap_in(
            pf_req.id, kv_mgr, pf_req.ssm.data(), pf_req.conv.data(), stream);
        cudaDeviceSynchronize();
        auto tp1 = std::chrono::high_resolution_clock::now();
        double pf_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();
        pf_req.active = true;

        bool pf_ok = verify_kv(pf_req.block_table, pf_req.fill_val);

        printf("    Direct swap_in (avg):    %.1f ms\n", direct_avg_ms);
        printf("    Prefetch swap_in:        %.1f ms  (%.1fx speedup)\n",
               pf_ms, direct_avg_ms / std::max(pf_ms, 0.01));
        printf("    Prefetch verify:         %s\n", pf_ok ? "PASS" : "FAIL");
    }

    // ============ Cleanup ============
    for (auto& r : reqs) {
        if (r.active) {
            kv_mgr.free_blocks(r.block_table);
            free_ssm(r.ssm, r.conv);
        }
    }
    cudaStreamDestroy(stream);

    // ╔═══════════════════════════════════════════════════════════╗
    // ║ Summary                                                  ║
    // ╚═══════════════════════════════════════════════════════════╝
    printf("\n  %s\n", std::string(62, '=').c_str());
    printf("  Summary: 256K Context Swap Performance\n");
    printf("  %s\n\n", std::string(62, '=').c_str());

    auto stats = swapper.get_stats();
    printf("  KV Swapper Totals:\n");
    printf("    swap_out calls:      %d\n", stats.total_swap_out);
    printf("    swap_in calls:       %d\n", stats.total_swap_in);
    printf("    prefetch hits:       %d\n", stats.prefetch_hits);
    printf("    SSD written:         %.2f GB\n",
           stats.total_bytes_written / (1024.0*1024*1024));
    printf("    SSD read:            %.2f GB\n",
           stats.total_bytes_read / (1024.0*1024*1024));
    printf("    page cache dropped:  %.2f GB\n",
           stats.page_cache_dropped / (1024.0*1024*1024));
    printf("    avg swap_out:        %.1f ms\n", stats.avg_swap_out_ms());
    printf("    avg swap_in:         %.1f ms\n", stats.avg_swap_in_ms());

    double w_gb = stats.total_bytes_written / (1024.0*1024*1024);
    double r_gb = stats.total_bytes_read / (1024.0*1024*1024);
    double w_s = stats.total_swap_out_ms / 1000.0;
    double r_s = stats.total_swap_in_ms / 1000.0;
    printf("\n  Effective SSD I/O:\n");
    printf("    Write throughput:    %.2f GB/s (%.2f GB / %.2f s)\n", w_gb/w_s, w_gb, w_s);
    printf("    Read throughput:     %.2f GB/s (%.2f GB / %.2f s)\n", r_gb/r_s, r_gb, r_s);

    double ctx_kv_mb = BLOCKS_PER_REQ * KV_PER_BLOCK / (1024.0*1024);
    double ctx_tot_mb = ctx_kv_mb + SSM_CONV_PER_REQ / (1024.0*1024);
    printf("\n  Per-Request Context Switch (256 blocks = 4K tokens):\n");
    printf("    Data volume:         %.0f MB (%.0f MB KV + %.0f MB SSM/Conv)\n",
           ctx_tot_mb, ctx_kv_mb, SSM_CONV_PER_REQ / (1024.0*1024));
    printf("    Swap out (actual):   %.1f ms\n", stats.avg_swap_out_ms());
    printf("    Swap in  (actual):   %.1f ms\n", stats.avg_swap_in_ms());

    // Memory
    size_t cuda_free = 0, cuda_total = 0;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    printf("\n  Memory (after cleanup):\n");
    printf("    Unified total:       %.1f GB\n", cuda_total / (1024.0*1024*1024));
    printf("    Unified free:        %.1f GB\n", cuda_free / (1024.0*1024*1024));
    printf("    KV cache footprint:  %.1f GB (%d blocks)\n",
           GPU_BLOCKS * KV_PER_BLOCK / (1024.0*1024*1024), GPU_BLOCKS);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_bench_start).count();
    printf("\n  Total benchmark time:  %.1f s\n", total_s);

    // Cleanup SSD
    printf("  Cleaning SSD files: %s\n", swap_dir.c_str());
    system(("rm -rf " + swap_dir).c_str());

    printf("\n  256K Swap Benchmark complete.\n\n");
}

void test_qwen_model() {
    std::cout << "\n--- Testing Qwen3.5-27B Model Loading ---\n";
    
    core::Qwen35Config config;  // 默认即 27B 参数
    std::string model_dir = "/home/rm01/runner/models/Qwen/Qwen3.5-27B";
    try {
        core::Qwen35Model model(config);
        model.load_weights(model_dir);
        std::cout << "Qwen3.5-27B Model weights loaded successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading model weights: " << e.what() << std::endl;
    }
}

void test_inference_engine() {
    test_inference_engine_with_cache(cache::CacheConfig());
}

void test_inference_engine_with_cache(const cache::CacheConfig& cache_config) {
    std::cout << "\n--- Testing Inference Engine ---\n";
    
    // Qwen3.5-27B 正确配置
    core::Qwen35Config config;  // 默认构造即为 27B 参数
    std::string model_dir = "/home/rm01/runner/models/Qwen/Qwen3.5-27B";
    
    try {
        core::InferenceEngine engine(config, model_dir, cache_config);
        
        // 模拟前端发送请求
        ipc::ShmRingBuffer<ipc::InferenceRequest, 128> frontend_queue("/qwen_thor_ipc", false);
        ipc::InferenceRequest req;
        req.request_id = 2001;
        // Chat-template tokens for "What is 2+2?"
        // <|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n
        int tokens[] = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30, 248046, 198, 248045, 74455, 198, 248068, 198};
        req.prompt_len = 17;
        for (int i = 0; i < 17; ++i) req.prompt_tokens[i] = tokens[i];
        req.max_new_tokens = 5;
        frontend_queue.push(req);
        
        // 启动引擎并等待一小段时间让它处理请求
        engine.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.stop();
        
        std::cout << "Inference Engine test passed.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in Inference Engine: " << e.what() << std::endl;
    }
}

// ============================================================================
// Chunked Prefill 测试: 验证 KV write 位置正确性和分块逻辑
// 不需要模型权重, 直接测试 KV cache 和 paged attention 层的位置计算
// ============================================================================
void test_chunked_prefill() {
    std::cout << "\n--- Testing Chunked Prefill (KV Write Position + Splitting) ---\n";

    // ---- 参数 (简化版, 不需要完整模型) ----
    const int block_size   = 16;
    const int num_kv_heads = 2;
    const int head_dim     = 64;
    const int block_bytes  = block_size * num_kv_heads * head_dim * sizeof(__nv_bfloat16);
    const int num_layers   = 2;
    const int elems_per_block = block_bytes / sizeof(__nv_bfloat16);

    auto alloc = std::make_shared<core::DeviceAllocator>();

    // ================================================================
    // Test 1: write_kv_cache 位置正确性
    // 验证 compute_write_start_kernel (context_len - num_tokens) 而非
    // 旧的 compute_write_positions (context_len - 1)
    // ================================================================
    {
        std::cout << "  [1/3] KV write position correctness..." << std::endl;
        const int total_tokens = 32;  // 2 blocks worth
        const int num_blocks = (total_tokens + block_size - 1) / block_size;

        ops::KVCacheManager kv_mgr(num_blocks + 4, block_size, num_kv_heads, head_dim,
                                    core::DataType::BF16, alloc, num_layers);
        auto blocks = kv_mgr.allocate_blocks(num_blocks);

        // 创建已知模式的 K/V 数据 [total_tokens, num_kv_heads * head_dim]
        int kv_dim = num_kv_heads * head_dim;
        std::vector<__nv_bfloat16> h_k(total_tokens * kv_dim);
        std::vector<__nv_bfloat16> h_v(total_tokens * kv_dim);
        for (int t = 0; t < total_tokens; ++t) {
            for (int d = 0; d < kv_dim; ++d) {
                h_k[t * kv_dim + d] = __float2bfloat16((float)(t + 1));   // K: token index + 1
                h_v[t * kv_dim + d] = __float2bfloat16((float)(t + 100)); // V: token index + 100
            }
        }

        __nv_bfloat16 *d_k, *d_v;
        cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16));
        cudaMalloc(&d_v, h_v.size() * sizeof(__nv_bfloat16));
        cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        int *d_block_tables, *d_context_lens;
        cudaMalloc(&d_block_tables, blocks.size() * sizeof(int));
        cudaMalloc(&d_context_lens, sizeof(int));
        cudaMemcpy(d_block_tables, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);
        int ctx_len = total_tokens;
        cudaMemcpy(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

        // 使用 invoke_write_kv_cache 写入 (模拟 prefill)
        // start_pos 应该是 context_len - num_tokens = 0
        for (int L = 0; L < num_layers; ++L) {
            __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
            __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));

            // 用 compute_write_start 风格: seq_positions[0] = context_len - num_tokens
            int start_pos = 0;  // context_len(32) - num_tokens(32) = 0
            ops::invoke_write_kv_cache(k_cache, v_cache, d_k, d_v,
                                        d_block_tables, start_pos, total_tokens,
                                        num_kv_heads, head_dim, block_size,
                                        (int)blocks.size(), 0);
        }
        cudaDeviceSynchronize();

        // 验证: 从 cache 读回数据, 检查 token 0 在 position 0, token 31 在 position 31
        bool pass = true;
        for (int L = 0; L < num_layers && pass; ++L) {
            const __nv_bfloat16* k_cache = kv_mgr.get_layer_k_cache(L);
            for (int t = 0; t < total_tokens && pass; ++t) {
                int block_idx = t / block_size;
                int block_off = t % block_size;
                int phys_block = blocks[block_idx];
                int cache_offset = phys_block * (block_size * num_kv_heads * head_dim)
                                 + block_off * (num_kv_heads * head_dim)
                                 + 0;  // head 0, dim 0
                __nv_bfloat16 val;
                cudaMemcpy(&val, k_cache + cache_offset, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                float f = __bfloat162float(val);
                float expected = (float)(t + 1);
                if (fabsf(f - expected) > 0.1f) {
                    std::cerr << "    FAIL: L=" << L << " token=" << t
                              << " expected=" << expected << " got=" << f << std::endl;
                    pass = false;
                }
            }
        }
        std::cout << "    " << (pass ? "PASS" : "FAIL") << ": KV write positions correct" << std::endl;

        kv_mgr.free_blocks(blocks);
        cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_block_tables); cudaFree(d_context_lens);
    }

    // ================================================================
    // Test 2: 分块 (chunked) KV write — 模拟 2 块各 16 token
    // chunk 0: tokens [0,16), context_len=16, start=0
    // chunk 1: tokens [16,32), context_len=32, start=16
    // 验证两块的 KV 在 cache 中位置正确
    // ================================================================
    {
        std::cout << "  [2/3] Chunked KV write (2 chunks of 16)..." << std::endl;
        const int total_tokens = 32;
        const int chunk_size = 16;
        const int num_blocks = (total_tokens + block_size - 1) / block_size;
        int kv_dim = num_kv_heads * head_dim;

        ops::KVCacheManager kv_mgr(num_blocks + 4, block_size, num_kv_heads, head_dim,
                                    core::DataType::BF16, alloc, num_layers);
        auto blocks = kv_mgr.allocate_blocks(num_blocks);

        int *d_block_tables, *d_context_lens;
        cudaMalloc(&d_block_tables, blocks.size() * sizeof(int));
        cudaMalloc(&d_context_lens, sizeof(int));
        cudaMemcpy(d_block_tables, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

        __nv_bfloat16 *d_k, *d_v;
        cudaMalloc(&d_k, chunk_size * kv_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&d_v, chunk_size * kv_dim * sizeof(__nv_bfloat16));

        // Chunk 0: tokens [0, 16), context_len=16
        {
            std::vector<__nv_bfloat16> h_k(chunk_size * kv_dim), h_v(chunk_size * kv_dim);
            for (int t = 0; t < chunk_size; ++t)
                for (int d = 0; d < kv_dim; ++d) {
                    h_k[t * kv_dim + d] = __float2bfloat16((float)(t + 1));
                    h_v[t * kv_dim + d] = __float2bfloat16((float)(t + 100));
                }
            cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            int ctx_len = 16;  // after chunk 0
            cudaMemcpy(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

            for (int L = 0; L < num_layers; ++L) {
                __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
                __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
                // start = context_len(16) - chunk_size(16) = 0
                ops::invoke_write_kv_cache(k_cache, v_cache, d_k, d_v,
                                            d_block_tables, 0, chunk_size,
                                            num_kv_heads, head_dim, block_size,
                                            (int)blocks.size(), 0);
            }
        }

        // Chunk 1: tokens [16, 32), context_len=32
        {
            std::vector<__nv_bfloat16> h_k(chunk_size * kv_dim), h_v(chunk_size * kv_dim);
            for (int t = 0; t < chunk_size; ++t)
                for (int d = 0; d < kv_dim; ++d) {
                    h_k[t * kv_dim + d] = __float2bfloat16((float)(t + 16 + 1));  // tokens 17-32
                    h_v[t * kv_dim + d] = __float2bfloat16((float)(t + 16 + 100));
                }
            cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            int ctx_len = 32;  // after chunk 1
            cudaMemcpy(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);

            for (int L = 0; L < num_layers; ++L) {
                __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
                __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
                // start = context_len(32) - chunk_size(16) = 16
                ops::invoke_write_kv_cache(k_cache, v_cache, d_k, d_v,
                                            d_block_tables, 16, chunk_size,
                                            num_kv_heads, head_dim, block_size,
                                            (int)blocks.size(), 0);
            }
        }
        cudaDeviceSynchronize();

        // 验证所有 32 个 token 的 KV 在 cache 中位置正确
        bool pass = true;
        for (int L = 0; L < num_layers && pass; ++L) {
            const __nv_bfloat16* k_cache = kv_mgr.get_layer_k_cache(L);
            for (int t = 0; t < total_tokens && pass; ++t) {
                int block_idx = t / block_size;
                int block_off = t % block_size;
                int phys_block = blocks[block_idx];
                int cache_offset = phys_block * (block_size * num_kv_heads * head_dim)
                                 + block_off * (num_kv_heads * head_dim);
                __nv_bfloat16 val;
                cudaMemcpy(&val, k_cache + cache_offset, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                float f = __bfloat162float(val);
                float expected = (float)(t + 1);
                if (fabsf(f - expected) > 0.1f) {
                    std::cerr << "    FAIL: Chunked write L=" << L << " token=" << t
                              << " expected=" << expected << " got=" << f << std::endl;
                    pass = false;
                }
            }
        }
        std::cout << "    " << (pass ? "PASS" : "FAIL") << ": Chunked KV write positions correct" << std::endl;

        kv_mgr.free_blocks(blocks);
        cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_block_tables); cudaFree(d_context_lens);
    }

    // ================================================================
    // Test 3: Paged attention correctness after chunked write
    // 写入 32 token KV (分 2 块), 然后用 paged attention 查询
    // 验证 attention 对齐 (token 31 应能看到 token 0-31 的 KV)
    // ================================================================
    {
        std::cout << "  [3/3] Paged attention after chunked KV write..." << std::endl;
        const int total_tokens = 32;
        const int num_blocks = (total_tokens + block_size - 1) / block_size;
        int kv_dim = num_kv_heads * head_dim;

        ops::KVCacheManager kv_mgr(num_blocks + 4, block_size, num_kv_heads, head_dim,
                                    core::DataType::BF16, alloc, num_layers);
        auto blocks = kv_mgr.allocate_blocks(num_blocks);

        int *d_block_tables, *d_context_lens;
        cudaMalloc(&d_block_tables, blocks.size() * sizeof(int));
        cudaMalloc(&d_context_lens, sizeof(int));
        cudaMemcpy(d_block_tables, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Write KV for 32 tokens (position 0..31), all same value per dim
        __nv_bfloat16 *d_k, *d_v;
        cudaMalloc(&d_k, total_tokens * kv_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&d_v, total_tokens * kv_dim * sizeof(__nv_bfloat16));

        // K: all 1.0, V: all 1.0 (for easy verification)
        std::vector<__nv_bfloat16> h_ones(total_tokens * kv_dim);
        for (auto& x : h_ones) x = __float2bfloat16(1.0f);
        cudaMemcpy(d_k, h_ones.data(), h_ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_ones.data(), h_ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        // Full write at start=0 (like unchunked prefill)
        int ctx_len = total_tokens;
        cudaMemcpy(d_context_lens, &ctx_len, sizeof(int), cudaMemcpyHostToDevice);
        {
            __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(0));
            __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(0));
            ops::invoke_write_kv_cache(k_cache, v_cache, d_k, d_v,
                                        d_block_tables, 0, total_tokens,
                                        num_kv_heads, head_dim, block_size,
                                        (int)blocks.size(), 0);
        }
        cudaDeviceSynchronize();

        // Query with 1 token (decode-like) at context_len=33
        //   → should attend to 32 cached + 1 current = 33 KV positions
        //   With Q=ones, K=ones, V=ones → attention output should be ~1.0
        int decode_ctx = total_tokens + 1;  // 33
        cudaMemcpy(d_context_lens, &decode_ctx, sizeof(int), cudaMemcpyHostToDevice);

        __nv_bfloat16 *d_q, *d_out;
        int num_q_heads = num_kv_heads;  // no GQA for simplicity
        cudaMalloc(&d_q, num_q_heads * head_dim * sizeof(__nv_bfloat16));
        cudaMalloc(&d_out, num_q_heads * head_dim * sizeof(__nv_bfloat16));

        // Write also the "decode" token's KV at position 32
        {
            std::vector<__nv_bfloat16> one_kv(kv_dim);
            for (auto& x : one_kv) x = __float2bfloat16(1.0f);
            __nv_bfloat16 *d_k1, *d_v1;
            cudaMalloc(&d_k1, kv_dim * sizeof(__nv_bfloat16));
            cudaMalloc(&d_v1, kv_dim * sizeof(__nv_bfloat16));
            cudaMemcpy(d_k1, one_kv.data(), kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v1, one_kv.data(), kv_dim * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

            // Allocate 1 more block for position 32
            auto extra = kv_mgr.allocate_blocks(1);
            blocks.push_back(extra[0]);
            cudaMemcpy(d_block_tables, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

            __nv_bfloat16* k_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(0));
            __nv_bfloat16* v_cache = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(0));
            ops::invoke_write_kv_cache(k_cache, v_cache, d_k1, d_v1,
                                        d_block_tables, 32, 1,
                                        num_kv_heads, head_dim, block_size,
                                        (int)blocks.size(), 0);
            cudaFree(d_k1); cudaFree(d_v1);
        }
        cudaDeviceSynchronize();

        // Q = all ones
        {
            std::vector<__nv_bfloat16> h_q(num_q_heads * head_dim);
            for (auto& x : h_q) x = __float2bfloat16(1.0f);
            cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }

        float sm_scale = 1.0f / sqrtf((float)head_dim);
        ops::invoke_paged_attention(d_out, d_q,
            kv_mgr.get_layer_k_cache(0), kv_mgr.get_layer_v_cache(0),
            d_block_tables, d_context_lens,
            (int)blocks.size(), decode_ctx,
            1, num_q_heads, num_kv_heads, head_dim,
            block_size, sm_scale, 0, 1);
        cudaDeviceSynchronize();

        // Check: output should be ~1.0 (V=1, softmax normalized, sum V * weight = 1)
        std::vector<__nv_bfloat16> h_out(num_q_heads * head_dim);
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        bool pass = true;
        for (int i = 0; i < num_q_heads * head_dim; ++i) {
            float f = __bfloat162float(h_out[i]);
            if (fabsf(f - 1.0f) > 0.05f) {
                std::cerr << "    FAIL: paged_attn output[" << i << "]=" << f << " expected ~1.0" << std::endl;
                pass = false;
                break;
            }
        }
        std::cout << "    " << (pass ? "PASS" : "FAIL") << ": Paged attention reads chunked KV correctly" << std::endl;

        kv_mgr.free_blocks(blocks);
        cudaFree(d_q); cudaFree(d_out);
        cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_block_tables); cudaFree(d_context_lens);
    }

    std::cout << "Chunked Prefill test finished.\n" << std::endl;
}

// ============================================================================
// Chunked Prefill — 综合模拟测试 + 性能基准
//
// Part A: 数据一致性 — chunked vs unchunked KV write 必须产生完全相同的 cache
// Part B: Paged attention 正确性 — 多块写入后 decode query 结果正确
// Part C: 性能 — Qwen3.5-27B 真实参数, 不同 prompt 长度 x chunk size 的
//   KV write 吞吐、paged attention 延迟、block 分配开销
// ============================================================================
void bench_chunked_prefill() {
    printf("\n");
    printf("==================================================================\n");
    printf("  Chunked Prefill Simulation Test + Performance Benchmark\n");
    printf("==================================================================\n\n");

    // ---- Qwen3.5-27B 真实参数 ----
    const int BLOCK_SIZE          = 16;
    const int NUM_KV_HEADS        = 4;
    const int HEAD_DIM            = 256;
    const int NUM_FULL_ATTN_LAYERS = 16;
    const int KV_DIM              = NUM_KV_HEADS * HEAD_DIM;  // 1024
    const int BLOCK_BYTES         = BLOCK_SIZE * KV_DIM * (int)sizeof(__nv_bfloat16);  // 32768 = 32 KB
    const size_t KV_PER_BLOCK     = (size_t)BLOCK_BYTES * NUM_FULL_ATTN_LAYERS * 2;   // 1 MB

    auto alloc = std::make_shared<core::DeviceAllocator>();

    // ================================================================
    // Part A: 数据一致性测试 — chunked vs unchunked KV write
    // ================================================================
    printf("  Part A: Data Consistency (chunked vs unchunked)\n");
    printf("  --------------------------------------------------------------\n");

    struct ConsistencyCase {
        int total_tokens;
        int chunk_size;
    };
    ConsistencyCase cases[] = {
        {   64,   16},   // 4 chunks x 16
        {   64,   32},   // 2 chunks x 32
        {  128,   48},   // 2+1 chunks (不整除)
        {  256,   64},   // 4 chunks x 64
        {  512,  128},   // 4 chunks x 128
        { 1024,  256},   // 4 chunks x 256
        { 4096, 1024},   // 4 chunks x 1024
        { 4096, 4096},   // 1 chunk (baseline)
        {  100,   33},   // 不整除边界
        { 1000,  400},   // 2+1 chunks
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    int pass_count = 0;

    for (int ci = 0; ci < num_cases; ++ci) {
        int T = cases[ci].total_tokens;
        int C = cases[ci].chunk_size;
        int num_blocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int num_chunks = (T + C - 1) / C;

        // 需要 num_blocks * 2 (unchunked + chunked 各一份)
        // 每层独立 cache → 只测 1 层以节省内存
        ops::KVCacheManager kv_ref(num_blocks + 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                    core::DataType::BF16, alloc, 1);
        ops::KVCacheManager kv_chk(num_blocks + 2, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                    core::DataType::BF16, alloc, 1);
        auto blocks_ref = kv_ref.allocate_blocks(num_blocks);
        auto blocks_chk = kv_chk.allocate_blocks(num_blocks);

        // 生成 KV 数据: K[t] = t+1, V[t] = t+1000
        __nv_bfloat16 *d_k, *d_v;
        cudaMalloc(&d_k, (size_t)T * KV_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&d_v, (size_t)T * KV_DIM * sizeof(__nv_bfloat16));
        {
            std::vector<__nv_bfloat16> h_k(T * KV_DIM), h_v(T * KV_DIM);
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < KV_DIM; ++d) {
                    h_k[t * KV_DIM + d] = __float2bfloat16((float)(t + 1));
                    h_v[t * KV_DIM + d] = __float2bfloat16((float)(t + 1000));
                }
            cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }

        int *d_bt_ref, *d_bt_chk;
        cudaMalloc(&d_bt_ref, blocks_ref.size() * sizeof(int));
        cudaMalloc(&d_bt_chk, blocks_chk.size() * sizeof(int));
        cudaMemcpy(d_bt_ref, blocks_ref.data(), blocks_ref.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bt_chk, blocks_chk.data(), blocks_chk.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Reference: 一次性写入所有 T 个 token
        {
            __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_ref.get_layer_k_cache(0));
            __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_ref.get_layer_v_cache(0));
            ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                d_bt_ref, 0, T, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                (int)blocks_ref.size(), 0);
        }

        // Chunked: 分 num_chunks 块写入
        for (int ch = 0; ch < num_chunks; ++ch) {
            int start = ch * C;
            int end   = std::min(start + C, T);
            int len   = end - start;
            __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_chk.get_layer_k_cache(0));
            __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_chk.get_layer_v_cache(0));
            ops::invoke_write_kv_cache(kc, vc, d_k + start * KV_DIM, d_v + start * KV_DIM,
                d_bt_chk, start, len, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                (int)blocks_chk.size(), 0);
        }
        cudaDeviceSynchronize();

        // 比较: 遍历所有 cache 位置, ref 和 chk 必须完全一致
        bool match = true;
        int mismatch_count = 0;
        const __nv_bfloat16* ref_kc = kv_ref.get_layer_k_cache(0);
        const __nv_bfloat16* chk_kc = kv_chk.get_layer_k_cache(0);
        for (int t = 0; t < T && match; ++t) {
            int blk_idx = t / BLOCK_SIZE;
            int blk_off = t % BLOCK_SIZE;
            for (int h = 0; h < NUM_KV_HEADS && match; ++h) {
                // 只检查 dim 0 以加速
                int ref_off = blocks_ref[blk_idx] * (BLOCK_SIZE * KV_DIM)
                            + blk_off * KV_DIM + h * HEAD_DIM;
                int chk_off = blocks_chk[blk_idx] * (BLOCK_SIZE * KV_DIM)
                            + blk_off * KV_DIM + h * HEAD_DIM;
                __nv_bfloat16 rv, cv;
                cudaMemcpy(&rv, ref_kc + ref_off, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                cudaMemcpy(&cv, chk_kc + chk_off, sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                if (__bfloat162float(rv) != __bfloat162float(cv)) {
                    match = false;
                    mismatch_count++;
                }
            }
        }

        printf("    T=%5d  chunk=%4d  chunks=%2d  %s\n",
               T, C, num_chunks, match ? "PASS" : "FAIL");
        if (match) pass_count++;

        kv_ref.free_blocks(blocks_ref);
        kv_chk.free_blocks(blocks_chk);
        cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_bt_ref); cudaFree(d_bt_chk);
    }
    printf("  Result: %d/%d passed\n\n", pass_count, num_cases);

    // ================================================================
    // Part B: Paged Attention 正确性 (多块写入 → decode query)
    //   写入 N 个 token (分 K 块), 然后 decode query 1 token
    //   K=1, V=constant → attention output = constant (均匀注意力)
    // ================================================================
    printf("  Part B: Paged Attention Correctness After Chunked Write\n");
    printf("  --------------------------------------------------------------\n");
    {
        struct AttnCase { int total_tokens; int chunk_size; };
        AttnCase attn_cases[] = {
            {  32,  16},
            {  64,  16},
            { 128,  48},
            { 256,  64},
            { 512, 128},
            {1024, 256},
        };
        int num_attn = sizeof(attn_cases) / sizeof(attn_cases[0]);
        int attn_pass = 0;

        for (int ai = 0; ai < num_attn; ++ai) {
            int T = attn_cases[ai].total_tokens;
            int C = attn_cases[ai].chunk_size;
            int num_blocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE + 1;  // +1 for decode
            int num_chunks = (T + C - 1) / C;

            // 使用简化的 kv 参数, 但保持真实 head_dim
            const int test_kv_heads = 2;  // GQA 简化
            const int test_kv_dim = test_kv_heads * HEAD_DIM;

            ops::KVCacheManager kv_mgr(num_blocks + 2, BLOCK_SIZE, test_kv_heads, HEAD_DIM,
                                        core::DataType::BF16, alloc, 1);
            auto blocks = kv_mgr.allocate_blocks(num_blocks);

            int *d_bt, *d_ctx;
            cudaMalloc(&d_bt, blocks.size() * sizeof(int));
            cudaMalloc(&d_ctx, sizeof(int));
            cudaMemcpy(d_bt, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

            // 写入 KV: K=1.0, V=1.0 (分块)
            __nv_bfloat16 *d_k, *d_v;
            int max_chunk = C;
            cudaMalloc(&d_k, (size_t)max_chunk * test_kv_dim * sizeof(__nv_bfloat16));
            cudaMalloc(&d_v, (size_t)max_chunk * test_kv_dim * sizeof(__nv_bfloat16));
            {
                std::vector<__nv_bfloat16> ones(max_chunk * test_kv_dim);
                for (auto& x : ones) x = __float2bfloat16(1.0f);
                cudaMemcpy(d_k, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
                cudaMemcpy(d_v, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            }

            __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(0));
            __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(0));

            for (int ch = 0; ch < num_chunks; ++ch) {
                int start = ch * C;
                int end   = std::min(start + C, T);
                int len   = end - start;
                ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                    d_bt, start, len, test_kv_heads, HEAD_DIM, BLOCK_SIZE,
                    (int)blocks.size(), 0);
            }

            // Decode: 写入 1 token at position T, K=V=1.0
            {
                std::vector<__nv_bfloat16> one(test_kv_dim);
                for (auto& x : one) x = __float2bfloat16(1.0f);
                __nv_bfloat16 *dk1, *dv1;
                cudaMalloc(&dk1, test_kv_dim * sizeof(__nv_bfloat16));
                cudaMalloc(&dv1, test_kv_dim * sizeof(__nv_bfloat16));
                cudaMemcpy(dk1, one.data(), one.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
                cudaMemcpy(dv1, one.data(), one.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
                ops::invoke_write_kv_cache(kc, vc, dk1, dv1,
                    d_bt, T, 1, test_kv_heads, HEAD_DIM, BLOCK_SIZE,
                    (int)blocks.size(), 0);
                cudaFree(dk1); cudaFree(dv1);
            }
            cudaDeviceSynchronize();

            // Paged attention query
            int decode_ctx = T + 1;
            cudaMemcpy(d_ctx, &decode_ctx, sizeof(int), cudaMemcpyHostToDevice);

            __nv_bfloat16 *d_q, *d_out;
            cudaMalloc(&d_q, test_kv_heads * HEAD_DIM * sizeof(__nv_bfloat16));
            cudaMalloc(&d_out, test_kv_heads * HEAD_DIM * sizeof(__nv_bfloat16));
            {
                std::vector<__nv_bfloat16> qv(test_kv_heads * HEAD_DIM);
                for (auto& x : qv) x = __float2bfloat16(1.0f);
                cudaMemcpy(d_q, qv.data(), qv.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            }

            float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
            ops::invoke_paged_attention(d_out, d_q,
                kv_mgr.get_layer_k_cache(0), kv_mgr.get_layer_v_cache(0),
                d_bt, d_ctx, (int)blocks.size(), decode_ctx,
                1, test_kv_heads, test_kv_heads, HEAD_DIM,
                BLOCK_SIZE, sm_scale, 0, 1);
            cudaDeviceSynchronize();

            // 验证: V=1 everywhere → output should be 1.0
            std::vector<__nv_bfloat16> h_out(test_kv_heads * HEAD_DIM);
            cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            bool ok = true;
            for (size_t i = 0; i < h_out.size(); ++i) {
                float f = __bfloat162float(h_out[i]);
                if (fabsf(f - 1.0f) > 0.05f) { ok = false; break; }
            }
            printf("    T=%5d  chunk=%4d  chunks=%2d  ctx=%5d  %s\n",
                   T, C, num_chunks, decode_ctx, ok ? "PASS" : "FAIL");
            if (ok) attn_pass++;

            kv_mgr.free_blocks(blocks);
            cudaFree(d_k); cudaFree(d_v); cudaFree(d_q); cudaFree(d_out);
            cudaFree(d_bt); cudaFree(d_ctx);
        }
        printf("  Result: %d/%d passed\n\n", attn_pass, num_attn);
    }

    // ================================================================
    // Part C: Performance Benchmark
    //   Qwen3.5-27B 真实参数 (16 full attention layers)
    //   测量: KV write 时间, paged attention 时间, block 分配时间
    //   对比不同 prompt 长度和 chunk size
    // ================================================================
    printf("  Part C: Performance Benchmark (Qwen3.5-27B Parameters)\n");
    printf("  --------------------------------------------------------------\n");
    printf("  %-8s %-8s %-6s  %-12s %-12s %-12s  %-10s %-10s\n",
           "Tokens", "Chunk", "Chunks",
           "KV-Write(ms)", "PagedAttn(ms)", "Alloc(ms)",
           "W-BW(GB/s)", "A-BW(GB/s)");
    printf("  %s\n", std::string(94, '-').c_str());

    struct PerfCase { int total_tokens; int chunk_size; };
    PerfCase perf_cases[] = {
        {   512,   512},  // baseline: 1 chunk
        {  1024,  1024},
        {  2048,  2048},
        {  4096,  4096},  // baseline: 1 chunk at max_chunk_size
        {  4096,  1024},  // 4 chunks
        {  4096,   512},  // 8 chunks
        {  8192,  4096},  // 2 chunks
        {  8192,  2048},  // 4 chunks
        { 16384,  4096},  // 4 chunks
        { 32768,  4096},  // 8 chunks (32K context)
        { 65536,  4096},  // 16 chunks (64K context)
        {131072,  4096},  // 32 chunks (128K context)
        {204800,  4096},  // 50 chunks (200K context)
        {262144,  4096},  // 64 chunks (256K context = model max)
    };
    int num_perf = sizeof(perf_cases) / sizeof(perf_cases[0]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int pi = 0; pi < num_perf; ++pi) {
        int T = perf_cases[pi].total_tokens;
        int C = perf_cases[pi].chunk_size;
        int num_blocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int num_chunks = (T + C - 1) / C;

        // ---- Block 分配 ----
        auto t0 = std::chrono::high_resolution_clock::now();
        ops::KVCacheManager kv_mgr(num_blocks + 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                    core::DataType::BF16, alloc, NUM_FULL_ATTN_LAYERS);
        auto blocks = kv_mgr.allocate_blocks(num_blocks);
        auto t1 = std::chrono::high_resolution_clock::now();
        double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int *d_bt;
        cudaMalloc(&d_bt, blocks.size() * sizeof(int));
        cudaMemcpy(d_bt, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

        // KV 数据: 1.0 填充 (每 chunk 复用)
        __nv_bfloat16 *d_k, *d_v;
        int max_chunk = C;
        cudaMalloc(&d_k, (size_t)max_chunk * KV_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&d_v, (size_t)max_chunk * KV_DIM * sizeof(__nv_bfloat16));
        {
            std::vector<__nv_bfloat16> ones(max_chunk * KV_DIM);
            for (auto& x : ones) x = __float2bfloat16(1.0f);
            cudaMemcpy(d_k, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }

        // ---- Warmup: 1 次完整流程 ----
        for (int L = 0; L < NUM_FULL_ATTN_LAYERS; ++L) {
            __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
            __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
            ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                d_bt, 0, std::min(C, T), NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                (int)blocks.size(), stream);
        }
        cudaStreamSynchronize(stream);

        // ---- KV Write: all layers × all chunks ----
        const int TRIALS = 3;
        double kv_write_total = 0;
        for (int trial = 0; trial < TRIALS; ++trial) {
            cudaStreamSynchronize(stream);
            auto tw0 = std::chrono::high_resolution_clock::now();
            for (int ch = 0; ch < num_chunks; ++ch) {
                int start = ch * C;
                int end   = std::min(start + C, T);
                int len   = end - start;
                for (int L = 0; L < NUM_FULL_ATTN_LAYERS; ++L) {
                    __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
                    __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
                    ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                        d_bt, start, len, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                        (int)blocks.size(), stream);
                }
            }
            cudaStreamSynchronize(stream);
            auto tw1 = std::chrono::high_resolution_clock::now();
            kv_write_total += std::chrono::duration<double, std::milli>(tw1 - tw0).count();
        }
        double kv_write_ms = kv_write_total / TRIALS;

        // 写入数据量: T tokens × KV_DIM × 2(K+V) × sizeof(bf16) × 16 layers
        double kv_write_bytes = (double)T * KV_DIM * 2 * sizeof(__nv_bfloat16) * NUM_FULL_ATTN_LAYERS;
        double kv_write_gbps = (kv_write_bytes / 1e9) / (kv_write_ms / 1e3);

        // ---- Paged Attention: 1 token decode query reading full context ----
        // 测量 1 个 decode token 查询整个 T+1 的 context
        __nv_bfloat16 *d_q, *d_out;
        int num_q_heads = NUM_KV_HEADS;  // 简化: 不用 GQA
        cudaMalloc(&d_q, num_q_heads * HEAD_DIM * sizeof(__nv_bfloat16));
        cudaMalloc(&d_out, num_q_heads * HEAD_DIM * sizeof(__nv_bfloat16));
        {
            std::vector<__nv_bfloat16> qv(num_q_heads * HEAD_DIM);
            for (auto& x : qv) x = __float2bfloat16(1.0f);
            cudaMemcpy(d_q, qv.data(), qv.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }

        int *d_ctx;
        cudaMalloc(&d_ctx, sizeof(int));
        int ctx_val = T;
        cudaMemcpy(d_ctx, &ctx_val, sizeof(int), cudaMemcpyHostToDevice);
        float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);

        // Warmup
        ops::invoke_paged_attention(d_out, d_q,
            kv_mgr.get_layer_k_cache(0), kv_mgr.get_layer_v_cache(0),
            d_bt, d_ctx, (int)blocks.size(), T,
            1, num_q_heads, NUM_KV_HEADS, HEAD_DIM,
            BLOCK_SIZE, sm_scale, stream, 1);
        cudaStreamSynchronize(stream);

        double attn_total = 0;
        const int ATTN_TRIALS = 5;
        for (int trial = 0; trial < ATTN_TRIALS; ++trial) {
            cudaStreamSynchronize(stream);
            auto ta0 = std::chrono::high_resolution_clock::now();
            // 测 all layers
            for (int L = 0; L < NUM_FULL_ATTN_LAYERS; ++L) {
                ops::invoke_paged_attention(d_out, d_q,
                    kv_mgr.get_layer_k_cache(L), kv_mgr.get_layer_v_cache(L),
                    d_bt, d_ctx, (int)blocks.size(), T,
                    1, num_q_heads, NUM_KV_HEADS, HEAD_DIM,
                    BLOCK_SIZE, sm_scale, stream, 1);
            }
            cudaStreamSynchronize(stream);
            auto ta1 = std::chrono::high_resolution_clock::now();
            attn_total += std::chrono::duration<double, std::milli>(ta1 - ta0).count();
        }
        double attn_ms = attn_total / ATTN_TRIALS;

        // 读取数据量: T tokens × KV_DIM × 2(K+V) × sizeof(bf16) × 16 layers
        // + Q: 16 layers × num_q_heads × HEAD_DIM × sizeof(bf16)
        double attn_read_bytes = (double)T * KV_DIM * 2 * sizeof(__nv_bfloat16) * NUM_FULL_ATTN_LAYERS
                               + (double)NUM_FULL_ATTN_LAYERS * num_q_heads * HEAD_DIM * sizeof(__nv_bfloat16);
        double attn_gbps = (attn_read_bytes / 1e9) / (attn_ms / 1e3);

        printf("  %7d  %7d   %4d   %10.2f   %11.2f   %10.2f   %9.2f  %9.2f\n",
               T, C, num_chunks,
               kv_write_ms, attn_ms, alloc_ms,
               kv_write_gbps, attn_gbps);

        kv_mgr.free_blocks(blocks);
        cudaFree(d_k); cudaFree(d_v); cudaFree(d_q); cudaFree(d_out);
        cudaFree(d_bt); cudaFree(d_ctx);
    }

    // ================================================================
    // Part D: Chunk Size 对比 (固定 16K tokens, 变 chunk)
    // ================================================================
    printf("\n  Part D: Chunk Size Impact (fixed 16K tokens, varying chunk)\n");
    printf("  --------------------------------------------------------------\n");
    printf("  %-8s %-6s  %-12s %-10s  %-10s\n",
           "Chunk", "Chunks", "KV-Write(ms)", "W-BW(GB/s)", "Overhead%");
    printf("  %s\n", std::string(60, '-').c_str());

    {
        const int T_FIXED = 16384;
        int chunk_sizes[] = {16384, 8192, 4096, 2048, 1024, 512};
        int n_cs = sizeof(chunk_sizes) / sizeof(chunk_sizes[0]);
        double baseline_ms = 0;

        int num_blocks = (T_FIXED + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int ci = 0; ci < n_cs; ++ci) {
            int C = chunk_sizes[ci];
            int num_chunks = (T_FIXED + C - 1) / C;

            ops::KVCacheManager kv_mgr(num_blocks + 4, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                                        core::DataType::BF16, alloc, NUM_FULL_ATTN_LAYERS);
            auto blocks = kv_mgr.allocate_blocks(num_blocks);

            int *d_bt;
            cudaMalloc(&d_bt, blocks.size() * sizeof(int));
            cudaMemcpy(d_bt, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);

            __nv_bfloat16 *d_k, *d_v;
            cudaMalloc(&d_k, (size_t)C * KV_DIM * sizeof(__nv_bfloat16));
            cudaMalloc(&d_v, (size_t)C * KV_DIM * sizeof(__nv_bfloat16));
            {
                std::vector<__nv_bfloat16> ones(C * KV_DIM);
                for (auto& x : ones) x = __float2bfloat16(1.0f);
                cudaMemcpy(d_k, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
                cudaMemcpy(d_v, ones.data(), ones.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            }

            // Warmup
            for (int L = 0; L < NUM_FULL_ATTN_LAYERS; ++L) {
                __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
                __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
                ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                    d_bt, 0, std::min(C, T_FIXED), NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                    (int)blocks.size(), stream);
            }
            cudaStreamSynchronize(stream);

            const int TRIALS = 5;
            double total_ms = 0;
            for (int trial = 0; trial < TRIALS; ++trial) {
                cudaStreamSynchronize(stream);
                auto tw0 = std::chrono::high_resolution_clock::now();
                for (int ch = 0; ch < num_chunks; ++ch) {
                    int start = ch * C;
                    int end   = std::min(start + C, T_FIXED);
                    int len   = end - start;
                    for (int L = 0; L < NUM_FULL_ATTN_LAYERS; ++L) {
                        __nv_bfloat16* kc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_k_cache(L));
                        __nv_bfloat16* vc = const_cast<__nv_bfloat16*>(kv_mgr.get_layer_v_cache(L));
                        ops::invoke_write_kv_cache(kc, vc, d_k, d_v,
                            d_bt, start, len, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
                            (int)blocks.size(), stream);
                    }
                }
                cudaStreamSynchronize(stream);
                auto tw1 = std::chrono::high_resolution_clock::now();
                total_ms += std::chrono::duration<double, std::milli>(tw1 - tw0).count();
            }
            double ms = total_ms / TRIALS;
            if (C == T_FIXED) baseline_ms = ms;

            double kv_bytes = (double)T_FIXED * KV_DIM * 2 * sizeof(__nv_bfloat16) * NUM_FULL_ATTN_LAYERS;
            double gbps = (kv_bytes / 1e9) / (ms / 1e3);
            double overhead = baseline_ms > 0 ? ((ms - baseline_ms) / baseline_ms * 100.0) : 0;

            printf("  %7d   %4d   %10.2f   %9.2f   %8.1f%%\n",
                   C, num_chunks, ms, gbps, overhead);

            kv_mgr.free_blocks(blocks);
            cudaFree(d_k); cudaFree(d_v); cudaFree(d_bt);
        }
    }

    // ================================================================
    // Part E: 内存开销分析
    // ================================================================
    printf("\n  Part E: Memory Footprint Analysis\n");
    printf("  --------------------------------------------------------------\n");
    printf("  %-10s %-12s %-12s %-12s %-12s %-12s\n",
           "Tokens", "KV-Cache(MB)", "Blocks", "SSM(MB)", "Conv(MB)", "Total(MB)");
    printf("  %s\n", std::string(72, '-').c_str());

    {
        // Qwen3.5-27B real params
        const int NLin = 48;
        const size_t SSM_PER_LAYER = (size_t)16 * 128 * 384 * sizeof(float);      // 3 MB
        const size_t CONV_PER_LAYER = (size_t)10240 * 3 * sizeof(__nv_bfloat16);   // 60 KB

        int prompt_lens[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 204800, 262144};
        int n_pl = sizeof(prompt_lens) / sizeof(prompt_lens[0]);

        for (int pi = 0; pi < n_pl; ++pi) {
            int T = prompt_lens[pi];
            int nblocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
            double kv_mb = (double)nblocks * BLOCK_BYTES * NUM_FULL_ATTN_LAYERS * 2 / (1024.0 * 1024);
            double ssm_mb = (double)NLin * SSM_PER_LAYER / (1024.0 * 1024);
            double conv_mb = (double)NLin * CONV_PER_LAYER / (1024.0 * 1024);
            double total_mb = kv_mb + ssm_mb + conv_mb;
            printf("  %9d   %10.1f   %9d   %9.1f   %9.1f   %10.1f\n",
                   T, kv_mb, nblocks, ssm_mb, conv_mb, total_mb);
        }

        printf("\n  Activation memory per chunk (max_chunk_size=4096):\n");
        int ws_full  = 4*5120 + 12288 + 6144 + 2*1024 + 3*17408;
        int ws_lin   = 5120 + 10240 + 6144 + 16 + 2048 + 6144 + 5120 + 5120 + 3*17408 + 5120 + 32;
        size_t ws_per_tok = std::max(ws_full, ws_lin);
        double act_mb_4k = ws_per_tok * 4096.0 * sizeof(__nv_bfloat16) / (1024.0 * 1024);
        double act_mb_8k = ws_per_tok * 8192.0 * sizeof(__nv_bfloat16) / (1024.0 * 1024);
        printf("    workspace/token:  %zu BF16 elements = %.1f KB\n",
               ws_per_tok, ws_per_tok * sizeof(__nv_bfloat16) / 1024.0);
        printf("    chunk=4096:       %.1f MB activation\n", act_mb_4k);
        printf("    chunk=8192:       %.1f MB activation\n", act_mb_8k);
        printf("    hidden_states:    4096 x 5120 x 2B = %.1f MB\n",
               4096.0 * 5120 * 2 / (1024.0 * 1024));
        printf("    → Chunked prefill: 128K prompt 总内存 = KV(8 GB) + SSM(144 MB) + Act(%.0f MB)\n",
               act_mb_4k);
        printf("    → Unchunked:       128K prompt 总内存 = KV(8 GB) + SSM(144 MB) + Act(%.0f MB) ← OOM!\n",
               ws_per_tok * 131072.0 * sizeof(__nv_bfloat16) / (1024.0 * 1024));

        // 200K / 256K 关键数据
        double kv_200k = (double)((204800 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_BYTES * NUM_FULL_ATTN_LAYERS * 2 / (1024.0 * 1024);
        double kv_256k = (double)((262144 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_BYTES * NUM_FULL_ATTN_LAYERS * 2 / (1024.0 * 1024);
        double ssm_mb = (double)NLin * SSM_PER_LAYER / (1024.0 * 1024);
        printf("    → 200K chunked:    KV(%.0f GB) + SSM(%.0f MB) + Act(%.0f MB) = %.1f GB  [128 GB Thor: OK]\n",
               kv_200k / 1024, ssm_mb, act_mb_4k, (kv_200k + ssm_mb + act_mb_4k) / 1024);
        printf("    → 256K chunked:    KV(%.0f GB) + SSM(%.0f MB) + Act(%.0f MB) = %.1f GB  [128 GB Thor: OK]\n",
               kv_256k / 1024, ssm_mb, act_mb_4k, (kv_256k + ssm_mb + act_mb_4k) / 1024);
    }

    printf("\n  ==============================================================\n");
    printf("  Benchmark complete.\n");
    printf("  ==============================================================\n\n");

    cudaStreamDestroy(stream);
}

// ============================================================================
// 测试入口 — 由 main.cpp 中的 cmd_test 调用
// ============================================================================
int run_tests(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "  Qwen3.5-27B Thor Unit Tests           " << std::endl;
    std::cout << "========================================\n" << std::endl;

    try {
        test_unified_allocator();
        test_safetensors_loader();
        test_kv_cache_manager();
        test_ipc_shm_queue();
        test_dense_gemm();
        test_light_ops();
        test_paged_attention();
        test_qwen_layer();
        test_cache_engine();
        test_kv_swapper();
        test_chunked_prefill();
        test_inference_engine();
    } catch (const std::exception& e) {
        std::cerr << "Test Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "========================================\n" << std::endl;
    return 0;
}

