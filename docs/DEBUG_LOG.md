# Debug 日志

记录推理引擎开发过程中发现和修复的关键 Bug。

---

## Bug #1: 多轮对话图片分析胡乱回答

**发现时间**: 2026-03-02  
**严重程度**: HIGH  
**影响范围**: 所有多轮多模态对话 (OpenAI + Ollama API)

### 现象

用户反馈：多轮对话中，图片分析完全不准确，模型"胡乱分析"。

### 排查过程

#### 第一步：确认单轮正常

发送包含 Shure SM7B 麦克风图片的单轮请求，模型正确识别为"麦克风"。

→ 结论：ViT 编码器、图像预处理管线正常。

#### 第二步：复现多轮异常

构造多轮请求（第一轮带图片，第二轮纯文本追问），发现第二轮回答仍能"看到"图片。

→ 疑点：服务端有图片缓存。

#### 第三步：定位根因 — 全局图片缓存

审查 `serve.h` 和 `serve.cpp`，发现之前为改善多轮体验添加的服务端图片缓存：

```cpp
// serve.h (已删除)
std::mutex vision_cache_mutex_;
std::vector<core::ProcessedImage> cached_images_;
std::chrono::steady_clock::time_point vision_cache_time_;
```

**问题**：`cached_images_` 是 `ServeApp` 的全局成员，没有按会话/请求隔离。
- 用户 A 上传图片 → 全局缓存
- 用户 B 发起无图多轮对话 → 检测到 `messages.size() > 2` → 注入 A 的缓存图片
- 结果：B 的对话中出现完全无关的图片上下文

#### 第四步：控制变量实验证明

1. **实验 A**：先发送真实图片请求（Sunburst 渐变色吉他），让缓存生效。再发送无图多轮请求追问"什么颜色" → 模型正确回答"Sunburst 渐变色"
2. **实验 B**：清除缓存（发送 1×1 像素 dummy PNG），再发送同样的无图多轮请求 → 模型回答"纯色红色"（完全是 hallucination）

→ **确认根因**：全局图片缓存导致跨会话图片污染。

### 修复方案

删除所有服务端图片缓存，遵循标准 LLM API 无状态设计（客户端每次请求都需重新发送图片）。

**修改文件**：
- `src/serve/serve.h`：删除 `vision_cache_mutex_`、`cached_images_`、`vision_cache_time_` 成员
- `src/serve/serve.cpp`：删除 OpenAI chat handler 和 Ollama chat handler 中的缓存注入逻辑（两处）

### 设计决策

参考 OpenAI、Anthropic、Google Gemini 等主流 LLM API：**所有 API 都是无状态的**，多轮对话的上下文（包括图片）由客户端在每次请求中完整发送。服务端不保存跨请求状态。

理由：
- 去除全局/会话状态后本质上消除了跨会话污染
- HTTP API 天然无状态，符合 RESTful 设计
- 客户端（如 Open WebUI、ChatBox）本身就会在每次请求中重新发送完整上下文

---

## Bug #2: SSM/Conv 状态池并发共享导致状态损坏

**发现时间**: 2026-03-02  
**严重程度**: CRITICAL  
**影响范围**: 所有并发请求场景下的 DeltaNet 层输出

### 现象

在 Bug #1 的修复过程中，对整个引擎进行会话隔离审计时发现。

没有明显的外部症状（因为单并发时不会触发），但在多并发场景下会导致：
- DeltaNet SSM 状态被意外清零
- 48 个 Linear Attention 层输出损坏
- 生成质量严重退化 / 胡言乱语

### 根因分析

#### 架构背景

Qwen3.5-27B 有 48 层 Linear Attention（Gated DeltaNet SSM），每层维护一个 SSM 状态矩阵：
- SSM state: `[num_key_heads × key_head_dim × lin_v_per_kh]` = ~3 MB/层 (FP32)
- Conv state: `[in_qkv_conv × (kernel_dim - 1)]` = ~60 KB/层 (BF16)
- 总计每请求: ~147 MB

为避免 Jetson 统一内存上频繁 `cudaMalloc`/`cudaFree` 导致页面回收崩溃，引擎采用预分配池化策略。

#### Bug 所在

**原始代码（`engine.cpp` 构造函数）**：
```cpp
// 只分配了 1 份 SSM/Conv 缓冲区
cudaMalloc(&ssm_pool_base_,  num_linear_layers_ * ssm_size_per_layer_);   // ~144 MB
cudaMalloc(&conv_pool_base_, num_linear_layers_ * conv_size_per_layer_);  // ~2.8 MB

// 每层一个指针，指向池内偏移
for (int li = 0; li < num_linear_layers_; ++li) {
    pooled_ssm_states_[li]  = (float*)((char*)ssm_pool_base_ + li * ssm_size_per_layer_);
    pooled_conv_states_[li] = ...;
}
```

**新请求创建时**：
```cpp
// 所有请求指向同一份物理内存！
for (int li = 0; li < num_linear_layers_; ++li) {
    ctx->ssm_states[li] = pooled_ssm_states_[li];   // 同一地址
    ctx->conv_states[li] = pooled_conv_states_[li];  // 同一地址
    cudaMemsetAsync(ctx->ssm_states[li], 0, ...);    // 清零 = 破坏其它请求的状态！
}
```

### 竞态时序

```
时间线:
  t0: 请求 A 创建 → ssm_states[li] = pooled[li], memset 清零 ✓
  t1: 请求 A prefill → SSM 状态逐层累积，写入 pooled[li]
  t2: 请求 A decode step 1 → 读取 pooled[li]，正常
  t3: 请求 B 创建 → ssm_states[li] = pooled[li] (同一地址!)
                   → cudaMemsetAsync 清零 pooled[li]
                   → 请求 A 的 SSM 状态被破坏！
  t4: 请求 A decode step 2 → 读取已被清零的 pooled[li] → 输出损坏
```

虽然 `inference_loop` 是单线程的，但 **poll 新请求（step 1）和 step 执行（step 2）在同一次循环迭代中串行发生**。当请求 B 在请求 A 的 decode 过程中到达 IPC 队列时，B 的初始化代码会在 A 的下一次 step 之前清零 A 的 SSM 状态。

### 修复方案：多槽位 SSM/Conv 状态池

预分配 `MAX_SSM_SLOTS = 8` 个独立的 SSM/Conv 缓冲区（LIFO free stack 管理），每个活跃请求占用独立 slot。

#### 内存布局变化

```
修复前 (1 slot, ~147 MB):
  ssm_pool_base_ ─► [layer_0][layer_1]...[layer_47]

修复后 (8 slots, ~1175 MB):
  ssm_pool_base_ ─► [slot_0: layer_0..47][slot_1: layer_0..47]...[slot_7: layer_0..47]
```

128 GB 统一内存系统上 ~1.17 GB 完全可接受。

#### 修改文件

**`src/engine/engine.h`**：
- 新增 `MAX_SSM_SLOTS = 8` 常量
- `RequestContext` 新增 `int ssm_slot = -1` 字段
- `pooled_ssm_states_` 从 `vector<float*>` 改为 `vector<vector<float*>>`（`[slot][layer]`）
- 新增 `free_ssm_slots_` 空闲槽位栈

**`src/engine/engine.cpp`**：

| 位置 | 修改 |
|------|------|
| 构造函数（池分配） | 分配 `8 × 48 × 3 MB` 连续内存，初始化 `[slot][layer]` 指针视图，填充空闲栈 |
| 请求创建 | 从 `free_ssm_slots_` 弹出 slot，赋值 `pooled_ssm_states_[slot][li]`。无可用 slot 时尝试换出其它请求或拒绝 |
| Cleanup | 回收 `ctx->ssm_slot` 到 `free_ssm_slots_` |
| Swap-out | 保存 SSM/Conv 到 SSD 后回收 slot |
| Swap-in | 重新从 free stack 分配新 slot |

### 验证

启动服务器日志确认 8 slot 分配：
```
[Engine] SSM/Conv state pool: 8 slots × 48 layers, SSM=3072.0 KB/layer, Conv=60.0 KB/layer, total=1174.5 MB
```

并发请求测试（2 个同时发送）：
```
[Engine] Assigned SSM slot 0 to request 2 (free slots: 7)
[Engine] Assigned SSM slot 1 to request 3 (free slots: 6)    ← 独立 slot!
...
[Engine] Cleanup: req=2 returned SSM slot 0 (free slots: 7)
[Engine] Cleanup: req=3 returned SSM slot 1 (free slots: 8)  ← 正确回收
```

推理结果正确：
- "1+1等于几？" → `2` ✓
- "法国的首都是哪里？" → `巴黎` ✓

---

## 审计附录：其它隔离检查项

在排查上述两个 Bug 期间，对整个引擎进行了全面的会话隔离审计：

| 检查项 | 状态 | 说明 |
|--------|------|------|
| KV Cache Block 分配 | ✅ 正确 | `paged_attention.cpp` 中每请求独立 `block_table`，`allocate_blocks`/`free_blocks` 正确追踪 |
| 响应路由 | ✅ 正确 | `resp_queues_[request_id]` 按请求 ID 隔离，mutex 保护 |
| 图像数据 | ✅ 正确 | `pending_images_[request_id]` 按请求 ID 隔离，mutex 保护 |
| Prefix Cache | ⚠️ 中等风险 | SSM 状态仅在最后一个 chunk 保存。部分前缀匹配时可能恢复 KV 但无 SSM 状态。实践中大多数匹配为完整前缀（相同 system prompt），风险较低 |
| 采样 RNG | ⚠️ 低风险 | `sampling_rng_` 全局共享。仅影响非确定性采样顺序，不影响正确性 |
| SSM/Conv 状态池 | ✅ 已修复 | 见 Bug #2 |
| 服务端图片缓存 | ✅ 已修复 | 见 Bug #1 |

---

## Bug #3: 连续推理 GPU Hard Reset（系统强制复位）

**发现时间**: 2026-02 (早期开发阶段)  
**严重程度**: CRITICAL — 整个系统被硬件看门狗强制复位  
**影响范围**: 所有多轮对话 / 连续请求场景

### 现象

第一个推理请求正常完成。发送第二个请求时（即多轮对话或连续请求），Jetson AGX Thor 整机 hard reset（硬件复位重启），无任何用户态日志。

复位后 `dmesg` 中仅有 Tegra186 WDT (看门狗定时器) 超时记录，无 GPU error 或 kernel oops。

### 根因分析

**这不是单一 Bug，而是 Jetson 统一内存架构 + CUDA stream + 看门狗 三者交互的复合崩溃链**。最终定位到 5 个独立的子问题，每个都可能单独导致系统复位。

#### 子问题 A: cudaMemset 在 default stream 与 compute_stream_ 竞态

**背景**：SSM/Conv 状态池需要在每个新请求开始前清零。

**错误代码**：
```cpp
// 新请求初始化时 — 使用 default stream (stream 0) 清零
cudaMemset(ctx->ssm_states[li], 0, ssm_size_per_layer_);  // stream 0
```

**问题**：推理 kernel 在 `compute_stream_` 上异步执行。`cudaMemset` 默认使用 stream 0，与 `compute_stream_` 无序列化关系。当上一个请求的 cleanup 尚未完成（或下一个请求的 prefill kernel 已在 `compute_stream_` 排队时），stream 0 的 memset 可能与 `compute_stream_` 上的 kernel 同时访问同一内存 → GPU 内部状态损坏 → hang → 看门狗超时 → 系统复位。

**修复**：
```cpp
// 所有 SSM/Conv 清零使用 compute_stream_，与推理 kernel 序列化
cudaMemsetAsync(ctx->ssm_states[li], 0, ssm_size_per_layer_, compute_stream_);
```

#### 子问题 B: Cleanup 未同步 compute_stream_ 就复用 SSM 池

**错误代码**：
```cpp
// 请求完成后立即断开 SSM 指针，不等 GPU kernel 完成
ctx->ssm_states.clear();
ctx->conv_states.clear();
// 下一个请求立即复用同一池 → 清零还在执行中的 kernel 数据
```

**问题**：GPU 上最后一个 decode step 的 kernel 可能还在访问 SSM/Conv 内存，而 CPU 端已经进入 cleanup → 下一个请求的 `cudaMemsetAsync` 再次清零同一块内存 → kernel 读到全零数据 → 后续 kernel 产生 NaN/Inf → GPU pipeline 卡死。

**修复**：在 cleanup 前增加 `sync_stream_with_timeout(compute_stream_, 90, "cleanup_pre_sync")`，确保 GPU 上所有 kernel 完成后再释放资源。

#### 子问题 C: cudaMalloc/cudaFree 频繁调用导致统一内存页面回收崩溃

**背景**：Jetson AGX Thor 使用 128 GB 统一内存，CPU 和 GPU 共享同一物理内存池。每次 `cudaMalloc`/`cudaFree` 都触发 SMMU/IOMMU 页表修改。

**问题**：原始代码在每个请求创建/销毁时对 SSM/Conv 状态执行 `cudaMalloc` (48 层 × 2 buffers = 96 次) 和 `cudaFree` (96 次)。在连续请求频率较高时，SMMU 页表频繁修改 + 统一内存 page cache 争用（driver 需要回收已死进程的 managed pages）→ 内核态相关代码路径中出现不可恢复的状态 → 硬件看门狗超时。

**修复**：改为**预分配池化**策略——构造函数中一次性 `cudaMalloc` 整个 SSM/Conv 池（2 次 `cudaMalloc` 替代 96 次独立分配），推理过程中仅做指针赋值 + `cudaMemsetAsync`，析构函数中一次性 `cudaFree`。后来进一步增加了预清零触发页面映射：
```cpp
// 预清零: 触发统一内存页面分配，避免首次请求时延迟缺页引发看门狗
cudaMemset(ssm_pool_base_, 0, total_ssm_bytes);
cudaMemset(conv_pool_base_, 0, total_conv_bytes);
```

#### 子问题 D: cudaStreamSynchronize spin-wait 饿死看门狗

**背景**：Jetson 上 `systemd` 负责喂硬件看门狗（~2 分钟超时）。推理引擎在等待 GPU 完成时调用 `cudaStreamSynchronize`。

**问题**：CUDA 默认调度模式为 spin-wait（`cudaDeviceScheduleAuto` ≈ `cudaDeviceScheduleSpin`），`cudaStreamSynchronize` 在等待期间 100% 烧满一个 CPU 核心。在长 prefill (>30s) 期间，spin-wait 不让出 CPU → systemd 无法调度到该核心喂狗 → 看门狗超时 → 硬件复位。

**修复**（两层防御）：

1. **BlockingSync 模式**：在 CUDA context 首次初始化前设置 `cudaDeviceScheduleBlockingSync`，使 `cudaStreamSynchronize` 时 CPU 让出（yield/sleep）而非 spin-wait。
```cpp
// backend.cpp — 必须在第一个 CUDA API 调用前
cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
cudaFree(0);  // 初始化 CUDA context
```

2. **Polling + timeout 替代 sync**：实现 `sync_stream_with_timeout()` 函数，使用 `cudaStreamQuery` 轮询 + 10ms sleep 间隔代替阻塞等待。即使 GPU kernel 真的 hang 住，CPU 也不会被阻塞，可以检测超时并做 graceful failure。
```cpp
static bool sync_stream_with_timeout(cudaStream_t stream, int timeout_seconds, const char* context) {
    for (;;) {
        cudaError_t err = cudaStreamQuery(stream);
        if (err == cudaSuccess) return true;
        if (err != cudaErrorNotReady) { /* clear error, return false */ }
        if (elapsed >= timeout_seconds) {
            fprintf(stderr, "*** GPU SYNC TIMEOUT (%s) ***\n", context);
            return false;  // 不阻塞，允许上层做 graceful recovery
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 让出 CPU
    }
}
```

#### 子问题 E: 请求间缺少 GPU idle 间隙

**问题**：第一个请求完成后，cleanup 立即允许下一个请求开始 prefill。但 Jetson 统一内存子系统需要时间完成页面迁移/回收（SMMU 页表更新、TLB 刷新等）。如果 GPU 在上一个请求的页面回收尚未完成时就开始新请求的内存访问，可能触发 SMMU fault → GPU hang。

**修复**：在所有活跃请求完成后、下一个请求开始前，插入全设备同步 + 200ms idle 间隙：
```cpp
if (active_requests.empty()) {
    sync_stream_with_timeout(compute_stream_, 60, "inter_request_sync");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));  // 让统一内存页面安定
}
```

### 修复文件汇总

| 文件 | 修改 |
|------|------|
| `src/engine/backend.cpp` | `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` + `cudaFree(0)` 初始化 CUDA context, `malloc_trim` 禁用 |
| `src/engine/engine.cpp` | `sync_stream_with_timeout()` 实现, SSM 池预分配, `cudaMemsetAsync` 替代 `cudaMemset`, cleanup 前 sync, inter-request 200ms idle 间隙 |

### 经验教训

1. **Jetson 统一内存不是"简化版 GPU"**：CPU/GPU 共享物理内存意味着 SMMU 页表管理是关键路径。频繁的 alloc/free 和跨 stream 访问在独立显存 GPU 上可能无害，在 Jetson 上可能导致硬件级崩溃。
2. **看门狗是硬约束**：嵌入式平台上 `systemd` 喂狗不能被推理 workload 抢占。所有长时间 GPU 等待必须让出 CPU。
3. **"第二次请求崩溃"模式**：如果第一次推理正常但第二次崩溃，优先检查：(a) stream 序列化、(b) 内存复用竞态、(c) GPU 状态残留、(d) 页面管理延迟。
4. **多层防御**：不依赖单一修复，而是在 stream 调度、同步策略、内存分配、idle 间隙四个层面同时加固。

### 验证

修复后连续运行 9 轮稳定性测试（每轮包含 prefill + 多步 decode），全部通过，无任何 GPU hang 或系统复位。

---

## Bug #4: SIGPIPE 导致服务端进程崩溃

**发现时间**: 2026-02 (早期开发阶段)  
**严重程度**: MEDIUM  
**影响范围**: 客户端断开连接时

### 现象

客户端在推理过程中断开 TCP 连接（例如用户关闭浏览器、curl 超时），服务端进程直接退出，无错误信息。

### 根因

Linux 上向已关闭的 socket 写入数据时，内核发送 `SIGPIPE` 信号，默认行为是终止进程。HTTP streaming response 写入时，如果客户端已断开，`send()`/`write()` 触发 `SIGPIPE` → 进程被杀。

### 修复

```cpp
// serve.cpp — run() 入口
signal(SIGPIPE, SIG_IGN);  // 忽略 SIGPIPE，让 send() 返回 EPIPE 错误码
```

同时在 streaming 写入回调中添加了客户端断开检测：
```cpp
std::atomic<bool> client_disconnected{false};
// 写入失败时:
if (send_result < 0) {
    client_disconnected.store(true, std::memory_order_relaxed);
    backend_.cancel(request_id);  // 通知引擎取消该请求
    return;
}
```

### 修改文件

- `src/serve/serve.cpp`：`signal(SIGPIPE, SIG_IGN)` + 各 handler 中的 `client_disconnected` 检测 + `cancel()` 调用

---

## Bug #5: CUTLASS GEMM 对齐要求导致 Prefill Crash

**发现时间**: 2026-02  
**严重程度**: HIGH  
**影响范围**: 特定 prompt 长度的 prefill 阶段

### 现象

Prefill 在特定 token 数量（如 T=478）时 segfault 崩溃，其它长度正常。

### 根因

Prefill attention 中的 Score GEMM `[hpg×T, T, head_dim]`：当 N 维度（= T）不满足 CUTLASS `TileShape(128)` 的 128 对齐要求时，`can_implement()` 返回 false，但代码没有处理这个情况就直接调用了 kernel → 未定义行为 → crash。

### 修复

在 `invoke_dense_gemm` 和 `invoke_dense_gemm_add` 中添加 `can_implement()` 检查，失败时自动回退到 cuBLAS：

```cpp
if (!gemm_op.can_implement(args)) {
    // 回退到 cuBLAS
    cublasGemmEx(get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, ...);
}
```

cuBLAS handle 使用懒初始化 (`get_cublas_handle()`)。

### 修改文件

- `src/engine/dense_gemm_sm110.cu`：`can_implement()` 检查 + cuBLAS 回退路径
