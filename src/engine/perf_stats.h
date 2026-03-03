#pragma once
// perf_stats.h — 推理性能统计与系统资源监控
// 提供: CUDA 事件计时、CPU 利用率、GPU 利用率/功耗/温度、内存统计

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace qwen_thor {
namespace perf {

// ============================================================================
// CUDA 事件计时器 — 异步、零开销 (除了事件本身)
// ============================================================================
struct CudaTimer {
    cudaEvent_t start_event, stop_event;
    bool created = false;

    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        created = true;
    }
    ~CudaTimer() {
        if (created) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
    }
    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;
    CudaTimer(CudaTimer&& o) noexcept
        : start_event(o.start_event), stop_event(o.stop_event), created(o.created) {
        o.created = false;
    }

    void start(cudaStream_t stream) { cudaEventRecord(start_event, stream); }
    void stop(cudaStream_t stream)  { cudaEventRecord(stop_event, stream); }
    
    // 阻塞等待并返回毫秒数
    float elapsed_ms() {
        cudaEventSynchronize(stop_event);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

// ============================================================================
// 阶段计时统计 — 累积平均 + 最近值
// ============================================================================
struct PhaseStat {
    float last_ms     = 0.0f;
    float total_ms    = 0.0f;
    float min_ms      = 1e9f;
    float max_ms      = 0.0f;
    int   count       = 0;

    void record(float ms) {
        last_ms = ms;
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
        count++;
    }

    float avg_ms() const { return count > 0 ? total_ms / count : 0.0f; }

    void reset() {
        last_ms = total_ms = max_ms = 0.0f;
        min_ms = 1e9f;
        count = 0;
    }
};

// ============================================================================
// 系统资源快照
// ============================================================================
struct SystemSnapshot {
    // CPU
    float cpu_util_percent = 0.0f;   // 所有核心平均利用率
    int   cpu_num_cores    = 0;

    // GPU (via NVML or tegrastats fallback)
    float gpu_util_percent = 0.0f;
    float gpu_power_w      = 0.0f;
    float gpu_temp_c       = 0.0f;

    // 内存 (统一内存)
    size_t mem_used_mb     = 0;
    size_t mem_total_mb    = 0;
    float  mem_util_percent = 0.0f;

    // CUDA 内存 (cudaMemGetInfo)
    size_t cuda_used_mb    = 0;
    size_t cuda_total_mb   = 0;
};

// ============================================================================
// SystemMonitor — 读取 /proc/stat, NVML, cudaMemGetInfo
// ============================================================================
class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // 采样一次系统状态 (非线程安全, 由调用者序列化)
    SystemSnapshot sample();

private:
    // CPU 利用率: 两次 /proc/stat 差分
    struct CpuTick {
        long long user = 0, nice = 0, system = 0, idle = 0;
        long long iowait = 0, irq = 0, softirq = 0, steal = 0;
        long long total() const { return user + nice + system + idle + iowait + irq + softirq + steal; }
        long long active() const { return total() - idle - iowait; }
    };
    CpuTick prev_cpu_;
    int num_cores_ = 0;

    CpuTick read_cpu_stat();

    // NVML handle (opaque, cast from nvmlDevice_t)
    void* nvml_device_ = nullptr;
    bool  nvml_ok_     = false;
};

// ============================================================================
// PerfProfiler — 整合计时与系统监控, 提供格式化报告
// ============================================================================
class PerfProfiler {
public:
    PerfProfiler();
    ~PerfProfiler() = default;

    // ---- 阶段计时 (CUDA 异步) ----
    // 开始/结束一个命名阶段 (嵌套不支持, 但可以并列)
    void begin(const std::string& phase, cudaStream_t stream);
    void end(const std::string& phase, cudaStream_t stream);

    // ---- 请求级计时 (wall-clock) ----
    void request_start();
    void request_prefill_done(int num_tokens);
    void request_decode_step();
    void request_done();

    // ---- 系统快照 ----
    // 调用 SystemMonitor::sample(), 通常在每 N 个 step 调用一次
    SystemSnapshot snapshot();

    // ---- 报告 ----
    // 打印当前请求的 decode 统计
    void print_step_report(int step_num);

    // 打印请求完成时的摘要
    void print_request_summary();

    // 打印所有阶段的累积统计 (通常在进程结束时)
    void print_phase_summary();

    // 获取内部统计 (for test)
    const PhaseStat* get_phase(const std::string& name) const;

    // 重置所有统计
    void reset();

private:
    // 阶段 -> (CudaTimer, PhaseStat)
    struct PhaseCtx {
        CudaTimer timer;
        PhaseStat stat;
        bool pending = false;  // stop event recorded but not yet read
    };
    std::unordered_map<std::string, PhaseCtx> phases_;
    std::mutex mu_;  // 保护 phases_ 的并发插入

    // 有序键列表 (保持插入顺序以便打印)
    std::vector<std::string> phase_order_;

    // 请求级统计
    using Clock = std::chrono::steady_clock;
    Clock::time_point req_start_;
    Clock::time_point prefill_done_;
    int prefill_tokens_ = 0;
    int decode_steps_   = 0;

    SystemMonitor monitor_;
};

// ============================================================================
// 便捷宏 — 在一对花括号内自动计时
// 用法: PERF_SCOPE(profiler, "rmsnorm", stream) { ... kernel launch ... }
// ============================================================================
#define PERF_SCOPE(profiler, name, stream)                          \
    for (bool _ps_once = ((profiler).begin(name, stream), true);    \
         _ps_once;                                                  \
         _ps_once = ((profiler).end(name, stream), false))

} // namespace perf
} // namespace qwen_thor
