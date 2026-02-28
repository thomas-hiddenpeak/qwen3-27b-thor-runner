// perf_stats.cpp — 性能统计实现
// SystemMonitor: /proc/stat (CPU), NVML (GPU), cudaMemGetInfo (内存)
// PerfProfiler: CUDA 事件计时 + 系统监控 + 报告输出

#include "perf_stats.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <dlfcn.h>  // dlopen/dlsym for NVML

namespace qwen_thor {
namespace perf {

// ============================================================================
// NVML 动态加载 (避免编译时链接依赖)
// ============================================================================
namespace nvml_api {

// NVML 类型定义 (最小子集)
using nvmlDevice_t = void*;
using nvmlReturn_t = int;
enum { NVML_SUCCESS = 0 };

struct nvmlUtilization_t { unsigned int gpu; unsigned int memory; };

// 函数指针
static nvmlReturn_t (*nvmlInit)(void)                                    = nullptr;
static nvmlReturn_t (*nvmlShutdown)(void)                                = nullptr;
static nvmlReturn_t (*nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*) = nullptr;
static nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
static nvmlReturn_t (*nvmlDeviceGetPowerUsage)(nvmlDevice_t, unsigned int*) = nullptr;
static nvmlReturn_t (*nvmlDeviceGetTemperature)(nvmlDevice_t, int, unsigned int*) = nullptr;

static void* lib_handle = nullptr;
static bool  loaded     = false;

static bool load() {
    if (loaded) return lib_handle != nullptr;
    loaded = true;

    // 尝试多个可能的库路径
    const char* paths[] = {
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "/usr/lib/aarch64-linux-gnu/nvidia/libnvidia-ml.so.1",
        nullptr
    };
    for (const char** p = paths; *p; ++p) {
        lib_handle = dlopen(*p, RTLD_LAZY);
        if (lib_handle) break;
    }
    if (!lib_handle) {
        std::cerr << "[PerfStats] WARNING: NVML not found, GPU stats unavailable\n";
        return false;
    }

    // 加载函数符号
    #define LOAD_SYM(name) \
        *(void**)(&name) = dlsym(lib_handle, #name); \
        if (!name) { std::cerr << "[PerfStats] WARNING: " #name " not found in NVML\n"; }

    LOAD_SYM(nvmlInit)
    LOAD_SYM(nvmlShutdown)
    LOAD_SYM(nvmlDeviceGetHandleByIndex)
    LOAD_SYM(nvmlDeviceGetUtilizationRates)
    LOAD_SYM(nvmlDeviceGetPowerUsage)
    LOAD_SYM(nvmlDeviceGetTemperature)

    #undef LOAD_SYM

    if (!nvmlInit || !nvmlDeviceGetHandleByIndex) {
        dlclose(lib_handle);
        lib_handle = nullptr;
        return false;
    }

    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "[PerfStats] WARNING: nvmlInit failed\n";
        dlclose(lib_handle);
        lib_handle = nullptr;
        return false;
    }

    return true;
}

} // namespace nvml_api

// ============================================================================
// SystemMonitor 实现
// ============================================================================

SystemMonitor::SystemMonitor() {
    // 读取 CPU 核心数
    std::ifstream f("/proc/stat");
    std::string line;
    num_cores_ = 0;
    while (std::getline(f, line)) {
        if (line.substr(0, 3) == "cpu" && line[3] != ' ') num_cores_++;
    }
    if (num_cores_ == 0) num_cores_ = 1;

    // 初始化 CPU baseline
    prev_cpu_ = read_cpu_stat();

    // 尝试加载 NVML
    if (nvml_api::load()) {
        nvml_api::nvmlDevice_t dev = nullptr;
        if (nvml_api::nvmlDeviceGetHandleByIndex(0, &dev) == nvml_api::NVML_SUCCESS) {
            nvml_device_ = dev;
            nvml_ok_ = true;
        }
    }
}

SystemMonitor::~SystemMonitor() {
    if (nvml_api::lib_handle && nvml_api::nvmlShutdown) {
        nvml_api::nvmlShutdown();
    }
}

SystemMonitor::CpuTick SystemMonitor::read_cpu_stat() {
    CpuTick tick{};
    std::ifstream f("/proc/stat");
    std::string line;
    if (std::getline(f, line)) {
        // "cpu  user nice system idle iowait irq softirq steal ..."
        if (line.substr(0, 4) == "cpu ") {
            std::istringstream iss(line.substr(4));
            iss >> tick.user >> tick.nice >> tick.system >> tick.idle
                >> tick.iowait >> tick.irq >> tick.softirq >> tick.steal;
        }
    }
    return tick;
}

SystemSnapshot SystemMonitor::sample() {
    SystemSnapshot snap;
    snap.cpu_num_cores = num_cores_;

    // ---- CPU 利用率 (全核平均) ----
    CpuTick cur = read_cpu_stat();
    long long dt = cur.total()  - prev_cpu_.total();
    long long da = cur.active() - prev_cpu_.active();
    if (dt > 0) {
        snap.cpu_util_percent = 100.0f * (float)da / (float)dt;
    }
    prev_cpu_ = cur;

    // ---- GPU (NVML) ----
    if (nvml_ok_ && nvml_device_) {
        auto dev = (nvml_api::nvmlDevice_t)nvml_device_;

        if (nvml_api::nvmlDeviceGetUtilizationRates) {
            nvml_api::nvmlUtilization_t util{};
            if (nvml_api::nvmlDeviceGetUtilizationRates(dev, &util) == nvml_api::NVML_SUCCESS) {
                snap.gpu_util_percent = (float)util.gpu;
            }
        }
        if (nvml_api::nvmlDeviceGetPowerUsage) {
            unsigned int mw = 0;
            if (nvml_api::nvmlDeviceGetPowerUsage(dev, &mw) == nvml_api::NVML_SUCCESS) {
                snap.gpu_power_w = mw / 1000.0f;
            }
        }
        if (nvml_api::nvmlDeviceGetTemperature) {
            unsigned int temp = 0;
            // NVML_TEMPERATURE_GPU = 0
            if (nvml_api::nvmlDeviceGetTemperature(dev, 0, &temp) == nvml_api::NVML_SUCCESS) {
                snap.gpu_temp_c = (float)temp;
            }
        }
    }

    // ---- 系统内存 (/proc/meminfo) ----
    {
        std::ifstream mi("/proc/meminfo");
        std::string line;
        size_t total_kb = 0, avail_kb = 0;
        while (std::getline(mi, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line.substr(9));
                iss >> total_kb;
            } else if (line.find("MemAvailable:") == 0) {
                std::istringstream iss(line.substr(13));
                iss >> avail_kb;
            }
        }
        snap.mem_total_mb = total_kb / 1024;
        snap.mem_used_mb  = (total_kb - avail_kb) / 1024;
        if (snap.mem_total_mb > 0)
            snap.mem_util_percent = 100.0f * snap.mem_used_mb / snap.mem_total_mb;
    }

    // ---- CUDA 内存 (cudaMemGetInfo) ----
    {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            snap.cuda_total_mb = total_bytes / (1024 * 1024);
            snap.cuda_used_mb  = (total_bytes - free_bytes) / (1024 * 1024);
        }
    }

    return snap;
}

// ============================================================================
// PerfProfiler 实现
// ============================================================================

PerfProfiler::PerfProfiler() = default;

void PerfProfiler::begin(const std::string& phase, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = phases_.find(phase);
    if (it == phases_.end()) {
        phase_order_.push_back(phase);
        it = phases_.emplace(phase, PhaseCtx{}).first;
    }
    it->second.timer.start(stream);
}

void PerfProfiler::end(const std::string& phase, cudaStream_t stream) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = phases_.find(phase);
    if (it == phases_.end()) return;
    it->second.timer.stop(stream);
    float ms = it->second.timer.elapsed_ms();
    it->second.stat.record(ms);
}

void PerfProfiler::request_start() {
    req_start_ = Clock::now();
    prefill_tokens_ = 0;
    decode_steps_ = 0;
}

void PerfProfiler::request_prefill_done(int num_tokens) {
    prefill_done_ = Clock::now();
    prefill_tokens_ = num_tokens;
}

void PerfProfiler::request_decode_step() {
    decode_steps_++;
}

void PerfProfiler::request_done() {
    // 在 print_request_summary 中使用
}

SystemSnapshot PerfProfiler::snapshot() {
    return monitor_.sample();
}

void PerfProfiler::print_step_report(int step_num) {
    auto snap = snapshot();

    // 收集本 step 的各阶段耗时
    float total_step_ms = 0.0f;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& name : phase_order_) {
            total_step_ms += phases_[name].stat.last_ms;
        }
    }

    fprintf(stderr,
        "[Step %3d] total=%.1fms | CPU %.0f%% | GPU %.0f%% %.1fW %.0f°C | "
        "MEM %zu/%zuMB (%.0f%%) | CUDA %zu/%zuMB\n",
        step_num, total_step_ms,
        snap.cpu_util_percent,
        snap.gpu_util_percent, snap.gpu_power_w, snap.gpu_temp_c,
        snap.mem_used_mb, snap.mem_total_mb, snap.mem_util_percent,
        snap.cuda_used_mb, snap.cuda_total_mb);
}

void PerfProfiler::print_request_summary() {
    auto now = Clock::now();
    double total_s   = std::chrono::duration<double>(now - req_start_).count();
    double prefill_s = std::chrono::duration<double>(prefill_done_ - req_start_).count();
    double decode_s  = total_s - prefill_s;
    double tps       = decode_steps_ > 0 ? decode_steps_ / decode_s : 0.0;

    fprintf(stderr,
        "\n╔══════════════════════════════════════════════════════════════╗\n"
        "║                   REQUEST SUMMARY                           ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  Prefill : %5d tokens in %8.2f ms  (%7.1f tok/s)       ║\n"
        "║  Decode  : %5d tokens in %8.2f ms  (%7.1f tok/s)       ║\n"
        "║  Total   :              %8.2f ms                        ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n",
        prefill_tokens_, prefill_s * 1000.0,
        prefill_tokens_ > 0 ? prefill_tokens_ / prefill_s : 0.0,
        decode_steps_, decode_s * 1000.0, tps,
        total_s * 1000.0);

    // 阶段细分
    {
        std::lock_guard<std::mutex> lk(mu_);
        fprintf(stderr, "║  %-20s %8s %8s %8s %8s %5s ║\n",
                "Phase", "Last", "Avg", "Min", "Max", "Count");
        fprintf(stderr, "║  %-20s %8s %8s %8s %8s %5s ║\n",
                "────────────────────", "────────", "────────", "────────", "────────", "─────");
        for (auto& name : phase_order_) {
            auto& s = phases_[name].stat;
            fprintf(stderr, "║  %-20s %7.2fms %7.2fms %7.2fms %7.2fms %5d ║\n",
                    name.c_str(), s.last_ms, s.avg_ms(), s.min_ms, s.max_ms, s.count);
        }
    }

    // 最终系统状态
    auto snap = snapshot();
    fprintf(stderr,
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  CPU: %.0f%% (%d cores) | GPU: %.0f%% %.1fW %.0f°C              ║\n"
        "║  MEM: %zu/%zu MB (%.0f%%) | CUDA: %zu/%zu MB                  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n\n",
        snap.cpu_util_percent, snap.cpu_num_cores,
        snap.gpu_util_percent, snap.gpu_power_w, snap.gpu_temp_c,
        snap.mem_used_mb, snap.mem_total_mb, snap.mem_util_percent,
        snap.cuda_used_mb, snap.cuda_total_mb);
}

void PerfProfiler::print_phase_summary() {
    fprintf(stderr, "\n=== Phase Timing Summary ===\n");
    fprintf(stderr, "  %-24s %10s %10s %10s %10s %6s\n",
            "Phase", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)", "Count");
    fprintf(stderr, "  %-24s %10s %10s %10s %10s %6s\n",
            "────────────────────────", "──────────", "──────────", "──────────", "──────────", "──────");
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& name : phase_order_) {
            auto& s = phases_[name].stat;
            fprintf(stderr, "  %-24s %9.1fms %9.2fms %9.2fms %9.2fms %6d\n",
                    name.c_str(), s.total_ms, s.avg_ms(), s.min_ms, s.max_ms, s.count);
        }
    }
    fprintf(stderr, "\n");
}

const PhaseStat* PerfProfiler::get_phase(const std::string& name) const {
    auto it = phases_.find(name);
    return it != phases_.end() ? &it->second.stat : nullptr;
}

void PerfProfiler::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    phases_.clear();
    phase_order_.clear();
    prefill_tokens_ = 0;
    decode_steps_ = 0;
}

} // namespace perf
} // namespace qwen_thor
