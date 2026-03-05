#pragma once
// nvtx_utils.h — NVTX 标记工具
// 4 级嵌套: Request → Phase → Layer → Op
// 不挂 profiler 时 NVTX 调用退化为 NOP, 零开销
//
// 用法:
//   NVTX_SCOPE("Embedding");                    // 简单 RAII scope
//   NVTX_SCOPE_COLOR("Forward", NVTX_GREEN);    // 带颜色
//   NVTX_LAYER_SCOPE(3, true);   // "GQA_03"   // 层级 scope (自动命名)
//   NVTX_LAYER_SCOPE(0, false);  // "GDN_00"
//   nvtx_push("Prefill"); ... nvtx_pop();       // 手动 push/pop (跨分支用)

#include <nvtx3/nvToolsExt.h>
#include <cstdio>

namespace qwen_thor {

// ============================================================================
// 预定义颜色 (ARGB)
// ============================================================================
constexpr uint32_t NVTX_BLUE    = 0xFF1E90FF;  // Request
constexpr uint32_t NVTX_GREEN   = 0xFF32CD32;  // Phase: Forward
constexpr uint32_t NVTX_ORANGE  = 0xFFFF8C00;  // Phase: Embedding/Sample
constexpr uint32_t NVTX_RED     = 0xFFDC143C;  // Phase: LM Head
constexpr uint32_t NVTX_PURPLE  = 0xFF9370DB;  // Layer: GQA
constexpr uint32_t NVTX_CYAN    = 0xFF00CED1;  // Layer: GDN
constexpr uint32_t NVTX_YELLOW  = 0xFFFFD700;  // Op: GEMV/GEMM
constexpr uint32_t NVTX_GRAY    = 0xFFA9A9A9;  // Op: misc

// ============================================================================
// 底层 push/pop
// ============================================================================
inline void nvtx_push(const char* name) {
    nvtxRangePushA(name);
}

inline void nvtx_push_color(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {};
    attr.version       = NVTX_VERSION;
    attr.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType     = NVTX_COLOR_ARGB;
    attr.color         = color;
    attr.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

inline void nvtx_pop() {
    nvtxRangePop();
}

// ============================================================================
// RAII Scope Guard
// ============================================================================
struct NvtxScope {
    NvtxScope(const char* name)                    { nvtx_push(name); }
    NvtxScope(const char* name, uint32_t color)    { nvtx_push_color(name, color); }
    ~NvtxScope()                                   { nvtx_pop(); }
    NvtxScope(const NvtxScope&) = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;
};

// ============================================================================
// Layer Scope: "GDN_00" / "GQA_03" 等, 带颜色区分
// ============================================================================
struct NvtxLayerScope {
    NvtxLayerScope(int layer_idx, bool is_full_attn) {
        snprintf(buf_, sizeof(buf_), "%s_%02d",
                 is_full_attn ? "GQA" : "GDN", layer_idx);
        nvtx_push_color(buf_, is_full_attn ? NVTX_PURPLE : NVTX_CYAN);
    }
    ~NvtxLayerScope() { nvtx_pop(); }
    NvtxLayerScope(const NvtxLayerScope&) = delete;
    NvtxLayerScope& operator=(const NvtxLayerScope&) = delete;
private:
    char buf_[16];
};

// ============================================================================
// 便捷宏
// ============================================================================
#define NVTX_SCOPE(name)              qwen_thor::NvtxScope _nvtx_##__LINE__(name)
#define NVTX_SCOPE_COLOR(name, color) qwen_thor::NvtxScope _nvtx_##__LINE__(name, color)
#define NVTX_LAYER_SCOPE(idx, is_fa)  qwen_thor::NvtxLayerScope _nvtx_layer_##__LINE__(idx, is_fa)

} // namespace qwen_thor
