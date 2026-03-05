#pragma once
// =============================================================================
// SM110a Hardware Primitives — Abstraction Layer
//
// 基于 Level 0 micro-benchmark 实测数据填充。
// Jetson AGX Thor: SM110a (Blackwell), 20 SM, 128 GB LPDDR5X 统一内存
//
// 依赖: CUTLASS CuTe (TMEM/UMMA/TMA 指令封装)
// =============================================================================

#include <cstdint>

namespace sm110a {

// ─────────────────────────────────────────────────────────────────────────────
// TMEM (Tensor Memory) 容量
// ─────────────────────────────────────────────────────────────────────────────
// cta_group::1 (单 CTA, 128 threads = 4 warps)
constexpr int TMEM_COLUMNS_TOTAL     = 512;     // 最大可分配列数
constexpr int TMEM_DP_PER_CTA        = 128;     // 深度位置数 (= CTA 线程数)
constexpr int TMEM_BYTES_PER_COLUMN  = TMEM_DP_PER_CTA * 4;  // 512 bytes (FP32)
constexpr int TMEM_TOTAL_BYTES       = TMEM_COLUMNS_TOTAL * TMEM_BYTES_PER_COLUMN; // 256 KB
constexpr int TMEM_ALLOC_GRANULARITY = 32;      // 分配最小粒度 (列)
constexpr int TMEM_ALLOC_MIN         = 32;      // 最小分配量
constexpr int TMEM_ALLOC_MAX         = 512;     // 最大分配量

// ─────────────────────────────────────────────────────────────────────────────
// GDN State ↔ TMEM 映射
// ─────────────────────────────────────────────────────────────────────────────
// State[KD=128, VD=128] FP32:
//   DP 维度 = KD = 128 (映射到 128 个线程/DP)
//   列维度 = VD = 128 (映射到 128 个 TMEM 列)
// 总占用 = 128 列 = TMEM 总容量的 25%
constexpr int GDN_STATE_KD           = 128;
constexpr int GDN_STATE_VD           = 128;
constexpr int GDN_STATE_TMEM_COLUMNS = GDN_STATE_VD;   // 128 列
constexpr int GDN_STATE_TMEM_BYTES   = GDN_STATE_TMEM_COLUMNS * TMEM_BYTES_PER_COLUMN; // 64 KB

// ─────────────────────────────────────────────────────────────────────────────
// UMMA (Unified MMA) BF16 Tile 约束
// ─────────────────────────────────────────────────────────────────────────────
// SS 模式 (A=SMEM, B=SMEM → C=TMEM), cta_group::1
//   M ∈ {64, 128}
//   N ∈ [8, 256], 步长 8
//   K_ATOM = 16 (BF16: 256 bits / 16 bits = 16 elements)
//
// TS 模式 (A=TMEM, B=SMEM → C=TMEM), cta_group::1
//   M ∈ {64, 128}
//   M=64:  N ∈ [8, 256], 步长 8
//   M=128: N ∈ [16, 256], 步长 16
//   A 必须 K-major
constexpr int UMMA_BF16_K_ATOM       = 16;      // BF16 每 MMA atom 的 K 维度
constexpr int UMMA_SS_M_MIN          = 64;
constexpr int UMMA_SS_M_MAX          = 128;
constexpr int UMMA_SS_N_MIN          = 8;
constexpr int UMMA_SS_N_MAX          = 256;

// GDN rank-1 update 无法使用 UMMA (K=1 < K_ATOM=16)
// GDN WY prefill GEMMs 可以使用 UMMA (inner dim = kd = 128 >> 16)
// FullAttn QK^T/PV 可以使用 UMMA (inner dim = head_dim = 256 >> 16)

// ─────────────────────────────────────────────────────────────────────────────
// 硬件参数 (Jetson AGX Thor SM110a)
// ─────────────────────────────────────────────────────────────────────────────
constexpr int    NUM_SM              = 20;
constexpr float  GPC_CLOCK_GHZ      = 1.575f;
constexpr float  DRAM_PEAK_GBS      = 273.0f;   // LPDDR5X 理论峰值
constexpr int    L2_CACHE_BYTES     = 32 * 1024 * 1024;  // 32 MB
constexpr int    SMEM_PER_SM_BYTES  = 228 * 1024;         // 228 KB/SM
constexpr int    SMEM_PER_BLOCK_MAX = 48 * 1024;          // 48 KB/block (default)

// ─────────────────────────────────────────────────────────────────────────────
// 实测数据 (sm110a_probe, 2026-03-05, MAXN power mode, 电源策略优化后)
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// 实测数据 (sm110a_probe Statistical Edition, 2026-03-05)
// MAXN power mode, 电源策略优化后
// 统计方法: 20 trials, 5 warmup, 报告 median ± stddev
// 全部 CV < 5% 除 DRAM 大 payload (CV < 12%) 外
// ─────────────────────────────────────────────────────────────────────────────

// TMEM lifecycle (128 threads / 4 warps CTA, cta_group::1)
// 列数必须 2 的幂次 (32/64/128), 上限 128 (TMEM_DP_PER_CTA)
constexpr float TMEM_ALLOC_NS            = 221.6f;  // 列数无关, CV=0.0%
constexpr float TMEM_DEALLOC_NS          = 119.4f;  // 列数无关, CV=0.0%
constexpr float TMEM_STORE_NS_PER_COL   = 5.3f;    // @128cols, CV=0.0%
constexpr float TMEM_LOAD_NS_PER_COL    = 10.5f;   // @128cols, CV=0.0%
// GDN state 128 cols: store 677ns, load 1344ns (total lifecycle ~2.4μs)

// UMMA throughput (单 SM, BF16 → FP32, SS mode)
constexpr float UMMA_SS_64x64_TFLOPS    = 1.6f;    // 100~10000 iters 均稳定 1.6, CV=0.0%

// UMMA throughput (单 SM, BF16 → FP32, TS mode: A=TMEM, B=SMEM)
constexpr float UMMA_TS_64x64_TFLOPS    = 1.6f;    // 与 SS 一致, CV=0.0% ★★★
// ★ TS 模式吞吐 = SS, 但省去 A 的 SMEM→ALU 读取, 可用于 chain MMA

// UTCCP (SMEM→TMEM, tcgen05.cp, 单 SM)
constexpr float UTCCP_128x256b_NS      = 162.6f;   // 4 KB/op, CV=0.0% ★★★
constexpr float UTCCP_4x256b_NS        = 10.2f;    // 128 B/op, CV=0.0% ★★★
constexpr float UTCCP_128x256b_GBS     = 25.2f;    // 4096 B / 162.6 ns ≈ 25.2 GB/s/SM
constexpr float UTCCP_4x256b_GBS       = 12.6f;    // 128 B / 10.2 ns ≈ 12.6 GB/s/SM

// TMEM pack::16b Load / unpack::16b Store (单 SM)
// pack: FP32 in TMEM → BF16 in registers (隐式类型转换)
// unpack: BF16 in registers → FP32 in TMEM
constexpr float TMEM_NORMAL_LD_NS      = 1.0f;     // 32x32b.x4, CV=0.0% ★★★
constexpr float TMEM_PACK_LD_NS        = 1.4f;     // 16x256b.x1.pack::16b, CV=0.0%
constexpr float TMEM_NORMAL_ST_NS      = 2.0f;     // 32x32b.x4, CV=0.0%
constexpr float TMEM_UNPACK_ST_NS      = 10.9f;    // 16x256b.x1.unpack::16b, CV=0.0%
// ★ pack load 仅 1.4× normal, 几乎 FP32→BF16 免费
// ⚠️ unpack store 5.5× normal — 避免频繁 BF16→FP32 TMEM 回写

// L1::no_allocate DRAM Read (v8.f32, 256-bit, L1 bypass, 64MB, 20 SM)
constexpr float DRAM_READ_NOALLOC_GBS  = 260.5f;   // L1::no_allocate, CV=7.3% ★☆☆
constexpr float DRAM_READ_STANDARD_GBS = 261.0f;   // standard v4.f32, CV=9.1% ★☆☆
// ★ L1 bypass 对 streaming 访问无显著优势/劣势
//   统一内存 LPDDR5X 下 L1 cache 对 DRAM 访问本就影响有限

// f32x2 SIMD FMA (单 SM, 128 threads)
constexpr float FP32_SCALAR_FMA_NS     = 2.8f;     // ns per FMA, CV=0.0% ★★★
constexpr float F32X2_FMA_NS           = 2.9f;     // ns per f32x2 op (= 2 FMAs), CV=0.0%
constexpr float F32X2_SPEEDUP          = 1.97f;    // 接近 2.0× 理论值
// ★ fma.rn.f32x2 有效 2× throughput: RMSNorm/SiLU 用 f32x2 可省 50% 指令

// LDSM/STSM b8 (SM100+ 8-bit matrix load/store from SMEM)
constexpr float LDSM_B8_M16N16_NS     = 1.2f;     // ldmatrix m16n16.x1.trans, CV=0.0% ★★★
constexpr float STSM_B8_M16N8_NS      = 2.9f;     // stmatrix m16n8.x1.trans, CV=0.1%
constexpr float LDSM_B8_GBS           = 222.2f;    // 256B / 1.2ns ≈ 222 GB/s/SM
constexpr float STSM_B8_GBS           = 44.8f;     // 128B / 2.9ns ≈ 45 GB/s/SM
// ★ 8-bit LDSM 极快 (222 GB/s/SM), 适合 INT8/FP8 数据搬运

// UMMA Fence / Commit overhead (单 SM, 无 pending MMA)
constexpr float UMMA_WAIT_LD_NS        = 1.1f;     // tcgen05.wait::ld, CV=0.0% ★★★
constexpr float UMMA_WAIT_ST_NS        = 7.5f;     // tcgen05.wait::st, CV=0.0%
constexpr float UMMA_COMMIT_NS         = 12.6f;    // tcgen05.commit, CV=0.0%
// ★ wait::ld 仅 1.1ns (几乎免费), wait::st 7.5ns, commit 12.6ns
//   pipeline 中可安全使用 wait/commit 做同步, 开销可忽略

// Warpgroup Register Reconfiguration (setmaxnreg inc+dec, 128 threads)
constexpr float REG_RECONFIG_INC_DEC_NS = 57.3f;   // inc(32)+dec(32) pair, CV=0.0% ★★★
// ★ 每次 reconfig ~57ns = ~90 cycles: 适合 warp-specialization
//   producer 释放寄存器 → consumer 获取, 每层切换仅 1 次, 开销可忽略

// Grid Dependency Control (PDL, griddepcontrol)
constexpr float PDL_LAUNCH_DEPENDENTS_NS = 5.2f;   // launch_dependents, CV=0.2% ★★★
constexpr float PDL_WAIT_NS              = 1.1f;   // wait, CV=0.0% ★★★
// ★ PDL 指令延迟极低 (5.2ns + 1.1ns): 64 层 forward 的 kernel 间隙可用 PDL 消除

// DRAM Read Bandwidth (cudaEvent timing, 全 20 SM)
constexpr float DRAM_READ_256MB_GBS     = 228.8f;  // median, CV=11.4% (★☆☆)
constexpr float DRAM_READ_64MB_GBS      = 263.0f;  // median, CV=9.5%
constexpr float DRAM_READ_512MB_GBS     = 228.1f;  // median, CV=8.6%
constexpr float DRAM_MEASURED_GBS       = 249.0f;  // 20 SM 综合 (Probe 5), CV=3.3%

// DRAM Write Bandwidth — 读写非对称, 写 BW = 读的 ~60%
constexpr float DRAM_WRITE_256MB_GBS    = 176.6f;  // median, CV=3.7% (★★☆)
constexpr float DRAM_WRITE_512MB_GBS    = 149.3f;  // median, CV=1.9% (★★★)
constexpr float DRAM_WRITE_64MB_GBS     = 169.0f;  // median, CV=3.8%
// ★ LPDDR5X 写带宽仅 55-64% of peak (149-177 GB/s vs 273 peak)

// L2 Cache Bandwidth (fits-in-L2 payload)
constexpr float L2_BW_1MB_GBS          = 3244.0f;  // median, CV=4.9% (★★☆)
constexpr float L2_BW_2MB_GBS          = 3253.0f;  // median, CV=2.3%
constexpr float L2_BW_4MB_GBS          = 3093.0f;  // median, CV=6.5%
constexpr float L2_BW_8MB_GBS          = 1569.0f;  // 部分 L2 命中
constexpr float L2_BW_16MB_GBS         = 1425.0f;
constexpr float L2_BW_32MB_GBS         = 1425.0f;  // 32MB = L2 size, 仍全命中

// DRAM Multi-SM Scaling (256MB read, per-SM bandwidth)
// 1 SM:  27 GB/s (10% peak) — 单 SM DRAM 上限
// 2 SM:  54 GB/s → 近线性
// 10 SM: 180 GB/s (66%) → 带宽效率下降
// 20 SM: 249 GB/s (91%) → 饱和
// ★ per-SM 效率: 1 SM=100%, 10 SM=66%, 20 SM=45% → 并行度换带宽

// Memory Access Latency (pointer chasing, 1 thread, 100K steps)
constexpr float LATENCY_L1_NS          = 24.7f;    // ≤16KB working set, CV=0.0%
constexpr float LATENCY_L2_NEAR_NS     = 26.4f;    // 64KB working set
constexpr float LATENCY_L2_FAR_NS      = 85.5f;    // 256KB working set
constexpr float LATENCY_L2_DEEP_NS     = 161.7f;   // 16MB working set
constexpr float LATENCY_DRAM_NS        = 169.9f;   // 256MB working set, CV=0.3%
// L1 → L2 断崖: 64KB→256KB (26→86 ns, 3.3×)
// L2 → DRAM 过渡: 渐进式 (32MB→256MB, 162→170 ns, 仅 5% 差异)
// ★ 统一内存下 "DRAM" 延迟接近 L2 深层 — TLB/page fault 主导

// TMA Bulk Copy (cp.async.bulk, 单 SM, GMEM L2 cached)
constexpr float TMA_LOAD_FIXED_NS      = 202.0f;   // 512B tile, 近固定开销
constexpr float TMA_LOAD_16KB_NS       = 286.3f;   // 交叉点, CV=0.2%
constexpr float TMA_LOAD_32KB_NS       = 369.8f;   // 32KB tile, CV=0.8%
constexpr float TMA_STORE_32KB_NS      = 704.5f;   // CV=0.0%
constexpr int   TMA_CROSSOVER_BYTES    = 16 * 1024; // ≥16KB: TMA wins
constexpr int   TMA_MAX_TILE_BYTES     = 32 * 1024; // 48KB fails (SMEM/block limit)

// ─────────────────────────────────────────────────────────────────────────────
// 关键推论 (基于实测数据)
// ─────────────────────────────────────────────────────────────────────────────
// 1. TMEM 可完整存储 GDN state [128×128] FP32:
//    仅占 128/512 = 25% TMEM, 剩余 384 列可供 K/Q 缓存或 WY 中间结果
//
// 2. TMEM → SMEM roundtrip 约 15 ns/col (store+load)
//    GDN state 128 cols 完整 roundtrip ~2 μs, 远快于 GMEM 访问
//
// 3. UMMA SS 64×64 实测 1.61 TFLOPS/SM
//    20 SM 理论全占: 32.2 TFLOPS (需验证多 SM 并行)
//    GDN WY [128,128] 矩阵乘 = 128×128×128×2 = 4.2M FLOPs
//    单 SM 耗时 ≈ 2.6 μs (1.61 TFLOPS basis)
//    vs 当前标量: O(ChunkSize² × kd) 循环
//
// 4. DRAM 实测 249 GB/s (20 SM, 91% of 273 peak) → 51GB 权重读需 205 ms
//    DRAM 写带宽仅 149-177 GB/s (55-64% peak) — LPDDR5X 读写非对称
//    当前 decode ~232 ms/step → 说明有 ~27 ms 非权重开销
//    UMMA 不直接帮助 decode (rank-1 update K=1 < K_ATOM=16)
//    但 TMEM state 缓存省 GMEM I/O: 每 decode step 省 144 MB (48 层 × 3 MB)
//    节省: 144 MB / 249 GB/s = 0.58 ms → ~0.25% 加速 (较小)
//
// 4b. L2 cache 带宽 ~3.2 TB/s (@1-2MB), 32MB 全 L2 仍 ~1.4 TB/s
//     对于 fits-in-L2 的中间结果 (softmax, norm residuals) 可忽略
//
// 4c. DRAM 读写非对称结论:
//     KV cache 写入/SSM state 写回将受限于写带宽
//     Write BW ~60% of Read → 需写入优化 (batch, streaming store)
//
// 4d. Memory Latency 分层:
//     L1 ≤16KB:     24.7 ns (40.5M accesses/s)
//     L2 256KB:     85.5 ns (11.7M accesses/s)  ← 3.5× L1
//     L2 deep 16MB: 161.7 ns (6.2M accesses/s) ← 6.5× L1
//     DRAM 256MB:   169.9 ns (5.9M accesses/s) ← 仅比 L2 deep 高 5%
//     ★ 统一内存 DRAM 延迟接近 L2 深层 — pointer chasing 场景下
//       TLB/page table walk 而非 cache miss 主导延迟
//
// 5. GDN rank-1 update 的 UMMA 映射: K=1 不满足 K_ATOM=16 约束
//    ★ 结论: rank-1 update 必须保持标量实现, UMMA 仅用于:
//       - WY prefill GEMMs (inner dim = kd = 128, 满足 K_ATOM=16)
//       - FullAttn QK^T/PV (inner dim = 256, 满足)
//
// 6. TMA cp.async.bulk 跨越分析 (单 SM, L2 cached 数据源):
//    ≤ 8KB: 手动 cooperative copy (128 threads float4) 更快
//    16KB: 大致持平 (TMA 286ns vs Manual 299ns, 1.05×)
//    32KB: TMA 4.31× 胜出 (TMA 371ns vs Manual 1597ns)
//    ★ 结论: GDN state tile (32KB BF16 / 64KB FP32) 应使用 TMA
//       KV block (8KB) 保持手动 copy
//       TMA LOAD 释放 warp 做其他计算, 适合计算-传输重叠
//
// 7. TMA STORE 带宽约为 LOAD 的 53% (32KB: 46.5 vs 88.4 GB/s/SM)
//    State 写回 GMEM 比读取慢 ~2×, 但仍远快于线程写入
//
// 8. UMMA TS 模式 = SS 吞吐 (均 1.6 TFLOPS/SM):
//    TS 省去 A 的 SMEM 读, 可用于 chain MMA (上一步输出 TMEM→下一步输入 TMEM)
//    GDN WY fused kernel: 消除中间 TMEM→SMEM→TMEM 往返
//
// 9. UTCCP (SMEM→TMEM): 128x256b = 4KB/162.6ns = 25.2 GB/s/SM
//    TS 模式前置: 将 SMEM 权重拷贝到 TMEM 作为 A 输入
//    4x256b = 128B/10.2ns = 12.6 GB/s/SM (小数据更灵活)
//
// 10. TMEM pack::16b 几乎免费 (1.4 vs 1.0 ns):
//     FP32 accumulator → BF16 输出零开销, epilogue 直接用 pack 读出
//     ⚠️ unpack store 5.5× 慢 — 尽量避免 BF16→FP32 回写 TMEM
//
// 11. L1::no_allocate ≈ standard (260 vs 261 GB/s):
//     统一内存下 L1 bypass 无额外收益 — LPDDR5X 无独立 L1 cache 效应
//
// 12. f32x2 SIMD FMA: 1.97× 加速
//     RMSNorm/SiLU/sigmoid 等标量循环改用 f32x2 可消减 50% 指令数
//
// 13. LDSM b8 极快 (222 GB/s/SM), STSM b8 较慢 (45 GB/s/SM)
//     量化路径的 INT8 SMEM↔register 传输已有基线
//
// 14. UMMA wait::ld/st + commit 开销极低 (1.1 / 7.5 / 12.6 ns)
//     Pipeline 同步可放心使用, 不影响吞吐
//
// 15. Reg Reconfig: 57.3 ns/pair (inc+dec)
//     Warp-specialization 每层仅需 1 次切换, 开销 <0.03% of decode step
//
// 16. PDL: launch_dependents 5.2ns + wait 1.1ns
//     64 层 forward 中 kernel 间隙可用 PDL 消除
//     理论省: 64 层 × ~5μs kernel launch gap = 320 μs → PDL 可减至 ~0

// ─────────────────────────────────────────────────────────────────────────────
// SM110a Roofline Model (基于实测数据)
// ─────────────────────────────────────────────────────────────────────────────
//
// 算力天花板 (compute ceilings):
//   UMMA BF16→FP32: 1.60 TFLOPS/SM × 20 SM = 32.0 TFLOPS (实测, sustained, CV=0.0%)
//   CUDA FP32:     128 cores/SM × 20 SM × 1.575 GHz × 2 = 8.06 TFLOPS (理论)
//   实测 FP32 标量吞吐约为理论 ~60-70%: ~5.0 TFLOPS (保守估计)
//
// 带宽天花板 (memory ceilings):
//   DRAM Read:  249 GB/s (实测 median, 20 SM, 91% of 273 peak, CV=3.3%)
//   DRAM Write: 149~177 GB/s (实测, 55-65% peak) ← 非对称!
//   L2:         ~3200 GB/s (@1-2MB fits-in-L2, CV<5%)
//   SMEM:       ~0.1 TB/s (实测, 128 threads, 受 bank conflict 影响)
//   TMEM:       store ~5 ns/col, load ~10.5 ns/col (片上, 不走 memory hierarchy)
//
// 算术强度临界点 (FLOP/Byte, DRAM Read-Limited):
//   UMMA:  32000 GFLOPS / 249 GB/s = **128.5 FLOP/Byte**
//   FP32:   5000 GFLOPS / 249 GB/s = **20.1 FLOP/Byte**
//   (低于此值为 bandwidth-bound, 高于为 compute-bound)
//
// ── 各操作 Roofline 分析 ──
//
// Operation                     | Data (bytes) | FLOPs       | AI (F/B) | Bound    | Time (理论)
// ------------------------------|-------------|-------------|----------|----------|------------
// Decode GEMV [1,K]×[K,N]      | K×N×2 (BF16)| 2×K×N       | 2.0      | BW       | data/249 GB/s
//   gate_proj [1,5120]×[5120,17408]| 178 MB   | 178 MFLOP   | 2.0      | BW       | 0.71 ms
//   全 64 层权重读取 ~51 GB     | 51 GB       | ~51 GFLOP   | 2.0      | BW       | 205 ms
//
// GDN rank-1 update [128×128]   | State 64KB  | 2×128²=33K  | 0.5      | BW       | N/A (片上)
//   当 state 在 TMEM: 0 DRAM IO | 0           | 33K         | ∞        | Compute  | ~6.5 ns
//
// GDN WY GEMM [C,128]×[128,128]| 2×C×128 BF16| 2×C×128²    | 128      | Compute* | C×128²×2/1.61T
//   C=64 (chunk_size):          | 16 KB       | 2.1 MFLOP   | 128      | Compute  | 1.3 μs/SM
//   C=256:                      | 64 KB       | 8.4 MFLOP   | 128      | Compute  | 5.2 μs/SM
//   * AI=128 > 125.8 临界点, 恰好跨入 compute-bound (UMMA 模式)
//   * 输入数据需从 GMEM→SMEM: TMA 32KB=371ns, 可与上一 chunk 计算重叠
//
// FullAttn QK^T [S,256]×[256,T] | S×256×2 BF16| 2×S×256×T   | 2×T      | "depends"
//   Decode (T=1, S=seq):        | S×512 BF16  | S×512       | 2.0      | BW       | data/256 GB/s
//   Prefill chunk (T=256):      | T×256×2     | 2×S×256×T   | S        | Compute  | if S≥128
//
// RMSNorm [1, 5120]             | 10 KB read  | 5120×3 ≈15K | 1.5      | L2/BW    | ~3 ns (L2)
// SiLU+Gate [1, 17408]          | 34 KB read  | 17408×2=35K | 1.0      | L2/BW    | ~10 ns (L2)
//   这些小 kernel 完全 fit in L2, 3.4 TB/s → 可忽略
//
// ── 关键结论 ──
//
// 1. Decode 完全 BW-bound (AI=2.0 << 19.5): 优化方向是 batch (读一次服务多 token)
// 2. WY prefill GEMM 在 AI=128 处, 恰好 UMMA compute-bound → 最大受益于 UMMA
// 3. GDN state TMEM 缓存: rank-1 update 变成纯 compute (AI=∞)
// 4. L2 带宽 3.4 TB/s >> DRAM 256 GB/s: fits-in-L2 操作时间可忽略
// 5. TMA LOAD + UMMA compute overlap: 几乎可隐藏 state/chunk 传输延迟

} // namespace sm110a
