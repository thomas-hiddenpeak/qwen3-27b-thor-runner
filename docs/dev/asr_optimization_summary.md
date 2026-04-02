# ASR V4 优化方案总结 & 后续行动计划

> **创建**: 2026-03-16  
> **参考**: tmp/asr_eval_report.md, docs/ASR_PIPELINE_STATUS.md, tmp/asr_strategy_discussion.md

---

## 一、用户意见融合

### 核心原则 (已确认)
1. ✓ **分段忠实性**: 同一说话人无打断时不额外分段，有打断必须体现
2. ✓ **标点方案**: 优先 ASR 本身能力，次选 FunASR 模型
3. ✓ **口语规范化**: API 可选，默认关闭，后续 LLM 处理
4. ✓ **音频前处理**: 保持现状 (resample 充分)
5. ⚠ **并行执行**: Phase 2&3 可能有 25% 加速，但需评估 GPU 资源

---

## 二、当前现状总结

### 质量指标 (asrTest.m4a)
| 指标 | 当前 | 目标 |
|------|------|------|
| 段落数 | 22 | 保持 (不人为拆分) |
| 说话人数 | 4 | ✓ 合理 (从 7→4) |
| 段尾标点 | 100% | ✓ 满足 |
| 跨段断句 | 0 | ✓ 满足 |
| 句内标点质量 | 70% (规则限制) | 需改进 |
| 口语冗余 | 忠实转写 | ✓ 可选规范化 |

### Pipeline 依赖关系
```
ASR (Phase 1) ──→ ForcedAligner (Phase 2)
     ↓                    ↓
   (并行可能)      Phase 4 (分配 speaker)
     ↑                    ↑
CAM++ (Phase 3) ────────────
```

**关键发现**: Phase 2&3 互不依赖，可并行执行 → 预期 25% 加速

---

## 三、5 个优化方向优先级调整

### 方向 A: 标点质量 ⚠ 重新设计
**修订**: 
1. **P0 (验证)**: 确认 ASR 是否有标点能力
   - 方法: 单个音频片段测试 (现有 server)
   - 预期: 知晓是否需要额外标点模型
   
2. **P1 (若 ASR 无标点)**: 集成标点模型
   - **选项 A**: FunASR 轻量标点模型 (推荐)
   - **选项 B**: 改进规则引擎
   - **工作量**: 1-2 周

### 方向 B: 说话人分离精度
**修订**: Phase 3b 已实现基础版本 ✓
- 时序平滑可快速加强 P0 (1 小时)
- 动态 threshold 可作 P2 (可选)

### 方向 C: 长段拆分 ⚠ 用户未确认需求
**修订**:
- **待确认**: 是否需要? 若需，按何标准?
  - 我们的建议: **保持现状**, 不人为拆分
  - 若阅读体验差，让消费端 (LLM) 处理
  
- **备选**: 若要拆分，规则为
  - 同一 speaker, 累计 >N 秒时按句号拆分 (N=60s 推荐)

### 方向 D: 口语规范化
**修订**: **D3 (API 可选)** ✓
- 默认保留原始转写
- `clean=true` 时触发规则去重
- 实现成本: <2h

### 方向 E: 端到端延迟
**修订**: Phase 2&3 并行 (P1, 2-3h)
- 预期收益: ASR + ForcedAligner 从串行 ~90s → 并行 ~70s
- 风险: GPU 资源冲突 (需测试)

---

## 四、立即行动清单 (Next 48h)

### ✓ 已完成
- [x] 碎片吸收聚类 (Phase 3b) — commit `2028212`
- [x] 完整评估报告 — tmp/asr_eval_report.md
- [x] 优化方向分析 — docs/ASR_PIPELINE_STATUS.md
- [x] 深度讨论文档 — tmp/asr_strategy_discussion.md

### 待执行 (优先级)

#### P0 (需要用户确认)
- [ ] **分段策略**: 保持现状 (不人为拆分), 还是按 N 秒阈值拆分?
- [ ] **标点方案**: 验证 ASR 标点能力后再决定 (FunASR 还是改进规则)
- [ ] **并行执行**: 是否愿意承受 GPU 资源风险换取 25% 加速?

#### P1 (验证 ASR 标点，1 小时)
```bash
# 操作步骤:
1. 创建 tests/test_asr_punctuation.py
2. 调用现有 /v1/audio/transcriptions 端点 (plain mode)
3. 检查输出是否包含标点 (。？！等)
4. 通过多个例子 (陈述/疑问/感叹) 验证模式
```

#### P2 (说话人时序平滑，1 小时)
```cpp
// 在 Phase 5 后增加:
void smooth_speaker_islands(std::vector<Segment>& segments) {
    // Rule: 若某 segment 时长 < 5s 且被同一 speaker 前后包围,
    //       合并到前后 speaker (取出现频次多的)
}
```

#### P3 (Phase 2&3 并行，2-3 小时)
```cpp
// 在 serve.cpp Phase 1 结束后:
std::thread t2([...] { Phase_2_ForcedAligner(); });
std::thread t3([...] { Phase_3_CAM_Plus(); });
t2.join();
t3.join();
// Phase 4 继续
```

#### P4 (口语规范化 API，2 小时)
```cpp
// API 参数: ?clean=true
bool clean_text = form.fields.count("clean") && form.fields["clean"] == "true";
if (clean_text) {
    full_text = denoise_oral_speech(full_text);  // 规则去重
}
```

---

## 五、决策关键点 (需与用户讨论)

### Q1: 长段拆分
**现状**: 22 段，最长 86s
**选项 A**: 保持现状，不拆分
```
优点: 保证说话人连续
缺点: 段落过长 (但可由消费端处理)
```

**选项 B**: 同一说话人超 60s 按句号拆分
```
优点: 阅读体验改善
缺点: 人为打破连续性
```

**建议**: 选项 A (保持现状)

---

### Q2: 标点方案
**现状**: 规则引擎 (PunctuationRestorer)
**验证**: 需确认 ASR 是否已输出标点

**调查结果待补充** (需运行 P0 测试)

---

### Q3: 并行执行影响
**Qwen3-ASR**: GPU 内存占用 ~8-10GB, VRAM 密集 (内存带宽)
**CAM++**: GPU 内存占用 ~2-3GB, 算力密集 (Tensor cores)

**风险**:
- 同时运行可能触发统一内存 page fault
- 需进行性能实测

**建议**: 先运行 P1&P2 的小投入改进，再考虑并行

---

## 六、后续迭代计划

### 短期 (1 周)
- P0: ASR 标点验证
- P1: 说话人时序平滑
- P2: 口语规范化 API

### 中期 (2 周)
- 标点模型集成 (若 P0 验证需要)
- Phase 2&3 并行 (若性能测试可行)

### 长期 (1 月)
- 多语言支持
- 端到端微调 (若需提升标点质量)

---

## 七、建议的讨论议题

1. **分段策略**: A (保持现状) 还是 B (N 秒拆分)?
2. **标点方案**: 等待 P0 验证后决定
3. **并行执行**: 是否接受 GPU 资源风险?
4. **优先级**: 口语规范化 > 时序平滑 > 并行?

---

## 附: 相关文件位置

| 文件 | 说明 |
|------|------|
| tmp/asr_eval_report.md | 评估报告 (22 段详情 + 质量指标) |
| docs/ASR_PIPELINE_STATUS.md | V4 架构 + 优化历程 |
| tmp/asr_strategy_discussion.md | 深度分析 (Pipeline 依赖 + 方案对比) |
| src/serve/serve.cpp | V4 pipeline 实现 (~L3860-4560) |
| src/plugins/asr/speaker_encoder.h | SpeakerManager + Phase 3b |
| src/plugins/asr/punctuation.h | 标点恢复规则引擎 |

---

**总体评估**: ASR V4 pipeline 已达到生产级别 (7/10 综合分数)。后续优化主要聚焦标点质量和说话人精度微调，无需大规模重构。
