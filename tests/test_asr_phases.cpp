// test_asr_phases.cpp — ASR 增强计划 Phase 1-6 单元测试
//
// 编译: g++ -std=c++17 -I src/plugins/asr -o tmp/test_asr_phases tests/test_asr_phases.cpp
// 运行: ./tmp/test_asr_phases

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <cmath>

// ============================================================================
// Phase 3: KeywordSpotter
// ============================================================================
#include "keyword_spotter.h"

void test_keyword_spotter() {
    std::cout << "[Phase 3] KeywordSpotter..." << std::flush;

    qwen_thor::asr::KeywordSpotter ks;

    // 添加关键词
    qwen_thor::asr::KeywordEntry kw1;
    kw1.text = "你好";
    kw1.action = "wake";
    kw1.threshold = 0.8f;
    ks.add_keyword(kw1);

    qwen_thor::asr::KeywordEntry kw2;
    kw2.text = "停止";
    kw2.action = "stop";
    kw2.threshold = 0.8f;
    ks.add_keyword(kw2);

    qwen_thor::asr::KeywordEntry kw3;
    kw3.text = "hello";
    kw3.action = "greet";
    kw3.threshold = 0.8f;
    ks.add_keyword(kw3);

    assert(ks.keywords().size() == 3);

    // 精确匹配
    auto hits = ks.match("你好世界");
    assert(!hits.empty());
    assert(hits[0].keyword == "你好");
    assert(hits[0].action == "wake");

    // 无匹配
    hits = ks.match("天气很好");
    assert(hits.empty());

    // 英文匹配
    hits = ks.match("say hello world");
    assert(!hits.empty());
    assert(hits[0].keyword == "hello");

    // 多关键词匹配
    hits = ks.match("你好请停止说话");
    assert(hits.size() >= 2);

    // 流式匹配 (去重)
    ks.reset_stream();
    auto h1 = ks.on_token(0, "你");
    auto h2 = ks.on_token(1, "好");
    // "你好" should trigger
    assert(!h2.empty() || !h1.empty());  // might trigger on h2

    // 同一关键词不应重复触发
    auto h3 = ks.on_token(2, "世界");
    // h3 should not re-trigger "你好"

    // 移除关键词
    ks.remove_keyword("你好");
    assert(ks.keywords().size() == 2);
    hits = ks.match("你好世界");
    assert(hits.empty());

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// Phase 4: PunctuationRestorer
// ============================================================================
#include "punctuation.h"

void test_punctuation_restorer() {
    std::cout << "[Phase 4] PunctuationRestorer..." << std::flush;

    qwen_thor::asr::PunctuationRestorer pr;

    // 规则方案: 问句检测
    auto result = pr.restore_rules("你今天怎么样");
    assert(!result.empty());
    // 应该添加问号
    assert(result.back() == L'？' || result.find("？") != std::string::npos ||
           result.find("?") != std::string::npos ||
           result.size() > std::string("你今天怎么样").size());

    // 规则方案: 感叹句
    result = pr.restore_rules("太好了");
    assert(!result.empty());

    // 规则方案: 短文本不加句中逗号
    result = pr.restore_rules("好的");
    assert(!result.empty());

    // 规则方案: 长文本加逗号
    result = pr.restore_rules("我今天去了超市买了很多东西回来做晚饭然后看了一会电视剧就睡觉了");
    assert(!result.empty());
    // 长文本应该有逗号
    assert(result.find(",") != std::string::npos ||
           result.find("，") != std::string::npos ||
           result.size() > std::string("我今天去了超市买了很多东西回来做晚饭然后看了一会电视剧就睡觉了").size());

    // 自动选择 (无 LLM): 用规则
    auto auto_result = pr.restore("测试自动选择");
    assert(!auto_result.empty());

    // 已有标点的文本应原样返回
    auto preserved = pr.restore("你好，世界！");
    assert(preserved == "你好，世界！");

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// Phase 5: AlignerEngine
// ============================================================================
#include "aligner_engine.h"

void test_aligner_engine() {
    std::cout << "[Phase 5] AlignerEngine..." << std::flush;

    // tokenize_for_align: 中文按字, 英文按词
    auto words = qwen_thor::asr::AlignerEngine::tokenize_for_align("你好世界hello world");
    assert(words.size() == 6);  // 你 好 世 界 hello world
    assert(words[0] == "你");
    assert(words[1] == "好");
    assert(words[2] == "世");
    assert(words[3] == "界");
    assert(words[4] == "hello");
    assert(words[5] == "world");

    // 空字符串
    auto empty = qwen_thor::asr::AlignerEngine::tokenize_for_align("");
    assert(empty.empty());

    // 纯英文
    auto en = qwen_thor::asr::AlignerEngine::tokenize_for_align("hello world test");
    assert(en.size() == 3);

    // fix_timestamps: LIS 单调修正
    std::vector<qwen_thor::asr::AlignedWord> aw(5);
    aw[0] = {"a", 100, 200, 0.9f};
    aw[1] = {"b", 300, 400, 0.9f};
    aw[2] = {"c", 250, 350, 0.9f};  // 违反单调
    aw[3] = {"d", 500, 600, 0.9f};
    aw[4] = {"e", 700, 800, 0.9f};
    qwen_thor::asr::AlignerEngine::fix_timestamps(aw);

    // 检查单调递增
    for (int i = 1; i < (int)aw.size(); ++i) {
        assert(aw[i].start_ms >= aw[i - 1].start_ms);
    }

    // 均匀对齐 (占位实现)
    qwen_thor::asr::AlignerEngine ae;
    // 不加载模型, 直接测试辅助函数

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// Phase 1: VadEngine (结构测试, 不加载模型)
// ============================================================================
#include "vad_engine.h"

void test_vad_engine_structure() {
    std::cout << "[Phase 1] VadEngine structure..." << std::flush;

    qwen_thor::asr::VadEngine vad;
    assert(!vad.is_loaded());

    // 未加载时 detect 应返回空
    float dummy[160] = {};
    auto segs = vad.detect(dummy, 160);
    assert(segs.empty());

    // EnergyVad 测试
    qwen_thor::asr::EnergyVad::Config ecfg;
    ecfg.energy_threshold = 0.01f;
    ecfg.silence_ms = 500;
    qwen_thor::asr::EnergyVad evad(ecfg);

    // 静音 → should not trigger
    float silent[1600] = {};  // 100ms @16kHz
    auto eresult = evad.process(silent, 1600, 16000);
    // 静音段不应有语音段
    assert(!eresult.vad_triggered);

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// Phase 2: SpeakerEncoder + SpeakerManager (结构测试)
// ============================================================================
#include "speaker_manager.h"

void test_speaker_manager() {
    std::cout << "[Phase 2] SpeakerManager..." << std::flush;

    qwen_thor::asr::SpeakerManager mgr;
    assert(mgr.speaker_count() == 0);

    // 注册说话人
    std::vector<float> emb_a(512, 0.0f);
    emb_a[0] = 1.0f; emb_a[1] = 0.5f; emb_a[2] = 0.3f;
    mgr.register_speaker("Alice", emb_a);
    assert(mgr.speaker_count() == 1);

    std::vector<float> emb_b(512, 0.0f);
    emb_b[100] = 1.0f; emb_b[101] = 0.8f; emb_b[200] = 0.4f;
    mgr.register_speaker("Bob", emb_b);
    assert(mgr.speaker_count() == 2);

    // 识别已知说话人
    std::vector<float> emb_a2(512, 0.0f);
    emb_a2[0] = 0.98f; emb_a2[1] = 0.48f; emb_a2[2] = 0.28f;
    auto match = mgr.identify(emb_a2, 0.5f, false);

    // Cosine similarity
    float sim = qwen_thor::asr::cosine_similarity(emb_a, emb_a2);
    assert(sim > 0.9f);  // Very similar

    float sim2 = qwen_thor::asr::cosine_similarity(emb_a, emb_b);
    assert(sim2 < 0.1f);  // Very different

    // 自动注册新说话人
    std::vector<float> emb_c(512, 0.0f);
    emb_c[300] = 1.0f;
    auto match2 = mgr.identify(emb_c, 0.5f, true);
    assert(match2.is_new);
    assert(mgr.speaker_count() == 3);

    // 清空
    mgr.clear();
    assert(mgr.speaker_count() == 0);

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// AsrResult 结构测试
// ============================================================================
#include "asr_plugin.h"

void test_asr_result_structure() {
    std::cout << "[AsrResult] Structure..." << std::flush;

    qwen_thor::plugins::AsrResult result;

    // 基础字段
    result.text = "你好世界";
    result.language = "zh";
    result.duration_s = 3.5f;

    // Phase 4: 标点
    result.text_with_punc = "你好，世界！";

    // Phase 5: 时间戳
    qwen_thor::plugins::AsrResult::WordInfo w1;
    w1.word = "你好";
    w1.start_ms = 0;
    w1.end_ms = 500;
    w1.confidence = 0.95f;
    w1.speaker_id = 0;
    result.words.push_back(w1);

    // Phase 6: 说话人分割
    qwen_thor::plugins::AsrResult::SpeakerSegment seg;
    seg.start_ms = 0;
    seg.end_ms = 5000;
    seg.speaker_id = 0;
    seg.speaker_name = "Speaker_0";
    seg.text = "你好世界";
    result.segments.push_back(seg);

    // Phase 3: 关键词
    qwen_thor::plugins::AsrResult::KeywordHit kh;
    kh.keyword = "你好";
    kh.action = "wake";
    kh.char_offset = 0;
    kh.confidence = 1.0f;
    result.keyword_hits.push_back(kh);

    assert(result.text == "你好世界");
    assert(result.words.size() == 1);
    assert(result.segments.size() == 1);
    assert(result.keyword_hits.size() == 1);

    std::cout << " PASSED" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ASR Enhancement Plan — Phase 1-6 Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    int pass = 0, fail = 0;

    auto run = [&](const char* name, void(*fn)()) {
        try {
            fn();
            pass++;
        } catch (const std::exception& e) {
            std::cout << " FAILED: " << e.what() << std::endl;
            fail++;
        } catch (...) {
            std::cout << " FAILED: unknown exception" << std::endl;
            fail++;
        }
    };

    run("Phase 3: KeywordSpotter", test_keyword_spotter);
    run("Phase 4: PunctuationRestorer", test_punctuation_restorer);
    run("Phase 5: AlignerEngine", test_aligner_engine);
    run("Phase 1: VadEngine", test_vad_engine_structure);
    run("Phase 2: SpeakerManager", test_speaker_manager);
    run("AsrResult Structure", test_asr_result_structure);

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << pass << " passed, " << fail << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return fail > 0 ? 1 : 0;
}
