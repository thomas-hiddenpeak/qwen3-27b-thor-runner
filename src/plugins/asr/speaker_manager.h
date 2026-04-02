// speaker_manager.h — 说话人注册/识别管理
//
// 在线说话人管理: 注册、匹配、碎片吸收/合并。
// 与具体 encoder 无关, 仅操作 embedding 向量。

#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdio>
#include <utility>

namespace qwen_thor {
namespace asr {

// 余弦相似度
inline float cosine_similarity(const std::vector<float>& a,
                               const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0;
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-12f);
}

// ============================================================================
// SpeakerManager — 说话人注册/识别管理
// ============================================================================
class SpeakerManager {
public:
    struct MatchResult {
        std::string name;
        int   speaker_id  = -1;
        float similarity  = 0;
        bool  is_new      = false;
    };

    SpeakerManager() = default;

    // 注册说话人
    void register_speaker(const std::string& name, const std::vector<float>& embedding) {
        Speaker s;
        s.name = name;
        s.embedding = embedding;
        s.id = next_id_++;
        s.seen_count = 1;
        speakers_.push_back(s);
    }

    // 识别: 返回最匹配的说话人 (或注册新说话人)
    MatchResult identify(const std::vector<float>& embedding,
                         float threshold = 0.65f,
                         bool auto_register = true) {
        MatchResult best;
        best.similarity = -1;

        for (auto& s : speakers_) {
            float sim = cosine_similarity(embedding, s.embedding);
            if (sim > best.similarity) {
                best.similarity = sim;
                best.name = s.name;
                best.speaker_id = s.id;
            }
        }

        if (best.similarity >= threshold) {
            // 更新 embedding (moving average)
            for (auto& s : speakers_) {
                if (s.id == best.speaker_id) {
                    update_embedding(s, embedding, 0.1f);
                    break;
                }
            }
            best.is_new = false;
            return best;
        }

        // 新说话人
        if (auto_register) {
            best.speaker_id = next_id_;
            best.name = "Speaker_" + std::to_string(next_id_);
            best.is_new = true;
            register_speaker(best.name, embedding);
        } else {
            best.speaker_id = -1;
            best.name = "Unknown";
            best.is_new = false;
        }
        return best;
    }

    // 列表
    int speaker_count() const { return (int)speakers_.size(); }

    // 获取所有说话人名称
    std::vector<std::string> speaker_names() const {
        std::vector<std::string> names;
        for (const auto& s : speakers_) names.push_back(s.name);
        return names;
    }

    // 按名称删除说话人
    bool remove_by_name(const std::string& name) {
        for (auto it = speakers_.begin(); it != speakers_.end(); ++it) {
            if (it->name == name) {
                speakers_.erase(it);
                return true;
            }
        }
        return false;
    }

    // 按名称获取 embedding
    std::vector<float> get_embedding(const std::string& name) const {
        for (const auto& s : speakers_) {
            if (s.name == name) return s.embedding;
        }
        return {};
    }

    // 按 ID 获取名称和 embedding
    std::pair<std::string, std::vector<float>> get_embedding_by_id(int id) const {
        for (const auto& s : speakers_) {
            if (s.id == id) return {s.name, s.embedding};
        }
        return {};
    }

    // 重置
    void clear() {
        speakers_.clear();
        next_id_ = 0;
    }

    // 二次聚类: 碎片吸收 + 已确立合并
    std::unordered_map<int, int> merge_similar(float merge_threshold = 0.55f,
                                                int min_established = 5,
                                                float established_merge_threshold = -1.0f) {
        std::unordered_map<int, int> id_map;
        for (auto& s : speakers_) id_map[s.id] = s.id;

        if (speakers_.size() > 1) {
            fprintf(stderr, "[SpeakerMerge] pairwise similarity (fragment absorption, "
                    "threshold=%.2f, min_established=%d):\n",
                    merge_threshold, min_established);
            for (size_t i = 0; i < speakers_.size(); ++i) {
                const char* tag_i = speakers_[i].seen_count >= min_established ? "E" : "F";
                for (size_t j = i + 1; j < speakers_.size(); ++j) {
                    const char* tag_j = speakers_[j].seen_count >= min_established ? "E" : "F";
                    float sim = cosine_similarity(
                        speakers_[i].embedding, speakers_[j].embedding);
                    fprintf(stderr, "  %s[%s]↔%s[%s]: %.3f (seen %d,%d)\n",
                        speakers_[i].name.c_str(), tag_i,
                        speakers_[j].name.c_str(), tag_j,
                        sim, speakers_[i].seen_count, speakers_[j].seen_count);
                }
            }
        }

        std::vector<size_t> established_idx, fragment_idx;
        for (size_t i = 0; i < speakers_.size(); ++i) {
            if (speakers_[i].seen_count >= min_established)
                established_idx.push_back(i);
            else
                fragment_idx.push_back(i);
        }
        fprintf(stderr, "[SpeakerMerge] %zu established, %zu fragments\n",
                established_idx.size(), fragment_idx.size());

        if (established_idx.empty()) return id_map;

        std::vector<size_t> to_remove;
        for (size_t fi : fragment_idx) {
            float best_sim = -1;
            size_t best_ei = 0;
            for (size_t ei : established_idx) {
                float sim = cosine_similarity(
                    speakers_[fi].embedding, speakers_[ei].embedding);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_ei = ei;
                }
            }

            if (best_sim >= merge_threshold) {
                fprintf(stderr, "[SpeakerMerge] absorb %s (seen %d) → %s (sim %.3f)\n",
                        speakers_[fi].name.c_str(), speakers_[fi].seen_count,
                        speakers_[best_ei].name.c_str(), best_sim);
                int old_id = speakers_[fi].id;
                int new_id = speakers_[best_ei].id;
                for (auto& [k, v] : id_map) {
                    if (v == old_id) v = new_id;
                }
                auto& si = speakers_[best_ei];
                auto& sj = speakers_[fi];
                float wi = (float)si.seen_count;
                float wj = (float)sj.seen_count;
                float total = wi + wj;
                for (size_t k = 0; k < si.embedding.size() && k < sj.embedding.size(); ++k) {
                    si.embedding[k] = (wi * si.embedding[k] + wj * sj.embedding[k]) / total;
                }
                float norm = 0;
                for (float v : si.embedding) norm += v * v;
                norm = sqrtf(norm + 1e-12f);
                for (float& v : si.embedding) v /= norm;
                si.seen_count = (int)total;
                to_remove.push_back(fi);
            } else {
                fprintf(stderr, "[SpeakerMerge] keep %s (seen %d, best sim %.3f < %.2f)\n",
                        speakers_[fi].name.c_str(), speakers_[fi].seen_count,
                        best_sim, merge_threshold);
            }
        }

        std::sort(to_remove.rbegin(), to_remove.rend());
        for (size_t idx : to_remove)
            speakers_.erase(speakers_.begin() + idx);

        // Pass 2: established-established merge
        if (established_merge_threshold > 0) {
            bool merged_any = true;
            while (merged_any) {
                merged_any = false;
                float best_sim = -1;
                size_t best_i = 0, best_j = 0;
                for (size_t i = 0; i < speakers_.size(); ++i) {
                    if (speakers_[i].seen_count < min_established) continue;
                    for (size_t j = i + 1; j < speakers_.size(); ++j) {
                        if (speakers_[j].seen_count < min_established) continue;
                        float sim = cosine_similarity(
                            speakers_[i].embedding, speakers_[j].embedding);
                        if (sim > best_sim) {
                            best_sim = sim;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                if (best_sim >= established_merge_threshold) {
                    size_t keep = best_i, absorb = best_j;
                    if (speakers_[keep].seen_count < speakers_[absorb].seen_count)
                        std::swap(keep, absorb);
                    fprintf(stderr, "[SpeakerMerge] merge established %s (seen %d) → %s (seen %d, sim %.3f)\n",
                            speakers_[absorb].name.c_str(), speakers_[absorb].seen_count,
                            speakers_[keep].name.c_str(), speakers_[keep].seen_count, best_sim);
                    int old_id = speakers_[absorb].id;
                    int new_id = speakers_[keep].id;
                    for (auto& [k, v] : id_map) {
                        if (v == old_id) v = new_id;
                    }
                    auto& si = speakers_[keep];
                    auto& sj = speakers_[absorb];
                    float wi = (float)si.seen_count, wj = (float)sj.seen_count;
                    float total = wi + wj;
                    for (size_t k = 0; k < si.embedding.size() && k < sj.embedding.size(); ++k)
                        si.embedding[k] = (wi * si.embedding[k] + wj * sj.embedding[k]) / total;
                    float norm = 0;
                    for (float v : si.embedding) norm += v * v;
                    norm = sqrtf(norm + 1e-12f);
                    for (float& v : si.embedding) v /= norm;
                    si.seen_count = (int)total;
                    speakers_.erase(speakers_.begin() + absorb);
                    merged_any = true;
                }
            }
        }

        return id_map;
    }

private:
    struct Speaker {
        std::string name;
        std::vector<float> embedding;
        int id = 0;
        int seen_count = 0;
    };

    std::vector<Speaker> speakers_;
    int next_id_ = 0;

    void update_embedding(Speaker& s, const std::vector<float>& new_emb, float alpha) {
        for (size_t i = 0; i < s.embedding.size() && i < new_emb.size(); ++i) {
            s.embedding[i] = (1 - alpha) * s.embedding[i] + alpha * new_emb[i];
        }
        float norm = 0;
        for (float v : s.embedding) norm += v * v;
        norm = sqrtf(norm + 1e-12f);
        for (float& v : s.embedding) v /= norm;
        s.seen_count++;
    }
};

} // namespace asr
} // namespace qwen_thor
