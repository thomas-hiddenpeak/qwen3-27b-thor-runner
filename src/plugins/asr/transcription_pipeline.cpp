// transcription_pipeline.cpp — V4/V2/Plain 转录管线实现
//
// 从 serve.cpp handle_audio_transcriptions() 提取的全部 pipeline 逻辑。

#include "transcription_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <set>

#include <cuda_runtime.h>

namespace qwen_thor {
namespace asr {

// ============================================================================
// 统一入口: 根据参数和可用组件选择 V4 / Plain
// ============================================================================
TranscriptionResult TranscriptionPipeline::transcribe(
        const audio::AudioData& wav, const TranscriptionParams& params) {
    bool has_speaker = deps_.speaker_encoder != nullptr || deps_.eres2netv2_encoder != nullptr || deps_.eres2netv2_gpu_encoder != nullptr;
    bool has_vad = (deps_.vad_engine && deps_.vad_engine->is_loaded()) ||
                   (deps_.gpu_vad_engine && deps_.gpu_vad_engine->is_loaded());

    if (params.identify_speaker && has_speaker && has_vad) {
        return run_v4_pipeline(wav, params);
    }
    return run_plain_mode(wav, params);
}

// ============================================================================
// ASR with energy-based split (shared by V4 Phase 1 and Plain mode)
// ============================================================================
std::string TranscriptionPipeline::run_asr_with_energy_split(
        const audio::AudioData& wav, const std::string& language) {
    float total_duration_s = (float)wav.samples.size() / wav.sample_rate;
    std::string full_text;

    if (total_duration_s > 100.0f) {
        const int sr = wav.sample_rate;
        const int total_samples = (int)wav.samples.size();
        const int target_chunk_samples = 90 * sr;
        const int search_window = 10 * sr;
        const int energy_window = (int)(0.1f * sr);

        std::vector<int> split_points;
        split_points.push_back(0);

        int pos = 0;
        while (pos + target_chunk_samples < total_samples) {
            int center = pos + target_chunk_samples;
            int search_start = std::max(pos + target_chunk_samples / 2, center - search_window);
            int search_end = std::min(total_samples - energy_window, center + search_window);

            if (search_start >= search_end) {
                split_points.push_back(std::min(center, total_samples));
                pos = split_points.back();
                continue;
            }

            float min_energy = 1e30f;
            int best_pos = center;
            int step = std::max(1, energy_window / 4);
            for (int s = search_start; s < search_end; s += step) {
                float energy = 0.0f;
                int end = std::min(s + energy_window, total_samples);
                for (int k = s; k < end; k++) {
                    float v = wav.samples[k];
                    energy += v * v;
                }
                if (energy < min_energy) {
                    min_energy = energy;
                    best_pos = s + energy_window / 2;
                }
            }
            split_points.push_back(best_pos);
            pos = best_pos;
        }
        split_points.push_back(total_samples);

        int num_chunks = (int)split_points.size() - 1;
        fprintf(stderr, "[Pipeline] Energy split: %d chunks (target=90s, total=%.1fs)\n",
                num_chunks, total_duration_s);

        std::vector<std::vector<float>> all_chunk_pcms;
        for (int ci = 0; ci < num_chunks; ci++) {
            int s = split_points[ci];
            int e = split_points[ci + 1];
            if (e - s < sr / 5) continue;
            all_chunk_pcms.emplace_back(wav.samples.begin() + s, wav.samples.begin() + e);
        }

        auto* native_plugin = dynamic_cast<plugins::NativeAsrPlugin*>(deps_.asr_plugin);
        if (native_plugin && all_chunk_pcms.size() >= 2) {
            std::vector<plugins::NativeAsrPlugin::PcmChunk> pcm_chunks;
            for (auto& pcm : all_chunk_pcms)
                pcm_chunks.push_back({pcm.data(), (int)pcm.size()});
            auto batch_results = native_plugin->transcribe_batch_pcm(
                pcm_chunks, wav.sample_rate, language, true);
            for (auto& r : batch_results)
                if (r.error_code == 0 && !r.text.empty())
                    full_text += r.text;
        } else {
            for (auto& chunk_pcm : all_chunk_pcms) {
                auto seg_result = deps_.asr_plugin->transcribe_pcm(
                    chunk_pcm.data(), (int)chunk_pcm.size(), wav.sample_rate, language, true);
                if (seg_result.error_code == 0 && !seg_result.text.empty())
                    full_text += seg_result.text;
            }
        }
    } else {
        auto result = deps_.asr_plugin->transcribe_pcm(
            wav.samples.data(), (int)wav.samples.size(), wav.sample_rate, language, true);
        if (result.error_code == 0) full_text = result.text;
    }

    return full_text;
}

// ============================================================================
// Forced Alignment (V4 Phase 2 / Plain mode word timestamps)
// ============================================================================
std::vector<AlignedWord> TranscriptionPipeline::run_forced_alignment(
        const audio::AudioData& wav, const std::string& text) {
    std::vector<AlignedWord> words;
    if (!deps_.aligner_engine || !deps_.aligner_engine->is_loaded() || text.empty())
        return words;

    const int max_align_samples = wav.sample_rate * 180;
    if ((int)wav.samples.size() <= max_align_samples) {
        std::lock_guard<std::mutex> lock(*deps_.aligner_mutex);
        words = deps_.aligner_engine->align(
            wav.samples.data(), (int)wav.samples.size(),
            wav.sample_rate, text, "Chinese");
    } else {
        const int seg_samples = wav.sample_rate * 150;
        auto all_chars = AlignerEngine::tokenize_for_align(text);
        int total_chars = (int)all_chars.size();
        int total_samples = (int)wav.samples.size();
        int num_segs = (total_samples + seg_samples - 1) / seg_samples;
        int chars_per_seg = (total_chars + num_segs - 1) / num_segs;
        int char_offset = 0;
        for (int si = 0; si < num_segs && char_offset < total_chars; ++si) {
            int sample_start = si * seg_samples;
            int sample_end = std::min(sample_start + seg_samples, total_samples);
            int seg_chars = std::min(chars_per_seg, total_chars - char_offset);
            std::string seg_text;
            for (int ci = char_offset; ci < char_offset + seg_chars; ++ci)
                seg_text += all_chars[ci];
            int offset_ms = (int)((float)sample_start / wav.sample_rate * 1000);
            std::vector<AlignedWord> seg_aligned;
            {
                std::lock_guard<std::mutex> lock(*deps_.aligner_mutex);
                seg_aligned = deps_.aligner_engine->align(
                    wav.samples.data() + sample_start, sample_end - sample_start,
                    wav.sample_rate, seg_text, "Chinese");
            }
            for (auto& w : seg_aligned) {
                w.start_ms += offset_ms;
                w.end_ms += offset_ms;
                words.push_back(w);
            }
            char_offset += seg_chars;
        }
    }

    return words;
}

// ============================================================================
// V4 Pipeline: ASR-first + ForcedAligner + CAM++ spectral clustering
// ============================================================================
TranscriptionResult TranscriptionPipeline::run_v4_pipeline(
        const audio::AudioData& wav, const TranscriptionParams& params) {
    TranscriptionResult result;
    float total_duration_s = (float)wav.samples.size() / wav.sample_rate;
    result.duration_s = total_duration_s;

    fprintf(stderr, "[Pipeline] v4: audio %.1fs, %zu samples\n",
            total_duration_s, wav.samples.size());

    // 每次请求清空说话人注册
    if (deps_.speaker_manager && deps_.speaker_mutex) {
        std::lock_guard<std::mutex> lock(*deps_.speaker_mutex);
        deps_.speaker_manager->clear();
    }

    auto v4_t0 = std::chrono::steady_clock::now();

    // ================================================================
    // Phase 1: ASR (energy split for long audio)
    // ================================================================
    auto phase_t0 = v4_t0;
    std::string full_text = run_asr_with_energy_split(wav, params.language);

    if (full_text.empty()) {
        result.error_code = 1;
        result.error_message = "ASR transcription produced no text";
        return result;
    }

    fprintf(stderr, "[Pipeline] v4 Phase 1: ASR text = %zu chars (%.1fs)\n",
            full_text.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_t0).count());

    // ================================================================
    // Phase 2 & 3: 并行执行 ForcedAligner || VAD+CAM++
    // ================================================================
    struct SpkInterval {
        int start_ms, end_ms;
        int speaker_id;
        std::string speaker_name;
    };
    std::vector<AlignedWord> aligned_words;
    std::vector<SpkInterval> spk_intervals;
    int phase3_speaker_count = 0;
    auto phase2_t0 = std::chrono::steady_clock::now();

    // Phase 2 在后台线程执行
    auto phase2_future = std::async(std::launch::async, [&]() -> std::vector<AlignedWord> {
        return run_forced_alignment(wav, full_text);
    });

    // Phase 3 在主线程并发执行
    {
        auto phase3_t0 = std::chrono::steady_clock::now();

        // Step 3a: VAD
        struct VadResult { int start_ms; int end_ms; };
        std::vector<VadResult> vad_results;
        auto vad_t0 = std::chrono::steady_clock::now();

        if (deps_.gpu_vad_engine && deps_.gpu_vad_engine->is_loaded()) {
            auto gpu_segs = deps_.gpu_vad_engine->detect_all(
                wav.samples.data(), (int)wav.samples.size(), 300, 8000);
            for (auto& gs : gpu_segs)
                vad_results.push_back({gs.start_ms, gs.end_ms});
        } else if (deps_.vad_engine) {
            std::lock_guard<std::mutex> lock(*deps_.vad_mutex);
            auto& cfg = deps_.vad_engine->mutable_config();
            int orig_silence = cfg.max_end_silence_time;
            int orig_segment = cfg.max_single_segment_time;
            cfg.max_end_silence_time = 300;
            cfg.max_single_segment_time = 8000;
            auto vad_segments = deps_.vad_engine->detect_all(wav.samples.data(), (int)wav.samples.size());
            cfg.max_end_silence_time = orig_silence;
            cfg.max_single_segment_time = orig_segment;
            for (auto& vs : vad_segments)
                vad_results.push_back({vs.start_ms, vs.end_ms});
        }

        double vad_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - vad_t0).count();

        // Step 3b: Mel + CAM++ speaker embedding (chunk-level → spectral clustering)
        auto mel_t0 = std::chrono::steady_clock::now();
        int mel_segments = 0;
        std::vector<std::vector<float>> seg_embeddings;

        struct ChunkInfo { int abs_start_ms, abs_end_ms; };
        std::vector<ChunkInfo> chunk_infos;
        float total_speech_sec = 0;

        const int CHUNK_FRAMES = 300;
        const int MIN_CHUNK_FRAMES = 150;

        // === Batch GPU extraction (CAM++ or ERes2NetV2 GPU) ===
        using CamBatchChunk = GpuSpeakerEncoder::BatchChunk;
        using ERes2BatchChunk = GpuERes2NetV2Encoder::BatchChunk;
        std::vector<CamBatchChunk> cam_batch_chunks;
        std::vector<ERes2BatchChunk> eres2_batch_chunks;
        std::vector<ChunkInfo> batch_chunk_infos;

        bool use_gpu_mel = deps_.gpu_mel && deps_.gpu_mel->is_initialized();
        bool use_cam_batch = use_gpu_mel && deps_.speaker_encoder && !deps_.eres2netv2_gpu_encoder && !deps_.eres2netv2_encoder;
        bool use_eres2_gpu_batch = use_gpu_mel && deps_.eres2netv2_gpu_encoder;
        bool use_gpu_batch = use_cam_batch || use_eres2_gpu_batch;
        float* d_batch_mels = nullptr;
        int batch_mel_capacity = (int)(total_duration_s * 100 + 10000) * 80;
        if (use_gpu_batch) {
            if (cudaMalloc(&d_batch_mels, (size_t)batch_mel_capacity * sizeof(float)) != cudaSuccess) {
                fprintf(stderr, "[Pipeline] batch mel buffer alloc failed, falling back to serial\n");
                use_gpu_batch = false;
                use_cam_batch = false;
                use_eres2_gpu_batch = false;
            }
        }
        int batch_mel_offset = 0;
        int embed_chunk_count = 0;

        for (size_t vi = 0; vi < vad_results.size(); ++vi) {
            auto& vr = vad_results[vi];
            if (vr.end_ms - vr.start_ms < 200) continue;

            int64_t start_sample = (int64_t)vr.start_ms * wav.sample_rate / 1000;
            int64_t end_sample = (int64_t)vr.end_ms * wav.sample_rate / 1000;
            start_sample = std::max((int64_t)0, std::min(start_sample, (int64_t)wav.samples.size()));
            end_sample = std::max(start_sample, std::min(end_sample, (int64_t)wav.samples.size()));
            int seg_samples = (int)(end_sample - start_sample);
            if (seg_samples < 1600) continue;

            const float* seg_pcm = wav.samples.data() + start_sample;

            float rms = 0.0f;
            for (int i = 0; i < seg_samples; ++i) rms += seg_pcm[i] * seg_pcm[i];
            rms = std::sqrt(rms / seg_samples);
            if (rms < 0.005f) continue;

            int num_frames = 0;
            float* d_mel = nullptr;
            std::vector<float> mel;
            bool need_cpu_mel = deps_.eres2netv2_encoder && !deps_.eres2netv2_gpu_encoder;
            if (deps_.gpu_mel && deps_.gpu_mel->is_initialized() && !need_cpu_mel) {
                auto mel_result = deps_.gpu_mel->compute_gpu(seg_pcm, seg_samples);
                num_frames = mel_result.num_frames;
                d_mel = mel_result.d_mel;
                deps_.gpu_mel->sync();
            } else {
                SpeakerService::compute_mel_80(seg_pcm, seg_samples, wav.sample_rate, mel, num_frames);
            }
            if (num_frames < 10) continue;
            ++mel_segments;
            total_speech_sec += (vr.end_ms - vr.start_ms) / 1000.0f;

            // 拆成 3s chunk
            std::vector<std::pair<int,int>> chunk_ranges;
            if (num_frames <= CHUNK_FRAMES + MIN_CHUNK_FRAMES) {
                chunk_ranges.push_back({0, num_frames});
            } else {
                int pos = 0;
                while (pos < num_frames) {
                    int end = pos + CHUNK_FRAMES;
                    if (num_frames - end < MIN_CHUNK_FRAMES) end = num_frames;
                    chunk_ranges.push_back({pos, std::min(end, num_frames)});
                    pos = end;
                    if (pos >= num_frames) break;
                }
            }

            const float MS_PER_FRAME = 10.0f;
            for (auto& [f_start, f_end] : chunk_ranges) {
                int chunk_frames = f_end - f_start;
                if (chunk_frames < 10) continue;

                int abs_start = vr.start_ms + (int)(f_start * MS_PER_FRAME);
                int abs_end = vr.start_ms + (int)(f_end * MS_PER_FRAME);
                abs_end = std::min(abs_end, vr.end_ms);

                if (use_gpu_batch && d_mel &&
                    (batch_mel_offset + chunk_frames) * 80 <= batch_mel_capacity) {
                    cudaMemcpy(d_batch_mels + (size_t)batch_mel_offset * 80,
                               d_mel + f_start * 80,
                               (size_t)chunk_frames * 80 * sizeof(float),
                               cudaMemcpyDeviceToDevice);
                    if (use_cam_batch) {
                        cam_batch_chunks.push_back({d_batch_mels + (size_t)batch_mel_offset * 80, chunk_frames});
                    } else {
                        eres2_batch_chunks.push_back({d_batch_mels + (size_t)batch_mel_offset * 80, chunk_frames});
                    }
                    batch_chunk_infos.push_back({abs_start, abs_end});
                    batch_mel_offset += chunk_frames;
                } else if (deps_.eres2netv2_encoder && !deps_.eres2netv2_gpu_encoder) {
                    // ERes2NetV2 CPU path: needs CPU mel data
                    std::vector<float> embedding;
                    if (!mel.empty()) {
                        embedding = deps_.eres2netv2_encoder->extract(mel.data() + f_start * 80, chunk_frames);
                    } else if (d_mel) {
                        std::vector<float> cpu_mel(chunk_frames * 80);
                        cudaMemcpy(cpu_mel.data(), d_mel + f_start * 80,
                                   chunk_frames * 80 * sizeof(float), cudaMemcpyDeviceToHost);
                        embedding = deps_.eres2netv2_encoder->extract(cpu_mel.data(), chunk_frames);
                    }
                    ++embed_chunk_count;
                    if (embed_chunk_count % 200 == 0) {
                        fprintf(stderr, "[Pipeline] ERes2NetV2 CPU extracted %d chunks...\n", embed_chunk_count);
                    }
                    if (!embedding.empty()) {
                        bool valid = true;
                        for (float v : embedding)
                            if (std::isnan(v) || std::isinf(v)) { valid = false; break; }
                        if (valid) {
                            seg_embeddings.push_back(std::move(embedding));
                            chunk_infos.push_back({abs_start, abs_end});
                        }
                    }
                } else if (deps_.speaker_encoder) {
                    std::vector<float> embedding;
                    {
                        std::lock_guard<std::mutex> spk_lock(*deps_.speaker_mutex);
                        if (d_mel) {
                            embedding = deps_.speaker_encoder->extract_gpu(d_mel + f_start * 80, chunk_frames);
                        } else if (!mel.empty()) {
                            embedding = deps_.speaker_encoder->extract(mel.data() + f_start * 80, chunk_frames);
                        }
                    }
                    if (!embedding.empty()) {
                        bool valid = true;
                        for (float v : embedding)
                            if (std::isnan(v) || std::isinf(v)) { valid = false; break; }
                        if (valid) {
                            seg_embeddings.push_back(std::move(embedding));
                            chunk_infos.push_back({abs_start, abs_end});
                        }
                    }
                }
            }
        }

        // Pass 2: Batch extract (CAM++ or ERes2NetV2 GPU)
        if (use_cam_batch && !cam_batch_chunks.empty()) {
            std::lock_guard<std::mutex> spk_lock(*deps_.speaker_mutex);
            auto embeddings = deps_.speaker_encoder->extract_batch_gpu(cam_batch_chunks);
            for (int i = 0; i < (int)embeddings.size(); i++) {
                if (!embeddings[i].empty()) {
                    seg_embeddings.push_back(std::move(embeddings[i]));
                    chunk_infos.push_back(batch_chunk_infos[i]);
                }
            }
        }
        if (use_eres2_gpu_batch && !eres2_batch_chunks.empty()) {
            auto embeddings = deps_.eres2netv2_gpu_encoder->extract_batch_gpu(eres2_batch_chunks);
            for (int i = 0; i < (int)embeddings.size(); i++) {
                if (!embeddings[i].empty()) {
                    seg_embeddings.push_back(std::move(embeddings[i]));
                    chunk_infos.push_back(batch_chunk_infos[i]);
                }
            }
        }

        if (d_batch_mels) cudaFree(d_batch_mels);

        size_t total_batch = cam_batch_chunks.size() + eres2_batch_chunks.size();
        fprintf(stderr, "[Pipeline] Step 3b: %d VAD segs → %zu chunks (%.0fs speech) [batch=%zu]\n",
                mel_segments, seg_embeddings.size(), total_speech_sec, total_batch);

        // Debug: Export embeddings
        {
            const char* dbg = getenv("QWEN_EXPORT_EMBEDDINGS");
            if (dbg && seg_embeddings.size() > 0) {
                FILE* fp = fopen(dbg, "wb");
                if (fp) {
                    int n = (int)seg_embeddings.size();
                    int d = (int)seg_embeddings[0].size();
                    fwrite(&n, 4, 1, fp);
                    fwrite(&d, 4, 1, fp);
                    for (auto& e : seg_embeddings)
                        fwrite(e.data(), 4, d, fp);
                    for (auto& ci : chunk_infos) {
                        float cs = (float)ci.abs_start_ms / 1000.0f;
                        float ce2 = (float)ci.abs_end_ms / 1000.0f;
                        fwrite(&cs, 4, 1, fp);
                        fwrite(&ce2, 4, 1, fp);
                    }
                    fclose(fp);
                    fprintf(stderr, "[Pipeline] Exported %d embeddings to %s\n", n, dbg);
                }
            }
        }

        // ============================================================
        // Phase 3b: Spectral Clustering (chunk-level)
        // ============================================================
        if (seg_embeddings.size() >= 2) {
            const int n_segs = (int)seg_embeddings.size();
            const int emb_dim = (int)seg_embeddings[0].size();

            // 1. 余弦相似度矩阵
            std::vector<float> sim_matrix(n_segs * n_segs, 0.0f);
            for (int i = 0; i < n_segs; ++i) {
                sim_matrix[i * n_segs + i] = 1.0f;
                for (int j = i + 1; j < n_segs; ++j) {
                    float dot = 0;
                    for (int k = 0; k < emb_dim; ++k)
                        dot += seg_embeddings[i][k] * seg_embeddings[j][k];
                    sim_matrix[i * n_segs + j] = dot;
                    sim_matrix[j * n_segs + i] = dot;
                }
            }

            // 1b. Temporal proximity mixing
            {
                constexpr float TEMPORAL_ALPHA = 0.65f;
                constexpr float TEMPORAL_TAU = 12.0f;
                constexpr float INV_TAU = 1.0f / TEMPORAL_TAU;
                for (int i = 0; i < n_segs; ++i) {
                    float mid_i = (chunk_infos[i].abs_start_ms + chunk_infos[i].abs_end_ms) * 0.5e-3f;
                    for (int j = i + 1; j < n_segs; ++j) {
                        float mid_j = (chunk_infos[j].abs_start_ms + chunk_infos[j].abs_end_ms) * 0.5e-3f;
                        float t_prox = expf(-fabsf(mid_i - mid_j) * INV_TAU);
                        float cos_val = sim_matrix[i * n_segs + j];
                        float combined = (1.0f - TEMPORAL_ALPHA) * cos_val + TEMPORAL_ALPHA * t_prox;
                        sim_matrix[i * n_segs + j] = combined;
                        sim_matrix[j * n_segs + i] = combined;
                    }
                }
            }

            // 2. p-pruning
            int p = std::max(3, n_segs * 6 / 100);
            p = std::min(p, n_segs - 1);
            for (int i = 0; i < n_segs; ++i) {
                std::vector<float> row_vals;
                for (int j = 0; j < n_segs; ++j) {
                    if (j != i) row_vals.push_back(sim_matrix[i * n_segs + j]);
                }
                std::sort(row_vals.rbegin(), row_vals.rend());
                float threshold = (p < (int)row_vals.size()) ? row_vals[p] : -2.0f;
                for (int j = 0; j < n_segs; ++j) {
                    if (j != i && sim_matrix[i * n_segs + j] < threshold)
                        sim_matrix[i * n_segs + j] = 0;
                }
            }

            // 3. 对称化
            for (int i = 0; i < n_segs; ++i) {
                for (int j = i + 1; j < n_segs; ++j) {
                    float val = (sim_matrix[i * n_segs + j] + sim_matrix[j * n_segs + i]) * 0.5f;
                    val = std::max(0.0f, val);
                    sim_matrix[i * n_segs + j] = val;
                    sim_matrix[j * n_segs + i] = val;
                }
            }

            // 孤立节点修复
            for (int i = 0; i < n_segs; ++i) {
                float row_sum = 0;
                for (int j = 0; j < n_segs; ++j)
                    if (j != i) row_sum += sim_matrix[i * n_segs + j];
                if (row_sum < 1e-12f) {
                    float best = -2; int best_j = 0;
                    for (int j = 0; j < n_segs; ++j) {
                        if (j == i) continue;
                        float dot = 0;
                        for (int k = 0; k < emb_dim; ++k)
                            dot += seg_embeddings[i][k] * seg_embeddings[j][k];
                        if (dot > best) { best = dot; best_j = j; }
                    }
                    sim_matrix[i * n_segs + best_j] = std::max(0.01f, best);
                    sim_matrix[best_j * n_segs + i] = std::max(0.01f, best);
                }
            }

            // 4. Normalized Laplacian eigendecomposition
            std::vector<float> D(n_segs, 0.0f);
            for (int i = 0; i < n_segs; ++i)
                for (int j = 0; j < n_segs; ++j)
                    D[i] += sim_matrix[i * n_segs + j];

            std::vector<float> D_inv_sqrt(n_segs);
            for (int i = 0; i < n_segs; ++i)
                D_inv_sqrt[i] = (D[i] > 1e-12f) ? 1.0f / sqrtf(D[i]) : 0.0f;

            std::vector<float> Lsym(n_segs * n_segs, 0.0f);
            for (int i = 0; i < n_segs; ++i)
                for (int j = 0; j < n_segs; ++j)
                    Lsym[i * n_segs + j] = D_inv_sqrt[i] * sim_matrix[i * n_segs + j] * D_inv_sqrt[j];

            // 5. Power iteration top-k eigenvectors
            const int max_k = 8;
            int actual_max = std::min(max_k, n_segs);
            std::vector<std::vector<float>> eigenvectors(actual_max, std::vector<float>(n_segs, 0));
            std::vector<float> eigenvalues(actual_max, 0);

            std::vector<float> Lwork = Lsym;
            for (int k = 0; k < actual_max; ++k) {
                std::vector<float> v(n_segs);
                for (int i = 0; i < n_segs; ++i) v[i] = (float)(i + k * 7 + 1);
                float vnorm = 0;
                for (float x : v) vnorm += x * x;
                vnorm = sqrtf(vnorm);
                for (float& x : v) x /= vnorm;

                for (int iter = 0; iter < 200; ++iter) {
                    std::vector<float> Av(n_segs, 0);
                    for (int i = 0; i < n_segs; ++i)
                        for (int j = 0; j < n_segs; ++j)
                            Av[i] += Lwork[i * n_segs + j] * v[j];
                    float norm = 0;
                    for (float x : Av) norm += x * x;
                    norm = sqrtf(norm + 1e-12f);
                    for (int i = 0; i < n_segs; ++i) v[i] = Av[i] / norm;
                }

                float lambda = 0;
                for (int i = 0; i < n_segs; ++i) {
                    float Av_i = 0;
                    for (int j = 0; j < n_segs; ++j)
                        Av_i += Lwork[i * n_segs + j] * v[j];
                    lambda += v[i] * Av_i;
                }
                eigenvalues[k] = lambda;
                eigenvectors[k] = v;

                for (int i = 0; i < n_segs; ++i)
                    for (int j = 0; j < n_segs; ++j)
                        Lwork[i * n_segs + j] -= lambda * v[i] * v[j];
            }

            // 6. NME eigengap
            int optimal_k = 2;
            float max_nme = 0;
            for (int k = 0; k + 1 < actual_max; ++k) {
                float gap = eigenvalues[k] - eigenvalues[k + 1];
                if (eigenvalues[k] < 0.01f) continue;
                float nme = gap / (k + 1);
                if (nme > max_nme) {
                    max_nme = nme;
                    optimal_k = k + 1;
                }
            }
            optimal_k = std::max(2, std::min(optimal_k, 8));

            // Duration heuristic
            int min_k_heuristic = 2;
            if (total_speech_sec > 1200 && n_segs > 100) min_k_heuristic = 4;
            else if (total_speech_sec > 300 && n_segs > 50) min_k_heuristic = 3;
            if (optimal_k < min_k_heuristic) {
                fprintf(stderr, "[Pipeline] Phase 3b: duration heuristic bumps k %d→%d (%.0fs, %d chunks)\n",
                        optimal_k, min_k_heuristic, total_speech_sec, n_segs);
                optimal_k = min_k_heuristic;
            }

            fprintf(stderr, "[Pipeline] Phase 3b spectral: eigenvalues:");
            for (int k = 0; k < actual_max; ++k)
                fprintf(stderr, " %.3f", eigenvalues[k]);
            fprintf(stderr, " → k=%d (nme=%.4f)\n", optimal_k, max_nme);

            // 7. Spectral features + K-means
            std::vector<float> features(n_segs * optimal_k);
            for (int i = 0; i < n_segs; ++i) {
                float row_norm = 0;
                for (int k = 0; k < optimal_k; ++k) {
                    features[i * optimal_k + k] = eigenvectors[k][i];
                    row_norm += eigenvectors[k][i] * eigenvectors[k][i];
                }
                row_norm = sqrtf(row_norm + 1e-12f);
                for (int k = 0; k < optimal_k; ++k)
                    features[i * optimal_k + k] /= row_norm;
            }

            std::vector<int> labels(n_segs, 0);
            float best_inertia = 1e30f;

            for (int restart = 0; restart < 10; ++restart) {
                std::vector<std::vector<float>> cur_centroids(optimal_k, std::vector<float>(optimal_k, 0));
                std::vector<int> cur_labels(n_segs, 0);

                // k-means++ init
                int seed_idx = restart * 137 % n_segs;
                for (int j = 0; j < optimal_k; ++j)
                    cur_centroids[0][j] = features[seed_idx * optimal_k + j];
                for (int c = 1; c < optimal_k; ++c) {
                    float best_d = -1;
                    int best_i = 0;
                    for (int i = 0; i < n_segs; ++i) {
                        float min_d = 1e30f;
                        for (int prev = 0; prev < c; ++prev) {
                            float d = 0;
                            for (int j = 0; j < optimal_k; ++j) {
                                float diff = features[i * optimal_k + j] - cur_centroids[prev][j];
                                d += diff * diff;
                            }
                            min_d = std::min(min_d, d);
                        }
                        if (min_d > best_d) {
                            best_d = min_d;
                            best_i = i;
                        }
                    }
                    for (int j = 0; j < optimal_k; ++j)
                        cur_centroids[c][j] = features[best_i * optimal_k + j];
                }

                for (int iter = 0; iter < 30; ++iter) {
                    int changed = 0;
                    for (int i = 0; i < n_segs; ++i) {
                        float best_d = 1e30f;
                        int best_c = 0;
                        for (int c = 0; c < optimal_k; ++c) {
                            float d = 0;
                            for (int j = 0; j < optimal_k; ++j) {
                                float diff = features[i * optimal_k + j] - cur_centroids[c][j];
                                d += diff * diff;
                            }
                            if (d < best_d) { best_d = d; best_c = c; }
                        }
                        if (best_c != cur_labels[i]) ++changed;
                        cur_labels[i] = best_c;
                    }

                    for (int c = 0; c < optimal_k; ++c)
                        std::fill(cur_centroids[c].begin(), cur_centroids[c].end(), 0.0f);
                    std::vector<int> cnt(optimal_k, 0);
                    for (int i = 0; i < n_segs; ++i) {
                        cnt[cur_labels[i]]++;
                        for (int j = 0; j < optimal_k; ++j)
                            cur_centroids[cur_labels[i]][j] += features[i * optimal_k + j];
                    }
                    for (int c = 0; c < optimal_k; ++c)
                        if (cnt[c] > 0)
                            for (int j = 0; j < optimal_k; ++j)
                                cur_centroids[c][j] /= cnt[c];

                    if (changed == 0) break;
                }

                float inertia = 0;
                for (int i = 0; i < n_segs; ++i) {
                    for (int j = 0; j < optimal_k; ++j) {
                        float diff = features[i * optimal_k + j] - cur_centroids[cur_labels[i]][j];
                        inertia += diff * diff;
                    }
                }
                if (inertia < best_inertia) {
                    best_inertia = inertia;
                    labels = cur_labels;
                }
            }

            // 8. Cluster embedding centroids + log
            std::vector<std::vector<float>> cluster_emb(optimal_k, std::vector<float>(emb_dim, 0));
            std::vector<int> cluster_cnt(optimal_k, 0);
            for (int i = 0; i < n_segs; ++i) {
                cluster_cnt[labels[i]]++;
                for (int j = 0; j < emb_dim; ++j)
                    cluster_emb[labels[i]][j] += seg_embeddings[i][j];
            }
            for (int c = 0; c < optimal_k; ++c) {
                if (cluster_cnt[c] > 0) {
                    for (int j = 0; j < emb_dim; ++j)
                        cluster_emb[c][j] /= cluster_cnt[c];
                    float norm = 0;
                    for (float v : cluster_emb[c]) norm += v * v;
                    norm = sqrtf(norm + 1e-12f);
                    for (float& v : cluster_emb[c]) v /= norm;
                }
            }

            fprintf(stderr, "[Pipeline] Phase 3b: cluster centroid similarities:\n");
            for (int i = 0; i < optimal_k; ++i) {
                for (int j = i + 1; j < optimal_k; ++j) {
                    float sim = 0;
                    for (int k = 0; k < emb_dim; ++k)
                        sim += cluster_emb[i][k] * cluster_emb[j][k];
                    fprintf(stderr, "  c%d-c%d: %.3f", i, j, sim);
                }
            }
            fprintf(stderr, "\n");

            // Dump embeddings
            {
                FILE* fdump = fopen("tmp/speaker_dump.bin", "wb");
                if (fdump) {
                    int32_t hdr[3] = {n_segs, emb_dim, optimal_k};
                    fwrite(hdr, sizeof(int32_t), 3, fdump);
                    for (int i = 0; i < n_segs; ++i) {
                        float ts[2] = {(float)chunk_infos[i].abs_start_ms, (float)chunk_infos[i].abs_end_ms};
                        fwrite(ts, sizeof(float), 2, fdump);
                    }
                    for (int i = 0; i < n_segs; ++i)
                        fwrite(seg_embeddings[i].data(), sizeof(float), emb_dim, fdump);
                    fwrite(labels.data(), sizeof(int), n_segs, fdump);
                    fclose(fdump);
                }
            }

            // 8b. Temporal consistency smoothing
            {
                std::vector<int> order(n_segs);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) { return chunk_infos[a].abs_start_ms < chunk_infos[b].abs_start_ms; });

                const int WINDOW = 2;
                const float COS_MARGIN = -0.04f;
                const int MAX_SMOOTH_ITER = 2;
                int total_smoothed = 0;

                for (int iter = 0; iter < MAX_SMOOTH_ITER; ++iter) {
                    int changed = 0;
                    for (int oi = 0; oi < n_segs; ++oi) {
                        int idx = order[oi];
                        int lo = std::max(0, oi - WINDOW);
                        int hi = std::min(n_segs - 1, oi + WINDOW);

                        int neighbor_cnt[64] = {};
                        for (int ni = lo; ni <= hi; ++ni) {
                            if (ni == oi) continue;
                            neighbor_cnt[labels[order[ni]]]++;
                        }

                        int maj_label = -1, maj_count = 0;
                        for (int c = 0; c < optimal_k; ++c) {
                            if (neighbor_cnt[c] > maj_count) {
                                maj_count = neighbor_cnt[c];
                                maj_label = c;
                            }
                        }

                        if (maj_label >= 0 && maj_label != labels[idx] && maj_count >= WINDOW) {
                            float old_sim = 0, new_sim = 0;
                            for (int dd = 0; dd < emb_dim; ++dd) {
                                old_sim += seg_embeddings[idx][dd] * cluster_emb[labels[idx]][dd];
                                new_sim += seg_embeddings[idx][dd] * cluster_emb[maj_label][dd];
                            }
                            if (new_sim >= old_sim + COS_MARGIN) {
                                labels[idx] = maj_label;
                                ++changed;
                            }
                        }
                    }
                    total_smoothed += changed;
                    if (changed == 0) break;
                }

                if (total_smoothed > 0) {
                    fprintf(stderr, "[Pipeline] Phase 3b: temporal smoothing: %d chunks corrected\n",
                            total_smoothed);
                    // Update cluster_emb after smoothing
                    for (int c = 0; c < optimal_k; ++c) {
                        std::fill(cluster_emb[c].begin(), cluster_emb[c].end(), 0.0f);
                        cluster_cnt[c] = 0;
                    }
                    for (int i = 0; i < n_segs; ++i) {
                        cluster_cnt[labels[i]]++;
                        for (int dd = 0; dd < emb_dim; ++dd)
                            cluster_emb[labels[i]][dd] += seg_embeddings[i][dd];
                    }
                    for (int c = 0; c < optimal_k; ++c) {
                        if (cluster_cnt[c] > 0) {
                            for (int dd = 0; dd < emb_dim; ++dd)
                                cluster_emb[c][dd] /= cluster_cnt[c];
                            float norm = 0;
                            for (float v : cluster_emb[c]) norm += v * v;
                            norm = sqrtf(norm + 1e-12f);
                            for (float& v : cluster_emb[c]) v /= norm;
                        }
                    }
                }
            }

            // Build spk_intervals from chunk labels
            spk_intervals.clear();
            for (int i = 0; i < n_segs; ++i) {
                SpkInterval si;
                si.start_ms = chunk_infos[i].abs_start_ms;
                si.end_ms = chunk_infos[i].abs_end_ms;
                si.speaker_id = labels[i];
                si.speaker_name = "Speaker_" + std::to_string(labels[i]);
                spk_intervals.push_back(si);
            }

            std::sort(spk_intervals.begin(), spk_intervals.end(),
                [](const SpkInterval& a, const SpkInterval& b) { return a.start_ms < b.start_ms; });

            // Merge adjacent same-speaker intervals (gap ≤ 500ms)
            {
                std::vector<SpkInterval> merged;
                for (auto& si : spk_intervals) {
                    if (!merged.empty() && merged.back().speaker_id == si.speaker_id &&
                        si.start_ms - merged.back().end_ms <= 500) {
                        merged.back().end_ms = std::max(merged.back().end_ms, si.end_ms);
                    } else {
                        merged.push_back(si);
                    }
                }
                spk_intervals = std::move(merged);
            }

            // Renumber speaker_id by appearance order
            std::map<int, int> renumber;
            int next_num = 0;
            for (auto& si : spk_intervals) {
                if (renumber.find(si.speaker_id) == renumber.end())
                    renumber[si.speaker_id] = next_num++;
                si.speaker_id = renumber[si.speaker_id];
                si.speaker_name = "Speaker_" + std::to_string(si.speaker_id);
            }
            phase3_speaker_count = next_num;

            // Auto-register diarization speakers to global manager
            if (!seg_embeddings.empty() && deps_.speaker_manager) {
                std::map<int, std::vector<float>> cluster_centroids;
                std::map<int, int> cluster_counts;
                int edim_reg = (int)seg_embeddings[0].size();
                for (size_t ci = 0; ci < seg_embeddings.size(); ++ci) {
                    int final_id = renumber.count(labels[ci]) ? renumber[labels[ci]] : labels[ci];
                    if (cluster_centroids.find(final_id) == cluster_centroids.end()) {
                        cluster_centroids[final_id].assign(edim_reg, 0.0f);
                        cluster_counts[final_id] = 0;
                    }
                    cluster_counts[final_id]++;
                    for (int j = 0; j < edim_reg; ++j)
                        cluster_centroids[final_id][j] += seg_embeddings[ci][j];
                }
                for (auto& [id, c] : cluster_centroids) {
                    float cnt_f = (float)cluster_counts[id];
                    for (float& v : c) v /= cnt_f;
                    float norm = 0;
                    for (float v : c) norm += v * v;
                    norm = sqrtf(norm + 1e-12f);
                    for (float& v : c) v /= norm;
                    std::string name = "Speaker_" + std::to_string(id);
                    bool exists = false;
                    for (auto& gn : deps_.speaker_manager->speaker_names()) {
                        if (gn == name) { exists = true; break; }
                    }
                    if (!exists) {
                        deps_.speaker_manager->register_speaker(name, c);
                    }
                }
            }
        }

        double mel_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - mel_t0).count();
        double phase3_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - phase3_t0).count();

        fprintf(stderr, "[Pipeline] v4 Phase 3: %zu speaker intervals, %d speakers (%.1fs)\n",
                spk_intervals.size(), phase3_speaker_count, phase3_ms / 1000.0);
    }

    // Wait for Phase 2
    aligned_words = phase2_future.get();
    fprintf(stderr, "[Pipeline] v4 Phase 2+3: %zu words, %zu intervals (%.1fs)\n",
            aligned_words.size(), spk_intervals.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - phase2_t0).count());

    // ================================================================
    // Phase 4: Word → Speaker attribution
    // ================================================================
    struct WordWithSpeaker {
        std::string word;
        int start_ms, end_ms;
        int speaker_id;
        std::string speaker_name;
    };
    std::vector<WordWithSpeaker> word_list;
    word_list.reserve(aligned_words.size());

    // 4a-pre: 零时长聚集重分布
    {
        size_t idx = 0;
        while (idx < aligned_words.size()) {
            if (aligned_words[idx].start_ms >= 0 &&
                aligned_words[idx].start_ms == aligned_words[idx].end_ms) {
                int cluster_ts = aligned_words[idx].start_ms;
                size_t zero_start = idx;
                while (idx < aligned_words.size() &&
                       aligned_words[idx].start_ms >= 0 &&
                       aligned_words[idx].start_ms == aligned_words[idx].end_ms &&
                       std::abs(aligned_words[idx].start_ms - cluster_ts) <= 200) {
                    ++idx;
                }
                size_t cluster_size = idx - zero_start;
                if (cluster_size >= 5) {
                    size_t ext_start = zero_start;
                    while (ext_start > 0) {
                        if (aligned_words[ext_start - 1].start_ms < 0) break;
                        int gap_before = 0;
                        if (ext_start >= 2)
                            gap_before = aligned_words[ext_start - 1].start_ms -
                                         aligned_words[ext_start - 2].end_ms;
                        if (gap_before > 5000) { --ext_start; } else break;
                    }
                    size_t total = idx - ext_start;
                    int bound_left = std::max(0, cluster_ts - (int)total * 200);
                    for (int j = (int)ext_start - 1; j >= 0; --j) {
                        if (aligned_words[j].start_ms >= 0 &&
                            aligned_words[j].end_ms > aligned_words[j].start_ms) {
                            bound_left = aligned_words[j].end_ms;
                            break;
                        }
                    }
                    int bound_right = cluster_ts;
                    int range = bound_right - bound_left;
                    if (range < (int)total * 60)
                        bound_right = bound_left + (int)total * 150;
                    int step = std::max(60, (bound_right - bound_left) / (int)total);
                    for (size_t j = ext_start; j < idx; ++j) {
                        aligned_words[j].start_ms = bound_left + (int)(j - ext_start) * step;
                        aligned_words[j].end_ms = aligned_words[j].start_ms + std::min(step, 100);
                    }
                }
            } else { ++idx; }
        }
    }

    // 4a: 零时长词时间戳平滑
    for (size_t i = 0; i < aligned_words.size(); ++i) {
        auto& aw = aligned_words[i];
        if (aw.start_ms >= 0 && aw.start_ms == aw.end_ms) {
            int left = (i > 0 && aligned_words[i-1].end_ms > 0) ? aligned_words[i-1].end_ms : aw.start_ms;
            int right = (i + 1 < aligned_words.size() && aligned_words[i+1].start_ms > 0)
                        ? aligned_words[i+1].start_ms : aw.start_ms;
            if (right > aw.start_ms) {
                aw.end_ms = std::min(right, aw.start_ms + 80);
            } else if (aw.start_ms > left) {
                aw.start_ms = std::max(left, aw.end_ms - 80);
            }
        }
    }

    // 4b: 主分配
    for (auto& aw : aligned_words) {
        WordWithSpeaker wws;
        wws.word = aw.word;
        wws.start_ms = aw.start_ms;
        wws.end_ms = aw.end_ms;
        wws.speaker_id = -1;
        wws.speaker_name = "Unknown";
        if (aw.start_ms >= 0 && !spk_intervals.empty()) {
            int word_mid = (aw.start_ms + aw.end_ms) / 2;
            int best_overlap = 0;
            for (auto& si : spk_intervals) {
                int os = std::max(aw.start_ms, si.start_ms);
                int oe = std::min(aw.end_ms, si.end_ms);
                int overlap = oe - os;
                if (overlap > best_overlap) {
                    best_overlap = overlap;
                    wws.speaker_id = si.speaker_id;
                    wws.speaker_name = si.speaker_name;
                }
            }
            if (wws.speaker_id < 0) {
                for (auto& si : spk_intervals) {
                    if (word_mid >= si.start_ms && word_mid < si.end_ms) {
                        wws.speaker_id = si.speaker_id;
                        wws.speaker_name = si.speaker_name;
                        break;
                    }
                }
            }
            if (wws.speaker_id < 0) {
                int min_dist = INT_MAX;
                for (auto& si : spk_intervals) {
                    int dist = 0;
                    if (word_mid < si.start_ms) dist = si.start_ms - word_mid;
                    else if (word_mid > si.end_ms) dist = word_mid - si.end_ms;
                    if (dist < min_dist) {
                        min_dist = dist;
                        wws.speaker_id = si.speaker_id;
                        wws.speaker_name = si.speaker_name;
                    }
                }
            }
        }
        word_list.push_back(wws);
    }

    // ================================================================
    // Phase 5: Build speaker segments
    // ================================================================
    struct V4Segment {
        int start_ms, end_ms;
        int speaker_id;
        std::string speaker_name;
        std::string text;
    };
    std::vector<V4Segment> v4_segments;

    // 5a: 连续同 speaker 合并 (gap ≤ 2s)
    for (auto& w : word_list) {
        bool extend = !v4_segments.empty() &&
                      w.speaker_id == v4_segments.back().speaker_id &&
                      w.speaker_id >= 0 &&
                      w.start_ms - v4_segments.back().end_ms <= 1200;
        if (extend) {
            v4_segments.back().end_ms = std::max(v4_segments.back().end_ms, w.end_ms);
            v4_segments.back().text += w.word;
        } else {
            V4Segment seg;
            seg.start_ms = w.start_ms;
            seg.end_ms = w.end_ms;
            seg.speaker_id = w.speaker_id;
            seg.speaker_name = w.speaker_name;
            seg.text = w.word;
            v4_segments.push_back(std::move(seg));
        }
    }

    // 5b: 短段吸收
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<V4Segment> merged;
        for (size_t i = 0; i < v4_segments.size(); ++i) {
            auto& seg = v4_segments[i];
            int char_count = 0;
            for (size_t j = 0; j < seg.text.size(); ) {
                unsigned char c = seg.text[j];
                if (c < 0x80) { ++j; } else if (c < 0xE0) { j += 2; } else if (c < 0xF0) { j += 3; } else { j += 4; }
                ++char_count;
            }
            if (char_count <= 3 && (seg.end_ms - seg.start_ms) < 2000) {
                if (!merged.empty() &&
                    seg.start_ms - merged.back().end_ms <= 2000 &&
                    (seg.speaker_id == merged.back().speaker_id || seg.speaker_id < 0)) {
                    merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                    merged.back().text += seg.text;
                    continue;
                }
                if (i + 1 < v4_segments.size() &&
                    v4_segments[i+1].start_ms - seg.end_ms <= 2000 &&
                    (seg.speaker_id == v4_segments[i+1].speaker_id || seg.speaker_id < 0)) {
                    v4_segments[i+1].start_ms = std::min(v4_segments[i+1].start_ms, seg.start_ms);
                    v4_segments[i+1].text = seg.text + v4_segments[i+1].text;
                    continue;
                }
            }
            merged.push_back(seg);
        }
        v4_segments = std::move(merged);
    }

    // 5c: 二次合并
    {
        std::vector<V4Segment> merged;
        for (auto& seg : v4_segments) {
            if (!merged.empty() &&
                seg.speaker_id == merged.back().speaker_id &&
                seg.start_ms - merged.back().end_ms <= 2000) {
                merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                merged.back().text += seg.text;
            } else {
                merged.push_back(seg);
            }
        }
        v4_segments = std::move(merged);
    }

    // 5d: 长段落拆分 (>15s 的段落在词间隙>800ms处拆分)
    {
        bool did_split = true;
        while (did_split) {
            did_split = false;
            std::vector<V4Segment> split_result;
            for (auto& seg : v4_segments) {
                int seg_dur = seg.end_ms - seg.start_ms;
                if (seg_dur <= 15000) {
                    split_result.push_back(seg);
                    continue;
                }
                // 找这个段落时间范围内的所有词 (不限 speaker_id)
                std::vector<int> seg_word_indices;
                for (int wi = 0; wi < (int)word_list.size(); ++wi) {
                    if (word_list[wi].start_ms >= seg.start_ms - 100 &&
                        word_list[wi].end_ms <= seg.end_ms + 100) {
                        seg_word_indices.push_back(wi);
                    }
                }
                if (seg_word_indices.size() < 4) {
                    split_result.push_back(seg);
                    continue;
                }
                // 找最大词间隙(>800ms)
                int best_gap = 0, best_gap_idx = -1;
                for (int i = 1; i < (int)seg_word_indices.size(); ++i) {
                    int gap = word_list[seg_word_indices[i]].start_ms -
                              word_list[seg_word_indices[i-1]].end_ms;
                    if (gap > best_gap && gap >= 800) {
                        best_gap = gap;
                        best_gap_idx = i;
                    }
                }
                if (best_gap_idx < 0) {
                    split_result.push_back(seg);
                    continue;
                }
                // 拆分
                V4Segment seg1, seg2;
                seg1.start_ms = seg.start_ms;
                seg1.end_ms = word_list[seg_word_indices[best_gap_idx - 1]].end_ms;
                seg1.speaker_id = seg.speaker_id;
                seg1.speaker_name = seg.speaker_name;
                seg2.start_ms = word_list[seg_word_indices[best_gap_idx]].start_ms;
                seg2.end_ms = seg.end_ms;
                seg2.speaker_id = seg.speaker_id;
                seg2.speaker_name = seg.speaker_name;
                // 分配文字
                int split_word = seg_word_indices[best_gap_idx];
                seg1.text.clear();
                seg2.text.clear();
                for (int wi : seg_word_indices) {
                    if (wi < split_word) seg1.text += word_list[wi].word;
                    else seg2.text += word_list[wi].word;
                }
                if (!seg1.text.empty()) split_result.push_back(seg1);
                if (!seg2.text.empty()) split_result.push_back(seg2);
                did_split = true;
            }
            v4_segments = std::move(split_result);
        }
    }

    // ================================================================
    // Phase 6: 标点恢复 + speaker/sentence 重分段
    // ================================================================
    if (deps_.punctuation_restorer) {
        // 6a: 拼接全文, 建立 char→word 映射
        std::string full_text_concat;
        std::vector<int> word_char_map;
        for (size_t wi = 0; wi < word_list.size(); ++wi) {
            const auto& w = word_list[wi];
            for (size_t j = 0; j < w.word.size(); ) {
                unsigned char c = (unsigned char)w.word[j];
                int clen = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                full_text_concat += w.word.substr(j, clen);
                word_char_map.push_back((int)wi);
                j += clen;
            }
        }

        // 6b: 全文标点恢复
        std::string punctuated = deps_.punctuation_restorer->restore(full_text_concat);

        // 6c: 对齐
        struct CharInfo {
            std::string ch;
            int word_idx;
            bool is_punc;
        };
        std::vector<CharInfo> char_infos;
        {
            auto split_u8 = [](const std::string& s) {
                std::vector<std::string> out;
                for (size_t i = 0; i < s.size(); ) {
                    unsigned char c = (unsigned char)s[i];
                    int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                    out.push_back(s.substr(i, len));
                    i += len;
                }
                return out;
            };
            auto orig_chars = split_u8(full_text_concat);
            auto punc_chars = split_u8(punctuated);

            size_t oi = 0;
            for (size_t pi = 0; pi < punc_chars.size(); ++pi) {
                if (oi < orig_chars.size() && punc_chars[pi] == orig_chars[oi]) {
                    char_infos.push_back({punc_chars[pi], word_char_map[oi], false});
                    ++oi;
                } else {
                    int inherit_wi = (char_infos.empty()) ? 0 :
                        (char_infos.back().word_idx >= 0) ? char_infos.back().word_idx : 0;
                    char_infos.push_back({punc_chars[pi], inherit_wi, true});
                }
            }
        }

        // 6d: 重建 segments
        auto is_sentence_end = [](const std::string& c) {
            return c == "\xe3\x80\x82" || c == "\xef\xbc\x9f" || c == "\xef\xbc\x81";
        };

        v4_segments.clear();
        if (!char_infos.empty()) {
            int first_wi = 0;
            for (auto& ci : char_infos)
                if (ci.word_idx >= 0) { first_wi = ci.word_idx; break; }

            V4Segment cur;
            cur.speaker_id = word_list[first_wi].speaker_id;
            cur.speaker_name = word_list[first_wi].speaker_name;
            cur.start_ms = word_list[first_wi].start_ms;
            cur.end_ms = word_list[first_wi].end_ms;

            for (size_t i = 0; i < char_infos.size(); ++i) {
                auto& ci = char_infos[i];
                int wi = ci.word_idx >= 0 ? ci.word_idx : -1;

                bool split_here = false;
                bool is_sent_end = is_sentence_end(ci.ch) && i + 1 < char_infos.size();
                bool is_comma = (ci.ch == "\xef\xbc\x8c") && i + 1 < char_infos.size();

                int cur_char_count = 0;
                {
                    for (size_t ci2 = 0; ci2 < cur.text.size(); ) {
                        unsigned char uc = (unsigned char)cur.text[ci2];
                        int cl = (uc < 0x80) ? 1 : (uc < 0xE0) ? 2 : (uc < 0xF0) ? 3 : 4;
                        ci2 += cl;
                        cur_char_count++;
                    }
                }

                if (is_sent_end || is_comma) {
                    int next_wi = -1;
                    for (size_t j = i + 1; j < char_infos.size(); ++j) {
                        if (char_infos[j].word_idx >= 0) {
                            next_wi = char_infos[j].word_idx;
                            break;
                        }
                    }
                    if (is_sent_end && next_wi >= 0 && word_list[next_wi].speaker_id != cur.speaker_id)
                        split_here = true;
                    if (!split_here && is_sent_end &&
                        ((cur.end_ms - cur.start_ms) > 10000 || cur_char_count > 30))
                        split_here = true;
                    if (!split_here && is_comma &&
                        ((cur.end_ms - cur.start_ms) > 15000 || cur_char_count > 40))
                        split_here = true;
                }

                if (!split_here && !ci.is_punc && wi >= 0 &&
                    word_list[wi].speaker_id != cur.speaker_id) {
                    if (!cur.text.empty()) v4_segments.push_back(cur);
                    cur = V4Segment();
                    cur.speaker_id = word_list[wi].speaker_id;
                    cur.speaker_name = word_list[wi].speaker_name;
                    cur.start_ms = word_list[wi].start_ms;
                    cur.end_ms = word_list[wi].end_ms;
                    cur.text = ci.ch;
                    continue;
                }

                // 时间间隙强制拆分: 当前词与段落末尾差 >5s 时强制新段
                if (!split_here && !ci.is_punc && wi >= 0 &&
                    !cur.text.empty() && cur.end_ms > 0 &&
                    word_list[wi].start_ms - cur.end_ms > 5000) {
                    v4_segments.push_back(cur);
                    cur = V4Segment();
                    cur.speaker_id = word_list[wi].speaker_id;
                    cur.speaker_name = word_list[wi].speaker_name;
                    cur.start_ms = word_list[wi].start_ms;
                    cur.end_ms = word_list[wi].end_ms;
                    cur.text = ci.ch;
                    continue;
                }

                cur.text += ci.ch;
                if (wi >= 0)
                    cur.end_ms = std::max(cur.end_ms, word_list[wi].end_ms);

                if (split_here) {
                    if (!cur.text.empty()) v4_segments.push_back(cur);
                    int next_wi = -1;
                    for (size_t j = i + 1; j < char_infos.size(); ++j) {
                        if (char_infos[j].word_idx >= 0) {
                            next_wi = char_infos[j].word_idx;
                            break;
                        }
                    }
                    cur = V4Segment();
                    if (next_wi >= 0) {
                        cur.speaker_id = word_list[next_wi].speaker_id;
                        cur.speaker_name = word_list[next_wi].speaker_name;
                        cur.start_ms = word_list[next_wi].start_ms;
                        cur.end_ms = word_list[next_wi].end_ms;
                    }
                }
            }
            if (!cur.text.empty()) v4_segments.push_back(cur);
        }

        // 6e: 段末标点修正
        for (auto& seg : v4_segments) {
            if (seg.text.empty()) continue;
            bool ends_sent = false;
            if (seg.text.size() >= 3) {
                std::string last3 = seg.text.substr(seg.text.size() - 3);
                ends_sent = (last3 == "\xe3\x80\x82" || last3 == "\xef\xbc\x9f" || last3 == "\xef\xbc\x81");
            }
            if (!ends_sent) {
                if (seg.text.size() >= 3 && seg.text.substr(seg.text.size() - 3) == "\xef\xbc\x8c") {
                    seg.text.replace(seg.text.size() - 3, 3, "\xe3\x80\x82");
                } else {
                    seg.text += "\xe3\x80\x82";
                }
            }
        }
    }

    // ================================================================
    // Phase 6.5: 说话人 island 平滑
    // ================================================================
    {
        std::vector<V4Segment> smoothed;
        int island_merged = 0;

        for (size_t i = 0; i < v4_segments.size(); ++i) {
            auto& seg = v4_segments[i];
            int dur_ms = seg.end_ms - seg.start_ms;
            bool is_island = dur_ms < 1000 && seg.speaker_id >= 0 &&
                             !smoothed.empty() && i + 1 < v4_segments.size();
            bool surrounded = is_island &&
                              smoothed.back().speaker_id >= 0 &&
                              smoothed.back().speaker_id == v4_segments[i+1].speaker_id &&
                              smoothed.back().speaker_id != seg.speaker_id;
            if (surrounded) {
                smoothed.back().end_ms = std::max(smoothed.back().end_ms, seg.end_ms);
                smoothed.back().text += seg.text;
                ++island_merged;
            } else {
                smoothed.push_back(seg);
            }
        }

        if (island_merged > 0)
            fprintf(stderr, "[Pipeline] v4 Phase 6.5: smoothed %d speaker islands\n", island_merged);

        std::vector<V4Segment> merged;
        for (auto& seg : smoothed) {
            if (!merged.empty() &&
                seg.speaker_id == merged.back().speaker_id &&
                seg.start_ms - merged.back().end_ms <= 2000) {
                int prev_dur = merged.back().end_ms - merged.back().start_ms;
                int cur_dur = seg.end_ms - seg.start_ms;
                int merged_dur = std::max(merged.back().end_ms, seg.end_ms) - merged.back().start_ms;
                if ((prev_dur <= 5000 || cur_dur <= 5000) && merged_dur <= 20000) {
                    merged.back().end_ms = std::max(merged.back().end_ms, seg.end_ms);
                    merged.back().text += seg.text;
                } else {
                    merged.push_back(seg);
                }
            } else {
                merged.push_back(seg);
            }
        }
        v4_segments = std::move(merged);
    }

    // ================================================================
    // Phase 6.52: 超长段拆分 (>30s 且有句末标点时按标点拆)
    // ================================================================
    {
        auto split_u8_chars = [](const std::string& s) -> std::vector<std::string> {
            std::vector<std::string> out;
            for (size_t i = 0; i < s.size(); ) {
                unsigned char c = (unsigned char)s[i];
                int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
                out.push_back(s.substr(i, len));
                i += len;
            }
            return out;
        };
        auto is_sent_end_ch = [](const std::string& c) {
            return c == "\xe3\x80\x82" || c == "\xef\xbc\x9f" || c == "\xef\xbc\x81";
        };

        std::vector<V4Segment> split_out;
        int split_count = 0;
        for (auto& seg : v4_segments) {
            int dur_ms = seg.end_ms - seg.start_ms;
            if (dur_ms <= 30000) {
                split_out.push_back(seg);
                continue;
            }
            auto chars = split_u8_chars(seg.text);
            // 找所有句末标点位置 (不含最后一个字符)
            std::vector<int> split_points;
            for (int ci = 0; ci + 1 < (int)chars.size(); ++ci) {
                if (is_sent_end_ch(chars[ci])) split_points.push_back(ci);
            }
            if (split_points.empty()) {
                split_out.push_back(seg); // 没标点可拆, 保留原样
                continue;
            }
            // 按标点拆分, 时间按字符比例分配
            int prev_ci = 0;
            for (int sp : split_points) {
                V4Segment sub;
                sub.speaker_id = seg.speaker_id;
                sub.speaker_name = seg.speaker_name;
                for (int j = prev_ci; j <= sp; ++j)
                    sub.text += chars[j];
                double ratio_start = (double)prev_ci / chars.size();
                double ratio_end = (double)(sp + 1) / chars.size();
                sub.start_ms = seg.start_ms + (int)(dur_ms * ratio_start);
                sub.end_ms = seg.start_ms + (int)(dur_ms * ratio_end);
                if (!sub.text.empty()) split_out.push_back(sub);
                prev_ci = sp + 1;
                ++split_count;
            }
            // 剩余部分
            if (prev_ci < (int)chars.size()) {
                V4Segment sub;
                sub.speaker_id = seg.speaker_id;
                sub.speaker_name = seg.speaker_name;
                for (int j = prev_ci; j < (int)chars.size(); ++j)
                    sub.text += chars[j];
                double ratio_start = (double)prev_ci / chars.size();
                sub.start_ms = seg.start_ms + (int)(dur_ms * ratio_start);
                sub.end_ms = seg.end_ms;
                if (!sub.text.empty()) split_out.push_back(sub);
            }
        }
        if (split_count > 0)
            fprintf(stderr, "[Pipeline] v4 Phase 6.52: split %d ultra-long segments\n", split_count);
        v4_segments = std::move(split_out);
    }

    // ================================================================
    // Phase 6.55: 口语规范化
    // ================================================================
    if (params.clean_oral) {
        static const std::vector<std::string> oral_patterns = {
            "\xe6\x88\x91\xe6\x83\xb3\xe8\xaf\xb4",
            "\xe4\xbd\xa0\xe7\x9f\xa5\xe9\x81\x93\xe5\x90\x97",
            "\xef\xbc\x8c\xe5\x92\x8b",
            "\xef\xbc\x8c\xe5\xb0\xb1\xe6\x98\xaf",
            "\xe7\x84\xb6\xe5\x90\x8e\xe5\x91\xa2",
            "\xe5\xaf\xb9\xe5\x90\xa7",
            "\xe8\xbf\x99\xe6\xa0\xb7\xe7\x9a\x84\xe8\xaf\x9d",
            "\xef\xbc\x8c\xe6\x80\x8e\xe6\xa0\xb7",
            "\xef\xbc\x8c\xe8\xae\xb2",
            "\xe7\xad\x89\xe7\xad\x89",
            "\xe4\xb9\x9f\xe6\x98\xaf\xef\xbc\x8c",
            "\xef\xbc\x8c\xe6\x9c\x89\xe4\xb8\x00",
            "\xe8\xbf\x99\xe4\xb8\xaa\xe8\xaf\xb8",
            "\xef\xbc\x8c\xe5\x9c\xa8\xe4\xba\x8e",
            "\xe7\xb1\xbb\xe4\xba\x8b",
            "\xef\xbc\x8c\xe5\x8f\xab\xe4\xbb\x80\xe4\xb9\x88",
        };
        int oral_removed = 0;
        for (auto& seg : v4_segments) {
            for (const auto& pattern : oral_patterns) {
                size_t pos = 0;
                while ((pos = seg.text.find(pattern, pos)) != std::string::npos) {
                    seg.text.erase(pos, pattern.size());
                    ++oral_removed;
                }
            }
        }
        if (oral_removed > 0)
            fprintf(stderr, "[Pipeline] v4 Phase 6.55: removed %d oral redundancies\n", oral_removed);
    }

    // ================================================================
    // Phase 6.6: 空白区间填充
    // ================================================================
    if (!spk_intervals.empty() && !v4_segments.empty()) {
        std::stable_sort(v4_segments.begin(), v4_segments.end(),
            [](const V4Segment& a, const V4Segment& b) { return a.start_ms < b.start_ms; });

        std::vector<V4Segment> filled;
        int gap_filled = 0;
        float gap_duration_ms = 0;

        auto find_nearest_speaker = [&](int gap_mid, int default_spk_id, const std::string& default_name) {
            int best_dist = INT_MAX;
            int best_spk = default_spk_id;
            std::string best_name = default_name;
            for (auto& si : spk_intervals) {
                int center = (si.start_ms + si.end_ms) / 2;
                int dist = std::abs(center - gap_mid);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_spk = si.speaker_id;
                    best_name = si.speaker_name;
                }
            }
            return std::make_pair(best_spk, best_name);
        };

        // Fill gap before first segment
        if (v4_segments[0].start_ms > 0) {
            int gap_end = v4_segments[0].start_ms;
            if (gap_end >= 200) {
                auto [spk, name] = find_nearest_speaker(gap_end / 2,
                    v4_segments[0].speaker_id, v4_segments[0].speaker_name);
                V4Segment gap_seg;
                gap_seg.start_ms = 0;
                gap_seg.end_ms = gap_end;
                gap_seg.speaker_id = spk;
                gap_seg.speaker_name = name;
                filled.push_back(gap_seg);
                ++gap_filled;
                gap_duration_ms += gap_end;
            }
        }

        for (size_t i = 0; i < v4_segments.size(); ++i) {
            filled.push_back(v4_segments[i]);
            if (i + 1 < v4_segments.size()) {
                int gap_start = v4_segments[i].end_ms;
                int gap_end = v4_segments[i + 1].start_ms;
                if (gap_end - gap_start >= 200) {
                    auto [spk, name] = find_nearest_speaker((gap_start + gap_end) / 2,
                        v4_segments[i].speaker_id, v4_segments[i].speaker_name);
                    V4Segment gap_seg;
                    gap_seg.start_ms = gap_start;
                    gap_seg.end_ms = gap_end;
                    gap_seg.speaker_id = spk;
                    gap_seg.speaker_name = name;
                    filled.push_back(gap_seg);
                    ++gap_filled;
                    gap_duration_ms += gap_end - gap_start;
                }
            }
        }

        // Fill gap after last segment
        int audio_end_ms = (int)((float)wav.samples.size() / wav.sample_rate * 1000);
        if (v4_segments.back().end_ms < audio_end_ms - 200) {
            int gap_start = v4_segments.back().end_ms;
            auto [spk, name] = find_nearest_speaker((gap_start + audio_end_ms) / 2,
                v4_segments.back().speaker_id, v4_segments.back().speaker_name);
            V4Segment gap_seg;
            gap_seg.start_ms = gap_start;
            gap_seg.end_ms = audio_end_ms;
            gap_seg.speaker_id = spk;
            gap_seg.speaker_name = name;
            filled.push_back(gap_seg);
            ++gap_filled;
            gap_duration_ms += audio_end_ms - gap_start;
        }

        if (gap_filled > 0) {
            fprintf(stderr, "[Pipeline] v4 Phase 6.6: filled %d gaps (%.1fs)\n",
                    gap_filled, gap_duration_ms / 1000.0f);
            v4_segments = std::move(filled);
        }
    }

    // Final sort
    std::stable_sort(v4_segments.begin(), v4_segments.end(),
                     [](const V4Segment& a, const V4Segment& b) {
                         if (a.start_ms != b.start_ms) return a.start_ms < b.start_ms;
                         if (a.end_ms != b.end_ms) return a.end_ms < b.end_ms;
                         return a.speaker_id < b.speaker_id;
                     });

    fprintf(stderr, "[Pipeline] v4 done: %zu segments, %zu words, total %.1fs\n",
            v4_segments.size(), word_list.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - v4_t0).count());

    // Build result
    for (auto& seg : v4_segments) {
        result.full_text += seg.text;
        result.segments.push_back({seg.start_ms, seg.end_ms, seg.speaker_id,
                                   seg.speaker_name, seg.text, 0});
    }
    for (auto& w : word_list) {
        result.words.push_back({w.word, w.start_ms, w.end_ms, w.speaker_id, w.speaker_name});
    }

    return result;
}

// ============================================================================
// Plain Mode: Simple ASR (energy split for long audio, optional word timestamps)
// ============================================================================
TranscriptionResult TranscriptionPipeline::run_plain_mode(
        const audio::AudioData& wav, const TranscriptionParams& params) {
    TranscriptionResult result;
    float total_duration_s = (float)wav.samples.size() / wav.sample_rate;
    result.duration_s = total_duration_s;

    // Long audio: energy split
    if (total_duration_s > 100.0f) {
        fprintf(stderr, "[Pipeline] Plain mode: long audio %.1fs, energy-based split\n",
                total_duration_s);

        const int sr = wav.sample_rate;
        const int total_samples = (int)wav.samples.size();
        const int target_chunk_samples = 100 * sr;
        const int search_window = 15 * sr;
        const int energy_window = (int)(0.1f * sr);

        std::vector<int> split_points;
        split_points.push_back(0);

        int pos = 0;
        while (pos + target_chunk_samples < total_samples) {
            int center = pos + target_chunk_samples;
            int search_start = std::max(pos + target_chunk_samples / 2, center - search_window);
            int search_end = std::min(total_samples - energy_window, center + search_window);

            if (search_start >= search_end) {
                split_points.push_back(std::min(center, total_samples));
                pos = split_points.back();
                continue;
            }

            float min_energy = 1e30f;
            int best_pos = center;
            int step = std::max(1, energy_window / 4);
            for (int s = search_start; s < search_end; s += step) {
                float energy = 0.0f;
                int end = std::min(s + energy_window, total_samples);
                for (int k = s; k < end; k++) {
                    float v = wav.samples[k];
                    energy += v * v;
                }
                if (energy < min_energy) {
                    min_energy = energy;
                    best_pos = s + energy_window / 2;
                }
            }
            split_points.push_back(best_pos);
            pos = best_pos;
        }
        split_points.push_back(total_samples);

        int num_chunks = (int)split_points.size() - 1;
        std::string full_text;
        std::vector<std::vector<float>> all_plain_pcms;
        for (int ci = 0; ci < num_chunks; ci++) {
            int s = split_points[ci];
            int e = split_points[ci + 1];
            if (e - s < sr / 5) continue;
            all_plain_pcms.emplace_back(wav.samples.begin() + s, wav.samples.begin() + e);
        }

        auto* native_plain = dynamic_cast<plugins::NativeAsrPlugin*>(deps_.asr_plugin);
        if (native_plain && all_plain_pcms.size() >= 2) {
            std::vector<plugins::NativeAsrPlugin::PcmChunk> pcm_chunks;
            for (auto& pcm : all_plain_pcms)
                pcm_chunks.push_back({pcm.data(), (int)pcm.size()});
            auto batch_results = native_plain->transcribe_batch_pcm(
                pcm_chunks, wav.sample_rate, params.language, true);
            for (auto& r : batch_results)
                if (r.error_code == 0 && !r.text.empty())
                    full_text += r.text;
        } else {
            for (auto& chunk_pcm : all_plain_pcms) {
                auto seg_result = deps_.asr_plugin->transcribe_pcm(
                    chunk_pcm.data(), (int)chunk_pcm.size(),
                    wav.sample_rate, params.language, true);
                if (seg_result.error_code == 0 && !seg_result.text.empty())
                    full_text += seg_result.text;
            }
        }

        result.full_text = full_text;
    } else {
        // Short audio: direct transcribe
        auto asr_result = deps_.asr_plugin->transcribe_pcm(
            wav.samples.data(), (int)wav.samples.size(),
            wav.sample_rate, params.language, params.suppress_early_eos);
        if (asr_result.error_code != 0) {
            result.error_code = asr_result.error_code;
            result.error_message = asr_result.error_message;
            return result;
        }
        result.full_text = asr_result.text;
    }

    // Punctuation
    if (params.punctuate && deps_.punctuation_restorer && !result.full_text.empty())
        result.full_text_with_punc = deps_.punctuation_restorer->restore(result.full_text);

    // Word timestamps (optional)
    if (params.want_word_timestamps && !result.full_text.empty()) {
        auto aligned = run_forced_alignment(wav, result.full_text);
        for (auto& aw : aligned)
            result.words.push_back({aw.word, aw.start_ms, aw.end_ms, -1, ""});
    }

    if (result.full_text.empty()) {
        result.error_code = 3;
        result.error_message = "ASR transcription produced no text";
    }

    return result;
}

} // namespace asr
} // namespace qwen_thor
