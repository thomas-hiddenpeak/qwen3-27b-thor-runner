# P3: Phase 2&3 Parallelization Analysis

## Objective
Parallelize ForcedAligner (Phase 2) and VAD+CAM++ (Phase 3) to achieve ~25% TTFT speedup.

## Current Pipeline (Sequential)
```
Phase 1: ASR → Phase 2: ForcedAligner → Phase 3: VAD+CAM++ → Phase 4+
         (parallel already)        (sequential)      (sequential)
```

## Proposed Pipeline (Parallel)
```
Phase 1: ASR ├→ Phase 2: ForcedAligner ─┐
         └→ Phase 3: VAD+CAM++      ─→ Phase 4+
```

## Implementation Questions
1. Can Phase 2 (ForcedAligner) run independently on ASR output?
2. Can Phase 3 (VAD+CAM++) run independently on ASR output?
3. Are there shared resources that would cause contention?
4. What synchronization is needed before Phase 4?

## Technical Feasibility
- Phase 2: Input = ASR output text + timing. Can start immediately after ASR.
- Phase 3: Input = Audio signal + ASR output. Can start immediately after ASR.
- Shared resources: None identified yet (need code review).
- Synchronization point: Phase 4 needs results from both Phase 2 AND Phase 3.

## Next Steps for P3 Implementation
1. [ ] Code review: Identify Phase 2 & 3 input/output dependencies
2. [ ] GPU memory analysis: Can both phases run concurrently?
3. [ ] Threading implementation: Use std::thread or thread pool
4. [ ] Synchronization: Add barrier before Phase 4
5. [ ] Benchmarking: Measure speedup vs sequential

## Estimated Effort
- Analysis: 2 hours (code review, dependency graph)
- Implementation: 4-6 hours (threading, synchronization, testing)
- Benchmarking: 2 hours (latency measurement)
- Total: 8-10 hours

## Expected Improvement
- Current TTFT (sequential): ~250-500ms (estimate)
- Target TTFT (parallel): ~190-375ms (25% reduction)
- Throughput impact: Neutral to +5% (GPU shared resources)

## Status
✅ P3 Analysis Started
⏳ P3 Implementation: Ready to begin in next session
