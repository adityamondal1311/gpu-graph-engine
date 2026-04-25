// gpu_dijkstra.cu  ─ Phase 5: Parallel Dijkstra via Delta-Stepping
// ─────────────────────────────────────────────────────────────────────────────
// Why naive Dijkstra doesn't parallelize:
//   The priority queue has a serial "pop minimum" operation.
//   You can't extract the global minimum from millions of threads at once.
//
// Delta-stepping idea (Meyer & Sanders, 1998):
//   Instead of a priority queue, use B buckets of width Δ.
//   Bucket b holds nodes with tentative distance in [b*Δ, (b+1)*Δ).
//   Process the smallest non-empty bucket. Within a bucket, ALL nodes can
//   be relaxed IN PARALLEL — they're within Δ of each other so no ordering
//   violation occurs for "light" edges (weight ≤ Δ).
//   "Heavy" edges (weight > Δ) are deferred to a later bucket.
//   Δ ≈ 1/max_degree is a good starting point.
//
// GPU mapping:
//   One CUDA thread per node in the current bucket.
//   atomicMin updates distances (using integer representation of floats
//   since IEEE 754 floats are monotone under uint32 comparison for positives).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include "graph.h"
#include <vector>

std::vector<float> gpu_dijkstra(const CSRGraph& g, int source, float delta = -1.0f);
// delta = -1 → auto-select as (max_weight / avg_degree)
