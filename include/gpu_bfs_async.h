#pragma once
#include "graph.h"
#include <vector>
#include <cuda_runtime.h>

// Async multi-query BFS using CUDA streams.
// Pipelines N queries so transfer and kernel overlap across queries.
// Returns distances for each source node.
struct AsyncBfsResult {
    std::vector<std::vector<int>> distances;  // distances[i] = BFS from sources[i]
    double total_ms;                          // wall time for all queries
};

AsyncBfsResult gpu_bfs_multi_stream(
    const CSRGraph&        g,
    const std::vector<int>& sources,
    int                    num_streams = 2
);
