#pragma once
#include "graph.h"
#include <vector>

// GPU BFS using CUDA.
// Returns hop-distance from source to every node (-1 = unreachable).
// Internally:
//   1. Copies CSR arrays to device memory.
//   2. Launches a BFS kernel per level: each thread handles one frontier node.
//   3. Uses atomicCAS to safely mark visited nodes without double-work.
std::vector<int> gpu_bfs(const CSRGraph& g, int source);

// Returns the time breakdown for the last call (host-device transfer + kernel time).
struct GpuBfsTiming {
    double transfer_ms;
    double kernel_ms;
};
GpuBfsTiming gpu_bfs_last_timing();
