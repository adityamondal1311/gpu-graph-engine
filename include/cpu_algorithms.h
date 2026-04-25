#pragma once
#include "graph.h"
#include <vector>

// CPU BFS — returns distance (hop count) from source to every node.
// Unreachable nodes get distance = -1.
std::vector<int> cpu_bfs(const CSRGraph& g, int source);

// CPU Dijkstra — returns shortest path distance (sum of weights) from source.
// Unreachable nodes get distance = std::numeric_limits<float>::infinity().
std::vector<float> cpu_dijkstra(const CSRGraph& g, int source);