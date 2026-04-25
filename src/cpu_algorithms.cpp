#include "cpu_algorithms.h"
#include "graph.h"
#include <queue>
#include <limits>
#include <vector>

// ─── BFS ─────────────────────────────────────────────────────────────────────
std::vector<int> cpu_bfs(const CSRGraph& g, int source) {
    std::vector<int> dist(g.num_nodes, -1);
    dist[source] = 0;

    std::queue<int> frontier;
    frontier.push(source);

    while (!frontier.empty()) {
        int u = frontier.front(); frontier.pop();
        for (int e = g.row_ptr[u]; e < g.row_ptr[u+1]; e++) {
            int v = g.col_idx[e];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                frontier.push(v);
            }
        }
    }
    return dist;
}

// ─── Dijkstra ─────────────────────────────────────────────────────────────────
// Classic binary-heap Dijkstra. O((V + E) log V).
// This is our CPU baseline. The GPU version will beat this.
std::vector<float> cpu_dijkstra(const CSRGraph& g, int source) {
    const float INF = std::numeric_limits<float>::infinity();
    std::vector<float> dist(g.num_nodes, INF);
    dist[source] = 0.0f;

    // min-heap: (distance, node)
    using pfi = std::pair<float, int>;
    std::priority_queue<pfi, std::vector<pfi>, std::greater<pfi>> pq;
    pq.push({0.0f, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;  // stale entry

        for (int e = g.row_ptr[u]; e < g.row_ptr[u+1]; e++) {
            int   v   = g.col_idx[e];
            float nd  = dist[u] + g.weights[e];
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    return dist;
}