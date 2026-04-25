#include <iostream>
#include <string>
#include <chrono>
#include <numeric>
#include <cstring>
#include "graph.h"
#include "cpu_algorithms.h"

// ── Tiny timer helper ────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

struct Timer {
    std::chrono::time_point<Clock> t0;
    void start() { t0 = Clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
};

// ── Validate BFS on a tiny hand-crafted graph ────────────────────────────────
void validate_bfs() {
    // 0→1, 0→2, 1→3, 2→3, 3→4
    // BFS from 0: dist = [0, 1, 1, 2, 3]
    CSRGraph g;
    g.num_nodes = 5; g.num_edges = 5;
    g.row_ptr = {0, 2, 3, 4, 5, 5};
    g.col_idx = {1, 2, 3, 3, 4};
    g.weights = {1,1,1,1,1};
    auto d = cpu_bfs(g, 0);
    bool ok = (d[0]==0 && d[1]==1 && d[2]==1 && d[3]==2 && d[4]==3);
    std::cout << "[validate_bfs] " << (ok ? "PASS" : "FAIL") << "\n";
}

void validate_dijkstra() {
    // Same graph, weights all 1
    CSRGraph g;
    g.num_nodes = 5; g.num_edges = 5;
    g.row_ptr = {0, 2, 3, 4, 5, 5};
    g.col_idx = {1, 2, 3, 3, 4};
    g.weights = {1,1,1,1,1};
    auto d = cpu_dijkstra(g, 0);
    bool ok = (d[0]==0 && d[1]==1 && d[2]==1 && d[3]==2 && d[4]==3);
    std::cout << "[validate_dijkstra] " << (ok ? "PASS" : "FAIL") << "\n";
}

// ── Benchmark one graph size ─────────────────────────────────────────────────
void benchmark(int N, int avg_deg, int runs = 3) {
    std::cout << "\n=== N=" << N << " avg_degree=" << avg_deg << " ===\n";
    CSRGraph g = CSRGraph::random_graph(N, avg_deg);
    g.print_stats();

    int source = 0;
    Timer t;

    // --- BFS ---
    double bfs_total = 0;
    for (int r = 0; r < runs; r++) {
        t.start();
        auto d = cpu_bfs(g, source);
        bfs_total += t.ms();
        (void)d;
    }
    printf("  CPU BFS:      %.2f ms  (avg of %d runs)\n", bfs_total/runs, runs);

    // --- Dijkstra ---
    double dijk_total = 0;
    for (int r = 0; r < runs; r++) {
        t.start();
        auto d = cpu_dijkstra(g, source);
        dijk_total += t.ms();
        (void)d;
    }
    printf("  CPU Dijkstra: %.2f ms  (avg of %d runs)\n", dijk_total/runs, runs);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::cout << "== Graph Engine Phase 1: CPU Baseline ==\n\n";

    validate_bfs();
    validate_dijkstra();

    if (argc >= 2 && strcmp(argv[1], "--file") == 0) {
        // Usage: ./graph_engine --file graph.txt [--weighted] [--source 0]
        std::string path = argv[2];
        bool weighted = false;
        int  source   = 0;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--weighted") == 0) weighted = true;
            if (strcmp(argv[i], "--source")   == 0 && i+1 < argc) source = atoi(argv[++i]);
        }
        CSRGraph g = CSRGraph::load_from_file(path, weighted);

        if (g.num_nodes == 0) {
            printf("ERROR: graph loaded 0 nodes. Check the file path and format.\n");
            printf("  Expected: one edge per line, format: 'src dst' or 'src dst weight'\n");
            printf("  Comment lines starting with '#' are skipped.\n");
            return 1;
        }

        g.print_stats();

        if (source >= g.num_nodes) {
            printf("ERROR: source node %d >= num_nodes %d\n", source, g.num_nodes);
            return 1;
        }

        printf("\n--- Running on: %s ---\n", path.c_str());
        Timer t;
        t.start(); auto bd = cpu_bfs(g, source);
        double bfs_ms = t.ms();
        t.start(); auto dd = cpu_dijkstra(g, source);
        double dijk_ms = t.ms();

        printf("CPU BFS:      %.2f ms\n", bfs_ms);
        printf("CPU Dijkstra: %.2f ms\n", dijk_ms);

        // Count reachable nodes
        int reachable = 0;
        for (int d : bd) if (d >= 0) reachable++;
        printf("Reachable from node %d: %d / %d nodes\n", source, reachable, g.num_nodes);
        (void)dd;
    } else {
        // Default: benchmark synthetic graphs
        benchmark(10'000,  8);
        benchmark(100'000, 8);
    }

    return 0;
}