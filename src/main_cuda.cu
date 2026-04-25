// main_cuda.cu  ── Phase 5: BFS + Dijkstra benchmark harness

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#include "graph.h"
#include "cpu_algorithms.h"
#include "gpu_bfs.h"
#include "gpu_bfs_async.h"
#include "gpu_dijkstra.h"

using Clock = std::chrono::high_resolution_clock;

struct Timer {
    std::chrono::time_point<Clock> t0;
    void   start() { t0 = Clock::now(); }
    double ms() const {
        return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    }
};

static void print_gpu_info() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU : %s\n", p.name);
    printf("SMs : %d   |   Global mem: %.1f GB   |   sm_%d%d\n\n",
        p.multiProcessorCount, p.totalGlobalMem/1e9, p.major, p.minor);
}

// ── BFS benchmark ─────────────────────────────────────────────────────────────
static void run_bfs(const CSRGraph& g, int source, const char* label) {
    printf("=== BFS | %s ===\n", label);
    g.print_stats();

    const int RUNS = 5;
    Timer t;

    double cpu_total = 0; std::vector<int> cpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); cpu_dist = cpu_bfs(g, source); cpu_total += t.ms(); }

    double gpu_total = 0; std::vector<int> gpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); gpu_dist = gpu_bfs(g, source); gpu_total += t.ms(); }

    GpuBfsTiming tm = gpu_bfs_last_timing();

    int mismatches = 0;
    for (int i = 0; i < g.num_nodes; i++) if (cpu_dist[i] != gpu_dist[i]) mismatches++;
    int reachable = 0; for (int d : cpu_dist) if (d >= 0) reachable++;

    printf("  [VALIDATE] %s\n", mismatches == 0 ? "PASS" : "FAIL");
    printf("  Reachable  : %d / %d\n", reachable, g.num_nodes);
    printf("  CPU        : %.3f ms\n", cpu_total / RUNS);
    printf("  GPU        : %.3f ms  [transfer %.3f | kernel %.3f]\n",
           gpu_total/RUNS, tm.transfer_ms, tm.kernel_ms);
    printf("  Speedup    : %.2fx\n\n", (cpu_total/RUNS) / (gpu_total/RUNS));
}

// ── Dijkstra benchmark ────────────────────────────────────────────────────────
static void run_dijkstra(const CSRGraph& g, int source, const char* label) {
    printf("=== Dijkstra | %s ===\n", label);

    const int RUNS = 5;
    Timer t;

    // CPU Dijkstra
    double cpu_total = 0; std::vector<float> cpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); cpu_dist = cpu_dijkstra(g, source); cpu_total += t.ms(); }

    // GPU delta-stepping (auto delta)
    double gpu_total = 0; std::vector<float> gpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); gpu_dist = gpu_dijkstra(g, source); gpu_total += t.ms(); }

    // Validate: allow small floating point epsilon
    int mismatches = 0;
    for (int i = 0; i < g.num_nodes; i++) {
        float c = cpu_dist[i], gd = gpu_dist[i];
        bool both_inf = (c == std::numeric_limits<float>::infinity())
                     && (gd == std::numeric_limits<float>::infinity());
        if (!both_inf && std::fabs(c - gd) > 1e-3f) mismatches++;
    }

    int reachable = 0;
    for (float d : cpu_dist) if (d < std::numeric_limits<float>::infinity()) reachable++;

    printf("  [VALIDATE] %s%s\n",
        mismatches == 0 ? "PASS" : "FAIL",
        mismatches > 0  ? " — check delta value" : "");
    printf("  Reachable  : %d / %d\n", reachable, g.num_nodes);
    printf("  CPU        : %.3f ms\n", cpu_total / RUNS);
    printf("  GPU        : %.3f ms\n", gpu_total / RUNS);
    printf("  Speedup    : %.2fx\n\n", (cpu_total/RUNS) / (gpu_total/RUNS));
}

// ── Streams benchmark ─────────────────────────────────────────────────────────
static void run_streams(const CSRGraph& g, const char* label) {
    printf("=== Streams | %s ===\n", label);
    int K = 8;
    std::vector<int> sources;
    for (int i = 0; i < K; i++) sources.push_back((i * g.num_nodes / K) % g.num_nodes);

    Timer t;
    t.start();
    for (int s : sources) { auto d = gpu_bfs(g, s); (void)d; }
    double seq_ms = t.ms();

    auto async2 = gpu_bfs_multi_stream(g, sources, 2);

    printf("  Sequential : %.3f ms  (%.3f ms/query)\n", seq_ms, seq_ms/K);
    printf("  2-stream   : %.3f ms  (%.3f ms/query)\n", async2.total_ms, async2.total_ms/K);
    printf("  Stream spd : %.2fx\n\n", seq_ms / async2.total_ms);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("== Graph Engine Phase 5: BFS + Delta-Stepping Dijkstra ==\n\n");
    print_gpu_info();

    if (argc >= 2 && strcmp(argv[1], "--file") == 0) {
        if (argc < 3) { fprintf(stderr, "Usage: --file <path> [--source N]\n"); return 1; }
        std::string path = argv[2];
        int source = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--source") == 0 && i+1 < argc)
                source = atoi(argv[++i]);

        CSRGraph g = CSRGraph::load_from_file(path, /*weighted=*/true);
        if (g.num_nodes == 0) return 1;
        run_bfs(g, source, path.c_str());
        run_dijkstra(g, source, path.c_str());

    } else {
        for (int N : {10000, 100000}) {
            char label[64]; sprintf(label, "Synthetic N=%d avg_deg=8", N);
            CSRGraph g = CSRGraph::random_graph(N, 8);
            run_bfs(g, 0, label);
            run_dijkstra(g, 0, label);
        }
        CSRGraph g100k = CSRGraph::random_graph(100000, 8);
        run_streams(g100k, "Synthetic N=100000 avg_deg=8");
    }
    return 0;
}
