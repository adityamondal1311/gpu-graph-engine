// main_cuda.cu  ── Phase 4: CUDA Streams benchmark
// Compares sequential GPU BFS vs pipelined multi-stream GPU BFS.
// Shows transfer/kernel overlap benefit across multiple queries.

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "graph.h"
#include "cpu_algorithms.h"
#include "gpu_bfs.h"
#include "gpu_bfs_async.h"

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

static bool validate(const std::vector<int>& cpu, const std::vector<int>& gpu) {
    int mismatches = 0;
    for (int i = 0; i < (int)cpu.size(); i++)
        if (cpu[i] != gpu[i]) mismatches++;
    if (mismatches) printf("  [VALIDATE] FAIL — %d mismatches\n", mismatches);
    else            printf("  [VALIDATE] PASS\n");
    return mismatches == 0;
}

// ── Single-query benchmark (Phase 2/3 baseline) ───────────────────────────────
static void run_single(const CSRGraph& g, int source, const char* label) {
    printf("=== %s ===\n", label);
    g.print_stats();

    const int RUNS = 5;
    Timer t;

    double cpu_total = 0; std::vector<int> cpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); cpu_dist = cpu_bfs(g, source); cpu_total += t.ms(); }

    double gpu_total = 0; std::vector<int> gpu_dist;
    for (int r = 0; r < RUNS; r++) { t.start(); gpu_dist = gpu_bfs(g, source); gpu_total += t.ms(); }

    GpuBfsTiming tm = gpu_bfs_last_timing();
    validate(cpu_dist, gpu_dist);

    int reachable = 0;
    for (int d : cpu_dist) if (d >= 0) reachable++;
    printf("  Reachable        : %d / %d\n", reachable, g.num_nodes);
    printf("  CPU BFS          : %.3f ms\n", cpu_total / RUNS);
    printf("  GPU BFS (single) : %.3f ms  [transfer %.3f | kernel %.3f]\n",
        gpu_total/RUNS, tm.transfer_ms, tm.kernel_ms);
    printf("  Speedup (vs CPU) : %.2fx\n\n", (cpu_total/RUNS) / (gpu_total/RUNS));
}

// ── Multi-query streams benchmark (Phase 4) ───────────────────────────────────
static void run_streams(const CSRGraph& g, const char* label) {
    printf("=== Phase 4 Streams: %s ===\n", label);

    // Use 8 source nodes spread across the graph
    int K = 8;
    std::vector<int> sources;
    for (int i = 0; i < K; i++)
        sources.push_back((i * g.num_nodes / K) % g.num_nodes);

    // Sequential baseline: K individual gpu_bfs calls
    Timer t;
    t.start();
    std::vector<std::vector<int>> seq_results;
    for (int s : sources) seq_results.push_back(gpu_bfs(g, s));
    double seq_ms = t.ms();

    // Streamed: 2-stream pipeline
    auto async2 = gpu_bfs_multi_stream(g, sources, 2);

    // Validate: streamed results must match sequential
    int mismatches = 0;
    for (int i = 0; i < K; i++)
        for (int v = 0; v < g.num_nodes; v++)
            if (seq_results[i][v] != async2.distances[i][v]) mismatches++;

    printf("  Queries          : %d sources\n", K);
    printf("  Sequential       : %.3f ms  (%.3f ms/query)\n",
        seq_ms, seq_ms/K);
    printf("  2-stream async   : %.3f ms  (%.3f ms/query)\n",
        async2.total_ms, async2.total_ms/K);
    printf("  Stream speedup   : %.2fx\n",  seq_ms / async2.total_ms);
    printf("  Validate         : %s\n\n", mismatches == 0 ? "PASS" : "FAIL");
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("== Graph Engine Phase 4: CUDA Streams ==\n\n");
    print_gpu_info();

    if (argc >= 2 && strcmp(argv[1], "--file") == 0) {
        if (argc < 3) { fprintf(stderr, "Usage: --file <path> [--source N]\n"); return 1; }
        std::string path = argv[2];
        int source = 0;
        for (int i = 3; i < argc; i++)
            if (strcmp(argv[i], "--source") == 0 && i+1 < argc)
                source = atoi(argv[++i]);

        CSRGraph g = CSRGraph::load_from_file(path);
        if (g.num_nodes == 0) return 1;
        run_single(g, source, path.c_str());
        run_streams(g, path.c_str());

    } else {
        // Single-query speedup
        for (int N : {10000, 100000}) {
            char label[64]; sprintf(label, "Synthetic N=%d avg_deg=8", N);
            CSRGraph g = CSRGraph::random_graph(N, 8);
            run_single(g, 0, label);
        }
        // Streams benefit on 100K graph
        {
            CSRGraph g = CSRGraph::random_graph(100000, 8);
            run_streams(g, "Synthetic N=100000 avg_deg=8");
        }
    }
    return 0;
}
