// main_cuda.cu  ── Phase 2: GPU benchmark harness
// Runs CPU BFS, then GPU BFS, validates they agree, prints speedup.
// Usage:
//   ./graph_engine_cuda                          (synthetic benchmark)
//   ./graph_engine_cuda --file graph.txt         (SNAP file)
//   ./graph_engine_cuda --file graph.txt --source 42

#include <cstdio>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "graph.h"
#include "cpu_algorithms.h"
#include "gpu_bfs.h"

using Clock = std::chrono::high_resolution_clock;

// ── Wall-clock timer ──────────────────────────────────────────────────────────
struct Timer {
    std::chrono::time_point<Clock> t0;
    void  start() { t0 = Clock::now(); }
    double ms() const {
        return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    }
};

// ── Print GPU info ─────────────────────────────────────────────────────────────
static void print_gpu_info() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU : %s\n", p.name);
    printf("SMs : %d   |   Global mem: %.1f GB   |   sm_%d%d\n\n",
        p.multiProcessorCount,
        p.totalGlobalMem / 1e9,
        p.major, p.minor);
}

// ── Validate: GPU BFS must match CPU BFS exactly ──────────────────────────────
static bool validate(const std::vector<int>& cpu, const std::vector<int>& gpu) {
    if (cpu.size() != gpu.size()) return false;
    int mismatches = 0;
    for (int i = 0; i < (int)cpu.size(); i++) {
        // Both must agree on reachability; distances may differ only if
        // the graph has multiple shortest paths (same hop count is fine).
        if ((cpu[i] == -1) != (gpu[i] == -1)) mismatches++;
        else if (cpu[i] != -1 && cpu[i] != gpu[i])  mismatches++;
    }
    if (mismatches > 0)
        printf("  [VALIDATE] FAIL — %d mismatches\n", mismatches);
    else
        printf("  [VALIDATE] PASS — CPU and GPU BFS agree on all distances\n");
    return mismatches == 0;
}

// ── Run one benchmark (CPU vs GPU) on a loaded graph ─────────────────────────
static void run_benchmark(const CSRGraph& g, int source, const char* label) {
    printf("=== %s ===\n", label);
    g.print_stats();

    Timer t;
    const int RUNS = 5;

    // ── CPU BFS ──────────────────────────────────────────────────────────────
    double cpu_total = 0;
    std::vector<int> cpu_dist;
    for (int r = 0; r < RUNS; r++) {
        t.start();
        cpu_dist = cpu_bfs(g, source);
        cpu_total += t.ms();
    }
    double cpu_avg = cpu_total / RUNS;

    // ── GPU BFS ──────────────────────────────────────────────────────────────
    double gpu_total = 0;
    std::vector<int> gpu_dist;
    for (int r = 0; r < RUNS; r++) {
        t.start();
        gpu_dist = gpu_bfs(g, source);
        gpu_total += t.ms();
    }
    double gpu_avg = gpu_total / RUNS;

    // Breakdown of last GPU run
    GpuBfsTiming timing = gpu_bfs_last_timing();

    validate(cpu_dist, gpu_dist);

    // Reachable nodes
    int reachable = 0;
    for (int d : cpu_dist) if (d >= 0) reachable++;

    printf("  Source node    : %d\n", source);
    printf("  Reachable      : %d / %d\n", reachable, g.num_nodes);
    printf("  CPU BFS        : %.3f ms  (avg of %d runs)\n", cpu_avg, RUNS);
    printf("  GPU BFS total  : %.3f ms  (avg of %d runs)\n", gpu_avg, RUNS);
    printf("    transfer     : %.3f ms\n", timing.transfer_ms);
    printf("    kernel       : %.3f ms\n", timing.kernel_ms);
    printf("  Speedup        : %.2fx\n\n", cpu_avg / gpu_avg);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    printf("== Graph Engine Phase 2: GPU BFS ==\n\n");
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
        if (source >= g.num_nodes) {
            fprintf(stderr, "ERROR: source %d >= num_nodes %d\n", source, g.num_nodes);
            return 1;
        }
        run_benchmark(g, source, path.c_str());

    } else {
        // Synthetic benchmarks — same sizes as Phase 1 so speedup is comparable
        printf("Running synthetic benchmarks (same random seed as Phase 1)...\n\n");
        for (int N : {10000, 100000}) {
            char label[64]; sprintf(label, "Synthetic N=%d avg_deg=8", N);
            CSRGraph g = CSRGraph::random_graph(N, 8);
            run_benchmark(g, 0, label);
        }
    }
    return 0;
}
