// benchmark_blocksize.cu  ── Phase 6: block size tuning
// Sweeps block sizes 64/128/256/512 for BFS and prints throughput table.
// Compile into a separate binary: graph_engine_tune

#include <cstdio>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "graph.h"
#include "cpu_algorithms.h"

using Clock = std::chrono::high_resolution_clock;

// ── Templated BFS kernel — block size is a compile-time template param ────────
template<int BSIZE>
__global__ void bfs_kernel_tuned(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*       dist,
    const int* frontier,
    int*       next_frontier,
    int*       next_size,
    int        frontier_size,
    int        current_level
) {
    __shared__ int s_frontier[BSIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    if (tid < frontier_size) s_frontier[lid] = frontier[tid];
    __syncthreads();
    if (tid >= frontier_size) return;

    int u         = s_frontier[lid];
    int row_start = __ldg(&row_ptr[u]);
    int row_end   = __ldg(&row_ptr[u+1]);

    for (int e = row_start; e < row_end; e++) {
        int v = __ldg(&col_idx[e]);
        if (atomicCAS(&dist[v], -1, current_level + 1) == -1) {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// ── Run BFS with a given block size, return average ms ───────────────────────
template<int BSIZE>
double run_bfs_tuned(
    int* d_row_ptr, int* d_col_idx, int* d_dist,
    int* d_frontier, int* d_next_frontier, int* d_next_size,
    int N, int source, int runs
) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        cudaMemset(d_dist, -1, N * sizeof(int));
        int zero = 0;
        cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice);
        int frontier_size = 1, level = 0;

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        while (frontier_size > 0) {
            cudaMemset(d_next_size, 0, sizeof(int));
            int blocks = (frontier_size + BSIZE - 1) / BSIZE;
            bfs_kernel_tuned<BSIZE><<<blocks, BSIZE>>>(
                d_row_ptr, d_col_idx, d_dist,
                d_frontier, d_next_frontier, d_next_size,
                frontier_size, level);
            cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost);
            std::swap(d_frontier, d_next_frontier);
            level++;
        }

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        total += ms;
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    return total / runs;
}

int main() {
    printf("== Phase 6: Block Size Sweep ==\n\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s  sm_%d%d\n\n", p.name, p.major, p.minor);

    for (int N : {100000, 500000}) {
        CSRGraph g = CSRGraph::random_graph(N, 8);
        printf("--- N=%d  Edges=%d ---\n", N, g.num_edges);
        printf("%-12s %-12s %-12s %-12s\n",
               "BlockSize", "Kernel(ms)", "Speedup", "Occupancy-hint");

        int *d_rp, *d_ci, *d_dist, *d_fr, *d_nfr, *d_ns;
        cudaMalloc(&d_rp,  (N+1)*sizeof(int));
        cudaMalloc(&d_ci,   g.num_edges*sizeof(int));
        cudaMalloc(&d_dist, N*sizeof(int));
        cudaMalloc(&d_fr,   N*sizeof(int));
        cudaMalloc(&d_nfr,  N*sizeof(int));
        cudaMalloc(&d_ns,   sizeof(int));
        cudaMemcpy(d_rp, g.row_ptr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ci, g.col_idx.data(), g.num_edges*sizeof(int), cudaMemcpyHostToDevice);

        double b64  = run_bfs_tuned< 64>(d_rp,d_ci,d_dist,d_fr,d_nfr,d_ns,N,0,5);
        double b128 = run_bfs_tuned<128>(d_rp,d_ci,d_dist,d_fr,d_nfr,d_ns,N,0,5);
        double b256 = run_bfs_tuned<256>(d_rp,d_ci,d_dist,d_fr,d_nfr,d_ns,N,0,5);
        double b512 = run_bfs_tuned<512>(d_rp,d_ci,d_dist,d_fr,d_nfr,d_ns,N,0,5);

        double best = std::min({b64, b128, b256, b512});
        printf("%-12d %-12.3f %-12s %-12s\n", 64,  b64,  b64==best?"<< BEST":"", "low reg pressure");
        printf("%-12d %-12.3f %-12s %-12s\n", 128, b128, b128==best?"<< BEST":"", "balanced");
        printf("%-12d %-12.3f %-12s %-12s\n", 256, b256, b256==best?"<< BEST":"", "balanced");
        printf("%-12d %-12.3f %-12s %-12s\n", 512, b512, b512==best?"<< BEST":"", "high occupancy");
        printf("\n");

        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_dist);
        cudaFree(d_fr); cudaFree(d_nfr); cudaFree(d_ns);
    }
    return 0;
}
