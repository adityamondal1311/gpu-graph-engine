// gpu_bfs.cu  ─ Phase 2: Parallel BFS on GPU
// ─────────────────────────────────────────────────────────────────────────────
// Algorithm (level-synchronous BFS):
//   Each iteration = one BFS level.
//   Launch one thread per node in the current frontier.
//   Each thread scans its neighbor list and tries to claim unvisited neighbors
//   using atomicCAS(visited[v], -1, current_level+1).
//
// Why atomicCAS and not atomicExch?
//   atomicCAS(addr, expected, desired) only writes if *addr == expected.
//   This means the first thread to reach node v wins; all others see the node
//   is already claimed and skip it.  No race, no double-work.
// ─────────────────────────────────────────────────────────────────────────────

#include "gpu_bfs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <cstring>

// ── CUDA error checking macro ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error: ")               \
                + cudaGetErrorString(err) + " at " + __FILE__                  \
                + ":" + std::to_string(__LINE__));                              \
        }                                                                       \
    } while (0)

// ── Global timing store ──────────────────────────────────────────────────────
static GpuBfsTiming g_last_timing = {0, 0};

// ── BFS kernel ───────────────────────────────────────────────────────────────
// Each thread is assigned one node from the current frontier.
// It scans that node's neighbor list and enqueues unvisited neighbors.
__global__ void bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*       dist,          // dist[v] = -1 means unvisited
    const int* frontier,      // current frontier (list of node IDs)
    int*       next_frontier,
    int*       next_size,     // atomic counter for next frontier size
    int        frontier_size,
    int        current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];

    for (int e = row_ptr[u]; e < row_ptr[u+1]; e++) {
        int v = col_idx[e];
        // Claim v atomically: only one thread can set dist[v] from -1
        if (atomicCAS(&dist[v], -1, current_level + 1) == -1) {
            // We won the race — enqueue v into next frontier
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// ── Host function ─────────────────────────────────────────────────────────────
std::vector<int> gpu_bfs(const CSRGraph& g, int source) {
    int N = g.num_nodes;
    int M = g.num_edges;

    // ── CUDA events for timing ────────────────────────────────────────────────
    cudaEvent_t t0, t1, t2;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));

    // ── Allocate device memory ────────────────────────────────────────────────
    int *d_row_ptr, *d_col_idx, *d_dist;
    int *d_frontier, *d_next_frontier, *d_next_size;

    CUDA_CHECK(cudaEventRecord(t0));  // start transfer timer

    CUDA_CHECK(cudaMalloc(&d_row_ptr,       (N+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,         M   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist,            N   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier,        N   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier,   N   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_size,             sizeof(int)));

    // ── Copy graph to device ──────────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(),   M  *sizeof(int), cudaMemcpyHostToDevice));

    // Init dist = -1 everywhere, then set source = 0
    CUDA_CHECK(cudaMemset(d_dist, -1, N * sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Seed frontier with source node
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int frontier_size = 1;

    CUDA_CHECK(cudaEventRecord(t1));  // end transfer, start kernel

    // ── BFS level loop ────────────────────────────────────────────────────────
    // TUNING NOTE (Phase 6): try BLOCK_SIZE = 64, 128, 256, 512.
    // For sparse graphs 128 or 256 usually wins.
    const int BLOCK_SIZE = 256;
    int level = 0;

    while (frontier_size > 0) {
        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));

        int blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_kernel<<<blocks, BLOCK_SIZE>>>(
            d_row_ptr, d_col_idx,
            d_dist,
            d_frontier, d_next_frontier, d_next_size,
            frontier_size, level
        );
        CUDA_CHECK(cudaGetLastError());

        // Read back next frontier size (small copy, necessary for the loop condition)
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap frontier pointers
        std::swap(d_frontier, d_next_frontier);
        level++;
    }

    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaEventSynchronize(t2));

    // ── Copy results back ─────────────────────────────────────────────────────
    std::vector<int> dist(N);
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, N * sizeof(int), cudaMemcpyDeviceToHost));

    // ── Record timings ────────────────────────────────────────────────────────
    float transfer_ms, kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&transfer_ms, t0, t1));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms,   t1, t2));
    g_last_timing = {(double)transfer_ms, (double)kernel_ms};

    // ── Free device memory ────────────────────────────────────────────────────
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_dist);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_size);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);

    return dist;
}

GpuBfsTiming gpu_bfs_last_timing() { return g_last_timing; }
