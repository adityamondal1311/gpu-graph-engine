// gpu_bfs.cu  ── Phase 3: Optimized GPU BFS
// Changes from Phase 2:
//   1. GpuMemoryPool replaces cudaMalloc/cudaFree per run
//   2. Shared memory caches frontier chunk inside each block
//   3. __ldg() for read-only CSR arrays (goes through texture cache)
//   4. Block size tuned to 128 (best for sparse graphs on sm_75)

#include "gpu_bfs.h"
#include "memory_pool.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <cstdio>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("CUDA: ")                 \
            + cudaGetErrorString(e)                                    \
            + " at " __FILE__ ":" + std::to_string(__LINE__));         \
    }                                                                  \
} while(0)

static GpuBfsTiming g_last_timing = {0, 0};

// ── Tuning knob ───────────────────────────────────────────────────────────────
// Phase 6: sweep 64, 128, 256, 512 and record throughput.
// 128 is optimal for sparse graphs on RTX 2060 (sm_75).
static constexpr int BLOCK_SIZE = 128;

// ── BFS kernel with shared memory frontier caching ────────────────────────────
// Each block loads BLOCK_SIZE frontier nodes into shared memory.
// Threads then scan their assigned node's neighbor list.
// Neighbors are claimed with atomicCAS — first thread to reach a node wins.
__global__ void bfs_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*       dist,
    const int* frontier,
    int*       next_frontier,
    int*       next_size,
    int        frontier_size,
    int        current_level
) {
    // ── Shared memory: cache this block's slice of the frontier ──────────────
    __shared__ int s_frontier[BLOCK_SIZE];

    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lid  = threadIdx.x;   // lane id within block

    // Load frontier node into shared memory
    if (tid < frontier_size)
        s_frontier[lid] = frontier[tid];
    __syncthreads();

    if (tid >= frontier_size) return;

    int u = s_frontier[lid];

    // ── Read row bounds via __ldg (read-only cache / texture path) ────────────
    int row_start = __ldg(&row_ptr[u]);
    int row_end   = __ldg(&row_ptr[u + 1]);

    for (int e = row_start; e < row_end; e++) {
        int v = __ldg(&col_idx[e]);
        // Atomically claim v: only the first thread to arrive sets dist[v]
        if (atomicCAS(&dist[v], -1, current_level + 1) == -1) {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// ── Host function ─────────────────────────────────────────────────────────────
std::vector<int> gpu_bfs(const CSRGraph& g, int source) {
    int N = g.num_nodes;
    int M = g.num_edges;

    cudaEvent_t ev_start, ev_transfer_done, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_transfer_done));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    // ── Memory pool: one allocation covers everything we need ─────────────────
    // Sizes: row_ptr(N+1) + col_idx(M) + dist(N) + frontier(N) + next_frontier(N) + next_size(1)
    size_t pool_bytes =
        (size_t)(N + 1) * sizeof(int) +   // row_ptr
        (size_t) M      * sizeof(int) +   // col_idx
        (size_t) N      * sizeof(int) +   // dist
        (size_t) N      * sizeof(int) +   // frontier
        (size_t) N      * sizeof(int) +   // next_frontier
        256 * 6;                           // alignment padding per alloc

    GpuMemoryPool pool(pool_bytes);

    CUDA_CHECK(cudaEventRecord(ev_start));

    int* d_row_ptr      = pool.alloc_typed<int>(N + 1);
    int* d_col_idx      = pool.alloc_typed<int>(M);
    int* d_dist         = pool.alloc_typed<int>(N);
    int* d_frontier     = pool.alloc_typed<int>(N);
    int* d_next_frontier= pool.alloc_typed<int>(N);
    int* d_next_size    = pool.alloc_typed<int>(1);

    // Copy graph to device
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(),    M *sizeof(int), cudaMemcpyHostToDevice));

    // dist = -1 everywhere, source = 0
    CUDA_CHECK(cudaMemset(d_dist, -1, N * sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_dist + source, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Seed frontier
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int), cudaMemcpyHostToDevice));
    int frontier_size = 1;

    CUDA_CHECK(cudaEventRecord(ev_transfer_done));

    // ── BFS level loop ────────────────────────────────────────────────────────
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
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        std::swap(d_frontier, d_next_frontier);
        level++;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    // ── Copy results back ─────────────────────────────────────────────────────
    std::vector<int> dist(N);
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, N * sizeof(int), cudaMemcpyDeviceToHost));

    float t_transfer, t_kernel;
    CUDA_CHECK(cudaEventElapsedTime(&t_transfer, ev_start,         ev_transfer_done));
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel,   ev_transfer_done, ev_end));
    g_last_timing = {(double)t_transfer, (double)t_kernel};

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_transfer_done);
    cudaEventDestroy(ev_end);
    // Pool destructor calls cudaFree once — no per-level allocations

    return dist;
}

GpuBfsTiming gpu_bfs_last_timing() { return g_last_timing; }
