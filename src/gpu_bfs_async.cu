// gpu_bfs_async.cu  ── Phase 4: CUDA Streams
// ─────────────────────────────────────────────────────────────────────────────
// Problem Phase 3 still has:
//   Each BFS query is fully sequential:
//     [transfer graph] → [kernel] → [transfer results back] → next query
//   The GPU sits idle during host-device transfers.
//
// Phase 4 fix — pipeline with 2 streams:
//
//   Stream 0:  [transfer Q0] → [kernel Q0] → [readback Q0]
//   Stream 1:          [transfer Q1] → [kernel Q1] → [readback Q1]
//                              ↑ overlaps with kernel Q0
//
//   Net effect: transfer latency of Q1 is hidden behind kernel of Q0.
//   For K queries, we save (K-1) × transfer_time.
//
// Key CUDA APIs used:
//   cudaMemcpyAsync   — non-blocking copy, returns before transfer completes
//   cudaStreamCreate  — creates an independent HW queue on the GPU
//   cudaStreamSynchronize — wait for all work in one stream to finish
//   cudaEventRecord   — timestamp inside a stream for accurate timing
//
// IMPORTANT: cudaMemcpyAsync requires PINNED (page-locked) host memory.
//   Regular malloc'd memory can't be DMA'd asynchronously.
//   We use cudaMallocHost / cudaFreeHost for all host buffers here.
// ─────────────────────────────────────────────────────────────────────────────

#include "gpu_bfs_async.h"
#include "memory_pool.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <string>
#include <chrono>
#include <cstdio>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                            \
    if (e != cudaSuccess)                                              \
        throw std::runtime_error(std::string("CUDA: ")                 \
            + cudaGetErrorString(e)                                    \
            + " at " __FILE__ ":" + std::to_string(__LINE__));         \
} while(0)

static constexpr int BLOCK_SIZE = 128;

// ── BFS kernel (same as Phase 3, stream-aware via launch parameter) ───────────
__global__ void bfs_kernel_async(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    int*       dist,
    const int* frontier,
    int*       next_frontier,
    int*       next_size,
    int        frontier_size,
    int        current_level
) {
    __shared__ int s_frontier[BLOCK_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    if (tid < frontier_size)
        s_frontier[lid] = frontier[tid];
    __syncthreads();

    if (tid >= frontier_size) return;

    int u         = s_frontier[lid];
    int row_start = __ldg(&row_ptr[u]);
    int row_end   = __ldg(&row_ptr[u + 1]);

    for (int e = row_start; e < row_end; e++) {
        int v = __ldg(&col_idx[e]);
        if (atomicCAS(&dist[v], -1, current_level + 1) == -1) {
            int pos = atomicAdd(next_size, 1);
            next_frontier[pos] = v;
        }
    }
}

// ── Per-stream state ──────────────────────────────────────────────────────────
// Each stream gets its own device memory slab and pinned host result buffer.
struct StreamState {
    cudaStream_t stream;

    // Device memory (from pool)
    int* d_row_ptr;
    int* d_col_idx;
    int* d_dist;
    int* d_frontier;
    int* d_next_frontier;
    int* d_next_size;

    // Pinned host memory for async readback
    int* h_dist_pinned;
    int  N;

    void init(int n, int m, GpuMemoryPool& pool) {
        N = n;
        CUDA_CHECK(cudaStreamCreate(&stream));
        d_row_ptr       = pool.alloc_typed<int>(n + 1);
        d_col_idx       = pool.alloc_typed<int>(m);
        d_dist          = pool.alloc_typed<int>(n);
        d_frontier      = pool.alloc_typed<int>(n);
        d_next_frontier = pool.alloc_typed<int>(n);
        d_next_size     = pool.alloc_typed<int>(1);
        CUDA_CHECK(cudaMallocHost(&h_dist_pinned, n * sizeof(int)));
    }

    void destroy() {
        cudaStreamDestroy(stream);
        cudaFreeHost(h_dist_pinned);
    }
};

// ── Run BFS for one source on one stream (fully async) ───────────────────────
static void launch_bfs_async(StreamState& s, const CSRGraph& g, int source) {
    int N = g.num_nodes;

    // Async transfer: graph CSR to device (non-blocking on host)
    CUDA_CHECK(cudaMemcpyAsync(s.d_row_ptr, g.row_ptr.data(),
        (N+1)*sizeof(int), cudaMemcpyHostToDevice, s.stream));
    CUDA_CHECK(cudaMemcpyAsync(s.d_col_idx, g.col_idx.data(),
        g.num_edges*sizeof(int), cudaMemcpyHostToDevice, s.stream));

    // Init dist = -1 (async memset)
    CUDA_CHECK(cudaMemsetAsync(s.d_dist, -1, N*sizeof(int), s.stream));

    // Set source distance = 0 (async)
    int zero = 0;
    CUDA_CHECK(cudaMemcpyAsync(s.d_dist + source, &zero,
        sizeof(int), cudaMemcpyHostToDevice, s.stream));

    // Seed frontier with source
    CUDA_CHECK(cudaMemcpyAsync(s.d_frontier, &source,
        sizeof(int), cudaMemcpyHostToDevice, s.stream));

    // BFS level loop — kernels are queued into the stream
    // NOTE: frontier_size readback requires a sync, so levels are still serial.
    // The overlap happens BETWEEN queries (transfer Q1 while kernel Q0 runs).
    int frontier_size = 1;
    int level = 0;

    // Sync just the stream to get initial frontier size (source only = 1, known)
    while (frontier_size > 0) {
        CUDA_CHECK(cudaMemsetAsync(s.d_next_size, 0, sizeof(int), s.stream));

        int blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfs_kernel_async<<<blocks, BLOCK_SIZE, 0, s.stream>>>(
            s.d_row_ptr, s.d_col_idx,
            s.d_dist,
            s.d_frontier, s.d_next_frontier, s.d_next_size,
            frontier_size, level
        );

        // Read back next frontier size — this sync is unavoidable for level loop
        CUDA_CHECK(cudaMemcpyAsync(&frontier_size, s.d_next_size,
            sizeof(int), cudaMemcpyDeviceToHost, s.stream));
        CUDA_CHECK(cudaStreamSynchronize(s.stream));  // wait for size only

        std::swap(s.d_frontier, s.d_next_frontier);
        level++;
    }

    // Async readback of results into pinned memory
    CUDA_CHECK(cudaMemcpyAsync(s.h_dist_pinned, s.d_dist,
        N*sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    // Caller calls cudaStreamSynchronize to collect results
}

// ── Main API ──────────────────────────────────────────────────────────────────
AsyncBfsResult gpu_bfs_multi_stream(
    const CSRGraph&         g,
    const std::vector<int>& sources,
    int                     num_streams
) {
    int N = g.num_nodes;
    int M = g.num_edges;
    int K = (int)sources.size();

    // One pool per stream: row_ptr + col_idx + dist + frontier + next_frontier + next_size
    size_t per_stream = (size_t)(N+1)*sizeof(int)   // row_ptr
                      + (size_t) M   *sizeof(int)   // col_idx
                      + (size_t) N   *sizeof(int)   // dist
                      + (size_t) N   *sizeof(int)   // frontier
                      + (size_t) N   *sizeof(int)   // next_frontier
                      + 256 * 6;                    // alignment padding

    GpuMemoryPool pool(per_stream * num_streams);

    std::vector<StreamState> streams(num_streams);
    for (int s = 0; s < num_streams; s++)
        streams[s].init(N, M, pool);

    AsyncBfsResult result;
    result.distances.resize(K, std::vector<int>(N));

    auto wall_start = std::chrono::high_resolution_clock::now();

    // ── Pipeline: assign queries to streams in round-robin ───────────────────
    // Query i runs on stream (i % num_streams).
    // While stream 0 runs kernel for query 0, stream 1 transfers query 1's data.
    for (int i = 0; i < K; i++) {
        int si = i % num_streams;
        // If this stream is busy with a previous query, collect its result first
        if (i >= num_streams) {
            int prev = i - num_streams;
            int si_prev = prev % num_streams;
            CUDA_CHECK(cudaStreamSynchronize(streams[si_prev].stream));
            // Copy pinned result to output
            int* src = streams[si_prev].h_dist_pinned;
            result.distances[prev].assign(src, src + N);
        }
        launch_bfs_async(streams[si], g, sources[i]);
    }

    // Collect remaining in-flight queries
    for (int i = std::max(0, K - num_streams); i < K; i++) {
        int si = i % num_streams;
        CUDA_CHECK(cudaStreamSynchronize(streams[si].stream));
        int* src = streams[si].h_dist_pinned;
        result.distances[i].assign(src, src + N);
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    result.total_ms = std::chrono::duration<double,std::milli>(wall_end - wall_start).count();

    for (auto& s : streams) s.destroy();
    return result;
}
