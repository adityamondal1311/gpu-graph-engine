#include "gpu_dijkstra.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
} while(0)

// We store distances as uint32 (bit_cast of float) so we can use atomicMin.
// Valid because positive IEEE 754 floats are monotone under unsigned comparison.
__device__ __forceinline__ unsigned float_to_uint(float f) {
    return __float_as_uint(f);
}

__device__ __forceinline__ float uint_to_float(unsigned u) {
    return __uint_as_float(u);
}

// ── Relaxation kernel ─────────────────────────────────────────────────────────
// Each thread processes one node from the active bucket.
// It relaxes all outgoing edges and updates dist[] via atomicMin.
// If dist[v] changed AND v is in the next bucket, it marks v as active.
__global__ void relax_kernel(
    const int*   __restrict__ row_ptr,
    const int*   __restrict__ col_idx,
    const float* __restrict__ weights,
    unsigned*    dist_uint,     // distances as uint (for atomicMin)
    const int*   bucket,        // current bucket nodes
    int          bucket_size,
    int*         next_bucket,
    int*         next_size,
    float        delta,
    int          current_b      // current bucket index
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bucket_size) return;

    int u = bucket[tid];
    float d_u = uint_to_float(dist_uint[u]);

    for (int e = row_ptr[u]; e < row_ptr[u+1]; e++) {
        int   v   = col_idx[e];
        float nd  = d_u + weights[e];
        unsigned nd_uint = float_to_uint(nd);

        unsigned old = atomicMin(&dist_uint[v], nd_uint);
        if (nd_uint < old) {
            // v was relaxed — check if it belongs in a nearby bucket
            int bv = (int)(nd / delta);
            if (bv <= current_b + 1) {  // within next bucket range
                int pos = atomicAdd(next_size, 1);
                next_bucket[pos] = v;
            }
        }
    }
}

std::vector<float> gpu_dijkstra(const CSRGraph& g, int source, float delta) {
    int N = g.num_nodes;
    int M = g.num_edges;

    // Auto-select delta
    if (delta < 0) {
        float max_w = *std::max_element(g.weights.begin(), g.weights.end());
        delta = max_w / (float)(g.num_edges / g.num_nodes + 1);
        delta = std::max(delta, 0.01f);
    }

    // ── Device allocations ────────────────────────────────────────────────────
    int      *d_row_ptr, *d_col_idx;
    float    *d_weights;
    unsigned *d_dist_uint;
    int      *d_bucket, *d_next_bucket, *d_next_size;

    CUDA_CHECK(cudaMalloc(&d_row_ptr,      (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx,        M  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_weights,        M  *sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist_uint,      N  *sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&d_bucket,         N  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_bucket,    N  *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_size,          sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(),   M  *sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, g.weights.data(),   M  *sizeof(float), cudaMemcpyHostToDevice));

    // Init dist = INF (as uint)
    unsigned INF_uint = float_to_uint(std::numeric_limits<float>::infinity());
    // cudaMemset sets bytes — use a kernel instead for correctness
    std::vector<unsigned> host_dist(N, INF_uint);
    host_dist[source] = float_to_uint(0.0f);
    CUDA_CHECK(cudaMemcpy(d_dist_uint, host_dist.data(), N*sizeof(unsigned), cudaMemcpyHostToDevice));

    // Seed bucket 0 with source
    CUDA_CHECK(cudaMemcpy(d_bucket, &source, sizeof(int), cudaMemcpyHostToDevice));
    int bucket_size = 1;

    const int BLOCK = 256;
    int current_b = 0;

    while (bucket_size > 0) {
        CUDA_CHECK(cudaMemset(d_next_size, 0, sizeof(int)));

        int blocks = (bucket_size + BLOCK - 1) / BLOCK;
        relax_kernel<<<blocks, BLOCK>>>(
            d_row_ptr, d_col_idx, d_weights,
            d_dist_uint,
            d_bucket, bucket_size,
            d_next_bucket, d_next_size,
            delta, current_b
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&bucket_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        std::swap(d_bucket, d_next_bucket);
        current_b++;
    }

    // ── Copy distances back ───────────────────────────────────────────────────
    std::vector<unsigned> result_uint(N);
    CUDA_CHECK(cudaMemcpy(result_uint.data(), d_dist_uint, N*sizeof(unsigned), cudaMemcpyDeviceToHost));

    std::vector<float> dist(N);
    for (int i = 0; i < N; i++) dist[i] = uint_to_float(result_uint[i]);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_weights);
    cudaFree(d_dist_uint); cudaFree(d_bucket); cudaFree(d_next_bucket); cudaFree(d_next_size);

    return dist;
}
