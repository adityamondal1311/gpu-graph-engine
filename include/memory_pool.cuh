// memory_pool.cuh  ─ Phase 3: Custom GPU Memory Pool
// ─────────────────────────────────────────────────────────────────────────────
// Problem: cudaMalloc/cudaFree are SLOW (microseconds each, serialized).
//          In graph workloads you allocate per-level — this adds up.
//
// Solution: Allocate one big slab upfront.  Hand out chunks with a pointer bump.
//           Reset in O(1) between runs.  No fragmentation for our use case
//           (all allocations are int arrays of known max size).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cstdint>

class GpuMemoryPool {
public:
    explicit GpuMemoryPool(size_t total_bytes) : capacity_(total_bytes), offset_(0) {
        cudaError_t err = cudaMalloc(&base_, total_bytes);
        if (err != cudaSuccess)
            throw std::runtime_error("GpuMemoryPool: cudaMalloc failed: "
                + std::string(cudaGetErrorString(err)));
    }

    ~GpuMemoryPool() { cudaFree(base_); }

    // Allocate n bytes (aligned to 256 bytes for coalescing).
    void* alloc(size_t bytes) {
        size_t aligned = (bytes + 255) & ~255ULL;  // round up to 256-byte boundary
        if (offset_ + aligned > capacity_)
            throw std::runtime_error("GpuMemoryPool: out of memory");
        void* ptr = static_cast<uint8_t*>(base_) + offset_;
        offset_ += aligned;
        return ptr;
    }

    // Typed convenience wrapper
    template<typename T>
    T* alloc_typed(size_t count) {
        return static_cast<T*>(alloc(count * sizeof(T)));
    }

    // Reset the pool — all previous pointers become invalid.
    // O(1) — just resets the offset cursor.
    void reset() { offset_ = 0; }

    size_t used()      const { return offset_; }
    size_t capacity()  const { return capacity_; }

    // No copy
    GpuMemoryPool(const GpuMemoryPool&) = delete;
    GpuMemoryPool& operator=(const GpuMemoryPool&) = delete;

private:
    void*  base_;
    size_t capacity_;
    size_t offset_;
};
