# High-Performance Graph Processing Engine

CUDA-accelerated parallel BFS and Dijkstra on GPU with **10x+ speedup** over CPU baseline on large sparse graphs.

**Stack:** CUDA 13.2 · C++17 · CMake · Python wrapper  
**GPU tested:** NVIDIA GeForce RTX 2060 (sm_75, 30 SMs, 6.4GB)

---

## Architecture

```
CPU (host)                          GPU (device)
──────────────────────────────────────────────────────────
Graph loader (CSR)                  BFS kernel
CPU BFS baseline          ───────►  shared memory frontier cache
CPU Dijkstra baseline               atomicCAS node claiming
                                    __ldg() read-only cache
                          ◄───────  Delta-stepping Dijkstra
Benchmark harness                   atomicMin on uint-cast floats
Python CLI wrapper                  Custom memory pool (bump alloc)
CUDA Streams pipeline               2-stream async overlap
```

---

## Results

### BFS — CPU vs GPU

| Dataset | Nodes | Edges | CPU | GPU | Speedup |
|---|---|---|---|---|---|
| Synthetic | 10K | 80K | 2.3ms | 35ms | 0.07x (overhead dominates) |
| Synthetic | 100K | 800K | 27ms | 2.5ms | **10.86x** |
| Synthetic | 500K | 4M | — | 3.1ms | — |
| ca-GrQc (SNAP) | 5K | 29K | 0.8ms | 34ms | 0.02x (graph too small) |
| amazon0302 (SNAP) | 262K | 1.2M | 57ms | 41ms | 1.39x |

### Dijkstra (delta-stepping) — CPU vs GPU

| Dataset | Nodes | Edges | CPU | GPU | Speedup |
|---|---|---|---|---|---|
| Synthetic (weighted) | 100K | 800K | 104ms | 21ms | **4.78x** |
| ca-GrQc (SNAP) | 5K | 29K | 1.7ms | 5.4ms | 0.32x (graph too small) |
| amazon0302 (SNAP) | 262K | 1.2M | 137ms | 187ms | 0.73x (unweighted — delta-stepping degenerates) |

### Block size sweep (BFS kernel, RTX 2060 sm_75)

| Block Size | N=100K kernel | N=500K kernel |
|---|---|---|
| 64 | 0.748ms | 3.350ms |
| **128** | **0.727ms ★** | 3.784ms |
| 256 | 0.731ms | 3.352ms |
| **512** | 0.774ms | **3.121ms ★** |

**Finding:** optimal block size scales with frontier density. 128 wins for sparse graphs (N=100K), 512 wins when frontier is large (N=500K).

### CUDA Streams (multi-query BFS pipeline)

| Mode | 8 queries on N=100K | Per query |
|---|---|---|
| Sequential | 21ms | 2.6ms |
| 2-stream async | 12.7ms | 1.6ms |
| **Stream speedup** | **1.66x** | |

Transfer (0.6ms) overlaps with kernel (0.7ms) across streams.

---

## Key optimizations

**Phase 3 — Memory:**
- `GpuMemoryPool`: single `cudaMalloc` slab, bump allocator, eliminates per-run allocation overhead
- `__shared__ int s_frontier[BLOCK_SIZE]`: cooperative frontier load per block before neighbor scan
- `__ldg()` on CSR arrays: forces reads through read-only texture cache

**Phase 4 — Streams:**
- `cudaMemcpyAsync` + pinned host memory (`cudaMallocHost`)
- Round-robin stream assignment pipelines K queries — transfer Q[i+1] overlaps kernel Q[i]

**Phase 5 — Dijkstra:**
- Delta-stepping: bucket-based frontier replaces serial priority queue
- `atomicMin` on uint-cast floats (IEEE 754 positive floats are monotone under uint comparison)
- Auto-selects delta = max_weight / 10 for correctness-performance balance

**Phase 6 — Profiling (Nsight Systems + Nsight Compute):**
- Identified level-sync gap (frontier_size readback per BFS level) as primary CPU overhead
- Block size sweep: 128 optimal for sparse graphs on sm_75
- Warp divergence present in neighbor-list scan for high-degree nodes (known sparse graph characteristic)

---

## Build

**Prerequisites:** CUDA Toolkit 12+, CMake 3.18+, C++17 compiler

```bash
# CPU-only baseline
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release --target graph_engine

# GPU (CUDA) — edit CMAKE_CUDA_ARCHITECTURES in CMakeLists.txt for your GPU
# sm_75 = RTX 2060/2080, sm_86 = RTX 3090, sm_89 = RTX 4090, sm_80 = A100
cmake --build . --config Release --target graph_engine_cuda

# Block size tuning binary
cmake --build . --config Release --target graph_engine_tune
```

**Find your GPU architecture:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## Usage

```bash
# CPU baseline
./graph_engine                                    # synthetic benchmark
./graph_engine --file graph.txt                   # SNAP edge-list file

# GPU
./graph_engine_cuda                               # synthetic benchmark
./graph_engine_cuda --file graph.txt              # SNAP file, BFS + Dijkstra
./graph_engine_cuda --file graph.txt --source 42  # custom source node

# Block size sweep
./graph_engine_tune

# Python wrapper
python graph_engine.py --algo bfs --input graph.txt --source 0
python graph_engine.py --benchmark
```

**Graph file format** (SNAP-compatible):
```
# comment lines ignored
# FromNodeId  ToNodeId  [Weight]
0    1    2.5
0    2    1.0
1    3    3.7
```

---

## SNAP Datasets

| Dataset | Nodes | Edges | Download |
|---|---|---|---|
| ca-GrQc | 5K | 29K | https://snap.stanford.edu/data/ca-GrQc.html |
| amazon0302 | 262K | 1.2M | https://snap.stanford.edu/data/amazon0302.html |
| com-Youtube | 1.1M | 3M | https://snap.stanford.edu/data/com-youtube.html |

---

## File structure

```
graph_engine/
├── include/
│   ├── graph.h                # CSR graph struct + loader
│   ├── cpu_algorithms.h       # CPU BFS + Dijkstra
│   ├── gpu_bfs.h              # Phase 2+3: CUDA BFS
│   ├── gpu_bfs_async.h        # Phase 4: multi-stream async BFS
│   ├── gpu_dijkstra.h         # Phase 5: delta-stepping Dijkstra
│   └── memory_pool.cuh        # Phase 3: GPU bump allocator
├── src/
│   ├── graph.cpp
│   ├── cpu_algorithms.cpp
│   ├── gpu_bfs.cu
│   ├── gpu_bfs_async.cu
│   ├── gpu_dijkstra.cu
│   ├── benchmark_blocksize.cu # Phase 6: block size sweep
│   ├── main.cpp               # CPU harness
│   └── main_cuda.cu           # GPU harness
├── graph_engine.py            # Python CLI wrapper
└── CMakeLists.txt
```