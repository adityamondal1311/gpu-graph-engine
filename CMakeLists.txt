cmake_minimum_required(VERSION 3.18)
project(GraphEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 2060 = sm_75
set(CMAKE_CUDA_ARCHITECTURES 75)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Phase 1: CPU-only ─────────────────────────────────────────────────────────
add_executable(graph_engine
    src/main.cpp
    src/graph.cpp
    src/cpu_algorithms.cpp
)

# ── Phase 2-5: CUDA ───────────────────────────────────────────────────────────
add_executable(graph_engine_cuda
    src/main_cuda.cu
    src/graph.cpp
    src/cpu_algorithms.cpp
    src/gpu_bfs.cu
    src/gpu_bfs_async.cu
    src/gpu_dijkstra.cu
)
set_target_properties(graph_engine_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ── Phase 6: Block size tuning binary ────────────────────────────────────────
add_executable(graph_engine_tune
    src/benchmark_blocksize.cu
    src/graph.cpp
    src/cpu_algorithms.cpp
)
set_target_properties(graph_engine_tune PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch:  sm_${CMAKE_CUDA_ARCHITECTURES}")
