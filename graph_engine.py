#!/usr/bin/env python3
"""
graph_engine.py — Python wrapper for the CUDA Graph Processing Engine.

Usage:
    python graph_engine.py --algo bfs   --input graph.txt --source 0
    python graph_engine.py --algo dijkstra --input graph.txt --source 0 --weighted
    python graph_engine.py --benchmark --sizes 10000,100000

Requires the compiled binary at: ./build/graph_engine_cuda  (or graph_engine for CPU-only)
"""

import argparse
import subprocess
import sys
import os
import time

BINARY_CUDA = "./build/graph_engine_cuda"
BINARY_CPU  = "./build/graph_engine"

def find_binary():
    if os.path.exists(BINARY_CUDA):
        return BINARY_CUDA
    if os.path.exists(BINARY_CPU):
        return BINARY_CPU
    raise FileNotFoundError(
        "No compiled binary found. Run:\n"
        "  mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)"
    )

def run_engine(args_list: list[str]) -> tuple[str, float]:
    binary = find_binary()
    cmd = [binary] + args_list
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print("ERROR:", result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout, elapsed

def main():
    parser = argparse.ArgumentParser(description="Graph Engine CLI wrapper")
    parser.add_argument("--algo",     choices=["bfs", "dijkstra"], help="Algorithm to run")
    parser.add_argument("--input",    type=str, help="Edge-list file path")
    parser.add_argument("--source",   type=int, default=0, help="Source node (default 0)")
    parser.add_argument("--weighted", action="store_true", help="Graph has edge weights")
    parser.add_argument("--benchmark",action="store_true", help="Run synthetic benchmark")
    parser.add_argument("--sizes",    type=str, default="10000,100000",
                        help="Comma-separated node counts for benchmark")
    args = parser.parse_args()

    if args.benchmark:
        for n in args.sizes.split(","):
            print(f"\n── Benchmark N={n} ──")
            out, wall = run_engine(["--benchmark", "--nodes", n.strip()])
            print(out)
            print(f"Wall time (including process start): {wall*1000:.1f} ms")
        return

    if not args.input:
        parser.error("--input required unless --benchmark is set")

    engine_args = ["--file", args.input, "--algo", args.algo, "--source", str(args.source)]
    if args.weighted:
        engine_args.append("--weighted")

    print(f"Running {args.algo.upper()} from node {args.source} on {args.input} ...")
    out, wall = run_engine(engine_args)
    print(out)
    print(f"Total wall time: {wall*1000:.1f} ms")

if __name__ == "__main__":
    main()
