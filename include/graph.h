#pragma once
#include <vector>
#include <string>
#include <limits>

// Compressed Sparse Row (CSR) graph representation.
// For a graph with N nodes and M edges:
//   row_ptr[i]..row_ptr[i+1]-1 are indices into col_idx[] for node i's neighbors.
//   col_idx[j] is the destination node of edge j.
//   weights[j] is the weight of edge j (1.0 for unweighted graphs).
struct CSRGraph {
    int num_nodes;
    int num_edges;
    std::vector<int>   row_ptr;   // size: num_nodes + 1
    std::vector<int>   col_idx;   // size: num_edges
    std::vector<float> weights;   // size: num_edges

    // Load from an edge-list file.
    // Format (SNAP-style):
    //   # comment lines start with #
    //   src dst          (unweighted)
    //   src dst weight   (weighted)
    // Nodes are re-indexed to 0..N-1 automatically.
    static CSRGraph load_from_file(const std::string& path, bool weighted = false);

    // Generate a random sparse graph (for quick testing without a dataset).
    static CSRGraph random_graph(int num_nodes, int avg_degree, unsigned seed = 42);

    void print_stats() const;
};
