#include "graph.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <iostream>
#include <cassert>
#include <cstdio>

CSRGraph CSRGraph::load_from_file(const std::string& path, bool weighted) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", path.c_str());
        fprintf(stderr, "       Check the path is correct and the file exists.\n");
        return CSRGraph{};
    }

    std::unordered_map<int, int> id_map;
    std::vector<std::tuple<int,int,float>> edges;
    int next_id = 0;
    int lines_read = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        for (char& c : line) if (c == '\t') c = ' ';

        std::istringstream ss(line);
        int u, v; float w = 1.0f;
        if (!(ss >> u >> v)) continue;
        if (weighted) ss >> w;
        lines_read++;

        if (!id_map.count(u)) id_map[u] = next_id++;
        if (!id_map.count(v)) id_map[v] = next_id++;
        edges.emplace_back(id_map[u], id_map[v], w);
    }

    printf("  Parsed %d edge lines from file.\n", lines_read);

    int N = next_id;
    int M = (int)edges.size();
    std::vector<int> degree(N, 0);
    for (auto& [u, v, w] : edges) degree[u]++;

    CSRGraph g;
    g.num_nodes = N;
    g.num_edges = M;
    g.row_ptr.resize(N + 1, 0);
    g.col_idx.resize(M);
    g.weights.resize(M, 1.0f);

    for (int i = 0; i < N; i++) g.row_ptr[i+1] = g.row_ptr[i] + degree[i];
    std::vector<int> cursor(g.row_ptr.begin(), g.row_ptr.begin() + N);
    for (auto& [u, v, w] : edges) {
        int pos = cursor[u]++;
        g.col_idx[pos] = v;
        g.weights[pos] = w;
    }
    return g;
}

CSRGraph CSRGraph::random_graph(int N, int avg_degree, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> node_dist(0, N-1);
    std::uniform_real_distribution<float> weight_dist(1.0f, 10.0f);

    std::vector<std::tuple<int,int,float>> edges;
    edges.reserve((size_t)N * avg_degree);
    for (int u = 0; u < N; u++) {
        for (int d = 0; d < avg_degree; d++) {
            int v = node_dist(rng);
            if (v != u) edges.emplace_back(u, v, weight_dist(rng));
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end(),
        [](auto& a, auto& b){ return std::get<0>(a)==std::get<0>(b) && std::get<1>(a)==std::get<1>(b); }),
        edges.end());

    int M = (int)edges.size();
    std::vector<int> degree(N, 0);
    for (auto& [u, v, w] : edges) degree[u]++;

    CSRGraph g;
    g.num_nodes = N; g.num_edges = M;
    g.row_ptr.resize(N+1, 0); g.col_idx.resize(M); g.weights.resize(M);
    for (int i = 0; i < N; i++) g.row_ptr[i+1] = g.row_ptr[i] + degree[i];
    std::vector<int> cursor(g.row_ptr.begin(), g.row_ptr.begin() + N);
    for (auto& [u, v, w] : edges) { int pos = cursor[u]++; g.col_idx[pos]=v; g.weights[pos]=w; }
    return g;
}

void CSRGraph::print_stats() const {
    printf("Nodes: %d  Edges: %d  Avg degree: %.2f\n",
        num_nodes, num_edges, num_edges / (double)num_nodes);
}