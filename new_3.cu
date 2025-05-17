#include <iostream>
#include <bits/stdc++.h>
// #define n 13
// #define m 23
using namespace std;
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;
__global__ void bfs_round_kernel(
    const int*  vertex,
    const int*  edges,
    const int*  frontier,
    int          k,
    int          n,
    int*         visited,
    int*         level,
    int*         next_frontier,
    int*         nextCount,
    int          depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k) return;
    int u = frontier[tid];
    for (int e = vertex[u]; e < vertex[u+1]; e++) {
        int v = edges[e];
        if (atomicExch(&visited[v], 1) == 0) {
            level[v] = depth + 1;
            int pos = atomicAdd(nextCount, 1);
            if (pos < n) next_frontier[pos] = v;
        }
    }
}

// CPU-side “tail” of BFS
void cpu_bfs_round(
    const int*  frontier,
    int         frontier_size,
    int         start_idx,
    const int*  vertex,
    const int*  edges,
    int*        visited,
    int*        level,
    int*        next_frontier,
    int&        nextCount,
    int         depth
) {
    for (int idx = start_idx; idx < frontier_size; idx++) {
        int u = frontier[idx];
        for (int e = vertex[u]; e < vertex[u+1]; e++) {
            int v = edges[e];
            if (!visited[v]) {
                visited[v]    = 1;
                level[v]      = depth + 1;
                next_frontier[nextCount++] = v;
            }
        }
    }
}
void createNewCSR(
    int                       n,
    const int*                vertex,
    const int*                edges,
    const std::vector<int>&   W,
    const std::vector<int>&   S,
    int*                      new_vertex,
    int*&                     new_edges,
    int&                      new_m
) {
    int k = (int)W.size();

    // 1) Build a lookup to tell which vertices are sinks
    std::vector<bool> isSink(n, false);
    for (int u : S) {
        isSink[u] = true;
    }

    // 2) Build inverse‐map for W → new‐IDs [0..k-1]
    //    (we leave invPerm[v] = -1 for sinks)
    std::vector<int> invPerm(n, -1);
    for (int i = 0; i < k; i++) {
        invPerm[W[i]] = i;
    }

    // 3) First pass: count, per W‐row, how many *non‑sink* edges it has
    new_vertex[0] = 0;
    for (int i = 0; i < k; i++) {
        int oldV = W[i];
        int start = vertex[oldV];
        int end   = vertex[oldV+1];
        int cnt   = 0;
        for (int e = start; e < end; e++) {
            int nbr = edges[e];
            if (!isSink[nbr]) {
                ++cnt;
            }
        }
        new_vertex[i+1] = new_vertex[i] + cnt;
    }

    // 4) Allocate edges array
    new_m      = new_vertex[k];
    new_edges  = new int[new_m];

    // 5) Second pass: actually fill them in
    //    track per‐row fill position
    std::vector<int> fillPos(k, 0);
    for (int i = 0; i < k; i++) {
        int oldV    = W[i];
        int start   = vertex[oldV];
        int end     = vertex[oldV+1];
        int baseOff = new_vertex[i];

        for (int e = start; e < end; e++) {
            int nbr = edges[e];
            if (!isSink[nbr]) {
                // remap neighbor to its new‐ID
                int newNbr = invPerm[nbr];
                int pos    = baseOff + fillPos[i]++;
                new_edges[pos] = newNbr;
            }
        }
    }
}


void reorder(int n, int m, int* vertex, int* edges, vector<int>& W , vector<int>& S) {
     // Dynamic list
    int* N = new int[n]; // Dynamic array of size n
    int head = 0;
    int tail = 0;

    int* count = new int[n]();

    for (int i = 0; i < m; i++) {
        count[edges[i]]++;
    }

    int* V = new int[n];
    for (int i = 0; i < n; i++) {
        V[i] = count[i];
    }

    std::sort(V, V + n, std::greater<int>());

    /*for (int i = 0; i < n; i++) {
        cout << V[i] << endl;
    }
    cout << endl;*/
    W.resize(n, -1); // Resize W and initialize with -1
    vector<bool> inW(n, false); // To check if a vertex is in W
    vector<bool> inS(n, false); // To check if a vertex is in S

    // Sort vertices by in-degree in descending order
    vector<int> sortedVertices(n);
    iota(sortedVertices.begin(), sortedVertices.end(), 0);
    sort(sortedVertices.begin(), sortedVertices.end(), [&](int a, int b) {
        return count[a] > count[b];
    });

    for (int vi : sortedVertices) {
        if (!inW[vi] && !inS[vi]) {
            bool hasOutNeighbors = (vertex[vi + 1] > vertex[vi]);
            if (hasOutNeighbors) {
                W[tail++] = vi;
                inW[vi] = true;
            } else {
                S.push_back(vi);
                inS[vi] = true;
            }

            while (head < tail) {
                int vcur = W[head++];
                vector<int> neighbors;

                // Collect all out-neighbors of vcur
                for (int j = vertex[vcur]; j < vertex[vcur + 1]; j++) {
                    neighbors.push_back(edges[j]);
                }

                // Sort neighbors by in-degree in descending order
                sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
                    return count[a] > count[b];
                });

                for (int vj : neighbors) {
                    if (!inW[vj] && !inS[vj]) {
                        bool vjHasOutNeighbors = (vertex[vj + 1] > vertex[vj]);
                        if (vjHasOutNeighbors) {
                            W[tail++] = vj;
                            inW[vj] = true;
                        } else {
                            S.push_back(vj);
                            inS[vj] = true;
                        }
                    }
                }
            }
        }
    }

    // Resize W to the actual size
    W.resize(tail);

    // Print W at the end
    /*cout << "W: ";
    for (int i = 0; i < tail; i++) {
        cout << W[i] << " ";
    }
    cout << endl;*/

    delete[] N;
    delete[] count;
    delete[] V;
}


void createCSR(int N, int arr1[], int arr2[], int edgeCount, int* vertex, int* edges) {
    // Initialize vertex array with zeros
    for (int i = 0; i <= N; i++) {
        vertex[i] = 0;
    }

    // Count the number of edges for each vertex
    for (int i = 0; i < edgeCount; i++) {
        if (arr1[i] >= 0 && arr1[i] < N) {
            vertex[arr1[i] + 1]++;
        } else {
            std::cerr << "Error: arr1[" << i << "] is out of bounds." << std::endl;
            exit(1);
        }
    }

    // Compute prefix sum to determine starting indices in edges array
    for (int i = 1; i <= N; i++) {
        vertex[i] += vertex[i - 1];
    }

    // Fill the edges array
    for (int i = 0; i < edgeCount; i++) {
        int u = arr1[i];
        int index = vertex[u]++;
        edges[index] = arr2[i];
    }

    // Restore vertex array to correct starting indices
    for (int i = N; i > 0; i--) {
        vertex[i] = vertex[i - 1];
    }
    vertex[0] = 0;
}



int main(int argc, char* argv[]) {
    int n=0;
    int m=0;

    #include <fstream>
    #include <sstream>

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <k> <src>" << std::endl;
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open file " << argv[1] << std::endl;
        return 1;
    }

    int k = std::stoi(argv[2]);
    if (k <= 0) {
        std::cerr << "Error: k must be a positive integer." << std::endl;
        return 1;
    }

    int src = std::stoi(argv[3]);
    if (src < 0) {
        std::cerr << "Error: src must be a non-negative integer." << std::endl;
        return 1;
    }

    inputFile >> n >> m;
    int* arr1 = new int[m];
    int* arr2 = new int[m];




    for (int i = 0; i < m; ++i) {
        int u, v, t, x;

        inputFile >> u >> v >> t >> x;
        //cout << u << "  " << v <<endl;
        arr1[i] = u;
        arr2[i] = v;
    }

    inputFile.close();

    int* vertex = new int[n + 1];
    int* edges = new int[m];

    createCSR(n, arr1, arr2, m, vertex, edges);


    vector<int> S;
    std::vector<int> W;
    reorder(n, m, vertex, edges, W,S);

    // Print W
    cout << "W: ";
    for (int i = 0; i < W.size(); i++) {
        cout << W[i] << " ";
    }
    cout << endl;

    // Print S
    cout <<endl << "S: " << endl;
    for (int i = 0; i < S.size(); i++) {
        cout << S[i] << " ";
    }
    cout << endl;


    int* new_vertex = new int[n+1];
    int* new_edges  = new int[m];

createNewCSR(n, vertex, edges, W, S, new_vertex, new_edges, m);




    // Print new_vertex
    cout << "new_vertex: ";
    for (int i = 0; i <= n; i++) {
        cout << new_vertex[i] << " ";
    }
    cout << endl;

    // Print new_edges
    cout << "new_edges: ";
    for (int i = 0; i < m; i++) {
        cout << new_edges[i] << " ";
    }
    cout << endl;

    int *visited_cpu = new int[n]();      // zero‐inited
    int *level_cpu   = new int[n];
    fill(level_cpu, level_cpu + n, -1);
    visited_cpu[src] = 1;
    level_cpu[src]   = 0;

    int *frontier_cpu      = new int[n];
    int  frontier_size     = 1;
    frontier_cpu[0]        = src;
    int *next_frontier_cpu = new int[n];
    int  nextCount_cpu     = 0;

    // --- Allocate+copy to GPU ---
    int *d_vertex, *d_edges;
    int *d_frontier, *d_visited, *d_level, *d_next_frontier, *d_nextCount;
    cudaMalloc(&d_vertex,      (n+1)*sizeof(int));
    cudaMalloc(&d_edges,       m*sizeof(int));
    cudaMalloc(&d_frontier,    n*sizeof(int));
    cudaMalloc(&d_visited,     n*sizeof(int));
    cudaMalloc(&d_level,       n*sizeof(int));
    cudaMalloc(&d_next_frontier, n*sizeof(int));
    cudaMalloc(&d_nextCount,   sizeof(int));

    cudaMemcpy(d_vertex, new_vertex, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,  new_edges,   m*sizeof(int),   cudaMemcpyHostToDevice);

    // zero‐init device visited/level
    cudaMemset(d_visited, 0, n*sizeof(int));
    cudaMemset(d_level,  -1, n*sizeof(int));

    // mark source on device
    int one = 1;
    cudaMemcpy(&d_frontier[0], &src,         sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_visited[src], &one,        sizeof(int), cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(&d_level[src],   &zero,       sizeof(int), cudaMemcpyHostToDevice);

    int depth = 0;
    const int THREADS = 256;

    // --- BFS main loop ---
    while (frontier_size > 0) {
        // reset GPU’s nextCount = 0
        cudaMemcpy(d_nextCount, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // copy current frontier to device
        cudaMemcpy(d_frontier, frontier_cpu,
                   frontier_size*sizeof(int), cudaMemcpyHostToDevice);

        // GPU handles up to k_gpu = min(k_user, frontier_size)
        int k_gpu = min(k, frontier_size);
        int blocks = (k_gpu + THREADS - 1)/THREADS;
        bfs_round_kernel<<<blocks, THREADS>>>(
            d_vertex, d_edges, d_frontier, k_gpu, n,
            d_visited, d_level, d_next_frontier, d_nextCount, depth
        );
        cudaDeviceSynchronize();

        // fetch how many the GPU actually enqueued
        cudaMemcpy(&k_gpu, d_nextCount, sizeof(int), cudaMemcpyDeviceToHost);

        // fetch those frontier items back to host
        cudaMemcpy(next_frontier_cpu, d_next_frontier,
                   k_gpu * sizeof(int), cudaMemcpyDeviceToHost);

        // CPU handles the tail [k_gpu .. frontier_size)
        nextCount_cpu = 0;
        if (frontier_size > k_gpu) {
            cpu_bfs_round(
                frontier_cpu, frontier_size, k_gpu,
                new_vertex, new_edges,
                visited_cpu, level_cpu,
                next_frontier_cpu, nextCount_cpu,
                depth
            );
        }

        // total new frontier size = GPU’s + CPU’s
        int new_frontier_size = k_gpu + nextCount_cpu;

        // swap frontiers: next_frontier_cpu is now our frontier_cpu
        memcpy(frontier_cpu, next_frontier_cpu,
               new_frontier_size * sizeof(int));
        frontier_size = new_frontier_size;

        depth++;
    }

    // --- copy final levels back from GPU and print both sides ---
    int* final_level_gpu = new int[n];
    cudaMemcpy(final_level_gpu, d_level, n*sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Depth by GPU/CPU hybrid:\n";
    for (int i = 0; i < n; i++)
        cout << final_level_gpu[i] << " ";
    cout << endl;

    // cleanup…
    cudaFree(d_vertex); cudaFree(d_edges);
    cudaFree(d_frontier); cudaFree(d_visited);
    cudaFree(d_level);   cudaFree(d_next_frontier);
    cudaFree(d_nextCount);

    delete[] new_vertex; delete[] new_edges;
    delete[] visited_cpu; delete[] level_cpu;
    delete[] frontier_cpu; delete[] next_frontier_cpu;
    delete[] final_level_gpu;

    return 0;

    return 0;
}
