#include <iostream>
#include <bits/stdc++.h>
// #define n 13
// #define m 23
using namespace std;
#include <vector>
#include <algorithm>
#include <queue>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
__global__ void bfs_bitmap_gpu(
    const int*        vertex,         // CSR row‐pointer [0..k]
    const int*        edges,          // CSR col‐indices [0..new_vertex[k]-1]
    const int*        frontier,       // current frontier flags [0..k-1]
    int               k,              // number of vertices GPU should process
    int*              visited,        // visited flags [0..n-1]
    int*              next_frontier,  // next‐frontier flags [0..k-1]
    int*              level,          // BFS level [0..n-1]
    int               depth           // current BFS depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= k || frontier[tid] == 0) return;

    int start = vertex[tid];
    int end   = vertex[tid+1];
    for (int e = start; e < end; ++e) {
        int v = edges[e];
        // mark visited and enqueue
        if (atomicExch(&visited[v], 1) == 0) {
            level[v]           = depth + 1;
            next_frontier[v]   = 1;
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
    int&                      new_m,
    const std::string&        outputFolder
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
    std::vector<int> fillPos(k, 0);
    for (int i = 0; i < k; i++) {
        int oldV    = W[i];
        int start   = vertex[oldV];
        int end     = vertex[oldV+1];
        int baseOff = new_vertex[i];

        for (int e = start; e < end; e++) {
            int nbr = edges[e];
            if (!isSink[nbr]) {
                int newNbr = invPerm[nbr];
                int pos    = baseOff + fillPos[i]++;
                new_edges[pos] = newNbr;
            }
        }
    }

    // 6) Write (old_id, new_id) mapping for all vertices:
    std::ofstream mapFile("mapping.txt");

    // First, W‐vertices get IDs [0..k-1]
    for (int i = 0; i < k; ++i) {
        mapFile << "(" << W[i] << "," << i << ")\n";
    }
    mapFile << "Here on wardes sink\n";
    // Then, sink‐vertices in S get IDs [k..n-1]
    for (int j = 0; j < (int)S.size(); ++j) {
        mapFile << "(" << S[j] << "," << (k + j) << ")\n";
    }

    mapFile.close();
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    #include <fstream>
    #include <sstream>
    bool any_set;

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

    std::string inputFileName = argv[1];
    size_t lastSlash = inputFileName.find_last_of("/\\");
    std::string baseName = (lastSlash == std::string::npos) ? inputFileName : inputFileName.substr(lastSlash + 1);
    std::string outputFolder = "Out_" + baseName;

    if (mkdir(outputFolder.c_str(), 0777) == -1) {
        if (errno != EEXIST) {
            std::cerr << "Error: Could not create output folder " << outputFolder << std::endl;
            return 1;
        }
    }


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

    // Create and write to the output file
    std::ofstream outputFile(outputFolder + "/W_and_S.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not create file " << outputFolder + "/W_and_S.txt" << std::endl;
        return 1;
    }

    // Write W to the file
    outputFile << "W: ";
    for (int i = 0; i < W.size(); i++) {
        outputFile << W[i] << " ";
    }
    outputFile << std::endl;

    // Write S to the file
    outputFile << "S: ";
    for (int i = 0; i < S.size(); i++) {
        outputFile << S[i] << " ";
    }
    outputFile << std::endl;

    outputFile.close();

    int* new_vertex = new int[W.size() + 1];
    int* new_edges;
    int new_m;

    createNewCSR(n, vertex, edges, W, S, new_vertex, new_edges, new_m, outputFolder);
// Write the new CSR to the output folder
std::ofstream csrFile(outputFolder + "/new_CSR.txt");
if (!csrFile.is_open()) {
    std::cerr << "Error: Could not create file " << outputFolder + "/new_CSR.txt" << std::endl;
    return 1;
}

// Write new_vertex to the file
csrFile << "new_vertex: ";
for (int i = 0; i <= W.size(); i++) {
    csrFile << new_vertex[i] << " ";
}
csrFile << std::endl;

// Write new_edges to the file
csrFile << "new_edges: ";
for (int i = 0; i < m; i++) {
    csrFile << new_edges[i] << " ";
}
csrFile << std::endl;

csrFile.close();


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
    int* frontier      = new int[n]();  // initialized to 0
    int* next_frontier = new int[n]();
    int* visited       = new int[n]();  // initialized to 0
    int* level         = new int[n];
    std::fill(level, level + n, -1);

    frontier[src] = 1;
    visited[src]  = 1;
    level[src]    = 0;

    // --- 3) Device allocations ---
    int *d_vertex, *d_edges;
    int *d_frontier, *d_next_frontier, *d_visited, *d_level;

    cudaMalloc(&d_vertex,        (k+1) * sizeof(int));
    cudaMalloc(&d_edges,         vertex[k+1]      * sizeof(int));
    cudaMalloc(&d_frontier,      n      * sizeof(int));
    cudaMalloc(&d_next_frontier, n      * sizeof(int));
    cudaMalloc(&d_visited,       n      * sizeof(int));
    cudaMalloc(&d_level,         n      * sizeof(int));

    // copy sub-CSR once
    cudaMemcpy(d_vertex, new_vertex, (k+1)*sizeof(int),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,  new_edges,  vertex[k+1]      *sizeof(int),        cudaMemcpyHostToDevice);
    // initialize visited and level on device
    cudaMemcpy(d_visited, visited,   n * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_level,   level,     n * sizeof(int),            cudaMemcpyHostToDevice);

    const int THREADS = 1024;
    int depth = 0;

    // --- 4) BFS loop ---
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // zero next_frontier on host & device
        std::memset(next_frontier, 0, n * sizeof(int));
        cudaMemset(d_next_frontier, 0, n * sizeof(int));

        // copy frontier[0..k-1] and visited[0..n-1] to device
        cudaMemcpy(d_frontier, frontier, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_visited,  visited,  n * sizeof(int), cudaMemcpyHostToDevice);

        // launch GPU phase
        int blocks = (k + THREADS - 1) / THREADS;
        bfs_bitmap_gpu<<<blocks, THREADS>>>(
            d_vertex, d_edges,
            d_frontier, k,
            d_visited, d_next_frontier,
            d_level, depth
        );

        // CPU phase on vertices [k..n-1]

        #pragma omp parallel for
        for (int u = k; u < n; ++u) {
            if (frontier[u] == 0) continue;
            for (int e = new_vertex[u]; e < new_vertex[u+1]; ++e) {
                int v = new_edges[e];
                if (visited[v] == 0) {
                    visited[v]        = 1;
                    level[v]          = depth + 1;
                    next_frontier[v]  = 1;
                }
            }
        }
        cudaDeviceSynchronize();

        // fetch GPU’s next frontier flags for v<k
        cudaMemcpy(next_frontier, d_next_frontier, n * sizeof(int), cudaMemcpyDeviceToHost);
        // fetch updated visited and level for all vertices
        cudaMemcpy(visited, d_visited, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(level,   d_level,   n * sizeof(int), cudaMemcpyDeviceToHost);
        // Reset frontier for the next iteration
        std::memset(frontier, 0, n * sizeof(int));
        std::memcpy(frontier, next_frontier, n * sizeof(int));


        any_set = false;

        #pragma omp parallel for reduction(||:any_set)
        for (int i = 0; i < n; ++i) {
            frontier[i] = frontier[i] || next_frontier[i];  // OR-merge
            any_set = any_set || frontier[i];              // Reduction check
        }
        
        // Break if no more active frontiers
        if (!any_set) break;

        depth++;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Elapsed time for BFS: " << elapsed_time.count() << " seconds\n";

    // --- 5) Print levels ---
    std::cout << "BFS Levels from src=" << src << ":\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "vertex " << i << " -> level " << level[i] << "\n";
    }

    // Write levels to the output file
    std::ofstream levelFile(outputFolder + "/out_" + baseName);
    if (!levelFile.is_open()) {
        std::cerr << "Error: Could not create file " << outputFolder + "/out_" + baseName << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        levelFile << level[i] << "\n";
    }

    levelFile.close();

    for (int i = 0; i < n; ++i) {
        cout << level[i] << "\n";
    }
    std::cout << "Elapsed time for BFS: " << (elapsed_time.count() * 1000) << " milliseconds\n";

    // --- 6) Cleanup ---
    cudaFree(d_vertex);
    cudaFree(d_edges);
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_visited);
    cudaFree(d_level);

    delete[] frontier;
    delete[] next_frontier;
    delete[] visited;
    delete[] level;
    return 0;
}