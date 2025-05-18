#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <map>
#include <cuda_runtime.h>

#include "utils.h" // file handling utility functions
using namespace std;

size_t getAvailableGPUMemory() {
    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    return free;
}

// Global variables
bool debug = false; // Set to true for debugging
vector<int> original_map; // Maps reordered vertex indices to original indices
vector<int> reorder_map;  // Maps original vertex indices to reordered indices

/*  ------------------------------------------------------------------------------
                        Read Input and Convert to CSR Format
    ------------------------------------------------------------------------------  */
struct Edge {
    int u, v;
    float w;
};

void read_input(const string& filepath, vector<Edge>& edges, int& n, int& m, bool ignore_weights = true) {
    ifstream fin(filepath);
    if (!fin) {
        cerr << "Error opening input file from path: "<< filepath << endl;
        exit(1);
    }
    fin >> n >> m; edges.resize(m);

    // Use a map to store unique edges (u, v) -> w (keep min weight if duplicate)
    map<pair<int, int>, float> edge_map;
    for (int i = 0; i < m; ++i) {
        int u, v;
        float w = 1.0f;
        fin >> u >> v;
        if (!ignore_weights) {
            if (!(fin >> w)) {
                w = 1.0f;
                fin.clear();
            }
        }
        // Skip to end of line to avoid issues if weight is missing
        fin.ignore(numeric_limits<streamsize>::max(), '\n');
        auto key = make_pair(u, v);
        if (edge_map.count(key)) {
            if (!ignore_weights) {
                edge_map[key] = min(edge_map[key], w);
            }
            // else: do nothing, keep w=1
        } else {
            edge_map[key] = ignore_weights ? 1.0f : w;
        }
    }
    // Now fill the edges vector with unique edges
    edges.clear();
    for (const auto& kv : edge_map) {
        edges.push_back({kv.first.first, kv.first.second, ignore_weights ? 1.0f : kv.second});
    }
    m = edges.size();
}

void convert_to_csr(const vector<Edge>& edges, int n, vector<int>& offset, vector<int>& dest, vector<float>& weights) {
    // Initialize offset array
    offset.assign(n + 1, 0);

    // Count out-degree per vertex
    vector<int> out_degree(n, 0);
    for (const auto& e : edges) {
        out_degree[e.u]++;
    }

    // Build offset (start from 1, as 0th index is 0 always)
    for (int i = 1; i <= n; ++i) {
        offset[i] = offset[i - 1] + out_degree[i - 1];
    }

    // Initialize destination and weights
    dest.resize(offset[n]);
    weights.resize(offset[n]);

    // Fill in the CSR arrays
    vector<int> index(n, 0);
    for (const auto& e : edges) {
        int pos = offset[e.u] + index[e.u]++;
        dest[pos] = e.v;
        weights[pos] = e.w;
    }
}

/*  ------------------------------------------------------------------------------
                            Graph Reordering Algorithm
    ------------------------------------------------------------------------------  */
// Helper to sort vertices by descending in-degree
bool compare_indegree(const pair<int, int>& a, const pair<int, int>& b) {
    return a.second > b.second;
}

// Reorders a graph such that vertices with higher in-degree come first,
// and sink vertices (no out-neighbors) are placed at the end.
// Input: offset and dest represent the CSR of the original graph
// Output: reorder_map stores the reordered vertex list
int reorder_graph(const vector<int>& offset, const vector<int>& dest, int n) {
    // Step 1: Compute in-degrees of all vertices
    vector<int> in_degree(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            in_degree[dest[j]]++;
        }
    }

    // Step 2: Sort all vertices by in-degree (descending order)
    vector<pair<int, int>> sorted_vertices;
    for (int i = 0; i < n; ++i) {
        sorted_vertices.emplace_back(i, in_degree[i]);
    }
    sort(sorted_vertices.begin(), sorted_vertices.end(), compare_indegree);
    cout << "Sorted vertices by in-degree (vertex, in-degree, out-degree):" << endl;
    for (const auto& p : sorted_vertices) {
        int i = p.first;
        cout << "(v:" << i << ", i:" << in_degree[i] << ", o:" << offset[i + 1] - offset[i] << ") ";
    }
    cout << endl << endl;

    // Step 3: Initialize structures
    vector<bool> visited(n, false);  // For each vertex (true => belongs to {W or/union S})
    vector<int> W(n);                // List to store ordered vertices
    vector<int> S;                   // List to store sink vertices
    int tail = 0;                    // Tail index for W
    int head = 0;                    // Head index for BFS

    // Step 4: BFS-style traversal to expand W
    for (auto& p : sorted_vertices) {
        int v = p.first;
        if (visited[v]) continue;

        // Mark current vertex v as sink or not
        if (offset[v] == offset[v + 1]) {
            S.push_back(v);  // Sink vertex
        } else {
            W[tail++] = v;   // Add to main ordered list
        }
        visited[v] = true;   // Mark as visited
    
        // processes unvisited neighbors that has been enqueued from current vertex processes in FIFO order.
        if (debug) cout << ">> Processing BFS from vertex: " << v << endl;
        while (head < tail) {
            if (debug) cout << "BFS queue (W[head:tail]): ";
            for (int idx = head; idx < tail; ++idx) {
                cout << W[idx] << " ";
            }
            if (debug) cout << endl;
            int cur = W[head++]; // [dequeue]

            // Collect all neighbors (out-edges) of the current vertex
            vector<int> neighbors;
            for (int j = offset[cur]; j < offset[cur + 1]; ++j) {
                neighbors.push_back(dest[j]);
            }
            // Print neighbors of current vertex
            if (debug) cout << "Neighbors of vertex " << cur << ": ";
            for (int nb : neighbors) {
                if (debug) cout << nb << " ";
            }
            if (debug) cout << endl;

            // Sort neighbors by in-degree (descending)
            vector<pair<int, int>> sorted_neighbors;
            for (int v : neighbors) {
                if (!visited[v])
                    sorted_neighbors.emplace_back(v, in_degree[v]);
            }
            sort(sorted_neighbors.begin(), sorted_neighbors.end(), compare_indegree);
            // Print sorted unvisited neighbors with their in-degree
            if (debug) cout << "Sorted unvisited neighbors (vertex, in-degree): ";
            for (const auto& nb : sorted_neighbors) {
                if (debug) cout << "(" << nb.first << ", " << nb.second << ") ";
            }
            if (debug) cout << endl;

            // Process each unvisited neighbor
            for (auto& p : sorted_neighbors) {
                int v = p.first;
                if (visited[v]) continue;

                if (offset[v] == offset[v + 1]) {  //isSink(v) condition
                    S.push_back(v);  // Sink vertex
                } else {
                    W[tail++] = v;   // [enqueue] (v) based on bfs-indeg(desc)-ordered
                }
                visited[v] = true;   // belongs to {W or/union S}
            }
        }
    }
    if (debug) cout << endl;

    // Print W (orderlist) and S (sink array)
    cout << "BFS based reordering completed!" << endl << endl;
    cout << "Orderlist (W) => (new_idx, old_idx): ";
    for (int i = 0; i < tail; ++i) {
        cout << "(" << i << ", " << W[i] << ") ";
    }
    cout << endl;
    cout << "Sink vertices (S) => (new_idx, old_idx): ";
    for (int i = 0; i < S.size(); ++i) {
        cout << "(" << tail + i << ", " << S[i] << ") ";
    }
    cout << endl << endl;

    // Step 5: Combine W and S into final reordered list
    original_map.clear();
    original_map.insert(original_map.end(), W.begin(), W.begin() + tail);  // Ordered list
    original_map.insert(original_map.end(), S.begin(), S.end());           // Sink vertices
    reorder_map.assign(n, -1);
    for (int i = 0; i < n; ++i)
        reorder_map[original_map[i]] = i; // reverse mapping
    return tail;  // Return index where sinks start
}


/*  ------------------------------------------------------------------------------
                Subgraph Extraction Based on GPU Memory Constraints
    ------------------------------------------------------------------------------  */
// Helper function to calculate memory required for a subgraph
size_t calculateSubgraphMemory(const vector<int>& offset, const vector<int>& dest, 
                              const vector<int>& map, int selected_vertices, size_t min_buffer) {
    size_t total_memory = 0;
    
    // Memory for offset array (int)
    total_memory += (selected_vertices + 1) * sizeof(int);
    
    // Memory for destination array (int)
    int total_edges = 0;
    for (int i = 0; i < selected_vertices; ++i) {
        int v = map[i];
        total_edges += offset[v + 1] - offset[v];
    }
    total_memory += total_edges * sizeof(int);
    
    // Memory for weights (float)
    total_memory += total_edges * sizeof(float);
    
    // Add some buffer for auxiliary data
    total_memory += min_buffer; // 1KB buffer for safety
    
    return total_memory;
}

void extract_subgraph(const vector<int>& offset, const vector<int>& dest, const vector<float>& weights,
                     int n, size_t max_gpu_memory, size_t min_gpu_buffer,
                     vector<int>& subgraph_offset, vector<int>& subgraph_dest, vector<float>& subgraph_weights) {
    // Step 1: Precompute out-degrees for reordered vertices
    vector<int> out_degrees(n, 0);
    for (int i = 0; i < n; ++i) {
        int v = original_map[i]; // map new vertex ids to old vertex ids
        out_degrees[i] = offset[v + 1] - offset[v];
    }
    cout << "Out-degrees of reordered vertices (new_vertex_id: out_degree):" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "(" << i << ": " << out_degrees[i] << ") ";
    }
    cout << endl << endl;
    // Calculate cumulative sum of out-degree array (later use prefix sum)
    vector<int> cum_out_deg_sum(n, 0);
    cum_out_deg_sum[0] = out_degrees[0];
    for (int i = 1; i < n; ++i) {
        cum_out_deg_sum[i] = out_degrees[i] + cum_out_deg_sum[i-1];
    }
    cout << "Cumulative sum of out-degrees: ";
    for (int i = 0; i < n; ++i) {
        cout << cum_out_deg_sum[i] << " ";
    }
    cout << endl << endl;

    // Step 2: Binary search to find maximum number of vertices
    int left = 0, right = n;
    int selected_vertices = 0;
    
    while (left <= right) {
        int mid = (left + right) / 2;        
        // Calculate memory required for first 'mid' vertices
        size_t required_memory = 0;
        int total_edges = (mid == 0) ? 0 : cum_out_deg_sum[mid - 1]; // total edges for first 'mid' vertices
                
        // Calculate memory (offset + dest + weights)
        required_memory = (mid + 1) * sizeof(int) + 
                         total_edges * sizeof(int) + 
                         total_edges * sizeof(float) + min_gpu_buffer; // 1KB buffer for safety
        
        if (required_memory <= max_gpu_memory) {
            selected_vertices = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    } // selected_vertices now holds the max no. of vertices (wrt to reordered vertices)
    cout << "Selected vertices: " << selected_vertices << endl;
    int total_edges = (selected_vertices == 0) ? 0 : cum_out_deg_sum[selected_vertices - 1];
    cout << "Total edges: " << total_edges << endl;
    cout << "Memory required: " << calculateSubgraphMemory(offset, dest, original_map, selected_vertices, min_gpu_buffer) << endl << endl;

    // Step 3: Build subgraph => new offset, dest, and weights (wrt to reordered vertices)
    subgraph_offset.clear();
    subgraph_dest.clear();
    subgraph_weights.clear();
    
    if (debug) cout << "Subgraph edges (original -> reordered):" << endl;
    for (int i = 0; i < selected_vertices; ++i) {
        int u = original_map[i]; // i:reordered from_vertex (new_id), u: original from_vertex (old_id)
        for (int k = offset[u]; k < offset[u + 1]; ++k) {
            int v = dest[k];
            int j = reorder_map[v]; // j: reordered to_vertex (new_id), v:original to_vertex (old_id)
                        
            // Check if j is in the subgraph
            if (j < selected_vertices) {
                // Print mapping: edge (i, j) => (u, v) with weight
                if (debug) cout << "edge(" << u << "," << v << ") => (" << i << "," << j << ") w=" << weights[k] << endl;
                subgraph_dest.push_back(j);
                subgraph_weights.push_back(weights[k]);
            }
        }

        subgraph_offset.push_back(subgraph_dest.size());
    } //NOTE: subgraph CSR is declared fully on reordered vertices
    subgraph_offset.insert(subgraph_offset.begin(), 0); // offset array of csr always starts with 0
    if (debug) cout << endl;
    cout << "Subgraph CSR remapping (old_id=>new_id) completed!" << endl << endl;

    // Step 4: Validate memory usage
    if (calculateSubgraphMemory(offset, dest, reorder_map, selected_vertices, min_gpu_buffer) > max_gpu_memory) {
        throw runtime_error("Subgraph exceeds GPU memory constraints");
    }
}

/*  ------------------------------------------------------------------------------
                                    Main Function
    ------------------------------------------------------------------------------  */
// Function to print CSR format graph
void print_csr_graph(const vector<int>& offset, const vector<int>& dest, const vector<float>& weights) {
    int n = offset.size() - 1;
    for (int i = 0; i < n; ++i) {
        cout << "Vertex " << i << ":";
        for (int j = offset[i]; j < offset[i + 1]; ++j) {
            cout << " -> (" << dest[j] << ", w=" << weights[j] << ")";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    // Helper description for --help or -h
    // Parse debug flag
    bool read_weights = false;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "-d") {
            debug = true;
        }
        if (string(argv[i]) == "-w") {
            read_weights = true;
        }
    }

    if (argc >= 2 && (string(argv[1]) == "--help" || string(argv[1]) == "-h")) {
        cout << "Usage: " << argv[0] << " <input_file_path> [max_gpu_memory|-g<size>|-g <size>] [-d] [-w]" << endl;
        cout << "  <input_file_path>   : Path to the input graph file." << endl;
        cout << "  [max_gpu_memory]    : (Optional) Maximum GPU memory in bytes for subgraph extraction." << endl;
        cout << "  -g<size> or -g <size>: (Optional) Specify GPU memory in bytes (e.g., -g104857600 or -g 104857600)." << endl;
        cout << "  -d                  : Enable debug mode (prints reordering bfs iteration & subgraph csr remapping)." << endl;
        cout << "  -w                  : Read edge weights from the input file (defualt:false => all weights = 1, i.e. unweighted)." << endl;
        cout << "  --help, -h          : Show this help message." << endl;
        return 0;
    }

    // Step 1: Read input
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file_path>" << endl;
        return 1;
    }
    string file_name, abs_path, rel_path=argv[1];
    get_abs_path_and_file_name(rel_path, abs_path, file_name);
    cout << "Absolute path: " << abs_path << endl;
    cout << "File Name (without extension): " << file_name << endl << endl;

    vector<Edge> edges; int n, m;
    read_input(abs_path, edges, n, m, !read_weights);
    cout << "Input graph: " << n << " vertices, " << m << " edges." << endl << endl;

    // Step 2: Convert to CSR
    vector<int> offset, dest;
    vector<float> weights;  ofstream fout;
    convert_to_csr(edges, n, offset, dest, weights);
    cout << "Given Data's CSR Graph:" << endl;
    print_csr_graph(offset, dest, weights);// Print the CSR graph

    // Step 3: Write CSR to binary files
    write_bin_file("offset.bin", file_name, offset);
    write_bin_file("destination.bin", file_name, dest);
    write_bin_file("weights.bin", file_name, weights);

    // Step 4: Reorder the graph
    reorder_map.reserve(n); original_map.reserve(n);
    int tail = reorder_graph(offset, dest, n);

    // Step 5: Save reordered CSR
    // For simplicity, we'll just save the order, not the actual reordered CSR
    fout.open("1_GraphReordering.txt");
    fout << "Map vertices (new_idx --> old_idx): \n";
    for (int i = 0; i < original_map.size(); i++) {
        if (i == 0)
            fout << "Ordered vertices: \n";
        if (i == tail)
            fout << "\nSink verticies: \n";
        fout << "(" << i << ", " << original_map[i] << ") ";
    }
    fout << endl << endl;
    fout << "Map vertices (old_idx --> new_idx): \n";
    for (int i = 0; i < reorder_map.size(); i++) {
        fout << "(" << i << ", " << reorder_map[i] << ") ";
    }
    fout.close();
    write_bin_file("original_map.bin", file_name, original_map);
    write_bin_file("reorder_map.bin", file_name, reorder_map);
    cout << "Reordered vertices Output saved to \'1_GraphReordering.txt\' :)" << endl << endl;

    // Step 6: Extract subgraph
    size_t min_gpu_buffer = 0; // 1024B buffer for safety
    size_t max_gpu_memory = getAvailableGPUMemory(); // adjust based on GPU available memory
    cout << "Available GPU memory: " << max_gpu_memory << " Bytes" << endl;
    // Parse max_gpu_memory from command line if provided // For experimenting
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-d" || arg == "-w") continue; // skip other flags
        if (arg.rfind("-g", 0) == 0) { // starts with -g
            if (arg.length() > 2) {
                // Format: -g<size>
                max_gpu_memory = stoull(arg.substr(2));
            } else if (i + 1 < argc) {
                // Format: -g <size>
                max_gpu_memory = stoull(argv[i + 1]);
            }
            cout << "Max GPU memory (experimental): " << max_gpu_memory << " Bytes" << endl;
            break;
        } else {
            // If it's not -g or -d, treat as direct size argument
            max_gpu_memory = stoull(arg);
            cout << "Max GPU memory (experimental): " << max_gpu_memory << " Bytes" << endl;
            break;
        }
    }

    vector<int> subgraph_offset, subgraph_dest;
    vector<float> subgraph_weights;
    extract_subgraph(offset, dest, weights, n, max_gpu_memory, min_gpu_buffer, subgraph_offset, subgraph_dest, subgraph_weights);
    
    fout.open("2_SubgraphExtraction.txt");
    fout << "Subgraph CSR arrays:" << endl;
    fout << "Offset: ";
    for (size_t i = 0; i < subgraph_offset.size(); ++i) {
        fout << subgraph_offset[i] << " ";
    }
    fout << "\nDestination: ";
    for (size_t i = 0; i < subgraph_dest.size(); ++i) {
        fout << subgraph_dest[i] << " ";
    }
    fout << "\nWeights: ";
    for (size_t i = 0; i < subgraph_weights.size(); ++i) {
        fout << subgraph_weights[i] << " ";
    }
    fout << endl << endl;
    fout.close();

    // Step 7: Write subgraph to binary
    write_bin_file("subgraph_offset.bin", file_name, subgraph_offset);
    write_bin_file("subgraph_destination.bin", file_name, subgraph_dest);
    write_bin_file("subgraph_weights.bin", file_name, subgraph_weights);
    cout << "Size-constrained Subgraph Output saved to \'2_SubgraphExtraction.txt\' :)" << endl << endl;
    
    cout << "Subgraph CSR graph (wrt new_vertex_ids):" << endl;
    print_csr_graph(subgraph_offset, subgraph_dest, subgraph_weights); // Print the subgraph CSR
    cout << "PreProcessing completed successfully!" << endl;
    return 0;
}