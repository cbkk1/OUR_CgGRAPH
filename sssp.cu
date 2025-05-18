// sssp.cu
// CPU-GPU cooperative Single-Source Shortest Paths (SSSP)
// Builds on preprocess.cu: loads CSR and subgraph binaries, and reorder maps

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <string>
#include <cuda_runtime.h>

#include "utils.h"  // file handling utility functions
using namespace std;
#define INF numeric_limits<float>::infinity()
#define BLOCK_SIZE 128

// Atomic min for float (CUDA doesn't provide atomicMin for float on all archs)
__device__ float atomicMinFloat(float* addr, float value) {
    float old = *addr, assumed;
    int attempts = 0;
    while (value < old) {
        ++attempts;
        assumed = old;
        old = atomicCAS((int*)addr, __float_as_int(assumed), __float_as_int(value));
        if (__int_as_float(old) == assumed) break;
    }
    printf("atomicMinFloat attempts: %d\n", attempts);
    return old;
}

// Mapping arrays
static vector<int> original_map;  // new_idx -> old_idx
static vector<int> reorder_map;   // old_idx -> new_idx

// Host-side full-graph CSR
static vector<int> h_offset, h_dest;
static vector<float> h_weight;

// Host-side subgraph CSR (G')
static vector<int> h_sub_offset, h_sub_dest;
static vector<float> h_sub_weight;

// Distance arrays on CPU (for full graph)
static vector<float> h_dist, h_next_dist;
static vector<bool> h_mask, h_next_mask; // visited array

// Device-side subgraph arrays
static int *d_sub_offset = nullptr;
static int *d_sub_dest   = nullptr;
static float *d_sub_weight = nullptr;
static bool *d_sub_mask = nullptr;
static float *d_sub_dist  = nullptr;
static float *d_sub_out   = nullptr;

//-----------------------------------------------------------------------------
//     SSSP Kernels based on Harish and PJ Narayan Paper (skip parent path)
// Inputs:
//   sub_n      = number of vertices in subgraph
//   sub_offset = CSR offset array of G'
//   sub_dest   = CSR destination array of G'
//   sub_weight = CSR weights array of G'
//   d_mask     = mask array to track relaxed vertices
//   d_in       = current distance vector (size sub_n)
// Outputs:
//   d_out      = updated distance vector after one relaxation pass
//-----------------------------------------------------------------------------
__global__ void scatter_relax_kernel(int sub_n,
    const int *sub_offset, const int *sub_dest, const float *sub_weight,
    bool *d_mask, const float *d_in, float *d_out)
{
    int u = blockIdx.x*blockDim.x + threadIdx.x;
    if (u >= sub_n) return;

    float du = d_in[u];
    if (!d_mask[u] || du == INFINITY) return; // skip not masked & unreachable
    d_mask[u] = false; // reset mask for this vertex
    printf("GPU vertex %d: degree = %d\n", u, sub_offset[u+1] - sub_offset[u]);
    
    int total_attempts = 0;
    for (int e = sub_offset[u]; e < sub_offset[u+1]; ++e) {
        int v    = sub_dest[e];
        float w  = sub_weight[e];
        float cand = du + w;

        // atomicMin on floats: CUDA has intrinsic __int_as_float etc.
        float old = atomicMinFloat(&d_out[v], cand);
        if (cand < old) {
            d_mask[v] = true;
        }
        // Count attempts for this edge
        total_attempts += 1; // Each atomicMinFloat call is one attempt per edge
        printf("GPU vertex %d: total edge relax attempts = %d\n", u, total_attempts);
    }
}
__global__ void update_iteration_kernel(int sub_n,
    const int *sub_offset, const int *sub_dest, const float *sub_weight,
    bool *d_mask, float *d_in, float *d_out)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= sub_n) return;

    if (d_out[tid] < d_in[tid]){
        d_in[tid] = d_out[tid];
        d_mask[tid] = true;
    }
    // reset d_in and d_out to be same
    d_out[tid] = d_in[tid];
}



//-----------------------------------------------------------------------------
// Print distances neatly per iteration
//-----------------------------------------------------------------------------
void print_iteration(int iter, const vector<float> &cpu_dist, int n, const vector<float> &gpu_dist, int sub_n, const vector<bool> &mask) {
    cout << "Iteration " << iter << ":\n  vertex:";
    for (int i = 0; i < n; ++i) cout << " \t" << i;
    cout << "\n  |--CPU:";
    for (float d : cpu_dist) {
        if (d == INF) 
            cout << " \tInf";
        else 
            cout << " \t" << d;
    }
    cout << "\n  |--GPU:";
    for (int i = 0; i < n; ++i) {
        if (i < sub_n) {
            if (gpu_dist[i] == INF) 
                cout << " \tInf";
            else 
                cout << " \t" << gpu_dist[i];
        } else {
            cout << " \t-";
        }
    }
    cout << "\n  |-mask:"; //print mask for debugging
    for (bool d : mask) 
        cout << " \t" << (d ? "T" : "F");
    cout << endl << endl;
}

int main(int argc, char** argv) {
    
    // Step 1: Read input
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file_path>" << endl;
        return 1;
    }
    string folder_name, abs_path, rel_path=argv[1];
    get_abs_path_and_file_name(rel_path, abs_path, folder_name);
    cout << "Absolute path: " << abs_path << endl;
    cout << "Folder Name: " << folder_name << endl << endl;

    // Step 2: Load all data
    read_bin_file("original_map.bin", folder_name, original_map); // Load original_map (new->old)
    read_bin_file("reorder_map.bin", folder_name, reorder_map); // Load reorder map (old->new)
    
    // Load full-graph CSR
    read_bin_file("offset.bin", folder_name, h_offset);
    read_bin_file("destination.bin", folder_name, h_dest);
    read_bin_file("weights.bin", folder_name, h_weight);    

    // Load subgraph CSR
    read_bin_file("subgraph_offset.bin", folder_name, h_sub_offset);
    read_bin_file("subgraph_destination.bin", folder_name, h_sub_dest);
    read_bin_file("subgraph_weights.bin", folder_name, h_sub_weight);

    int n = h_offset.size() - 1, m = h_dest.size(); // number of vertices & edges in full graph
    int sub_n = h_sub_offset.size() - 1, sub_m = h_sub_dest.size(); // number of vertices & edges in subgraph

    // Step 3: Initialize distances for host-side and device-side
    h_dist.assign(n, INF);
    h_mask.assign(n, false);
    
    // Parse source vertex from command line (old_id)
    int src_old, src_new;
    if (argc >= 3) {
        src_old = stoi(argv[2]);
        if (src_old < 0 || src_old >= m) {
            cerr << "Invalid source vertex id (old_id): " << src_old << endl;
            cerr << "Valid range: [0, " << m-1 << "]\n";
            return 1;
        }
        src_new = reorder_map[src_old];
    } else {
        cout << "No source vertex specified, defaulting to 0 (new_id)\n";
        src_new = 0;
        src_old = original_map[src_new];
    }
    cout << "Source vertex (old_id, new_id): (" << src_old << ", " << src_new << ")\n";
    h_dist[src_new] = 0.0f; // Do all computation wrt new_ids and finally map to old_ids
    h_mask[src_new] = true; // Set source vertex as visited
    h_next_dist = h_dist; h_next_mask = h_mask; // copy next arrays same as current

    // Allocate and copy subgraph to GPU
    cudaMalloc(&d_sub_offset, sizeof(int)*(sub_n+1));
    cudaMalloc(&d_sub_dest,   sizeof(int)*sub_m);
    cudaMalloc(&d_sub_weight, sizeof(float)*sub_m);
    cudaMemcpy(d_sub_offset, h_sub_offset.data(), sizeof(int)*(sub_n+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sub_dest,   h_sub_dest.data(),   sizeof(int)*sub_m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sub_weight, h_sub_weight.data(), sizeof(float)*sub_m, cudaMemcpyHostToDevice);
    cudaMalloc(&d_sub_mask, sizeof(bool)*sub_n); // subgraph mask/frontier array
    cudaMalloc(&d_sub_dist, sizeof(float)*sub_n); // subgraph dist input
    cudaMalloc(&d_sub_out,  sizeof(float)*sub_n); // subgraph dist output

    // Copy initial subgraph distances
    vector<float> temp_cost(sub_n); // to hold distances for subgraph vertices computed by GPU inbetween kernel calls
    for (int i = 0; i < sub_n; ++i) temp_cost[i] = h_dist[i];
    cudaMemcpy(d_sub_dist, temp_cost.data(), sizeof(float)*sub_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sub_out,  temp_cost.data(), sizeof(float)*sub_n, cudaMemcpyHostToDevice);
    // Similarly, copy mask / visited array
    vector<bool> temp_mask(sub_n, false);
    for (int i = 0; i < sub_n; ++i) temp_mask[i] = h_mask[i];
    cudaMemcpy(d_sub_mask, &temp_mask[0], sizeof(bool)*sub_n, cudaMemcpyHostToDevice); // (different template this is not guaranteed to work)

    // Step 4: Apply SSSP algorithm {G'=>GPU, G\G'->CPU}
    // Print initial state
    cout << "=== SSSP iterations (CPU-GPU cooperative) ===\n";
    print_iteration(0, h_dist, n, temp_cost, sub_n, h_mask);
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (sub_n + threadsPerBlock - 1) / threadsPerBlock;

    // Main iteration loop
    bool converged = false;
    
    int iter = 0;
    while (!converged && iter < n) {
        ++iter;

        // -----------------------
        // 1) GPU: 2 Parts
        //    - relax all edges in G' (subgraph)
        //   => compare results of common subgraph vertices 
        //    & take min in the end after CPU's computation
        // ------------------------
        scatter_relax_kernel<<<blocksPerGrid, threadsPerBlock>>>(sub_n, d_sub_offset, d_sub_dest, d_sub_weight, d_sub_mask, d_sub_dist, d_sub_out);
        // cudaDeviceSynchronize(); // put in last to utilize parallelism
        // cout << "Relaxed subgraph edges in GPU kernel.\n";

        // ----------------------
        // 2) CPU: 2 Parts (need to multithread this parallelly without waiting)
        //    - relax all edges in the residual graph G - G′
        //   => Residual edges are those where (u → v) is NOT fully inside G′,
        //      i.e. NOT (u < sub_n AND v < sub_n).
        // ----------------------

        // First: all edges whose source is outside G′ (u >= sub_n)
        //   These vertices u are CPU-only, so relax ALL of their edges.
        for (int u = sub_n; u < n; ++u) {
            if (!h_mask[u]) continue; // skip vertices not in the frontier
            h_next_mask[u] = false;
            int u_old = original_map[u];        // map reordered index → original index
            float du  = h_dist[u];              // current distance of u

            // Walk through all of u_old's outgoing edges in the full CSR
            for (int e = h_offset[u_old]; e < h_offset[u_old+1]; ++e) {
                int v_old = h_dest[e];           // original neighbor
                int v     = reorder_map[v_old];  // neighbor in reordered space
                cout << "Relaxing edges from G - G'("<< u <<") to G("<< v <<") for vertex " << endl;
                float cand = du + h_weight[e];
                if (cand < h_next_dist[v]) {
                    cout << "===> relaxing successful!\n";
                    h_next_dist[v] = cand;
                    h_next_mask[v] = true;
                }
            }
        }

        // Second: edges from G′ into the residual graph
        //   These are edges whose source u is in G′ (u < sub_n) but whose
        //   destination v lies outside G′ (v >= sub_n).
        for (int u = 0; u < sub_n; ++u) {
            if (!h_mask[u]) continue; // skip vertices not in the frontier
            h_next_mask[u] = false;
            int u_old = original_map[u]; // new_id → old_id
            float du  = h_dist[u];

            for (int e = h_offset[u_old]; e < h_offset[u_old+1]; ++e) {
                int v_old = h_dest[e];
                int v     = reorder_map[v_old]; // new_id to old_id
                cout << "Relaxing edges from G'("<< u <<") to G - G'("<< v <<") for vertex " << endl;

                // If v is also in G′, GPU already handled (u→v) in its kernel:
                if (v < sub_n) continue;

                // Otherwise this edge is in the residual graph → relax it on CPU
                float cand = du + h_weight[e];
                if (cand < h_next_dist[v]) {
                    cout << "===> relaxing successful!\n";
                    h_next_dist[v] = cand;
                    h_next_mask[v] = true;
                }
            }
        }
        cout << "Relaxed residual graph edges in CPU.\n";
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cerr << "CUDA error after scatter_relax_kernel: " << cudaGetErrorString(err) << endl;
            exit(1);
        }
        cout << "Relaxed subgraph edges in GPU kernel.\n";

        // ----------------------
        // need to apply wait call to synchronize CPU and GPU
        // 3) Aggregate and prepare next iteration
        // ----------------------
        
        // Copy back GPU results & compare
        cudaMemcpy(temp_cost.data(), d_sub_out, sizeof(float)*sub_n, cudaMemcpyDeviceToHost);
        h_dist = h_next_dist; // Update next distances for subgraph vertices
        for (int u = 0; u < sub_n; ++u) {
            float cand = temp_cost[u];
            if (cand < h_dist[u]) {
                h_dist[u] = cand;
                h_next_mask[u] = true; // mark to be visited again as it was relaxed
            }
        }
        h_next_dist = h_dist; 
        print_iteration(iter, h_next_dist, n, temp_cost, sub_n, h_next_mask); // just to see mask for CPU & GPU aggregate
        h_mask = h_next_mask;

        // Sync updated subgraph distances back to GPU
        for (int i = 0; i < sub_n; ++i) temp_cost[i] = h_dist[i];
        cudaMemcpy(d_sub_out, temp_cost.data(), sizeof(float)*sub_n, cudaMemcpyHostToDevice);
        for (int i = 0; i < sub_n; ++i) temp_mask[i] = h_mask[i];
        cudaMemcpy(d_sub_mask, &temp_mask[0], sizeof(bool)*sub_n, cudaMemcpyHostToDevice);
        update_iteration_kernel<<<blocksPerGrid, threadsPerBlock>>>(sub_n, d_sub_offset, d_sub_dest, d_sub_weight, d_sub_mask, d_sub_dist, d_sub_out);
        cudaDeviceSynchronize();

        /// ------------------------
        // cudaMemcpy(&temp_mask[0], d_sub_mask, sizeof(bool)*sub_n, cudaMemcpyDeviceToHost);
        // cudaMemcpy(temp_cost.data(), d_sub_dist, sizeof(float)*sub_n, cudaMemcpyDeviceToHost);

        // cout << "d_sub_mask: ";
        // for (int i = 0; i < sub_n; ++i) {
        //     cout << (temp_mask[i] ? "T" : "F") << " ";
        // }
        // cout << endl;

        // cout << "d_sub_dist: ";
        // for (int i = 0; i < sub_n; ++i) {
        //     if (temp_cost[i] == INF)
        //         cout << "Inf ";
        //     else
        //         cout << temp_cost[i] << " ";
        // }
        // cout << endl;
        /// ------------------------

        // Check if all elements of h_mask are false (i.e., no active frontier)
        converged = true;
        for (int i = 0; i < n; ++i) {
            if (h_mask[i]) {
                converged = false;
                break;
            }
        }
    }
    cout << endl;
    cout << "\nConverged in " << iter << " iterations.\n";

    // Print final distances
    cout << "Final distances:\n";
    for (int i = 0; i < n; ++i) {
        cout << "v" << i << ": ";
        if (h_dist[i] == INF) cout << "Inf";
        else cout << h_dist[i];
        cout << (i+1<n ? ", " : "\n");
    }

    // Cleanup
    cudaFree(d_sub_offset);
    cudaFree(d_sub_dest);
    cudaFree(d_sub_weight);
    cudaFree(d_sub_dist);
    cudaFree(d_sub_out);
    return 0;
}
