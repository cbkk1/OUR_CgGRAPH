#include <iostream>
#include <bits/stdc++.h>
// #define n 13
// #define m 23
using namespace std;
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

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
        vertex[arr1[i] + 1]++;
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
    int n=12;
    int m=23;

    #include <fstream>
    #include <sstream>

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open file " << argv[1] << std::endl;
        return 1;
    }

    inputFile >> n >> m;
    int* arr1 = new int[m];
    int* arr2 = new int[m];




    for (int i = 0; i < m; ++i) {
        int u, v, t, x;

        inputFile >> u >> v;
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

    



    return 0;
}