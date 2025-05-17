#include <bits/stdc++.h>
using namespace std;

// Build CSR using an auxiliary count+offset array so we don't overwrite prefix sums.
void createCSR(int N,
               const vector<int>& arr1,
               const vector<int>& arr2,
               vector<int>& vertex,
               vector<int>& edges)
{
    vector<int> count(N, 0);
    // 1) Count out‑degree for each u
    for (int u : arr1) {
        if (u < 0 || u >= N) {
            cerr << "Error: out‑of‑bounds vertex " << u << endl;
            exit(1);
        }
        count[u]++;
    }

    // 2) Prefix sum into vertex[]
    vertex[0] = 0;
    for (int i = 1; i <= N; i++)
        vertex[i] = vertex[i-1] + count[i-1];

    // 3) Fill edges[] by bumping an offset array
    vector<int> offset(vertex.begin(), vertex.begin() + N);
    for (size_t i = 0; i < arr1.size(); i++) {
        int u = arr1[i], v = arr2[i];
        edges[offset[u]++] = v;
    }
}

// Standard CPU BFS, starting from `source`
void bfs(const vector<int>& vertex,
         const vector<int>& edges,
         int N,
         int source,
         vector<int>& level)
{
    level.assign(N, -1);
    queue<int> q;

    level[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int e = vertex[u]; e < vertex[u+1]; e++) {
            int v = edges[e];
            if (level[v] == -1) {
                level[v] = level[u] + 1;
                q.push(v);
            }
        }
    }

    // print
    cout << "Vertex levels:\n";
    int highest = 0;
    for (int i = 0; i < N; i++) {
        cout << "  " << i << " → " << level[i] << "\n";
        highest = max(highest, level[i]);
    }
    cout << "Highest level = " << highest << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <infile> <source_vertex>\n";
        return 1;
    }

    string infile = argv[1];
    int source  = stoi(argv[2]);

    ifstream in(infile);
    if (!in) {
        cerr << "Error: cannot open " << infile << endl;
        return 1;
    }

    int N, M;
    in >> N >> M;

    vector<int> arr1(M), arr2(M);
    for (int i = 0; i < M; i++) {
        int u, v, t, x;
        in >> u >> v >> t >> x;
        arr1[i] = u;
        arr2[i] = v;
    }
    in.close();

    if (source < 0 || source >= N) {
        cerr << "Error: source_vertex out of range [0," << N-1 << "]\n";
        return 1;
    }

    // allocate CSR arrays
    vector<int> vertex(N+1), edges(M);
    createCSR(N, arr1, arr2, vertex, edges);

    // run BFS
    vector<int> level;
    bfs(vertex, edges, N, source, level);

    return 0;
}
