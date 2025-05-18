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
    auto start = chrono::high_resolution_clock::now();
    bfs(vertex, edges, N, source, level);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;
    cout << "BFS execution time: " << elapsed.count() << " seconds" << endl;
// 1) Read mapping.txt
std::string baseName = infile.substr(infile.find_last_of("/\\") + 1);
std::string mappingFilePath = "Out_" + baseName + "/mapping.txt";
std::ifstream mapFile(mappingFilePath);
if (!mapFile) {
    std::cerr << "Error: cannot open mapping.txt\n";
    return 1;
}

// First pass: find out how many entries (max new_id + 1)
int old_id, new_id;
char ch;
int max_new = -1;
std::vector<std::pair<int,int>> entries;
while (mapFile >> ch    // '('
       >> old_id
       >> ch            // ','
       >> new_id
       >> ch) {         // ')'
    entries.emplace_back(new_id, old_id);
    if (new_id > max_new) max_new = new_id;
}
mapFile.close();

// 2) Build new_levels, initialize to -1
std::vector<int> new_levels(max_new + 1, -1);
for (auto &pr : entries) {
    int nid = pr.first;
    int oid = pr.second;
    if (oid >= 0 && oid < (int)level.size())
        new_levels[nid] = level[oid];
}

// 3) Print as CSV: "3, 1, 2, -1, ..."
for (size_t i = 0; i < new_levels.size(); ++i) {
    std::cout << new_levels[i];
    if (i + 1 < new_levels.size())
        std::cout << ", ";
}
std::cout << "\n";
// Write the output to a file named "serial_out_<infile>"
std::string outputFileName = "serial_out_" + infile.substr(infile.find_last_of("/\\") + 1);
std::ofstream outFile(outputFileName);
if (!outFile) {
    std::cerr << "Error: cannot open output file\n";
    return 1;
}

for (size_t i = 0; i < new_levels.size(); ++i) {
    outFile << new_levels[i];
    if (i + 1 < new_levels.size())
        outFile << "\n";
}
outFile << "\n";
outFile.close();
cout << "BFS execution time: " << elapsed.count() * 1000 << " milliseconds" << endl;

    

    return 0;
}
