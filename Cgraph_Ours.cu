#include <iostream>
#include <bits/stdc++.h>
// #define n 13
// #define m 23
using namespace std;
#include <vector>
#include <algorithm>
#include <queue>

using namespace std;

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
}/*
void createCSR(int N,int arr1[],int arr2[],int edgeCount,int* vertex,int* edges)
{
    int index;

    for (int i = 0; i < N+1; i++) {
    vertex[i] = 0;
    }

    for (int i = 0; i < N; i++){
        vertex[arr1[i]+1]++;

    } 

    //PREFIX SUM BELOW

    for (int i = 1; i < N+1; i++) {
    vertex[i] += vertex[i - 1];
    }

    for (int i = 0; i < edgeCount; i++) {
    edges[i] = -1;
    }

    for(int i=0;i<N;i++)
    {
        index= vertex[arr1[i]];
        while(edges[index]!=-1)
        {
            index++;
        }
        edges[index]=i;
    }
}*/

void bfs(
    int num_vertices,
    const int vertex[],       // size = num_vertices + 1
    const int edges[],     // size = num_edges
    int source,            // start vertex
    int dist[]             // output array of size num_vertices, unvisited = -1
)
{
    std::queue<int> q;

    // Initialize distances
    for (int i = 0; i < num_vertices; ++i)
        dist[i] = -1;

    // Start BFS
    dist[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int idx = vertex[u]; idx < vertex[u + 1]; ++idx) {
            int v = edges[idx];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
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

        inputFile >> u >> v >> t >> x;
        //cout << u << "  " << v <<endl;
        arr1[i] = u;
        arr2[i] = v;
    }

    inputFile.close();

    int* vertex = new int[n + 1];
    int* edges = new int[m];

    createCSR(n, arr1, arr2, m, vertex, edges);


    // Print the CSR representation
   /* std::cout << "CSR Representation:" << std::endl;

    std::cout << "Vertex array:" << std::endl;
    for (int i = 0; i < n + 1; i++) {
        std::cout << "vertex[" << i << "] = " << vertex[i] << std::endl;
    }

    std::cout << "Edges array:" << std::endl;
    for (int i = 0; i < m; i++) {
        std::cout << "edges[" << i << "] = " << edges[i] << std::endl;
    }*/

    // // Example initialization (optional)
    // int vertex[n]={0,5,8,8,9,9,10,14,15,18,19,21,23};
    // int edges[m]={1,5,8,3,7,9,5,10,6,0,0,4,8,5,5,2,4,11,5,2,8,5,2};
    // int weight[m]={2,4,9,1,8,1,2,5,2,4,3,5,2,4,3,1,2,4,1,6,1};

    //bfs(vertex, edges, 1); // Call BFS with the first vertex as source
    //int dist[n]; // Array to store distances from the source vertex

    // Call BFS with the first vertex (index 0) as the source
    //bfs(n, ver, edges, 5, dist);
    bool flag=1;
    int* count = new int[n](); // Dynamically allocate and initialize all elements to 0
    std::vector<int> sink(n, 0); // Initialize all elements to 0
    sink.clear(); // Clear the vector to ensure it's empty before pushing back
    for (int i = 0; i < n - 1; i++) {
        if (vertex[i] == vertex[i + 1]) {
            sink.push_back(i);
        }
    }
    // Print the sink array
    std::cout << "Sink vertices:" << std::endl;
    for (size_t i = 0; i < sink.size(); i++) {
        std::cout << "sink[" << i << "] = " << sink[i] << std::endl;
    }
    for (int i = 0; i < m; i++) {
        count[edges[i]]++;
    }


    cout<<"Count of incoming edges"<<endl;

        // Print the array count
        /*for (int i = 0; i < n; i++) {
            std::cout << "count[" << i << "] = " << count[i] << std::endl;
        }*/
        std::vector<std::pair<int, int>> countPairs;

        for (int i = 0; i < n; i++) {
            countPairs.push_back(std::make_pair(count[i], i));
        }
        int* dist = new int[n]; // Array to store distances from the source vertex

        int* is_sink = new int[n](); // Dynamically allocate and initialize all elements to 0
        for (size_t i = 0; i < sink.size(); i++) {
            is_sink[sink[i]] = 1;
        }

            // Print the is_sink array
    /*std::cout << "is_sink array:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "is_sink[" << i << "] = " << is_sink[i] << std::endl;
    }*/
    


while(flag==1){
    flag=0;
   




    // Print the distances
    // std::cout << "Distances from source vertex 0:" << std::endl;
    // for (int i = 0; i < n; i++) {
    //     std::cout << "Vertex " << i << ": " << dist[i] << std::endl;
    // }
    //     int sum = 0;
    //     for (int i = 0; i < n; i++) {
    //         sum += count[i];
    //     }

    // std::cout << "Summation of count array: " << sum << std::endl;

        
    // Create a vector of pairs to store count values along with their indices



    // Sort the pairs based on the first value (count) in descending order
    std::sort(countPairs.begin(), countPairs.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first;
    });

    // Print the sorted pairs
    std::cout << "Sorted count array with indices:" << std::endl;
    for (const auto& p : countPairs) {
        std::cout << "count: " << p.first << ", index: " << p.second << std::endl;
    }

    n = countPairs.size();

    int v = -1;
    for (size_t i = 0; i < countPairs.size(); ++i) {
        if (is_sink[countPairs[i].second] == 0) { // Check if it's not a sink
            v = countPairs[i].second; // Get the index with the highest count that's not a sink
            break;
        }
    }
    if (v == -1) {
        std::cerr << "Error: No valid vertex found to start BFS." << std::endl;
        break; // Exit the loop if no valid vertex is found
    }
   // std::cout << "Starting BFS from vertex with highest count: " << v << std::endl;

    bfs(n, vertex, edges, v, dist); // Call BFS with the vertex having the highest count

    // Print the distances
    std::cout << "Distances from source vertex " << v << ":" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << "Vertex " << i << ": " << dist[i] << std::endl;
    }




    // Find the highest number in the dist array
    int max_dist = -1;
    for (int i = 0; i < n; i++) {
        if (dist[i] > max_dist) {
            max_dist = dist[i];
        }
    }

    std::cout << "The highest number in the dist array is: " << max_dist << std::endl;

    std::vector<int> reordered_array; // Dynamic size array for non-sink vertices
    for (int i = 0; i < max_dist+10; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[j] == i) {
                if (is_sink[j] == 1) {
                    continue; // Skip sink vertices
                }
                reordered_array.push_back(j);

                //printf("Pushed %d to reordered_array\n", j);
                
                count[j] = -1; // Mark as visited
            } else {
                continue;
            }
        }
    }

    // Print the reordered array
    std::cout << "Reordered array:" << std::endl;
    for (size_t i = 0; i < reordered_array.size(); i++) {
        std::cout << "reordered_array[" << i << "] = " << reordered_array[i] << std::endl;
    }

    // Remove pairs with count -1 from sorted pairs
    // Update countPairs to reflect changes in count
    for (auto& p : countPairs) {
        p.first = count[p.second];
    }

    std::vector<std::pair<int, int>> filteredPairs;
    for (const auto& p : countPairs) {
        if (p.first != -1) {
            filteredPairs.push_back(p);
            flag=1;
        }
    }
    flag=0;
    countPairs=filteredPairs;

    // Print the filtered pairs
    /*std::cout << "Filtered count array with indices (excluding -1):" << std::endl;
    for (const auto& p : countPairs) {
        std::cout << "count: " << p.first << ", index: " << p.second << std::endl;
    }*/

   // delete[] dist;
    
printf("Done\n");
}

    /*bfs(n, vertex, edges, countPairs[0].second, dist); // Call BFS with the vertex having the highest count

    // Print the distances
    std::cout << "Distances from source vertex " << countPairs[0].second << ":" << std::endl;
    for (int i = 0; i < n; i++) {
        if(count[i] == -1){
            continue;
        }
        else
        std::cout << "Vertex " << i << ": " << dist[i] << std::endl;
    }*/

    // // Print the count array
    // std::cout << "Count array:" << std::endl;
    // for (int i = 0; i < n; i++) {
    //     std::cout << "count[" << i << "] = " << count[i] << std::endl;
    // }





     // Clean up dynamically allocated memory


/*
    // // Resize the count array to reflect the reduced size
    // int* resizedCount = new int[m];
    // for (int i = 0; i < m; i++) {
    //     resizedCount[i] = count[i];
    // }

    // // Replace the old count array with the resized one
    // delete[] count;
    // count = resizedCount;
/*
    std::vector<std::pair<int, int>> countPairs;

    for (int i = 0; i < n; i++) {
        countPairs.push_back(std::make_pair(count[i], i));
    }

    // Sort the pairs based on the first value in reverse order
    std::sort(countPairs.begin(), countPairs.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first;
    });

    cout<<endl<<"After sorting"<<endl;

    // Print the sorted pairs
    for (const auto& p : countPairs) {
        std::cout << "count: " << p.first << ", index: " << p.second << std::endl;
    }

    // Print the array count
    // for (int i = 0; i < n; i++) {
    //     std::cout << "count[" << i << "] = " << count[i] << std::endl;
    // }

    cout<<endl<<"Sink vertices"<<endl;

    for (size_t i = 0; i < sink.size(); i++) {
        std::cout << "sink[" << i << "] = " << sink[i] << std::endl;
    }

    std::vector<int> check_sink(n, 1); // Initialize all elements to 1
    // Iterate through the sink array and update check_sink

    cout << endl<<"Sink array" << endl;
    for (size_t i = 0; i < sink.size(); i++) {
        check_sink[sink[i]] = 0;
    }

    // Print the check_sink array
    for (size_t i = 0; i < check_sink.size(); i++) {
        std::cout << "check_sink[" << i << "] = " << check_sink[i] << std::endl;
    }

    //bfs(n, vertex, edges, 4,dist); // Call BFS with the first vertex as source

    //Print the distances


        int v = countPairs[0].second;cout<<"v= "<<v <<endl; // Access the second element of the first pair
        bfs(n, vertex, edges, v, dist); // Call BFS with the first vertex as source

        std::cout << endl<< "Distances from source vertex :" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "Vertex " << i << ": " << dist[i] << std::endl;
        }


    std::vector<int> non_sink;

        for(int k=0;k<n;k++){
    for (int i = 0; i < n; i++) {
        if (dist[i] == k) {
            if(check_sink[i] == 0){
                continue;
            }
            else
            non_sink.push_back(i);
        }
    }}

    // Print the non_sink array
    std::cout << endl << "Non-sink vertices: Renamed" << std::endl << endl;
    for (size_t i = 0; i < non_sink.size(); i++) {
        std::cout << "non_sink[" << i << "] = " << non_sink[i] << std::endl;
    }

    // Print the sink vertices
    std::cout << std::endl <<endl << "Sink vertices:" << std::endl<<endl;
    for (size_t i = 0; i < sink.size(); i++) {
        std::cout << "sink[" << i << "] = " << sink[i] << std::endl;
    }





    // Placeholder for further implementation
   /// std::cout << "Arrays initialized." << std::endl;
*/
    return 0;
}