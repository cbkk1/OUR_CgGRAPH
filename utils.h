#include <sys/stat.h>
#ifdef _WIN32
    #include <direct.h>
#endif
#include <filesystem>

using namespace std;
namespace fs = std::filesystem; // requires C++17 stdlib, thus use -std=c++17 flag
#define MAX_PATH 1024

//-----------------------------------------------------------------------------
// Helper: store vector<T> into binary file
//-----------------------------------------------------------------------------
template<typename T> 
void write_bin_file(const string& filename, const string& folder, const vector<T>& data) {
    fs::create_directories(folder); // Create output directory if it doesn't exist
    fs::path full_path = fs::path(folder) / filename;
    ofstream fout(full_path, ios::binary);
    if (!fout) {
        cerr << "Error writing binary file: " << full_path << endl;
        exit(1);
    }
    fout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

//-----------------------------------------------------------------------------
// Helper: load binary file into vector<T>
//-----------------------------------------------------------------------------
template<typename T>
void read_bin_file(const string& filename, const string& folder, vector<T>& data) {
    fs::path full_path = fs::path(folder) / filename;
    if (!fs::exists(full_path)) {
        throw runtime_error("Missing file: " + full_path.string());
    }
    ifstream fin(full_path, ios::binary);
    if (!fin) {
        cerr << "Error reading binary file: " << full_path << endl;
        exit(1);
    }
    fin.seekg(0, ios::end);
    size_t bytes = fin.tellg();
    fin.seekg(0, ios::beg);
    size_t count = bytes / sizeof(T);
    data.resize(count);
    fin.read(reinterpret_cast<char*>(data.data()), bytes);
}

// Utility function to get absolute path and file name
void get_abs_path_and_file_name(const string& rel_path, string& abs_path, string& file_name) {
    fs::path p = fs::absolute(rel_path);
    abs_path = p.string();
    file_name = p.stem().string(); // stem() gives filename without extension
}
