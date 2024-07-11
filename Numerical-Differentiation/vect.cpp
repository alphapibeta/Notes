#include "Matmul.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>


using namespace std;


int main() {
     int N = 1 << 16;
    size_t bytes = sizeof(int) * N;
    // vector<int> * vp = new vector<int>;
    // cout<< sizeof(vp) <<  endl;
    // vp->push_back(10);
    // cout<<(*vp)<<endl;
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);

    

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    vectorAdd(d_a, d_b, d_c, N);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // verify_result(a, b, c);

    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);

    // std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;
}