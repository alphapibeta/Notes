#include "Matmul.h"
#include <vector>
#include <cassert>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(const int *__restrict a, const int *__restrict b, int *__restrict c, int N) {
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < N) c[tid] = a[tid] + b[tid];
}

void vectorAdd(const int* a, const int* b, int* c, int N) {
    int NUM_THREADS = 1024;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
    vectorAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(a, b, c, N);
}

void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}