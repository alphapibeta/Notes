#include <cmath>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

__global__ void computeDistancesKernel(float* x_coords, float* y_coords, float* distances, int num_points) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pairs = (static_cast<unsigned long long>(num_points) * (num_points - 1)) / 2;
    
    if (idx < num_pairs) {
        int p1 = 0, p2 = 1;
        unsigned long long sum = 0;
        for (int i = 0; i < num_points - 1; ++i) {
            sum += num_points - i - 1;
            if (idx < sum) {
                p1 = i;
                p2 = num_points - (sum - idx);
                break;
            }
        }
        
        float dx = x_coords[p1] - x_coords[p2];
        float dy = y_coords[p1] - y_coords[p2];
        distances[idx] = sqrtf(dx * dx + dy * dy);
    }
}

__global__ void reduceMaxKernel(float* distances, float* max_distance, int num_pairs) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned long long i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = 0.0f;
    
    if (i < num_pairs) {
        sdata[tid] = distances[i];
    }
    if (i + blockDim.x < num_pairs) {
        sdata[tid] = fmaxf(sdata[tid], distances[i + blockDim.x]);
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_distance, __float_as_int(sdata[0]));
    }
}

void launchComputeDistances(float* d_x_coords, float* d_y_coords, float* d_distances, int num_points, int threads_x, int threads_y) {
    unsigned long long num_pairs = (static_cast<unsigned long long>(num_points) * (num_points - 1)) / 2;
    int threads = threads_x * threads_y;
    int blocks = (num_pairs + threads - 1) / threads;

    nvtxRangePush("Compute Distances Kernel");
    computeDistancesKernel<<<blocks, threads>>>(d_x_coords, d_y_coords, d_distances, num_points);
    nvtxRangePop();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in computeDistancesKernel: " << cudaGetErrorString(err) << std::endl;
    }
}

void launchReduceMax(float* d_distances, float* d_max_distance, int num_pairs, int threads_x, int threads_y) {
    int threads = threads_x * threads_y;
    int blocks = (num_pairs + (threads * 2) - 1) / (threads * 2);

    nvtxRangePush("Reduce Max Kernel");
    reduceMaxKernel<<<blocks, threads, threads * sizeof(float)>>>(d_distances, d_max_distance, num_pairs);
    nvtxRangePop();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in reduceMaxKernel: " << cudaGetErrorString(err) << std::endl;
    }
}