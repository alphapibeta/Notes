#include "HeatEquationSolver.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

__global__ void gpu_kernel(float *grid, float *new_grid, int N, float alpha, float dt, float dx2, float dy2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1) {
        int index = i * N + j;
        float temp = grid[index];
        float temp_left = grid[i * N + (j - 1)];
        float temp_right = grid[i * N + (j + 1)];
        float temp_top = grid[(i - 1) * N + j];
        float temp_bottom = grid[(i + 1) * N + j];
        new_grid[index] = temp + alpha * dt * ((temp_left - 2.0f * temp + temp_right) / dx2 +
                                               (temp_top - 2.0f * temp + temp_bottom) / dy2);
    } else {
        int index = i * N + j;
        new_grid[index] = grid[index];
    }
}

void gpu_heat_equation_solver(int N, int num_steps, float *grid, float alpha, float dt, float dx2, float dy2, int block_thread_x, int block_thread_y) {
    float *d_grid, *d_new_grid;
    cudaError_t err;

    err = cudaMalloc(&d_grid, N * N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (d_grid allocation): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_new_grid, N * N * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (d_new_grid allocation): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_grid);
        return;
    }

    err = cudaMemcpy(d_grid, grid, N * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (memcpy to device): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_grid);
        cudaFree(d_new_grid);
        return;
    }

    dim3 threads_per_block(block_thread_x, block_thread_y);
    dim3 num_blocks((N + block_thread_x - 1) / block_thread_x, (N + block_thread_y - 1) / block_thread_y);

    std::cout << "Grid dimensions: " << num_blocks.x << "x" << num_blocks.y << std::endl;
    std::cout << "Block dimensions: " << threads_per_block.x << "x" << threads_per_block.y << std::endl;

    nvtxRangeId_t gpu_solver_range = nvtxRangeStartA("GPU Solver");  
    for (int step = 0; step < num_steps; ++step) {
        gpu_kernel<<<num_blocks, threads_per_block>>>(d_grid, d_new_grid, N, alpha, dt, dx2, dy2);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error (kernel launch): " << cudaGetErrorString(err) << std::endl;
            break;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error (device sync): " << cudaGetErrorString(err) << std::endl;
            break;
        }

        std::swap(d_grid, d_new_grid);
    }
    nvtxRangeEnd(gpu_solver_range);

    err = cudaMemcpy(grid, d_grid, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (memcpy to host): " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_grid);
    cudaFree(d_new_grid);
}