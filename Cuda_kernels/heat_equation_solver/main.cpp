#include "HeatEquationSolver.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <cstring>  // for memcpy

// Initialize the grid
void initialize_grid(float *grid, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            grid[i * N + j] = (i == N / 2 && j == N / 2) ? 100.0f : 25.0f;
        }
    }
}


// void print_grid(float *grid, int N) {
//     for (int i = 0; i < N; ++i) {
//         for (int j = 0; j < N; ++j) {
//             std::cout << grid[i * N + j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// Compare two grids to check if they are the same within a small tolerance
bool compare_grids(float *cpu_grid, float *gpu_grid, int N, float tolerance = 1e-5f) {
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(cpu_grid[i] - gpu_grid[i]) > tolerance) {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpu_grid[i] << ", GPU = " << gpu_grid[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <block_thread_x> <block_thread_y> <N> <num_steps> <verify_cpu> [num_cores]" << std::endl;
        return 1;
    }

    int block_thread_x = std::stoi(argv[1]);
    int block_thread_y = std::stoi(argv[2]);
    int N = std::stoi(argv[3]);
    int num_steps = std::stoi(argv[4]);
    bool verify_cpu = std::stoi(argv[5]) != 0;
    int num_cores = (argc == 7) ? std::stoi(argv[6]) : omp_get_max_threads();

    float alpha = 0.5f;
    float dx = 0.01f, dy = 0.01f;
    float dt = dx * dy / (2.0f * alpha * (dx * dx + dy * dy));

    // Allocate memory for GPU grid and also to CPU grid if verification is enabled
    float *cpu_grid = nullptr;
    float *gpu_grid = new float[N * N];
    initialize_grid(gpu_grid, N);

    if (verify_cpu) {
        cpu_grid = new float[N * N];
        memcpy(cpu_grid, gpu_grid, N * N * sizeof(float)); // Copy initial state to CPU grid

        std::cout << "Running CPU solver with " << num_cores << " cores..." << std::endl;
        omp_set_num_threads(num_cores);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_heat_equation_solver(N, num_steps, cpu_grid, alpha, dt, dx * dx, dy * dy, num_cores);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;
        std::cout << "CPU elapsed time: " << cpu_elapsed.count() << " seconds" << std::endl;
    }

    // GPU Test 
    std::cout << "Running GPU solver..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_heat_equation_solver(N, num_steps, gpu_grid, alpha, dt, dx * dx, dy * dy, block_thread_x, block_thread_y);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;
    std::cout << "GPU elapsed time: " << gpu_elapsed.count() << " seconds" << std::endl;

    // Verify convergence between CPU and GPU results (if verification is enabled)
    if (verify_cpu) {
        bool converged = compare_grids(cpu_grid, gpu_grid, N);
        if (converged) {
            std::cout << "CPU and GPU results match!" << std::endl;
        } else {
            std::cout << "CPU and GPU results do not match!" << std::endl;
        }
    }

    // Clean up
    if (verify_cpu) {
        delete[] cpu_grid;
    }
    delete[] gpu_grid;

    return 0;
}