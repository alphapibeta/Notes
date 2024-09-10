#include "HeatEquationSolver.h"
#include <iostream>
#include <omp.h>
#include <nvtx3/nvToolsExt.h>  // NVTX header for profiling

void cpu_heat_equation_solver(int N, int num_steps, float *grid, float alpha, float dt, float dx2, float dy2, int num_threads) {
    float *new_grid = new float[N * N];
    omp_set_num_threads(num_threads);

    // Add NVTX marker for the main loop
    nvtxRangeId_t cpu_range = nvtxRangeStartA("CPU Solver");

    for (int step = 0; step < num_steps; ++step) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int index = i * N + j;
                float temp = grid[index];
                float temp_left = grid[i * N + (j - 1)];
                float temp_right = grid[i * N + (j + 1)];
                float temp_top = grid[(i - 1) * N + j];
                float temp_bottom = grid[(i + 1) * N + j];
                new_grid[index] = temp + alpha * dt * ((temp_left - 2.0f * temp + temp_right) / dx2 +
                                                       (temp_top - 2.0f * temp + temp_bottom) / dy2);
            }
        }
        std::swap(grid, new_grid);
    }

    nvtxRangeEnd(cpu_range);  // End NVTX marker for the CPU solver
    delete[] new_grid;
}
