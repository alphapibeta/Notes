#ifndef HEAT_EQUATION_SOLVER_H
#define HEAT_EQUATION_SOLVER_H

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>  
#endif

void cpu_heat_equation_solver(int N, int num_steps, float *grid, float alpha, float dt, float dx2, float dy2, int num_threads);
void gpu_heat_equation_solver(int N, int num_steps, float *grid, float alpha, float dt, float dx2, float dy2, int block_thread_x, int block_thread_y);

#endif // HEAT_EQUATION_SOLVER_H
