import numpy as np
from heat_solver import cpu_heat_solver, gpu_heat_solver

class HeatEquationSolver:
    def __init__(self, N, block_thread_x=16, block_thread_y=16):
        self.N = N
        self.block_thread_x = block_thread_x
        self.block_thread_y = block_thread_y
        self.alpha = 0.5
        self.dx = self.dy = 0.01
        self.dt = self.dx * self.dy / (2.0 * self.alpha * (self.dx * self.dx + self.dy * self.dy))
        self.dx2 = self.dx * self.dx
        self.dy2 = self.dy * self.dy

    def initialize_grid(self):
        grid = np.full((self.N, self.N), 25.0, dtype=np.float32)
        grid[self.N // 2, self.N // 2] = 100.0
        return grid

    def solve_cpu(self, num_steps, num_threads=None):
        grid = self.initialize_grid()
        cpu_heat_solver(self.N, num_steps, self.alpha, self.dt, self.dx2, self.dy2, num_threads or 0, grid)
        return grid

    def solve_gpu(self, num_steps):
        grid = self.initialize_grid()
        gpu_heat_solver(self.N, num_steps, self.alpha, self.dt, self.dx2, self.dy2, 
                        self.block_thread_x, self.block_thread_y, grid)
        return grid