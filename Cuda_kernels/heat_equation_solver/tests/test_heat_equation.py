import numpy as np
import time
import argparse
import heat_solver

# Initialize the grid 
def initialize_grid(N):
    grid = np.full((N, N), 25.0, dtype=np.float32)
    grid[N // 2, N // 2] = 100.0  # Central heat spot
    return grid

# Compare CPU and GPU grids, print the first mismatch if any
def compare_grids(cpu_grid, gpu_grid, tolerance=1e-5):
    diff = np.abs(cpu_grid - gpu_grid)
    if np.any(diff > tolerance):
        mismatch = np.where(diff > tolerance)
        i, j = mismatch[0][0], mismatch[1][0]
        print(f"Mismatch at index ({i}, {j}): CPU = {cpu_grid[i, j]}, GPU = {gpu_grid[i, j]}")
        return False
    return True


# Main function 
def main():
    parser = argparse.ArgumentParser(description="Heat Equation Solver")
    parser.add_argument("block_thread_x", type=int, help="Block thread x dimension")
    parser.add_argument("block_thread_y", type=int, help="Block thread y dimension")
    parser.add_argument("N", type=int, help="Grid size")
    parser.add_argument("num_steps", type=int, help="Number of time steps")
    parser.add_argument("verify_cpu", type=int, help="1 to verify CPU results, 0 otherwise")
    parser.add_argument("num_cores", type=int, nargs='?', default=None, help="Number of CPU cores to use")
    
    args = parser.parse_args()

    alpha = 0.5
    dx = dy = 0.01
    dt = dx * dy / (2.0 * alpha * (dx * dx + dy * dy))  # Time step size
    dx2 = dx * dx
    dy2 = dy * dy

    # Initialize GPU grid and CPU grid (if verification enabled)
    gpu_grid = initialize_grid(args.N)
    cpu_grid = None

    # Run CPU solver if verification is enabled
    if args.verify_cpu:
        cpu_grid = gpu_grid.copy()
        print(f"Running CPU solver with {args.num_cores or 'max'} cores...")
        start_time = time.time()
        heat_solver.cpu_heat_solver(args.N, args.num_steps, alpha, dt, dx2, dy2, args.num_cores or 0, cpu_grid)
        end_time = time.time()
        print(f"CPU elapsed time: {end_time - start_time:.6f} seconds")

    # Run GPU solver
    print("Running GPU solver...")
    start_time = time.time()
    heat_solver.gpu_heat_solver(args.N, args.num_steps, alpha, dt, dx2, dy2, 
                                args.block_thread_x, args.block_thread_y, gpu_grid)
    end_time = time.time()
    print(f"GPU elapsed time: {end_time - start_time:.6f} seconds")

    # Compare CPU and GPU grids (if verification enabled)
    if args.verify_cpu:
        if compare_grids(cpu_grid, gpu_grid, args.N):
            print("CPU and GPU results match!")
        else:
            print("CPU and GPU results do not match!")
        diff = np.abs(cpu_grid - gpu_grid)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        print(f"Maximum difference: {max_diff}")
        print(f"Average difference: {avg_diff}")

if __name__ == "__main__":
    main()
