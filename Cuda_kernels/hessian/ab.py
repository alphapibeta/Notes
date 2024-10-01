import numpy as np
import time
from hessian_inversion import HessianInversion, CPU_AVAILABLE, GPU_AVAILABLE

import sys
sys.path.append("/home/tesla/exp/Notes/Cuda_kernels/hessian/python/")


# Function to generate a random matrix
def generate_random_matrix(size):
    return np.random.rand(size, size).tolist()


# Function to compare matrices with tolerance
def compare_matrices(cpu_matrix, gpu_matrix, size, abs_tolerance=1e-5, rel_tolerance=0.01):
    mismatches = 0
    max_abs_diff = 0
    max_rel_diff = 0
    sum_abs_diff = 0
    sum_rel_diff = 0

    for i in range(size):
        for j in range(size):
            cpu_val = cpu_matrix[i][j]
            gpu_val = gpu_matrix[i * size + j]
            abs_diff = abs(cpu_val - gpu_val)
            rel_diff = abs_diff / max(abs(cpu_val), abs(gpu_val))

            sum_abs_diff += abs_diff
            sum_rel_diff += rel_diff

            if abs_diff > abs_tolerance and rel_diff > rel_tolerance:
                mismatches += 1
                max_abs_diff = max(max_abs_diff, abs_diff)
                max_rel_diff = max(max_rel_diff, rel_diff)
                if mismatches <= 5:
                    print(f"Mismatch at ({i}, {j}): CPU = {cpu_val}, GPU = {gpu_val}, abs diff = {abs_diff}, rel diff = {rel_diff}")

    avg_abs_diff = sum_abs_diff / (size * size)
    avg_rel_diff = sum_rel_diff / (size * size)

    print(f"Total mismatches: {mismatches}\nMax absolute difference: {max_abs_diff}\nMax relative difference: {max_rel_diff}\nAverage absolute difference: {avg_abs_diff}\nAverage relative difference: {avg_rel_diff}")

    return mismatches == 0


def run_gpu_inversion(matrix_size, run_cpu):
    print(f"Running GPU Hessian Inversion with matrix size: {matrix_size}x{matrix_size}")
    
    # Generate a random matrix and add regularization
    matrix = generate_random_matrix(matrix_size)
    for i in range(matrix_size):
        matrix[i][i] += 1e-6  # Add a small value to the diagonal for stability

    if GPU_AVAILABLE:
        hessian_gpu = HessianInversion(matrix_size, use_gpu=True)
        hessian_gpu.setMatrix(matrix)

        start_gpu = time.time()
        hessian_gpu.invert()
        end_gpu = time.time()

        gpu_inverse = hessian_gpu.getInverse()
        duration_gpu = end_gpu - start_gpu

        print(f"GPU Inversion Time: {duration_gpu * 1000:.2f} ms")
        # You can access additional GPU-specific information here if needed:
        # print(f"GPU Memory Throughput: {hessian_gpu.getGPUMemoryThroughput()} GB/s")
        # print(f"GPU Computational Throughput: {hessian_gpu.getGPUComputationalThroughput()} GFLOP/s")

        if run_cpu and CPU_AVAILABLE:
            hessian_cpu = HessianInversion(matrix_size, use_gpu=False)
            hessian_cpu.setMatrix(matrix)

            start_cpu = time.time()
            hessian_cpu.invert()
            end_cpu = time.time()

            cpu_inverse = hessian_cpu.getInverse()
            duration_cpu = end_cpu - start_cpu

            print(f"CPU Inversion Time: {duration_cpu * 1000:.2f} ms")
            
            # Compare CPU and GPU results
            are_equal = compare_matrices(cpu_inverse, gpu_inverse, matrix_size)
            if are_equal:
                print("CPU and GPU results match within tolerance!")
            else:
                print("CPU and GPU results do NOT match within tolerance!")
        elif run_cpu:
            print("CPU module not available for inversion.")
    else:
        print("GPU module not available for inversion.")


if __name__ == "__main__":
    # Example usage: running both GPU and CPU inversion
    matrix_size = 4
    run_cpu = True  # Set to True if you want to run CPU inversion
    run_gpu_inversion(matrix_size, run_cpu)
