import numpy as np
import pytest
from hessian_inversion import cpu_inversion, gpu_inversion

# Global tolerances
ABS_TOL_FLOAT = 1e-3
REL_TOL_FLOAT = 0.1
ABS_TOL_DOUBLE = 1e-3
REL_TOL_DOUBLE = 0.01
STRICT_ABS_TOL = 1e-3
STRICT_REL_TOL = 0.001

def compare_matrices_hybrid(cpu_matrix, gpu_matrix, abs_tolerance, rel_tolerance):
    size = len(cpu_matrix)
    mismatches = 0
    max_abs_diff = 0
    max_rel_diff = 0
    sum_abs_diff = 0
    sum_rel_diff = 0

    gpu_matrix_flat = np.array(gpu_matrix).flatten()

    for i in range(size):
        for j in range(size):
            cpu_val = cpu_matrix[i][j]
            gpu_val = gpu_matrix_flat[i * size + j]
            abs_diff = abs(cpu_val - gpu_val)
            rel_diff = abs_diff / max(abs(cpu_val), abs(gpu_val)) if (cpu_val != 0 or gpu_val != 0) else 0

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

    print(f"Total mismatches: {mismatches}")
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Max relative difference: {max_rel_diff}")
    print(f"Average absolute difference: {avg_abs_diff}")
    print(f"Average relative difference: {avg_rel_diff}")

    return mismatches == 0


@pytest.mark.parametrize("size", [2, 5, 10, 50, 100])
@pytest.mark.parametrize("precision", ["float", "double"])
def test_random_matrix_inversion(size, precision):
    np.random.seed(42)
    matrix = np.random.rand(size, size).tolist()

    cpu_result = cpu_inversion(matrix)
    gpu_result = gpu_inversion(matrix, precision=precision)

    assert len(cpu_result) == size
    assert len(gpu_result) == size * size  # GPU result is flattened
    abs_tol = ABS_TOL_FLOAT if precision == "float" else ABS_TOL_DOUBLE
    rel_tol = REL_TOL_FLOAT if precision == "float" else REL_TOL_DOUBLE
    assert compare_matrices_hybrid(cpu_result, gpu_result, abs_tolerance=abs_tol, rel_tolerance=rel_tol)


@pytest.mark.parametrize("size", [2, 5, 10, 50, 100])
@pytest.mark.parametrize("precision", ["float", "double"])
def test_identity_matrix_inversion(size, precision):
    identity_matrix = np.identity(size).tolist()

    cpu_result = cpu_inversion(identity_matrix)
    gpu_result = gpu_inversion(identity_matrix, precision=precision)

    assert len(cpu_result) == size
    assert len(gpu_result) == size * size  # GPU result is flattened
    abs_tol = ABS_TOL_FLOAT if precision == "float" else ABS_TOL_DOUBLE
    rel_tol = REL_TOL_FLOAT if precision == "float" else REL_TOL_DOUBLE
    assert compare_matrices_hybrid(cpu_result, gpu_result, abs_tolerance=abs_tol, rel_tolerance=rel_tol)


@pytest.mark.parametrize("precision", ["float", "double"])
def test_small_matrix_inversion(precision):
    # Testing small matrix with higher precision
    matrix = [[4.0, 2.0], [3.0, 1.0]]

    cpu_result = cpu_inversion(matrix)
    gpu_result = gpu_inversion(matrix, precision=precision)

    assert len(cpu_result) == len(matrix)
    abs_tol = ABS_TOL_FLOAT if precision == "float" else ABS_TOL_DOUBLE
    rel_tol = REL_TOL_FLOAT if precision == "float" else REL_TOL_DOUBLE
    assert compare_matrices_hybrid(cpu_result, gpu_result, abs_tolerance=abs_tol, rel_tolerance=rel_tol)


@pytest.mark.parametrize("size", [50, 100])
@pytest.mark.parametrize("precision", ["float", "double"])
def test_large_matrix_inversion(size, precision):
    np.random.seed(42)
    matrix = np.random.rand(size, size).tolist()

    cpu_result = cpu_inversion(matrix)
    gpu_result = gpu_inversion(matrix, precision=precision)

    assert len(cpu_result) == size
    assert len(gpu_result) == size * size  # GPU result is flattened
    abs_tol = ABS_TOL_FLOAT if precision == "float" else ABS_TOL_DOUBLE
    rel_tol = REL_TOL_FLOAT if precision == "float" else REL_TOL_DOUBLE
    assert compare_matrices_hybrid(cpu_result, gpu_result, abs_tolerance=abs_tol, rel_tolerance=rel_tol)


@pytest.mark.parametrize("size", [100,500,1000])
@pytest.mark.parametrize("precision", ["float", "double"])
def test_error_tolerance(size, precision):
    np.random.seed(42)
    matrix = np.random.rand(size, size).tolist()

    cpu_result = cpu_inversion(matrix)
    gpu_result = gpu_inversion(matrix, precision=precision)

    # Using stricter tolerance to catch differences
    assert compare_matrices_hybrid(cpu_result, gpu_result, abs_tolerance=STRICT_ABS_TOL, rel_tolerance=STRICT_REL_TOL)
