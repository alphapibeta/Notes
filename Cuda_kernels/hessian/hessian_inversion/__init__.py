# Import the shared libraries (bindings) for CPU and GPU
from .hessian_inversion_cpu_py import HessianInversionCPU
from .hessian_inversion_gpu_py import HessianInversionGPU_float, HessianInversionGPU_double

# Provide Python wrapper functions to make it easier to use
def cpu_inversion(matrix):
    cpu_solver = HessianInversionCPU(len(matrix))
    cpu_solver.setMatrix(matrix)
    cpu_solver.invert()
    return cpu_solver.getInverse()

def flatten_matrix(matrix):
    return [element for row in matrix for element in row]

def gpu_inversion(matrix, precision="double"):
    if precision == "float":
        gpu_solver = HessianInversionGPU_float(len(matrix))
    elif precision == "double":
        gpu_solver = HessianInversionGPU_double(len(matrix))
    else:
        raise ValueError("Precision must be 'float' or 'double'")
    
    flattened_matrix = flatten_matrix(matrix)
    gpu_solver.setMatrix(flattened_matrix)
    gpu_solver.invert()
    return gpu_solver.getInverse()
