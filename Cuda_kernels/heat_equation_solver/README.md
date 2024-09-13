
# Heat Equation Solver

This repository contains a CUDA-based heat equation solver, supporting both Python bindings and a standalone C++ executable. The solver can run on both the CPU and GPU and can be used to compare the performance between the two implementations.

## Features

- GPU-accelerated heat equation solver using CUDA.
- CPU-based solver using OpenMP for multi-threading.
- Python bindings for easy integration into Python workflows.
- Standalone C++ version for direct execution.
- Configurable grid sizes, time steps, and solver parameters.
- Verification between CPU and GPU results.

## Installation

### Prerequisites

- CMake (Version 3.10 or higher)
- CUDA Toolkit (Version 12.3 or higher)
- GCC (GNU Compiler)
- Python 3.6+
- Python packages: `numpy`, `pycuda`

### Clone the Repository

```bash
git clone https://github.com/your-username/heat-equation-solver.git
cd heat-equation-solver
```

### Development Mode vs Final Installation

The project supports both development mode (for quick builds) and final installation mode (for production use).

### Building for Development

#### Step 1: Create the Build Directory and Configure CMake

```bash
mkdir build && cd build
cmake -DDEVELOPMENT=ON -DSTANDALONE_CPP=ON ..
```

#### Step 2: Build the Project

```bash
make -j$(nproc)
```

#### Step 3: Install the Project

```bash
make install
```

The `dev_install` directory will contain the necessary files for both the Python bindings and the standalone C++ executable.

### Building for Final Installation

#### Step 1: Create the Build Directory and Configure CMake

```bash
mkdir build && cd build
cmake -DDEVELOPMENT=OFF -DSTANDALONE_CPP=ON ..
```

#### Step 2: Build the Project

```bash
make -j$(nproc)
```

#### Step 3: Install the Project

```bash
make install
```

The `final_install` directory will contain the final production-ready files.

## Python Installation

To use the Python bindings, you need to install the package as a Python wheel.

### Building the Python Wheel

After performing the final installation:

```bash
cd /path/to/repo/heat-equation-solver
python setup.py bdist_wheel
```

### Installing the Python Wheel

```bash
pip install dist/heat_solver-1.0-py3-none-any.whl
```

## Running the Standalone C++ Executable

After installing the standalone C++ version, you can run the executable with the following command:

```bash
./bin/heat_solver_cpp <block_thread_x> <block_thread_y> <grid_size> <num_steps> <verify_cpu> [num_cores]
```

### Example:

```bash
./bin/heat_solver_cpp 16 16 1024 400 1 12
```

This runs the heat equation solver on a 1024x1024 grid for 400 time steps, comparing CPU and GPU results, using 12 CPU cores.

## Running the Python Version

You can run the Python version of the solver by importing the `heat_solver` module:

```python
import heat_solver
```

### Example Python Script

```python
import numpy as np
import heat_solver

# Initialize grid
N = 1024
grid = np.full((N, N), 25.0, dtype=np.float32)
grid[N // 2, N // 2] = 100.0  # Initial heat source

# Run the GPU solver
heat_solver.gpu_heat_solver(N, 400, 0.5, 0.01, 0.01, 0.01, 16, 16, grid)

# Print final grid
print(grid)
```

### Testing the Python Version

To test the Python version, use the provided test script:

```bash
python tests/test_heat_equation.py <block_thread_x> <block_thread_y> <grid_size> <num_steps> <verify_cpu> [num_cores]
```

### Example:

```bash
python tests/test_heat_equation.py 16 16 1024 400 1 12
```

This runs the test for the heat equation solver on a 1024x1024 grid for 400 time steps, comparing the CPU and GPU results, using 12 CPU cores for the CPU version.

## Documentation

- The solver uses the finite difference method to solve the heat equation on a 2D grid.
- The grid is updated at each time step using either the CPU (with OpenMP multi-threading) or the GPU (with CUDA).
- The solver verifies the correctness of the GPU results by comparing them with the CPU results.

## Debugging

If you encounter errors related to library linking, ensure that:
- The correct CUDA Toolkit version is installed.
- The environment variables are correctly set for CUDA and OpenMP.
- You are linking to the correct shared library paths.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the open-source community for their contributions to CUDA, OpenMP, and Python.

For questions or issues, please open a GitHub issue.
