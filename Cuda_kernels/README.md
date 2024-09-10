
# Project Overview

This project contains four main components:
1. **Heat Equation Solver**: A CUDA-based solver for the heat equation, with OpenMP support for parallel CPU computations.
2. **Max Distance Kernel**: A CUDA-based distance calculator.
3. **X86 Profiler**: A Python script for profiling the performance of the CUDA kernels.
4. **Streamlit App**: A Streamlit-based interactive dashboard for visualizing the profiling results.

## Project Structure

```
.
├── heat_equation_solver
│   ├── CMakeLists.txt
│   ├── include
│   │   └── HeatEquationSolver.h
│   ├── main.cpp
│   └── src
│       ├── HeatEquationSolver.cpp
│       └── HeatEquationSolver.cu
├── max-distance
│   ├── CMakeLists.txt
│   ├── include
│   │   └── DistanceCalculator.h
│   ├── main.cpp
│   └── src
│       ├── DistanceCalculator.cpp
│       └── DistanceCalculator.cu
├── README.md
├── run_project.sh
├── X86_profiler.py
├── x86_results
│   ├── test_distance_final_x86
│   ├── test_distance_final_x86.csv
│   ├── test_distance_final_x86.txt
│   ├── test_heat_final_x86
│   ├── test_heat_final_x86.csv
│   └── test_heat_final_x86.txt
└── x86_streamlit_app.py

7 directories, 20 files
```

## Installation and Execution

### 1. Build the CUDA Projects

Follow the steps below to build both the `heat_equation_solver` and `max-distance` projects.

#### Step 1: Build the Heat Equation Solver

```bash
cd heat_equation_solver
mkdir -p build
cd build
cmake ../
make -j$(nproc)
```

#### Step 2: Build the Max Distance Kernel

```bash
cd ../../max-distance
mkdir -p build
cd build
cmake ../
make -j$(nproc)
```

### 2. Run the CUDA Binaries

Once the projects are built, you can execute the binaries to run the CUDA kernels.

#### Usage of the Heat Equation Solver (with OpenMP):

```bash
./heat_solver <block_thread_x> <block_thread_y> <N> <num_steps> <verify_cpu> [num_cores]
```

- **block_thread_x**: Number of threads per block in the X direction.
- **block_thread_y**: Number of threads per block in the Y direction.
- **N**: Size of the grid (number of points).
- **num_steps**: Number of time steps for the solver.
- **verify_cpu**: Whether to verify the results using CPU (0 or 1).
- **num_cores**: (Optional) Number of CPU cores to use for the OpenMP parallel CPU solver.

The heat equation is solved in parallel using OpenMP by specifying the number of cores, with NVTX markers used for profiling in Nsight Compute and Nsight Systems.

Example:

```bash
./heat_solver 32 32 4096 100 0 4
```

#### Usage of the Max Distance Kernel:

```bash
./max_distance <threads_x> <threads_y> <num_points> <init_mode>
```

- **threads_x**: Number of threads per block in the X direction.
- **threads_y**: Number of threads per block in the Y direction.
- **num_points**: Number of points for distance calculation.
- **init_mode**: Initialization mode for points (0 for random, 1 for sequential).

Example:

```bash
./max_distance 32 32 4096 0
```

### 3. Run the X86 Profiler

You can use the `X86_profiler.py` to profile the performance of both kernels. This profiler uses Nsight Compute to gather detailed performance metrics.

#### Usage of the X86 Profiler:

```bash
python3 X86_profiler.py --build_dirs <build_directory> --kernels <kernel_name> --exec_name <binary_name> --kernel_args <args> --output_file <output_file>
```

Options:
- **`--build_dirs`**: Path to the build directory.
- **`--kernels`**: Name of the kernel to profile.
- **`--exec_name`**: Name of the executable file.
- **`--kernel_args`**: Arguments for the kernel (e.g., block size, grid size).
- **`--output_file`**: File to store the profiling results.

#### Example for the Heat Equation Solver:

```bash
python3 X86_profiler.py --build_dirs ./heat_equation_solver/build --kernels HeatEquationSolver --exec_name ./heat_solver --kernel_args block-x=32 block-y=32 N=4096 nsteps=100 --output_file ./x86_results/test_heat_final_x86
```

#### Example for the Max Distance Kernel:

```bash
python3 X86_profiler.py --build_dirs ./max-distance/build --kernels computeDistancesKernel --exec_name ./max_distance --kernel_args block-x=32 block-y=32 N=4096 --output_file ./x86_results/test_distance_final_x86
```

### 4. Run the Streamlit App

After profiling the kernels, you can use the Streamlit app to visualize the results interactively.

#### Usage of the Streamlit App:

```bash
streamlit run x86_streamlit_app.py <path_to_csv_file>
```

Example:

```bash
streamlit run x86_streamlit_app.py ./x86_results/test_heat_final_x86.csv
```

The app will visualize various performance metrics, such as execution time, memory throughput, and occupancy, allowing you to analyze the kernel's performance with different configurations.

---

## Profiling Tools: Nsight Compute and Nsight Systems

### Nsight Compute

Nsight Compute is used for in-depth kernel analysis and performance profiling. This tool provides insights into kernel efficiency, including SM efficiency, memory throughput, and occupancy.

### Nsight Systems

Nsight Systems is a system-wide profiler used to capture CPU, GPU, and memory activity. It can be used to understand the interaction between different components of your application and detect performance bottlenecks across the system.

To enable profiling markers, **NVTX** markers are used in the source code (as seen in the `cpu_heat_equation_solver`).

---

## Files Overview

### 1. `heat_equation_solver`

This folder contains the implementation of the CUDA-based heat equation solver.

- **`CMakeLists.txt`**: Configuration file for building the project with CMake.
- **`HeatEquationSolver.h`**: Header file defining the solver's functionality.
- **`HeatEquationSolver.cpp` & `HeatEquationSolver.cu`**: C++ and CUDA source files implementing the solver.
- **OpenMP Support**: The solver uses OpenMP for parallel CPU computations.

### 2. `max-distance`

This folder contains the implementation of the CUDA-based distance calculator.

- **`CMakeLists.txt`**: Configuration file for building the project with CMake.
- **`DistanceCalculator.h`**: Header file defining the distance calculator functionality.
- **`DistanceCalculator.cpp` & `DistanceCalculator.cu`**: C++ and CUDA source files implementing the distance calculator.

### 3. `X86_profiler.py`

This Python script profiles the performance of CUDA kernels by running the binaries and collecting Nsight Compute metrics. It outputs results in CSV format for further analysis.

### 4. `x86_streamlit_app.py`

This Streamlit-based app visualizes the profiling results in an interactive web interface. It supports a wide range of metrics and allows for custom graph creation.

### 5. `x86_results`

This folder stores the CSV and text files generated by the profiler for both the heat equation solver and the distance calculator.

---

## Conclusion

This project is part of my learning process, where I have implemented CUDA kernel performance analysis with OpenMP support for parallel CPU processing. By using tools like Nsight Compute and Nsight Systems for profiling and an interactive Streamlit app for visualizing results, the project aims to explore ways to optimize block sizes, memory configurations, and kernel efficiency.

Feel free to experiment with different configurations and analyze the results using the provided tools!
