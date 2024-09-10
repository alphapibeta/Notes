cmake_minimum_required(VERSION 3.10)
project(HeatEquationSolver)

# Enable CUDA and OpenMP
enable_language(CUDA)
find_package(OpenMP REQUIRED)

# Set the CUDA architecture (adjust according to your GPU's compute capability)
set(CMAKE_CUDA_ARCHITECTURES 75) # For NVIDIA RTX 2060, change if needed

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Add the executable
add_executable(heat_solver main.cpp src/HeatEquationSolver.cpp src/HeatEquationSolver.cu)

# Link CUDA and OpenMP libraries
target_link_libraries(heat_solver CUDA::cudart OpenMP::OpenMP_CXX)
