cmake_minimum_required(VERSION 3.10)
project(DistanceCalculatorCUDA)

# Enable CUDA
find_package(CUDA REQUIRED)

# Set the CUDA architecture based on your GPU
set(CUDA_ARCHITECTURE "75")  # For Turing-based GPUs like RTX 2060/2070/2080
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCHITECTURE}")

# Include directories
include_directories(include)

# Add NVTX flags
find_package(CUDA REQUIRED nvToolsExt)
include_directories(${CUDA_NVTOOLSEXT_INCLUDE})

# CUDA compilation
cuda_add_library(DistanceCalculator src/DistanceCalculator.cpp src/DistanceCalculator.cu)

# Build the main executable
add_executable(max_distance main.cpp)
target_link_libraries(max_distance DistanceCalculator ${CUDA_CUDA_LIBRARY} ${CUDA_NVTX3_LIBRARY})

