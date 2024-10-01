cmake_minimum_required(VERSION 3.12)
project(HessianInversion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(include ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS})

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

# Set where to output the shared libraries (inside the Python package directory)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/hessian_inversion)

# Source files
set(SOURCES_CPU src/HessianInversionCPU.cpp)
set(SOURCES_GPU src/HessianInversionGPU.cu)

# Create CPU library
add_library(hessian_inversion_cpu SHARED ${SOURCES_CPU})
target_link_libraries(hessian_inversion_cpu PRIVATE OpenMP::OpenMP_CXX)

# Create GPU library
add_library(hessian_inversion_gpu SHARED ${SOURCES_GPU})
target_link_libraries(hessian_inversion_gpu PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)

# Create Python module for CPU
pybind11_add_module(hessian_inversion_cpu_py src/cpu_pybind_wrapper.cpp)
set_target_properties(hessian_inversion_cpu_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/hessian_inversion)
target_link_libraries(hessian_inversion_cpu_py PRIVATE hessian_inversion_cpu)

# Create Python module for GPU
pybind11_add_module(hessian_inversion_gpu_py src/gpu_pybind_wrapper.cpp)
set_target_properties(hessian_inversion_gpu_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/hessian_inversion)
target_link_libraries(hessian_inversion_gpu_py PRIVATE hessian_inversion_gpu)

# Installation rules
install(TARGETS hessian_inversion_cpu hessian_inversion_gpu RUNTIME DESTINATION lib)
install(TARGETS hessian_inversion_cpu_py hessian_inversion_gpu_py LIBRARY DESTINATION "${Python3_SITEARCH}")
