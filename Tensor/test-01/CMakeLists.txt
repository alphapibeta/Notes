cmake_minimum_required(VERSION 3.10)
project(TensorTest VERSION 1.0 LANGUAGES CXX)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../install/include)

# Find the shared library in the install location
find_library(TENSOR_LIB_SHARED tensor_lib_shared PATHS ../install/lib)

# Find CUDA (if not already done)
find_package(CUDA REQUIRED)

# Create an executable for the shared library
add_executable(test_shared test-01.cpp)

# Link the shared library and CUDA
target_link_libraries(test_shared ${TENSOR_LIB_SHARED} ${CUDA_LIBRARIES})

# Set output directory for executables
set_target_properties(test_shared PROPERTIES OUTPUT_NAME "test_shared")
