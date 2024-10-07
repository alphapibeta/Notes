cmake_minimum_required(VERSION 3.10)
project(TensorTests LANGUAGES CXX CUDA)

# Include the main project
include_directories(../include)

# Find GoogleTest
find_package(GTest REQUIRED)

# Specify test sources
set(TEST_SOURCES
    test_cpu_tensor.cpp
    test_gpu_tensor.cu
)

# Specify the source files that are needed for linking
set(SOURCES
    ../src/Storage.cpp
    ../src/Tensor.cpp
    ../src/TensorFactory.cpp
    ../cuda/GPUStorage.cu
    ../cuda/GPUTensor.cu
)

# Add test executable
add_executable(tensor_tests ${TEST_SOURCES} ${SOURCES})

# Link GoogleTest and CUDA libraries
target_link_libraries(tensor_tests ${GTEST_LIBRARIES} GTest::Main ${CUDA_LIBRARIES})

# Enable CUDA support for test executable
set_target_properties(tensor_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
