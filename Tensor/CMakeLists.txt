cmake_minimum_required(VERSION 3.10)
project(TensorProject VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(include)

# Specify source files for the library
set(LIBRARY_SOURCES
    src/Storage.cpp
    src/Tensor.cpp
    src/TensorFactory.cpp
    cuda/GPUStorage.cu
    cuda/GPUTensor.cu
)

# Create a shared library
add_library(tensor_lib_shared SHARED ${LIBRARY_SOURCES})

# Link CUDA libraries
target_link_libraries(tensor_lib_shared ${CUDA_LIBRARIES})

# Enable CUDA support for the shared library
set_target_properties(tensor_lib_shared PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Specify CUDA architectures
set_target_properties(tensor_lib_shared PROPERTIES
    CUDA_ARCHITECTURES "75" # Adjust based on your GPU
)

# Specify the executable
add_executable(tensor_program main.cpp)

# Link the shared library to the executable
target_link_libraries(tensor_program tensor_lib_shared ${CUDA_LIBRARIES})

# Install rules (optional)
install(TARGETS tensor_lib_shared
    EXPORT TensorLibTargets
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/TensorLibrary.h DESTINATION include)

# Exporting the targets
install(EXPORT TensorLibTargets
    FILE TensorLibTargets.cmake
    NAMESPACE Tensor::
    DESTINATION lib/cmake/Tensor
)

# Add CMake config for find_package
include(CMakePackageConfigHelpers)

# Write version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TensorLibraryConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Configure package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/TensorLibraryConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorLibraryConfig.cmake"
    INSTALL_DESTINATION lib/cmake/Tensor
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TensorLibraryConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TensorLibraryConfigVersion.cmake"
    DESTINATION lib/cmake/Tensor
)
