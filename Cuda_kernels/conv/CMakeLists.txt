cmake_minimum_required(VERSION 3.10)
project(ConvolutionSolver LANGUAGES CXX CUDA)

# Enable CUDA
find_package(CUDA REQUIRED)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Include directories
include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust as needed for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add source files
set(SOURCES_CPU src/ConvolutionSolverCPU.cpp)
set(SOURCES_GPU src/ConvolutionSolverGPU.cpp src/ConvolutionSolverCUDA.cu)
set(HEADERS include/ConvolutionSolverCPU.h include/ConvolutionSolverGPU.h)

# Add executables
add_executable(convolution main.cpp ${SOURCES_CPU} ${SOURCES_GPU} ${HEADERS})

# Set CUDA properties
set_target_properties(convolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA, OpenMP, and NVTX
target_link_libraries(convolution ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

# Add OpenMP flags
target_compile_options(convolution PRIVATE ${OpenMP_CXX_FLAGS})