cmake_minimum_required(VERSION 3.10)
project(HeatEquationSolver)

# Enable CUDA and OpenMP
enable_language(CUDA)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)  # Ensure CUDA toolkit is found

# Set the CUDA architecture (adjust according to your GPU's compute capability)
set(CMAKE_CUDA_ARCHITECTURES 75) # For NVIDIA RTX 2060, change if needed

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Define option for choosing installation type (development vs final)
option(DEVELOPMENT "Enable development mode" OFF)

# Define option for building a standalone C++ version without Python bindings
option(STANDALONE_CPP "Build standalone C++ version without Python bindings" OFF)

# Include directories
include_directories(include)

# Add the shared library for Python bindings
add_library(heat_solver SHARED src/HeatEquationSolver.cpp src/HeatEquationSolver.cu)

# Link CUDA and OpenMP libraries to the shared library
target_link_libraries(heat_solver PRIVATE CUDA::cudart OpenMP::OpenMP_CXX)

# Add a standalone C++ executable if STANDALONE_CPP is ON
if(STANDALONE_CPP)
    add_executable(heat_solver_cpp main.cpp)
    # Link the standalone C++ executable to the shared heat_solver library
    target_link_libraries(heat_solver_cpp PRIVATE heat_solver CUDA::cudart OpenMP::OpenMP_CXX)
    # Install the C++ executable into the bin directory
    install(TARGETS heat_solver_cpp RUNTIME DESTINATION bin)
endif()

# Define different installation directories for development and final install
if(DEVELOPMENT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/dev_install" CACHE PATH "Development install directory" FORCE)
else()
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/final_install" CACHE PATH "Final install directory" FORCE)
endif()

# Install the shared object (.so) file into the lib directory
install(TARGETS heat_solver LIBRARY DESTINATION lib)

# Install Python files into the python directory
install(FILES src/HeatEquationSolver.py DESTINATION python)

# Install header files
install(FILES include/HeatEquationSolver.h DESTINATION include)
