CXX = g++
NVCC = nvcc
CXXFLAGS = -Isrc/include -Wall -std=c++17
CUDAFLAGS = -Isrc/include -arch=sm_86  # Update the architecture to match your GPU's capability
LDFLAGS = -lnvToolsExt
TESTFLAGS = -lgtest -lgtest_main -pthread

# Define executables
EXEC = derivative
CUDA_EXEC = derivative_cuda
TEST_EXEC = runTests

# Source and CUDA files
SRC_FILES = $(wildcard src/src/*.cpp)
CUDA_SRC_FILES = $(wildcard src/src/*.cu)
TEST_SRC_FILES = $(wildcard src/tests/*.cpp)
MAIN_SRC_FILE = src/main.cpp

# Object files
OBJS = $(SRC_FILES:.cpp=.o) $(MAIN_SRC_FILE:.cpp=.o)
CUDA_OBJS = $(CUDA_SRC_FILES:.cu=.o)
TEST_OBJS = $(filter-out $(MAIN_SRC_FILE:.cpp=.o), $(OBJS)) $(TEST_SRC_FILES:.cpp=.o)

# Default target
all: $(CUDA_EXEC)

# CUDA build target
$(CUDA_EXEC): $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $^

# Build the main executable (only when needed)
build_main: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Build the test executable
build_tests: $(TEST_EXEC)

$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(TESTFLAGS) -o $@ $^

# General rule for building object from source
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# General rule for building CUDA object from CUDA source
%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(TEST_OBJS) $(EXEC) $(CUDA_EXEC) $(TEST_EXEC)

.PHONY: all clean build_tests build_main run_tests