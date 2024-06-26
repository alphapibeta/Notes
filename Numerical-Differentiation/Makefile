CXX = g++
CXXFLAGS = -Isrc/include -Wall -std=c++11
LDFLAGS = 
TESTFLAGS = -lgtest -lgtest_main -pthread

# Define executables
EXEC = derivative
TEST_EXEC = runTests

# Source files
SRC_FILES = $(wildcard src/src/*.cpp)
TEST_SRC_FILES = $(wildcard src/tests/*.cpp)
MAIN_SRC_FILE = src/main.cpp

# Object files
OBJS = $(SRC_FILES:.cpp=.o) $(MAIN_SRC_FILE:.cpp=.o)
TEST_OBJS = $(filter-out $(MAIN_SRC_FILE:.cpp=.o), $(OBJS)) $(TEST_SRC_FILES:.cpp=.o)

# Default target
all: $(EXEC)

# Test build target
build_tests: $(TEST_EXEC)

# Run tests
run_tests: build_tests
	./$(TEST_EXEC)

# Build the main executable
$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Build the test executable
$(TEST_EXEC): $(TEST_OBJS)
	$(CXX) $(TESTFLAGS) -o $@ $^

# General rule for building object from source
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f $(OBJS) $(TEST_OBJS) $(EXEC) $(TEST_EXEC)

.PHONY: all clean build_tests run_tests