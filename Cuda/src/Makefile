# Specify the compiler to use
CC=nvcc
# Specify the target architecture
ARCH=sm_86

# Include directories
INCLUDES=-I./include

# Source files
SOURCES=src/Operations.cu main.cpp

# Output binary
OUT=program.out

# Build target
all:
	$(CC) $(SOURCES) -arch=$(ARCH) $(INCLUDES) -o $(OUT)

# Clean the build
clean:
	rm -f $(OUT)