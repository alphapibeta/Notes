#include "ConvolutionSolverCPU.h"
#include "ConvolutionSolverGPU.h"
#include <iostream>
#include <cassert>

// Test CPU and GPU Convolution results
void test_convolution() {
    CPUTensor input_cpu({1, 1, 5, 5});
    CPUTensor filter_cpu({1, 1, 3, 3});
    input_cpu.initialize(1);  // Initialize with ones
    filter_cpu.initialize(1); // Initialize with ones

    GPUTensor input_gpu({1, 1, 5, 5});
    GPUTensor filter_gpu({1, 1, 3, 3});
    input_gpu.initialize(1);  // Initialize with ones
    filter_gpu.initialize(1); // Initialize with ones

    CPUTensor output_cpu = convolution2DCPU(input_cpu, filter_cpu);
    GPUTensor output_gpu({1, 1, 3, 3});

    convolution2DGPU(input_gpu, filter_gpu, output_gpu, 16, 16);

    std::vector<float> gpu_output_data(9);
    output_gpu.copyToCPU(gpu_output_data);

    for (size_t i = 0; i < output_cpu.data.size(); ++i) {
        assert(abs(output_cpu.data[i] - gpu_output_data[i]) < 1e-4);
    }

    std::cout << "Test Passed: CPU and GPU outputs match!" << std::endl;
}

int main() {
    test_convolution();
    return 0;
}
