#include "ConvolutionSolverCPU.h"
#include "ConvolutionSolverGPU.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

// Compile-time print flag
constexpr bool ENABLE_PRINT = false;

void printShape(const std::string& name, const std::vector<int>& shape) {
    std::cout << name << " shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

bool verifyOutputs(const CPUTensor& cpu_output, const GPUTensor& gpu_output, float tolerance = 1e-5) {
    if (cpu_output.shape != gpu_output.shape) {
        std::cerr << "Error: CPU and GPU output shapes do not match." << std::endl;
        return false;
    }

    std::vector<float> gpu_data(gpu_output.shape[0] * gpu_output.shape[1] * gpu_output.shape[2] * gpu_output.shape[3]);
    gpu_output.copyToCPU(gpu_data);

    for (size_t i = 0; i < cpu_output.data.size(); ++i) {
        if (std::abs(cpu_output.data[i] - gpu_data[i]) > tolerance) {
            std::cerr << "Error: Mismatch at index " << i << ". CPU: " << cpu_output.data[i] << ", GPU: " << gpu_data[i] << std::endl;
            return false;
        }
    }

    std::cout << "Verification passed: CPU and GPU outputs match within tolerance." << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 10 || argc > 12) {
        std::cerr << "Usage: " << argv[0] 
                  << " <cuda-block-x> <cuda-block-y> <init-type>"
                  << " <input-channels> <input-height> <input-width>"
                  << " <output-channels> <filter-height> <filter-width>"
                  << " [batch-size] [cpu-cores]" << std::endl;
        return -1;
    }

    int cuda_block_x = std::stoi(argv[1]);
    int cuda_block_y = std::stoi(argv[2]);
    int init_type = std::stoi(argv[3]);
    int input_channels = std::stoi(argv[4]);
    int input_height = std::stoi(argv[5]);
    int input_width = std::stoi(argv[6]);
    int output_channels = std::stoi(argv[7]);
    int filter_height = std::stoi(argv[8]);
    int filter_width = std::stoi(argv[9]);
    int batch_size = (argc >= 11) ? std::stoi(argv[10]) : 1;
    int cpu_cores = (argc == 12) ? std::stoi(argv[11]) : 0;  // 0 means use max available cores

    // Print CLI arguments
    std::cout << "Command-line arguments:" << std::endl;
    std::cout << "CUDA block dimensions: " << cuda_block_x << "x" << cuda_block_y << std::endl;
    std::cout << "Initialization type: " << init_type << std::endl;
    std::cout << "Input dimensions: " << batch_size << "x" << input_channels << "x" << input_height << "x" << input_width << std::endl;
    std::cout << "Filter dimensions: " << output_channels << "x" << input_channels << "x" << filter_height << "x" << filter_width << std::endl;
    std::cout << "CPU cores: " << (cpu_cores == 0 ? "max available" : std::to_string(cpu_cores)) << std::endl;

    // Set a fixed seed for reproducibility
    unsigned int seed = 42;

    std::vector<int> input_shape = {batch_size, input_channels, input_height, input_width};
    std::vector<int> filter_shape = {output_channels, input_channels, filter_height, filter_width};

    // CPU Tensor creation and initialization
    // CPUTensor input_cpu(input_shape);
    // CPUTensor filter_cpu(filter_shape);
    // input_cpu.initialize(init_type, seed);
    // filter_cpu.initialize(2, seed);  // Initialize filter with random values


    // // CPU Convolution with timing
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // CPUTensor output_cpu = convolution2DCPU(input_cpu, filter_cpu, cpu_cores);
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    // printShape("CPU Output", output_cpu.shape);
    // std::cout << "CPU Convolution Time: " << cpu_duration.count() << " ms" << std::endl;


    // GPU Tensor creation and initialization
    GPUTensor input_gpu(input_shape);
    GPUTensor filter_gpu(filter_shape);
    input_gpu.initialize(init_type, seed);
    filter_gpu.initialize(2, seed);  // Initialize filter with random values

    printShape("Input", input_gpu.shape);
    printShape("Filter", filter_gpu.shape);

    

    // GPU Convolution with timing
    int output_height = input_height - filter_height + 1;
    int output_width = input_width - filter_width + 1;
    std::vector<int> output_shape = {batch_size, output_channels, output_height, output_width};
    GPUTensor output_gpu(output_shape);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    convolution2DGPU(input_gpu, filter_gpu, output_gpu, cuda_block_x, cuda_block_y);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;

    
    printShape("GPU Output", output_gpu.shape);

    
    std::cout << "GPU Convolution Time: " << gpu_duration.count() << " ms" << std::endl;

    if (ENABLE_PRINT) {
        // std::cout << "CPU Convolution Output:" << std::endl;
        // output_cpu.print();
        std::cout << "GPU Convolution Output:" << std::endl;
        output_gpu.print();
    }

    // Verify CPU and GPU outputs
    // verifyOutputs(output_cpu, output_gpu);

    return 0;
}