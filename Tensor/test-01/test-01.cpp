#include <iostream>
#include "TensorFactory.h"

void testTensorCreation() {
    constexpr int Rows = 4;
    constexpr int Cols = 4;

    // Create a CPU tensor using the static library
    auto cpuTensorStatic = TensorFactory<float>::createTensor({Rows, Cols}, Device::CPU);
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            (*cpuTensorStatic)[{i, j}] = static_cast<float>(i * Cols + j);
        }
    }
    std::cout << "Static Library - CPU Tensor:" << std::endl;
    cpuTensorStatic->print();

    // Create a GPU tensor using the static library
    auto gpuTensorStatic = TensorFactory<float>::createTensor({Rows, Cols}, Device::CUDA);
    std::cout << "Static Library - GPU Tensor:" << std::endl;
    gpuTensorStatic->print();

    // Create a CPU tensor using the shared library
    auto cpuTensorShared = TensorFactory<float>::createTensor({Rows, Cols}, Device::CPU);
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            (*cpuTensorShared)[{i, j}] = static_cast<float>(i * Cols + j + 10);  // Different values
        }
    }
    std::cout << "Shared Library - CPU Tensor:" << std::endl;
    cpuTensorShared->print();

    // Create a GPU tensor using the shared library
    auto gpuTensorShared = TensorFactory<float>::createTensor({Rows, Cols}, Device::CUDA);
    std::cout << "Shared Library - GPU Tensor:" << std::endl;
    gpuTensorShared->print();
}

int main() {
    testTensorCreation();
    return 0;
}
