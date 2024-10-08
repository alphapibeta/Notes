#include <iostream>
#include "TensorFactory.h"
#include <random>

// Function to initialize the tensor with random values (for both CPU and GPU)
template<typename T, int Rows, int Cols>
void initializeRandomTensor(std::unique_ptr<Tensor<T>>& tensor) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0.0, 1.0); 
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            (*tensor)[{i, j}] = distribution(generator); // Initialize tensor elements
        }
    }
}

void testTensorCreation() {
    constexpr int Rows = 4;
    constexpr int Cols = 4;

    // Create and initialize CPU tensor using the shared library
    auto cpuTensor = TensorFactory<float>::createTensor({Rows, Cols}, Device::CPU);
    initializeRandomTensor<float, Rows, Cols>(cpuTensor); // Initialize CPU tensor with random values
    std::cout << "CPU Tensor (Shared Library):" << std::endl;
    cpuTensor->print();

    // Create and initialize GPU tensor using the shared library
    auto gpuTensor = TensorFactory<float>::createTensor({Rows, Cols}, Device::CUDA);
    initializeRandomTensor<float, Rows, Cols>(gpuTensor); // Initialize GPU tensor with random values
    std::cout << "GPU Tensor (Shared Library):" << std::endl;
    gpuTensor->print();
}

int main() {
    testTensorCreation();
    return 0;
}
