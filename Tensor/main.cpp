#include "TensorFactory.h"
#include <iostream>
#include <random>

template<typename T, int Rows, int Cols>
void initializeRandomTensor(std::unique_ptr<Tensor<T>>& tensor) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0.0, 1.0); 
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            for (int k = 0; k < Rows; ++k){
            (*tensor)[{i, j,k}] = distribution(generator);
        }
    }
}
}

template<typename T, int Rows, int Cols>
void createAndPrintTensors() {
    // Create and initialize CPU tensor
    auto cpuTensor = TensorFactory<T>::createTensor({Rows, Cols, Rows}, Device::CPU);
    initializeRandomTensor<T, Rows, Cols>(cpuTensor);
    std::cout << "CPU Tensor (" << typeid(T).name() << "):" << std::endl;
    cpuTensor->print();

    // Create and initialize GPU tensor
    auto gpuTensor = TensorFactory<T>::createTensor({Rows, Cols,Rows}, Device::CUDA);
    initializeRandomTensor<T, Rows, Cols>(gpuTensor);
    std::cout << "GPU Tensor (" << typeid(T).name() << "):" << std::endl;
    gpuTensor->print();
}

int main() {
    constexpr int Rows = 4;
    constexpr int Cols = 4;

    // Create and print tensors for float
    createAndPrintTensors<float, Rows, Cols>();

    // Create and print tensors for double
    createAndPrintTensors<double, Rows, Cols>();

    return 0;
}
