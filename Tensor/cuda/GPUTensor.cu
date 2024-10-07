#include "GPUTensor.h"
#include <iostream>

template<typename T>
GPUTensor<T>::GPUTensor(const std::vector<int>& shape) : Tensor<T>(shape, Device::CUDA) {
    this->storage = std::make_unique<GPUStorage<T>>(this->size);
}

template<typename T>
void GPUTensor<T>::print() const {
    std::cout << "Tensor on GPU:" << std::endl;
    Tensor<T>::print();
}

// Explicit instantiation for float and double
template class GPUTensor<float>;
template class GPUTensor<double>;
