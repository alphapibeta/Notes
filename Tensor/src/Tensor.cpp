#include "Tensor.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& shape, Device device)
    : shape(shape), device(device) {
    size = computeSize(shape);
    if (device == Device::CPU) {
        storage = std::make_unique<Storage<T>>(size);
    } else {
        storage = std::make_unique<GPUStorage<T>>(size); // Ensure GPUStorage is recognized
    }
    storage->fill(0);
}

template<typename T>
int Tensor<T>::computeSize(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template<typename T>
T& Tensor<T>::operator[](const std::vector<int>& indices) {
    return storage->operator[](getFlatIndex(indices));
}

template<typename T>
const T& Tensor<T>::operator[](const std::vector<int>& indices) const {
    return storage->operator[](getFlatIndex(indices));
}

template<typename T>
int Tensor<T>::getFlatIndex(const std::vector<int>& indices) const {
    int flatIndex = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        flatIndex += indices[i] * stride;
        stride *= shape[i];
    }
    return flatIndex;
}

template<typename T>
void Tensor<T>::print() const {
    for (int i = 0; i < size; ++i) {
        std::cout << storage->operator[](i) << " ";
        if ((i + 1) % shape[1] == 0) {
            std::cout << std::endl;
        }
    }
}

// Explicit instantiation for float and double
template class Tensor<float>;
template class Tensor<double>;
