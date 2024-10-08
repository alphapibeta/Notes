#include "Tensor.h"
#include <iostream>
#include <numeric>
#include <stdexcept>

// Constructor: Allocates storage depending on the device (CPU or GPU)
template<typename T>
Tensor<T>::Tensor(const std::vector<int>& shape, Device device)
    : shape(shape), device(device) {
    size = computeSize(shape);

    // Allocate storage based on the device
    if (device == Device::CPU) {
        storage = std::make_unique<Storage<T>>(size);  // CPU memory
    } else if (device == Device::CUDA) {
        storage = std::make_unique<GPUStorage<T>>(size);  // GPU memory
    }

    storage->fill(0);  // Initialize tensor values to 0
}

// Compute the total size of the tensor
template<typename T>
int Tensor<T>::computeSize(const std::vector<int>& shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// Access tensor element using multi-dimensional indices
template<typename T>
T& Tensor<T>::operator[](const std::vector<int>& indices) {
    return storage->operator[](getFlatIndex(indices));
}

// Const access to tensor element
template<typename T>
const T& Tensor<T>::operator[](const std::vector<int>& indices) const {
    return storage->operator[](getFlatIndex(indices));
}

// Print the tensor values
template<typename T>
void Tensor<T>::print() const {
    for (int i = 0; i < size; ++i) {
        std::cout << storage->operator[](i) << " ";
        if ((i + 1) % shape[1] == 0) {
            std::cout << std::endl;
        }
    }
}

// Get the flat index for multi-dimensional indices
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

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;
