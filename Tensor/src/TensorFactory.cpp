#include "TensorFactory.h"

template<typename T>
std::unique_ptr<Tensor<T>> TensorFactory<T>::createTensor(const std::vector<int>& shape, Device device) {
    if (device == Device::CPU) {
        return std::make_unique<Tensor<T>>(shape);
    } else if (device == Device::CUDA) {
        return std::make_unique<GPUTensor<T>>(shape);
    }
    return nullptr; 
}

// Explicit instantiation for float and double
template class TensorFactory<float>;
template class TensorFactory<double>;
