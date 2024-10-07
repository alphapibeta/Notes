#ifndef TENSOR_FACTORY_H
#define TENSOR_FACTORY_H

#include "Tensor.h"
#include "GPUTensor.h"
#include "Device.h" 
#include <memory>

template<typename T>
class TensorFactory {
public:
    static std::unique_ptr<Tensor<T>> createTensor(const std::vector<int>& shape, Device device);
};

#endif // TENSOR_FACTORY_H
