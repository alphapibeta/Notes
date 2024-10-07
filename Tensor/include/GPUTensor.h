#ifndef GPU_TENSOR_H
#define GPU_TENSOR_H

#include "Tensor.h"
#include "GPUStorage.h"

template<typename T>
class GPUTensor : public Tensor<T> {
public:
    GPUTensor(const std::vector<int>& shape);
    void print() const; 
};

#endif // GPU_TENSOR_H
