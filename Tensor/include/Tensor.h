#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include "Storage.h"
#include "GPUStorage.h" 
#include "Device.h"

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);  // Constructor
    T& operator[](const std::vector<int>& indices);  // Element access
    const T& operator[](const std::vector<int>& indices) const;  // Const element access
    void print() const;  // Print the tensor
    virtual ~Tensor() = default;  // Virtual destructor

protected:
    std::vector<int> shape;
    std::unique_ptr<Storage<T>> storage;  // Smart pointer for storage
    Device device;
    int size;

    int computeSize(const std::vector<int>& shape);  // Compute tensor size
    int getFlatIndex(const std::vector<int>& indices) const;  // Compute flat index
};

#endif  // TENSOR_H
