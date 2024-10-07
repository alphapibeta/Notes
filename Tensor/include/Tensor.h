#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include "Storage.h"
#include "GPUStorage.h" // Include GPUStorage here
#include "Device.h"

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);
    T& operator[](const std::vector<int>& indices);
    const T& operator[](const std::vector<int>& indices) const;
    void print() const;
    virtual ~Tensor() = default;
    Storage<T>* getStorage() const {
        return storage.get();
    }

protected:
    std::vector<int> shape;
    std::unique_ptr<Storage<T>> storage; 
    Device device;
    int size;
    int computeSize(const std::vector<int>& shape);
    int getFlatIndex(const std::vector<int>& indices) const;
};

#endif // TENSOR_H
