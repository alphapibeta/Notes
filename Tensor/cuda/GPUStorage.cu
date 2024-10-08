#include "GPUStorage.h"
#include <cuda_runtime.h>

// Constructor: Allocate memory on the GPU
template<typename T>
GPUStorage<T>::GPUStorage(int size) : Storage<T>(size), deviceData(nullptr, CudaDeleter<T>()) {
    T* rawDeviceData;
    cudaMalloc(&rawDeviceData, size * sizeof(T));  // Allocate GPU memory
    deviceData.reset(rawDeviceData);  // Manage GPU memory with smart pointer
}

// Destructor: Automatically frees memory via the CudaDeleter
template<typename T>
GPUStorage<T>::~GPUStorage() {
    // Memory will be freed by CudaDeleter when the unique_ptr goes out of scope
}

// Get pointer to the device data
template<typename T>
T* GPUStorage<T>::getDevicePointer() {
    return deviceData.get();
}

// Explicit template instantiations
template class GPUStorage<float>;
template class GPUStorage<double>;
