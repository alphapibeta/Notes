#include "GPUStorage.h"

template<typename T>
GPUStorage<T>::GPUStorage(int size) : Storage<T>(size) {
    cudaMalloc(&deviceData, size * sizeof(T));
}

template<typename T>
GPUStorage<T>::~GPUStorage() {
    cudaFree(deviceData);
}

template<typename T>
T* GPUStorage<T>::getDevicePointer() {
    return deviceData;
}

// Explicit instantiation for float and double
template class GPUStorage<float>;
template class GPUStorage<double>;
