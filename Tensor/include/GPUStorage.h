#ifndef GPU_STORAGE_H
#define GPU_STORAGE_H

#include "Storage.h"
#include <cuda_runtime.h>
#include <memory>  // For smart pointers

// Generic CudaDeleter that works for any type T (GPU Memory)
template<typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        cudaFree(ptr);
    }
};

// GPUStorage class declaration
template<typename T>
class GPUStorage : public Storage<T> {
public:
    GPUStorage(int size);  // Constructor
    ~GPUStorage();         // Destructor

    T* getDevicePointer();  // Get pointer to device data

private:
    std::unique_ptr<T, CudaDeleter<T>> deviceData;  // Smart pointer for GPU data with the generic deleter
};

#endif  // GPU_STORAGE_H
