#ifndef GPU_STORAGE_H
#define GPU_STORAGE_H

#include "Storage.h"
#include <cuda_runtime.h>
#include <iostream>

template<typename T>
class GPUStorage : public Storage<T> {
public:
    GPUStorage(int size);
    ~GPUStorage();
    T* getDevicePointer();

private:
    T* deviceData; 
};

#endif // GPU_STORAGE_H
