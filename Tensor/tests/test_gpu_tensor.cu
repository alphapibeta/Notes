#include <gtest/gtest.h>
#include "GPUTensor.h"
#include "TensorFactory.h"

__global__ void setTensorValues(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (float)idx;
    }
}

TEST(GPUTensorTest, Initialization) {
    auto gpuTensor = TensorFactory::createTensor({100, 100}, Device::CUDA);
    

    // float* deviceData = static_cast<GPUStorage*>(gpuTensor->storage.get())->getDevicePointer();
    float* deviceData = static_cast<GPUStorage*>(gpuTensor->getStorage())->getDevicePointer();

    int size = 100 * 100;

    setTensorValues<<<(size + 255) / 256, 256>>>(deviceData, size);
    cudaDeviceSynchronize();

    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            float expectedValue = static_cast<float>(i * 100 + j);
            EXPECT_EQ(gpuTensor->operator[]({i, j}), expectedValue);
        }
    }
}

TEST(GPUTensorTest, AccessOutOfBounds) {
    auto gpuTensor = TensorFactory::createTensor({100, 100}, Device::CUDA);
    
    EXPECT_THROW(gpuTensor->operator[]({100, 100}), std::out_of_range);
}
