#include <gtest/gtest.h>
#include "Tensor.h"
#include "TensorFactory.h"

TEST(CPUTensorTest, Initialization) {
    auto cpuTensor = TensorFactory::createTensor({100, 100}, Device::CPU);
    
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            EXPECT_EQ(cpuTensor->operator[]({i, j}), 0);
        }
    }

    cpuTensor->operator[]({0, 0}) = 1.0f;
    EXPECT_EQ(cpuTensor->operator[]({0, 0}), 1.0f);
}

TEST(CPUTensorTest, AccessOutOfBounds) {
    auto cpuTensor = TensorFactory::createTensor({100, 100}, Device::CPU);
    EXPECT_THROW(cpuTensor->operator[]({100, 100}), std::out_of_range);
}
