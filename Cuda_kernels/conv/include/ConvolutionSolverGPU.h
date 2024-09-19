#ifndef CONVOLUTION_SOLVER_GPU_H
#define CONVOLUTION_SOLVER_GPU_H

#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

struct GPUTensor {
    std::vector<int> shape;  
    float* d_data = nullptr; 
    GPUTensor(std::vector<int> shape);
    ~GPUTensor();
    void initialize(int init_type, unsigned int seed = 0); 
    void copyToGPU(const std::vector<float>& host_data);
    void copyToCPU(std::vector<float>& host_data) const;
    void print() const;
};

void convolution2DGPU(GPUTensor& input, GPUTensor& filter, GPUTensor& output, int block_x, int block_y);

// NVTX marker functions
void nvtxStartRange(const char* name, uint32_t color);
void nvtxEndRange();

#endif