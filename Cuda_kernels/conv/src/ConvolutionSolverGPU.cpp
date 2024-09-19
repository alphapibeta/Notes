#include "ConvolutionSolverGPU.h"
#include <random>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

void nvtxStartRange(const char* name, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
}

void nvtxEndRange() {
    nvtxRangePop();
}

GPUTensor::GPUTensor(std::vector<int> shape) : shape(shape) {
    nvtxStartRange("GPUTensor Constructor", 0xFF0000FF);
    cudaMalloc(&d_data, shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float));
    nvtxEndRange();
}

GPUTensor::~GPUTensor() {
    nvtxStartRange("GPUTensor Destructor", 0xFF0000FF);
    if (d_data) {
        cudaFree(d_data);
    }
    nvtxEndRange();
}

void GPUTensor::initialize(int init_type, unsigned int seed) {
    nvtxStartRange("GPUTensor::initialize", 0xFF0000FF);
    std::vector<float> host_data(shape[0] * shape[1] * shape[2] * shape[3]);
    std::mt19937 gen(seed);
    if (init_type == 0) {
        std::fill(host_data.begin(), host_data.end(), 0.0f);
    } else if (init_type == 1) {
        std::fill(host_data.begin(), host_data.end(), 1.0f);
    } else if (init_type == 2) {
        std::uniform_real_distribution<> dis(0, 1);
        for (auto& val : host_data) {
            val = static_cast<float>(dis(gen));
        }
    }
    copyToGPU(host_data);
    nvtxEndRange();
}

void GPUTensor::copyToGPU(const std::vector<float>& host_data) {
    nvtxStartRange("GPUTensor::copyToGPU", 0x00FF00FF);
    cudaMemcpy(d_data, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    nvtxEndRange();
}

void GPUTensor::copyToCPU(std::vector<float>& host_data) const {
    nvtxStartRange("GPUTensor::copyToCPU", 0x0000FFFF);
    cudaMemcpy(host_data.data(), d_data, host_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxEndRange();
}

void GPUTensor::print() const {
    nvtxStartRange("GPUTensor::print", 0xFFFF00FF);
    std::vector<float> host_data(shape[0] * shape[1] * shape[2] * shape[3]);
    copyToCPU(host_data);
    for (int b = 0; b < shape[0]; ++b) {
        for (int c = 0; c < shape[1]; ++c) {
            std::cout << "Batch: " << b << ", Channel: " << c << std::endl;
            for (int h = 0; h < shape[2]; ++h) {
                for (int w = 0; w < shape[3]; ++w) {
                    std::cout << host_data[b * shape[1] * shape[2] * shape[3] +
                                          c * shape[2] * shape[3] +
                                          h * shape[3] + w] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    nvtxEndRange();
}