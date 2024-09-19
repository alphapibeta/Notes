#include "ConvolutionSolverCPU.h"
#include <random>
#include <iostream>
#include <omp.h>

CPUTensor::CPUTensor(std::vector<int> shape) : shape(shape) {
    data.resize(shape[0] * shape[1] * shape[2] * shape[3], 0.0f);
}

void CPUTensor::initialize(int init_type, unsigned int seed) {
    std::mt19937 gen(seed);
    if (init_type == 0) {
        std::fill(data.begin(), data.end(), 0.0f);
    } else if (init_type == 1) {
        std::fill(data.begin(), data.end(), 1.0f);
    } else if (init_type == 2) {
        std::uniform_real_distribution<> dis(0, 1);
        for (auto& val : data) {
            val = static_cast<float>(dis(gen));
        }
    }
}

float& CPUTensor::operator()(int b, int c, int h, int w) {
    int index = b * shape[1] * shape[2] * shape[3] +
                c * shape[2] * shape[3] +
                h * shape[3] + w;
    return data[index];
}

const float& CPUTensor::operator()(int b, int c, int h, int w) const {
    int index = b * shape[1] * shape[2] * shape[3] +
                c * shape[2] * shape[3] +
                h * shape[3] + w;
    return data[index];
}

void CPUTensor::print() const {
    for (int b = 0; b < shape[0]; ++b) {
        for (int c = 0; c < shape[1]; ++c) {
            std::cout << "Batch: " << b << ", Channel: " << c << std::endl;
            for (int h = 0; h < shape[2]; ++h) {
                for (int w = 0; w < shape[3]; ++w) {
                    std::cout << data[b * shape[1] * shape[2] * shape[3] +
                                      c * shape[2] * shape[3] +
                                      h * shape[3] + w] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

CPUTensor convolution2DCPU(const CPUTensor& input, const CPUTensor& filter, int num_cores) {
    int out_height = input.shape[2] - filter.shape[2] + 1;
    int out_width = input.shape[3] - filter.shape[3] + 1;
    CPUTensor output({input.shape[0], filter.shape[0], out_height, out_width});

    if (num_cores <= 0) {
        num_cores = omp_get_max_threads();
    }
    omp_set_num_threads(num_cores);

    #pragma omp parallel for collapse(4)
    for (int b = 0; b < input.shape[0]; ++b) {
        for (int oc = 0; oc < filter.shape[0]; ++oc) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < input.shape[1]; ++ic) {
                        for (int kh = 0; kh < filter.shape[2]; ++kh) {
                            for (int kw = 0; kw < filter.shape[3]; ++kw) {
                                int ih = oh + kh;
                                int iw = ow + kw;
                                sum += input(b, ic, ih, iw) * filter(oc, ic, kh, kw);
                            }
                        }
                    }
                    output(b, oc, oh, ow) = sum;
                }
            }
        }
    }
    return output;
}