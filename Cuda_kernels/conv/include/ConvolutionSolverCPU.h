#ifndef CONVOLUTION_SOLVER_CPU_H
#define CONVOLUTION_SOLVER_CPU_H

#include <vector>

// CPU Tensor structure representing multi-dimensional data
struct CPUTensor {
    std::vector<float> data; // Flattened tensor data
    std::vector<int> shape;  // Shape of the tensor [batch, channels, height, width]

    CPUTensor(std::vector<int> shape);
    void initialize(int init_type, unsigned int seed = 0); 
    float& operator()(int b, int c, int h, int w);
    const float& operator()(int b, int c, int h, int w) const;
    void print() const;
};

// CPU Convolution function
CPUTensor convolution2DCPU(const CPUTensor& input, const CPUTensor& filter,int cpu_cores);

#endif // CONVOLUTION_SOLVER_CPU_H
