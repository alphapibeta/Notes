#ifndef HESSIAN_INVERSION_GPU_H
#define HESSIAN_INVERSION_GPU_H

#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

template <typename T>
class HessianInversionGPU {
public:
    HessianInversionGPU(int size);
    ~HessianInversionGPU();
    
    void setMatrix(const std::vector<T>& matrix);
    std::vector<T> getInverse() const;
    void invert();
    void regularizeMatrix(T epsilon = 1e-6);
    void printMatrix(const std::vector<T>& matrix) const;

    double getGPUBandwidth() const;
    double getGPUComputationalThroughput() const;
    double getTheoreticalGPUBandwidth() const;
    double getTheoreticalGPUComputationalThroughput() const;
    double getGPUMemoryThroughput() const;
    double getArithmeticIntensity() const;

private:
    int size;
    std::vector<T> matrix;
    std::vector<T> inverse;
    T* d_matrix;
    T* d_inverse;
    int* d_ipiv;
    int* d_info;
    cusolverDnHandle_t cusolverH;
    
    void allocateMemory();
    void freeMemory();

    double gpu_bandwidth;
    double gpu_computational_throughput;
    double theoretical_gpu_bandwidth;
    double theoretical_gpu_computational_throughput;
    double gpu_memory_throughput;
    double arithmetic_intensity;
};

#endif