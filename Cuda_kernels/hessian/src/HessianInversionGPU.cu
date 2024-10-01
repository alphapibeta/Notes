#include "HessianInversionGPU.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at line " << __LINE__ << ": " << cudaGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER Error at line " << __LINE__ << ": " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

template <typename T>
HessianInversionGPU<T>::HessianInversionGPU(int size) : size(size), gpu_bandwidth(0), gpu_computational_throughput(0), gpu_memory_throughput(0), arithmetic_intensity(0) {
    matrix.resize(size * size);
    inverse.resize(size * size);
    allocateMemory();
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    
    // Calculate theoretical GPU bandwidth and computational throughput
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    theoretical_gpu_bandwidth = prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) * 2 / 1.0e9;
    theoretical_gpu_computational_throughput = prop.clockRate * 1000.0 * prop.multiProcessorCount * 32 * 2 / 1.0e9;
}

template <typename T>
HessianInversionGPU<T>::~HessianInversionGPU() {
    freeMemory();
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
}

template <typename T>
void HessianInversionGPU<T>::allocateMemory() {
    CHECK_CUDA(cudaMalloc(&d_matrix, size * size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_inverse, size * size * sizeof(T)));
    CHECK_CUDA(cudaMalloc(&d_ipiv, size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
}

template <typename T>
void HessianInversionGPU<T>::freeMemory() {
    CHECK_CUDA(cudaFree(d_matrix));
    CHECK_CUDA(cudaFree(d_inverse));
    CHECK_CUDA(cudaFree(d_ipiv));
    CHECK_CUDA(cudaFree(d_info));
}

template <typename T>
void HessianInversionGPU<T>::setMatrix(const std::vector<T>& matrix) {
    this->matrix = matrix;
    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(), size * size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> HessianInversionGPU<T>::getInverse() const {
    return inverse;
}

template <typename T>
void HessianInversionGPU<T>::regularizeMatrix(T epsilon) {
    for (int i = 0; i < size; ++i) {
        matrix[i * size + i] += epsilon;
    }
    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(), size * size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void HessianInversionGPU<T>::printMatrix(const std::vector<T>& matrix) const {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << matrix[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <>
void HessianInversionGPU<float>::invert() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int lwork = 0;
    float* d_work = nullptr;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolverH, size, size, d_matrix, size, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    CHECK_CUSOLVER(cusolverDnSgetrf(cusolverH, size, size, d_matrix, size, d_work, d_ipiv, d_info));

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "LU factorization failed: " << h_info << std::endl;
        exit(1);
    }

    std::vector<float> h_identity(size * size, 0.0f);
    for (int i = 0; i < size; ++i) {
        h_identity[i * size + i] = 1.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_inverse, h_identity.data(), size * size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUSOLVER(cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, size, size, d_matrix, size, d_ipiv, d_inverse, size, d_info));

    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "Matrix inversion failed: " << h_info << std::endl;
        exit(1);
    }

    CHECK_CUDA(cudaMemcpy(inverse.data(), d_inverse, size * size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_work));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate bandwidth and computational throughput
    double seconds = milliseconds / 1000.0;
    double flops = 2.0 * size * size * size; // Approximate FLOPs for matrix inversion
    double bytes_transferred = size * size * sizeof(float) * 2; // Read and write
    gpu_memory_throughput = bytes_transferred / seconds / 1.0e9; // GB/s
    gpu_computational_throughput = flops / seconds / 1.0e9; // GFLOP/s
    arithmetic_intensity = flops / bytes_transferred;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <>
void HessianInversionGPU<double>::invert() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int lwork = 0;
    double* d_work = nullptr;
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverH, size, size, d_matrix, size, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
    CHECK_CUSOLVER(cusolverDnDgetrf(cusolverH, size, size, d_matrix, size, d_work, d_ipiv, d_info));

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "LU factorization failed: " << h_info << std::endl;
        exit(1);
    }

    std::vector<double> h_identity(size * size, 0.0);
    for (int i = 0; i < size; ++i) {
        h_identity[i * size + i] = 1.0;
    }
    CHECK_CUDA(cudaMemcpy(d_inverse, h_identity.data(), size * size * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUSOLVER(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, size, size, d_matrix, size, d_ipiv, d_inverse, size, d_info));

    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "Matrix inversion failed: " << h_info << std::endl;
        exit(1);
    }

    CHECK_CUDA(cudaMemcpy(inverse.data(), d_inverse, size * size * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_work));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate bandwidth and computational throughput
    double seconds = milliseconds / 1000.0;
    double flops = 2.0 * size * size * size; // Approximate FLOPs for matrix inversion
    double bytes_transferred = size * size * sizeof(double) * 2; // Read and write
    gpu_memory_throughput = bytes_transferred / seconds / 1.0e9; // GB/s
    gpu_computational_throughput = flops / seconds / 1.0e9; // GFLOP/s
    arithmetic_intensity = flops / bytes_transferred;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename T>
double HessianInversionGPU<T>::getGPUBandwidth() const {
    return gpu_memory_throughput;  // Changed from gpu_bandwidth to gpu_memory_throughput
}

template <typename T>
double HessianInversionGPU<T>::getGPUComputationalThroughput() const {
    return gpu_computational_throughput;
}

template <typename T>
double HessianInversionGPU<T>::getTheoreticalGPUBandwidth() const {
    return theoretical_gpu_bandwidth;
}

template <typename T>
double HessianInversionGPU<T>::getTheoreticalGPUComputationalThroughput() const {
    return theoretical_gpu_computational_throughput;
}

template <typename T>
double HessianInversionGPU<T>::getGPUMemoryThroughput() const {
    return gpu_memory_throughput;
}

template <typename T>
double HessianInversionGPU<T>::getArithmeticIntensity() const {
    return arithmetic_intensity;
}

// Explicit instantiations
template class HessianInversionGPU<float>;
template class HessianInversionGPU<double>;