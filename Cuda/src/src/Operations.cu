
#include "Operations.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAddKernel(const float *a, const float *b, float *c, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        c[i] = a[i] + b[i];
    }
}

bool cudaOperation(const float *a, const float *b, float *c, int numElements, float &elapsedTime) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc((void **)&d_a, numElements * sizeof(float));
    cudaMalloc((void **)&d_b, numElements * sizeof(float));
    cudaMalloc((void **)&d_c, numElements * sizeof(float));

    cudaMemcpy(d_a, a, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, numElements * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, numElements);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Check for any errors launching the kernel
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    // Copy the device result vector in device memory to the host result vector
    cudaMemcpy(c, d_c, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory and events
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}




__global__ void runningSumKernel(const float *input, float *output, int numElements) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    for (int j = 0; j <= i && j < numElements; j++) {
        sum += input[j];
    }

    if (i < numElements) {
        output[i] = sum;
    }
}

bool cudaRunningSum(const float *input, float *output, int numElements, float &elapsedTime) {
    float *d_input = nullptr, *d_output = nullptr;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_input, numElements * sizeof(float));
    cudaMalloc((void **)&d_output, numElements * sizeof(float));

    cudaMemcpy(d_input, input, numElements * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    runningSumKernel<<<(numElements + 255) / 256, 256>>>(d_input, d_output, numElements);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(output, d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}