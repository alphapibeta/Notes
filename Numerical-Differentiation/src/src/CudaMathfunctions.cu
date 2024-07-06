#include "CudaMathFunctions.h"
#include <cmath>
#include <cuda_runtime.h>
#include "nvToolsExt.h"
// Sine Kernels
__global__ void forwardSinKernel(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x[idx] = sin(x[idx]);
        f_x_plus_h[idx] = sin(x[idx] + h);
    }
}

__global__ void backwardSinKernel(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x[idx] = sin(x[idx]);
        f_x_minus_h[idx] = sin(x[idx] - h);
    }
}

__global__ void centralSinKernel(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x_minus_h[idx] = sin(x[idx] - h);
        f_x_plus_h[idx] = sin(x[idx] + h);
    }
}

// Exponential Kernels
__global__ void forwardExponentialKernel(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x[idx] = exp(x[idx]);
        f_x_plus_h[idx] = exp(x[idx] + h);
    }
}

__global__ void backwardExponentialKernel(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x[idx] = exp(x[idx]);
        f_x_minus_h[idx] = exp(x[idx] - h);
    }
}

__global__ void centralExponentialKernel(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        f_x_minus_h[idx] = exp(x[idx] - h);
        f_x_plus_h[idx] = exp(x[idx] + h);
    }
}

// Polynomial Kernels
__global__ void forwardPolynomialKernel(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        double x_val = x[idx];
        f_x[idx] = x_val * x_val - 4 * x_val + 4;
        f_x_plus_h[idx] = (x_val + h) * (x_val + h) - 4 * (x_val + h) + 4;
    }
}

__global__ void backwardPolynomialKernel(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        double x_val = x[idx];
        f_x[idx] = x_val * x_val - 4 * x_val + 4;
        f_x_minus_h[idx] = (x_val - h) * (x_val - h) - 4 * (x_val - h) + 4;
    }
}

__global__ void centralPolynomialKernel(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        double x_val = x[idx] - h;
        f_x_minus_h[idx] = x_val * x_val - 4 * x_val + 4;
        x_val = x[idx] + h;
        f_x_plus_h[idx] = x_val * x_val - 4 * x_val + 4;
    }
}



void forwardCudaSinVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Forward Sine Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    forwardSinKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_plus_h);
}

void backwardCudaSinVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_minus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Backward Sine Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    backwardSinKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_minus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_minus_h);
}

void centralCudaSinVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x_minus_h, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Central Sine Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    centralSinKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x_minus_h, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x_minus_h);
    cudaFree(d_f_x_plus_h);
}


void backwardCudaExponentialVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_minus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Backward Exponential Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    backwardExponentialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_minus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_minus_h);
}


void centralCudaExponentialVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x_minus_h, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Central Exponential Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    centralExponentialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x_minus_h, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x_minus_h);
    cudaFree(d_f_x_plus_h);
}

void backwardCudaPolynomialVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_minus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Backward Polynomial Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    backwardPolynomialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_minus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_minus_h);
}


void centralCudaPolynomialVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x_minus_h, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x_minus_h, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Central Polynomial Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    centralPolynomialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x_minus_h, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x_minus_h, d_f_x_minus_h, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x_minus_h);
    cudaFree(d_f_x_plus_h);
}


void forwardCudaPolynomialVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Forward Polynomial Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    forwardPolynomialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_plus_h);
}


void forwardCudaExponentialVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size) {
    double *d_x, *d_f_x, *d_f_x_plus_h;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_f_x, size * sizeof(double));
    cudaMalloc(&d_f_x_plus_h, size * sizeof(double));

    cudaMemcpy(d_x, x, size * sizeof(double), cudaMemcpyHostToDevice);

    nvtxRangePush("Forward Exponential Kernel Launch");
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    forwardExponentialKernel<<<numBlocks, blockSize>>>(d_x, h, d_f_x, d_f_x_plus_h, size);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaMemcpy(f_x, d_f_x, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_x_plus_h, d_f_x_plus_h, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_x);
    cudaFree(d_f_x_plus_h);
}
