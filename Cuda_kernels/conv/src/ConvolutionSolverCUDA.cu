#include "ConvolutionSolverGPU.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel for 2D convolution
__global__
void convolution2DKernel(float* input, float* filter, float* output, 
                         int batch, int out_channels, int in_channels, 
                         int out_height, int out_width, 
                         int filter_height, int filter_width, 
                         int input_height, int input_width) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow < out_width && oh < out_height && oc < out_channels) {
        for (int b = 0; b < batch; ++b) {
            float sum = 0.0f;
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int kh = 0; kh < filter_height; ++kh) {
                    for (int kw = 0; kw < filter_width; ++kw) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                            sum += input[((b * in_channels + ic) * input_height + ih) * input_width + iw] * 
                                   filter[((oc * in_channels + ic) * filter_height + kh) * filter_width + kw];
                        }
                    }
                }
            }
            output[((b * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
        }
    }
}

// CUDA Convolution function definition
void convolution2DGPU(GPUTensor& input, GPUTensor& filter, GPUTensor& output, int block_x, int block_y) {
    nvtxRangePushA("convolution2DGPU");

    dim3 threadsPerBlock(block_x, block_y, 1);
    dim3 numBlocks((output.shape[3] + block_x - 1) / block_x, 
                   (output.shape[2] + block_y - 1) / block_y, 
                   output.shape[1]);

    nvtxRangePushA("CUDA Kernel Launch");
    convolution2DKernel<<<numBlocks, threadsPerBlock>>>(input.d_data, filter.d_data, output.d_data,
                                                        input.shape[0], output.shape[1], input.shape[1],
                                                        output.shape[2], output.shape[3],
                                                        filter.shape[2], filter.shape[3],
                                                        input.shape[2], input.shape[3]);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePop();
}