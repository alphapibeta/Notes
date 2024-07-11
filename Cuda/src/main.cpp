#include <iostream>
#include "Operations.h"

int main() {
    int numElements = 2<<8; // size of the vectors
    float *a = new float[numElements];
    float *b = new float[numElements];
    float *c = new float[numElements];
    int su =numElements;
    float *input = new float[su];
    float *output = new float[su];
    float elapsedTime;

    for (int i = 0; i < numElements; ++i) {
        a[i] = float(i);
        b[i] = float(i);
    }
    for (int i = 0; i < su; ++i) {
        input[i] = float(i);
    }

    if (cudaOperation(a, b, c, numElements, elapsedTime)) {
        std::cout << "Vector addition completed successfully. Time elapsed: " << elapsedTime << " ms" << std::endl;
    } else {
        std::cerr << "CUDA operation failed." << std::endl;
        return 1;
    }

    if (cudaRunningSum(input, output, su, elapsedTime)) {
        std::cout << "Running sum computed successfully. Time elapsed: " << elapsedTime << " ms" << std::endl;
    } else {
        std::cerr << "CUDA operation failed." << std::endl;
        return 1;
    }



    delete[] a;
    delete[] b;
    delete[] c;
    delete[] output;
    return 0;
}