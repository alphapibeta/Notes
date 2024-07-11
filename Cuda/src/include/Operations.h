#ifndef OPERATIONS_H
#define OPERATIONS_H

bool cudaOperation(const float *a, const float *b, float *c, int numElements, float &elapsedTime);
bool cudaRunningSum(const float *input, float *output, int numElements, float &elapsedTime);

#endif // OPERATIONS_H