#ifndef CUDA_MATH_FUNCTIONS_H
#define CUDA_MATH_FUNCTIONS_H

void forwardCudaSinVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size);
void backwardCudaSinVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size);
void centralCudaSinVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size);

void forwardCudaExponentialVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size);
void backwardCudaExponentialVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size);
void centralCudaExponentialVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size);

void forwardCudaPolynomialVector(const double* x, double h, double* f_x, double* f_x_plus_h, int size);
void backwardCudaPolynomialVector(const double* x, double h, double* f_x, double* f_x_minus_h, int size);
void centralCudaPolynomialVector(const double* x, double h, double* f_x_minus_h, double* f_x_plus_h, int size);

#endif