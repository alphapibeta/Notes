#ifndef HESSIAN_INVERSION_CPU_H
#define HESSIAN_INVERSION_CPU_H

#include <vector>
#include <iostream>
#include <omp.h>

template <typename T>
class HessianInversionCPU {
public:
    HessianInversionCPU(int size);
    void setMatrix(const std::vector<std::vector<T>>& matrix);
    std::vector<std::vector<T>> getInverse() const;
    void invert();
    void regularizeMatrix(T epsilon = 1e-6);
    void printMatrix(const std::vector<std::vector<T>>& matrix) const;

private:
    int size;
    std::vector<std::vector<T>> matrix;
    std::vector<std::vector<T>> inverse;
    void luDecompose(std::vector<std::vector<T>>& L, std::vector<std::vector<T>>& U);
    std::vector<T> forwardSubstitution(const std::vector<std::vector<T>>& L, const std::vector<T>& b);
    std::vector<T> backwardSubstitution(const std::vector<std::vector<T>>& U, const std::vector<T>& y);
};

#include "HessianInversionCPU.tpp"

#endif