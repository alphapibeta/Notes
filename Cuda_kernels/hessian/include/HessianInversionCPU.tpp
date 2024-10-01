#ifndef HESSIAN_INVERSION_CPU_TPP
#define HESSIAN_INVERSION_CPU_TPP

#include <cmath>
#include <omp.h>

template <typename T>
HessianInversionCPU<T>::HessianInversionCPU(int size) : size(size), matrix(size, std::vector<T>(size)), inverse(size, std::vector<T>(size)) {}

template <typename T>
void HessianInversionCPU<T>::setMatrix(const std::vector<std::vector<T>>& matrix) {
    this->matrix = matrix;
}

template <typename T>
std::vector<std::vector<T>> HessianInversionCPU<T>::getInverse() const {
    return inverse;
}

template <typename T>
void HessianInversionCPU<T>::invert() {
    std::vector<std::vector<T>> L(size, std::vector<T>(size, 0));
    std::vector<std::vector<T>> U(size, std::vector<T>(size, 0));
    luDecompose(L, U);

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        std::vector<T> e(size, 0);
        e[i] = 1;
        std::vector<T> y = forwardSubstitution(L, e);
        std::vector<T> x = backwardSubstitution(U, y);
        for (int j = 0; j < size; ++j) {
            inverse[j][i] = x[j];
        }
    }
}

template <typename T>
void HessianInversionCPU<T>::regularizeMatrix(T epsilon) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        matrix[i][i] += epsilon;
    }
}

template <typename T>
void HessianInversionCPU<T>::printMatrix(const std::vector<std::vector<T>>& matrix) const {
    for (const auto& row : matrix) {
        for (T val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void HessianInversionCPU<T>::luDecompose(std::vector<std::vector<T>>& L, std::vector<std::vector<T>>& U) {
    for (int i = 0; i < size; i++) {
        #pragma omp parallel for
        for (int j = 0; j < size; j++) {
            if (j < i)
                L[j][i] = 0;
            else {
                L[j][i] = matrix[j][i];
                for (int k = 0; k < i; k++) {
                    L[j][i] -= L[j][k] * U[k][i];
                }
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < size; j++) {
            if (j < i)
                U[i][j] = 0;
            else if (j == i)
                U[i][j] = 1;
            else {
                U[i][j] = matrix[i][j] / L[i][i];
                for (int k = 0; k < i; k++) {
                    U[i][j] -= ((L[i][k] * U[k][j]) / L[i][i]);
                }
            }
        }
    }
}

template <typename T>
std::vector<T> HessianInversionCPU<T>::forwardSubstitution(const std::vector<std::vector<T>>& L, const std::vector<T>& b) {
    std::vector<T> y(size, 0);
    for (int i = 0; i < size; ++i) {
        y[i] = b[i];
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] -= sum;
        y[i] /= L[i][i];
    }
    return y;
}

template <typename T>
std::vector<T> HessianInversionCPU<T>::backwardSubstitution(const std::vector<std::vector<T>>& U, const std::vector<T>& y) {
    std::vector<T> x(size, 0);
    for (int i = size - 1; i >= 0; --i) {
        x[i] = y[i];
        T sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int j = i + 1; j < size; ++j) {
            sum += U[i][j] * x[j];
        }
        x[i] -= sum;
        x[i] /= U[i][i];
    }
    return x;
}

#endif