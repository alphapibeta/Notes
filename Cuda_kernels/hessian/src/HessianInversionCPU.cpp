// #include "HessianInversionCPU.h"

// // Constructor to initialize the matrix and size
// HessianInversionCPU::HessianInversionCPU(int size) : size(size), matrix(size, std::vector<float>(size)), inverse(size, std::vector<float>(size)) {}

// // Set the input matrix
// void HessianInversionCPU::setMatrix(const std::vector<std::vector<float>>& matrix) {
//     this->matrix = matrix;
// }

// // Get the inverted matrix
// std::vector<std::vector<float>> HessianInversionCPU::getInverse() const {
//     return inverse;
// }

// // Perform LU decomposition and invert the matrix
// void HessianInversionCPU::invert() {
//     std::vector<std::vector<float>> L(size, std::vector<float>(size, 0));
//     std::vector<std::vector<float>> U(size, std::vector<float>(size, 0));

//     // Perform LU decomposition
//     luDecompose(L, U);

//     // Invert the matrix by solving linear systems for each column
//     for (int i = 0; i < size; ++i) {
//         std::vector<float> e(size, 0);
//         e[i] = 1;
//         std::vector<float> y = forwardSubstitution(L, e);
//         std::vector<float> x = backwardSubstitution(U, y);
//         for (int j = 0; j < size; ++j) {
//             inverse[j][i] = x[j];
//         }
//     }
// }

// // Regularize the matrix by adding a small value to the diagonal
// void HessianInversionCPU::regularizeMatrix(float epsilon) {
//     for (int i = 0; i < size; ++i) {
//         matrix[i][i] += epsilon;
//     }
// }

// // Print the matrix (for debugging purposes)
// void HessianInversionCPU::printMatrix(const std::vector<std::vector<float>>& matrix) const {
//     for (const auto& row : matrix) {
//         for (float val : row) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }
// }

// // LU decomposition
// void HessianInversionCPU::luDecompose(std::vector<std::vector<float>>& L, std::vector<std::vector<float>>& U) {
//     for (int i = 0; i < size; i++) {
//         for (int j = 0; j < size; j++) {
//             if (j < i)
//                 L[j][i] = 0;
//             else {
//                 L[j][i] = matrix[j][i];
//                 for (int k = 0; k < i; k++) {
//                     L[j][i] -= L[j][k] * U[k][i];
//                 }
//             }
//         }
//         for (int j = 0; j < size; j++) {
//             if (j < i)
//                 U[i][j] = 0;
//             else if (j == i)
//                 U[i][j] = 1;
//             else {
//                 U[i][j] = matrix[i][j] / L[i][i];
//                 for (int k = 0; k < i; k++) {
//                     U[i][j] -= ((L[i][k] * U[k][j]) / L[i][i]);
//                 }
//             }
//         }
//     }
// }

// // Forward substitution
// std::vector<float> HessianInversionCPU::forwardSubstitution(const std::vector<std::vector<float>>& L, const std::vector<float>& b) {
//     std::vector<float> y(size, 0);
//     for (int i = 0; i < size; ++i) {
//         y[i] = b[i];
//         for (int j = 0; j < i; ++j) {
//             y[i] -= L[i][j] * y[j];
//         }
//         y[i] /= L[i][i];
//     }
//     return y;
// }

// // Backward substitution
// std::vector<float> HessianInversionCPU::backwardSubstitution(const std::vector<std::vector<float>>& U, const std::vector<float>& y) {
//     std::vector<float> x(size, 0);
//     for (int i = size - 1; i >= 0; --i) {
//         x[i] = y[i];
//         for (int j = i + 1; j < size; ++j) {
//             x[i] -= U[i][j] * x[j];
//         }
//         x[i] /= U[i][i];
//     }
//     return x;
// }
