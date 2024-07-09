#include <iostream>
#include "CudaMathFunctions.h"
#include "nvToolsExt.h"
// #include "MathFunctions.h"
#include "Topology.h"
int main() {


    std::set<int> X = {1, 2, 3};
    std::vector<std::set<int>> openSets = {
        {}, {1}, {1, 2}, {1, 2, 3}
    };
    Topology topology(X, openSets);

    std::set<int> testSet = {1, 2};
    std::cout << "Is {1, 2} an open set? " << (topology.is_open_set(testSet) ? "Yes" : "No") << std::endl;

    // Add any numerical differentiation functionality here if needed


    const int size = 10;
    double xs[size], h = 0.01;
    double sin_values[size], sin_values_plus_h[size], sin_values_minus_h[size];
    double exp_values[size], exp_values_plus_h[size], exp_values_minus_h[size];
    double poly_values[size], poly_values_plus_h[size], poly_values_minus_h[size];
    double derivatives[size];


    for (int i = 0; i < size; i++) {
        xs[i] = 2;  
    }

    forwardCudaSinVector(xs, h, sin_values, sin_values_plus_h, size);
    forwardCudaExponentialVector(xs, h, exp_values, exp_values_plus_h, size);
    forwardCudaPolynomialVector(xs, h, poly_values, poly_values_plus_h, size);

    backwardCudaSinVector(xs, h, sin_values, sin_values_minus_h, size);
    backwardCudaExponentialVector(xs, h, exp_values, exp_values_minus_h, size);
    backwardCudaPolynomialVector(xs, h, poly_values, poly_values_minus_h, size);


    centralCudaSinVector(xs, h, sin_values_minus_h, sin_values_plus_h, size);
    centralCudaExponentialVector(xs, h, exp_values_minus_h, exp_values_plus_h, size);
    centralCudaPolynomialVector(xs, h, poly_values_minus_h, poly_values_plus_h, size);

    for (int i = 0; i < size; i++) {
        double forward_sin_derivative = (sin_values_plus_h[i] - sin_values[i]) / h;
        double forward_exp_derivative = (exp_values_plus_h[i] - exp_values[i]) / h;
        double forward_poly_derivative = (poly_values_plus_h[i] - poly_values[i]) / h;
        std::cout << "Forward x: " << xs[i]
                  << " Sin Derivative: " << forward_sin_derivative
                  << ", Exp Derivative: " << forward_exp_derivative
                  << ", Poly Derivative: " << forward_poly_derivative << std::endl;
    }

    for (int i = 0; i < size; i++) {
        double backward_sin_derivative = (sin_values[i] - sin_values_minus_h[i]) / h;
        double backward_exp_derivative = (exp_values[i] - exp_values_minus_h[i]) / h;
        double backward_poly_derivative = (poly_values[i] - poly_values_minus_h[i]) / h;
        std::cout << "Backward x: " << xs[i]
                  << " Sin Derivative: " << backward_sin_derivative
                  << ", Exp Derivative: " << backward_exp_derivative
                  << ", Poly Derivative: " << backward_poly_derivative << std::endl;
    }

    for (int i = 0; i < size; i++) {
        double central_sin_derivative = (sin_values_plus_h[i] - sin_values_minus_h[i]) / (2 * h);
        double central_exp_derivative = (exp_values_plus_h[i] - exp_values_minus_h[i]) / (2 * h);
        double central_poly_derivative = (poly_values_plus_h[i] - poly_values_minus_h[i]) / (2 * h);
        std::cout << "Central x: " << xs[i]
                  << " Sin Derivative: " << central_sin_derivative
                  << ", Exp Derivative: " << central_exp_derivative
                  << ", Poly Derivative: " << central_poly_derivative << std::endl;
    }

    return 0;
}


