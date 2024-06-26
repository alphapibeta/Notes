#include <iostream>
#include <cmath>
#include "ForwardDerivative.h"
#include "BackwardDerivative.h"
#include "CentralDerivative.h"
#include "MathFunctions.h"
double sin_function(double x) {
    return std::sin(x);
}

int main() {
    double x = 1.0;
    double h = 0.01;  

    double forward = forwardDerivative(sin_function, x, h);
    double backward = backwardDerivative(sin_function, x, h);
    double central = centralDerivative(sin_function, x, h);

    std::cout << "Forward Derivative of Sin(x): at  x = " << x << ": " << forward << std::endl;
    std::cout << "Backward Derivative of Sin(x): at  x= " << x << ": " << backward << std::endl;
    std::cout << "Central Derivative of Sin(x): at  x=" << x << ": " << central << std::endl;

        // polynomial function
    double forwardPoly = forwardDerivative(polynomial, x, h);
    double backwardPoly = backwardDerivative(polynomial, x, h);
    double centralPoly = centralDerivative(polynomial, x, h);

    std::cout << "Forward Derivative of Polynomial : (x^2 - 4x + 4) at x = " << x << ": " << forwardPoly << std::endl;
    std::cout << "Backward Derivative of Polynomial : (x^2 - 4x + 4) at x = " << x << ": " << backwardPoly << std::endl;
    std::cout << "Central Derivative of Polynomial : (x^2 - 4x + 4) at x = " << x << ": " << centralPoly << std::endl;

    // exponential function
    double forwardExp = forwardDerivative(exponential, x, h);
    double backwardExp = backwardDerivative(exponential, x, h);
    double centralExp = centralDerivative(exponential, x, h);

    std::cout << "Forward Derivative of Exponential at x = " << x << ": " << forwardExp << std::endl;
    std::cout << "Backward Derivative of Exponential at x = " << x << ": " << backwardExp << std::endl;
    std::cout << "Central Derivative of Exponential at x = " << x << ": " << centralExp << std::endl;

    return 0;
}