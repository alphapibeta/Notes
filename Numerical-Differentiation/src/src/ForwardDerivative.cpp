#include "ForwardDerivative.h"

double forwardDerivative(double (*func)(double), double x, double h) {
    return (func(x + h) - func(x)) / h;
}