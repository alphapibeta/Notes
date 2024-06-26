#include "BackwardDerivative.h"

double backwardDerivative(double (*func)(double), double x, double h) {
    return (func(x) - func(x - h)) / h;
}