#include "CentralDerivative.h"

double centralDerivative(double (*func)(double), double x, double h) {
    return (func(x + h) - func(x - h)) / (2 * h);
}