#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "HessianInversionCPU.h"

namespace py = pybind11;

template <typename T>
void bind_hessian_cpu(py::module& m) {
    py::class_<HessianInversionCPU<T>>(m, "HessianInversionCPU")
        .def(py::init<int>())
        .def("setMatrix", &HessianInversionCPU<T>::setMatrix)
        .def("invert", &HessianInversionCPU<T>::invert)
        .def("getInverse", &HessianInversionCPU<T>::getInverse)
        .def("regularizeMatrix", &HessianInversionCPU<T>::regularizeMatrix)
        .def("printMatrix", &HessianInversionCPU<T>::printMatrix);
}

PYBIND11_MODULE(hessian_inversion_cpu_py, m) {
    bind_hessian_cpu<double>(m);
}
