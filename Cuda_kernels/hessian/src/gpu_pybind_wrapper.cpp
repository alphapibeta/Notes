#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "HessianInversionGPU.h"

namespace py = pybind11;

template <typename T>
void bind_hessian_gpu(py::module& m, const std::string& precision) {
    py::class_<HessianInversionGPU<T>>(m, ("HessianInversionGPU_" + precision).c_str())  // Use a unique class name for each precision
        .def(py::init<int>())
        .def("setMatrix", &HessianInversionGPU<T>::setMatrix)
        .def("invert", &HessianInversionGPU<T>::invert)
        .def("getInverse", &HessianInversionGPU<T>::getInverse)
        .def("regularizeMatrix", &HessianInversionGPU<T>::regularizeMatrix)
        .def("printMatrix", &HessianInversionGPU<T>::printMatrix)
        .def("getGPUBandwidth", &HessianInversionGPU<T>::getGPUBandwidth)
        .def("getGPUComputationalThroughput", &HessianInversionGPU<T>::getGPUComputationalThroughput)
        .def("getTheoreticalGPUBandwidth", &HessianInversionGPU<T>::getTheoreticalGPUBandwidth)
        .def("getTheoreticalGPUComputationalThroughput", &HessianInversionGPU<T>::getTheoreticalGPUComputationalThroughput)
        .def("getGPUMemoryThroughput", &HessianInversionGPU<T>::getGPUMemoryThroughput)
        .def("getArithmeticIntensity", &HessianInversionGPU<T>::getArithmeticIntensity);
}

PYBIND11_MODULE(hessian_inversion_gpu_py, m) {
    bind_hessian_gpu<float>(m, "float");   // Bind float precision
    bind_hessian_gpu<double>(m, "double"); // Bind double precision
}
