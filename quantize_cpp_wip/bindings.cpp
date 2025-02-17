#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::tuple and other STL types
#include <pybind11/numpy.h>  // For handling NumPy arrays (if needed)
#include "quantize.h"  // Include the header file where QuantizeCore and QuantizationType are defined

namespace py = pybind11;

PYBIND11_MODULE(quantize_core, m) {
    // Bind the QuantizationType enum
    py::enum_<QuantizationType>(m, "QuantizationType")
        .value("Q_4_BIT", QuantizationType::Q_4_BIT)
        .value("Q_8_BIT", QuantizationType::Q_8_BIT)
        .export_values();

    // Bind the QuantizeCore class
    py::class_<QuantizeCore>(m, "QuantizeCore")
        .def(py::init<QuantizationType>())
        .def("quantizeTensor", &QuantizeCore::quantizeTensor);
}