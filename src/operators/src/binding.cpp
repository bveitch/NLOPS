#include <pybind11/pybind11.h>
#include "matrix_operator.hpp"
#include "matrix_operator_par_row.hpp"
#include "matrix_operator_par_col.hpp"
#include "matrix_operator_par_block.hpp"
#include "sparse_operator.hpp"
namespace py = pybind11;
using namespace std;

PYBIND11_MODULE(matrix_operator_cpp, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: matrix_operator_cpp

        .. autosummary::
           :toctree: _generate

           Matrix to Vec Linear Operators

    )pbdoc";
    py::class_<BaseOperator<double>>(m, "BaseOp")
    .def_property_readonly("shape", &MatrixOperator<double>::shape)
    .def_property_readonly("nthreads", &MatrixOperator<double>::get_num_threads);
    py::class_<MatrixOperator<double>, BaseOperator<double>>(m, "MatrixOp")
    .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const int&>())
    .def("forward", &MatrixOperator<double>::forward)
    .def("adjoint", &MatrixOperator<double>::adjoint);
    py::class_<MatrixOperatorParRow<double>, MatrixOperator<double>>(m, "MatrixOpParRow")
    .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const int&>())
    .def("forward", &MatrixOperator<double>::forward)
    .def("adjoint", &MatrixOperator<double>::adjoint);
    py::class_<MatrixOperatorParCol<double>, MatrixOperator<double>>(m, "MatrixOpParCol")
    .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const int&>())
    .def("forward", &MatrixOperator<double>::forward)
    .def("adjoint", &MatrixOperator<double>::adjoint);
    py::class_<MatrixOperatorParBlock<double>, MatrixOperator<double>>(m, "MatrixOpParBlk")
    .def(py::init<const py::array_t<double, py::array::c_style | py::array::forcecast>&, const int&>())
    .def("forward", &MatrixOperator<double>::forward)
    .def("adjoint", &MatrixOperator<double>::adjoint);
    py::class_<SparseOperator<double>, BaseOperator<double>>(m, "SparseOp")
    .def(py::init<const int&, const int&, const py::tuple&, const py::tuple&,const py::tuple&, const int&>())
    .def("forward", &SparseOperator<double>::forward)
    .def("adjoint", &SparseOperator<double>::adjoint);
}