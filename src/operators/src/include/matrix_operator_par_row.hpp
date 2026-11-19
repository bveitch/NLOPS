#pragma once
#include<iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matrix_operator.hpp"
using namespace std;

namespace py = pybind11;
template<typename T>
class MatrixOperatorParRow: public MatrixOperator<T>{

public:
        
    MatrixOperatorParRow(const py::array_t<T>& matrix, int nthreads):
    MatrixOperator<T>(matrix, nthreads){};

    void fwd_add(const T* const input, T* const output) const final {
        #pragma omp parallel for
        for (int irow=0; irow<m_nrow; irow++){
            const T* row_data = m_data[irow];
            for (int icol=0; icol < m_ncol; ++icol) {
                output[irow]+=row_data[icol]*input[icol];
            }
        }
    }

   void adj_add(const T* const inputT, T* const outputT) const final { 
        for (int icol=0; icol<m_ncol; icol++){
            const T* col_data = m_dataT[icol];
            #pragma omp parallel for shared (icol, col_data) reduction(+ : outputT[icol])
            for (int irow=0; irow < m_nrow; ++irow) {
                outputT[icol]+=col_data[irow]*inputT[irow];
            }
        }
    }

protected:
    using BaseOperator<T>::m_nrow;
    using BaseOperator<T>::m_ncol;
    using MatrixOperator<T>::m_data;
    using MatrixOperator<T>::m_dataT;
};