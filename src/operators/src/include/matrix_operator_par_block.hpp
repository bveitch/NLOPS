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
class MatrixOperatorParBlock: public MatrixOperator<T>{

public:
        
    MatrixOperatorParBlock(const py::array_t<T>& matrix, int nthreads):
    MatrixOperator<T>(matrix, nthreads){};


    void fwd_add(const T* const input, T* const output) const final {
        #pragma omp parallel for collapse(2) reduction(+ : output[:m_nrow])
        for (int irow=0; irow<m_nrow; irow++){
            for (int icol=0; icol < m_ncol; ++icol) {
                output[irow]+=m_data[irow][icol]*input[icol];
            }
        }
    }

   void adj_add(const T* const inputT, T* const outputT) const final { 
        #pragma omp parallel for collapse(2) reduction(+ : outputT[:m_ncol])
        for (int icol=0; icol<m_ncol; icol++){
            for (int irow=0; irow < m_nrow; ++irow) {
                outputT[icol]+=m_dataT[icol][irow]*inputT[irow];
            }
        }
    }

protected:
    using BaseOperator<T>::m_nrow;
    using BaseOperator<T>::m_ncol;
    using MatrixOperator<T>::m_data;
    using MatrixOperator<T>::m_dataT;

};