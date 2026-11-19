#pragma once
#include<iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "base_operator.hpp"
using namespace std;

namespace py = pybind11;
template<typename T>
class MatrixOperator: public BaseOperator<T>{

public:
        
    MatrixOperator(const py::array_t<T>& matrix, int nthreads):
    BaseOperator<T>(matrix.request().shape[0], matrix.request().shape[1], nthreads), 
    m_info(matrix.request())
    {
        m_data = new T *[m_nrow];
        m_dataT = new T *[m_ncol];
        T* buffer_ptr=static_cast<T*>(m_info.ptr);
        for (int irow=0; irow<m_nrow; irow++){
            m_data[irow]  = buffer_ptr + irow*m_ncol;
        }
        for (int icol=0; icol<m_ncol; icol++){
            m_dataT[icol]  = new T[m_nrow];
            for (int irow=0; irow<m_nrow; irow++){
                m_dataT[icol][irow]  = *(buffer_ptr + icol+irow*m_ncol);
            }
        }
    };

    ~MatrixOperator() override {
        delete[] m_data;
        for(auto icol=0;icol<m_ncol;icol++)
        {
            delete[] m_dataT[icol];
        }
        delete[] m_dataT;
    };

    void fwd_add(const T* const input, T* const output) const override {
        for (int irow=0; irow<m_nrow; irow++){
            const T* row_data = m_data[irow];
            for (int icol=0; icol < m_ncol; ++icol) {
                output[irow]+=row_data[icol]*input[icol];
            }
        }
    }

   void adj_add(const T* const inputT, T* const outputT) const override { 
        for (int icol=0; icol<m_ncol; icol++){
            const T* col_data = m_dataT[icol];
            for (int irow=0; irow < m_nrow; ++irow) {
                outputT[icol]+=col_data[irow]*inputT[irow];
            }
        }
    }

protected:
    using BaseOperator<T>::m_nrow;
    using BaseOperator<T>::m_ncol;
    T** m_data;
    T** m_dataT;

private:
    py::buffer_info m_info;
};