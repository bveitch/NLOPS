#pragma once
#include<iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
using namespace std;

namespace py = pybind11;
template<typename T>
class BaseOperator{

public:
        
    BaseOperator(int nrow, int ncol, int nthreads=1):
    m_nrow(nrow), 
    m_ncol(ncol),
    m_nthreads(nthreads)
    {
        omp_set_num_threads(m_nthreads);
    };

    virtual ~BaseOperator(){};

    py::tuple shape() const {
        return py::make_tuple(m_nrow, m_ncol);
    }

    int get_num_threads() const {
        int nthreads=1;
        #pragma omp parallel
        {
            #pragma omp single
            {
                nthreads = omp_get_num_threads();
            }
        }
        return nthreads;
    }
    
    virtual void fwd_add(const T* const input , T* const output) const =0;

    virtual void adj_add(const T* const inputT, T* const outputT) const =0;

    void fwd(const T* const input, T* const output) const {
        std::fill(output, output + m_nrow, static_cast<T>(0.)); 
        fwd_add(input, output);
    }

    void adj(const T* const inputT, T* const outputT) const { 
        std::fill(outputT, outputT + m_ncol, static_cast<T>(0.)); 
        adj_add(inputT, outputT);
    }

    py::array_t<T> forward(const py::array_t<T>& input) const{
        int m_rowshape[1]{m_nrow};
        int m_strides[1]{sizeof(T)};
        auto output = py::array_t<T>(m_rowshape, m_strides);
        auto input_buf = input.request();
        auto input_size = input_buf.shape[0]; 
        auto output_buf = output.request();
        check_colspace(input_size);
        fwd(static_cast<const T *>(input_buf.ptr),  static_cast<T *>(output_buf.ptr));
        return output;
    }

    py::array_t<T> adjoint(const py::array_t<T>& inputT) const{ 
        int m_colshape[1]{m_ncol};
        int m_strides[1]{sizeof(T)};
        auto outputT = py::array_t<T>(m_colshape, m_strides);
        auto inputT_buf = inputT.request();
        auto inputT_size = inputT_buf.shape[0]; 
        auto outputT_buf = outputT.request();
        check_rowspace(inputT_size);
        adj(static_cast< const T *>(inputT_buf.ptr),  static_cast<T *>(outputT_buf.ptr));
        return outputT;
    }

private:

    void check_colspace(int size) const {
        std::ostringstream error_message;
        error_message << "input size = "<< size <<" != " << m_ncol;
        if (size != m_ncol) {
            throw runtime_error(error_message.str());
        } 
    };

    void check_rowspace(int size) const {
        std::ostringstream error_message;
        error_message << "input size = "<< size <<" != " << m_nrow;
        if (size != m_nrow) {
            throw runtime_error(error_message.str());
        } 
    };
    
protected:
    int m_nrow;
    int m_ncol;
    int m_nthreads;
};