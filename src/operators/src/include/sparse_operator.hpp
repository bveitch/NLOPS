#pragma once
#include<iostream>
#include <numeric>
#include <vector>
#include <array>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "base_operator.hpp"
using namespace std;

namespace py = pybind11;
template<typename T>
class SparseOperator: public BaseOperator<T>{

public:
        
    SparseOperator(const int nrow, 
        const int ncol, 
        const py::tuple& row_locations, 
        const py::tuple& col_locations, 
        const py::tuple& values, 
        int nthreads):
    BaseOperator<T>(nrow, ncol, nthreads),
    m_nsparse(row_locations.size()),
    m_row_locs(row_locations.cast<vector<int>>()),
    m_col_locs(col_locations.cast<vector<int>>()),
    m_vals(values.cast<vector<T>>())
    {
     
        if (m_col_locs.size() !=  m_nsparse) {
            std::ostringstream error_message;
            error_message << "col locations size = "<< m_row_locs.size() <<" != row locations size" << m_nsparse << std::endl;
            throw runtime_error(error_message.str());
        } 
        if (m_vals.size() !=  m_nsparse) {
            std::ostringstream error_message;
            error_message << "vals size = "<< m_vals.size() <<" != row locations size" << m_nsparse << std::endl;
            throw runtime_error(error_message.str());
        } 
        for(size_t iloc=0; iloc<m_nsparse; iloc++){
            if (m_col_locs[iloc] < 0 || m_col_locs[iloc] >=m_ncol){
                std::ostringstream error_col_bound;
                error_col_bound << "col location = "<< m_col_locs[iloc] <<"oustide bound [0," << m_ncol << ")" << std::endl;
                throw runtime_error(error_col_bound.str());
            }
            if (m_row_locs[iloc] < 0 || m_row_locs[iloc] >=m_nrow){
                std::ostringstream error_row_bound;
                error_row_bound << "row location = "<< m_row_locs[iloc] <<"oustide bound [0," << m_nrow << ")" << std::endl;
                throw runtime_error(error_row_bound.str());
            }
        }
    };

    void fwd_add(const T* const input, T* const output) const final {
        #pragma omp parallel for reduction(+:output[:m_nrow])
        for (size_t iloc=0; iloc<m_nsparse; iloc++){
           output[m_row_locs[iloc]]+=m_vals[iloc]*input[m_col_locs[iloc]];
        }
    }

   void adj_add(const T* const inputT, T* const outputT) const final { 
         #pragma omp parallel for reduction(+:outputT[:m_ncol])
         for (size_t iloc=0; iloc<m_nsparse; iloc++){
           outputT[m_col_locs[iloc]]+=m_vals[iloc]*inputT[m_row_locs[iloc]];
        }
    }

protected:
    using BaseOperator<T>::m_nrow;
    using BaseOperator<T>::m_ncol;
private:

    size_t m_nsparse;
    vector<int> m_row_locs;
    vector<int> m_col_locs;
    vector<T> m_vals;

};