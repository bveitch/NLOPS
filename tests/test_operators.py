import os
import sys
# Adds the parent directory of 'src' to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
from src.operators.matrix import MatrixOperator
from src.operators.constraints import Sigmoid
from src.operators.chain import NLChain
from src.operators.vector import NLVector
from src.operators.base import check_dot_product, check_linearization
from src.operators.symbolic import SymbolicOperator
from src.operators.matrix_cpp import MatrixCppOperator, SparseMatrixCppOperator

def create_matrix_operator():
    M = np.ones((3,5))
    return MatrixOperator(M, "matop")

def create_sigmoid_operator():
    return Sigmoid((10))

def create_nlchain_operator():
    M1 = np.ones((5, 10))
    M3 = np.ones((3, 5))
    op1 = MatrixOperator(M1, "matop1")
    op2 = Sigmoid(op1.output_shape)
    op3 = MatrixOperator(M3, "matop2")
    return NLChain([op1, op2, op3])

def create_vector_operator():
    size = 5
    M1 = np.ones((size, size))
    M3 = 0.1*np.eye(size)
    op1 = MatrixOperator(M1, "matop1")
    op2 = Sigmoid(op1.input_shape)
    op3 = MatrixOperator(M3, "matop2")
    return NLVector([op1, op2, op3])

def create_sympy_operator():
    return SymbolicOperator(
        exprs = ["x*y*z", "x+y+z"],
        variables = ["x", "y", "z"],
        name = "xyz"
    )

def create_matrix_cpp_operator(arr2d):
    return MatrixCppOperator(arr2d)

def create_sparse_cpp_operator(arr2d):
    locs = np.nonzero(arr2d)
    values = arr2d[locs]
    return SparseMatrixCppOperator(arr2d.shape, locs, values)

def create_matrix_cpp_operator_3x3():
    arr2d_3x3 = np.array([[1.,2.,0], [-1.,1.,1.], [3.,0.,1.]])
    return create_matrix_cpp_operator(arr2d_3x3)

def create_matrix_cpp_operator_3x5():
    arr2d_3x5 = np.array(
        [[1.,2.,0], 
         [-1.,1.,1.], 
         [3.,0.,1.], 
         [0,1.,1.0],
         [-1,0,10]])
    return create_matrix_cpp_operator(arr2d_3x5)

def create_matrix_cpp_operator_large():
    arr2d_large =  np.ones((100,1000), dtype=np.float64)
    return create_matrix_cpp_operator(arr2d_large)

def create_sparse_cpp_operator_3x3():
    arr2d_3x3 = np.array([[1.,2.,0], [-1.,1.,1.], [3.,0.,1.]])
    return create_sparse_cpp_operator(arr2d_3x3)

def create_sparse_cpp_operator_3x5():
    arr2d_3x5 = np.array(
        [[1.,2.,0], 
         [-1.,1.,1.], 
         [3.,0.,1.], 
         [0,1.,1.0],
         [-1,0,10]])
    return create_sparse_cpp_operator(arr2d_3x5)

def create_sparse_cpp_operator_large():
    arr2d_large =  np.ones((100,1000), dtype=np.float64)
    return create_sparse_cpp_operator(arr2d_large)

@pytest.fixture
def create_operator():
    def _create_operator(name):
        operators = {
            "matrix"  : create_matrix_operator,
            "sigmoid" : create_sigmoid_operator,
            "nlchain" : create_nlchain_operator,
            "vector"  : create_vector_operator,
            "sympy"   : create_sympy_operator,
            "cpp_3x3" : create_matrix_cpp_operator_3x3,
            "cpp_3x5" : create_matrix_cpp_operator_3x5,
            "large"   : create_matrix_cpp_operator_large, 
            "sparse_3x3"   : create_sparse_cpp_operator_3x3,
            "sparse_3x5"   : create_sparse_cpp_operator_3x5,
            "sparse_large" : create_sparse_cpp_operator_large 
        }
        return operators[name]()

    return _create_operator

pytestmark = pytest.mark.parametrize("name", 
    ["matrix", "sigmoid", "nlchain", "vector", "sympy", 
     "cpp_3x3", "cpp_3x5", "large",
     "sparse_3x3", "sparse_3x5", "sparse_large"])

def test_dot_product(name, create_operator):
    operator = create_operator(name)
    input = np.ones((operator.input_shape))
    check_dot_product(operator, input)

def test_linearization(name, create_operator):
    operator = create_operator(name)
    input = np.ones((operator.input_shape))
    check_linearization(operator, input, atol=5.0e-6)