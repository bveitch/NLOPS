import numpy as np
from src.operators.MatrixOperator import MatrixOperator
from src.operators.Constraints import Sigmoid
from src.operators.NLBase import check_dot_product, check_linearization

def create_matrix_operator():
    M = np.ones((3,5))
    return MatrixOperator(M, "matop")

def create_sigmoid_operator():
    return Sigmoid((10))

def test_dot_product():
    #operator = create_matrix_operator()
    operator = create_sigmoid_operator()
    input = np.ones((operator.input_shape))
    check_dot_product(operator, input)

def test_linearization():
    operator = create_sigmoid_operator()
    input = np.ones((operator.input_shape))
    check_linearization(operator, input)