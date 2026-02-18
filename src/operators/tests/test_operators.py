import numpy as np
from src.operators.MatrixOperator import MatrixOperator
from src.operators.NLBase import check_dot_product

def create_matrix_operator():
    M = np.ones((3,5))
    print(M.shape)
    return MatrixOperator(M, "matop")

def test_dot_product():
    operator = create_matrix_operator()
    input = np.ones((operator.input_shape))
    check_dot_product(operator, input)