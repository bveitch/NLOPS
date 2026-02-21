import pytest
import numpy as np
from src.operators.matrix import MatrixOperator
from src.operators.constraints import Sigmoid
from src.operators.chain import NLChain
from src.operators.base import check_dot_product, check_linearization
from src.operators.symbolic import SymbolicOperator

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

def create_sympy_operator():
    return SymbolicOperator(
        exprs = ["x*y*z", "x+y+z"],
        variables = ["x", "y", "z"],
        name = "xyz"
    )

@pytest.fixture
def create_operator():
    def _create_operator(name):
        operators = {
            "matrix"  : create_matrix_operator,
            "sigmoid" : create_sigmoid_operator,
            "nlchain" : create_nlchain_operator,
            "sympy"   : create_sympy_operator,
        }
        return operators[name]()

    return _create_operator

pytestmark = pytest.mark.parametrize("name", ["matrix", "sigmoid", "nlchain", "sympy"])

def test_dot_product(name, create_operator):
    operator = create_operator(name)
    input = np.ones((operator.input_shape))
    check_dot_product(operator, input)

def test_linearization(name, create_operator):
    operator = create_operator(name)
    input = np.ones((operator.input_shape))
    check_linearization(operator, input)