import pytest
import numpy as np
from src.operators.matrix import MatrixOperator
from src.operators.constraints import Sigmoid
from src.operators.chain import NLChain
from src.objectives.base import L2ObjectiveFn, check_objective
from src.objectives.sum_objective import SumObjectiveFn

def create_matrix_operator():
    M = np.ones((10,5))
    return MatrixOperator(M, "op")

def create_l2():
    op = create_matrix_operator()
    dshape = op.output_shape
    mshape = op.input_shape
    data = np.random.rand(*dshape)
    return L2ObjectiveFn(shape=mshape, operator = op, data = data)

def create_constrained_l2():
    mop = create_matrix_operator()
    dshape = mop.output_shape
    mshape = mop.input_shape
    data = np.random.rand(*dshape)
    constraint = Sigmoid(shape = mshape)
    operator = NLChain([constraint, mop])
    dobjfn = L2ObjectiveFn(shape = mshape, operator = operator, data = data)
    robjfn = L2ObjectiveFn(shape = mshape)
    return SumObjectiveFn([dobjfn, robjfn], [1,0.1])

@pytest.fixture
def create_objective():
    def _create_objective(name):
        objectives = {
            "l2"  : create_l2,
            "constrained_l2" : create_constrained_l2,
        }
        return objectives[name]()

    return _create_objective

pytestmark = pytest.mark.parametrize("name", ["l2", "constrained_l2"])

def test_objective(name, create_objective):
    objective = create_objective(name)
    input = np.zeros((objective.xshape),dtype=np.float64)
    check_objective(objective, input)

