import pytest
import numpy as np
import numpy.testing as npt
from src.operators.matrix import MatrixOperator
from src.operators.constraints import Sigmoid
from src.operators.chain import NLChain, NLBase
from src.operators.vector import NLVector
from src.objectives.base import L2ObjectiveFn, check_objective
from src.objectives.sum_objective import SumObjectiveFn
from src.objectives.normal_fit import NormalFit, BinomialFit

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

def create_normal_fit():

    class VarianceTestOp(NLBase):

        def __init__(self, input_shape: tuple, output_shape: tuple,name = "VarianceTest"):
            self.input_size = input_shape[0]
            assert self.input_size <=  output_shape[0]
            super().__init__(input_shape = input_shape, 
                             output_shape = output_shape, 
                             name = name)

        def _check_shape(self, shape:tuple, is_fwd:bool):
            if is_fwd:
                assert shape == self.input_shape, f"{self.name}: {shape=} != {self.input_shape}"
            else:
                assert shape == self.output_shape, f"{self.name}: {shape=} != {self.output_shape}"

        def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
            output=0.1*np.ones(self.output_shape)
            output[0:self.input_size] += input**2
            return output
    
        def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
            output = np.zeros(self.output_shape)
            output[0:self.input_size] += 2*input*dinput
            return output
    
        def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
            outputT = np.zeros(self.input_shape)
            outputT += 2*input*dinputT[0:self.input_size]
            return outputT
        
    mean_op = create_matrix_operator()
    var_op = VarianceTestOp(mean_op.input_shape, mean_op.output_shape)
    op = NLVector([mean_op, var_op])
    dshape = mean_op.output_shape
    mshape = mean_op.input_shape
    data = 1.0+np.random.rand(*dshape)
    return NormalFit(shape=mshape, predictor = op, data = data)

def create_binomial_fit():
    dshape = (10,)
    mshape = dshape
    data = np.random.rand(*dshape)
    operator = Sigmoid(shape = mshape)
    return BinomialFit(shape=mshape, operator = operator, data = data)

def create_binomial_fit2():
    operator = create_matrix_operator()
    dshape = operator.output_shape
    mshape = operator.input_shape
    data = np.random.rand(*dshape)
    return BinomialFit(shape=mshape, operator = operator, data = data)

def create_chain_binomial_fit():
    mop = create_matrix_operator()
    dshape = mop.output_shape
    mshape = mop.input_shape
    data = np.random.rand(*dshape)
    sigmoid = Sigmoid(shape = dshape)
    operator = NLChain([mop, sigmoid])
    return BinomialFit(shape=mshape, operator = operator, data = data)

@pytest.fixture
def create_objective():
    def _create_objective(name):
        objectives = {
            "l2"  : create_l2,
            "constrained_l2" : create_constrained_l2,
            "normal": create_normal_fit,
            "binomial": create_binomial_fit,
            "binomial2": create_binomial_fit2,
            "chain_binomial": create_chain_binomial_fit,
        }
        return objectives[name]()

    return _create_objective

pytestmark = pytest.mark.parametrize("name", 
    [ "l2", "constrained_l2", "normal", "binomial", "binomial2", "chain_binomial"])
def test_objective(name, create_objective):
    objective = create_objective(name)
    input = np.zeros((objective.xshape),dtype=np.float64)
    check_objective(objective, input)

