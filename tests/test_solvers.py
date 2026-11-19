import os
import sys
# Adds the parent directory of 'src' to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
import numpy.testing as npt
from src.operators.matrix import MatrixOperator
from src.operators.constraints import Sigmoid
from src.operators.chain import NLChain
from src.objectives.base import L2ObjectiveFn
from src.objectives.sum_objective import SumObjectiveFn
from src.solvers.general import GeneralSolver
from src.solvers.direct import direct_solve, direct_solve_reverse
from src.operators.matrix_cpp import MatrixCppOperator, SparseMatrixCppOperator

def create_problem(N, M, l=None):
    A = np.eye(N=N,M=M)
    b = np.ones(N)
    expected = np.zeros(M)
    minshape = min(N, M)
    if l is not None and l > 0:
        expected[:minshape] += 1.0/(1.0+l)
    else:
        expected[:minshape] += np.ones(minshape)
    return A, b, expected

def solve_l2(A, b,l ):
    Aop = MatrixOperator(A, "A")
    Aobjfn = L2ObjectiveFn(shape = Aop.input_shape, operator = Aop, data = b)
    if l is not None and l>0:
        lobjfn = L2ObjectiveFn(shape = Aop.input_shape)
        objfn = SumObjectiveFn([Aobjfn, lobjfn], [1,l])
        return GeneralSolver(objfn).solve()
    else:
        return GeneralSolver(Aobjfn).solve()

def solve_l2_cpp_op(A, b, l):

    def from_arr2d(
        arr2d : npt.NDArray, 
        data: npt.NDArray):
        op = MatrixCppOperator(arr2d)
        assert data.shape == op.output_shape, f"{data.shape=} not equal to {op.output_shape}"
        return L2ObjectiveFn(shape = op.input_shape, operator = op, data = data)
    
    y=np.zeros(A.shape[1])
    T = np.sqrt(l)*np.eye(y.size)
    arr2d = np.concatenate([A,T], axis=0)
    assert arr2d.shape == (A.shape[0] + T.shape[0], T.shape[1])
    data = np.concatenate([b,y],axis=0)
    objfn = from_arr2d(arr2d= arr2d, data = data)
    return  GeneralSolver(objfn).solve()
 
def solve_l2_cpp_sparse_op(A, b, l):

    def from_sparse(
        nrow : int,
        ncol : int,
        locs : npt.NDArray,
        values : npt.NDArray, 
        data: npt.NDArray):
        assert data.size == nrow, f"{data.size=} not equal to {nrow}"
        op = SparseMatrixCppOperator((nrow, ncol), locs, values)
        return L2ObjectiveFn(shape = op.input_shape, operator = op, data = data)

    y=np.zeros(A.shape[1])
    T = np.sqrt(l)*np.eye(y.size)
    arr2d = np.concatenate([A,T], axis=0)
    assert arr2d.shape == (A.shape[0] + T.shape[0], T.shape[1])
    data = np.concatenate([b,y],axis=0)
    nrow, ncol = arr2d.shape
    locs = np.nonzero(arr2d)
    values = arr2d[locs]
    objfn = from_sparse(
        nrow=nrow, 
        ncol=ncol, 
        locs = locs,
        values = values,  
        data = data)
    return GeneralSolver(objfn).solve()

def solve_direct(A, b, l):
    return direct_solve(A, b, l=l)

def solve_reverse(A, b, l):
    return direct_solve_reverse(A, b, l)

pytestmark = pytest.mark.parametrize("shape, l, solver", 
    [ 
        ((10, 5), 0, "direct"),
        ((10, 5), 0, "l2"), 
        ((10, 5), 0.1, "direct"),
        ((10, 5), 0.1, "reverse"),
        ((10, 5), 0.1, "l2"), 
        ((10, 5), 0.1, "cpp"),
        ((10, 5), 0.1, "sparse"), 
        pytest.param((5, 10), 0, "direct", marks=pytest.mark.skip(reason="not PSD")),
        ((5, 10), 0, "l2"), 
        ((5, 10), 0.1, "direct"),
        ((5, 10), 0.1, "reverse"),
        ((5, 10), 0.1, "l2"), 
        ((5, 10), 0.1, "cpp"),
        ((5, 10), 0.1, "sparse"), 
    ])

def test_solver(shape, l, solver):
    N, M = shape
    A, b, expected = create_problem(N,M, l)
    if solver == "direct":
        answer = solve_direct(A, b, l)
    elif solver == "reverse":
        answer = solve_reverse(A, b, l)
    elif solver == "l2":
        answer = solve_l2(A, b, l)
    elif solver == "cpp":
        answer = solve_l2_cpp_op(A, b, l)
    elif solver == "sparse":
        answer = solve_l2_cpp_sparse_op(A, b, l)
    if l == 0:
        npt.assert_equal(answer, expected)
    else:
        npt.assert_almost_equal(answer, expected)

