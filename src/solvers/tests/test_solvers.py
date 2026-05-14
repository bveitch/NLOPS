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
        pytest.param((5, 10), 0, "direct", marks=pytest.mark.skip(reason="not PSD")),
        ((5, 10), 0, "l2"), 
        ((5, 10), 0.1, "direct"),
        ((5, 10), 0.1, "reverse"),
        ((5, 10), 0.1, "l2"), 
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
    if l == 0:
        npt.assert_equal(answer, expected)
    else:
        npt.assert_almost_equal(answer, expected)

