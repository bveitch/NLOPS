import numpy.typing as npt
from src.operators.base import LBase
import matrix_operator_cpp

NTHREADS = 4  

class MatrixCppOperator(LBase):

    def __init__(self, arr2d: npt.NDArray):
        self.op = matrix_operator_cpp.MatrixOp(arr2d, NTHREADS)
        output_size, input_size=self.op.shape
        super().__init__(input_shape=(input_size), output_shape=(output_size), name = "cpp_matrix")
        
    def _fwd(self, x):
        return self.op.forward(x)
    
    def _adj(self, y):
        return self.op.adjoint(y)

class SparseMatrixCppOperator(LBase):

    def __init__(self, shape:tuple, locs: npt.NDArray, values: npt.NDArray):
        nrow, ncol = shape
        self.op = matrix_operator_cpp.SparseOp(
             nrow, ncol, tuple(locs[0]), tuple(locs[1]), tuple(values), NTHREADS)
        output_size, input_size=self.op.shape
        super().__init__(input_shape=(input_size), output_shape=(output_size), name = "cpp_matrix")

    def _fwd(self, x):
        return self.op.forward(x)
    
    def _adj(self, y):
        return self.op.adjoint(y)