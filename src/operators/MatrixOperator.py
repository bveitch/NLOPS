import numpy as np
import numpy.typing as npt
from src.operators.NLBase import LBase

class MatrixOperator(LBase):
    
    def __init__(self, M: npt.NDArray, name:str):
        output_shape, input_shape=M.shape
        self._M = M
        super().__init__(input_shape=input_shape, output_shape=output_shape, name = name)


    def _check_shape(self, input_shape, is_fwd):
        size = input_shape[-1]
        if is_fwd:
            assert size == self._input_shape, f"{input_shape=}[-1] != {self._input_shape}"
        else:
            assert size == self._output_shape, f"{input_shape=}[-1] != {self._output_shape}"

    def _fwd(self, hsi):
        return np.dot(hsi, self._M.T)
    
    def _adj(self, rgb):
        return np.dot(rgb, self._M)