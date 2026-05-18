import numpy as np
import numpy.typing as npt
from src.operators.base import LBase

class MatrixOperator(LBase):
    
    def __init__(self, M: npt.NDArray, name:str, input_shape: tuple|None = None, ):
        output_size, input_size=M.shape
        self._M = M
        if input_shape is not None:
            assert input_size == input_shape[-1], f"{self.name}: {input_shape[-1]=} != {input_size}"
            output_shape = list(input_shape)
            output_shape[-1] = output_size 
        else:
            input_shape = (input_size)
            output_shape = (output_size)
        super().__init__(input_shape=input_shape, output_shape=output_shape, name = name)
        
    def _fwd(self, hsi):
        return np.dot(hsi, self._M.T)
    
    def _adj(self, rgb):
        return np.dot(rgb, self._M)