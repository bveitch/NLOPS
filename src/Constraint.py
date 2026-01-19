import numpy as np
import numpy.typing as npt
from src.NLBase import NLBase

class Constraint(NLBase):

    def __init__(self, min:float=0, max:float=1):
        self.min = min
        self.max = max

    @staticmethod
    def stable_sigmoid(x:npt.NDArray):
        return np.where(x> 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
    
    def _check_shape(self, shape:tuple, is_fwd:bool):
        pass

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        sigma=Constraint.stable_sigmoid(input)
        return self.min+self.max*sigma
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        sigma=Constraint.stable_sigmoid(input)
        return self.max*sigma*(1-sigma)*dinput
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
        return self._fwd_lin(input, dinputT)