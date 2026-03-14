
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class NLBase(ABC):

    def __init__(self, input_shape:tuple, output_shape:tuple, name:str = "NLBase"):
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._name = name

    def __str__(self):
        return f"{self._name}: {self._input_shape} -> {self._output_shape}"

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, val):
        self._name = val

    @property
    def input_shape(self):
        if isinstance(self._input_shape, int):
            return tuple([self._input_shape])
        return self._input_shape
    
    @input_shape.setter
    def input_shape(self, shape):
        self._input_shape = shape
    
    @property
    def output_shape(self):
        if isinstance(self._output_shape, int):
            return tuple([self._output_shape])
        return self._output_shape
    
    @abstractmethod
    def _check_shape(self, shape:tuple, is_fwd:bool):
        return NotImplemented
  
    @abstractmethod
    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        return NotImplemented

    @abstractmethod
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        return NotImplemented

    @abstractmethod
    def _adj_lin(self, input:npt.NDArray, dinput:npt.NDArray)->npt.NDArray:
        return NotImplemented
    
    def __call__(self, input:npt.NDArray)->npt.NDArray:
        self._check_shape(input.shape, True)
        return self._fwd_nl(input)
    
    def linear(self, input:npt.NDArray, dinput:npt.NDArray)->npt.NDArray:
        self._check_shape(input.shape, True)
        self._check_shape(dinput.shape, True)
        return self._fwd_lin(input, dinput)
    
    def adjoint(self, input:npt.NDArray, dinputT:npt.NDArray)->npt.NDArray:
        self._check_shape(input.shape, True)
        self._check_shape(dinputT.shape, False)
        return self._adj_lin(input, dinputT)

class LBase(NLBase):

    def __init__(self, input_shape:tuple, output_shape:tuple, name:str = "LBase"):
        super().__init__(input_shape, output_shape, name)
    
    @abstractmethod
    def _fwd(self, input:npt.NDArray) ->npt.NDArray:
        return NotImplemented

    @abstractmethod
    def _adj(self, input:npt.NDArray, dinput:npt.NDArray)->npt.NDArray:
        return NotImplemented
    
    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        return self._fwd(input)

    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        return self._fwd(dinput)
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray)->npt.NDArray:
        return self._adj(dinputT)
    
    def __call__(self, input:npt.NDArray)->npt.NDArray:
        self._check_shape(input.shape, True)
        return self._fwd(input)
    
    def adj(self, inputT:npt.NDArray)->npt.NDArray:
        self._check_shape(inputT.shape, False)
        return self._adj(inputT) 
 
        
def check_dot_product(operator: NLBase, input:npt.NDArray):
    input_shape = operator.input_shape
    output_shape = operator.output_shape
    x = np.random.random(input_shape)
    y = np.random.random(output_shape)
    yTAx=np.sum(y*operator.linear(input, x))
    xTATy=np.sum(x*operator.adjoint(input, y))
    np.testing.assert_allclose(yTAx, xTATy)

def check_linearization(operator: NLBase, input:npt.NDArray, eps=1.0e-6, atol=1.0e-6):
    input_shape = operator.input_shape
    dx = np.random.random(input_shape)
    fpx=operator(input+eps*dx)
    fmx=operator(input-eps*dx)
    df = (fpx-fmx)*(0.5/eps)
    Fmx=operator.linear(input,dx)
    np.testing.assert_allclose(df, Fmx, atol=atol, rtol=eps*eps*dx.dot(dx))