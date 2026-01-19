import copy
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class NLBase(ABC):

    def __init__(self, input_shape:tuple, output_shape:tuple):
        self._input_shape = input_shape
        self._output_shape = output_shape

    @property
    def input_shape(self):
        return self._input_shape
    
    @property
    def output_shape(self):
        return self._input_shape

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

    def __init__(self, input_shape:tuple, output_shape:tuple):
        self._input_shape = input_shape
        self._output_shape = output_shape
    
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
    
class NLChain(NLBase):

    def __init__(self, 
                 input_shape: tuple, 
                 output_shape: tuple, 
                 nl_operators: list[NLBase]):
        super().__init__(input_shape, output_shape)
        self.operators = nl_operators

    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self._input_shape, f"NLChain: {shape=} != {self._input_shape}"
        else:
            assert shape == self._output_shape, f"NLChain: {shape=} != {self._output_shape}"

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        output=input.copy()
        for operator in self.operators:
            output=operator(output)
        return output
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        output = input.copy()
        doutput = dinput.copy()
        for operator in self.operators:
            doutput=operator.linear(output, doutput)
            output=operator(output)
        return doutput
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
        dtemp = dinputT.copy()
        ops = copy.copy(self.operators)
        while len(ops) > 0:
            op = ops.pop()
            if len(ops) > 0:
                output_shape = op.input_shape
                chainop= NLChain(input_shape = self._input_shape,
                                 output_shape = output_shape,
                                 nl_operators=ops)
                temp = chainop(input)
            dtemp = op.adjoint(temp,dtemp)
        return dtemp
        
def check_dot_product(operator: NLBase, input:npt.NDArray):
    x=input.random()
    y=operator(input).random()
    yTAx=y.dot(operator.linear(input, x))
    xTATy=x.dot(operator.adjoint(input, y))
    np.testing.assert_allclose(yTAx, xTATy)

def check_linearization(operator: NLBase, input:npt.NDArray, eps=1.0e-6):
    dx=input.random()
    fpx=operator(input+eps*dx)
    fmx=operator(input-eps*dx)
    df = (fpx-fmx)*(0.5/eps)
    Fmx=operator.linear(input,dx)
    np.assert_close(df, Fmx, atol=1.0e-6, rtol=eps*eps*dx.dot(dx))