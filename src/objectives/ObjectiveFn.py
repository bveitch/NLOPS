from abc import ABC, abstractmethod
import math
import numpy as np
import numpy.typing as npt
from src.operators.NLBase import NLBase

class ObjectiveFn(ABC):

    def __init__(self, shape:tuple):
        self._shape = shape
    
    @property
    def xshape(self) ->tuple:
        return self._shape
    
    @property 
    def xsize(self) -> int:
        return math.prod(self._shape)
    
    def unravel(self, xflat: npt.NDArray) -> npt.NDArray:
        assert xflat.ndim == 1, f"{xflat.ndim =} should be 1"
        assert xflat.size == self.xsize, f"{xflat.size =} should be {self.xsize}"
        return xflat.reshape(self._shape)
    
    @abstractmethod
    def _value(self, x: npt.NDArray) -> np.float64:
        return NotImplemented

    @abstractmethod
    def _gradient(self, x: npt.NDArray) -> npt.NDArray:
        return NotImplemented
    
    def __call__(self, xflat: npt.NDArray):
        x = self.unravel(xflat)
        return self._value(x)
    
    def gradient(self, xflat: npt.NDArray) -> npt.NDArray:
        x = self.unravel(xflat)
        g = self._gradient(x)
        return np.ravel(g)

class L2ObjectiveFn(ObjectiveFn):

    def __init__(self, 
                 shape: tuple,
                 operator: NLBase  | None = None, 
                 data: npt.NDArray | None = None):
        self.op = operator
        self.d = data
        super().__init__(shape)

    @property
    def data(self):
        return self.d
    
    @property 
    def operator(self):
        return self.op
    
    def _eval(self, x):
        if self.op is not None:
            fx = self.op(x)
        else:
            fx = x
        if self.d is not None:
            return fx - self.d
        else:
            return fx

    def _value(self, x):
        r = self._eval(x)
        return 0.5*np.sum(r**2)
    
    def _gradient(self, x):
        r = self._eval(x)
        if self.op is not None:
            return self.op.adjoint(x, r)
        else:
            return r
