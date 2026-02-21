from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from src.objectives.base import ObjectiveFn
    
class SumObjectiveFn(ObjectiveFn):

    def __init__(self, 
                 objfns: list[ObjectiveFn], 
                 regularizers:list[np.float64]):
        assert len(objfns) == len(regularizers), "must be one regularizer for each objfn"
        shape = objfns[0].xshape
        super().__init__(shape)
        nobjfns =len(objfns)
        if nobjfns > 1: 
            for objfn in objfns[1:]:
                assert objfn.xshape == shape, "objective function must have same x shape"
        self._objfns = list(zip(objfns, regularizers))

    @classmethod
    def from_objfn(cls, objfn: ObjectiveFn):
        return cls([objfn],[np.float64(1)])
        
    def __add__(self, other: tuple[ObjectiveFn, np.float64]):
        zipped_objfns = self._objfns + other
        objfns, regularizers = zip(*zipped_objfns)
        return SumObjectiveFn(list(objfns), list(regularizers))

    def _value(self, x):
        v = np.float64(0)
        for objfn, reg in self._objfns:
            v += reg * objfn._value(x)
        return v

    def _gradient(self, x):
        g = np.zeros(x.shape, dtype=x.dtype)
        for objfn, reg in self._objfns:
            g += reg * objfn._gradient(x)
        return g


