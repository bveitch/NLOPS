import numpy as np
from src.operators.base import NLBase
from src.operators.sympy_wrapper import SympyWrap
 
class SymbolicOperator(NLBase):

    def __init__(self, exprs, variables, name): 
        self._vars = variables
        self._f = SympyWrap.from_strings(exprs)
        self._jac= self._f.jac(variables)
        nsymbols = len(variables)
        nexpr = len(exprs)
        super().__init__(input_shape=nsymbols, output_shape=nexpr, name = name)
        

    def _check_shape(self, input_shape, is_fwd):
        size = input_shape[0]
        if is_fwd:
            assert size == self._input_shape, f"{self.name}: {input_shape=}[-1] != {self._input_shape}"
        else:
            assert size == self._output_shape, f"{self.name}: {input_shape=}[-1] != {self._output_shape}"

    def _fwd_nl(self, input):
        return self._f(input, self._vars)
    
    def _eval_jac(self, input):
        return self._jac(input, self._vars)

    def _fwd_lin(self, input, dinput):
        jac = self._eval_jac(input)
        return np.einsum('ij...,j...->i...', jac, dinput)
    
    def _adj_lin(self, input, dinputT):
        jac = self._eval_jac(input)
        return np.einsum('ji...,j...->i...', jac, dinputT)
        


