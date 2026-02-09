import numpy as np
from NLBase import NLBase
from SympyWrapper import SympyWrap
 
class SymbolicOperator(NLBase):

    def __init__(self, expr, variables): 
        self._vars = variables
        self._f = SympyWrap.from_strings(expr)
        self._jac= self._f.jac(vars)

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
        


