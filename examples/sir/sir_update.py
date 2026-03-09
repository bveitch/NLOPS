from functools import partial
import numpy as np
from src.operators.sympy_wrapper import SympyWrap

class SIRUpdate:
    
    def __init__(self, a, b):
        self.a = a
        self.b = b 

    def _f_si(self, s, i):
        return np.array([-self.a*s*i, self.a*s*i-self.b*i, self.b*i])
    
    def _jac_si(self, s, i):
        return np.array([
            [-self.a*i, -self.a*s],
            [ self.a*i, self.a*s-self.b],
            [ 0,        self.b]])
    
    def _jac_mod(self, s, i):
        return np.array([
            [-s*i,  0],
            [ s*i, -i],
            [ 0,    i]])

    @np.errstate(all='ignore')
    def f(self, sir0):
        return self._f_si(sir0[0], sir0[1])
        
    @np.errstate(all='ignore')
    def df_dsir(self, sir0, dsir):
        jac = self._jac_si(sir0[0], sir0[1])
        return np.dot(jac, dsir[0:2])
    
    @np.errstate(all='ignore')
    def df_dmod(self, sir0, dmod):
        jac = self._jac_mod(sir0[0], sir0[1])
        return np.dot(jac, dmod)

    @np.errstate(all='ignore')
    def dsir_df(self, sir0, df):
        jac = self._jac_si(sir0[0], sir0[1])
        dsi = np.dot(jac.T, df)
        return np.array([dsi[0], dsi[1], 0])

    @np.errstate(all='ignore')
    def dmod_df(self, sir0, df):
        jac = self._jac_mod(sir0[0], sir0[1])
        return np.dot(jac.T, df)

class SIRSympyUpdate:

    exprs =  ["-a*s*i", "a*s*i-b*i", "b*i"]
    vars = ["s", "i"]
    baseop = SympyWrap.from_strings(exprs)
 
    def __init__(self, a, b): 
        self._func = SIRSympyUpdate.baseop.partial_eval({"a":a, "b":b})
        self._jac = self._func.jac(SIRSympyUpdate.vars)
        self._jacm, _ = SIRSympyUpdate.baseop.partial_jac({"a":a, "b":b})

    @classmethod
    def input_size(cls):
        return len(cls.vars)
    
    @classmethod
    def output_size(cls):
        return len(cls.exprs)
    
    @staticmethod
    def mod_size():
        return 2
    
    def _f_si(self, s, i):
        return self._func(np.array([s, i]), symbol_names=self.vars)
    
    def _jac_si(self, s, i):
        return self._jac(np.array([s, i]), symbol_names=self.vars)
    
    def _jac_mod(self, s, i):
        return self._jacm(np.array([s, i]), symbol_names=self.vars)
