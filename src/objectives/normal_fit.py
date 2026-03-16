import numpy as np
import numpy.typing as npt
from src.objectives.base import L2ObjectiveFn, NLBase
from src.operators.chain import NLChain

class NormalFit(L2ObjectiveFn):

    eps = 5.0e-4

    def __init__(self, 
                 shape: tuple,
                 predictor: NLBase, 
                 data: npt.NDArray):
        predictor_output_shape = predictor.output_shape
        assert (
            predictor_output_shape[0] == 2
        ), f"{predictor_output_shape[0] =} must be 2"
        assert (
            predictor_output_shape[1:] == data.shape 
        ), f"{predictor_output_shape[1:] =} must match {data.shape}"
        super().__init__(shape, predictor, data)
        
    def _eval(self, mu, var):
        r = mu - self.data
        eps = NormalFit.eps
        safevar = np.where(var > eps, var, eps*np.ones(var.shape))
        logvar=np.log(safevar)
        return 0.5*np.sum(r**2/safevar) + 0.5*np.sum(logvar)
                                
    def _grad(self, mu, var):
        r = mu - self.data
        eps = NormalFit.eps
        safevar = np.where(var > eps, var, eps*np.ones(var.shape))
        gmu = r/safevar
        gvar = np.where(var > eps, 0.5*(safevar-r**2)/safevar**2, np.zeros(var.shape))
        return np.stack([gmu, gvar])
     
    def _value(self, x):
        y = self.op(x)
        mu = y[0, ...]
        var = y[1, ...] 
        return self._eval(mu, var)
    
    def _gradient(self, x):
        y = self.op(x)
        mu = y[0, ...]
        var = y[1, ...] 
        gy = self._grad(mu, var)
        return self.op.adjoint(x , gy)
    
class BinomialFit(NormalFit):

    class BinomialToNormal(NLBase):
        
        def __init__(self, shape:tuple):
            output_shape = (2, *shape)
            super().__init__(shape, output_shape, "BinomialToNormal")
        
        def _check_shape(self, shape:tuple, is_fwd:bool):
            if is_fwd:
                assert shape == self.input_shape, f"{self.name}: {shape=} != {self.input_shape}"
            else:
                assert shape == self.output_shape, f"{self.name}: {shape=} != {self.output_shape}"
        
        def _fwd_nl(self, input):
            return np.stack([input, input*(1-input)], axis=0)
        
        def _fwd_lin(self, input0, dinput):
            return np.stack([dinput, (1-2.*input0)*dinput], axis=0)
        
        def _adj_lin(self, input0, dinputT):
            dinputT0, dinputT1 = np.unstack(dinputT, axis=0)
            return dinputT0+(1-2.*input0)*dinputT1
        
    def __init__(self, 
                 shape: tuple,
                 operator: NLBase, 
                 data: npt.NDArray):
        data_shape = data.shape
        if isinstance(operator, NLChain):
            predictor  = BinomialFit.BinomialToNormal(data_shape)*operator
        else:
            predictor =  NLChain([operator,BinomialFit.BinomialToNormal(data_shape)])
        super().__init__(shape, predictor, data)
