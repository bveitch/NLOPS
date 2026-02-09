import numpy as np
from src.objectives import ObjectiveFn

class NormalFit(ObjectiveFn):

    def __init__(self, data, predictor):
        self.data = data
        self.predictor = predictor

    def _val(self, mu, var):
        v = mu - self.data
        logvar=np.log(var)
        return 0.5*np.sum(v**2/var) + 0.5*np.sum(logvar)
                                
    def _grad(self, mu, var):
        v = self.data - mu
        gmu = v/var
        gvar = -0.5*(v/var)**2 + 0.5/var
        return np.stack([gmu, gvar])
     
    def _value(self, x):
        y = self.predictor(x)
        mu = y[0, ...]
        var = y[1, ...] 
        return self._val(mu, var)
    
    def _gradient(self, x):
        y = self.predictor(x)
        mu = y[0, ...]
        var = y[1, ...] 
        gy = self._grad(mu, var)
        return self.predictor.adjoint(x , gy)