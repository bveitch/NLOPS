import numpy as np
from scipy.optimize import minimize

class Callback:

    def __init__(self):
        self._iterations = []

    @property
    def iterations(self):
        return self._iterations

    def __call__(self, xk):
        self._iterations.append(xk)
        print(f"{len(self._iterations)} : {xk}")
    
class GeneralSolver:

    def __init__(self, objfn, method='BFGS', niter=10):
        self.objfn = objfn
        self.method = method
        self.niter = niter

    def solve(self, x0=None):
        if x0 is None:
            xsize = self.objfn.xsize
            x0 = np.zeros(xsize)
        if self.niter ==0:
            xsol = self.objfn.gradient(x0)
            return self.objfn.unravel(xsol)
        res = minimize(self.objfn, x0, method=self.method, jac=self.objfn.gradient, callback=Callback())
        xsol = res.x
        return self.objfn.unravel(xsol)
