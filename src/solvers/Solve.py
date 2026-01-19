import numpy as np
from scipy.optimize import minimize

class GeneralSolver:

    def __init__(self, objfn, method='Newton-CG', niter=10):
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
        res = minimize(self.objfn, x0, method=self.method, jac=self.objfn.gradient)
        xsol = res.x
        return self.objfn.unravel(xsol)
