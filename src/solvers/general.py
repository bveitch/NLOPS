import time
import numpy as np
from scipy.optimize import minimize

class Callback:

    def __init__(self, printk = None):
        self._iterations = []
        self.printk = printk

    @property
    def iterations(self):
        return self._iterations

    def __call__(self, intermediate_result):
        xk = intermediate_result.x
        fk = intermediate_result.fun
        self._iterations.append({"xk": xk, "fk":fk})
        niterations = len(self._iterations)
        if self.printk is not None and niterations % self.printk == 0:
            print(f"iteration {niterations} : {fk}")
    
class GeneralSolver:

    def __init__(self, objfn, method='BFGS', niter=10):
        self.objfn = objfn
        self.method = method
        self.niter = niter
        self.callback = Callback(10)
        self.runtime = None

    def solve(self, x0=None):
        if x0 is None:
            xsize = self.objfn.xsize
            x0 = np.zeros(xsize)
        else:
            assert x0.size == self.objfn.xsize
        if self.niter ==0:
            xsol = self.objfn.gradient(x0)
            return self.objfn.unravel(xsol)
        
        start = time.time()
        res = minimize(self.objfn, x0, method=self.method, jac=self.objfn.gradient, callback=self.callback)
        stop = time.time()
        self.runtime = stop - start
        xsol = res.x
        return self.objfn.unravel(xsol)
    
    @property
    def iterations(self):
        return self.callback.iterations
