import time
import numpy as np
from abc import ABC, abstractmethod
from src.solvers.direct import direct_solve

class BaseADMM(ABC):

    def __init__(self, xshape, A, b, l, rho, niter=10, xtol =1.0e-6):
        self.xshape = xshape
        self.A = A
        self.b = b
        self.l = l
        self.rho = rho
        self.niter = niter
        self.xtol = xtol
        self.runtime = None

    
    @classmethod
    def from_objfn(cls, objfn,  l, rho, niter=10, xtol =1.0e-6):
        xshape = objfn.xshape
        A = objfn.op._M
        b = objfn.d
        return cls(xshape, A, b, l, rho, niter, xtol)

    @abstractmethod
    def _update_x(self, z, u):
        return NotImplemented
    
    @abstractmethod
    def _update_z(self, x, u):
        return NotImplemented
    
    def _update_u(self, x, z ,u):
        return u + x - z
        
    def solve(self):
        z = np.zeros(self.xshape)
        u = np.zeros(self.xshape)
        xprev = np.zeros(self.xshape)
        ts = time.time()
        for iter in range(self.niter):
            x = self._update_x( z, u)
            z = self._update_z( x, u)
            u = self._update_u(x,z,u)
            dx = 0.5*np.sum((x-xprev)**2)
            if dx < self.xtol:
                break
            xprev = x
        self.runtime = time.time() - ts
        return x

def shrinkage(x, u, l=None):
    if l is not None:
        if u < l:
            return shrinkage(x, l, u) 
        yp = np.where(x>u, x-u, 0)
        ym = np.where(x<l, x-l, 0)
        return yp + ym
    else:
        return shrinkage(x, u, -u)

class LassoADMM(BaseADMM):

    def _update_x(self, z, u):
        return direct_solve(self.A, self.b, self.rho, z-u)

    def _update_z(self, x, u):
        return shrinkage(x+u, self.l/self.rho)
    
def bound(x, l, u, p):
    assert p >= 0, f"bound: {p=} < 0"
    if l > u:
        return bound(x, u, l,p)
    return np.where(x<u, l+shrinkage(x, l, l-p), u+shrinkage(x, u+p, u))

class BoundADMM(BaseADMM):

    def __init__(self, xshape, A, b, l, rho, niter, xtol=1.0e-6, lower=0, upper=1):
        super().__init__(xshape, A, b, l, rho, niter, xtol)
        self.lower = lower
        self.upper = upper

    @classmethod
    def from_objfn(cls, objfn,  l, rho, niter=10, xtol =1.0e-6, lower=0, upper=1):
        xshape = objfn.xshape
        A = objfn.op._M
        b = objfn.d
        return cls(xshape, A, b, l, rho, niter, xtol, lower, upper)

    def _update_x(self, z, u):
        return direct_solve(self.A, self.b, self.rho, z-u)

    def _update_z(self, x, u):
        return bound(x+u, self.lower, self.upper, self.l/self.rho)


