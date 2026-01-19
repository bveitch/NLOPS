import numpy as np
import numpy.typing as npt
from src.NLBase import NLBase

class SIRModel(NLBase):

    def __init__(self, nt: int, dt: float):
        self._keys = ["I0", "beta", "gamma"]
        self._compartments = ["s", "i", "r"]
        self._popsize = 1.0*10^7
        self._nt = nt
        self._dt = dt
        self._ushape=(self.ncompartments,self._nt)

    @property
    def ncompartments(self) -> int:
        return len(self._compartments)
    
    @property
    def nt(self) -> int:
        return self._nt
    
    def R0(self, input:npt.NDArray):
        return self._popsize*input[1]/input[2]

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        I0 = input[0]
        R0 = self.R0(input)
        u = np.zeros(self._ushape)
        u[0,0] = I0/self._popsize 
        for it in range(self._nt):
            w = [-R0*u[1,it,], (R0*u[0,it]-1),u[2,it]]
            u[:, it+1] += self._dt*w*u[:,it] 
        return u[1,:]
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        R0 = self.R0(input)
        I0 = input[0]
        dI0 = dinput[0]
        u0 = np.zeros(self._ushape)
        du = np.zeros(self._ushape)
        u0[0,0] = I0/self._popsize 
        du[0,0] = dI0/self._popsize
        for it in range(self._nt):
            dw = np.array([-R0*du[1,it,], (R0*du[0,it]-1),du[2,it]])
            u0[:, it+1] += self._dt*w0*u0[:,it] 
            du[:, it+1] = dt*w0*du[:,it] + dt*dw*u0[:,it] 
        return du[1,:]
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
       raise NotImplemented
    
class BinomialModel(NLBase):
     
    def __init__(self, nsamples):
        self.N = nsamples
    
    def sample(self, p, q, base_infectives):
        nt = len(base_infectives)
        p =  self._fwd_nl(self, p,q, base_infectives)
        samples = np.random.binomial(self.N, p, size=(self.N, nt))
        return np.sum(samples, axis=0, keepdims=False)
     
    def _predict(p,q, infectives):
        return p*infectives + q*(1-infectives)
    
    def _predict_lin(p, q, infectives, dp, dq, dinfectives):
        dv = dp*infectives + dq*(1-infectives)
        dv+=(p-q)*dinfectives
        return dv
    
    def _predict_adj(p, q, infectives, dv):
        dp = np.sum(dv, infectives)
        dq = np.sum(dv, infectives)
        dinfectives = (p-q)*dv
        return [dp, dq, dinfectives]
    
    def _fwd_nl(self, params):
        p = params[0]
        q = params[1]
        sir_params = params[2:]
        base_infectives = self.sir_mod(sir_params)[...,1]
        return self._predict(p, q, base_infectives)
    
    def _fwd_lin(self, params, dparams):
        p = params[0]
        q = params[1]
        sir_params = params[2:]
        dp = dparams[0]
        dq = dparams[1]
        dsir_params = dparams[2:]
        base_infectives = self.sir_mod(sir_params)[...,1]
        dbase_infectives = self.sir_mod.forward(sir_params, dsir_params)[...,1]
        return self._predict_lin(p, q, base_infectives, dp, dq, dbase_infectives)
    
    def _adj_lin(self, params, dpbin):
        p = params[0]
        q = params[1]
        sir_params = params[2:]
        base_infectives = self.sir_mod(sir_params)[...,1]
        [dp,dq, dinfectives] = self._predict_adj(p, q, base_infectives, dpbin)
        dsir_params = self.sir_mod.adjoint(sir_params, dinfectives)
        return np.array([dp, dq, dsir_params])

class BinomialToNormal(NLBase):

    def _fwd_nl(self, pbin):
        mu = self.N*pbin
        var = self.N*pbin*(1-pbin)
        return np.stack([mu, var])
    
    def _fwd_lin(self, pbin, dpbin):
        dmu = self.N*dpbin
        dvar = self.N*dpbin*(1-2.*pbin)
        return np.stack([dmu, dvar])
    
    def _adj_lin(self, pbin, dnormal):
        dmu = dnormal[...,0]
        dvar = dnormal[..., 1] 
        dpbin = dmu+ (1-2*pbin)*dvar 
        return self.N*dpbin
    
class SIRFit(ObjectiveFunction):

    def __init__(self, Nsamples, data):
        self.data = data
        self.N = Nsamples
        self.modeller = SIR()

    def mu(self, pars):
        return self.N*self.modeller(pars)

    def sigma2(self, pars):
        p = self.modeller(pars)
        return self.N*p*(1-p)

    def _value(self, x):
        y = self.modeller(x)
        mu = y[0]
        var = y[1] 
        v = mu - self.data
        logvar=np.log(var)
        return 0.5*np.sum(v**2/var) + 0.5*np.sum(logvar)
                                
    def _gradient(self, x):
        y = self.modeller(x)
        mu = y[0]
        var = y[1] 
        v = self.data - mu
        gmu = v/var
        gvar = -0.5*(v/var)**2 + 0.5/var
        gy = np.stack([gmu, gvar])
        return self.modeller.adjoint(x,gy)