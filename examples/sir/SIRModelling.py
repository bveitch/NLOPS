import numpy as np
import numpy.typing as npt
from src.operators.base import NLBase
from src.operators.sympy_wrapper import SympyWrap

class SIRUpdate:

    exprs =  ["-a*s*i", "a*s*i-b*i", "b*i"]
    vars = ["s", "i"]
    baseop = SympyWrap.from_strings(exprs)
 
    def __init__(self, a, b): 
        self._fsir = SIRUpdate.baseop.partial_eval({"a":a, "b":b})
        self._jac_sir = self._fsir.jac(SIRUpdate.vars)
        self._jac_mod, _ = SIRUpdate.baseop.partial_jac({"a":a, "b":b})

    @classmethod
    def input_size(cls):
        return len(cls.vars)
    
    @classmethod
    def output_size(cls):
        return len(cls.exprs)
    
    @staticmethod
    def mod_size():
        return 2

    def f(self, sir0):
        return self._fsir(sir0, SIRUpdate.vars)
        
    def df_dsir(self, sir0, dsir):
        jac = self._jac_sir(sir0, SIRUpdate.vars)
        return np.dot(jac, dsir)

    def df_dmod(self, sir0, dmod):
        jac = self._jac_mod(sir0, SIRUpdate.vars)
        return np.dot(jac, dmod)

    def dsir_df(self, sir0, df):
        jac = self._jac_sir(sir0, SIRUpdate.vars)
        return np.dot(jac.T, df)

    def dmod_df(self, sir0, df):
        jac = self._jac_mod(sir0, SIRUpdate.vars)
        return np.dot(jac.T, df)

class SIRSampler:

    @staticmethod
    def f(params):
        (tp, fp, sir) = params
        i =sir[1]
        return (tp + fp)*i

    @staticmethod
    def df_lin(params0, dparams):
        (tp0, fp0, sir0) = params0
        (dtp, dfp, dsir) = dparams
        i0 = sir0[1]
        di = dsir[1]
        return dtp*i0 + dfp*(1-i0) + (tp0-fp0)*di

    @staticmethod
    def df_adj(params0, data):
        (tp0, fp0, sir0) = params0
        i0 = sir0[1]
        dtp = i0*data
        dfp = (1-i0)*data
        dsir = np.zeros(sir0.shape)
        dsir[1] = (tp0-fp0)*data
        return (dtp, dfp, dsir)

class SIRModel(NLBase):

    def __init__(self, T: float, dt: float, Tsub: float):
        self._T = T
        self._dt = dt
        self._nt = int(T/dt)
        self._ntsub = int(T/Tsub)
        self._jtsub = int(Tsub/dt)
        self._sir_shape = SIRUpdate.input_size()
        self._mod_shape = SIRUpdate.mod_size()
        self._num_pvalues = 2
        self._dsize = SIRUpdate.output_size()
        super().__init__(input_shape =  (self._sir_shape+ self._mod_shape + self._num_pvalues), 
                         output_shape = (self._ntsub,self._dsize),
                         name = "SIRMod")
    
    @property
    def dt(self) -> int:
        return self._dt
    
    @property
    def nt(self) -> int:
        return self._nt
    
    @property
    def tvalues(self) -> int:
        return np.linspace(0, self._T, num = self._ntsub)
    
    @classmethod
    def nparams(cls):
        return cls._nparams
    
    @classmethod
    def nsir(cls):
        return cls._nsir
    
    def unpack_params(self, params, withN=False):
        [s0, i0, a, b, tp, fp] = params.tolist()  
        r0 = - s0 - i0
        if withN:
            r0 += 1
        return np.array([s0, i0, r0]), np.array([a,b]), tp, fp

    def pack_params(self, params):
        sir, mod, tp, fp = params
        a = mod[0]
        b = mod[1]
        s0 = sir[0] - sir[2]
        i0 = sir[1] - sir[2]
        return np.array([s0, i0, a, b,  tp, fp])

    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self.input_shape, f"SIRModel: {shape=} != {self.input_shape}"
        else:
            assert shape == self.output_shape, f"SIRModel: {shape=} != {self.output_shape}"
        
    def _fwd_nl(self, params:npt.NDArray) ->npt.NDArray:
        sir0, mod, tp, fp = self.unpack_params(params, True)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod)
        for it in range(self._nt):
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                #SIRSampler.f(jt=jt, data = data, params = (tp, fp, sir0))
                data[jt,:] = sir0
            sir0 += self._dt*updater.f(sir0[0:2])
        return data
    
    def _fwd_lin(self, params:npt.NDArray, dparams:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
        dsir, dmod, dtp, dfp = self.unpack_params(dparams)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod0)
        for it in range(self._nt):
            if((it % self.ntsub)==0):
                jt=int(it/self._jtsub)
                #SIRSampler.df_lin(jt, data, (tp0, fp0, sir0), (dtp, dfp, dsir))
                data[jt,:] = dsir
            dsir += self._dt*updater.df_dsir(sir0,dsir)
            dsir += self._dt*updater.df_dmod(sir0,dmod)
            sir0 += self._dt*updater(sir0)
        return data
    
    def _adj_lin(self, params:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
        updater = SIRUpdate(*mod0)
        sirdata = np.zeros((self.nt,self.dsize))
        for it in range(self.nt):
            sirdata[it,:]= sir0
            sir0 += self._dt*updater.f(sir0)

        dtp = 0
        dfp = 0
        dmod = np.zeros(self._nmod)
        dsir = np.zeros(self.dsize)
        for it in reversed(range(self._nt)):
            sir0 = sirdata[it,:]
            dmod += self.dt*updater.dmod_df( sir0 ,dsir)
            dsir += self.dt*updater.dsir_df( sir0, dsir)
            if(it% self.ntsub==0):
                jt=int(it/self._jtsub)
                dsir += data[jt,:]
                #SIRSampler.df_adj(jt, data, (tp0, fp0, sir0), (dtp, dfp, dsir))
        return self.pack_params(dsir, dmod, dtp, dfp)
  
def sample(pinfectives, tp, fp, nsamples):
    nt = pinfectives.shape[0]
    p = tp * pinfectives + (1-pinfectives) * fp
    true_samples  = np.random.binomial(nsamples, p, size=nt)
    return true_samples