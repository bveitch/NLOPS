import numpy as np
import numpy.typing as npt
from src.operators.base import NLBase
from examples.sir.sir_update import SIRUpdate

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

    def __init__(self, T: float, dt: float, Tsub: float, sample=False):
        self._T = T
        self._dt = dt
        self._nt = int(T/dt)
        self._ntsub = int(T/Tsub)
        self._jtsub = int(Tsub/dt)
        self._sir_size = SIRUpdate.sir_size()
        self._mod_size = SIRUpdate.mod_size()
        self._num_pvalues = 2
        self._dsize = self._sir_size
        self._sample = sample
        super().__init__(input_shape =  (self._sir_size - 1 + self._mod_size + self._num_pvalues), 
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
                if not self._sample:
                    #SIRSampler.f(jt=jt, data = data, params = (tp, fp, sir0))
                    data[jt,:] = sir0
                else:
                    data[jt,:] = SIRSampler.f(params = (tp, fp, sir0))
            sir0 += self._dt*updater.f(sir0)
        return data
    
    def _fwd_lin(self, params:npt.NDArray, dparams:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
        dsir, dmod, dtp, dfp = self.unpack_params(dparams)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod0)
        for it in range(self._nt):
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                if not self._sample:
                    data[jt,:] = dsir
                else:
                    data[jt,:] = SIRSampler.df_lin(params = (tp0, fp0, sir0), dparams=(dtp, dfp, dsir))
            dsir += self._dt*updater.df_dsir(sir0,dsir)
            dsir += self._dt*updater.df_dmod(sir0,dmod)
            sir0 += self._dt*updater.f(sir0)
        return data
    
    def _adj_lin(self, params:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
        updater = SIRUpdate(*mod0)
        sirdata = np.zeros((self.nt,self._dsize))
        for it in range(self.nt):
            sirdata[it,:]= sir0
            sir0 += self._dt*updater.f(sir0[0:2])

        dtp = 0
        dfp = 0
        dmod = np.zeros(self._mod_size)
        dsir = np.zeros(self._dsize)
        for it in reversed(range(self._nt)):
            sir0 = sirdata[it,:]
            dmod += self.dt*updater.dmod_df( sir0 ,dsir)
            dsir += self.dt*updater.dsir_df( sir0, dsir)
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                if not self._sample:
                    dsir += data[jt,:]
                else:
                    dparams = SIRSampler.df_adj(params=(tp0, fp0, sir0), data = data)
                    dtp += dparams[0]
                    dfp += dparams[1]
                    dsir += dparams[2]
                #SIRSampler.df_adj(jt, data, (tp0, fp0, sir0), (dtp, dfp, dsir))
        return self.pack_params((dsir, dmod, dtp, dfp))
  
def sample(pinfectives, tp, fp, nsamples, seed =1000):
    rng = np.random.default_rng(seed=seed)
    nt = pinfectives.shape[0]
    p = tp * pinfectives + (1-pinfectives) * fp
    true_samples  = rng.binomial(nsamples, p, size=nt)
    return true_samples