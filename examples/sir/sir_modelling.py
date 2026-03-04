import numpy as np
import numpy.typing as npt
from src.operators.base import NLBase
from examples.sir.sir_update import SIRUpdate

class SIRSampler:

    @staticmethod
    def f(params):
        (tp, fp, sir) = params
        i =sir[1]
        return tp*i + fp*(1-i)

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

def unpack_params(params, sample_infectives = False, sir_is_base=True):
    if sample_infectives:
        [s0, i0, a, b, tp, fp] = params.tolist() 
    else:
        [s0, i0, a, b] = params.tolist()
        tp, fp = None, None
    r0 = - s0 - i0
    if sir_is_base:
        r0 += 1
    return np.array([s0, i0, r0]), np.array([a,b]), tp, fp

def pack_params(params, sample_infectives):
    sir, mod, tp, fp = params
    a = mod[0]
    b = mod[1]
    s0 = sir[0] - sir[2]
    i0 = sir[1] - sir[2]
    if sample_infectives:
        return np.array([s0, i0, a, b,  tp, fp])
    else:
        return np.array([s0, i0, a, b]) 

class SIRModel(NLBase):

    def __init__(self, T: float, dt: float, Tsub: float, sample_i=False):
        self._T = T
        self._Tsub = Tsub
        self._dt = dt
        self._nt = int(T/dt)
        self._ntsub = int(T/Tsub)
        self._jtsub = int(Tsub/dt)
        self._sir_size = SIRUpdate.sir_size()
        self._mod_size = SIRUpdate.mod_size()
        self._sample_i = sample_i
        if self._sample_i:
            self._dsize = 1
            input_size = self._sir_size + self._mod_size + 1
        else:
            self._dsize = self._sir_size
            input_size = self._sir_size - 1 + self._mod_size
        super().__init__(input_shape =  (input_size), 
                         output_shape = (self._ntsub,self._dsize),
                         name = "SIRMod")
    
    def create_sampler(self):
        return SIRModel(T=self._T, 
                        dt=self.dt, 
                        Tsub=self._Tsub, 
                        sample_i=True)
    
    @property
    def dt(self) -> int:
        return self._dt
    
    @property
    def nt(self) -> int:
        return self._nt
    
    @property
    def tvalues(self, T0 = 0., units = "days") -> int:
        dtype = np.dtype(float, metadata={"units": units})
        return np.linspace(T0, self._T + T0, num = self._ntsub, dtype=dtype)
    
    @classmethod
    def from_tvalues(cls, tvalues, dt, sample_i):
        T = tvalues.max() - tvalues.min()
        ntsub = len(tvalues)
        Tsub = T/ntsub
        return cls(T =T, dt=dt, Tsub = Tsub, sample_i=sample_i)
    
    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self.input_shape, f"SIRModel: {shape=} != {self.input_shape}"
        else:
            assert shape == self.output_shape, f"SIRModel: {shape=} != {self.output_shape}"

    def _fwd_nl(self, params:npt.NDArray) ->npt.NDArray:
        sir0, mod, tp, fp = unpack_params(params, 
                                          sample_infectives=self._sample_i, 
                                          sir_is_base=True)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod)
        for it in range(self._nt):
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                if self._sample_i:
                    data[jt,:] = SIRSampler.f(params = (tp, fp, sir0))
                else:    
                    data[jt,:] = sir0
            sir0 += self._dt*updater.f(sir0)
            if not np.isfinite(sir0).all():
                return data.fill(np.nan)
        return data
    
    def _fwd_lin(self, params:npt.NDArray, dparams:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = unpack_params(params, 
                                            sample_infectives=self._sample_i, 
                                            sir_is_base=True)
        dsir, dmod, dtp, dfp = unpack_params(dparams, 
                                            sample_infectives=self._sample_i, 
                                            sir_is_base=False)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod0)
        for it in range(self._nt):
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                if self._sample_i:
                    data[jt,:] = SIRSampler.df_lin(params0 = (tp0, fp0, sir0), 
                                                   dparams = (dtp, dfp, dsir))
                else:
                    data[jt,:] = dsir
            dsir += self._dt*updater.df_dsir(sir0,dsir)
            dsir += self._dt*updater.df_dmod(sir0,dmod)
            sir0 += self._dt*updater.f(sir0)
            if not np.isfinite(sir0).all():
                return data.fill(np.nan)
        return data
    
    def _adj_lin(self, params:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = unpack_params(params, 
                                            sample_infectives=self._sample_i, 
                                            sir_is_base=True)
        updater = SIRUpdate(*mod0)
        dtp = 0
        dfp = 0
        dmod = np.zeros(self._mod_size)
        dsir = np.zeros(self._sir_size)
        sirdata = np.zeros((self.nt,self._sir_size))
        for it in range(self.nt):
            sirdata[it,:]= sir0
            sir0 += self._dt*updater.f(sir0)
            if np.isnan(sir0).any():
                dmod.fill(np.nan)
                return self.pack_params((dsir, dmod, dtp, dfp), 
                                        sample_infectives=self._sample_i)

        for it in reversed(range(self._nt)):
            sir0 = sirdata[it,:]
            dmod += self.dt*updater.dmod_df( sir0 ,dsir)
            dsir += self.dt*updater.dsir_df( sir0, dsir)
            if(it % self._jtsub==0):
                jt=int(it/self._jtsub)
                if self._sample_i:
                    dparams = SIRSampler.df_adj(params0 = (tp0, fp0, sir0), 
                                                data = data[jt,0])
                    dtp += dparams[0]
                    dfp += dparams[1]
                    dsir += dparams[2]
                else:
                    dsir += data[jt,:]
        return pack_params((dsir, dmod, dtp, dfp), 
                           sample_infectives=self._sample_i)
  
def sample(pinfectives, tp, fp, nsamples, seed =1000):
    rng = np.random.default_rng(seed=seed)
    nt = pinfectives.shape[0]
    p = tp * pinfectives + (1-pinfectives) * fp
    true_samples  = rng.binomial(nsamples, p, size=nt)
    return true_samples