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

class SIRModel(NLBase):

    def __init__(self, T0: float, T1: float, dt: float, Tsub: float, units = "days", sample_i=False):
        self._T0 = T0
        self._T1 = T1
        T = T1-T0
        self._dt = dt
        self._nt = int(T/dt)
        self._ntsub = int(T/Tsub)
        self._jtsub = int(Tsub/dt)
        # T = jtsub * ntsub * dt
        # ntsub = nt/jtsub = (T/Tsub)
        self._units = units
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
    
    @classmethod
    def from_tvalues(cls, tvalues, dt, sample_i = False):
        T0 = tvalues[0]
        T1 = tvalues[-1]
        ntsub = len(tvalues)
        Tsub = (T1 - T0)/ntsub
        units = tvalues.dtype.metadata["units"]
        return cls(T0=T0, T1=T1, dt=dt, Tsub = Tsub, units = units, sample_i = sample_i)

    def create_sampler(self):
        return self.__init__(T0=self._T0, 
                             T1=self._T1, 
                             dt=self.dt, 
                             Tsub=self._Tsub, 
                             units=self._units, 
                             sample_i=True)
    
    @property
    def dt(self) -> int:
        return self._dt
    
    @property
    def nt(self) -> int:
        return self._nt
    
    @property
    def tvalues(self) -> int:
        dtype = np.dtype(float, metadata={"units": self._units})
        return np.linspace(self._T0, self._T1, num = self._ntsub, dtype=dtype)
    
    def unpack_params(self, params, withN=False):
        if self._sample_i:
            [s0, i0, a, b, tp, fp] = params.tolist() 
        else:
            [s0, i0, a, b] = params.tolist()
            tp, fp = None, None
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
        if self._sample_i:
            return np.array([s0, i0, a, b,  tp, fp])
        else:
            return np.array([s0, i0, a, b]) 

    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self.input_shape, f"SIRModel: {shape=} != {self.input_shape}"
        else:
            assert shape == self.output_shape, f"SIRModel: {shape=} != {self.output_shape}"
        
    def _catch_nan(self, it, data):
        jt=int(it/self._jtsub) + 1
        data[jt:, :] = np.finfo(data.dtype).max

    def _fwd_nl(self, params:npt.NDArray) ->npt.NDArray:
        sir0, mod, tp, fp = self.unpack_params(params, True)
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
            if np.isnan(sir0).any():
                #self._catch_nan(it, data)
                return data
        return data
    
    def _fwd_lin(self, params:npt.NDArray, dparams:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
        dsir, dmod, dtp, dfp = self.unpack_params(dparams)
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
            if np.isnan(sir0).any():
                #self._catch_nan(it, data)
                return data
        return data
    
    def _adj_lin(self, params:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        sir0, mod0, tp0, fp0 = self.unpack_params(params, True)
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
                dmod.fill(np.finfo(dmod.dtype).max)
                return self.pack_params((dsir, dmod, dtp, dfp))

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
        return self.pack_params((dsir, dmod, dtp, dfp))
  
def sample(pinfectives, tp, fp, nsamples, seed =1000):
    rng = np.random.default_rng(seed=seed)
    nt = pinfectives.shape[0]
    p = tp * pinfectives + (1-pinfectives) * fp
    true_samples  = rng.binomial(nsamples, p, size=nt)
    return true_samples