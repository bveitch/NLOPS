import numpy as np
import numpy.typing as npt
from src.operators.base import NLBase
from examples.sir.sir_update import SIRUpdate
from examples.sir.sir_params import (
    SISIZE, 
    SIRSIZE, 
    SIRMODSIZE, 
    SiModPtfParams, 
    SiModPtfParser,
    SiPtfParams,
    SiPtfParser,
    si0_to_sir0,
    dsir_to_dsi,
    dsi_to_dsir
)

class SIRSampler:
    
    @staticmethod
    def ptf_size():
        return 2
    
    @staticmethod
    def data_size():
        return 1
    
    @staticmethod
    def f(params):
        (ptf, sir) = params
        i =sir[1]
        return ptf[0]*i + ptf[1]*(1-i)

    @staticmethod
    def df_lin(params0, dparams):
        (ptf0, sir0) = params0
        (dptf, dsir) = dparams
        i0 = sir0[1]
        di = dsir[1]
        return ( dptf[0]*i0 + dptf[1]*(1-i0) 
                + (ptf0[0]-ptf0[1])*di )

    @staticmethod
    def df_adj(params0, data):
        (dptf0, sir0) = params0
        i0 = sir0[1]
        dpt = i0*data
        dpf = (1-i0)*data
        dsir = np.zeros(sir0.shape)
        dsir[1] = (dptf0[0]-dptf0[1])*data
        return (np.array([dpt, dpf]), dsir)

class SIRModelling(NLBase):
        
    def __init__(self, T: float, dt: float, Tsub: float, sample_i=False):
        self.T = T
        self.Tsub = Tsub
        self.dt = dt
        self.nt = int(T/dt)
        self.ntsub = int(T/Tsub)
        self.jtsub = int(Tsub/dt)
        self.updater = SIRUpdate
        self.sample_i = sample_i
        if sample_i:
            self.mdims = (SISIZE, SIRMODSIZE, SIRSampler.ptf_size())
            data_size = SIRSampler.data_size()
        else:
            self.mdims = (SISIZE, SIRMODSIZE, 0)
            data_size = SIRSIZE 
        input_size = sum(self.mdims)
        super().__init__(input_shape =  (input_size), 
                         output_shape = (self.ntsub,data_size),
                         name = "SIRModelling")

    def tvalues(self, T0 = 0., units = "days") -> npt.NDArray:
        tvalues_dtype = np.dtype(float, metadata={"units": units})
        print(tvalues_dtype.metadata)
        return np.linspace(T0, self.T + T0, num = self.ntsub, dtype=tvalues_dtype)
    
    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self.input_shape, f"{self.name}: {shape=} != {self.input_shape}"
        else:
            assert shape == self.output_shape, f"{self.name}: {shape=} != {self.output_shape}"
    
    def create_sampler(self):
        return SIRModelling(T=self.T, 
                        dt=self.dt, 
                        Tsub=self.Tsub, 
                        sample_i=True)
    
    def _fwd_nl_tuple(self, params:SiModPtfParams) ->npt.NDArray:
        si0, mod, ptf = params.si, params.mod, params.ptf
        sir0 = si0_to_sir0(si0)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod)
        for it in range(self.nt):
            if(it % self.jtsub==0):
                jt=int(it/self.jtsub)
                if self.sample_i:
                    data[jt,:] = SIRSampler.f(params = (ptf, sir0))
                else:    
                    data[jt,:] = sir0
            sir0 += self.dt*updater.f(sir0)
            if not np.isfinite(sir0).all():
                data.fill(0.)
                return data
        return data
    
    def _fwd_lin_tuple(self, params0:SiModPtfParams, dparams:SiModPtfParams) ->npt.NDArray:
        si0, mod0, ptf0 = params0.si, params0.mod, params0.ptf
        dsi, dmod, dptf = dparams.si, dparams.mod, dparams.ptf
        sir0 = si0_to_sir0(si0)
        dsir = dsi_to_dsir(dsi)
        data = np.zeros(self.output_shape)
        updater = SIRUpdate(*mod0)
        for it in range(self.nt):
            if(it % self.jtsub==0):
                jt=int(it/self.jtsub)
                if self.sample_i:
                    data[jt,:] = SIRSampler.df_lin(params0 = (ptf0, sir0), 
                                                   dparams = (dptf, dsir))
                else:
                    data[jt,:] = dsir
            dsir += self.dt*updater.df_dsir(sir0,dsir)
            dsir += self.dt*updater.df_dmod(sir0,dmod)
            sir0 += self.dt*updater.f(sir0)
            if not np.isfinite(sir0).all():
                data.fill(0)
                return data
        return data
    
    def _adj_lin_tuple(self, params0:SiModPtfParams, data:npt.NDArray) ->npt.NDArray:
        si0, mod0, ptf0 = params0.si, params0.mod, params0.ptf
        sir0 = si0_to_sir0(si0)
        updater = SIRUpdate(*mod0)
        if ptf0 is not None:
            dptf = np.zeros((ptf0.size))
        else:
            dptf = None
        dmod = np.zeros(SIRMODSIZE)
        dsir = np.zeros(SIRSIZE)
        sirdata = np.zeros((self.nt,SIRSIZE))
        for it in range(self.nt):
            sirdata[it,:]= sir0
            sir0 += self.dt*updater.f(sir0)
            if np.isnan(sir0).any():
                dmod.fill(np.nan)
                dsi = dsir_to_dsi(dsir)
                return SiModPtfParams(dsi, dmod, dptf)

        for it in reversed(range(self.nt)):
            sir0 = sirdata[it,:]
            dmod += self.dt*updater.dmod_df( sir0 ,dsir)
            dsir += self.dt*updater.dsir_df( sir0, dsir)
            if(it % self.jtsub==0):
                jt=int(it/self.jtsub)
                if self.sample_i:
                    dparams = SIRSampler.df_adj(params0 = (ptf0, sir0), 
                                                data = data[jt,0])
                    dptf += dparams[0]
                    dsir += dparams[1]
                else:
                    dsir += data[jt,:]
            dsi = dsir_to_dsi(dsir)
        return SiModPtfParams(dsi, dmod, dptf)

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        params = SiModPtfParser.from_numpy(input, dims =self.mdims)
        return self._fwd_nl_tuple(params) 
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        params0 = SiModPtfParser.from_numpy(input, dims = self.mdims)
        dparams = SiModPtfParser.from_numpy(dinput, dims = self.mdims)
        return self._fwd_lin_tuple(params0, dparams) 
    
    def _adj_lin(self, input:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        params0 = SiModPtfParser.from_numpy(input, dims = self.mdims)
        dparams = self._adj_lin_tuple(params0, data)
        return SiModPtfParser.to_numpy(dparams, dims = self.mdims)
    
  
class SIRFixedModelling(SIRModelling):

    def __init__(self, T: float, dt: float, Tsub: float, model0:npt.NDArray, sample_i=False):
        super().__init__(T=T, dt=dt, Tsub=Tsub, sample_i = sample_i)
        assert model0.size == SIRMODSIZE
        self.model0 = model0 
        if sample_i:
            self.mfixed_dims = (SISIZE, SIRSampler.ptf_size())
        else:
            self.mfixed_dims = (SISIZE, 0)
        input_size = sum(self.mfixed_dims)
        self.input_shape = (input_size)
        self.name = "SIRFixedModelling"

    @classmethod
    def create_fixed_mod(cls, sirmod: SIRModelling, model0:npt.NDArray):
        return cls(T=sirmod.T, 
                dt=sirmod.dt, 
                Tsub=sirmod.Tsub, 
                model0 = model0,
                sample_i=sirmod.sample_i)
    
    def create_sampler(self):
        return SIRFixedModelling(T=self.T, 
                        dt=self.dt, 
                        Tsub=self.Tsub, 
                        sample_i=True)

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        params = SiPtfParser.from_numpy(input, dims = self.mfixed_dims)
        params = SiModPtfParams(params.si, self.model0, params.ptf)
        return self._fwd_nl_tuple(params)

    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        params0 = SiPtfParser.from_numpy(input, dims = self.mfixed_dims)
        dparams = SiPtfParser.from_numpy(dinput, dims = self.mfixed_dims)
        params0 = SiModPtfParams(params0.si, self.model0, params0.ptf)
        dparams = SiModPtfParams(dparams.si, np.zeros(self.model0.size), dparams.ptf)
        return self._fwd_lin_tuple(params0, dparams) 
    
    def _adj_lin(self, input:npt.NDArray, data:npt.NDArray) ->npt.NDArray:
        params0 = SiPtfParser.from_numpy(input, dims = self.mfixed_dims)
        params0 = SiModPtfParams(params0.si, self.model0, params0.ptf)
        dparams = self._adj_lin_tuple(params0, data)
        dparams = SiPtfParams(dparams.si, dparams.ptf)
        return SiPtfParser.to_numpy(dparams, dims = self.mfixed_dims)
 
def simulate_epidemic(T:float , dt:float, Tsub:float, Npopulation:float, sir_model:npt.NDArray):
    i0=1.0/Npopulation
    s0 = 1-i0
    initial_si = np.array([s0, i0])
    model = np.concatenate([initial_si, sir_model])
    modeller = SIRModelling(T, dt, Tsub)
    tvalues = modeller.tvalues()
    return modeller(model), tvalues, modeller

def sample(pinfectives, tp, fp, nsamples, seed = 1000):
    rng = np.random.default_rng(seed=seed)
    nt = pinfectives.shape[0]
    p = tp * pinfectives + (1-pinfectives) * fp
    true_samples  = rng.binomial(nsamples, p, size=nt)
    return true_samples