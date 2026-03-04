import numpy as np
import numpy.typing as npt
from examples.sir.sir_modelling import SIRModel
from src.objectives.base import L2ObjectiveFn
from src.solvers.Solve import GeneralSolver

def sir_objfn(data: npt.NDArray, 
              sirmod: SIRModel) -> L2ObjectiveFn:
    datashape = data.shape 
    assert sirmod.output_shape == datashape, f"{sirmod.output_shape=} doesnt match {datashape=}"
    mod_shape = sirmod.input_shape
    objfn = L2ObjectiveFn(mod_shape, operator = sirmod, data = data)
    return objfn

class SIRInversion:

    def __init__(self, tvalues, data, dt):
        self._tvalues = tvalues
        self._dt = dt
        ntsub, dsize = data.shape 
        assert ntsub == len(tvalues), f"{ntsub=} doesnt match {len(tvalues)=}"
        sample_i = (dsize ==1)
        sirmod = SIRModel.from_tvalues(tvalues, dt, sample_i)
        sirobjfn = sir_objfn(data, sirmod)
        self._solver = GeneralSolver(sirobjfn)

    def __call__(self):
        params = self._solver.solve()
        sirmod = SIRModel.from_tvalues(self._tvalues, self._dt, False)
        return sirmod(params[0:4]), params
        
