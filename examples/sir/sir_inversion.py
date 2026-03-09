import typing
import numpy as np
import numpy.typing as npt
from examples.sir.sir_modelling import SIRModelling, SIRFixedModelling
from src.objectives.base import L2ObjectiveFn
from src.solvers.Solve import GeneralSolver

def sir_objfn(data: npt.NDArray, 
              sirmod: SIRModelling) -> L2ObjectiveFn:
    datashape = data.shape 
    assert sirmod.output_shape == datashape, f"{sirmod.output_shape=} doesnt match {datashape=}"
    mod_shape = sirmod.input_shape
    objfn = L2ObjectiveFn(mod_shape, operator = sirmod, data = data)
    return objfn

class SIRInversion:

    def __init__(self, sirmod, data):
        self.sirmod = sirmod
        sirobjfn = sir_objfn(data, self.sirmod)
        self._solver = GeneralSolver(sirobjfn)

    def create_input(self, params):
        si_size, m_size, _ = self.sirmod.mdims
        input_size = si_size + m_size
        return params[:input_size]
    
    def __call__(self, x0):
        params = self._solver.solve(x0)
        input = self.create_input(params)
        print(input)
        simulator = SIRModelling(self.sirmod.T,
                                 self.sirmod.dt,
                                 self.sirmod.Tsub)
        return simulator(input), params

class SIRFixedInversion(SIRInversion):

    def __init__(self, sirfixedmod: SIRFixedModelling, data):
        self.model = sirfixedmod.model0
        super().__init__(sirfixedmod, data)

    @classmethod
    def from_sirmod(cls, sirmod: SIRModelling, data, model0:npt.NDArray):
        sirfixedmod = SIRFixedModelling.create_fixed_mod(sirmod, model0)
        return cls(sirfixedmod, data)

    def create_input(self, params):
        si_size, _ = self.sirmod.mfixed_dims
        input_size = si_size
        return np.concatenate((params[:input_size], self.model))

        
