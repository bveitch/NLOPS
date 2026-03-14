import typing
import numpy as np
import numpy.typing as npt
from examples.sir.sir_modelling import SIRModelling, SIRFixedModelling
from src.objectives.base import L2ObjectiveFn
from src.objectives.sum_objective import SumObjectiveFn
from src.objectives.normal_fit import BinomialFit
from src.solvers.Solve import GeneralSolver

def sir_objfn(data: npt.NDArray, 
              sirmod: SIRModelling) -> L2ObjectiveFn:
    datashape = data.shape 
    assert sirmod.output_shape == datashape, f"{sirmod.output_shape=} doesnt match {datashape=}"
    mod_shape = sirmod.input_shape
    return L2ObjectiveFn(mod_shape, operator = sirmod, data = data)

def sir_binomial_objfn(data: npt.NDArray, 
              sirmod: SIRModelling) -> L2ObjectiveFn:
    datashape = data.shape 
    assert sirmod.output_shape == datashape, f"{sirmod.output_shape=} doesnt match {datashape=}"
    mod_shape = sirmod.input_shape
    return BinomialFit(mod_shape, operator = sirmod, data = data)

class SIRInversion:

    def __init__(self, sirmod, data, datafit="L2", regularizer:tuple[npt.NDArray, np.float64]|None =None):
        self.sirmod = sirmod
        if datafit == "L2":
            dobjfn = sir_objfn(data, self.sirmod)
        elif datafit == "Binomial":
            dobjfn = sir_binomial_objfn(data, self.sirmod)
        else:
            raise ValueError(f"{datafit} is not defined")
        if regularizer is not None:
            xshape = dobjfn.xshape
            mreg, vreg = regularizer
            robjfn = L2ObjectiveFn(xshape, operator = None, data = mreg)
            objfn =  SumObjectiveFn([dobjfn, robjfn], [1.0, vreg])
        else:
            objfn = dobjfn
        self._solver = GeneralSolver(objfn)
        self._results = {
            "starting_model" : None, 
            "final_model": None,
            "iterations": None,
            "sirdata": None,
            "predicted_samples": None,
            "runtime": None}

    def create_input(self, params):
        si_size, m_size, _ = self.sirmod.mdims
        input_size = si_size + m_size
        return params[:input_size]
    
    def __call__(self, x0):
        self._results["starting_model"] = x0
        params = self._solver.solve(x0)
        self._results["final_model"] = params
        self._results["iterations"] = self.iterations
        self._results["runtime"] = self._solver.runtime
        prediction = self.sirmod(params)
        if self.sirmod.sample_i == True:
            input = self.create_input(params)
            simulator = SIRModelling(self.sirmod.T,
                                    self.sirmod.dt,
                                    self.sirmod.Tsub)
            sirprediction = simulator(input)
            self._results["predicted_samples"] = np.squeeze(prediction)
            self._results["sirdata"] = sirprediction
        else:
            self._results["sirdata"] = prediction
        return self._results
    
    @property
    def iterations(self):
        f0 = self._solver.iterations[0]["fk"]
        return [ iter["fk"]/f0 for iter in self._solver.iterations]

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
    
def run_infectives_inversion(samples, sir_sample_mod, initial_condition, starting_model, test_params, **kwargs):
    infectives_data = samples[:,np.newaxis]
    xstart = np.concatenate([initial_condition, starting_model, test_params])

    if kwargs.get("fixed_model", False):
        print("using fixed model")
        sirinv = SIRFixedInversion.from_sirmod(sirmod=sir_sample_mod, 
                                            data=infectives_data,
                                            model0= starting_model)
        xstart = np.concatenate([initial_condition, test_params])
    elif "regularization" in kwargs.keys():
        print(f"using regularization: {kwargs["regularization"]}")
        sirinv = SIRInversion(sirmod=sir_sample_mod, data=infectives_data, regularizer=kwargs["regularization"])
    else:
        print("standard mode")
        sirinv = SIRInversion(sirmod=sir_sample_mod, data=infectives_data)
    return sirinv(x0 = xstart)
        

        
