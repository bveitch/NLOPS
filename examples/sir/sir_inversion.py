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

def create_objfn(data, sirmod, datafit="L2"):
    if datafit == "L2":
        objfn =  sir_objfn(data, sirmod)
    elif datafit == "Binomial":
        objfn = sir_binomial_objfn(data, sirmod)
    else:
        raise ValueError(f"{datafit} is not defined")
    setattr(objfn, "sirmod", sirmod)
    return objfn
    
class SIRInversion:

    def __init__(self, dobjfn: L2ObjectiveFn, regularizer:tuple[L2ObjectiveFn, np.float64]|None =None):
        self.dobjfn = dobjfn 
        try:
            self.sirmod = getattr(dobjfn, "sirmod")
        except:
            AttributeError(f"sirmod not found in {dobjfn}") 
        if regularizer is not None:
            robjfn, vreg = regularizer
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

    @classmethod
    def from_sirmod(cls, sirmod, data, datafit="L2"):
        dobjfn = create_objfn(data, sirmod, datafit)
        return cls(dobjfn)
    
    def add_model_regularization(self, mregularization:tuple[npt.NDArray, np.float64]):
        xshape = self.dobjfn.xshape
        mreg, vreg = mregularization
        robjfn = L2ObjectiveFn(xshape, operator = None, data = mreg)
        return SIRInversion(self.dobjfn, [robjfn, vreg])

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
        if self._solver.iterations:
            f0 = self._solver.iterations[0]["fk"]
            return [ iter["fk"]/f0 for iter in self._solver.iterations]
        else:
            return None

class SIRFixedInversion(SIRInversion):

    def __init__(self, dobjfn: L2ObjectiveFn, model0: npt.NDArray):
        super().__init__(dobjfn)
        self.model0 = model0
    
    @classmethod
    def from_sirfixedmod(cls, sirfixedmod: SIRFixedModelling, data, datafit="L2"):
        dobjfn = create_objfn(data, sirfixedmod, datafit)
        model0 = sirfixedmod.model0
        return cls(dobjfn, model0)

    @classmethod
    def from_sirmod(cls, sirmod: SIRModelling, data, model0:npt.NDArray, datafit="L2"):
        sirfixedmod = SIRFixedModelling.create_fixed_mod(sirmod, model0)
        return cls.from_sirfixedmod(sirfixedmod, data, datafit=datafit)

    def create_input(self, params):
        si_size, _ = self.sirmod.mfixed_dims
        input_size = si_size
        return np.concatenate((params[:input_size], self.model0))
    
def run_infectives_inversion(samples, sir_sample_mod, initial_condition, starting_model, test_params, **kwargs):
    infectives_data = samples[:,np.newaxis]
    xstart = np.concatenate([initial_condition, starting_model, test_params])

    if kwargs.get("fixed_model", False):
        print("using fixed model")
        sirinv = SIRFixedInversion.from_sirmod(sirmod = sir_sample_mod, 
                                                data = infectives_data,
                                                model0 = starting_model,
                                                datafit = kwargs.get("datafit", "L2"),
                                                )
        xstart = np.concatenate([initial_condition, test_params])
    else:
        print("standard mode")
        sirinv = SIRInversion.from_sirmod(sirmod=sir_sample_mod, 
                                          data=infectives_data,
                                          datafit = kwargs.get("datafit", "L2"),
                                          )

    if "regularization" in kwargs.keys():
        print(f"using regularization: {kwargs["regularization"]}")
        sirinv = sirinv.add_model_regularization(mregularization=kwargs["regularization"])

    return sirinv(x0 = xstart)
        

        
