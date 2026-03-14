import numpy as np
import numpy.typing as npt
from src.operators.base import NLBase

class NLVector(NLBase):

    def __init__(self, 
                 nl_operators: list[NLBase],
                 name:str = "NLVector"):
        nops = len(nl_operators)
        input_shape = nl_operators[0].input_shape
        output_shape = nl_operators[0].output_shape
        for op in nl_operators:
            assert op.input_shape == input_shape, f"{op.name}: {op.input_shape} != {input_shape}"
            assert op.output_shape == output_shape, f"{op.name}: {op.output_shape} != {output_shape}"
        output_shape = (nops, *output_shape)
        super().__init__(input_shape, output_shape, name)
        self.operators = nl_operators
            
    def print_ops(self) -> str:
        return [ op.name for op in reversed(self.operators)]
    
    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self.input_shape, f"{self.name}: {shape=} != {self.input_shape}"
        else:
            assert shape == self.output_shape, f"{self.name}: {shape=} != {self.output_shape}"

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        output = [ operator(input) for operator in self.operators]
        return np.stack(output, axis=0)
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        doutput = [ operator.linear(input, dinput) for operator in self.operators]
        return np.stack(doutput, axis=0)
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
        dinputT = np.unstack(dinputT, axis=0)
        doutputT = [ operator.adjoint(input, dinputT[i]) for i, operator in enumerate(self.operators)]
        return np.sum(np.array(doutputT), axis=0)