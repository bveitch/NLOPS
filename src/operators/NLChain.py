import copy
import numpy.typing as npt
from src.operators.NLBase import NLBase

class NLChain(NLBase):

    def __init__(self, 
                 nl_operators: list[NLBase],
                 name:str = "NLChain"):
        input_shape = nl_operators[0].input_shape
        output_shape = nl_operators[-1].output_shape
        super().__init__(input_shape, output_shape, name)
        prev_output_shape = nl_operators[0].output_shape
        for operator in nl_operators[1:]:
            assert operator.input_shape == prev_output_shape,\
                f"shape failed : {operator.name}: {operator.input_shape} != {prev_output_shape}"
            prev_output_shape = operator.output_shape
        self.operators = nl_operators
        
    @classmethod
    def from_nlbase(cls, op: NLBase):
        return cls([op])
    
    def print_ops(self) -> str:
        names = [ op.name for op in reversed(self.operators)]
        return " * ".join(names) 
        
    def __mul__(self, front: NLBase):
        input_shape = front.input_shape
        output_shape = self.output_shape
        nl_operators = self.operators.insert(0, front)
        return NLChain(input_shape, output_shape, nl_operators)
    
    def __rmul__(self, back: NLBase):
        input_shape = self.input_shape
        output_shape = back.output_shape
        nl_operators = self.operators.append(back)
        return NLChain(input_shape, output_shape, nl_operators)
    
    def _check_shape(self, shape:tuple, is_fwd:bool):
        if is_fwd:
            assert shape == self._input_shape, f"{self.name}: {shape=} != {self._input_shape}"
        else:
            assert shape == self._output_shape, f"{self.name}: {shape=} != {self._output_shape}"

    def _fwd_nl(self, input:npt.NDArray) ->npt.NDArray:
        output=input.copy()
        for operator in self.operators:
            output=operator(output)
        return output
    
    def _fwd_lin(self, input:npt.NDArray, dinput:npt.NDArray) ->npt.NDArray:
        output = input.copy()
        doutput = dinput.copy()
        for operator in self.operators:
            doutput=operator.linear(output, doutput)
            output=operator(output)
        return doutput
    
    def _adj_lin(self, input:npt.NDArray, dinputT:npt.NDArray) ->npt.NDArray:
        dtemp = dinputT.copy()
        ops = copy.copy(self.operators)
        while len(ops) > 0:
            op = ops.pop()
            if len(ops) > 0:
                output_shape = op.input_shape
                chainop= NLChain(input_shape = self._input_shape,
                                 output_shape = output_shape,
                                 nl_operators=ops)
                temp = chainop(input)
            dtemp = op.adjoint(temp,dtemp)
        return dtemp