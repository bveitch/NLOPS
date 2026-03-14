from collections import namedtuple
import numpy as np
import numpy.typing as npt

SIRSIZE = 3
SISIZE = SIRSIZE - 1
SIRMODSIZE = 2

SiModParams = namedtuple('SiModParams', 'si, mod')
SiModPtfParams = namedtuple('SiModTpFpParams', 'si, mod, ptf')
SiPtfParams = namedtuple('SiModTpFpParams', 'si, ptf')

def dsi_to_dsir(si):
    s0 = si[0] 
    i0 = si[1]
    r0 = - s0 - i0
    return np.array([s0, i0, r0])

def dsir_to_dsi(sir):
    s0 = sir[0] - sir[2]
    i0 = sir[1] - sir[2]
    return np.array([s0, i0]) 
    
def si0_to_sir0(si):
    sir = dsi_to_dsir(si)
    sir[-1] += 1.0
    return sir

class ListParser:

    @staticmethod
    def from_numpy(array : npt.NDArray, dims : list) -> list:
        begin = 0
        list = [None] * len(dims) 
        for i, dim in enumerate(dims):
            if dim < 0:
                raise ValueError(f"{dim =} cannot be negative!")
            end = begin + dim
            if dim == 1:
                list[i] = array[begin]
            if dim > 1:
                list[i] = array[begin: end]
            begin = end
        return list

    @staticmethod
    def to_numpy(params : list, dims : list) -> npt.NDArray:
        lst = []
        for i, param in enumerate(params): 
            dim = dims[i]
            if dim < 0:
                raise ValueError(f"{dims[i] =} cannot be negative!")
            elif dim == 0:
                assert param is None, "param must be None if dim is 0"
            elif dim == 1:
                lst.append(np.array([param]))
            else:
                lst.append(param)
        return np.concatenate((lst))
    
class SiModPtfParser:

    @staticmethod
    def from_numpy(array : npt.NDArray, dims) -> SiModPtfParams:
        list = ListParser.from_numpy(array, dims)
        return SiModPtfParams._make(list)

    @staticmethod
    def to_numpy(params : SiModPtfParams, dims) -> npt.NDArray:
        return ListParser.to_numpy(params, dims)
    
class SiPtfParser:

    @staticmethod
    def from_numpy(array : npt.NDArray, dims : list) -> SiPtfParams:
        list = ListParser.from_numpy(array, dims)
        return SiPtfParams._make(list)

    @staticmethod
    def to_numpy(params : SiPtfParams, dims) -> npt.NDArray:
        return ListParser.to_numpy(params, dims)
