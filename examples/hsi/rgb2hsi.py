import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hsi2rgb import HSIToRGB
from src.operators.chain import NLChain
from src.operators.constraints import Sigmoid
from src.objectives.base import L2ObjectiveFn
from src.objectives.sum_objective import SumObjectiveFn
from src.solvers.general import GeneralSolver
from src.solvers.direct import direct_solve, direct_solve_reverse

def plot_rgb_filters():
    hsi2rgb = HSIToRGB()
    wavelengths = hsi2rgb.wavelengths
    rgbfilters = hsi2rgb.rgb_filters
    for color, filter in rgbfilters.items():
        plt.plot(wavelengths, filter,color=color)
    plt.xlabel("wavelength (nm)")
    plt.title("RGB response")
    plt.savefig("rgbfilters")

def convert_rgb_to_float32(rgb_raw: npt.NDArray):
    info = np.iinfo(rgb_raw.dtype)
    min = info.min
    max = info.max
    data = rgb_raw.astype(np.float32)
    return (data-min)/(max-min)

def hsi_objfn(rgb_raw: npt.NDArray, 
              reflectivity_bound: bool) -> L2ObjectiveFn:
    rgb_shape = rgb_raw.shape
    hsi2rgb = HSIToRGB(nx=rgb_shape[0], ny=rgb_shape[1])
    wavelengths = hsi2rgb.wavelengths
    hsi_shape = hsi2rgb.input_shape
    data = convert_rgb_to_float32(rgb_raw)
    robjfn = None
    if not reflectivity_bound:
        constraint = None
        operator = hsi2rgb
    else:
        constraint = Sigmoid(shape = hsi_shape, min=-0.1, max=1.0)
        operator = NLChain([constraint, hsi2rgb])
        robjfn = L2ObjectiveFn(hsi_shape)
    dobjfn = L2ObjectiveFn(hsi_shape, operator = operator, data = data)
    if robjfn is None:
        objfn=dobjfn
    else:
        objfn=SumObjectiveFn([dobjfn, robjfn], [1,0.01])
    setattr(objfn, 'wavelengths', wavelengths)
    setattr(objfn, 'constraint', constraint)
    return objfn

def rgb2hsi(rgb_raw: npt.NDArray, reflectivity_bound: bool=True): 
    objfn = hsi_objfn(rgb_raw, reflectivity_bound)
    wavelengths = objfn.wavelengths
    constraint = objfn.constraint
    #solver = GeneralSolver(objfn, method='CG')
    #hsi_data = solver.solve()
    A = objfn.op._M
    b = objfn.d
    hsi_data = direct_solve_reverse(A, b, 0.01)
    # if constraint is not None:
    #     hsi_data = constraint(hsi_data)
    return hsi_data, wavelengths
