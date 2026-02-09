import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hsi2rgb import HSIToRGB
from src.operators.NLChain import NLChain
from src.operators.Constraints import Sigmoid
from src.objectives.ObjectiveFn import L2ObjectiveFn
from src.objectives.SumObjectiveFn import SumObjectiveFn
from src.solvers.Solve import GeneralSolver

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
              reflectivity_bound: bool,
              hsi_constraint: bool=False) -> L2ObjectiveFn:
    hsi2rgb = HSIToRGB()
    wavelengths = hsi2rgb.wavelengths
    nchannels = hsi2rgb.nchannels
    rgb_shape = rgb_raw.shape
    hsi_shape = list(rgb_shape)
    hsi_shape[-1]=nchannels
    data = convert_rgb_to_float32(rgb_raw)
    robjfn = None
    if not reflectivity_bound:
        constraint = None
        operator = hsi2rgb
    else:
        constraint = Sigmoid(shape = hsi_shape, min=0.0, max=1.25)
        operator = NLChain([constraint, hsi2rgb])
    if hsi_constraint:
        robjfn = L2ObjectiveFn(hsi_shape)
    dobjfn = L2ObjectiveFn(hsi_shape, operator = operator, data = data)
    if robjfn is None:
        objfn=dobjfn
    else:
        objfn=SumObjectiveFn([dobjfn, robjfn], [1,0.0])
    setattr(objfn, 'wavelengths', wavelengths)
    setattr(objfn, 'constraint', constraint)
    return objfn

def rgb2hsi(rgb_raw: npt.NDArray, reflectivity_bound: bool=True): 
    objfn = hsi_objfn(rgb_raw, reflectivity_bound)
    wavelengths = objfn.wavelengths
    constraint = objfn.constraint
    solver = GeneralSolver(objfn)
    hsi_data = solver.solve()
    if constraint is not None:
        hsi_data = constraint(hsi_data)
    return hsi_data, wavelengths
