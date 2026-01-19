import numpy as np
import numpy.typing as npt
from src.NLBase import LBase

WMIN = 380
WMAX = 720
WMINBLUE = WMIN
WMAXBLUE = 550
WMINGREEN = 450
WMAXGREEN= 620
WMINRED = 550
WMAXRED = WMAX

def cosine_filter(npoints, start, end):
    filter = np.zeros(npoints)
    L= end-start
    x=np.pi*np.linspace(-0.5, 0.5, num=L)
    cosfilter = np.cos(x)
    filter[start:end]=cosfilter*cosfilter
    return filter/np.sum(filter)

class HSIToRGB(LBase):
    
    def __init__(self, nchannels=21):
        self._wavelengths = list(np.linspace(WMIN, WMAX, nchannels))
        self.rfilter = self._init_filter(WMINRED, WMAXRED)
        self.gfilter = self._init_filter(WMINGREEN, WMAXGREEN)
        self.bfilter = self._init_filter(WMINBLUE, WMAXBLUE)
        self._rgb_filters = np.array([self.rfilter, self.gfilter, self.bfilter])
        super().__init__(input_shape=nchannels, output_shape=3)

    def _init_filter(self, cmin, cmax):
        icmin = min(range(self.nchannels), key=lambda i: abs(self.wavelengths[i] - cmin))
        icmax = min(range(self.nchannels), key=lambda i: abs(self.wavelengths[i] - cmax))
        return cosine_filter(self.nchannels, icmin, icmax)
       
    @property
    def nchannels(self) -> int:
        return len(self._wavelengths)
    
    @property
    def wavelengths(self) -> list:
        return self._wavelengths
    
    @property
    def rgb_filters(self) ->dict[str, npt.NDArray]:
        color_keys ={"red":0, "green": 1, "blue":2}
        rgbfilters ={}
        for color, index in color_keys.items():
            rgbfilters[color] = self._rgb_filters[index, :]
        return rgbfilters

    def _check_shape(self, input_shape, is_fwd):
        size = input_shape[-1]
        if is_fwd:
            assert size == self._input_shape, f"{input_shape=}[-1] != {self._input_shape}"
        else:
            assert size == self._output_shape, f"{input_shape=}[-1] != {self._output_shape}"

    def _fwd(self, hsi):
        return np.dot(hsi, self._rgb_filters.T)
    
    def _adj(self, rgb):
        return np.dot(rgb, self._rgb_filters)


        

        
    

    
