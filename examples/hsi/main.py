import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib
import numpy as np
import skimage as ski
from .rgb2hsi import plot_rgb_filters, rgb2hsi
matplotlib.rcParams['font.size'] = 18

plot_rgb_filters()

images = (
    'astronaut',
)
Point2d = namedtuple('Point', ['x', 'y','color', 'marker'])
pt_of_interest = Point2d(100, 100, 'cyan', 'x')
spectra={}
for name in images:
    caller = getattr(ski.data, name)
    image = caller()
    image=image
    hsi_image, wavelengths = rgb2hsi(image)
    loc_min = np.unravel_index(np.argmin(hsi_image), hsi_image.shape)
    loc_max = np.unravel_index(np.argmax(hsi_image), hsi_image.shape)
    pt_min = Point2d(loc_min[0], loc_min[1], 'magenta', 'x')
    pt_max = Point2d(loc_max[0], loc_max[1], 'red', 'x')
    points = [pt_of_interest, pt_min, pt_max]
    spectra = [(point, hsi_image[point.x, point.y, :]) for point in points]
    print(hsi_image.shape)
    fig, [ax0,ax1] = plt.subplots(2)
    fig.suptitle(name)
    if image.ndim == 2:
        ax0.imshow(image, cmap=plt.cm.gray)
    else:
        ax0.imshow(image)
    for (pt, spectrum ) in spectra:
        ax0.scatter(pt.x, pt.y, marker=pt.marker, color=pt.color)
        ax1.plot(wavelengths, spectrum, color=pt.color)

plt.savefig(f"{name}")
plt.show()