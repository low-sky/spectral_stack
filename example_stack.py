from spectral_stack import stacking
# You can make sure that the spectral_stack code is in your 
# path by adding to your $PYTHONPATH system variable.  In BASH this looks like:
# export PYTHONPATH=$PYTHONPATH:/path/to/spectral_stack/
from spectral_cube import SpectralCube
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

cube = SpectralCube.read('NGC1333_NH3_11_DR1_rebase3_trim.fits')
cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
mom0 = cube.moment0()
# Stack on peak velocity
mom1 = cube.spectral_axis[cube.argmax(axis=0).ravel()]
mom1.shape = mom0.shape
# Bin the data into the 60th to 90th percentile by brightness

pcts = np.nanpercentile(mom0.value,
                        np.linspace(60, 100, 5))
bins = np.digitize(mom0.value, pcts)
# Blank NaN values
bins[bins==5] = 0

plt.imshow(bins, origin='lower')
plt.colorbar()
plt.savefig('./imgs/binsmap.png')
plt.close()
plt.clf()

stack, labelvals = stacking.BinByLabel(cube, bins, mom1,
                                       weight_map=mom0)

for d in stack:
    plt.plot(d['spectral_axis'], d['spectrum'], label="Label={0}".format(d['label']))
plt.legend()
plt.xlabel('Spectral axis ({0})'.format(cube.spectral_axis.unit.to_string()))
plt.ylabel('Brightness ({0})'.format(cube.unit.to_string()))
plt.savefig('./imgs/example_spex.png')
plt.close()
plt.clf()
