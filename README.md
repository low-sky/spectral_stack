# spectral-stack
Spectral Stacking utilities

First, download some data to do stacking.  Let's grab the GBT Ammonia survey data on NGC 1333 from this webpage
```
https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/IP1ZZL/C6IUO3&version=1.0
```

Next, let's load the data as a spectral cube:

```
cube = SpectralCube.read('NGC1333_NH3_11_DR1_rebase3_trim.fits')
cube = cube.with_spectral_unit(u.km / u.s,    velocity_convention='radio')
```

Next, let's generate two maps.  The `mom0` is the integrated intensity that we will use to weight our average and `mom1` is an estimate of where the centre of the line is.  Here, we'll take this to be where the spectrum is at a maximum.
```
mom0 = cube.moment0()
# Stack on peak velocity
mom1 = cube.spectral_axis[cube.argmax(axis=0).ravel()]
mom1.shape = mom0.shape
```

Next, we'll create a label map by looking at 60-90th percentiles of the data and assigning each position in the map to that bin.  Your label map can vary:

```
pcts = np.nanpercentile(mom0.value,
                        np.linspace(6, 100, 5))
bins = np.digitize(mom0.value, pcts)
bins[bins==5] = 0
```
Note that we have the convention that labels with a value of 0 are not included in the stack (i.e., they are "background").  This makes a map that looks like this:
![Spectral Stack](/imgs/binsmap.png)
Next, let's stack!  We pass in the data `cube`, the bins we want stacked spectra in (`bins`) and the estimate of where the centre of the spectrum should be (`mom1`).

```
stack, labelvals = stacking.BinByLabel(cube, bins, mom1,
                                       weight_map=mom0)

```

Finally, we can plot the average spectra:


```
for d in stack:
    plt.plot(d['spectral_axis'],d['spectrum'], label="Label={0}".format(d['label']))
plt.legend()
plt.xlabel('Spectral axis ({0})'.format(cube.spectral_axis.unit.to_string()))
plt.ylabel('Brightness ({0})'.format(cube.unit.to_string()))
```
This gives the resulting plot:
![Spectral Stack](/imgs/example_spex.png)