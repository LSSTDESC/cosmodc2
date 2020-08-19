#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import dtk 
import h5py
import sys
import os
import time
import healpy as hp
import astropy 

from astropy.cosmology import WMAP7 as cosmo

npix = hp.nside2npix(32)

sky_frac = 131/npix 
print("sky_frac: ", sky_frac)



print("here it is")
data_real = np.genfromtxt("count_real.txt", delimiter=' ')
data_uf = np.genfromtxt("count_uf.txt", delimiter=' ')
print(data_real.astype(np.int))


data_real_cumsum = np.cumsum(data_real[::-1,1], axis=0)
data_uf_cumsum = np.cumsum(data_uf[::-1,1], axis=0)


stepz = dtk.StepZ(sim_name='AlphaQ')
print(data_real[:,0])

redshifts = stepz.get_z(data_real[::-1,0])
comoving_vol = cosmo.comoving_volume(redshifts)*(0.7**3)

plt.figure()
plt.plot(redshifts, comoving_vol*sky_frac)
plt.xlabel('redshift')
plt.ylabel('volume [(h^-1 Mpc)^3]')
plt.figure()
plt.plot(redshifts, data_real_cumsum)
plt.plot(redshifts, data_uf_cumsum)
plt.xlabel('redshift')
plt.ylabel('accumlated normal galaxy count')

plt.figure()
plt.plot(redshifts, data_real_cumsum/(comoving_vol*sky_frac), label='normal')
plt.plot(redshifts, data_uf_cumsum/(comoving_vol*sky_frac), label = 'uf')
plt.xlabel('redshift')
plt.ylabel('galaxies/(h^-1 Mpc)^3')
plt.legend(loc='best')

plt.figure()
plt.plot(redshifts, data_real_cumsum/(comoving_vol*sky_frac), label='normal')
plt.plot(redshifts, data_uf_cumsum/(comoving_vol*sky_frac), label='uf')
plt.xlabel('redshift')
plt.ylabel('galaxies/(h^-1 Mpc)^3')
plt.yscale('log')
plt.legend(loc='best')

print(comoving_vol[-1]*sky_frac)
print(comoving_vol*sky_frac)

print(data_real_cumsum/(comoving_vol*sky_frac))

plt.figure()
plt.plot(redshifts[::-1], data_uf[:,1]/data_real[:,1])
plt.xlabel('redshift')
plt.ylabel('ultra faints galaxys/normal galaxies')
plt.show()
