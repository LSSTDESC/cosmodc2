#!/usr/bin/env python2.7
from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import time
import sys
import datetime
from astropy.table import Table
from scipy.spatial import cKDTree
from pecZ import pecZ
from astropy.cosmology import WMAP7 as cosmo
from scipy.interpolate import interp1d 
import galmatcher


if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    output_fname = param.get_string('output_fname').replace("${step}","all")
    print(output_fname)
    hgroup = h5py.File(output_fname,'a')['galaxyProperties']
    pos_angle = hgroup['morphology/positionAngle'].value/(180.0)*np.pi
    minor = hgroup['morphology/spheroidMinorAxisArcsec'].value
    major = hgroup['morphology/spheroidMajorAxisArcsec'].value
    q = minor/major
    e = (1.0-q)/(1.0+q)
    e1 = np.cos(2.0*pos_angle)*e
    e2 = np.sin(2.0*pos_angle)*e
    print(e)
    e_old = hgroup['morphology/spheroidEllipticity'].value

    print(__file__)
    print(np.sum(np.isfinite(e)))
    print(np.sum(np.isfinite(e_old)))
    print(pos_angle)
    # plt.figure()
    # plt.plot(pos_angle,e1,'.',alpha=0.1)
    # plt.plot(pos_angle,e2,'.',alpha=0.1)
    # plt.show()
    hgroup['morphology/spheroidEllipticity'][:] = e
    hgroup['morphology/spheroidEllipticity1'][:] = e1
    hgroup['morphology/spheroidEllipticity2'][:] = e2
    
    

