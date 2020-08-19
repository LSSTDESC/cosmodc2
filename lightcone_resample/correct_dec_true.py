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
    dec_true = hgroup['dec_true'].value
    hgroup['dec_true'][:] = dec_true - 85.0
    
