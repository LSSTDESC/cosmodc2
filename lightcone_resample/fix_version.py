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
    vmaj = param.get_int('version_major')
    vmin = param.get_int('version_minor')
    vmm  = param.get_int('version_minor_minor')
    print(output_fname)
    hgroup = h5py.File(output_fname,'r+')['metaData']
    del hgroup['versionMajor'] 
    del hgroup['versionMinor'] 
    del hgroup['versionMinorMinor'] 
    del hgroup['version'] 
    # del hgroup['versionChangeNotes']

    hgroup['versionMajor'] = vmaj
    hgroup['versionMinor'] = vmin
    hgroup['versionMinorMinor'] = vmm
    hgroup['version'] = "{}.{}.{}".format(vmaj,vmin,vmm)

    # hgroup.move('ra','ra_true')
    # hgroup.move('dec','dec_true')
    # hgroup.move('ra_lensed', 'ra')
    # hgroup.move('dec_lensed','dec')


