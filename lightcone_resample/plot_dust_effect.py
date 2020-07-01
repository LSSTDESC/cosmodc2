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

def plot_mag_dust(mag_delta, mag, name,obs= False,ybins=None):
    plt.figure()
    if obs:
        xbins = np.linspace(10,25,100)
    else:
        xbins = np.linspace(-25,-10,100)
    if ybins is None:
        ybins = np.linspace(-1,3,100)
    h,xbins,ybins = np.histogram2d(mag, mag_delta, bins=(xbins,ybins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel(name +' no dust');plt.ylabel(name+' (dust - no dust)')
    plt.grid()

def plot_clr_dust(mag_delta1, mag_delta2, mag, clr_name, mag_name,obs=False,xbins=None,ybins = None):
    plt.figure()
    if xbins is None:
        if obs:
            xbins = np.linspace(10,25,100)
        else:
            xbins = np.linspace(-25,-10,100)
    if ybins is None:
        ybins = np.linspace(-1,1,100)
        
    h,xbins,ybins = np.histogram2d(mag, mag_delta1 - mag_delta2, bins = (xbins,ybins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel(mag_name +' no dust');plt.ylabel(clr_name+' (dust - no dust)')
    plt.grid()


    
def plot_dust_effect(fname):
    t1 = time.time()
    gal_prop = {}
    hfile = h5py.File(fname,'r')
    hgp = hfile['galaxyProperties']
    m_star = np.log10(hgp['totalMassStellar'].value)
    incl = hgp['morphology/inclination'].value
    mag_gd = hgp['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    mag_rd = hgp['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_id = hgp['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    mag_gnd = hgp['SDSS_filters/magnitude:SDSS_g:rest'].value
    mag_rnd = hgp['SDSS_filters/magnitude:SDSS_r:rest'].value
    mag_ind = hgp['SDSS_filters/magnitude:SDSS_i:rest'].value
    mag_dgd = mag_gd - mag_gnd
    mag_drd = mag_rd - mag_rnd
    mag_did = mag_id - mag_ind
    plot_mag_dust(mag_dgd, mag_gnd, "Mag g rest", ybins=np.linspace(-.05,.05,100))
    plot_clr_dust(mag_dgd, mag_drd, mag_rnd, "g-r rest", "Mag r rest", ybins=np.linspace(-.05,.05,100))
    plot_clr_dust(mag_dgd, mag_drd, incl , "g-r rest", "inclination", xbins=np.linspace(0,90,100),ybins=np.linspace(-.06,.06,100))
    mag_gd = hgp['SDSS_filters/magnitude:SDSS_g:observed:dustAtlas'].value
    mag_rd = hgp['SDSS_filters/magnitude:SDSS_r:observed:dustAtlas'].value
    mag_id = hgp['SDSS_filters/magnitude:SDSS_i:observed:dustAtlas'].value
    mag_gnd = hgp['SDSS_filters/magnitude:SDSS_g:observed'].value
    mag_rnd = hgp['SDSS_filters/magnitude:SDSS_r:observed'].value
    mag_ind = hgp['SDSS_filters/magnitude:SDSS_i:observed'].value
    mag_dgd = mag_gd - mag_gnd
    mag_drd = mag_rd - mag_rnd
    mag_did = mag_id - mag_ind
    plot_mag_dust(mag_dgd, mag_gnd, "Mag g observed",obs=True)
    # plot_mag_dust(mag_drd, mag_rnd, "Mag r observed",obs=True)
    # plot_mag_dust(mag_did, mag_ind, "Mag i observed",obs=True)
    plot_clr_dust(mag_dgd, mag_drd, mag_rnd, "g-r observed", "Mag r observed",obs=True)
    # plot_clr_dust(mag_dgd, mag_drd, mag_rnd, "r-i observed", "Mag r observed",obs=True)

    plt.show()
    


if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    gltcs_fname = param.get_string("gltcs_fname")
    steps = param.get_string_list("steps")
    plot_dust_effect(gltcs_fname.replace('${step}',str(421)))
