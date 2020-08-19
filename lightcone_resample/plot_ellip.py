#!/usr/bin/env python2.7

from __future__ import division,print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys
import time
from numpy.random import normal



if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    gltcs_fname = param.get_string('gltcs_fname')
    output_fname = param.get_string("output_fname")
    output_fname = output_fname.replace("${step}","all")
    hgroup = h5py.File(output_fname,'r')['galaxyProperties']
    print( hgroup['morphology'].keys())
    minor = hgroup['morphology/spheroidMinorAxisArcsec'].value
    major = hgroup['morphology/spheroidMajorAxisArcsec'].value
    q = hgroup['morphology/spheroidAxisRatio'].value
    e = hgroup['morphology/spheroidEllipticity'].value
    inclination = hgroup['morphology/inclination'].value
    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    sm    = hgroup['totalMassStellar'].value
    redshift = hgroup['redshift'].value
    dust = hgroup['dustFactor'].value
    gltcs_fname = gltcs_fname.replace("${step}",str(487))
    print(gltcs_fname)
    gltcs_hgroup = h5py.File(gltcs_fname,'r')['galaxyProperties']
    gltcs_inclination = gltcs_hgroup['morphology/inclination'].value
    gltcs_mag_g = gltcs_hgroup['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    gltcs_mag_r = gltcs_hgroup['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    gltcs_mag_i = gltcs_hgroup['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    gltcs_sm    = gltcs_hgroup['totalMassStellar'].value 
   
    a  = (minor/major)
    b = (1.0-e)/(1.0+e)
    equal =  a==b 
    equal2 = np.isclose(a,b)
    print( equal)
    print(np.sum(equal),equal.size)
    print(np.sum(equal2),equal2.size)
    for i in range(0,10):
        print(equal[i], "  {} == {} \t\t diff: {}".format(a[i],b[i],a[i]-b[i] ))
    print("max: ", np.nanmax(np.abs(a-b)))
    print(np.sum(np.isfinite(a)))
    print(np.sum(np.isfinite(b)))
    
    h,xbins = np.histogram(inclination,bins=250)
    h_all = np.sum(h)
    h = h/np.sum(h)
    plt.figure()
    plt.plot(dtk.bins_avg(xbins), h,label = 'ProtoDC2 v3 all')
    slct = dust == 1
    h,xbins = np.histogram(inclination[slct],bins=250)
    plt.plot(dtk.bins_avg(xbins), h/h_all,label = 'ProtoDC2 v3 dust =1')
    slct = dust == 3
    h,xbins = np.histogram(inclination[slct],bins=250)
    plt.plot(dtk.bins_avg(xbins), h/h_all,label = 'ProtoDC2 v3 dust = 3')
    slct = dust == 6
    h,xbins = np.histogram(inclination[slct],bins=250)
    plt.plot(dtk.bins_avg(xbins), h/h_all,label = 'ProtoDC2 v3 dust = 6')
    
    h,xbins = np.histogram(gltcs_inclination,bins = xbins)
    plt.plot(dtk.bins_avg(xbins), h/np.sum(h), label = 'Galacticus Snapshots')
    x= dtk.bins_avg(xbins)
    y = np.sin(x/180.0 *np.pi)
    y = y/np.sum(y)
    plt.plot(x,y,'--',label='random inclination')
    plt.xlabel('inclination [Deg]');plt.ylabel('pdf')
    plt.grid()
    plt.legend(loc='best',framealpha=0.3)

    plt.figure()
    h,xbins, ybins = np.histogram2d(inclination,mag_g-mag_r,bins = (250,250))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('inclination [Deg]');plt.ylabel('g-r rest');
    plt.grid()

    plt.figure()
    h,xbins, ybins = np.histogram2d(inclination,mag_r-mag_i,bins = (250,250))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('inclination [Deg]');plt.ylabel('r-i rest');
    plt.grid()

    plt.figure()
    h,xbins, ybins = np.histogram2d(inclination,mag_r,bins = (250,250))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('inclination [Deg]');plt.ylabel('r rest');
    plt.grid()

    plt.figure()
    h,xbins, ybins = np.histogram2d(inclination,np.log10(sm),bins = (250,250))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('inclination [Deg]');plt.ylabel('Log10(Stellar mass/Msun)');
    plt.grid()

    plt.figure()
    h,xbins, ybins = np.histogram2d(inclination,redshift,bins = (250,250))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('inclination [Deg]');plt.ylabel('redshift');
    plt.grid()

    # ybins = np.linspace(-1,2,250)
    # plt.figure()
    # h,xbins, ybins = np.histogram2d(gltcs_inclination,gltcs_mag_g-gltcs_mag_r,bins = (250,ybins))
    # plt.title("Galacticus Snapshot")
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.xlabel('inclination [Deg]');plt.ylabel('g-r rest');
    # plt.grid()

    # plt.figure()
    # plt.title("Galacticus Snapshot")
    # h,xbins, ybins = np.histogram2d(gltcs_inclination,gltcs_mag_r-gltcs_mag_i,bins = (250,ybins))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.xlabel('inclination [Deg]');plt.ylabel('r-i rest');
    # plt.grid()

    # ybins = np.linspace(-30,-12,250)
    # plt.figure()
    # plt.title("Galacticus Snapshot")
    # h,xbins, ybins = np.histogram2d(gltcs_inclination,gltcs_mag_r,bins = (250,ybins))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.xlabel('inclination [Deg]');plt.ylabel('r rest');
    # plt.grid()

    # ybins = np.linspace(1,13,250)
    # plt.figure()
    # plt.title("Galacticus Snapshot")
    # h,xbins, ybins = np.histogram2d(gltcs_inclination,np.log10(gltcs_sm),bins = (250,ybins))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.xlabel('inclination [Deg]');plt.ylabel('Log10(Stellar mass/Msun)');
    # plt.grid()

    dtk.save_figs(path='figs/'+sys.argv[1]+'/'+__file__+'/')

    plt.show()
    
