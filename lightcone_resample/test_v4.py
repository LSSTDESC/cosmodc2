#!/usr/bin/env python2.7

from __future__ import print_function, division


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys
import time


if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    output_fname = param.get_string('output_fname').replace("${step}","all")
    hgp = h5py.File(output_fname,'r')['galaxyProperties']
    print(hgp.keys())
    redshift  = hgp['redshiftHubble'].value
    pos_angle = hgp['morphology/positionAngle'].value

    ellip     = hgp['morphology/totalEllipticity'].value
    ellip1    = hgp['morphology/totalEllipticity1'].value
    ellip2    = hgp['morphology/totalEllipticity2'].value

    dellip     = hgp['morphology/diskEllipticity'].value
    dellip1    = hgp['morphology/diskEllipticity1'].value
    dellip2    = hgp['morphology/diskEllipticity2'].value

    sellip     = hgp['morphology/spheroidEllipticity'].value
    sellip1    = hgp['morphology/spheroidEllipticity1'].value
    sellip2    = hgp['morphology/spheroidEllipticity2'].value

    mag_g = hgp['SDSS_filters/magnitude:SDSS_g:rest'].value
    mag_r = hgp['SDSS_filters/magnitude:SDSS_r:rest'].value
    mag_i = hgp['SDSS_filters/magnitude:SDSS_i:rest'].value
  
    fnt_mag = np.isfinite(mag_r)
    
    dsize = hgp['morphology/diskHalfLightRadius'].value
    ssize = hgp['morphology/spheroidHalfLightRadius'].value

    dsize_as = hgp['morphology/diskHalfLightRadiusArcsec'].value
    ssize_as = hgp['morphology/spheroidHalfLightRadiusArcsec'].value

    dsm = hgp['diskMassStellar'].value
    ssm = hgp['diskMassStellar'].value
    
    bhm = hgp['blackHoleMass'].value
    bhar= hgp['blackHoleAccretionRate'].value

    h,xbins = np.histogram(pos_angle,bins=100)
    plt.figure()
    plt.plot(dtk.bins_avg(xbins),h)
    plt.xlabel("pos_angle")
    plt.grid()
    
    # print(ellip)
    xbins = np.linspace(0,1,100)
    slct = np.isfinite(ellip)
    print(np.sum(slct)/np.size(slct))
    h,xbins = np.histogram(ellip[slct],bins=xbins)
    plt.figure()
    plt.plot(dtk.bins_avg(xbins),h)
    plt.xlabel("ellip");plt.ylabel('freq')
    plt.grid()

    print(np.min(dellip), np.average(dellip), np.max(dellip))

    plt.figure()
    h,xbins = np.histogram(dellip,bins=xbins)
    plt.plot(dtk.bins_avg(xbins),h,label='ellipticity')
    h,xbins = np.histogram((1-dellip)/(1+dellip),bins=xbins)
    plt.plot(dtk.bins_avg(xbins),h,label='axis ratio')
    plt.xlabel("disk ellip");plt.ylabel('freq')
    plt.legend(loc='best')
    plt.grid()

    slct = np.isfinite(sellip)
    print(np.sum(slct)/np.size(slct))
    print(np.min(sellip[slct]), np.average(sellip[slct]), np.max(sellip[slct]))
    plt.figure()
    h,xbins = np.histogram(sellip[slct],bins=xbins)
    plt.plot(dtk.bins_avg(xbins),h)
    h,xbins = np.histogram((1-sellip[slct])/(1+sellip[slct]),bins=xbins)
    plt.plot(dtk.bins_avg(xbins),h)
    plt.xlabel("sphere ellip");plt.ylabel('freq')
    plt.legend(loc='best')
    plt.grid()
    
    
    # Take too long to plot
    # plt.figure()
    # plt.plot(pos_angle,ellip1,',',alpha=0.3)
    # plt.plot(pos_angle,ellip2,',',alpha=0.3)

    # plt.figure()
    # plt.plot(pos_angle,dellip1,',',alpha=0.3)
    # plt.plot(pos_angle,dellip2,',',alpha=0.3) 

    # plt.figure()
    # plt.plot(pos_angle,sellip1,',',alpha=0.3)
    # plt.plot(pos_angle,sellip2,',',alpha=0.3) 
   
    rad_bins = np.logspace(-3,3,100)
    plt.figure()
    h,xbins,ybins = np.histogram2d(mag_r, dsize,bins=(100,rad_bins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('mag r');plt.ylabel('disk size [kpc]')
    plt.yscale('log')
    plt.grid()
    

    plt.figure()
    h,xbins,ybins = np.histogram2d(mag_r, ssize,bins=(100,rad_bins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('mag r');plt.ylabel('bulge size [kpc]')
    plt.yscale('log')
    plt.grid()

    plt.figure()
    h,xbins,ybins = np.histogram2d(mag_r, dsize_as,bins=(100,rad_bins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('mag r');plt.ylabel('disk size [arcsec]')
    plt.yscale('log')
    plt.grid()
    
    plt.figure()
    h,xbins,ybins = np.histogram2d(mag_r, ssize_as,bins=(100,rad_bins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('mag r');plt.ylabel('bulge size [arcsec]')
    plt.yscale('log')
    plt.grid()


    plt.figure()
    logbins = np.logspace(1,13,100)
    h,xbins,ybins = np.histogram2d(ssm,bhm,bins=(logbins,logbins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('bulge stellar mass');plt.ylabel('black hole mass')
    plt.xscale('log');plt.yscale('log')
    plt.grid()

    plt.figure()
    logbins = np.logspace(1,13,100)
    
    h,xbins,ybins = np.histogram2d(bhm,bhar,bins=(logbins,logbins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.xlabel('black hole mass');plt.ylabel('black hole accreation rate [Msun/Gyr]')
    plt.xscale('log');plt.yscale('log')
    plt.grid()

    plt.show()


    
