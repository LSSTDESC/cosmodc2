#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys

def plot_gr_ri(gr, ri, title):
    plt.figure()
    bins = np.linspace(-1,2,100)
    h,xbins,ybins = np.histogram2d(gr,ri,bins=(bins,bins))
    plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
    plt.grid()
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.title(title)

def load_protoDC2(fname):
    hgroup = h5py.File(fname,'r')['galaxyProperties']
    redshift = hgroup['redshiftHubble'].value
    result  = {}
    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    result['g rest'] = mag_g
    result['r rest'] = mag_r
    result['i rest'] = mag_i

    result['g-r rest'] = mag_g - mag_r
    result['r-i rest'] = mag_r - mag_i

    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:observed:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:observed:dustAtlas'].value
    mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:observed:dustAtlas'].value
    result['g obs'] = mag_g
    result['r obs'] = mag_r
    result['i obs'] = mag_i



    result['g-r obs'] = mag_g - mag_r
    result['r-i obs'] = mag_r - mag_i

    result['redshift'] = redshift
    return result

def load_umachine(fname):
    hfile = h5py.File(fname,'r')
    result = {}
    result['g-r'] = hfile['restframe_extincted_sdss_gr'].value
    result['r-i'] = hfile['restframe_extincted_sdss_ri'].value
    result['redshift'] = hfile['redshift']
    return result

def append_dics(dics):
    result = {}
    keys = dics[0].keys()
    for key in keys:
        result[key] = []
    for dic in dics:
        for key in keys:
            result[key].append(dic[key])
    for key in keys:
        result[key] = np.concatenate(result[key])
    return result

if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    lightcone_fname = param.get_string('lightcone_fname')
    output_fname = param.get_string('output_fname')
    steps = param.get_int_list('steps')
    protoDC2_list = []
    umachine_list = []

    for i in range(0,len(steps)-1):
        step = steps[i+1]
        print step
        protoDC2_list.append(load_protoDC2(output_fname.replace("${step}",str(step))))
        umachine_list.append(load_umachine(lightcone_fname.replace("${step}",str(step))))
    protoDC2 = append_dics(protoDC2_list)
    umachine = append_dics(umachine_list)
    plot_gr_ri(protoDC2['g-r'],protoDC2['r-i'],'ProtoDC2 v3')
    plot_gr_ri(umachine['g-r'],umachine['r-i'],'UMachine + SDSS')
    plt.figure()
    h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['i obs'],bins=(100,100))
    plt.pcolor(xbins,ybins,h.T,cmap="PuBu",norm=LogNorm())
    plt.colorbar()
    plt.xlabel('redshift');plt.ylabel('mag i observed')
    plt.grid()
    plt.show()
