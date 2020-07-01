#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import dtk 
import h5py
import sys
import time
from numpy.random import normal
from cosmodc2.mock_diagnostics import mean_des_red_sequence_gr_color_vs_redshift, mean_des_red_sequence_ri_color_vs_redshift, mean_des_red_sequence_iz_color_vs_redshift

# Galaxy dict:
# mg, mr, mi, gr, ri

def print_keys(hfile):
    for key in hfile.keys():
        print(key)
    return 

def load_cosmoDC2(fname, step):
    print(fname)
    cat = {}
    hfile = h5py.File(fname, 'r')['galaxyProperties/SDSS_filters']
    cat['mg'] = hfile['magnitude:SDSS_g:rest:dustAtlas'].value
    cat['mr'] = hfile['magnitude:SDSS_r:rest:dustAtlas'].value
    cat['mi'] = hfile['magnitude:SDSS_i:rest:dustAtlas'].value
    cat['gr'] = cat['mg']-cat['mr']
    cat['ri'] = cat['mr']-cat['mi']
    return cat

def load_baseDC2(fname, step):
    print(fname)
    cat = {}
    hfile = h5py.File(fname, 'r')['galaxyProperties/baseDC2']
    cat['mg'] = hfile['restframe_extincted_sdss_abs_magg'].value
    cat['mi'] = hfile['restframe_extincted_sdss_abs_magi'].value
    cat['mr'] = hfile['restframe_extincted_sdss_abs_magr'].value
    cat['gr'] = cat['mg']-cat['mr']
    cat['ri'] = cat['mr']-cat['mi']
    return cat

def load_galacticus(fname, step):
    print(fname)
    cat = {}
    hfile = h5py.File(fname, 'r')['galaxyProperties/SDSS_filters']
    cat['mg'] = hfile['magnitude:SDSS_g:rest:dustAtlas'].value
    cat['mr'] = hfile['magnitude:SDSS_r:rest:dustAtlas'].value
    cat['mi'] = hfile['magnitude:SDSS_i:rest:dustAtlas'].value
    cat['gr'] = cat['mg']-cat['mr']
    cat['ri'] = cat['mr']-cat['mi']
    return cat

def contour_plot(mr, gr, color, label, mag_bins, color_bins):
    mag_bins_cen = dtk.bins_avg(mag_bins)
    color_bins_cen = dtk.bins_avg(color_bins)
    h,_,_ = np.histogram2d(mr, gr, bins=(mag_bins, color_bins))
    h = dtk.smoothen_H(h)
    levels = dtk.conf_interval(h, [ 0.80, 0.99])[::-1]
    for level in levels:
        plt.contourf(mag_bins_cen, color_bins_cen, h.T, colors=color, levels=[level, 1e10], alpha=0.3)
    plt.contour(mag_bins_cen, color_bins_cen, h.T, colors=color, levels=levels)
    plt.plot([],[], label=label)

def plot_three_cat(cosmo_cat, base_cat, gltcs_cat):
    color_bins = np.linspace(-0.5, 1, 100)
    color_bins_cen = dtk.bins_avg(color_bins)
    mag_bins = np.linspace(-24, -12, 64)
    mag_bins_cen = dtk.bins_avg(mag_bins)

    print(color_bins)
    print(cosmo_cat['gr'])
    
    cosmo_h, _ = np.histogram(cosmo_cat['gr'][cosmo_cat['mr']<-11], bins=color_bins, range=(np.min(color_bins), np.max(color_bins)), normed=True)
    base_h,  _ = np.histogram(base_cat['gr'][base_cat['mr']<-11],  bins=color_bins, range=(np.min(color_bins), np.max(color_bins)), normed=True)
    gltcs_h, _ = np.histogram(gltcs_cat['gr'][gltcs_cat['mr']<-11], bins=color_bins, range=(np.min(color_bins), np.max(color_bins)), normed=True)
    
    plt.figure(figsize=(4,3))
    plt.plot(color_bins_cen, base_h,  '-',   color='tab:blue', label=r'BaseDC2')
    plt.fill_between(color_bins_cen, base_h, color='tab:blue', alpha=0.3)
    plt.plot(color_bins_cen, gltcs_h, '-',    color='tab:orange', label=r'Galacticus')
    plt.fill_between(color_bins_cen, gltcs_h, color='tab:orange', alpha=0.3)
    plt.plot(color_bins_cen, cosmo_h, '-',    color='tab:green', label=r'CosmoDC2')
    plt.fill_between(color_bins_cen, cosmo_h, color='tab:green',   alpha=0.3)
    ylim = plt.ylim()
    plt.ylim([0,ylim[1]])
    plt.legend(loc='best')
    plt.ylabel(r'PDF')
    plt.xlabel(r'SDSS rest-frame g-r color')
    plt.tight_layout()

    color_bins = np.linspace(-0.5, 1, 64)
    mag_bins = np.linspace(-24, -12, 40)

    plt.figure(figsize=(4,3))
    contour_plot(base_cat['mr'], base_cat['gr'], 'tab:blue', r"BaseDC2", mag_bins, color_bins)
    contour_plot(gltcs_cat['mr'], gltcs_cat['gr'], 'tab:orange', r"Galacticus", mag_bins, color_bins)
    contour_plot(cosmo_cat['mr'], cosmo_cat['gr'], 'tab:green', r"CosmoDC2", mag_bins, color_bins)
    plt.legend(loc='lower left')
    plt.xlabel(r'SDSS Mag r')
    plt.ylabel(r'SDSS rest-frame g-r color')
    plt.tight_layout()
    # plt.pcolor(mag_bins, color_bins, h.T, cmap='Blues')

    
    dtk.save_figs(path='figs/'+__file__+"/", extension=".pdf")

    plt.show()

def plot_color_match(param_fname, step = 323):
    param = dtk.Param(param_fname)
    cosmoDC2_fname = param.get_string("output_fname")
    gltcs_fname    = param.get_string("gltcs_fname")
    healpixs       = param.get_int_list("healpix_pixels")
    healpix = healpixs[0]
    cosmoDC2_fname = cosmoDC2_fname.replace('${step}', str(step)).replace('${healpix}', str(healpix))
    gltcs_fname = gltcs_fname.replace('${step}', str(step))
    cosmo_cat = load_cosmoDC2(cosmoDC2_fname, step)
    base_cat  = load_baseDC2(cosmoDC2_fname, step)
    gltcs_cat = load_galacticus(gltcs_fname, step)
    plot_three_cat(cosmo_cat, base_cat, gltcs_cat)

if __name__ == "__main__":
    plot_color_match(sys.argv[1])
