#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import dtk 
import h5py
import sys
import time
from scipy import stats




def get_hfiles(fname, healpix_pixels):
    if len(healpix_pixels) == 0:
        healpix_pixels = ['']
    hfiles =[]
    print(fname)
    for healpix_pixel in healpix_pixels:
        print(healpix_pixel)
        if "#z_range#" in fname:
            for z_range in ["0_1", "1_2", "2_3"]:
                ffname = fname.replace('#healpix#',str(healpix_pixel)).replace("#z_range#", z_range)
                hfiles.append(h5py.File(ffname, 'r'))
        else:
            hfiles.append(h5py.File(fname.replace('#healpix#',str(healpix_pixel)),'r'))
    return hfiles


def get_val(hfiles, var_name, remove_nan=None):
    sub_result = []
    for hfile in hfiles:
        for key in hfile.keys():
            if key != "metaData":
                # print(hfile[key].keys())
                sub_result.append(hfile[key][var_name].value)
    result = np.concatenate(sub_result)
    if remove_nan is not None:
        result[~np.isfinite(result)]=remove_nan
    return result


def get_selection(hfiles, title, central_cut=False, 
                  Mr_cut=None, mr_cut = None, 
                  mass_cut=None, rs_cut=False,
                  synthetic=None, ms_cut =None, 
                  synthetic_type = None):
   
    redshift = get_val(hfiles,'redshift')
    slct = (redshift == redshift)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if rs_cut:
        a = get_val(hfiles,'baseDC2/is_on_red_sequence_gr')
        b = get_val(hfiles,'baseDC2/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    if synthetic is not None:
        halo_id = get_val(hfiles,'halo_id')
        slct = slct & ((halo_id < 0) == synthetic)
        title = title +'Synth.'
    if synthetic_type is not None:
        halo_id = get_val(hfiles,'baseDC2/halo_id')
        slct = slct & (halo_id == synthetic_type)
        title = title +'halo_id == {}, '.format(synthetic_type) 

    if ms_cut is not None:
        stellar_mass = get_val(hfiles,'totalMassStellar')
        slct = slct & ( stellar_mass > ms_cut)
        title = title + "M* > {:.2e}".format(ms_cut)
    return slct, title


def plot_color_redshift_baseDC2_diagnostics(fname):
    hfile = h5py.File(fname,'r')
    magr = get_var(hfile, 'restframe_extincted_sdss_abs_magr')
    magg = get_var(hfile, 'restframe_extincted_sdss_abs_magg')
    redshift = get_var(hfile, 'redshift')
    htag = get_var(hfile, 'target_halo_fof_halo_id')

    plt.figure()
    h,xbins, ybins = np.histogram2d(redshift, magg-magr, bins=500)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.ylabel('g-r')
    plt.xlabel('redshift')
    plt.colorbar(label='population density')

    plt.figure()
    plt.hist(redshift, bins=256, label='All Galaxies')
    plt.legend(loc='best')

    print("calcing")
    unique, cnt = np.unique(htag, return_counts=True)
    indx = np.argmax(cnt)
    print(unique[indx])

    plt.figure()
    plt.hist(redshift[htag==0], bins=256, label='Synthetic Galaxies')
    plt.legend(loc='best')

    plt.figure()
    plt.hist(redshift[magr<-19], bins=256, label='Mr < -19')
    plt.legend(loc='best')

    plt.figure()
    plt.hist(redshift[magr>-19], bins=256, label = 'Mr > -19')
    plt.hist(redshift[magr>-18], bins=256, label = 'Mr > -18')
    plt.hist(redshift[magr>-17], bins=256, label = 'Mr > -17')
    plt.hist(redshift[magr>-16], bins=256, label = 'Mr > -16')
    plt.legend(loc='best')
    plt.show()

    print(hfile['487'].keys())


def plot_ra_dec(hfiles, mag_cut = None):
    ra = get_val(hfiles, 'ra')
    dec = get_val(hfiles, 'dec')
    plt.figure()
    plt.hist2d(ra,dec, bins=128)
    plt.show()
    

def plot_redshift(hfiles, slct, title):
    redshift = get_val(hfiles, 'redshift')
    magr = get_val(hfiles, 'restframe_extincted_sdss_abs_magr')
    magg = get_val(hfiles, 'restframe_extincted_sdss_abs_magg')
    
    plt.figure()
    plt.hist2d(redshift, magr, bins =256)
    #plt.show()


def plot_redshift_distance(hfiles, title):
    x = get_val(hfiles, 'x')
    y = get_val(hfiles, 'y')
    z = get_val(hfiles, 'z')
    r = np.sqrt(x*x + y*y + z*z)
    target_halo_x = get_val(hfiles, 'target_halo_x')
    target_halo_y = get_val(hfiles, 'target_halo_y')
    target_halo_z = get_val(hfiles, 'target_halo_z')
    target_halo_r = np.sqrt(target_halo_x**2 + target_halo_y**2 + target_halo_z**2)

    redshift = get_val(hfiles, 'target_halo_redshift')
    redshift_raw = get_val(hfiles, 'redshift')
    slct = (redshift < 2.5) & (r > 4230)
    halo_id = get_val(hfiles, 'halo_id')
    print('x', x[slct])
    print('y', y[slct])
    print('z', z[slct])
    print('halo_id', halo_id[slct])
    central = get_val(hfiles, 'upid')==-1
    print('central', central[slct])
    host_halo_mvir = get_val(hfiles, 'host_halo_mvir')
    print('host_halo_mvir', host_halo_mvir[slct])
    restframe_extincted_sdss_abs_magr = get_val(hfiles, 'restframe_extincted_sdss_abs_magr')
    print('restframe_extincted_sdss_abs_magr', restframe_extincted_sdss_abs_magr[slct])
    target_halo_fof_halo_id = get_val(hfiles, 'target_halo_fof_halo_id')
    print('target_halo_fof_halo_id',     target_halo_fof_halo_id[slct])
    for num in target_halo_fof_halo_id[slct]:
        print(num)
    print('redshift', redshift[slct])

    plt.figure()
    plt.plot(r, target_halo_r, ',')
    
    
    plt.figure()
    plt.plot(redshift[slct], r[slct], '.',  alpha=1.0)
    plt.xlabel('redshift')
    plt.ylabel('distance [Mpc/h]')
    plt.title(title)

    plt.figure()
    plt.plot(redshift, r, ',', alpha=1.0)
    plt.xlabel('redshift')
    plt.ylabel('distance [Mpc/h]')
    plt.title(title)

    plt.figure()
    plt.plot(redshift, redshift_raw, ',')
    plt.xlabel(redshift)
    plt.ylabel(redshift)

    indx = np.zeros(len(slct))
    indx[slct] = 1.0
    plt.figure()
    plt.plot(indx, alpha=0.3)
    indx = np.zeros(len(slct))
    syn_cluster = halo_id == -1
    indx[syn_cluster] = 1.0
    plt.plot(indx)

if __name__ == "__main__":
    fname = sys.argv[1]
    healpix_pixels = sys.argv[2:]

    hfiles = get_hfiles(fname, healpix_pixels)
    # plot_ra_dec(hfiles)
    #plot_redshift(hfiles)
    slct, title = get_selection(hfiles,  "")
    plot_redshift_distance(hfiles, fname)

    # plot_color_redshift_baseDC2_diagnostics(sys.argv[1])

    plt.show()
