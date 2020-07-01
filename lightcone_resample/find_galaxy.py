#!/usr/bin/env python2.7

from __future__ import print_function, division 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys
import time
from numpy.random import normal

def get_hfiles(fname, healpix_pixels):
    if len(healpix_pixels) == 0:
        healpix_pixels = ['']
    hfiles =[]
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
        sub_result.append(hfile['galaxyProperties/'+var_name].value)
    result = np.concatenate(sub_result)
    if remove_nan is not None:
        result[~np.isfinite(result)]=remove_nan
    return result


def get_mag(hfiles, filter_type, frame, band):
    remove_nan = None
    band_filter_frame = '{filter_type}_filters/magnitude:{filter_type}_{band}:{frame}:dustAtlas';
    model_band_filter_frame= 'baseDC2/restframe_extincted_sdss_abs_mag{band}'
    if frame == 'obs':
        frame = 'observed'
    if filter_type == 'model':
        assert frame=='rest'
        var_name = model_band_filter_frame.format(**{'band': band})
        remove_nan = -14
    else:
        var_name = band_filter_frame.format(**{'filter_type':filter_type,
                                               'frame':frame,
                                               'band':band,})
    return get_val(hfiles,var_name, remove_nan = remove_nan)


def get_av_vals(hfiles, component, band, dust):
    data = get_val(hfiles, 'otherLuminosities/'+component+'LuminositiesStellar:'+band+':rest'+dust)
    return np.log10(data)*-2.5


def print_Av_Rv(hfiles, component, slct):
    v_dust = get_av_vals(hfiles, component,"V", ":dustAtlas")[slct]
    v = get_av_vals(hfiles, component, "V", "")[slct]
    b_dust = get_av_vals(hfiles, component, "B", ":dustAtlas")[slct]
    b = get_av_vals(hfiles, component, "B", "")[slct]
    Av = v_dust - v
    Rv = Av/((b_dust - b) - (v_dust - v))
    print(component, "V dust", v_dust)
    print(component, "V     ", v_dust)
    print(component, "B dust", b_dust)
    print(component, "B     ", b)
    print(component, "Av    ", Av)
    print(component, "Rv    ", Rv)


if __name__ == "__main__":
    fname = sys.argv[1]
    healpix_pixels = sys.argv[2:]
    print("\n\n")
    print("71.02876567618159 -25.70934849853007 22.1763157")
    print('id :', 10241431378047)
    print("we are trying to find the above")

    # Strange Rv/Av galaxy
    # ra_target = 71.02876567618159
    # dec_target = -25.70934849853007
    # id_target = 10241431378047
    # id_target_guess = 10241431378
    z_target = 0.32086
    hfiles = get_hfiles(fname, healpix_pixels)
    ra = get_val(hfiles, 'ra')
    dec = get_val(hfiles, 'dec')
    ra_true = get_val(hfiles, 'ra_true')
    dec_true = get_val(hfiles, 'dec_true')

    ids = get_val(hfiles, 'galaxyID')
    z = get_val(hfiles, 'redshift')
    z_true = get_val(hfiles, 'redshiftHubble')
    print("ra: {:.2f} -> {:.2f}".format(np.min(ra), np.max(ra)))
    print("dec: {:.2f} -> {:.2f}".format(np.min(dec), np.max(dec)))
    hit_ra = (np.min(ra) < ra_target) & (ra_target < np.max(ra))
    hit_dec = (np.min(dec) < dec_target) & (dec_target < np.max(dec))
    print("hit: ", hit_ra & hit_dec)
    print("\t\t ra;",hit_ra, "dec:", hit_dec)

    slct_close_true = np.isclose(ra_true,ra_target,atol=0.001) & np.isclose(dec_true, dec_target,atol=0.001)
    slct_close_lensed = np.isclose(ra,ra_target,atol=0.001) & np.isclose(dec, dec_target,atol=0.001)
    slct_z = np.isclose(z, z_target, atol = 0.001)
    slct = slct_close_true & slct_z

    print(np.sum(slct))

    def print_mag(hfiles, filter_type, frame, color, slct):
        data = get_mag(hfiles, filter_type, frame, color)
        print(filter_type, frame, color, data[slct])

    print_mag(hfiles, 'LSST', 'obs', 'u', slct)
    print_mag(hfiles, 'LSST', 'obs', 'g', slct)
    print_mag(hfiles, 'LSST', 'obs', 'r', slct)
    print_mag(hfiles, 'LSST', 'obs', 'i', slct)
    print_mag(hfiles, 'LSST', 'obs', 'z', slct)
    print_mag(hfiles, 'LSST', 'obs', 'y', slct)
    print("")
    print_mag(hfiles, 'LSST', 'rest', 'u', slct)
    print_mag(hfiles, 'LSST', 'rest', 'g', slct)
    print_mag(hfiles, 'LSST', 'rest', 'r', slct)
    print_mag(hfiles, 'LSST', 'rest', 'i', slct)
    print_mag(hfiles, 'LSST', 'rest', 'z', slct)
    print_mag(hfiles, 'LSST', 'rest', 'y', slct)
    print("")
    print_mag(hfiles, 'model', 'rest', 'g', slct)
    print_mag(hfiles, 'model', 'rest', 'r', slct)
    print_mag(hfiles, 'model', 'rest', 'i', slct)

    print("")
    print_Av_Rv(hfiles, 'total', slct)

    print("")
    print_Av_Rv(hfiles, 'disk', slct)

    print("")
    print_Av_Rv(hfiles, 'spheroid', slct)
    print("")

    print("z        ", z[slct])
    print("z_true   ", z_true[slct])
    print("z_target ", [z_target])
    print("")
    print("ra         ", ra[slct])
    print("ra_true    ", ra_true[slct])
    print("ra_target  ", [ra_target])
    print("")
    print("dec        ", dec[slct])
    print("dec_true   ", dec_true[slct])
    print("dec_target  ", [dec_target])

    print("")
    print("galaxyID: ", ids[slct])
    print('guess id: ', id_target_guess)
    print('given id: ', id_target)

    # plt.figure()
    # #plt.hist2d(ra,dec,bins=100,cmap='Blues',norm=clr.LogNorm())
    # h,xbins,ybins = np.histogram2d(ra,dec,bins=(100))
    # plt.pcolor(xbins,ybins,h.T, cmap='Blues',norm=clr.LogNorm())
    # plt.plot(ra_target, dec_target, 'rx')
    # plt.show()
    
