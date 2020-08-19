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
import pandas as pd
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


if __name__ == "__main__":
    fname = sys.argv[1]
    healpix_pixels = sys.argv[2:]
    hfiles = get_hfiles(fname, healpix_pixels)
    print(len(hfiles))
    #Slack
    target_ra, target_dec = 54.37508357,-32.40874507
    #Tricia
    target_ra, target_dec = 54.37836208,-32.40704645
    # Lensed cluster pos
    # target_ra, target_dec = 54.3827, -32.4219
    tolerance = 0.1
    print("\n")
    print("we are trying to find the above")
    print("RA:{} Dec:{}".format(target_ra, target_dec))

    pos_true = True
    if pos_true:
        ra = get_val(hfiles, 'ra_true')
        dec = get_val(hfiles, 'dec_true')
    else:
        ra = get_val(hfiles, 'ra')
        dec = get_val(hfiles, 'dec')

    redshift = get_val(hfiles, 'redshift')
    halo_id = get_val(hfiles,'uniqueHaloID')
    mass = get_val(hfiles, 'hostHaloMass')
    isCentral = get_val(hfiles, 'isCentral')
    # x = get_val(hfiles, "x")
    # y = get_val(hfiles, "y")
    mag = get_mag(hfiles, "LSST", "obs", "r")
    mag_i = get_mag(hfiles, "LSST", "obs", "i")
    mag_cut = 25
    slct1 = np.abs(ra - target_ra) < tolerance
    slct2 = np.abs(dec - target_dec) < tolerance
    slct3 = mag < mag_cut
    slct4 = halo_id == 106387004279
    slct_central = isCentral == isCentral
    slct = slct1 & slct2 & slct3

    plt.figure()
    #plt.scatter(ra[slct],dec[slct], s = (28-mag), marker='o', alpha = 0.3)
    plt.scatter(ra[slct],dec[slct], marker='o', c=isCentral[slct], alpha = 1.0, label='galaxies')

    cb = plt.colorbar()
    cb.set_label('central')
    plt.plot(target_ra, target_dec, 'rx')
    plt.title("Mag_r < {}".format(mag_cut))
    if pos_true:
        plt.xlabel('Ra True')
        plt.ylabel('Dec True')
    else:
        plt.xlabel('Ra Lesned')
        plt.ylabel('Dec Lensed')
    plt.tight_layout()
    plt.figure()
    plt.scatter(ra[slct], redshift[slct], c=isCentral[slct], cmap='coolwarm')
    plt.ylabel('redshift')
    plt.xlabel('ra')
    # plt.figure()
    # plt.scatter(ra[slct], y[slct], alpha=0.3)
    # plt.axvline(x=target_ra,ls='--', c='r')

    # plt.figure()
    # plt.scatter(x[slct], dec[slct], alpha=0.3)
    # plt.axhline(y=target_dec, ls='--', c='r')
    print(halo_id[slct])
    print(mass[slct])
    print("showing..")
    pd_dict = {'redshift':redshift[slct],
               'mag_r': mag[slct],
               'mag_i': mag_i[slct],
               'ra': ra[slct],
               'dec': dec[slct],
               'central': isCentral[slct],
               'halo_id': halo_id[slct],
    }
    df = pd.DataFrame.from_dict(pd_dict,)
    df.to_csv("~/tmp/weird_cluster.csv", index=False)
    plt.show()
