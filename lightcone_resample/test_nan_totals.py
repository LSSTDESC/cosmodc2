#!/usr/bin/env python2.7

from __future__ import print_function, division


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys
import time

from lc_resample import get_keys

def test_nan_totals(param_file_name):
    param = dtk.Param(param_file_name)
    lightcone_fname = param.get_string('lightcone_fname')
    gltcs_fname = param.get_string('gltcs_fname')
    gltcs_metadata_ref = param.get_string('gltcs_metadata_ref')
    gltcs_slope_fname = param.get_string('gltcs_slope_fname')
    sod_fname = param.get_string("sod_fname")
    halo_shape_fname = param.get_string("halo_shape_fname")
    halo_shape_red_fname = param.get_string("halo_shape_red_fname")
    output_fname = param.get_string('output_fname')
    steps = param.get_int_list('steps')
    use_slope = param.get_bool('use_slope')
    substeps = param.get_int('substeps')
    use_substep_redshift = param.get_bool('use_substep_redshift')
    load_mask = param.get_bool("load_mask")
    mask_loc  = param.get_string("mask_loc")
    index_loc = param.get_string("index_loc")
    recolor = param.get_bool('recolor')
    short = param.get_bool('short')
    supershort = param.get_bool('supershort')
    cut_small_galaxies = param.get_bool('cut_small_galaxies')
    cut_small_galaxies_mass = param.get_float('cut_small_galaxies_mass')
    plot = param.get_bool('plot')
    plot_substep = param.get_bool('plot_substep')
    use_dust_factor = param.get_bool('use_dust_factor')
    dust_factors = param.get_float_list('dust_factors')
    ignore_mstar = param.get_bool('ignore_mstar')
    match_obs_color_red_seq = param.get_bool('match_obs_color_red_seq')
    rescale_bright_luminosity = param.get_bool('rescale_bright_luminosity')
    rescale_bright_luminosity_threshold = param.get_float('rescale_bright_luminosity_threshold')
    ignore_bright_luminosity = param.get_bool('ignore_bright_luminosity')
    ignore_bright_luminosity_threshold = param.get_float('ignore_bright_luminosity_threshold')

    version_major = param.get_int('version_major')
    version_minor = param.get_int('version_minor')
    version_minor_minor = param.get_int('version_minor_minor')
    
    hfile = h5py.File(output_fname.replace("${step}", "all"),'r')
    hgroup = hfile['galaxyProperties']

    keys = get_keys(hgroup)
    for key in keys:

        data = hgroup[key].value
        nan_count = np.sum(~np.isfinite(data))
        print("nan_coutn: ", nan_count, key)
        assert(nan_count == 0)

    for key in keys:
        if('total' in key and 'morphology' not in key):
            disk_key = key.replace('total', 'disk')
            print(key)
            print(disk_key)
            disk_data = hgroup[disk_key].value
            spheroid_data = hgroup[key.replace('total', 'spheroid')].value
            total_data = hgroup[key].value
            close = np.isclose(disk_data + spheroid_data, total_data)
            print("total mismatch: ", np.sum(~close))
            assert(np.sum(~close)==0)
            
            

if __name__ == "__main__":
    test_nan_totals(sys.argv[1])
