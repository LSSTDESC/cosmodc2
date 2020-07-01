#!/usr/bin/env python2.7
from __future__ import print_function, division
import sys

sys.path.insert(0, '/homes/dkorytov/.local/lib/python2.7/site-packages/halotools-0.7.dev4939-py2.7-linux-x86_64.egg')

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import pdb
import dtk
import h5py
import time
import sys
import datetime

from lc_resample import *

def refind_inclination(infall_index, gltcs_fname):
    t = time.time()
    print("refinding ellipticity")
    hfile = h5py.File(gltcs_fname,'r')
    gltcs_infall_index = hfile['galaxyProperties/infallIndex'].value
    print('sorting...')
    srt = np.argsort(gltcs_infall_index)
    indx = dtk.search_sorted(gltcs_infall_index, infall_index, sorter=srt)
    print(np.sum(indx==-1), indx.size)
    
    slct_notfound = indx==-1
    assert(np.sum(indx==-1) == 0)
    print('done. time: {:.2f}'.format(time.time()-t))
    return gltcs_infall_index[indx]

def recalculate_ellipticity(param_file_name):
    param = dtk.Param(param_file_name)
    steps = param.get_int_list('steps')
    gltcs_fname = param.get_string('gltcs_fname')
    gltcs_metadata_ref = param.get_string('gltcs_metadata_ref')
    output_fname = param.get_string('output_fname')
    ignore_mstar = param.get_bool('ignore_mstar')
    match_obs_color_red_seq = param.get_bool('match_obs_color_red_seq')
    rescale_bright_luminosity = param.get_bool('rescale_bright_luminosity')
    rescale_bright_luminosity_threshold = param.get_float('rescale_bright_luminosity_threshold')
    ignore_bright_luminosity = param.get_bool('ignore_bright_luminosity')
    ignore_bright_luminosity_threshold = param.get_float('ignore_bright_luminosity_threshold')
    version_major = param.get_int('version_major')
    version_minor = param.get_int('version_minor')
    version_minor_minor = param.get_int('version_minor_minor')
    output_file_list = []
    for i in range(0,len(steps)-1) :
        step = steps[i+1]
        print('working on step {}'.format(step))
        output_step_fname = output_fname.replace('${step}',str(step))
        output_file_list.append(output_step_fname)
        erase_ellipticity_quantities(output_step_fname)  
        add_ellipticity_quantities(output_step_fname)  
    output_all = output_fname.replace("${step}","all")
    combine_step_lc_into_one(output_file_list, output_all)
    add_metadata(gltcs_metadata_ref, output_all, version_major, version_minor, version_minor_minor)

    

if __name__ == "__main__":
    recalculate_ellipticity(sys.argv[1])

