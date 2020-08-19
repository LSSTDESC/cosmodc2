#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import dtk 
import h5py
import sys
import os
import time


def get_hfiles(dname, z_range):
    hfiles = []
    for file_name in sorted(os.listdir(dname)):
        if z_range in file_name:
            #print("opening ", file_name)
            hfiles.append(h5py.File(dname+"/"+file_name, 'r'))
    return hfiles

def get_variable(hfiles, step, var_name):
    result = []
    for hfile in hfiles:
        if var_name in  hfile[step]:
            values = hfile[step][var_name].value
            # print(hfile, np.sum(values!=0))
            result.append(values)
        else:
            print(var_name, "not in ", hfile[step].keys())
    return np.concatenate(result)

def get_step(hfiles):
    result = []
    for key in hfiles[0].keys():
        if key != 'metaData':
            result.append(key)
    return result

def baseDC2_galaxy_count(dname, z_range):
    hfiles = get_hfiles(dname, z_range)
    steps = get_step(hfiles)
    for step in steps:
        halo_id = get_variable(hfiles, step, 'halo_id')
        print(step, np.sum(halo_id >=0))
    
    


if __name__ == "__main__":
    baseDC2_galaxy_count(sys.argv[1], "z_2_3")
    baseDC2_galaxy_count(sys.argv[1], "z_1_2")
    baseDC2_galaxy_count(sys.argv[1], "z_0_1")
    

