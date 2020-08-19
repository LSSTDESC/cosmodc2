#!/usr/bin/env python2.7
from __future__ import print_function, division

import numpy as np
import scipy as sp
import pdb
import dtk
import h5py
import time
import sys
import datetime

import galmatcher




if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    steps = param.get_int_list("steps")
    gltcs_fname = param.get_string("gltcs_fname")
    mask_loc = param.get_string("mask_loc")
    selection1 = galmatcher.read_selections(yamlfile='galmatcher/yaml/vet_protoDC2.yaml')
    selection2 = galmatcher.read_selections(yamlfile='galmatcher/yaml/colors_protoDC2.yaml')
    selection3 = galmatcher.read_selections(yamlfile='galmatcher/yaml/observed_colors_protoDC2.yaml')
    selection4 = galmatcher.read_selections(yamlfile='galmatcher/yaml/Av_Rv.yaml')
    dtk.ensure_dir(mask_loc)
    print(mask_loc)
    hfile = h5py.File(mask_loc, 'a')
    wrote_something = False
    for step in steps:
        print("\n\n=============")
        print("STEP {}".format(str(step)))
        if '{}'.format(str(step)) not in hfile:
            gltcs_step_fname = gltcs_fname.replace('${step}',str(step))
            print("\n==== mask1 ====\n")
            mask1 = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection1)
            print("\n==== mask2 ====\n")
            mask2 = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection2)
            print("\n==== mask3 ====\n")
            mask3 = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection3)
            print("\n==== mask4 ====\n")
            mask4 = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection4)
            print("size: {:.2e}".format(mask1.size))
            print("mask1: {:.2e}\nmask2: {:.2e}\nmask3: {:.2e}\nmask4: {:.2e}".format(np.sum(mask1), np.sum(mask2), np.sum(mask3), np.sum(mask4)))
            mask_all = mask1 & mask2 & mask3 & mask4
            hfile['{}'.format(str(step))] = mask_all
            wrote_something = True
        else:
            print("already there...")
    if not wrote_something:
        print("=============================")
        print("we didn't write anything!!!!!")
        print("=============================")
