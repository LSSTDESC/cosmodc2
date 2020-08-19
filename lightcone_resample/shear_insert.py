#!/usr/bin/env python2.7
from __future__ import print_function, division

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


# if __name__ == "__main__2":
#     param = dtk.Param(sys.argv[1])
#     param2 = dtk.Param(sys.argv[2])
#     shear_new_loc = param.get_string('shear_loc')
#     shear_old_loc = param2.get_string('shear_bin_fname')
#     steps = param2.get_int_list('steps')
#     steps_shr = param2.get_string_list('steps_shr')
#     print(shear_new_loc)
#     print(shear_old_loc)
#     for i in range(1,5):
#         print(steps[i], steps_shr[i])
#         shear1_new = np.fromfile(shear_new_loc.replace("${step}",str(steps[i])).replace("${num}",str(1)), dtype='f8')
#         shear1_old = np.fromfile(
#             shear_old_loc
#             .replace("${step}",str(steps[i]))
#             .replace("${step_shr}",steps_shr[i])
#             .replace("${var}","sr1"),
#             dtype='f4')

#         plt.figure()
#         h, xbins, ybins = np.histogram2d(shear1_old, shear1_new, bins=250)
#         plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
#         plt.grid()
#         plt.xlabel('old shear');plt.ylabel('new_shear')
#         plt.show()
#     #0.043227
#     exit()
if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    cat_loc = param.get_string("cat_loc")
    shear_loc = param.get_string("shear_loc")
    id_loc = param.get_string("id_loc")
    steps = param.get_int_list("steps")
    #param2 = dtk.Param(sys.argv[2])
    shear_old_loc = param2.get_string("shear_bin_fname")
    t1 = time.time()
    hg = h5py.File(cat_loc,'r+')['galaxyProperties']
    cat_id = hg['UMachineNative/lightcone_id'].value
    print(cat_id.size)
    print(np.unique(cat_id).size)
    cat_step = hg['step'].value
    print("Done loading lc_id: {:.2f}".format(time.time()-t1))
    print("cat_lc_id size: ", cat_id.size)
    print("totalMassStellar size: ", hg['totalMassStellar'].value.size)
    for step in steps:
        print("Steps: ", step)
        print(cat_step)
        slct_step = cat_step == step
        print(np.sum(slct_step))
        cat_id_step = cat_id[slct_step]
        cat_id_step_srt = np.argsort(cat_id_step)
        lc_id  = np.fromfile(id_loc.replace("${step}",str(step)), dtype='i8')
        shear1_loc = shear_loc.replace("${step}",str(step)).replace("${num}","1")
        shear2_loc = shear_loc.replace("${step}",str(step)).replace("${num}","2")
        shear1 = np.fromfile(shear1_loc,dtype='f8')
        shear2 = -np.fromfile(shear2_loc,dtype='f8')
        srt = np.argsort(lc_id)
        indx = dtk.search_sorted(cat_id_step, lc_id, sorter=cat_id_step_srt)
        indx_into_mask = np.arange(cat_id.size)[slct_step][indx]
        bool_mask = np.zeros(cat_id.size,dtype=bool)
        bool_mask[indx_into_mask] = True
        print(np.sum(indx==-1),'/',indx.size, "not found")
        print(hg['shear1'].value[bool_mask] - shear1)
        
        plt.figure()
        plt.title(step)
        h,xbins = np.histogram(shear1 - hg['shear1'].value[bool_mask] , bins = 100, density=True)
        plt.plot(dtk.bins_avg(xbins), h, label='shear1')
        h,xbins = np.histogram(shear2 - hg['shear2'].value[bool_mask] , bins = 100, density=True)
        plt.plot(dtk.bins_avg(xbins), h, label='shear2')
        plt.legend(loc='best')
        plt.xlabel('shear new - shear old')
        plt.ylabel('count')
        plt.grid()
        plt.figure()
        h, xbins, ybins = np.histogram2d(hg['shear1'].value[bool_mask], shear1, bins=250)
        plt.pcolor(xbins,ybins,h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.grid()
        plt.title(step)
        plt.xlabel("old shear 1")
        plt.ylabel("new shear 1")

        plt.figure()
        h, xbins, ybins = np.histogram2d(hg['shear2'].value[bool_mask], shear2, bins=(xbins,ybins))
        plt.pcolor(xbins,ybins,h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.grid()
        plt.title(step)
        plt.xlabel("old shear 2")
        plt.ylabel("new shear 2")
        dtk.save_figs("figs/"+sys.argv[1]+"/"+__file__+"/")
        plt.show()

        hg['shear1'][bool_mask] = shear1
        hg['shear2'][bool_mask] = shear2
