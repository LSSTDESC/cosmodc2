#!/usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import sys
import time
import h5py

from k_corr_step import *





def get_keys(hgroup):
    keys = []
    def _collect_keys(name, obj):
        if isinstance(obj, h5py.Dataset): 
            keys.append(name)
    hgroup.visititems(_collect_keys)
    return keys



def check_keys(keys1, keys2, keys3):
    result = True
    k1n = len(keys1)
    k2n = len(keys2)
    k3n = len(keys3)
    if(k1n == k2n and k1n == k3n):
        print "all key lengths are the same", k1n, k2n, k3n
    else:
        print "key lengths don't match", k1n, k2n, k3n
        result = False
    for key in keys3:
        if(key in keys2 and key in keys3):
            #print key, "is good"
            pass
        else:
            print "================"
            print key, "is bad?!"
            result = False;
    if( not result):
        print "Not correct number of keys"
    return result

def check_keys_vals(hg_kcorr, hg_gltcs1, hg_gltcs2, del_a, check_num):
    result = True
    keys_kcorr = get_keys(hg_kcorr)
    keys_gltcs1 = get_keys(hg_gltcs1)
    keys_gltcs2 = get_keys(hg_gltcs2)
    if check_keys(keys_kcorr, keys_gltcs1, keys_gltcs2):
        result = True
        # num_list = np.arange(0,len(keys_kcorr))
        # keys = np.random.choice(keys_kcorr,check_num,replace=False)
        # for key in keys:
        #     print "checking values in", key
        #     kcorr = hg_kcorr[key].value
        #     gltcs1  = hg_gltcs1[key].value
        #     # gltcs2 = hg_gltcs2[key].value
        #     # del_val = gltcs2 - gltcs1
        #     # dv_da = del_val/del_a
        #     no_match = kcorr != dv_da
        #     if( np.sum(no_match)>0):
        #         print "Bad values! in ",key, np.sum(no_match)
        #         result = False
        #     else:
        #         print "Values are good: ",key, np.sum(no_match)
    else:
        result = False


if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    steps = param.get_int_list("steps")
    gltcs_snapshot_ptrn = param.get_string("gltcs_snapshots_ptrn")
    k_corr_ptrn = param.get_string("output_ptrn")
    stepz = dtk.StepZ(sim_name = "AlphaQ")
    for i in range(0,len(steps)-1):
        step1 = steps[i+1]
        step2 = steps[i]
        print "======================================="
        print step1, "->", step2
        a1 = stepz.get_a(step1)
        a2 = stepz.get_a(step2)
        del_a = a2-a1
        hg_kcorr = h5py.File(k_corr_ptrn.replace("${step}",str(step1)),'r')['galaxyProperties']
        hg_gltcs1 = h5py.File(gltcs_snapshot_ptrn.replace("${step}",str(step1)),'r')['galaxyProperties']
        hg_gltcs2 = h5py.File(gltcs_snapshot_ptrn.replace("${step}",str(step2)),'r')['galaxyProperties']
        check_keys_vals(hg_kcorr,hg_gltcs1, hg_gltcs2, del_a, 1);
        
        
        
