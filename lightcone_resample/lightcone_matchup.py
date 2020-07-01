#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import h5py 
import dtk
import sys
import time
import datetime
from astropy.table import Table

def load_lightcone(lc_fname, shr_fname, fake_shear):
    print("loading lightcone...",end='')
    t1 = time.time()
    lc = {}
    lc['x'] = np.fromfile(lc_fname.replace("${var}","x"),dtype='f4')
    lc['y'] = np.fromfile(lc_fname.replace("${var}","y"),dtype='f4')
    lc['z'] = np.fromfile(lc_fname.replace("${var}","z"),dtype='f4')
    lc['vx'] = np.fromfile(lc_fname.replace("${var}","vx"),dtype='f4')
    lc['vy'] = np.fromfile(lc_fname.replace("${var}","vy"),dtype='f4')
    lc['vz'] = np.fromfile(lc_fname.replace("${var}","vz"),dtype='f4')
    lc['id'] = np.fromfile(lc_fname.replace("${var}","id"),dtype='i8')
    lc['ra'] = np.fromfile(lc_fname.replace("${var}","theta"),dtype='f4')/3600.0-87.5
    lc['dec'] = (np.fromfile(lc_fname.replace("${var}","phi"),dtype='f4')/3600.0)-2.5
    lc['redshift'] = np.fromfile(lc_fname.replace("${var}","redshift"),dtype='f4')
    lc['lightcone_rotation'] = np.fromfile(lc_fname.replace("${var}","rotation"),dtype='i4')
    lc['lightcone_replication'] = np.fromfile(lc_fname.replace("${var}","replication"),dtype='i4')
    #shear info
    if fake_shear:
        print("Faking the shears!")
        zeros = np.zeros(lc['x'].size,dtype='f4')
        lc['ra_lensed']     = np.copy(lc['ra'])
        lc['dec_lensed']    = np.copy(lc['dec'])
        lc['shear1']        = np.copy(zeros)
        lc['shear2']        = np.copy(zeros)
        lc['magnification'] = np.copy(zeros)
        lc['convergence']   = np.copy(zeros)
    else:
        lc['ra_lensed'] = np.fromfile(shr_fname.replace("${var}","xr1"),dtype='f4')/3600.0
        lc['dec_lensed'] = np.fromfile(shr_fname.replace("${var}","xr2"),dtype='f4')/3600.0
        lc['shear1'] = np.fromfile(shr_fname.replace("${var}","sr1"),dtype='f4')
        lc['shear2'] = np.fromfile(shr_fname.replace("${var}","sr2"),dtype='f4')
        lc['magnification'] = np.fromfile(shr_fname.replace("${var}","mra"),dtype='f4')
        lc['convergence'] = np.fromfile(shr_fname.replace("${var}","kr0"),dtype='f4')
    keys = lc.keys()
    size = lc[keys[0]].size
    for key in keys:
        assert lc[key].size == size, '{}'.format(key) 
    print("done {}".format(time.time()-t1))
    return lc

def load_snapshot(ss_fname):
    print("loading snapshot...",end='')
    t1 = time.time()
    ss = {}
    tbl = Table.read(ss_fname,path='data')
    # hfile = h5py.File(ss_fname,'r')
    keys = tbl.keys()
    for key in keys:
         print("\t",key)
         ss[key]=tbl[key].quantity
    print("done. {}".format(time.time()-t1))
    return ss



def match_up(lc, ss, output):
    t1 = time.time()
    ss_id = ss['lightcone_id']
    srt = np.argsort(ss_id)
    indx = dtk.search_sorted(ss_id, lc['id'], sorter=srt)
    num_not_found = np.sum(indx == -1)
    print("num not found: ",num_not_found)
    # plt.figure()
    # plt.plot(ss['x'],ss['y'],'.',alpha=0.3)
    # plt.figure()
    # plt.plot(lc['x'],lc['y'],'.',alpha=0.3)
    # plt.show()
    # exit()
    print(lc['id'][indx ==-1])
    assert(num_not_found == 0)
    hfile = h5py.File(output,'w')
    lc_keys = lc.keys()
    for key in lc_keys:
        if(key != 'id'):
            hfile[key] = lc[key]
    ss_keys = ss.keys()
    for key in ss_keys:
        print("\t",key,end="")
        if(key not in lc_keys):
            t2 = time.time()
            hfile[key]=ss[key][indx]
            print("--",time.time()-t2)
        else:
            print(" not copied")
    print('done. ',time.time()-t1)
    hfile.close()

def cat_dics(dics, keys = None):
    new_dic = {}
    if keys is None:
        keys = dics[0].keys()
    for key in keys:
        new_dic[key] = []
        for dic in dics:
            new_dic[key].append(dic[key])
        new_dic[key] = np.concatenate(new_dic[key])
    return new_dic

def dic_to_hdf5(fname, dic):
    hfile = h5py.File(fname,'w')
    for key in dic.keys():
        hfile[key] = dic[key]
    hfile.close()

if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    lightcone_bin_fname = param.get_string("lightcone_bin_fname")
    shear_bin_fname = param.get_string("shear_bin_fname")
    snapshot_galaxy_fname = param.get_string("snapshot_galaxy_fname")
    lightcone_output_fname = param.get_string("lightcone_output_fname")
    steps = param.get_int_list("steps")
    steps_shr = param.get_string_list("steps_shr")
    fake_shears = param.get_bool("fake_shears")
    t0 =time.time()
    lcs = []
    for step,step_shr in zip(steps,steps_shr):
        t1 = time.time()
        print("\n\n=====================\n STEP: {}".format(step))
        lightcone_step_fname = lightcone_bin_fname.replace("${step}",str(step))
        shear_step_fname = shear_bin_fname.replace("${step}",str(step)).replace("${step_shr}",step_shr)
        lc = load_lightcone(lightcone_step_fname, shear_step_fname, fake_shears)
        lcs.append(lc)
        ss = load_snapshot(snapshot_galaxy_fname.replace("${step}",str(step)))
        output_fname = lightcone_output_fname.replace("${step}",str(step))
        match_up(lc, ss, output_fname)
        print("\n=== done: {}".format(time.time()-t1))
    #dic_to_hdf5(lightcone_output_fname.replace("${step}","all"),cat_dics(lcs))
    print("\n\n=======================\n=========================")
    print("All done: ",time.time()-t0)
