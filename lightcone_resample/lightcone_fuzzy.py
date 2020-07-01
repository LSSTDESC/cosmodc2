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
from halotools.utils import fuzzy_digitize

def load_hdf5(fname, step):
    hfile = h5py.File(fname,'r')
    dic = {}
    keys = hfile.keys()
    for key in keys:
        dic[key] = hfile[key].value
    dic['step'] = np.ones(dic[keys[0]].size)*step
    hfile.close()
    return dic

def save_hdf5(fname, dic):
    hfile = h5py.File(fname,'w')
    for key in dic.keys():
        hfile[key] = dic[key]
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

def slct_dic(dic,slct):
    new_dic = {}
    for key in dic.keys():
        new_dic[key] = dic[key][slct]
    return new_dic

if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    steps = param.get_int_list("steps")
    redshifts = []
    lightcone_output_fname = param.get_string("lightcone_output_fname")
    lightcone_output_fuzzy_fname = lightcone_output_fname.replace(".hdf5","")+".fuzzy.hdf5"
    dics = []
    stepz = dtk.StepZ(sim_name="AlphaQ")
    for step in steps:
        print("step: ",step)
        dic = load_hdf5(lightcone_output_fname.replace("${step}",str(step)),step)
        dics.append(dic)
        redshifts.append(stepz.get_z(step))
    lc = cat_dics(dics)
    max_z = stepz.get_z(steps[-1])
    min_z = stepz.get_z(steps[0])
    print(min_z, max_z)
    slct_too_high = lc['redshift']>max_z
    slct_too_low  = lc['redshift']<min_z
    slct_good = ~slct_too_high & ~slct_too_low
    print("size: ",slct_good.size)
    print("inside: ",np.sum(slct_good))
    print("lower: ",np.sum(slct_too_high))
    print("higher: ",np.sum(slct_too_low))
    index = np.zeros(lc['redshift'].size)
    index[slct_too_high] = steps[-1]
    index[slct_too_low] = steps[0]
    indx = fuzzy_digitize(lc['redshift'][slct_good],redshifts)
    index[slct_good] = steps[indx]
    print("zeros: ", np.sum(index == 0))
    plt.figure()
    bins = np.linspace(0,1,250)
    for step in steps:
        slct = index == step
        lc_step = slct_dic(lc,slct)
        fname = lightcone_output_fuzzy_fname.replace("${step}",str(step))
        print(fname)
        save_hdf5(fname,lc_step)
        h,xbins = np.histogram(lc['redshift'][slct],bins=bins)
        plt.plot(dtk.bins_avg(xbins),h,label='step {}'.format(step))
    plt.grid()
    plt.xlabel('redshift');plt.ylabel('count')
    plt.title("fuzzied")
    #plt.yscale('log')
    plt.figure()
    for step in steps:
        slct = lc['step']==step
        h,xbins = np.histogram(lc['redshift'][slct],bins=bins)
        plt.plot(dtk.bins_avg(xbins),h,label='step {}'.format(step))
    plt.grid()
    plt.xlabel('redshift');plt.ylabel('count')
    plt.title('Original')
    #plt.yscale('log')

    plt.show()
