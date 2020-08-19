#!/usr/bin/env python2.7
from __future__ import print_function, division

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import time
import sys
import datetime
from astropy.table import Table
from scipy.spatial import cKDTree
from pecZ import pecZ
from astropy.cosmology import WMAP7 as cosmo
from scipy.interpolate import interp1d 


def load_mags(fname,frame,use_dust=False):
    dic ={}
    hgroup = h5py.File(fname,'r')
    if use_dust:
        dust = ':dustAtlas'
    else:
        dust = ''
    dic['mag_g'] = hgroup['galaxyProperties/SDSS_filters/magnitude:SDSS_g:'+frame+dust].value
    dic['mag_r'] = hgroup['galaxyProperties/SDSS_filters/magnitude:SDSS_r:'+frame+dust].value
    dic['mag_i'] = hgroup['galaxyProperties/SDSS_filters/magnitude:SDSS_i:'+frame+dust].value
    dic['clr_gr'] = dic['mag_g']-dic['mag_r']
    dic['clr_ri'] = dic['mag_r']-dic['mag_i']
    dic['mstar'] = hgroup['galaxyProperties/totalMassStellar'].value
    return dic

def multiply(dic,val):
    new_dic = {}
    for key in dic.keys():
        new_dic[key] = dic[key]*val
    return new_dic

def add(dic1, dic2):
    new_dic = {}
    for key in dic1.keys():
        new_dic[key] = dic1[key]+dic2[key]
    return new_dic;

def select_dic(dic1, slct):
    new_dic = {}
    for key in dic1.keys():
        new_dic[key] = dic1[key][slct]

def plot_colors(mags,title,frame):
    plt.figure()
    ybins = np.linspace(-1,2,100)
    if frame == 'rest':
        xbins = np.linspace(-26,-12,100)
    elif frame == 'observed':
        xbins = np.linspace(12,26,100) 
    elif frame == 'slope':
        xbins = np.linspace(-5,5,100)
        ybins = np.linspace(-5,5,100)
    else:
        print("frame \"{}\" is not allowed".format(frame))
        raise KeyError
    h,xbins,ybins = np.histogram2d(mags['mag_r'],mags['clr_gr'],bins=(xbins,ybins))
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.grid()
    plt.xlabel('mag_r');plt.ylabel('g-r')
    plt.title(title)

def make_slope(mags1, mags2, index_2to1):
    result = {}
    for key in mags1.keys():
        mags1_2ndstep = np.copy(mags1[key])
        slct = index_2to1 != -1
        mags1_2ndstep[slct] = mags2[key][index_2to1][slct]
        diff =  mags1_2ndstep - mags1[key] 
        result[key] = diff
        if(key == 'mag_r'):
            plt.figure()
            bins = np.linspace(-25,-5, 100)
            h, xbins, ybins = np.histogram2d(mags1[key][slct], mags2[key][index_2to1][slct], bins=(bins,bins))
            plt.pcolor(xbins,ybins,h.T, cmap='PuBu', norm=clr.LogNorm())
            plt.plot((bins[0], bins[-1]),(bins[0], bins[-1]), '--k')
            plt.xlabel('Mag_r step t')
            plt.ylabel('Mag_r step t+1')
            
            plt.show()
    return result



if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    gltcs_fname = param.get_string('gltcs_fname')
    index_loc = param.get_string('index_loc')
    steps = param.get_int_list('steps')
    stepz = dtk.StepZ(sim_name='AlphaQ')
    frame ='rest'
    #frame ='observed'
    for i in range(0,len(steps)-1):
        step1 = steps[i+1]
        step2 = steps[i]
        print(step1, step2)
        mag1 = load_mags(gltcs_fname.replace('${step}',str(step1)),frame,use_dust=True)
        mag2_raw = load_mags(gltcs_fname.replace('${step}',str(step2)),frame,use_dust=True)
        index_2to1 = h5py.File(index_loc.replace('${step}', str(step1)),'r')['match_2to1'].value
        slope1 = make_slope(mag1, mag2_raw, index_2to1)
        #slope1 = load_mags(gltcs_slope_fname.replace('${step}',str(step1)),frame, use_dust=True)
        continue
        plot_colors(mag1,str(step1),frame)
        for factor in np.linspace(0,1,25):
            print(factor)
            mag_slope = add(mag1,multiply(slope1,factor))
            plot_colors(mag_slope,"{:.2f} of the way from {} to {}.".format(factor,step1,step2),frame)
        plot_colors(mag2_raw,str(step2),frame)
        for factor in np.linspace(1,0,25):
            print(factor)
            mag_slope = add(mag1,multiply(slope1,factor))
            plot_colors(mag_slope,"{:.2f} of the way from {} to {}.".format(factor,step1,step2),frame)

        dtk.save_figs('figs/'+sys.argv[1]+'/'+__file__+"/")
        plt.close('all')
        plot_colors(mag1,step1,frame)
        plot_colors(mag2_raw, step2, frame)
        dtk.save_figs('figs/'+sys.argv[1]+'/'+__file__+"/other/")

        
