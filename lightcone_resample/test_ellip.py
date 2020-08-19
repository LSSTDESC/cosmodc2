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
import os
from ellipticity_model_testing import *
from scipy.stats import johnsonsb
from halotools.utils import rank_order_percentile
dirname = "tmp/"
def load_ellipticity_pdf(source, sample):
    if source == 'cosmos':
        basename = 'ellipticity_{0}_cosmos.txt'.format(sample)
        fname = os.path.join(dirname, basename)
        X = np.loadtxt(fname)
        _e = X[:, 0]
        _counts = X[:, 1]
        mask = ~np.isnan(_e) & ~np.isnan(_counts)
        e = _e[mask]
        counts = _counts[mask]
        _x = np.diff(e)
        dx = np.insert(_x, 0, _x[0])
        pdf = counts/np.sum(counts)/dx
        return e, pdf, counts
    elif source == 'pdc2':
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def test_ellip():
    loc1 = "/homes/dkorytov/data/cosmoDC2/protoDC2_v4.5/v4.5_test/v4.5.full.all.hdf5"
    loc2 = "/homes/dkorytov/data/cosmoDC2/protoDC2_v4.5/old/v4.5.full.all.hdf5"
    hfile1 = h5py.File(loc1, 'r')
    hfile2 = h5py.File(loc2, 'r')
    d_ellip1 = hfile1['galaxyProperties/morphology/diskEllipticity'].value
    d_ellip2 = hfile2['galaxyProperties/morphology/diskEllipticity'].value
    b_ellip1 = hfile1['galaxyProperties/morphology/spheroidEllipticity'].value
    b_ellip2 = hfile2['galaxyProperties/morphology/spheroidEllipticity'].value
    t_ellip1 = hfile1['galaxyProperties/morphology/totalEllipticity'].value
    t_ellip2 = hfile2['galaxyProperties/morphology/totalEllipticity'].value

    d_ratio1 = hfile1['galaxyProperties/morphology/diskAxisRatio'].value
    d_ratio2 = hfile2['galaxyProperties/morphology/diskAxisRatio'].value
    b_ratio1 = hfile1['galaxyProperties/morphology/spheroidAxisRatio'].value
    b_ratio2 = hfile2['galaxyProperties/morphology/spheroidAxisRatio'].value
    t_ratio1 = hfile1['galaxyProperties/morphology/totalAxisRatio'].value
    t_ratio2 = hfile2['galaxyProperties/morphology/totalAxisRatio'].value


    plt.figure()
    plt.title('disk ellip')
    plt.hist(d_ellip1,bins=100, alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ellip2,bins=100, alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.figure()
    plt.title('bulge ellip')
    plt.hist(d_ellip1,bins=100,alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ellip2,bins=100,alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.figure()
    plt.title('total ellip')
    plt.hist(d_ellip1,bins=100,alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ellip2,bins=100,alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.figure()
    plt.title('disk ratio')
    plt.hist(d_ratio1,bins=100, alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ratio2,bins=100, alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.figure()
    plt.title('bulge ratio')
    plt.hist(d_ratio1,bins=100,alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ratio2,bins=100,alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.figure()
    plt.title('total ratio')
    plt.hist(d_ratio1,bins=100,alpha=0.3,lw=0.1,label='new')
    plt.hist(d_ratio2,bins=100,alpha=0.3,lw=0.1,label='old')
    plt.legend(loc='best')
    plt.yscale('log')

    plt.show()


def convert_e2_e(e):
    q = np.sqrt((1.0 - e*e)/(1.0+e*e))
    return (1-q)/(1+q)

def convert_e_e2(e):
    q = (1.0 - e)/(1.0+e)
    return np.sqrt((1-q*q)/(1+q*q))

def combine_e(ellip_b, lum_b, ellip_d, lum_d):
    e_tot = (ellip_b*lum_b + ellip_d*lum_d)/(lum_d + lum_b)
    return e_tot

def combine_e2(ellip_b, lum_b, ellip_d, lum_d):
    ellip_b = convert_e2_e(ellip_b)
    ellip_d = convert_e2_e(ellip_d)
    return convert_e_e2(combine_e(ellip_b, lum_b, ellip_d, lum_d))

def test_models():
    loc1 = "/homes/dkorytov/data/cosmoDC2/protoDC2_v4.5/v4.5_test/v4.5.full.all.hdf5"
    hfile = h5py.File(loc1, 'r')['galaxyProperties']
    mag_r = hfile['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_i = hfile['SDSS_filters/magnitude:SDSS_i:observed:dustAtlas'].value
    mag_v = -2.5*np.log10(hfile['otherLuminosities/totalLuminositiesStellar:V:rest:dustAtlas'].value)
    
    lum_bulge = hfile['SDSS_filters/spheroidLuminositiesStellar:SDSS_r:rest:dustAtlas'].value
    lum_disk  = hfile['SDSS_filters/diskLuminositiesStellar:SDSS_r:rest:dustAtlas'].value
    lum_bulge[~np.isfinite(lum_bulge)] = 0
    lum_disk[~np.isfinite(lum_disk)] = 0


    q_disk    = hfile['morphology/diskAxisRatio'].value
    q_bulge  = hfile['morphology/spheroidAxisRatio'].value
    e_tot     = hfile['morphology/totalEllipticity'].value
    ellip_disk_cat = convert_e_e2(hfile['morphology/diskEllipticity'].value)
    ellip_bulge_cat = convert_e_e2(hfile['morphology/spheroidEllipticity'].value)
    ellip_tot_cat = combine_e2(ellip_disk_cat, lum_disk, ellip_bulge_cat, lum_bulge)
    q_tot     = (1.0-e_tot)/(1.0+e_tot)
    ellip_tot_cat2  = convert_e_e2(e_tot)
    lum_bulge[~np.isfinite(lum_bulge)]=0
    lum_disk[~np.isfinite(lum_disk)]=0
    lum_total = lum_bulge + lum_disk
    lum_bt = lum_bulge/lum_total
    not_eq = lum_total != (lum_bulge + lum_disk)

    print("{:.2e}, {:.2e}".format(np.sum(not_eq), not_eq.size))
    fnt = np.isfinite(lum_total)
    ellip_bulge2 = monte_carlo_ellipticity_bulge(mag_r)
    ellip_disk2  = monte_carlo_ellipticity_disk(mag_r)
    #ellip_tot2 = (ellip_bulge2*lum_bulge + ellip_disk2*lum_disk)/lum_total
    ellip_tot2 = combine_e2(ellip_bulge2, lum_bulge, ellip_disk2, lum_disk)

    a_disk =  np.interp(mag_r, [-21,-19],[-0.4,-0.4])
    #a_disk = calculate_johnsonsb_params_disk(mag_r)
    b_disk = np.ones_like(a_disk)*0.7

    a_bulge = np.interp(mag_r, [-21,-19,-17],[.6,1.0,1.6])
    #a_bulge = calculate_johnsonsb_params_bulge(mag_r)
    b_bulge = np.interp(mag_r, [-19,-17],[1.0,1.0])
    #b_bulge = np.ones_like(a_bulge)

    urand = np.random.uniform(size=lum_total.size)
    urand2 = rank_order_percentile(1*urand + 0.6*np.random.uniform(size=lum_total.size))

    ellip_bulge_new = johnsonsb.isf(urand, a_bulge, b_bulge)
    ellip_disk_new =  johnsonsb.isf(urand2, a_disk, b_disk)
    #ellip_tot_new = (ellip_bulge_new*lum_bulge + ellip_disk_new*lum_disk)/lum_total
    ellip_tot_new = combine_e2(ellip_bulge_new,lum_bulge , ellip_disk_new,lum_disk)
    # plt.figure()
    # h,xbins,ybins = np.histogram2d(mag_v, ellip_bulge_new,bins=100)
    # plt.pcolor(xbins,ybins,h.T,cmap='Blues',norm=clr.LogNorm())
    # plt.xlabel('mag_v');plt.ylabel('ellip_bulge')

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(mag_v, a_bulge,bins=100)
    # plt.pcolor(xbins,ybins,h.T,cmap='Blues',norm=clr.LogNorm())
    # plt.xlabel('mag_v');plt.ylabel('a_bulge')

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(mag_r, ellip_bulge_new,bins=100)
    # plt.pcolor(xbins,ybins,h.T,cmap='Blues',norm=clr.LogNorm())
    # plt.xlabel('mag_r');plt.ylabel('ellip_bulge')

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(mag_r, a_bulge,bins=100)
    # plt.pcolor(xbins,ybins,h.T,cmap='Blues',norm=clr.LogNorm())
    # plt.xlabel('mag_r');plt.ylabel('a_bulge')

    #ellip_tot2 = (lum_bulge + lum_disk)/lum_total
    print(np.min(ellip_tot2), np.max(ellip_tot2), np.sum(~np.isfinite(ellip_tot2)))
    
    e_lrg_cosmos, pdf_lrg_cosmos, __ = load_ellipticity_pdf('cosmos', 'lrg')
    e_disk_cosmos, pdf_disk_cosmos, __ = load_ellipticity_pdf('cosmos', 'disk')
    e_early_cosmos, pdf_early_cosmos, __ = load_ellipticity_pdf('cosmos', 'early')
    e_late_cosmos, pdf_late_cosmos, __ = load_ellipticity_pdf('cosmos', 'late')



    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle("Current Model")
    slct_i = mag_i < 24.0
    slcta = mag_v < -19.0 
    slctb = (-21 < mag_v) & (mag_v< -17)
    slct_1 = slcta & (0.7 < lum_bt) & slct_i
    slct_2 = slctb & (0.7 < lum_bt) & slct_i
    slct_3 = slctb & (lum_bt < 0.2) & slct_i
    slct_4 = slctb & (0.4 < lum_bt) & (lum_bt < 0.7) & slct_i
    xbins = np.linspace(0,1,100)
    xbins_avg = dtk.bins_avg(xbins)

    ax1.set_title('LRG, 0.7 < B/T < 1.0, V<-19')
    ax1.plot(e_lrg_cosmos, pdf_lrg_cosmos, label = "COSMOS")
    h,_ = np.histogram(ellip_tot2[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, label='total')
    h,_ = np.histogram(ellip_disk2[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':b', label='disk')
    h,_ = np.histogram(ellip_bulge2[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':r',label='bulge')
    ax1.legend(loc='best',framealpha=0.3)
    #ax1.set_yscale('log')
    
    ax2.set_title('Early, 0.7 < B/T < 1.0, -21<V<-17')
    ax2.plot(e_early_cosmos, pdf_early_cosmos, label = "COSMOS early")
    h,_ = np.histogram(ellip_tot2[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk2[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge2[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':r',label='protoDC2')
    #ax2.set_yscale('log')

    ax3.set_title('Disk, 0.0 < B/T < 1.2, -21<V<-17')
    ax3.plot(e_disk_cosmos, pdf_disk_cosmos, label = "COSMOS disk")
    h,_ = np.histogram(ellip_tot2[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk2[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge2[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':r',label='protoDC2')
    #ax3.set_yscale('log')

    ax4.set_title('Late, 0.4 < B/T < 0.7, -21<V<-17')
    ax4.plot(e_late_cosmos, pdf_late_cosmos, label = "COSMOS late")
    h,_ = np.histogram(ellip_tot2[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk2[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge2[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':r',label='protoDC2')
    #ax4.set_yscale('log')



    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle("New Model")

    ax1.set_title('LRG, 0.7 < B/T < 1.0, V<-19')
    ax1.plot(e_lrg_cosmos, pdf_lrg_cosmos, label = "COSMOS")
    h,_ = np.histogram(ellip_tot_new[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, label='total')
    h,_ = np.histogram(ellip_disk_new[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':b', label='disk')
    h,_ = np.histogram(ellip_bulge_new[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':r',label='bulge')
    ax1.legend(loc='best',framealpha=0.3)
    # ax1.set_yscale('log')

    ax2.set_title('Early, 0.7 < B/T < 1.0, -21<V<-17')
    ax2.plot(e_early_cosmos, pdf_early_cosmos, label = "COSMOS early")
    h,_ = np.histogram(ellip_tot_new[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk_new[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_new[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':r',label='protoDC2')
    # ax2.set_yscale('log')

    ax3.set_title('Disk, 0.0 < B/T < 1.2, -21<V<-17')
    ax3.plot(e_disk_cosmos, pdf_disk_cosmos, label = "COSMOS disk")
    h,_ = np.histogram(ellip_tot_new[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk_new[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_new[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':r',label='protoDC2')
    # ax3.set_yscale('log')

    ax4.set_title('Late, 0.4 < B/T < 0.7, -21<V<-17')
    ax4.plot(e_late_cosmos, pdf_late_cosmos, label = "COSMOS late")
    h,_ = np.histogram(ellip_tot_new[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_disk_new[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_new[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':r',label='protoDC2')
    # ax4.set_yscale('log')



    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2,sharex=True,sharey=True)
    fig.suptitle("ProtoDC2 Catalog")

    ax1.set_title('LRG, 0.7 < B/T < 1.0, V<-19')
    ax1.plot(e_lrg_cosmos, pdf_lrg_cosmos, label = "COSMOS")
    h,_ = np.histogram(ellip_tot_cat[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, label='total calc')
    h,_ = np.histogram(ellip_tot_cat2[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, "--m",label='total written',lw=2)
    h,_ = np.histogram(ellip_disk_cat[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':b', label='disk')
    h,_ = np.histogram(ellip_bulge_cat[slct_1],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax1.plot(xbins_avg, h, ':r',label='bulge')
    ax1.legend(loc='best',framealpha=0.3)
    ax1.set_yscale('log')

    ax2.set_title('Early, 0.7 < B/T < 1.0, -21<V<-17')
    ax2.plot(e_early_cosmos, pdf_early_cosmos, label = "COSMOS early")
    h,_ = np.histogram(ellip_tot_cat[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_tot_cat2[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, "--m",label='protoDC2',lw=2)
    h,_ = np.histogram(ellip_disk_cat[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_cat[slct_2],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax2.plot(xbins_avg, h, ':r',label='protoDC2')
    ax2.set_yscale('log')

    ax3.set_title('Disk, 0.0 < B/T < 1.2, -21<V<-17')
    ax3.plot(e_disk_cosmos, pdf_disk_cosmos, label = "COSMOS disk")
    h,_ = np.histogram(ellip_tot_cat[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_tot_cat2[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, "--m",label='protoDC2',lw=2)
    h,_ = np.histogram(ellip_disk_cat[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_cat[slct_3],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax3.plot(xbins_avg, h, ':r',label='protoDC2')
    ax3.set_yscale('log')

    ax4.set_title('Late, 0.4 < B/T < 0.7, -21<V<-17')
    ax4.plot(e_late_cosmos, pdf_late_cosmos, label = "COSMOS late")
    h,_ = np.histogram(ellip_tot_cat[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, label='protoDC2')
    h,_ = np.histogram(ellip_tot_cat2[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, "--m",label='protoDC2')

    h,_ = np.histogram(ellip_disk_cat[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':b', label='protoDC2')
    h,_ = np.histogram(ellip_bulge_cat[slct_4],bins=xbins)
    h = h/np.sum(h)/(xbins[1:] - xbins[:-1])
    ax4.plot(xbins_avg, h, ':r',label='protoDC2')
    ax4.set_yscale('log')
    plt.show()



        

if __name__ == "__main__":
    #test_ellip()
    test_models();
