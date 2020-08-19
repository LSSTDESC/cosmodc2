#!/usr/bin/env python2.7
from __future__ import print_function, division
import sys

sys.path.insert(0, '/homes/dkorytov/.local/lib/python2.7/site-packages/halotools-0.7.dev4939-py2.7-linux-x86_64.egg')
import os
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
import subprocess
from astropy.table import Table
from scipy.spatial import cKDTree
from pecZ import pecZ
from astropy.cosmology import WMAP7 as cosmo
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from cosmodc2.black_hole_modeling import monte_carlo_bh_acc_rate, bh_mass_from_bulge_mass, monte_carlo_black_hole_mass
from cosmodc2.size_modeling import mc_size_vs_luminosity_late_type, mc_size_vs_luminosity_early_type
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from cosmodc2.mock_diagnostics import mean_des_red_sequence_gr_color_vs_redshift, mean_des_red_sequence_ri_color_vs_redshift, mean_des_red_sequence_iz_color_vs_redshift
from ellipticity_model import monte_carlo_ellipticity_bulge_disk
from halotools.utils import fuzzy_digitize
import galmatcher 

def construct_gal_prop(fname, verbose=False, mask = None, mag_r_cut =
                       False):
    t1 = time.time()
    gal_prop = {}
    hfile = h5py.File(fname,'r')
    hgp = hfile['galaxyProperties']
    m_star = np.log10(hgp['totalMassStellar'].value)
    mag_g = hgp['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    mag_r = hgp['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_i = hgp['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    if mask is None:
        mask = np.ones(mag_r.size,dtype=bool)
    if mag_r_cut:
        mask = (mag_r < -10) & mask
    gal_prop['m_star'] = m_star[mask]
    gal_prop['Mag_r']  = mag_r[mask]
    gal_prop['clr_gr'] = mag_g[mask]-mag_r[mask]
    gal_prop['clr_ri'] = mag_r[mask]-mag_i[mask]
    if verbose:
        print('done loading gal prop. {}'.format(time.time()-t1))
    return gal_prop,mask


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


def select_dic(dic, slct):
    new_dic = {}
    for key in dic.keys():
        new_dic[key]=dic[key][slct]
    return new_dic


def clean_up_gal_prop(gal_prop):
    """For each galaxy, if any property is not finite, set all other
    properties to some value (4) that will not be selected by the
    kdtree query.

    """
    print("Cleaning up gal prop: ",end="")
    slct_nfnt =  ~np.isfinite(gal_prop['m_star'])
    for key in gal_prop.keys():
        slct_nfnt = slct_nfnt | ~np.isfinite(gal_prop[key])
        print("bad vals: ", np.sum(slct_nfnt))
    for key in gal_prop.keys():
        gal_prop[key][slct_nfnt] = -4
    return gal_prop


def construct_gal_prop_redshift_dust_raw(fname, index, step1, step2,
                                         step1_a, step2_a, target_a,
                                         mask1, mask2,
                                         dust_factor=1.0,
                                         match_obs_color_red_seq = False,
                                         cut_small_galaxies_mass = None, 
                                         snapshot = False):
    """Constructs gal_prop using the interpolation scheme from the galacticus
    snapshots and index matching galaxies in step2 to galaxies in step1. 
    """
    h_in_gp1 = h5py.File(fname.replace("${step}", str(step1)), 'r')['galaxyProperties']
    h_in_gp2 = h5py.File(fname.replace("${step}", str(step2)), 'r')['galaxyProperties']
    ##======DEBUG========
    # stepz = dtk.StepZ(sim_name="AlphaQ")
    # step1_ab = stepz.get_a(step1)
    # step2_ab = stepz.get_a(step2)
    # print("\t\t step1/2: {} - {}".format(step1, step2))
    # print("\t\t step1/2_a: {:.4f} -> {:.4f}".format(step1_a, step2_a))
    # print("\t\t step1/2_ab: {:.4f} -> {:.4f}".format(step1_ab, step2_ab))
    # print("\t\t step1/2_z: {:.4f} -> {:.4f}".format(1.0/step1_a-1.0, 1.0/step2_a-1.0))
    # print("\t\t step1/2_z2:{:.4f} -> {:.4f}".format(stepz.get_z(step1), stepz.get_z(step2)))
    # print("\t\t target a: {:.4f}".format(target_a))
    # print("\t\t target z: {:.3f}".format(1.0/target_a -1.0))
    # step1_a = stepz.get_a(step1)
    # step2_a = stepz.get_a(step2)
    ##=========DEBUG========
    lum_g_d = get_column_interpolation_dust_raw(
        'SDSS_filters/diskLuminositiesStellar:SDSS_g:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_r_d = get_column_interpolation_dust_raw(
        'SDSS_filters/diskLuminositiesStellar:SDSS_r:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_i_d = get_column_interpolation_dust_raw(
        'SDSS_filters/diskLuminositiesStellar:SDSS_i:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_g_b = get_column_interpolation_dust_raw(
        'SDSS_filters/spheroidLuminositiesStellar:SDSS_g:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_r_b = get_column_interpolation_dust_raw(
        'SDSS_filters/spheroidLuminositiesStellar:SDSS_r:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_i_b = get_column_interpolation_dust_raw(
        'SDSS_filters/spheroidLuminositiesStellar:SDSS_i:rest:dustAtlas',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    lum_g = lum_g_d + lum_g_b
    lum_i = lum_i_d + lum_i_b
    lum_r = lum_r_d + lum_r_b
    ##=======DEBUG======
    # print("=============")
    # print("lum_r non-finite: ", np.sum(~np.isfinite(lum_r)), np.sum(lum_r<0), lum_r.size)
    # print("lum_r_d non-finite: ", np.sum(~np.isfinite(lum_r_d)), np.sum(lum_r_d<0), lum_r_d.size)
    # print("lum_r_b non-finite: ", np.sum(~np.isfinite(lum_r_b)), np.sum(lum_r_b<0), lum_r_b.size)
    # slct_neg = lum_r<0
    # print(lum_r[slct_neg])
    # print(lum_r_d[slct_neg])
    # print(lum_r_b[slct_neg])
    # print("=============")
    ##=======DEBUG======
    m_star = get_column_interpolation_dust_raw(
        'totalMassStellar',
        h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
        target_a, dust_factor, snapshot=snapshot)
    node_index =  get_column_interpolation_dust_raw('infallIndex', h_in_gp1,
                                                    h_in_gp2, index, mask1, mask2, step1_a, step2_a, target_a,
                                                    dust_factor, snapshot=snapshot)

    mag_g = -2.5*np.log10(lum_g)
    mag_r = -2.5*np.log10(lum_r)
    mag_i = -2.5*np.log10(lum_i)
    size = m_star.size
    gal_prop = {}
    gal_prop['m_star'] = np.log10(m_star)
    gal_prop['Mag_r']  = mag_r
    gal_prop['clr_gr'] = mag_g - mag_r
    gal_prop['clr_ri'] = mag_r - mag_i
    gal_prop['dust_factor'] = np.ones(size, dtype='f4')*dust_factor
    gal_prop['index']  = np.arange(size, dtype='i8')
    gal_prop['node_index'] = node_index

    if match_obs_color_red_seq:
        lum_g_obs_d = get_column_interpolation_dust_raw(
            'SDSS_filters/diskLuminositiesStellar:SDSS_g:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_r_obs_d = get_column_interpolation_dust_raw(
            'SDSS_filters/diskLuminositiesStellar:SDSS_r:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_i_obs_d = get_column_interpolation_dust_raw(
            'SDSS_filters/diskLuminositiesStellar:SDSS_i:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_z_obs_d = get_column_interpolation_dust_raw(
            'SDSS_filters/diskLuminositiesStellar:SDSS_z:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_g_obs_s = get_column_interpolation_dust_raw(
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_g:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_r_obs_s = get_column_interpolation_dust_raw(
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_r:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_i_obs_s = get_column_interpolation_dust_raw(
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_i:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_z_obs_s = get_column_interpolation_dust_raw(
            'SDSS_filters/spheroidLuminositiesStellar:SDSS_z:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_g_obs = lum_g_obs_d + lum_g_obs_s
        lum_r_obs = lum_r_obs_d + lum_r_obs_s
        lum_i_obs = lum_i_obs_d + lum_i_obs_s
        lum_z_obs = lum_z_obs_d + lum_z_obs_s
        # Luminosity distance factors cancle when we compute galaxy color, 
        # so I'm not including them the magnitude calculation
        mag_g_obs = -2.5*np.log10(lum_g_obs)
        mag_r_obs = -2.5*np.log10(lum_r_obs)
        mag_i_obs = -2.5*np.log10(lum_i_obs)
        mag_z_obs = -2.5*np.log10(lum_z_obs)
        gal_prop['clr_gr_obs'] = mag_g_obs - mag_r_obs
        gal_prop['clr_ri_obs'] = mag_r_obs - mag_i_obs
        gal_prop['clr_iz_obs'] = mag_i_obs - mag_z_obs
        # Record LSST g-r color
        lum_g_obs_d_lsst = get_column_interpolation_dust_raw(
            'LSST_filters/diskLuminositiesStellar:LSST_g:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        lum_r_obs_d_lsst = get_column_interpolation_dust_raw(
            'LSST_filters/diskLuminositiesStellar:LSST_r:observed:dustAtlas',
            h_in_gp1, h_in_gp2, index, mask1, mask2, step1_a, step2_a,
            target_a, dust_factor, snapshot=snapshot)
        gal_prop['clr_gr_obs_lsst'] = -2.5*np.log10(lum_g_obs_d_lsst) + 2.5*np.log10( lum_r_obs_d_lsst)

    if not (cut_small_galaxies_mass is None):
        print("cutting out small galaxies in gltcs")
        slct_gal = gal_prop['m_star']>cut_small_galaxies_mass
        gal_prop = dic_select(gal_prop, slct_gal)
    # print("nan test")
    # print(np.sum(np.isnan(gal_prop['m_star'])))
    # print("not finite test")
    # print(np.sum(~np.isfinite(gal_prop['m_star'])))
    # print(gal_prop['m_star'][np.isnan(gal_prop['m_star'])])
    return gal_prop


def construct_lc_data(fname, match_obs_color_red_seq = False, verbose
                      = False, recolor=False, internal_step=None,
                      cut_small_galaxies_mass = None,
                      red_sequence_transition_mass_start=13.0,
                      red_sequence_transition_mass_end=13.5, 
                      snapshot = False):
    t1 = time.time()
    lc_data = {}
    if snapshot: # Snapshot baseDC2 format
        hfile_snap = h5py.File(fname,'r')
        hfile = hfile_snap['galaxyProperties']
        snapshot_redshift = hfile_snap['metaData/redshift'].value
    elif internal_step is None: # flat-file baseDC2 format
        hfile = h5py.File(fname,'r')
    else: # healpix basedDC2 format
        hfile = h5py.File(fname,'r')[str(internal_step)]
    non_empty_step = "obs_sm" in hfile
    if non_empty_step:
        lc_data['m_star'] = np.log10(hfile['obs_sm'].value)
        lc_data['Mag_r'] = hfile['restframe_extincted_sdss_abs_magr'].value
        lc_data['clr_gr'] = hfile['restframe_extincted_sdss_gr'].value
        lc_data['clr_ri'] = hfile['restframe_extincted_sdss_ri'].value
        if not snapshot:
            lc_data['redshift'] = hfile['redshift'].value
        else:
            lc_data['redshift'] = np.ones(lc_data['m_star'].size)*snapshot_redshift
        lc_data['sfr_percentile'] = hfile['sfr_percentile'].value
    else:
        lc_data['m_star'] = np.zeros(0, dtype=np.float)
        lc_data['Mag_r'] = np.zeros(0, dtype=np.float)
        lc_data['clr_gr'] = np.zeros(0, dtype=np.float)
        lc_data['clr_ri'] = np.zeros(0, dtype=np.float)
        if not snapshot:
            lc_data['redshift'] = hfile['redshift'].value
        else:
            lc_data['redshift'] = np.ones(lc_data['m_star'].size)*snapshot_redshift
        lc_data['sfr_percentile'] = np.zeros(0, dtype=np.float)
    if recolor:
        upid_mock = hfile['upid'].value
        mstar_mock = hfile['obs_sm'].value
        sfr_percentile_mock = hfile['sfr_percentile'].value
        host_halo_mvir_mock = hfile['host_halo_mvir'].value    
        redshift_mock = lc_data['redshift']
        a,b,c = assign_restframe_sdss_gri(upid_mock, mstar_mock, sfr_percentile_mock,
                                          host_halo_mvir_mock, redshift_mock)
        # plt.figure()
        # h,xbins,ybins = np.histogram2d(lc_data['Mag_r'], a, bins=250)
        # plt.pcolor(xbins,ybins, h.T, cmap='PuBu', norm =clr.LogNorm())
        # plt.grid()

        # plt.figure()
        # h,xbins,ybins = np.histogram2d(lc_data['clr_gr'], b, bins=250)
        # plt.pcolor(xbins,ybins, h.T, cmap='PuBu', norm =clr.LogNorm())
        # plt.grid()

        # plt.figure()
        # h,xbins,ybins = np.histogram2d(lc_data['clr_ri'], c, bins=250)
        # plt.pcolor(xbins,ybins, h.T, cmap='PuBu', norm =clr.LogNorm())
        # plt.grid()
        # plt.show()
        lc_data['Mag_r'] = a 
        lc_data['clr_gr'] = b
        lc_data['clr_ri'] = c
        #lc_data['Mag_r'], lc_data['clr_gr'], lc_data['clr_ri'] = [a,b,c]
    if match_obs_color_red_seq and non_empty_step:
        # print("match obs color red seq")
        host_halo_mvir_mock = hfile['host_halo_mvir'].value    
        is_on_red_seq_gr = hfile['is_on_red_sequence_gr'].value
        is_on_red_seq_ri = hfile['is_on_red_sequence_ri'].value
        mass_rs = soft_transition(np.log10(host_halo_mvir_mock), red_sequence_transition_mass_start,  red_sequence_transition_mass_end)
        lc_data['is_cluster_red_sequence'] = mass_rs & is_on_red_seq_gr & is_on_red_seq_ri
        #lc_data['is_cluster_red_sequence'] = is_on_red_seq_gr & is_on_red_seq_ri
        lc_data['clr_gr_obs'] = mean_des_red_sequence_gr_color_vs_redshift(lc_data['redshift'])
        lc_data['clr_ri_obs'] = mean_des_red_sequence_ri_color_vs_redshift(lc_data['redshift'])
        lc_data['clr_iz_obs'] = mean_des_red_sequence_iz_color_vs_redshift(lc_data['redshift'])
    elif match_obs_color_red_seq:
        lc_data['is_cluster_red_sequence'] = np.zeros(0,dtype=bool)
        lc_data['clr_gr_obs'] = np.zeros(0,dtype=bool)
        lc_data['clr_ri_obs'] = np.zeros(0,dtype=bool)
        lc_data['clr_iz_obs'] = np.zeros(0,dtype=bool)
    if not (cut_small_galaxies_mass is None):
        print("cutting out small galaxies!")
        # Cutting out low mass galaxies so it runs fasters
        slct_gals = lc_data['m_star']>cut_small_galaxies_mass
        lc_data = dic_select(lc_data, slct_gals)
    #TODO remove once bug is fixed
    lc_data['Mag_r'][~np.isfinite(lc_data['Mag_r'])] = -14.0

    if verbose:
        print('done loading lc data. {}'.format(time.time()-t1))
    return lc_data


def construct_lc_data_healpix(fname, match_obs_color_red_seq = False,
                              verbose = False, recolor=False,
                              internal_step=None,
                              cut_small_galaxies_mass = None,
                              healpix_pixels=None,
                              red_sequence_transition_mass_start=13.0,
                              red_sequence_transition_mass_end=13.5, 
                              snapshot=False):
    print("Construicting light cone data.")
    print("Input lightcone file pattern: ", fname)
    print("Healpix files: ",healpix_pixels)
    if healpix_pixels is None:
        print("No healpix used")
        lc_data = construct_lc_data(fname, match_obs_color_red_seq = match_obs_color_red_seq,
                                    verbose = verbose, recolor=recolor, internal_step = internal_step,
                                    cut_small_galaxies_mass = cut_small_galaxies_mass,
                                    red_sequence_transition_mass_start = red_sequence_transition_mass_start,
                                    red_sequence_transition_mass_end = red_sequence_transition_mass_end, 
                                    snapshot=snapshot)
    else:
        lc_data_hps = []
        for healpix_pixel in healpix_pixels:
            fname_healpix = fname.replace('${healpix}', str(healpix_pixel))
            lc_data_hp = construct_lc_data(fname_healpix, 
                                           match_obs_color_red_seq = match_obs_color_red_seq,
                                           recolor=recolor, 
                                           internal_step = internal_step,
                                           cut_small_galaxies_mass = cut_small_galaxies_mass,
                                           red_sequence_transition_mass_start = red_sequence_transition_mass_start,
                                           red_sequence_transition_mass_end = red_sequence_transition_mass_end, verbose=False, 
                                           snapshot=snapshot)

            lc_data_hp['healpix_pixel'] = np.ones(lc_data_hp['m_star'].size, dtype='i4')*healpix_pixel
            lc_data_hps.append(lc_data_hp)
        lc_data = cat_dics(lc_data_hps)
    for key in lc_data.keys():
        num = np.sum(~np.isfinite(lc_data[key]))
        assert num == 0
    return lc_data


def dic_select(dic, slct):
    new_dic = {}
    for key in dic.keys():
        new_dic[key] = dic[key][slct]
    return new_dic


def select_by_index(data,index):
    new_data = {}
    for key in data.keys():
        new_data[key] = data[key][index]
    return new_data


def squash_magnitudes(mag_dic, lim, a):
    # I'm using a tanh function for the soft threshold. No magnitude will excessed 
    # 'lim'. Mags below lim-a aren't affect.  
    # plt.figure()
    # plt.hist2d(mag_dic[:,0],mag_dic[:,1],bins=250,cmap='Blues',norm=clr.LogNorm())
    # plt.axvline(x=lim+a,c='k',ls=':')
    # plt.axvline(x=lim,c='k',ls='--')
    # xlims = plt.xlim()
    # plt.title("before magnitude squash")
    # plt.xlabel('Mag_r')
    # plt.ylabel('g-r rest')
    if(a == 0.0):
        slct = mag_dic[:,0]<lim
        mag_dic[slct,0]= lim 
    else:
        slct = mag_dic[:,0]<lim+a
        mag_dic[slct,0]=a*np.tanh((mag_dic[slct,0]-lim-a)/a) + lim + a
    # plt.figure()
    # plt.hist2d(mag_dic[:,0],mag_dic[:,1],bins=250,cmap='Blues',norm=clr.LogNorm())
    # plt.axvline(x=lim+a,c='k',ls=':')
    # plt.axvline(x=lim,c='k',ls='--')
    # plt.xlim(xlims)
    # plt.title("after magntiude squash")
    # plt.xlabel('Mag_r')
    # plt.ylabel('g-r rest')

    # plt.show()
    return mag_dic


def resample_index(lc_data, gal_prop, ignore_mstar = False, nnk = 10,
                   verbose = False, ignore_bright_luminosity=False,
                   ignore_bright_luminosity_threshold=None,
                   ignore_bright_luminosity_softness=0.0, ):
    if verbose:
        t1 = time.time()
        print("Starting kdtree resampling")
        print("\nNum LC Galaxies: {:.2e} Num Gltcs Galaxies: {:.2e}".format(lc_data['m_star'].size, gal_prop['m_star'].size))
    m_star = lc_data['m_star']
    mag_r  = lc_data['Mag_r']
    clr_gr = lc_data['clr_gr']
    clr_ri = lc_data['clr_ri']
    if ignore_mstar:
        print("\tIgnoring Mstar!")
        lc_mat = np.stack((mag_r,clr_gr,clr_ri),axis=1)
        gal_mat = np.stack((gal_prop['Mag_r'],
                            gal_prop['clr_gr'],
                            gal_prop['clr_ri']),axis=1)
    else:
        lc_mat = np.stack((mag_r,clr_gr,clr_ri,m_star),axis=1)
        gal_mat = np.stack((gal_prop['Mag_r'],
                            gal_prop['clr_gr'],
                            gal_prop['clr_ri'],
                            gal_prop['m_star']),axis=1)
    if ignore_bright_luminosity:
        lc_mat = squash_magnitudes(lc_mat, ignore_bright_luminosity_threshold, ignore_bright_luminosity_softness)
        gal_mat = squash_magnitudes(gal_mat, ignore_bright_luminosity_threshold, ignore_bright_luminosity_softness)
        # slct_lc_mat = lc_mat[:,0]<ignore_bright_luminosity_threshold
        # lc_mat[slct_lc_mat,0] = ignore_bright_luminosity_threshold
        # slct_gal_mat = gal_mat[:,0]< ignore_bright_luminosity_threshold
        # gal_mat[slct_gal_mat,0] = ignore_bright_luminosity_threshold
    if verbose:
        t2 = time.time()
        print('\tdone formating data. {}'.format(t2-t1))
        print("data size: {:.2e}".format(m_star.size))

    # if the search size is large enough, it's saves total time to construct a 
    # faster to search tree. Otherwise build a quick tree. 
    if m_star.size > 3e7:
        if verbose:
            print("long balanced tree")
        ckdtree = cKDTree(gal_mat, balanced_tree = True, compact_nodes = True)
    elif m_star.size > 3e6: 
        if verbose:
            print("long tree")
        ckdtree = cKDTree(gal_mat, balanced_tree = False, compact_nodes = True)
    else:
        if verbose:
            print("quick tree")
        ckdtree = cKDTree(gal_mat, balanced_tree = False, compact_nodes = False)

    if verbose:
        t3 = time.time()
        print('\tdone making tree. {}'.format(t3-t2))
    dist_raw, index_raw = ckdtree.query(lc_mat, nnk, n_jobs=10)
    if verbose:
        t4= time.time()
        print('\tdone querying. {}'.format(t4-t3))
    if nnk > 1:
        rand = np.random.randint(nnk,size=index_raw.shape[0])
        aa = np.arange(index_raw.shape[0])
        #dist = dist[aa,rand]
        index = index_raw[aa,rand]
    else:
        index = index_raw
    ##======DEBUG===========
    # print("lc_data size:")
    # for k in lc_data:
    #     print(k,np.sum(~np.isfinite(lc_data[k])),'/',lc_data[k].size)
    # print("\n\ngal_prop size:")
    # for k in gal_prop:
    #     print(k,np.sum(~np.isfinite(gal_prop[k])),'/',gal_prop[k].size)

    # print("index min/max: ", np.min(index), np.max(index))
    # print("ckdtree size: ",gal_mat[:,0].size, gal_mat[0,:].size)
    # plt.figure()
    # h,xbins = np.histogram(index, bins=1000)
    # plt.plot(dtk.bins_avg(xbins), h)
    # plt.grid()
    # plt.xlabel('index num')
    # plt.ylabel('count')
    ##======DEBUG===========
    return index
   

def resample_index_cluster_red_squence(lc_data, gal_prop, ignore_mstar
                                       = False, nnk = 10, verbose =
                                       False,
                                       ignore_bright_luminosity=False,
                                       ignore_bright_luminosity_threshold=False,
                                       ignore_bright_luminosity_softness=0.0,
                                       rs_scatter_dict = {}):
    if verbose:
        t1 = time.time()
        print("Starting kdtree resampling with obs colors")
    lc_data_list = []
    gal_prop_list = []
    # We modify the lightcone/baseDC2/query data with rs scatter, if listed
    lc_data_list += (lc_data['Mag_r'],
                     lc_data['clr_gr'],
                     lc_data['clr_ri'],
                     modify_array_with_rs_scatter(lc_data, "query", "gr", rs_scatter_dict), #clr_gr_obs
                     modify_array_with_rs_scatter(lc_data, "query", "ri", rs_scatter_dict), #clr_ri_obs
                     modify_array_with_rs_scatter(lc_data, "query", "iz", rs_scatter_dict), #clr_iz_obs
    )
    # We modify the galaxy properties/galactics/tree data with rs scatter, if listed
    gal_prop_list += (gal_prop['Mag_r'],
                      gal_prop['clr_gr'],
                      gal_prop['clr_ri'],
                      modify_array_with_rs_scatter(gal_prop, "tree", "gr", rs_scatter_dict), #clr_gr_obs
                      modify_array_with_rs_scatter(gal_prop, "tree", "ri", rs_scatter_dict), #clr_ri_obs
                      modify_array_with_rs_scatter(gal_prop, "tree", "iz", rs_scatter_dict), #clr_iz_obs
    )

    if ignore_mstar:
        pass
    else:
        lc_data_list.append(lc_data['m_star'])
        gal_prop_list.append(gal_prop['m_star'])
    lc_mat = np.transpose(lc_data_list)
    gal_mat = np.transpose(gal_prop_list)
    if ignore_bright_luminosity:
        if(ignore_bright_luminosity_softness == 0.0):
            slct_lc_mat = lc_mat[:,0]<ignore_bright_luminosity_threshold
            lc_mat[slct_lc_mat,0] = ignore_bright_luminosity_threshold
            slct_gal_mat = gal_mat[:,0]< ignore_bright_luminosity_threshold
            gal_mat[slct_gal_mat,0] = ignore_bright_luminosity_threshold
        else:
            lim = ignore_bright_luminosity_threshold
            a = ignore_bright_luminosity_softness
            slct_lc_mat = lc_mat[:,0]<lim+a
            lc_mat[slct_lc_mat,0]=a*np.tanh((lc_mat[slct_lc_mat,0]-lim-a)/a) + lim +a
            slct_gal_mat = gal_mat[:,0]<lim+a
            gal_mat[slct_gal_mat,0]=a*np.tanh((gal_mat[slct_gal_mat,0]-lim-a)/a) + lim +a
    if verbose:
        t2 = time.time()
        print("\tdone formatting data. {}".format(t2-t1))
    if lc_data['m_star'].size > 3e6: 
        if verbose:
            print("long tree")
        ckdtree = cKDTree(gal_mat, balanced_tree = False, compact_nodes = True)
    else:
        if verbose:
            print("quick tree")
        ckdtree = cKDTree(gal_mat, balanced_tree = False, compact_nodes = False)
    if verbose:
        t3 = time.time()
        print("\tdone making kdtree. {}".format(t3-t2))
    dist, index = ckdtree.query(lc_mat, nnk, n_jobs=10)
    if verbose:
        t4 = time.time()
        print("\tdone querying. {}".format(t4-t3))
    if nnk > 1:
        rand = np.random.randint(nnk,size=dist.shape[0])
        aa = np.arange(dist.shape[0])
        #dist = dist[aa,rand]
        index = index[aa,rand]
    # return orignal_index[slct_valid][index]
    return index
        

def get_keys(hgroup):
    keys = []
    def _collect_keys(name, obj):
        if isinstance(obj, h5py.Dataset): 
            keys.append(name)
    hgroup.visititems(_collect_keys)
    return keys


def soft_transition(vals, trans_start, trans_end):
    if(vals.size ==0):
        return np.ones(vals.size,dytpe='bool')
    slct_between = (vals>trans_start) & (vals<trans_end)
    if(trans_start == trans_end or np.sum(slct_between) == 0):
        return vals>trans_start
    elif(trans_start > trans_end):
        raise ValueError('Trans_start value is greater than trans_end')
    else:
        # print(trans_start, trans_end)
        # print(vals.size)
        # print(np.sum(slct_between))
        # print(vals[slct_between])
        bins = fuzzy_digitize(vals[slct_between],[trans_start,trans_end], min_counts=0)
        result = np.ones(vals.size, dtype='bool')
        result[vals<=trans_start] = False
        result[slct_between] = bins==1
        return result


def get_rs_scatter_dict_from_param(param):
    """This function takes in a dtk.Param object and returns a dictionary 
    containing the scatter"""
    print("seaching param file for red squence scatter information")
    rs_scatter_dict = {}
    colors = ['gr', 'ri', 'iz']
    scatter_locations = ['query', 'tree']
    for scatter_loc in scatter_locations:
        for color in colors:
            key = "red_sequence_scatter_{}_{}".format(scatter_loc, color)
            if key in param:
                val = param.get_float(key)
                print("found {} = {:f} in param file".format(key, val))
                rs_scatter_dict[key] = val
    return rs_scatter_dict
    

def modify_data_with_rs_scatter(data_dict, data_type, rs_scatter_dict):
    data_dict = data_dict.copydeep()
    assert data_type == "query" or data_type == "tree", "Data type must be either \"query\" or \"tree\". Given data_type is {}".format(data_type)
    colors = ['gr', 'ri', 'iz']
    for color in colors:
        rs_scatter_key = 'rs_scatter_{}_{}'.format(data_type, color)
        if rs_scatter_key in rs_scatter_dict:
            data = query_dict["clr_{}_obs".format(color)]
            scatter = np.random.normal(scale=rs_scatter_dict[key],
                                       size =data.size)
            query_dict["clr_{}_obs".format(color)] = data + scatter
    return data_dict


def modify_array_with_rs_scatter(data_dict, data_type, color,
                                 rs_scatter_dict):
    assert data_type == "query" or data_type == "tree", "Data type must be either \"query\" or \"tree\". Given data_type is {}".format(data_type)
    data = data_dict['clr_{}_obs'.format(color)]
    rs_scatter_key = "red_sequence_scatter_{}_{}".format(data_type, color)
    if rs_scatter_key in rs_scatter_dict:
        print("modifying {} data: color {} ".format(data_type, color))
        scatter =  np.random.normal(scale=rs_scatter_dict[rs_scatter_key],
                                    size =data.size)
        return data+scatter
    else:
        return data

                     
copy_avoids = ('x','y','z','vx','vy','vz', 'peculiarVelocity','galaxyID','redshift',
               'redshiftHubble','placementType','isCentral','hostIndex', 
               'blackHoleAccretionRate','blackHoleMass', 'step','infallHaloMass','infallHaloTag')
#TODO re-allow nitrogen contamination
copy_avoids_ptrn = ('hostHalo','magnitude','ageStatistics','Radius','Axis','Ellipticity','positionAngle','total', 'ContinuumLuminosity', 'contam_nitrogenII6584', 'Sersic', 'morphology', 'contam_nitrogen')
no_slope_var = ('x','y','z','vx','vy','vz', 'peculiarVelocity','galaxyID','redshift','redshiftHubble','inclination','positionAngle')
no_slope_ptrn  =('morphology','hostHalo','infall')

def to_copy(key, short, supershort):
    if short:
        if "SED" in key or "other" in key or "Lines" in key:
            print("\tnot copied: short var cut")
            return False
    if supershort:
        if "SDSS" not in key and "total" not in key and ":rest" not in key and "MassStellar" not in key and "infallIndex" != key and "inclination" not in key:
            print("\tnot copied: supershort var cut")
            return False
    if any([ ca == key for ca in copy_avoids]) or any([ cap in key for cap in copy_avoids_ptrn ]):
        print("\tnot copied: by explicit listing")
        return False
    return True


# Keys that have their luminosity adjusted
luminosity_factors_keys = ['Luminosities', 'Luminosity']

_cached_column = {}

def get_column_interpolation_dust_raw(key, h_in_gp1, h_in_gp2, index,
                                      mask1, mask2, step1_a, step2_a,
                                      target_a, dust_factors,
                                      kdtree_index=None,
                                      luminosity_factors = None, cache
                                      = False, snapshot = False):
    """This function returns the interpolated quantity between two
    timesteps, from step1 to step2. Some galaxies are masked out: Any
    galaxy that doesn't pass the mask in step1 (mask1), any galaxy
    that doesn't a decendent in step2, or any galaxy that whose
    descendent doesn't pass the step2 mask (mask2).

    """
    print("\tLoading key: {}".format(key))
    # if luminosity_factors is None:
    #     print("\t\tluminosity factors is none")
    #print("dust_factors: ", dust_factors)
    t1 = time.time()
    step_del_a = step2_a - step1_a
    target_del_a = target_a - step1_a
    ##=========DEBUG==========
    # print("step del_a {:.3f} - {:.3f}  = {:.3f}".format(step2_a, step1_a, step_del_a))
    # print("target_del_a {:.3f} - {:.3f} = {:.3f}".format(target_a, step1_a, target_del_a))
    ##=========DEBUG==========
    # The masking all galaxies that fail galmatcher's requirements at
    # step1, galaxies that don't have a descndent, or if the
    # descendent galaxy at step2 doesn't pass galmatcher requirements.
    # If index is set None, we aren't doing interpolation at all. We
    # are just using
    if not snapshot:
        mask_tot = mask1 & (index != -1) & mask2[index]
        if (key in no_slope_var) or any(ptrn in key for ptrn in no_slope_ptrn):
            #print('\t\tno interpolation')
            data = h_in_gp1[key].value[mask_tot]
            if kdtree_index is None:
                val_out = data
            else:
                val_out = data[kdtree_index]
        elif ":dustAtlas" in key:
            #print('\t\tinterpolation with dust')
            key_no_dust = key.replace(":dustAtlas","")
            val1_no_dust = h_in_gp1[key_no_dust].value[mask_tot]
            val1_dust = h_in_gp1[key].value[mask_tot]
            if index is not None:
                val2_no_dust = h_in_gp2[key].value[index][mask_tot]
            else:
                val2_no_dust = val1_no_dust
            # val1_no_dust_lg = np.log(val1_no_dust)
            # val1_dust_lg = np.log(val1_dust)
            # val2_no_dust_lg = np.log(val2_no_dust)
            dust_effect = val1_dust/val1_no_dust
            dust_effect[val1_no_dust == 0] = 1
            slope = (val2_no_dust - val1_no_dust)/step_del_a
            slope[step_del_a ==0] =0
            # slope_mag = (val2_no_dust_lg  - val1_no_dust_lg)/step_del_a
            # slope_mag[step_del_a == 0] = 0
            ##=======DEBUG=======
            # def print_vals(label, data):
            #     print("\t\t{} below/zero/size: {}/{}/{}".format(label, np.sum(data<0), np.sum(data==0), data.size))
            # print_vals("val1_no_dust", val1_no_dust)
            # print_vals("val2_no_dust", val2_no_dust)
            # print_vals("val1_dust", val1_dust)
            ##=======DEBUG=======
            if kdtree_index is None:
                tot_dust_effect = dust_effect**dust_factors
                slct = dust_effect > 1.0
                tot_dust_effect[slct] = dust_effect[slct]
                val_out = (val1_no_dust + slope*target_del_a)*tot_dust_effect
                #val_out = np.exp((val1_no_dust_lg + slope_mag*target_del_a)*tot_dust_effect)
            else:
                tot_dust_effect = (dust_effect[kdtree_index]**dust_factors)
                slct = dust_effect[kdtree_index] > 1.0
                tot_dust_effect[slct] = dust_effect[kdtree_index][slct]
                val_out = (val1_no_dust[kdtree_index] + slope[kdtree_index]*target_del_a)*tot_dust_effect
                #val_out = np.exp((val1_no_dust_lg[kdtree_index] + slope_mag[kdtree_index]*target_del_a)*tot_dust_effect)
        else:
            #print('\t\tinerpolation without dust')
            val1_data = h_in_gp1[key].value[mask_tot]
            val2_data = h_in_gp2[key].value[index][mask_tot]
            # val1_data_lg = np.log(val1_data)
            # val2_data_lg = np.log(val2_data)
            slope = (val2_data - val1_data)/step_del_a
            slope[step_del_a==0]=0
            # slope_mag = (val1_data_lg-val2_data_lg)/step_del_a
            # slope_mag[step_del_a==0]=0
            if kdtree_index is None:
                val_out = val1_data + slope*target_del_a
                #val_out = np.log(val1_data_lg + slope_mag*target_del_a)
            else:
                val_out = val1_data[kdtree_index] + slope[kdtree_index]*target_del_a
                #val_out = np.log(val1_data_lg[kdtree_index] + slope_mag[kdtree_index]*target_del_a)
        #print('\t\t',val_out.dtype)
        
    # If running on snapshot, we don't need to interpolate
    else: 
        mask_tot = mask1
        val1_data = h_in_gp1[key].value[mask_tot]
        # reorder the data if it's a post-matchup index
        if kdtree_index is None:
            val_out = val1_data
        else:
            val_out = val1_data[kdtree_index]

    if not(luminosity_factors is None):
        if(any(l in key for l in luminosity_factors_keys)):
            #print("\t\tluminosity adjusted")
            val_out = val_out*luminosity_factors
        elif('Luminosities' in key or 'Luminosity' in key):
            #print("\t\tluminosity adjusted 2")
            val_out = val_out*luminosity_factors
        else:
            pass
            #print("\t\tluminosity untouched")
    if np.sum(~np.isfinite(val_out))!=0:
        print(key, "has a non-fininte value")
        print("{:.2e} {:.2e}".format(np.sum(~np.isfinite(val_out)), val_out.size))
        if ":dustAtlas" in key:
            print(np.sum(~np.isfinite(val1_no_dust)))
            print(np.sum(~np.isfinite(dust_effect)))
            slct = ~np.isfinite(dust_effect)
            print(val1_no_dust[slct])
            print(val1_dust[slct])
        print(np.sum(~np.isfinite(slope_mag)))
        print(np.sum(~np.isfinite(target_del_a)))
        if "emissionLines" in key:
            print("overwriting non-finite values with 0")
            val_out[~np.isfinite(val_out)]=0.0
        else:
            raise
    #print("\t\toutput size: {:.2e}".format(val_out.size))
    print("\t\t mask size:{:.1e}/{:.1e} data size:{:.1e} read + format time: {}".format(np.sum(mask_tot), mask_tot.size, val_out.size, time.time()-t1))
    ##=======DEBUG======
    # print("\t\t non-finite: {}/{}/{}".format(np.sum(~np.isfinite(val_out)), np.sum(val_out<0), val_out.size))
    # print("\t\t below/zero/size: {}/{}/{}".format(np.sum(val_out<0), np.sum(val_out==0), val_out.size))
    ##=======DEBUG======
    return val_out


def copy_columns_interpolation_dust_raw(input_fname, output_fname,
                                        kdtree_index, step1, step2,
                                        step1_a, step2_a, mask1, mask2, 
                                        index_2to1, lc_a, 
                                        verbose = False, 
                                        short = False, supershort = False, 
                                        step = -1, dust_factors = 1.0,
                                        luminosity_factors = None,
                                        library_index = None,
                                        node_index = None,
                                        snapshot = False):
    print("===================================")
    print("copy columns interpolation dust raw")
    # lc_a = 1.0/(1.0+lc_redshift)
    # input_a = 1.0/(1.0 + input_redshift)
    del_a = lc_a-step1_a
    dtk.ensure_dir(output_fname)
    h_out = h5py.File(output_fname,'w')
    h_out_gp = h_out.create_group('galaxyProperties')
    h_out_gp['matchUp/dustFactor'] = dust_factors
    if luminosity_factors is not None:
        h_out_gp['matchUp/luminosityFactor'] = luminosity_factors
    if library_index is not None:
        h_out_gp['matchUp/libraryIndex'] = library_index
    if node_index is not None:
        h_out_gp['matchUp/GalacticusNodeIndex'] = node_index
    h_in_gp1 = h5py.File(input_fname.replace("${step}",str(step1)),'r')['galaxyProperties']
    h_in_gp2 = h5py.File(input_fname.replace("${step}",str(step2)),'r')['galaxyProperties']

    keys = get_keys(h_in_gp1)
    max_float = np.finfo(np.float32).max #The max float size
    for i in range(0,len(keys)):
        t1 = time.time()
        key = keys[i]
        if verbose:
            print('{}/{} [{}] {}'.format(i,len(keys),step, key))
        if not to_copy(key, short, supershort):
            continue
        new_data = get_column_interpolation_dust_raw(
            key, h_in_gp1, h_in_gp2, index_2to1, mask1, mask2, step1_a, step2_a, lc_a, dust_factors, 
            kdtree_index = kdtree_index, luminosity_factors = luminosity_factors, snapshot=snapshot)
        slct_finite = np.isfinite(new_data)
        #If the data is a double, record it as a float to save on disk space
        if(new_data.dtype == np.float64 and np.sum(new_data[slct_finite]>max_float) == 0):
            h_out_gp[key]= new_data.astype(np.float32)
        else:
            h_out_gp[key] = new_data
        print("\t\tDone writing. read+format+write: {}".format(time.time()-t1))
    return


def copy_columns_interpolation_dust_raw_healpix(input_fname, output_fname,
                                                kdtree_index, step1, step2,
                                                step1_a, step2_a, mask1, mask2, 
                                                index_2to1, lc_a, 
                                                healpix_pixels, lc_healpix,
                                                verbose = False, 
                                                short = False, supershort = False, 
                                                step = -1, dust_factors = 1.0,
                                                luminosity_factors = None,
                                                library_index = None,
                                                node_index = None, 
                                                snapshot= False):
    print("===================================")
    print("copy columns interpolation dust raw")
    # lc_a = 1.0/(1.0+lc_redshift)
    # input_a = 1.0/(1.0 + input_redshift)
    del_a = lc_a-step1_a
    #print("del_a: ", del_a)
    h_out_gps = {}
    h_out_gps_slct = {}
    for healpix_pixel in healpix_pixels:
        hp_fname = output_fname.replace("${healpix}", str(healpix_pixel))
        dtk.ensure_dir(hp_fname)
        h_out = h5py.File(hp_fname,'w')
        h_out_gps[healpix_pixel] = h_out.create_group('galaxyProperties')
        slct = lc_healpix == healpix_pixel
        h_out_gps[healpix_pixel]['matchUp/dustFactor'] = dust_factors[slct]
        h_out_gps[healpix_pixel]['matchUp/luminosityFactor'] = luminosity_factors[slct]
        h_out_gps[healpix_pixel]['matchUp/libraryIndex'] = library_index[slct]
        h_out_gps[healpix_pixel]['matchUp/GalacticusNodeIndex'] = node_index[slct]
        h_out_gps_slct[healpix_pixel] = slct
    h_in_gp1 = h5py.File(input_fname.replace("${step}",str(step1)),'r')['galaxyProperties']
    h_in_gp2 = h5py.File(input_fname.replace("${step}",str(step2)),'r')['galaxyProperties']

    keys = get_keys(h_in_gp1)
    max_float = np.finfo(np.float32).max #The max float size
    for i in range(0,len(keys)):
        t1 = time.time()
        key = keys[i]
        if verbose:
            print('{}/{} [{}] {}'.format(i,len(keys),step, key))
        if not to_copy(key, short, supershort):
            continue
        new_data = get_column_interpolation_dust_raw(
            key, h_in_gp1, h_in_gp2, index_2to1, mask1, mask2, step1_a, step2_a, lc_a, dust_factors, 
            kdtree_index = kdtree_index, luminosity_factors = luminosity_factors, snapshot = snapshot)
        slct_finite = np.isfinite(new_data)
        #If the data is a double, record it as a float to save on disk space
        if(new_data.dtype == np.float64 and np.sum(new_data[slct_finite]>max_float) == 0):
            new_data= new_data.astype(np.float32)
        if 'LineLuminosity' in key:
            key = key.replace(':rest', '')
        for healpix_pixel in healpix_pixels:
            h_out_gps[healpix_pixel][key] = new_data[h_out_gps_slct[healpix_pixel]]
        print("\t\tDone writing. read+format+write: {}".format(time.time()-t1))
    return


def overwrite_columns(input_fname, output_fname, ignore_mstar = False,
                      verbose=False, cut_small_galaxies_mass = None,
                      internal_step=None, fake_lensing=False,
                      healpix=False, step = None, healpix_shear_file =
                      None, no_shear_steps=None,
                      snapshot = False,
                      snapshot_redshift = None):
    t1 = time.time()
    if verbose:
        print("Overwriting columns.")
        #sdss = Table.read(input_fname,path='data')
    if snapshot:
        assert snapshot_redshift is not None, "Snapshot redshift must be specified in snapshot mode"
    h_out = h5py.File(output_fname, 'a')
    h_out_gp = h_out['galaxyProperties']
    if snapshot:
        h_in = h5py.File(input_fname,'r')['galaxyProperties']
    elif internal_step is None:
        h_in = h5py.File(input_fname,'r')
    else:
        h_in = h5py.File(input_fname,'r')[str(internal_step)]
    # if the input file has no galaxies, it doesn't have any columns
    step_has_data = "obs_sm" in h_in
    if step_has_data:
        sm = h_in['obs_sm'].value
    else:
        sm = np.zeros(0, dtype=float)
    if cut_small_galaxies_mass is None:
        mask = np.ones(sm.size, dtype=bool)
    else: 
        mask = np.log10(sm) > cut_small_galaxies_mass
    #redshift = np.ones(sdss['x'].quantity.size)*0.1
    t2 = time.time()
    if verbose:
        print("\t done reading in data", t2-t1)
    #xyz,v(xyz)
    if step_has_data:
        x = h_in['x'].value[mask]
        y = h_in['y'].value[mask]
        z = h_in['z'].value[mask]
        vx = h_in['vx'].value[mask]
        vy = h_in['vy'].value[mask]
        vz = h_in['vz'].value[mask]
        if snapshot:
            redshift = np.ones(x.size)*snapshot_redshift
        else:
            redshift  =h_in['redshift'].value[mask]
            size = h_in['x'].size
        if not snapshot:
            h_out_gp['lightcone_rotation'] = h_in['lightcone_rotation'].value[mask]
            h_out_gp['lightcone_replication'] = h_in['lightcone_replication'].value[mask]
    else:
        x = np.zeros(0, dtype=float)
        y = np.zeros(0, dtype=float)
        z = np.zeros(0, dtype=float)
        vx = np.zeros(0, dtype=float)
        vy = np.zeros(0, dtype=float)
        vz = np.zeros(0, dtype=float)
        size =  0
        redshift  =np.zeros(0, dtype=float)
        if not snapshot:
            h_out_gp['lightcone_rotation'] = np.zeros(0, dtype=int)
            h_out_gp['lightcone_replication'] = np.zeros(0, dtype=int)
    print('step: ', step)
    assert step is not None, "Step is not specified"
    if not snapshot:
        h_out_gp['step']=np.ones(size,dtype='i4')*step
    h_out_gp['x']=x
    h_out_gp['y']=y
    h_out_gp['z']=z
    h_out_gp['vx']=vx
    h_out_gp['vy']=vy
    h_out_gp['vz']=vz
    keys = get_keys(h_out_gp)
    for key in keys:
        if "spheroid" in key:
            spheroid_key = key
            disk_key = key.replace('spheroid','disk')
            total_key = key.replace('spheroid','total')
            h_out_gp[total_key] = np.array(h_out_gp[disk_key].value + h_out_gp[spheroid_key].value, dtype='f4')
    if ignore_mstar:
        if step_has_data:
            print("Ignoring M* in stellar mass assignment!")
            # m*_delta = M*_new/M*_old
            mstar_delta = h_in['obs_sm'].value[mask]/h_out_gp['totalMassStellar'].value
            h_out_gp['totalMassStellar'][:] = h_out_gp['totalMassStellar'].value*mstar_delta
            h_out_gp['diskMassStellar'][:] = h_out_gp['diskMassStellar'].value*mstar_delta
            h_out_gp['spheroidMassStellar'][:] = h_out_gp['spheroidMassStellar'].value*mstar_delta
        else:
            # No need to modify data on disk if the data sets are empty
            pass 

    t3 = time.time()
    #peculiar velocity
    if not snapshot:
        _,z_obs,v_pec,_,_,_,_ = pecZ(x,y,z,vx,vy,vz,redshift)
        h_out_gp['peculiarVelocity'] = np.array(v_pec, dtype='f4')
    #obs mag
    #Calculate the oringal redshift 
    stepz = dtk.StepZ(200,0,500)
    # Precompute redshfit to luminosity distance relationship
    zs = np.linspace(0,3.5,1000)
    z_to_dl = interp1d(zs,cosmo.luminosity_distance(zs))
    dl = z_to_dl(redshift)
    adjust_mag = -2.5*np.log10(1.0+redshift)+5*np.log10(dl)+25.0
    t4 = time.time()
    keys = get_keys(h_out_gp)
    for key in keys:
        # Calculating new observer frame magnitudes
        if("totalLuminositiesStellar" in key and  ":observed" in key and ("SDSS" in key or "LSST" in key)):
            new_key = key.replace("totalLuminositiesStellar",'magnitude',1)
            #print("making: "+new_key+" from "+key)
            h_out_gp[new_key]=np.array(adjust_mag -2.5*np.log10(h_out_gp[key].value), dtype='f4')
        # Calculating new rest frame magnitudes
        if("totalLuminositiesStellar" in key and  ":rest" in key and ("SDSS" in key or "LSST" in key)):
            new_key = key.replace("totalLuminositiesStellar","magnitude",1)
            #print("making: "+new_key+" from "+key)
            h_out_gp[new_key]=np.array(-2.5*np.log10(h_out_gp[key].value), dtype='f4')

    t5 = time.time()
    if verbose:
        print("\t done rewriting mags",t5-t4)
    #redshift
    if snapshot:
        h_out_gp['redshift'] = redshift
    else:
        h_out_gp['redshift'] = z_obs
    h_out_gp['redshiftHubble'] = redshift
    if not snapshot:
        if step_has_data:
            h_out_gp['ra_true'] = h_in['ra'].value[mask]
            h_out_gp['dec_true'] = h_in['dec'].value[mask]
        else:
            h_out_gp['ra_true'] = np.zeros(0, dtype=np.float)
            h_out_gp['dec_true'] = np.zeros(0, dtype=np.float)
        
    # Skip shears for this step if it's in the list 
    if no_shear_steps is not None:
        fake_lensing_step = (step in no_shear_steps)
        if (step in no_shear_steps):
            print("==== no shear step ====", step)
            pritn("\t step {} is list as in no_shear_steps".format(step), no_shear_steps)
    
    if fake_lensing or fake_lensing_step:
        print("\tfake shears")
        if step_has_data:
            size = h_in['x'].size
        else:
            size = 0
        if not snapshot:
            if step_has_data:
                h_out_gp['ra'] = h_in['ra'].value[mask]
                h_out_gp['dec'] = h_in['dec'].value[mask]
            else:
                h_out_gp['ra'] = np.zeros(0,dtype=float)
                h_out_gp['dec'] =np.zeros(0,dtype=float)
        h_out_gp['shear1'] = np.zeros(size, dtype='f4')
        h_out_gp['shear2'] = np.zeros(size, dtype='f4')
        h_out_gp['magnification'] = np.ones(size, dtype='f4')
        h_out_gp['convergence'] = np.zeros(size, dtype='f4')
    elif healpix:
        print("\thealpix shears")
        h_shear = h5py.File(healpix_shear_file, 'r')[str(step)]
        shear_id = h_shear['galaxy_id'].value
        if step_has_data:
            print("\t\tassigning shears ")
            base_id = h_in['galaxy_id'].value[mask]
            srt = np.argsort(shear_id)
            shear_indx = dtk.search_sorted(shear_id, base_id, sorter=srt)
            assert np.sum(shear_indx==-1) == 0, "a baseDC2 galaxy wasn't found in shear catalog?"
            h_out_gp['ra'] = h_shear['ra_lensed'].value[shear_indx]
            h_out_gp['dec'] = h_shear['dec_lensed'].value[shear_indx]
            s1 = h_shear['shear_1'].value[shear_indx]
            s2 = h_shear['shear_2'].value[shear_indx]
            k = h_shear['conv'].value[shear_indx]
            u = 1.0/((1.0-k)**2 - s1**2 -s2**2)
            h_out_gp['shear1'] = s1
            h_out_gp['shear2'] = s2
            h_out_gp['magnification'] = u
            h_out_gp['convergence'] = k
            # h_out_gp['ra'] = h_shear['ra_lensed'].value[mask]
            # h_out_gp['dec'] = h_shear['dec_lensed'].value[mask]
            # s1 = h_shear['shear_1'].value[mask]
            # s2 = h_shear['shear_2'].value[mask]
            # k = h_shear['conv'].value[mask]
        else:
            print("\t\tno data in this step, setting it all to zeros")
            h_out_gp['shear1']        = np.zeros(0, dtype=np.float)
            h_out_gp['shear2']        = np.zeros(0, dtype=np.float)
            h_out_gp['magnification'] = np.zeros(0, dtype=np.float)
            h_out_gp['convergence']   = np.zeros(0, dtype=np.float)

    else:
        print('\tprotoDC2 shear style')
        #No protection against empty step. This is only a feature of CosmoDC2
        h_out_gp['ra'] = h_in['ra_lensed'].value[mask]
        h_out_gp['dec'] = h_in['dec_lensed'].value[mask]
        h_out_gp['shear1'] = h_in['shear1'].value[mask]
        h_out_gp['shear2'] = h_in['shear2'].value[mask]
        h_out_gp['magnification'] = h_in['magnification'].value[mask]
        h_out_gp['convergence'] = h_in['convergence'].value[mask]

    if healpix:
        if step_has_data:
            h_out_gp['galaxyID'] = h_in['galaxy_id'].value[mask]
        else:
            h_out_gp['galaxyID'] = np.zeros(0, dtype=np.int64)

    if step_has_data:
        central = h_in['upid'].value[mask] == -1
        h_out_gp['isCentral'] = central
        if snapshot:
            pass
            h_out_gp['hostHaloTag'] = h_in['target_halo_id'].value[mask]
        else:
            h_out_gp['hostHaloTag'] = h_in['target_halo_fof_halo_id'].value[mask]
        h_out_gp['uniqueHaloID'] = h_in['target_halo_id'].value[mask]
        h_out_gp['hostHaloMass'] = h_in['target_halo_mass'].value[mask]
        unq, indx, cnt = np.unique(h_out_gp['infallIndex'].value, return_inverse=True, return_counts = True)
        h_out_gp['matchUp/NumberSelected'] = cnt[indx]
        central_2 = (h_in['host_centric_x'].value[mask] ==0) & (h_in['host_centric_y'].value[mask] ==0) & (h_in['host_centric_z'].value[mask] == 0)
        assert np.sum(central_2 != central) == 0, "double centrals? centrals by upid {}, centrals by halo centric position {}".format(np.sum(central), np.sum(central_2))
    else:
        h_out_gp['isCentral'] = np.zeros(0, dtype=bool)
        h_out_gp['hostHaloTag'] = np.zeros(0, dtype=np.int64)
        h_out_gp['uniqueHaloID'] = np.zeros(0, dtype=np.int64)
        h_out_gp['hostHaloMass'] = np.zeros(0, dtype=np.float)
        h_out_gp['matchUp/NumberSelected'] = np.zeros(0,dtype=np.int)
    tf = time.time()
    if verbose:
        print("\tDone overwrite columns", tf-t1)


def swap(slct, x1, x2):
    xa = x1[slct]
    x1[slct] = x2[slct]
    x2[slct]=xa


def rotate_host_halo(rot, x,y,z):
    """"
    From the documentation:
    
    // 0 = no rotation
    // 1 = swap(x, y) rotation applied
    // 2 = swap(y, z) rotation applied
    // 3 = swap(x, y) then swap(y, z) rotations applied 
    // 4 = swap(z, x) rotation applied
    // 5 = swap(x, y) then swap(z, x) rotations applied 

    // There are also two other possibilities:
    // 6 = swap(y, z) then swap(z, x) rotations applied 
    // 7 = swap(x, y), then swap(y, z), then swap(z, x) rotations applied 
    // which are equivalent to rotations 3 and 2 above, respectively

    """
    slct1 = rot == 1
    swap(slct1, x, y)

    slct2 = (rot == 2) | (rot == 7)
    swap(slct2, y, z)

    slct3 = (rot == 3) | (rot == 6)
    swap(slct3, x, y )
    swap(slct3, y, z)

    slct4 = rot == 4
    swap(slct4, z, x)

    slct5 = rot == 5  
    swap(slct5, x, y)
    swap(slct5, z, x)
    return


def overwrite_host_halo(output_fname, sod_loc, halo_shape_loc,
                        halo_shape_red_step_loc, verbose=False, snapshot = False):
    hgroup = h5py.File(output_fname,'r+')['galaxyProperties']

    halo_tag = hgroup['hostHaloTag'].value
    size = halo_tag.size
    if not snapshot:
        halo_rot = hgroup['lightcone_rotation'].value
    # fof_tag =  dtk.read_gio(fof_loc,'fof_halo_tag')
    # fof_mass = dtk.read_gio(fof_loc,'fof_mass')
    # fof_srt = np.argsort(fof_tag)
    sod_mass = -1*np.ones(halo_tag.size)
    if os.path.isfile(sod_loc):
        sod_cat_tag = dtk.gio_read(sod_loc,'fof_halo_tag')
        sod_cat_mass = dtk.gio_read(sod_loc,'sod_halo_mass')
        sod_cat_srt = np.argsort(sod_cat_tag)

        indx = dtk.search_sorted(sod_cat_mass,halo_tag,sorter=sod_cat_srt)
        slct = indx != -1
        sod_mass[slct] = sod_cat_mass[indx[slct]]
        hgroup['hostHaloSODMass']=sod_mass

    print("Num of galaxies: ", halo_tag.size)
    eg_cat_eg1 = np.zeros(size,dtype='f4')
    eg_cat_eg2 = np.zeros(size,dtype='f4')
    eg_cat_eg3 = np.zeros(size,dtype='f4')
    eg_cat_eg1_x =np.zeros(size,dtype='f4')
    eg_cat_eg1_y =np.zeros(size,dtype='f4')
    eg_cat_eg1_z =np.zeros(size,dtype='f4')
    eg_cat_eg2_x =np.zeros(size,dtype='f4')
    eg_cat_eg2_y =np.zeros(size,dtype='f4')
    eg_cat_eg2_z =np.zeros(size,dtype='f4')
    eg_cat_eg3_x = np.zeros(size,dtype='f4')
    eg_cat_eg3_y =np.zeros(size,dtype='f4')
    eg_cat_eg3_z =np.zeros(size,dtype='f4')
    has_halo_shape_files = os.path.isfile(halo_shape_loc) and os.path.isfile(halo_shape_red_step_loc)
    if has_halo_shape_files:
        eg_cat_htag = dtk.gio_read(halo_shape_loc,'halo_id')
        srt = np.argsort(eg_cat_htag)
        indx = dtk.search_sorted(eg_cat_htag,halo_tag,sorter=srt)
        slct_indx = indx != -1
        indx_slct = indx[slct_indx]
        print("num selected: ",np.sum(slct_indx))
        eg_cat_eg1[slct_indx] = dtk.gio_read(halo_shape_loc,'eval1')[indx_slct]
        eg_cat_eg2[slct_indx] = dtk.gio_read(halo_shape_loc,'eval2')[indx_slct]
        eg_cat_eg3[slct_indx] = dtk.gio_read(halo_shape_loc,'eval3')[indx_slct]
        eg_cat_eg1_x[slct_indx] = dtk.gio_read(halo_shape_loc,'evec1x')[indx_slct]
        eg_cat_eg1_y[slct_indx] = dtk.gio_read(halo_shape_loc,'evec1y')[indx_slct]
        eg_cat_eg1_z[slct_indx] = dtk.gio_read(halo_shape_loc,'evec1z')[indx_slct]
        eg_cat_eg2_x[slct_indx] = dtk.gio_read(halo_shape_loc,'evec2x')[indx_slct]
        eg_cat_eg2_y[slct_indx] = dtk.gio_read(halo_shape_loc,'evec2y')[indx_slct]
        eg_cat_eg2_z[slct_indx] = dtk.gio_read(halo_shape_loc,'evec2z')[indx_slct]
        eg_cat_eg3_x[slct_indx] = dtk.gio_read(halo_shape_loc,'evec3x')[indx_slct]
        eg_cat_eg3_y[slct_indx] = dtk.gio_read(halo_shape_loc,'evec3y')[indx_slct]
        eg_cat_eg3_z[slct_indx] = dtk.gio_read(halo_shape_loc,'evec3z')[indx_slct]
        if not snapshot:
            rotate_host_halo(halo_rot, eg_cat_eg1_x, eg_cat_eg1_y, eg_cat_eg1_z)
            rotate_host_halo(halo_rot, eg_cat_eg2_x, eg_cat_eg2_y, eg_cat_eg2_z)
            rotate_host_halo(halo_rot, eg_cat_eg3_x, eg_cat_eg3_y, eg_cat_eg3_z)
        hgroup['hostHaloEigenValue1'] = eg_cat_eg1
        hgroup['hostHaloEigenValue2'] = eg_cat_eg2
        hgroup['hostHaloEigenValue3'] = eg_cat_eg3
        hgroup['hostHaloEigenVector1X'] = eg_cat_eg1_x
        hgroup['hostHaloEigenVector1Y'] = eg_cat_eg1_y
        hgroup['hostHaloEigenVector1Z'] = eg_cat_eg1_z
        hgroup['hostHaloEigenVector2X'] = eg_cat_eg2_x
        hgroup['hostHaloEigenVector2Y'] = eg_cat_eg2_y
        hgroup['hostHaloEigenVector2Z'] = eg_cat_eg2_z
        hgroup['hostHaloEigenVector3X'] = eg_cat_eg3_x
        hgroup['hostHaloEigenVector3Y'] = eg_cat_eg3_y
        hgroup['hostHaloEigenVector3Z'] = eg_cat_eg3_z
    if has_halo_shape_files:
        eg_cat_htag = dtk.gio_read(halo_shape_red_step_loc,'halo_id')[indx_slct]
        eg_cat_eg1[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'eval1')[indx_slct]
        eg_cat_eg2[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'eval2')[indx_slct]
        eg_cat_eg3[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'eval3')[indx_slct]
        eg_cat_eg1_x[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec1x')[indx_slct]
        eg_cat_eg1_y[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec1y')[indx_slct]
        eg_cat_eg1_z[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec1z')[indx_slct]
        eg_cat_eg2_x[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec2x')[indx_slct]
        eg_cat_eg2_y[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec2y')[indx_slct]
        eg_cat_eg2_z[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec2z')[indx_slct]
        eg_cat_eg3_x[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec3x')[indx_slct]
        eg_cat_eg3_y[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec3y')[indx_slct]
        eg_cat_eg3_z[slct_indx] = dtk.gio_read(halo_shape_red_step_loc,'evec3z')[indx_slct]
        if not snapshot:
            rotate_host_halo(halo_rot, eg_cat_eg1_x, eg_cat_eg1_y, eg_cat_eg1_z)
            rotate_host_halo(halo_rot, eg_cat_eg2_x, eg_cat_eg2_y, eg_cat_eg2_z)
            rotate_host_halo(halo_rot, eg_cat_eg3_x, eg_cat_eg3_y, eg_cat_eg3_z)
        hgroup['hostHaloEigenValueReduced1'] = eg_cat_eg1
        hgroup['hostHaloEigenValueReduced2'] = eg_cat_eg2
        hgroup['hostHaloEigenValueReduced3'] = eg_cat_eg3
        hgroup['hostHaloEigenVectorReduced1X'] = eg_cat_eg1_x
        hgroup['hostHaloEigenVectorReduced1Y'] = eg_cat_eg1_y
        hgroup['hostHaloEigenVectorReduced1Z'] = eg_cat_eg1_z
        hgroup['hostHaloEigenVectorReduced2X'] = eg_cat_eg2_x
        hgroup['hostHaloEigenVectorReduced2Y'] = eg_cat_eg2_y
        hgroup['hostHaloEigenVectorReduced2Z'] = eg_cat_eg2_z
        hgroup['hostHaloEigenVectorReduced3X'] = eg_cat_eg3_x
        hgroup['hostHaloEigenVectorReduced3Y'] = eg_cat_eg3_y
        hgroup['hostHaloEigenVectorReduced3Z'] = eg_cat_eg3_z
    return

   
def add_native_umachine(output_fname, umachine_native,
                        cut_small_galaxies_mass = None,
                        internal_step=None, snapshot=False):
    t1 = time.time()
    if snapshot:
        h_in = h5py.File(umachine_native, 'r')['galaxyProperties']
    elif internal_step is None:
        h_in = h5py.File(umachine_native,'r')
    else:
        h_in = h5py.File(umachine_native,'r')[str(internal_step)]
    hgroup = h5py.File(output_fname, 'r+')['galaxyProperties']
    if cut_small_galaxies_mass is None:
        for key in h_in.keys():
            hgroup['baseDC2/'+key] = h_in[key].value
    else:
        sm = h_in['obs_sm'].value # in linear units
        slct = sm > 10**cut_small_galaxies_mass #convert cut_small.. from log to linear
        for key in h_in.keys():
            hgroup['baseDC2/'+key] = h_in[key].value[slct]
    print("done addign baseDC2 quantities. time: {:.2f}".format(time.time()-t1))
    return


def add_blackhole_quantities(output_fname, redshift, percentile_sfr):
    hgroup = h5py.File(output_fname,'r+')['galaxyProperties']
    print(hgroup.keys())
    bhm = monte_carlo_black_hole_mass(hgroup['spheroidMassStellar'].value)
    eddington_ratio, bh_acc_rate = monte_carlo_bh_acc_rate(redshift, bhm, percentile_sfr)
    hgroup['blackHoleMass'] = bhm
    hgroup['blackHoleAccretionRate'] = bh_acc_rate*1e9
    hgroup['blackHoleEddingtonRatio'] = eddington_ratio


def add_size_quantities(output_fname):
    hgroup = h5py.File(output_fname,'r+')['galaxyProperties']
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:rest'].value
    redshift = hgroup['redshift'].value
    if len(redshift) > 0:
        arcsec_per_kpc = interp1d(redshift,cosmo.arcsec_per_kpc_proper(redshift).value)
        f = arcsec_per_kpc(redshift)
    else:
        f = np.zeros(0, dtype=np.float)
    size_disk = mc_size_vs_luminosity_late_type(mag_r, redshift)
    size_sphere = mc_size_vs_luminosity_early_type(mag_r, redshift)

    hgroup['morphology/spheroidHalfLightRadius'] =       size_sphere
    hgroup['morphology/spheroidHalfLightRadiusArcsec'] = size_sphere*f
    hgroup['morphology/diskHalfLightRadius'] =       size_disk
    hgroup['morphology/diskHalfLightRadiusArcsec'] = size_disk*f


def erase_ellipticity_quantities(output_fname):
    print(output_fname)
    def erase_if_has(hfile, output_fname):
        if 'galaxyProperties/'+output_fname in hfile:
            del hfile['galaxyProperties/'+output_fname]
    hfile = h5py.File(output_fname,'r+')
    erase_if_has(hfile, 'morphology/spheroidAxisRatio')
    erase_if_has(hfile, 'morphology/spheroidAxisRatio')
    erase_if_has(hfile, 'morphology/spheroidMajorAxisArcsec')
    erase_if_has(hfile, 'morphology/spheroidMinorAxisArcsec')
    erase_if_has(hfile, 'morphology/spheroidEllipticity') 
    erase_if_has(hfile, 'morphology/spheroidEllipticity1')
    erase_if_has(hfile, 'morphology/spheroidEllipticity2')
    erase_if_has(hfile, 'morphology/diskAxisRatio')
    erase_if_has(hfile, 'morphology/diskMajorAxisArcsec') 
    erase_if_has(hfile, 'morphology/diskMinorAxisArcsec') 
    erase_if_has(hfile, 'morphology/diskEllipticity') 
    erase_if_has(hfile, 'morphology/diskEllipticity1')
    erase_if_has(hfile, 'morphology/diskEllipticity2')
    erase_if_has(hfile, 'morphology/totalEllipticity')  
    erase_if_has(hfile, 'morphology/totalAxisRatio')    
    erase_if_has(hfile, 'morphology/totalEllipticity1') 
    erase_if_has(hfile, 'morphology/totalEllipticity2') 
    erase_if_has(hfile, 'morphology/positionAngle') 


def add_ellipticity_quantities(output_fname, verbose = False):
    if verbose:
        print("\tadding ellipticity")
    def gal_zoo_dist(x):
        val = np.zeros_like(x)
        a = 2
        slct = x<0.2
        val[slct] = 0
        
        slct = (0.1<=x) & (x<0.7)
        val[slct] = np.tanh((x[slct]-.3)*np.pi*a) - np.tanh((-0.2)*np.pi*a)
        
        slct = (0.7<=x) & (x<1.0)
        val[slct] = np.tanh(-(x[slct]-.95)*np.pi*6.) - np.tanh((-0.2)*np.pi*a) -(np.tanh(-(0.7-0.95)*np.pi*6)-np.tanh(0.4*np.pi*a))
        
        slct = 1.0<=x
        val[slct] = 0
        return val
    hgroup = h5py.File(output_fname, 'r+')['galaxyProperties']
    if 'inclination' in hgroup['morphology']:
        inclination = hgroup['morphology/inclination'].value
    else:
        inclination = None
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    size = np.size(mag_r)

    pos_angle = np.random.uniform(size=size)*np.pi
    print("pos_angle: ", pos_angle.size)
    if False: # Old code for ellipticity
        spheroid_axis_ratio = dtk.clipped_gaussian(0.8, 0.2, size, max_val = 1.0, min_val=0.0)
        dist,lim = dtk.make_distribution(-inclination)
        resamp = dtk.resample_distribution(dist,gal_zoo_dist,lim,[0.0,1.0])
        disk_axis_ratio = resamp(-inclination)
    else:
        # Returns ellip = 1-q^2 / 1+q^2
        # spheroid_ellip_cosmo = monte_carlo_ellipticity_bulge(mag_r)
        # disk_ellip_cosmo = monte_carlo_ellipticity_disk(mag_r, inclination)
        # We need to convert to q = sqrt((1-e)/(1+e))
        spheroid_ellip_cosmos, disk_ellip_cosmos = monte_carlo_ellipticity_bulge_disk(mag_r)
        spheroid_axis_ratio = np.sqrt((1-spheroid_ellip_cosmos**2)/(1+spheroid_ellip_cosmos**2))
        disk_axis_ratio = np.sqrt((1-disk_ellip_cosmos**2)/(1+disk_ellip_cosmos**2))
    # Calculate ellipticity from the axis ratios
    ellip_disk = (1.0 - disk_axis_ratio)/(1.0 + disk_axis_ratio)
    ellip_spheroid = (1.0 - spheroid_axis_ratio)/(1.0 + spheroid_axis_ratio)

    hgroup['morphology/spheroidAxisRatio'] = np.array(spheroid_axis_ratio, dtype='f4')
    hgroup['morphology/spheroidMajorAxisArcsec'] = np.array(hgroup['morphology/spheroidHalfLightRadiusArcsec'].value, dtype='f4')
    hgroup['morphology/spheroidMinorAxisArcsec'] = np.array(hgroup['morphology/spheroidHalfLightRadiusArcsec'].value*spheroid_axis_ratio, dtype='f4')
    hgroup['morphology/spheroidEllipticity'] = np.array(ellip_spheroid, dtype='f4')
    hgroup['morphology/spheroidEllipticity1'] =np.array( np.cos(2.0*pos_angle)*ellip_spheroid, dtype='f4')
    hgroup['morphology/spheroidEllipticity2'] =np.array( np.sin(2.0*pos_angle)*ellip_spheroid, dtype='f4')

    hgroup['morphology/diskAxisRatio'] = np.array(disk_axis_ratio, dtype='f4')
    hgroup['morphology/diskMajorAxisArcsec'] = np.array(hgroup['morphology/diskHalfLightRadiusArcsec'].value, dtype='f4')
    hgroup['morphology/diskMinorAxisArcsec'] = np.array(hgroup['morphology/diskHalfLightRadiusArcsec'].value*disk_axis_ratio, dtype='f4')
    hgroup['morphology/diskEllipticity'] = np.array(ellip_disk, dtype='f4')
    hgroup['morphology/diskEllipticity1'] =np.array( np.cos(2.0*pos_angle)*ellip_disk, dtype='f4')
    hgroup['morphology/diskEllipticity2'] =np.array( np.sin(2.0*pos_angle)*ellip_disk, dtype='f4')

    lum_disk = hgroup['SDSS_filters/diskLuminositiesStellar:SDSS_r:rest'].value
    lum_sphere = hgroup['SDSS_filters/spheroidLuminositiesStellar:SDSS_r:rest'].value
    lum_tot = lum_disk + lum_sphere
    tot_ellip =  (lum_disk*ellip_disk + lum_sphere*ellip_spheroid)/(lum_tot)
    hgroup['morphology/totalEllipticity']  = np.array(tot_ellip, dtype='f4')
    hgroup['morphology/totalAxisRatio']    = np.array((1.0 - tot_ellip)/(1.0 + tot_ellip), dtype='f4')
    hgroup['morphology/totalEllipticity1'] = np.array(np.cos(2.0*pos_angle)*tot_ellip, dtype='f4')
    hgroup['morphology/totalEllipticity2'] = np.array(np.sin(2.0*pos_angle)*tot_ellip, dtype='f4')
    hgroup['morphology/positionAngle'] = np.array(pos_angle*180.0/np.pi, dtype='f4')
    # print("position angle writen: ", np.array(pos_angle*180.0/np.pi, dtype='f4'))
    # print("position angle writen: ", np.array(pos_angle*180.0/np.pi, dtype='f4').size)
    srsc_indx_disk = 1.0*np.ones(lum_disk.size,dtype='f4')
    srsc_indx_sphere = 4.0*np.ones(lum_disk.size,dtype='f4')
    srsc_indx_tot = (srsc_indx_disk*lum_disk + srsc_indx_sphere*lum_sphere)/(lum_tot)
    hgroup['morphology/diskSersicIndex'] = srsc_indx_disk
    hgroup['morphology/spheroidSersicIndex'] = srsc_indx_sphere
    hgroup['morphology/totalSersicIndex'] = srsc_indx_tot
    return


def combine_step_lc_into_one(step_fname_list, out_fname, healpix=False):
    print("combining into one file")
    print(out_fname)
    print(step_fname_list)
    hfile_out = h5py.File(out_fname,'w')
    hfile_gp_out = hfile_out.create_group('galaxyProperties')
    hfile_steps = []
    hfile_steps_gp = []
    for fname in step_fname_list:
        hfile = h5py.File(fname,'r')
        gp = hfile['galaxyProperties']
        hfile_steps.append(hfile)
        hfile_steps_gp.append(gp)
    keys = get_keys(hfile_steps_gp[-1])
    for i,key in enumerate(keys):
        t1 = time.time()
        print("{}/{} {}".format(i,len(keys),key))
        if key == 'inclination':
            print('skipping in final output')
        data_list = []
        #units = None
        for h_gp in hfile_steps_gp:
            if key in h_gp:
                data_list.append(h_gp[key].value)
        data = np.concatenate(data_list)
        hfile_gp_out[key]=data
        #hfile_gp_out[key].attrs['units']=units
        print("\t time: {:.2f}".format(time.time()-t1))
    if not healpix:
        hfile_gp_out['galaxyID'] = np.arange(hfile_gp_out['redshift'].size,dtype='i8')
    return 


def add_metadata(gal_ref_fname, out_fname, version_major,
                 version_minor, version_minor_minor, healpix_ref=None,
                 param_file=None, snapshot=False):
    """
    Takes the metadata group and copies it over the final output product. 
    Also for each data column, copies the units attribute. 
    """
    add_units(out_fname)
    hfile_gf = h5py.File(gal_ref_fname,'r')
    hfile_out = h5py.File(out_fname,'a')
    if 'metaData' in hfile_out:
        del hfile_out['/metaData']
    hfile_out.copy(hfile_gf['metaData/GalacticusParameters'],'/metaData/GalacticusParameters/')
    hfile_out['/metaData/versionMajor'] = version_major
    hfile_out['/metaData/versionMinor'] = version_minor
    hfile_out['/metaData/versionMinorMinor'] = version_minor_minor
    hfile_out['/metaData/version'] = "{}.{}.{}".format(version_major, version_minor, version_minor_minor)

    hfile_out['/metaData/catalogCreationDate']=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if healpix_ref is not None:
        hfile_hp = h5py.File(healpix_ref,'r')['metaData']
        hfile_out['metaData/H_0'] = hfile_hp['H_0'].value
        hfile_out['metaData/Omega_b'] = hfile_hp['Omega_b'].value
        hfile_out['metaData/Omega_matter'] = hfile_hp['Omega_matter'].value
        if not snapshot:
            hfile_out['metaData/skyArea'] = hfile_hp['skyArea'].value
            hfile_out['metaData/cosmoDC2_Model/synthetic_halo_minimum_mass'] = hfile_hp['synthetic_halo_minimum_mass'].value
        hfile_out['metaData/cosmoDC2_Model/commit_hash'] = hfile_hp['commit_hash'].value
        hfile_out['metaData/cosmoDC2_Model/seed'] = hfile_hp['seed'].value


    else:
        if not snapshot:
            hfile_out['metaData/skyArea'] = 25
    try:
        cmd = 'git rev-parse HEAD'
        commit_hash = subprocess.check_output(cmd, shell=True).strip()
    except subprocess.CalledProcessError as cpe:
        with open('git_commit_hash.txt') as gcf:
            commit_hash = gcf.read()
    print("commit hash: ", commit_hash)
    hfile_out['/metaData/cosmodDC2_Matchup/commit_hash']= commit_hash
    if param_file is not None:
        with open(param_file, 'r') as pfile:
            data = pfile.read()
        hfile_out['/metaData/cosmodDC2_Matchup/config_file'] = data


def add_units(out_fname):
    hfile = h5py.File(out_fname,'a')['galaxyProperties']
    #################################
    ###Add units to all fields
    #################################
    mag_list = ['magnitude']; mag_unit = 'AB magnitude'
    arcsec_list= ['Arcsec']; arcsec_unit = 'arcsecond'
    rad_list = []; rad_unit ='radians'
    deg_list = ['ra','dec','ra_true', 'dec_true', 'morphology/positionAngle','inclination']; deg_unit = 'degrees'
    phys_kpc_list = ['Radius']; phys_kpc_unit = 'physical kpc'
    phys_mpc_list = []; phys_mpc_unit = 'physical Mpc'
    reduced_dist_list =['Reduced','EigenVector', 'Eddington'];reduced_dist_unit = 'unitless'
    eigen_val_list = ['EigenValue'];eigen_val_unit = 'comoving Mpc/h'
    comv_mpc_list = ['x','y','z']; comv_mpc_unit = 'comoving Mpc/h'
    vel_list = ['vx','vy','vz','Velocity']; vel_unit = 'km/s'
    timeSFR_list =['TimeWeightedIntegratedSFR']; timeSFR_unit = 'Gyr*Msun'
    sfr_list =['SFR','blackHoleAccretionRate','StarFormationRate']; sfr_unit = 'Msun/Gyr'
    mass_list =['Mass','IntegratedSFR']; mass_unit = 'Msun'
    abundance_list =['Abundance'];abundance_unit = 'Msun'
    luminosity_list =['Luminosities','Luminosity']; luminosity_unit = 'AB luminosity (4.4659e13 W/Hz)'
    unitless_list = ['redshift','shear','magnification','convergence','Ellipticity','Sersic','AxisRatio','dustFactor']; unitless_unit ='unitless'
    id_list = ['Index','Tag','placementType','galaxyID','lightcone_replication','lightcone_rotation', 'uniqueHaloID','isCentral']; id_unit = 'id/index'
    angular_list = ['angularMomentum'];angular_unit = 'Msun*km/s*Mpc'
    bool_list =['nodeIsIsolated'];bool_unit = 'boolean'
    spinSpin_list =['spinSpin'];spinSpin_unit ='lambda'
    step_list = ['step'];step_unit = 'simluation step'
    umachine_list = ['UMachineNative', 'baseDC2', 'matchUp'];umachine_unit = 'Unspecified'
    count_list =['NumberSelected'];count_unit = 'count'
    print("assigning units")
    keys = get_keys(hfile)
    print( keys)
    for key in keys:
        print(key)
        print('\t',hfile[key].dtype)
        #add magnitude units
        if(any(l in key for l in mag_list)):
            hfile[key].attrs['units']=mag_unit
            print("\t mag")
            # umachine list
        elif(any(l in key for l in umachine_list)):
            hfile[key].attrs['units']=umachine_unit
            #add arcsec units
        elif(any(l in key for l in arcsec_list)):
            hfile[key].attrs['units']=arcsec_unit
            print( "\t ",arcsec_unit)
            #add rad units
        elif(any(l in key for l in rad_list)):
            hfile[key].attrs['units']=rad_unit
            print( "\t ",rad_unit)
            #add degree units
        elif(any(l == key for l in deg_list)):
            hfile[key].attrs['units']=deg_unit
            print( '\t',deg_unit)
            #add kpc units
        elif(any(l in key for l in phys_kpc_list)):
            hfile[key].attrs['units']=phys_kpc_unit
            print( "\t ",phys_kpc_unit)
            #add mpc units
        elif(any(l in key for l in phys_mpc_list)):
            hfile[key].attrs['units']=phys_mpc_unit
            print ("\t ",phys_mpc_unit)
            #reduced distances units
        elif(any(l in key for l in reduced_dist_list)):
            hfile[key].attrs['units']=reduced_dist_unit
            print ("\t ",reduced_dist_unit)
            #eigen val units
        elif(any(l in key for l in eigen_val_list)):
            hfile[key].attrs['units']=eigen_val_unit
            print ("\t ",reduced_dist_unit)
            #add comoving mpc units
        elif(any(l == key for l in comv_mpc_list)):
            hfile[key].attrs['units']=comv_mpc_unit
            print ("\t ",comv_mpc_unit)
            #add velocity units
        elif(any(l in key for l in vel_list)):
            hfile[key].attrs['units']=vel_unit
            print ("\t ",vel_unit)
            #add timesfr
        elif(any(l in key for l in timeSFR_list)):
            hfile[key].attrs['units']=timeSFR_unit
            print ("\t ",timeSFR_unit)
            #add sfr
        elif(any(l in key for l in sfr_list)):
            hfile[key].attrs['units']=sfr_unit
            print ("\t ",sfr_unit)
            #add mass
        elif(any(l in key for l in mass_list)):
            hfile[key].attrs['units']=mass_unit
            print ("\t ",mass_unit)
            #add abundance
        elif(any(l in key for l in abundance_list)):
            hfile[key].attrs['units']=abundance_unit
            print ("\t ",abundance_unit)
            #add luminosity units
        elif(any(l in key for l in luminosity_list)):
            hfile[key].attrs['units']=luminosity_unit
            print ("\t ",luminosity_unit)
            #add unit less
        elif(any(l in key for l in unitless_list)):
            hfile[key].attrs['units']=unitless_unit
            print ("\t ",unitless_unit)
            #add mass units
        elif(any(l in key for l in id_list)):
            hfile[key].attrs['units']=id_unit
            print ("\t ",id_unit)
            #angular momentum 
        elif(any(l in key for l in angular_list)):
            hfile[key].attrs['units']=angular_unit
            print ("\t ",angular_unit)
            #boolean
        elif(any(l in key for l in bool_list)):
            hfile[key].attrs['units']=bool_unit
            print ("\t", bool_unit)
            #spinSpin
        elif(any(l in key for l in spinSpin_list)):
            hfile[key].attrs['units']=spinSpin_unit
            # step
        elif(any(l in key for l in step_list)):
            hfile[key].attrs['units']=step_unit
            #counts
        elif(any(l in key for l in count_list)):
            hfile[key].attrs['units']=count_unit
            #Everything should have a unit!
        else:
            print("column", key, "was not assigned a unit :(")
            print("===================")
            #raise;

def plot_differences(lc_data, gal_prop, index):
    keys = ['Mag_r','clr_gr','clr_ri','m_star']
    dist = {}
    dist_all = None
    for key in keys:
        d = lc_data[key]-gal_prop[key][index]
        dist[key] = d
        if(dist_all is None):
            dist_all = d*d
        else:
            dist_all += d*d
    dist_all = np.sqrt(dist_all)
    plt.figure()
    for key in keys:
        slct_fnt = np.isfinite(dist[key])
        bins = np.linspace(np.min(dist[key][slct_fnt]), np.max(dist[key][slct_fnt]), 100)
        h,xbins = np.histogram(dist[key][slct_fnt],bins=bins)
        plt.plot(dtk.bins_avg(xbins),h,label=key)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('original value - matched value')
    plt.ylabel('count')
    plt.figure()
    slct_fnt = np.isfinite(dist_all)
    bins = np.linspace(np.min(dist_all[slct_fnt]), np.max(dist_all[slct_fnt]), 100)
    h,xbins = np.histogram(dist_all[slct_fnt],bins=bins)
    plt.plot(dtk.bins_avg(xbins),h,label='all',lw=2.0)
    for key in keys:
        slct_fnt = np.isfinite(dist[key])
        h,xbins = np.histogram(dist[key][slct_fnt],bins=xbins)
        plt.plot(dtk.bins_avg(xbins),h,label=key)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('distance in match')
    plt.ylabel('count')
    return


def plot_differences_obs_color(lc_data, gal_prop, index):
    slct = lc_data['is_cluster_red_sequence']
    keys = ['Mag_r','clr_gr','clr_ri', 'clr_gr_obs', 'clr_ri_obs', 'clr_iz_obs']
    dist = {}
    dist_all = None
    for key in keys:
        d = lc_data[key][slct]-gal_prop[key][index][slct]
        dist[key] = d
        if(dist_all is None):
            dist_all = d*d
        else:
            dist_all += d*d
    dist_all = np.sqrt(dist_all)
    plt.figure()
    for key in keys:
        slct_fnt = np.isfinite(dist[key])
        bins = np.linspace(np.min(dist[key][slct_fnt]), np.max(dist[key][slct_fnt]), 100)
        h,xbins = np.histogram(dist[key][slct_fnt],bins=bins)
        plt.plot(dtk.bins_avg(xbins),h,label=key)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('original value - matched value')
    plt.ylabel('count')

    plt.figure()
    slct_fnt = np.isfinite(dist_all)
    bins = np.linspace(np.min(dist_all[slct_fnt]), np.max(dist_all[slct_fnt]), 100)
    for key in keys:
        slct_fnt = np.isfinite(dist[key])
        h,xbins = np.histogram(dist[key][slct_fnt],bins=xbins)
        plt.plot(dtk.bins_avg(xbins),h,label=key)
    h,xbins = np.histogram(dist_all[slct_fnt],bins=bins)
    plt.plot(dtk.bins_avg(xbins),h,label='all',lw=2.0)
    plt.yscale('log')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('distance in match')
    plt.ylabel('count')
    return


def plot_differences_2d(lc_data, gal_prop,index, x='Mag_r'):
    keys = ['Mag_r','clr_gr','clr_ri','m_star']
    for key in keys:
        # if key == x:
        #     continue
        plt.figure()
        h,xbins,ybins = np.histogram2d(lc_data[x],lc_data[key]-gal_prop[key][index],bins=(100,100))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm = clr.LogNorm())
        plt.ylabel("diff {} (orginal-new)".format(key))
        plt.xlabel(x)
        plt.grid()
    return

 
def plot_side_by_side(lc_data, gal_prop, index, x='Mag_r'):
    keys =  ['Mag_r','clr_gr','clr_ri','m_star']
    for key in keys:
        if key == x:
            continue
        fig,axs = plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
        h,xbins,ybins = np.histogram2d(lc_data[x],lc_data[key],bins=(100,100))
        axs[0].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        axs[0].grid()
        axs[0].set_title('UMachine + SDSS')
        axs[0].set_xlabel(x)
        axs[0].set_ylabel(key)

        h,xbins,ybins = np.histogram2d(gal_prop[x][index],gal_prop[key][index],bins=(xbins,ybins))
        axs[1].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        axs[1].grid()
        axs[1].set_title('Matched Galacticus')
        axs[1].set_xlabel(x)
        axs[1].set_ylabel(key)

        h,xbins,ybins = np.histogram2d(gal_prop[x],gal_prop[key],bins=(xbins,ybins))
        axs[2].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        axs[2].grid()
        axs[2].set_title('Galacticus ')
        axs[2].set_xlabel(x)
        axs[2].set_ylabel(key)
        
    fig,axs = plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
    h,xbins,ybins = np.histogram2d(lc_data['clr_gr'],lc_data['clr_ri'],bins=(100,100))
    axs[0].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[0].grid()
    axs[0].set_title('UMachine + SDSS')
    axs[0].set_xlabel('clr_gr')
    axs[0].set_ylabel('clr_ri')
    
    h,xbins,ybins = np.histogram2d(gal_prop['clr_gr'][index],gal_prop['clr_ri'][index],bins=(xbins,ybins))
    axs[1].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[1].grid()
    axs[1].set_title('Matched Galacticus')
    axs[1].set_xlabel('clr_gr')
    axs[1].set_ylabel('clr_ri')
    
    h,xbins,ybins = np.histogram2d(gal_prop['clr_gr'],gal_prop['clr_ri'],bins=(xbins,ybins))
    axs[2].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[2].grid()
    axs[2].set_title('Galacticus ')
    axs[2].set_xlabel('clr_gr')
    axs[2].set_ylabel('clr_ri')
        
    return


def plot_mag_r(lc_data,gal_prop,index):
    plt.figure()
    slct_fnt_lc = np.isfinite(lc_data['Mag_r'])
    slct_fnt_gp = np.isfinite(gal_prop['Mag_r'][index])
    max_r = np.max((np.max(lc_data['Mag_r'][slct_fnt_lc]),np.max(gal_prop['Mag_r'][index][slct_fnt_gp])))
    min_r = np.min((np.min(lc_data['Mag_r'][slct_fnt_lc]),np.min(gal_prop['Mag_r'][index][slct_fnt_gp])))
    bins = np.linspace(min_r,max_r,100)
    h_lc,_ = np.histogram(lc_data['Mag_r'][slct_fnt_lc],bins=bins)
    h_mg,_ = np.histogram(gal_prop['Mag_r'][index][slct_fnt_gp],bins=bins)
    plt.plot(dtk.bins_avg(bins),h_lc, 'b', label='UMachine-SDSS')
    plt.plot(dtk.bins_avg(bins),h_mg, 'r', label='Matched Glctcs')
    plt.grid()
    plt.xlabel("Mr")
    plt.ylabel('Count')
    plt.legend(loc='best')
    return


def plot_m_star(lc_data,gal_prop,index):
    plt.figure()
    max_r = np.max((np.max(lc_data['m_star']),np.max(gal_prop['m_star'][index])))
    min_r = np.min((np.min(lc_data['m_star']),np.min(gal_prop['m_star'][index])))
    bins = np.linspace(min_r,max_r,100)
    h_lc,_ = np.histogram(lc_data['m_star'],bins=bins)
    h_mg,_ = np.histogram(gal_prop['m_star'][index],bins=bins)
    plt.plot(dtk.bins_avg(bins),h_lc, 'b', label='UMachine-SDSS')
    plt.plot(dtk.bins_avg(bins),h_mg, 'r', label='Matched Glctcs')
    plt.grid()
    plt.xlabel("log10(Stellar Mass)")
    plt.ylabel('Count')
    plt.legend(loc='best')
    return


def plot_single_dist(lc_data,gal_prop,index,key_name,key_label,bins =
                     None):
    plt.figure()
    if bins is None:
        max_r = np.max((np.max(lc_data[key_name]),np.max(gal_prop[key_name][index])))
        min_r = np.min((np.min(lc_data[key_name]),np.min(gal_prop[key_name][index])))
        bins = np.linspace(min_r,max_r,100)
    h_lc,_ = np.histogram(lc_data[key_name],bins=bins)
    h_mg,_ = np.histogram(gal_prop[key_name][index],bins=bins)
    plt.plot(dtk.bins_avg(bins),h_lc, 'b', label='UMachine-SDSS')
    plt.plot(dtk.bins_avg(bins),h_mg, 'r', label='Matched Glctcs')
    plt.grid()
    plt.xlabel(key_label)
    plt.ylabel('Count')
    plt.legend(loc='best')
    return


def plot_clr_mag(lc_data,gal_prop,index,mag_bins,data_key, data_name):
    fig,ax = plt.subplots(1,len(mag_bins),figsize=(15,5))
    # max_gr = np.max((np.max(lc_data[data_key]),np.max(gal_prop[data_key])))
    # min_gr = np.min((np.min(lc_data[data_key]),np.min(gal_prop[data_key])))
    bins = np.linspace(0.0, 1.1, 50)
    for i in range(0, len(mag_bins)):
        if i == 0:
            slct_lc = lc_data['Mag_r']<mag_bins[i]
            slct_mg = gal_prop['Mag_r'][index]<mag_bins[i]
            ax[i].set_title('Mr < {}'.format(mag_bins[i]))
        else:
            slct_lc = (mag_bins[i-1] < lc_data['Mag_r']) & ( lc_data['Mag_r'] < mag_bins[i])
            slct_mg = (mag_bins[i-1] < gal_prop['Mag_r'][index]) & ( gal_prop['Mag_r'][index] < mag_bins[i])
            ax[i].set_title('{} < Mr < {}'.format(mag_bins[i-1], mag_bins[i]))
        h_lc, _ = np.histogram(lc_data[data_key][slct_lc],bins=bins)
        h_mg, _ = np.histogram(gal_prop[data_key][index][slct_mg],bins=bins)
        ax[i].plot(dtk.bins_avg(bins),h_lc,'b', label = 'UMachine-SDSS')
        ax[i].plot(dtk.bins_avg(bins),h_mg,'r', label = 'Matched Glctcs')
        if i ==0:
            ax[i].legend(loc='best', framealpha=0.3)
        ax[i].set_xlabel(data_name)
        ax[i].set_ylabel('Count')
        ax[i].grid()
        

def plot_ri_gr_mag(lc_data, gal_prop, index, mag_bins):
    fig,ax = plt.subplots(1,len(mag_bins),figsize=(15,5))
    for i in range(0, len(mag_bins)):
        if i == 0:
            slct_lc = lc_data['Mag_r']<mag_bins[i]
            slct_mg = gal_prop['Mag_r'][index]<mag_bins[i]
            ax[i].set_title('Mr < {}'.format(mag_bins[i]))
        else:
            slct_lc = (mag_bins[i-1] < lc_data['Mag_r']) & ( lc_data['Mag_r'] < mag_bins[i])
            slct_mg = (mag_bins[i-1] < gal_prop['Mag_r'][index]) & ( gal_prop['Mag_r'][index] < mag_bins[i])
            ax[i].set_title('{} < Mr < {}'.format(mag_bins[i-1], mag_bins[i]))
        # print('{} < Mr < {}'.format(mag_bins[i], mag_bins[i-1]))
        # print(np.average(lc_data['Mag_r'][slct_lc]))
        ax[i].plot(lc_data['clr_gr'][slct_lc], lc_data['clr_ri'][slct_lc],'.b',alpha=0.5,label='UMachine-SDSS',ms=4)
        ax[i].plot(gal_prop['clr_gr'][index][slct_mg], gal_prop['clr_ri'][index][slct_mg],'.r',alpha=0.5,label='Matched Glctcs', ms=4)
        if i ==0:
            ax[i].legend(loc='best', framealpha=0.3)
        ax[i].set_xlabel('g-r color')
        ax[i].set_ylabel('r-i color')
        ax[i].grid()


def plot_clr_z(lc_data, gal_prop, index,clr='clr_gr'):
    fig,axs = plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
    h,xbins,ybins = np.histogram2d(lc_data['redshift'],lc_data[clr],bins=(100,100))
    axs[0].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[0].grid()
    axs[0].set_title('UMachine + SDSS')
    axs[0].set_xlabel('redshift')
    axs[0].set_ylabel(clr)
    h,xbins,ybins = np.histogram2d(gal_prop['redshift'][index],gal_prop[clr][index],bins=(xbins,ybins))
    axs[1].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[1].grid()
    axs[1].set_title('Matched Galacticus')
    axs[1].set_xlabel('redshift')
    axs[1].set_ylabel(clr)
    h,xbins,ybins = np.histogram2d(gal_prop['redshift'],gal_prop[clr],bins=(xbins,ybins))
    axs[2].pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    axs[2].grid()
    axs[2].set_title('Galacticus ')
    axs[2].set_xlabel('redshift')
    axs[2].set_ylabel(clr)
    

def plot_gal_prop_dist(gal_props, gal_props_names):
    num = len(gal_props)
    num_list = range(0, num)
    plt.figure()
    bins = np.linspace(5,14,100)
    for i in num_list:
        h,xbins = np.histogram(gal_props[i]['m_star'], bins = bins, normed=True)
        plt.plot(dtk.bins_avg(xbins), h, label=gal_props_names[i])
    plt.grid()
    plt.xlabel('m_star')
    plt.ylabel('count')
    plt.legend(loc='best', framealpha=0.3)

    plt.figure()
    bins = np.linspace(-25,-10, 100)
    for i in num_list:
        h,xbins = np.histogram(gal_props[i]['Mag_r'], bins = bins, normed=True)
        plt.plot(dtk.bins_avg(xbins), h, label=gal_props_names[i])
    plt.grid()
    plt.xlabel('Mag_r')
    plt.ylabel('count')
    plt.legend(loc='best', framealpha=0.3)

    plt.figure()
    bins = np.linspace(-.5,3, 100)
    for i in num_list:
        h,xbins = np.histogram(gal_props[i]['clr_gr'], bins = bins, normed=True)
        plt.plot(dtk.bins_avg(xbins), h, label=gal_props_names[i])
    plt.grid()
    plt.xlabel('clr_gr')
    plt.ylabel('count')
    plt.legend(loc='best', framealpha=0.3)

    plt.figure()
    bins = np.linspace(-.5,3, 100)
    for i in num_list:
        h,xbins = np.histogram(gal_props[i]['clr_ri'], bins = bins, normed=True)
        plt.plot(dtk.bins_avg(xbins), h, label=gal_props_names[i])
    plt.grid()
    plt.xlabel('clr_ri')
    plt.ylabel('count')
    plt.legend(loc='best', framealpha=0.3)
    
    xbins, ybins = (np.linspace(-25,-10,250), np.linspace(-.5,2,250))
    fig, axs = plt.subplots(1,num,figsize=(num*5,5), sharex = True)
    for i in num_list:
        h,xbins,ybins = np.histogram2d(gal_props[i]['Mag_r'], gal_props[i]['clr_gr'],bins=(xbins,ybins))
        axs[i].pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        axs[i].grid()
        axs[i].set_title(gal_props_names[i])
        axs[i].set_xlabel('Mag_r')
        axs[i].set_ylabel('clr_gr')

    fig, axs = plt.subplots(1,num,figsize=(num*5,5), sharex = True)
    for i in num_list:
        h,xbins,ybins = np.histogram2d(gal_props[i]['Mag_r'], gal_props[i]['clr_ri'],bins=(xbins,ybins))
        axs[i].pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        axs[i].grid()
        axs[i].set_title(gal_props_names[i])
        axs[i].set_xlabel('Mag_r')
        axs[i].set_ylabel('clr_ri')
    

def lightcone_resample(param_file_name):
    t00 = time.time()
    # Loading in all the parameters from the parameter file
    param = dtk.Param(param_file_name)
    lightcone_fname = param.get_string('lightcone_fname')
    gltcs_fname = param.get_string('gltcs_fname')
    gltcs_metadata_ref = param.get_string('gltcs_metadata_ref')
    gltcs_slope_fname = param.get_string('gltcs_slope_fname')
    sod_fname = param.get_string("sod_fname")
    halo_shape_fname = param.get_string("halo_shape_fname")
    halo_shape_red_fname = param.get_string("halo_shape_red_fname")
    output_fname = param.get_string('output_fname')
    healpix_file = param.get_bool('healpix_file')
    if healpix_file:
        healpix_pixels = param.get_int_list('healpix_pixels')
    fake_lensing  = param.get_bool('fake_lensing')
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
    if 'ignore_bright_luminosity_softness' in param:
        ignore_bright_luminosity_softness = param.get_float('ignore_bright_luminosity_softness')
    else:
        print('default value for ignore_bright_luminosity_softness')
        ignore_bright_luminosity_softness = 0.0
    version_major = param.get_int('version_major')
    version_minor = param.get_int('version_minor')
    version_minor_minor = param.get_int('version_minor_minor')

    if "concatenate_only" in param:
        concatenate_only = param.get_bool("concatenate_only")
    else:
        concatenate_only = False

    if "resume_at_step" in param:
        resume_run = True
        resume_at_step = param.get_int("resume_at_step")
    else:
        resume_run = False
        resume_at_step = 0
    
    if 'no_shear_steps' in param:
        no_shear_steps = param.get_int_list('no_shear_steps')
    else:
        no_shear_steps = np.zeros(0,'i4')
    
    if 'healpix_shear_file' in param:
        healpix_shear_file = param.get_string('healpix_shear_file')
    else:
        healpix_shear_file = None
    if "red_sequence_transition_mass_start" in param:
        red_sequence_transition_mass_start = param.get_float('red_sequence_transition_mass_start')
        red_sequence_transition_mass_end = param.get_float('red_sequence_transition_mass_end')
    else:
        red_sequence_transition_mass_start = 13.5
        red_sequence_transition_mass_end = 13.5
    if "metadata_only" in param:
        metadata_only = param.get_bool('metadata_only')
    else:
        metadata_only = False
    if "snapshot" in param:
        snapshot = param.get_bool("snapshot")
    else:
        snapshot = False
    # Adding scatter to the red squence
    rs_scatter_dict = get_rs_scatter_dict_from_param(param)
    red_sequence_transition_mass_start = red_sequence_transition_mass_start,
    red_sequence_transition_mass_end = red_sequence_transition_mass_end


    # The other options are depricated
    assert use_dust_factor & use_slope, "Must set use_dust_factor and use_slope to true. The other settings are depricated"
    assert ("${step}" in output_fname), "Must have ${step} in output_fname to generate sperate files for each step"
    if healpix_file:
        assert ("${healpix}" in output_fname), "Must have ${healpix} string in output while using healpix"
    if fake_lensing:
        assert ((healpix_shear_file is None) or healpix_shear_file == "NULL"), "If `fake_lensing` is set to true, healpix_shear_file must either not in the param file or set to NULL."
    if not cut_small_galaxies: # if we don't cut out small galaxies, set the mass limit
        cut_small_galaxies_mass = None # to None as a flag for other parts in the code
    # Check for setting for snapshot runs
    if snapshot:
        assert substeps == 1, "For snapshots, there is no reason to have more than 1 substep. Maybe this is suppposed to be a lightcone run?"
        assert use_substep_redshift, "For snapshots, there is not reason not use the substep redshift for all galaxies. If this is set to false, maybe this is supposed to be a light conerun?"
        
    # Load Eve's galmatcher mask. Another script writes the mask to file (Need to check which one)
    if load_mask:
        hfile_mask = h5py.File(mask_loc,'r')
    else:
        selection1 = galmatcher.read_selections(yamlfile='galmatcher/yaml/vet_protoDC2.yaml')
        selection2 = galmatcher.read_selections(yamlfile='galmatcher/yaml/colors_protoDC2.yaml')
    # This object converts simulation steps into redshift or scale factor
    stepz = dtk.StepZ(sim_name='AlphaQ')
    output_step_list = [] # A running list of the output genereated for each time step. 
    # After all the steps run, the files are concatenated into one file. 

    # If it's on a snapshot, we will run on every single step. So we
    # will iterate the same as number of times as steps given
    if snapshot: 
        step_i_limit = steps.size
    # If we are on a ligtcone, we will interpolate between step[i+1]
    # and step[i], where steps are are decreasing. So we will iterate
    # one less than the array size
    else:
        step_i_limit = steps.size-1

    for i in range(0,step_i_limit):
        # Since the interpolation scheme needs to know the earlier and later time step
        # that are interpolated between, we iterate over all pairs of steps. The outputs
        # are are labeled with the earlier time step i.e. the interpolation between 487
        # and 475 are output is labeled with 475
        t0 = time.time()
        if snapshot: # We will be "interpolating" between the same two snapshots
            step = steps[i]
            step2 = steps[i]
        else: # lightcone interpolation 
            step = steps[i+1]
            step2 = steps[i]
        print("\n\n=================================")
        print(" STEP: ",step)
        gltcs_step_fname = gltcs_fname.replace("${step}",str(step)) 
        lightcone_step_fname = lightcone_fname.replace("${step}",str(step))
        output_step_loc = output_fname.replace("${step}",str(step))
        output_step_list.append(output_step_loc)
        sod_step_loc = sod_fname.replace("${step}",str(step))
        halo_shape_step_loc = halo_shape_fname.replace("${step}",str(step))
        halo_shape_red_step_loc = halo_shape_red_fname.replace("${step}",str(step))
        if concatenate_only or metadata_only: 
            print("Skipping: concatenate_only is true ")
            continue

        if resume_run and step > resume_at_step :
            print("Skipping: we haven't reach the resume step: step {} > resume {}".format(step,resume_at_step))
            continue

        elif resume_run:
            print("resuming: step {} > resume {}".format(step,resume_at_step))
        else:
            print("not a resume run")
   
        if load_mask:
            mask1 = hfile_mask['{}'.format(step)].value
            mask2 = hfile_mask['{}'.format(step2)].value
        else:
            mask_a = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection1)
            mask_b = galmatcher.mask_cat(h5py.File(gltcs_step_fname, 'r'), selections=selection2)
            mask1 = mask_a & mask_b
            gltcs_step2_fname = gltcs_fname.replace("${step}",str(step2))
            mask_a = galmatcher.mask_cat(h5py.File(gltcs_step2_fname, 'r'), selections=selection1)
            mask_b = galmatcher.mask_cat(h5py.File(gltcs_step2_fname, 'r'), selections=selection2)
            mask2 = mask_a & mask_b
        verbose = True
        # Healpix cutouts/files have the step saved inside of them.
        if healpix_file:
            internal_file_step = step
        else:
            internal_file_step = None
        # Load the mock (UMachine + Color + Shear) into dict of arrays. 
        if not(healpix_file):
            lc_data = construct_lc_data(lightcone_step_fname, verbose = verbose, recolor=recolor, 
                                        match_obs_color_red_seq = match_obs_color_red_seq,
                                        cut_small_galaxies_mass = cut_small_galaxies_mass, 
                                        internal_step=internal_file_step,
                                        red_sequence_transition_mass_start = red_sequence_transition_mass_start,
                                        red_sequence_transition_mass_end = red_sequence_transition_mass_end, 
                                        snapshot=snapshot)

        else:
            lc_data = construct_lc_data_healpix(lightcone_step_fname, verbose = verbose, recolor=recolor, 
                                                match_obs_color_red_seq = match_obs_color_red_seq,
                                                cut_small_galaxies_mass = cut_small_galaxies_mass, 
                                                internal_step=internal_file_step,
                                                healpix_pixels = healpix_pixels,
                                                red_sequence_transition_mass_start = red_sequence_transition_mass_start,
                                                red_sequence_transition_mass_end = red_sequence_transition_mass_end, 
                                                snapshot=snapshot)
        # The index remap galaxies in step2 to the same order as they
        # were in step1. This is needed fore interpolation. If we
        # making comoDC2 for a snapshot, we don't need interpolation.
        if snapshot:
            index_2to1 = None
            snapshot_redshift = lc_data['redshift'][0] # Get the redshift of 
            # the snapshot from the metadata of the baseDC2. They all have the same value
        else:
            index_2to1 = h5py.File(index_loc.replace("${step}",str(step)), 'r')['match_2to1'].value
            snapshot_redshift = None
        #There is no other option. I just don't want to re-indent this entire block of code--
        #emacs doesn't re-indent python code well
        if(use_slope): 
            print("using interpolation on step", step)
            lc_a = 1.0/(1.0 +lc_data['redshift'])
            lc_a_cc = np.copy(lc_a) # galaxy scale factor for copy columns
            del_lc_a =  np.max(lc_a) - np.min(lc_a)
            step_a = np.min(lc_a)-0.01*del_lc_a #so no galaxy is exactly on the egdge of the bins
            step2_a = np.max(lc_a)+0.01*del_lc_a
            print('=======')
            print("lightcone min a:       {}".format(step_a))
            # print("lc raw min a:   {}".format(np.min(lc_a)))
            # print("dtk step min a: {}".format(stepz.get_a(step)))
            # print("gltcs        a: {}".format(1.0/(2.0180+1.0)) )  
            # print('=======')
            print("lightcone max a        {}".format(step2_a))
            # print("lc raw max a:   {}".format(np.max(lc_a)))
            # print("dtk step max a: {}".format(stepz.get_a(step2)))
            # print("gltcs        a: {}".format(1.0/(1.9472+1.0)))
            print("===================")

            abins = np.linspace(step_a, step2_a,substeps+1)
            abins_avg = dtk.bins_avg(abins)
            index = -1*np.ones(lc_data['redshift'].size,dtype='i8')
            match_dust_factors = -np.ones(lc_data['redshift'].size,dtype='f4')
            match_luminosity_factors = -1*np.ones(lc_data['redshift'].size,dtype='f4')
            match_library_index = -1*np.ones(lc_data['redshift'].size,dtype='i8')
            match_node_index = -1*np.ones(lc_data['redshift'].size,dtype='i8')
            for k in range(0, substeps):
                print("substep {}/{}".format(k, substeps))
                # if it's a snapshot, we will select all galaxies
                # within this one substep. There is no need for
                # interpolation (thus substepping) for snapshots
                if snapshot: 
                    slct_lc_abin = np.ones(lc_data['m_star'].size, dtype=bool)
                    assert substeps == 1, "Just double checking that there is only one substep"
                else:
                    print("\t{}/{} substeps".format(k,abins_avg.size))
                    slct_lc_abins1 = (abins[k]<=lc_a) 
                    slct_lc_abins2 = (lc_a<abins[k+1])
                    print("\t\t {} -> {}".format(abins[k],abins[k+1]))
                    print("\t\t center a: {}".format(abins_avg[k]))
                    print("\t\t step a: {} -> {}".format(step_a, step2_a))
                    slct_lc_abin = slct_lc_abins1 & slct_lc_abins2
                print("\t\t num gals: {}/{}".format(np.sum(slct_lc_abin), slct_lc_abin.size))


                lc_data_a = dic_select(lc_data, slct_lc_abin)
                if lc_data_a['redshift'].size == 0:
                    print("\t\t\t no galaxies for this redshift bin")
                    continue #nothing to match for this redshift bin
                if use_dust_factor:
                    gal_prop_list = [] 
                    for dust_factor in np.concatenate(([1.0],dust_factors)):
                        print("\tdust_factor********=",dust_factor)
                        gal_prop_tmp2 = construct_gal_prop_redshift_dust_raw(
                            gltcs_fname, index_2to1, step, step2, step_a, step2_a, abins_avg[k],
                            mask1, mask2, dust_factor, match_obs_color_red_seq,
                            cut_small_galaxies_mass = cut_small_galaxies_mass, snapshot=snapshot)
                        gal_prop_list.append(gal_prop_tmp2)
                    gal_prop_a = cat_dics(gal_prop_list)
                # Find the closest Galacticus galaxy
                index_abin = resample_index(lc_data_a, gal_prop_a, 
                                            ignore_mstar = ignore_mstar, 
                                            verbose = verbose, 
                                            ignore_bright_luminosity=ignore_bright_luminosity, 
                                            ignore_bright_luminosity_threshold = ignore_bright_luminosity_threshold,
                                            ignore_bright_luminosity_softness = ignore_bright_luminosity_softness)
                #If we are matching on observed colors for cluster red seqence guys:
                if match_obs_color_red_seq:
                    print("Matching on obs red seq")
                    #Create a lc_data with only cluster red sequence galaxies
                    slct_clstr_red_squence = lc_data_a['is_cluster_red_sequence']
                    if np.sum(slct_clstr_red_squence) > 0:
                        lc_data_a_crs = dic_select(lc_data_a, slct_clstr_red_squence)
                        # Find the closest Galacticus galaxy as before but also match on 
                        # observed g-r, r-i, and i-z colors
                        index_abin_crs = resample_index_cluster_red_squence(
                            lc_data_a_crs, gal_prop_a, 
                            ignore_mstar = ignore_mstar,
                            verbose = verbose,
                            ignore_bright_luminosity=ignore_bright_luminosity,
                            ignore_bright_luminosity_threshold = ignore_bright_luminosity_threshold,
                            ignore_bright_luminosity_softness = ignore_bright_luminosity_softness,
                            rs_scatter_dict = rs_scatter_dict)
                        index_abin[slct_clstr_red_squence] = index_abin_crs
                    else:
                        print("\tno red squence galaxies, so skipping...")
                if use_dust_factor:
                    # Get the Galacticus galaxy index, the division is to correctly
                    # offset the index for the extra dust gal_prop 
                    index[slct_lc_abin] = gal_prop_a['index'][index_abin]
                    # = index_abin%(index_abin.size//(1+len(dust_factors)))
                    # Record the dust factor for the matched galaxy so that it can be applied 
                    # to other columns in copy_columns()
                    match_dust_factors[slct_lc_abin] = gal_prop_a['dust_factor'][index_abin]
                    match_library_index[slct_lc_abin] = gal_prop_a['index'][index_abin]
                    match_node_index[slct_lc_abin] = gal_prop_a['node_index'][index_abin]
                    if use_substep_redshift:
                        lc_a_cc[slct_lc_abin] = abins_avg[k]
                # By default use the same Galacticus luminosity
                match_luminosity_factors[slct_lc_abin] = 1.0
                # For the brightest galaxies, adjust all luminosities by the same factor
                # so that the r-band matches
                if rescale_bright_luminosity:
                    slct_rescale_galaxies = lc_data_a['Mag_r'] < rescale_bright_luminosity_threshold
                    if np.sum(slct_rescale_galaxies) > 0:
                        print("num bright galaxies to rescale luminosity: {}".format(np.sum(slct_rescale_galaxies)))
                        tmp = 10**((-lc_data_a['Mag_r'][slct_rescale_galaxies] + gal_prop_a['Mag_r'][index_abin][slct_rescale_galaxies])/2.5)
                        slct_tmp = np.copy(slct_lc_abin)
                        slct_tmp[slct_lc_abin]=slct_rescale_galaxies
                        match_luminosity_factors[slct_tmp]=tmp
                if plot_substep:
                    plot_differences(lc_data_a, gal_prop_a, index_abin);
                    plot_differences_obs_color(lc_data_a, gal_prop_a, index_abin);
                    plot_differences_2d(lc_data_a, gal_prop_a, index_abin);
                    plot_side_by_side(lc_data_a, gal_prop_a, index_abin);
                    mag_bins = (-21,-20,-19);
                    plot_mag_r(lc_data_a, gal_prop_a, index_abin);
                    plot_clr_mag(lc_data_a, gal_prop_a, index_abin, mag_bins, 'clr_gr', 'g-r color');
                    #plot_clr_mag(lc_data, gal_prop_a, index_abin, mag_bins, 'clr_ri', 'r-i color')
                    plot_ri_gr_mag(lc_data_a, gal_prop_a, index_abin, mag_bins);
                    plt.show()
            slct_neg = index == -1
            print("assigned: {}/{}: {:.2f}".format( np.sum(~slct_neg), slct_neg.size, np.float(np.sum(slct_neg))/np.float(slct_neg.size)))
            assert(np.sum(slct_neg) == 0)
            

        if not(healpix_file):
            copy_columns_interpolation_dust_raw(gltcs_fname, output_step_loc, index, 
                                                step, step2, step_a, step2_a, mask1, mask2, 
                                                index_2to1, lc_a_cc, verbose = verbose,
                                                short = short, supershort = supershort,
                                                dust_factors = match_dust_factors, step = step,
                                                luminosity_factors = match_luminosity_factors,
                                                library_index = match_library_index,
                                                node_index = match_node_index, snapshot=snapshot)
            overwrite_columns(lightcone_step_fname, output_step_loc, ignore_mstar = ignore_mstar,
                              verbose = verbose, cut_small_galaxies_mass = cut_small_galaxies_mass,
                              internal_step = internal_file_step, fake_lensing=fake_lensing, step = step,
                              no_shear_steps = no_shear_steps,
                              snapshot=snapshot,
                              snapshot_redshift = snapshot_redshift)
            overwrite_host_halo(output_step_loc,sod_step_loc,
                                halo_shape_step_loc,
                                halo_shape_red_step_loc,
                                verbose=verbose, snapshot=snapshot)
            add_native_umachine(output_step_loc, lightcone_step_fname, cut_small_galaxies_mass = cut_small_galaxies_mass,
                                internal_step = internal_file_step, 
                                snapshot=snapshot)
            add_blackhole_quantities(output_step_loc, np.average(lc_data['redshift']), lc_data['sfr_percentile'])
            add_size_quantities(output_step_loc)
            add_ellipticity_quantities(output_step_loc)
        else:
            copy_columns_interpolation_dust_raw_healpix(gltcs_fname, output_step_loc, index, 
                                                        step, step2, step_a, step2_a, mask1, mask2, 
                                                        index_2to1, lc_a_cc, 
                                                        healpix_pixels, lc_data['healpix_pixel'],
                                                        verbose = verbose,
                                                        short = short, supershort = supershort,
                                                        dust_factors = match_dust_factors, step = step,
                                                        luminosity_factors = match_luminosity_factors,
                                                        library_index = match_library_index,
                                                        node_index = match_node_index, snapshot=snapshot)

            for healpix_pixel in healpix_pixels:
                output_healpix_loc = output_step_loc.replace("${healpix}",str(healpix_pixel))
                lightcone_healpix_fname =lightcone_step_fname.replace("${healpix}", str(healpix_pixel))
                overwrite_columns(lightcone_healpix_fname, output_healpix_loc, ignore_mstar = ignore_mstar,
                                  verbose = verbose, cut_small_galaxies_mass = cut_small_galaxies_mass,
                                  internal_step = internal_file_step, fake_lensing=fake_lensing, step = step, 
                                  healpix = True,
                                  no_shear_steps = no_shear_steps,
                                  healpix_shear_file = healpix_shear_file.replace("${healpix}", str(healpix_pixel)),
                                  snapshot = snapshot,
                                  snapshot_redshift = snapshot_redshift)
                overwrite_host_halo(output_healpix_loc,sod_step_loc,
                                    halo_shape_step_loc, halo_shape_red_step_loc,
                                    verbose=verbose, snapshot=snapshot)
                add_native_umachine(output_healpix_loc, lightcone_healpix_fname, cut_small_galaxies_mass = cut_small_galaxies_mass,
                                    internal_step = internal_file_step,
                                    snapshot=snapshot)
                slct_healpix = lc_data['healpix_pixel'] == healpix_pixel
                add_blackhole_quantities(output_healpix_loc, np.average(lc_data['redshift']), lc_data['sfr_percentile'][slct_healpix])
                add_size_quantities(output_healpix_loc)
                add_ellipticity_quantities(output_healpix_loc)

        if plot:
            if healpix_file:
                output_step_tmp = output_step_loc.replace("${healpix}", str(healpix_pixels[-1]))
                lc_data = select_dic(lc_data, lc_data['healpix_pixel']==healpix_pixels[-1])
            else:
                output_step_tmp = output_step_loc 

            dummy_mask = np.ones(lc_data['redshift'].size,dtype=bool)
            #new_gal_prop,new_mask = construct_gal_prop(output_step_loc, verbose=verbose,mask=dummy_mask)
            new_gal_prop,new_mask = construct_gal_prop(output_step_tmp, verbose=verbose,mask=dummy_mask)
            index = np.arange(lc_data['redshift'].size)
            plt.figure()
            plt.title("Post match")
            # plt.figure()
            # plt.plot(lc_data['clr_gr'], new_gal_prop['clr_gr'], '.', alpha=0.3)
            # plt.figure()
            # plt.plot(lc_data['Mag_r'], new_gal_prop['Mag_r'], '.', alpha=0.3)
            # plt.figure()
            # plt.plot(lc_data['m_star'], new_gal_prop['m_star'], '.', alpha=0.3)
            # plt.figure()
            # plt.title(" org gal prop vs new gal prop")
            # plt.plot(new_gal_prop['m_star'][index], new_gal_prop['m_star'],'.',alpha=0.3)
            mag_bins = (-21,-20,-19)
            plot_differences(lc_data, new_gal_prop, index)
            plot_differences_2d(lc_data, new_gal_prop, index)
            plot_side_by_side(lc_data, new_gal_prop, index)
            plot_mag_r(lc_data, new_gal_prop, index)
            plot_side_by_side(lc_data, new_gal_prop, index)
            plot_clr_mag(lc_data, new_gal_prop, index, mag_bins, 'clr_gr', 'g-r color')
            #plot_clr_mag(lc_data, new_gal_prop, index, mag_bins, 'clr_ri', 'r-i color')
            plot_ri_gr_mag(lc_data, new_gal_prop, index, mag_bins)
        if plot or plot_substep:
            dtk.save_figs('figs/'+param_file_name+"/"+__file__+"/")
            plt.show()
        print("\n=====\ndone. {}".format(time.time()-t0))

    ########################################################
    # Concatenation of the all steps into a single catalog #
    ######################################################## 

    if not(healpix_file):
        output_all = output_fname.replace("${step}","all")
        if not metadata_only:
            combine_step_lc_into_one(output_step_list, output_all)
        add_metadata(gltcs_metadata_ref, output_all, version_major,
                     version_minor, version_minor_minor, param_file =
                     param_file_name, snapshot = snapshot)
    else:
        for healpix_pixel in healpix_pixels:
            healpix_ref = lightcone_fname.replace("${healpix}",str(healpix_pixel))
            output_healpix_loc = output_fname.replace("${healpix}",str(healpix_pixel))
            output_all = output_healpix_loc.replace("${step}","all")
            output_step_list = []
            for i, step in enumerate(steps):
                if i == 0 and not snapshot:
                    continue
                output_step_list.append(output_healpix_loc.replace("${step}", str(step)))
            if not metadata_only:
                combine_step_lc_into_one(output_step_list, output_all, healpix=True)
            add_metadata(gltcs_metadata_ref, output_all, version_major, version_minor, version_minor_minor, healpix_ref = healpix_ref,
                         param_file = param_file_name, 
                         snapshot = snapshot)
        
    print("\n\n========\nALL DONE. Answer correct. \ntime: {:.2f}".format(time.time()-t00))


if __name__ == "__main__":
    lightcone_resample(sys.argv[1])
