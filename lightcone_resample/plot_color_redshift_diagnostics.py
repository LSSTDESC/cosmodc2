#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import dtk 
import h5py
import sys
import time
from numpy.random import normal
from cosmodc2.mock_diagnostics import mean_des_red_sequence_gr_color_vs_redshift, mean_des_red_sequence_ri_color_vs_redshift, mean_des_red_sequence_iz_color_vs_redshift


def plot_colors(fname,z_min,z_max,title,mass_cut):
    hgroup = h5py.File(fname,'r')['galaxyProperties']
    redshift = hgroup['redshift'].value
    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:observed:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:observed:dustAtlas'].value
    #mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    host_mass = hgroup['hostHaloMass'].value
    slct = (z_min < redshift) & (redshift < z_max) & (mass_cut < host_mass)
    r = mag_r[slct]
    gr = mag_g[slct] - mag_r[slct]
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(14,30,250)
    h,xbins,ybins = np.histogram2d(r,gr,bins=(xbins,ybins))
    plt.figure()
    plt.title(title+"\n{:.2f} < z < {:.2f})".format(z_min,z_max))
    #plt.plot(r,gr,'.',alpha=0.3)
    plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    plt.colorbar()
    plt.grid()
    plt.xlabel('mag r');plt.ylabel('g-r color')
    plt.tight_layout()
    
band_filter_frame = '{filter_type}_filters/magnitude:{filter_type}_{band}:{frame}:dustAtlas';
model_band_filter_frame= 'baseDC2/restframe_extincted_sdss_abs_mag{band}'

def get_norm(vals):
    val_max = np.max(np.abs(np.percentile(vals,[0.05,0.95])))                
    val_max = 0.2
    return clr.Normalize(vmin=-val_max, vmax=val_max)


def get_selection(fname, healpix_pixels, title, central_cut=False, 
                  Mr_cut=None, mr_cut = None, 
                  mass_cut=None, rs_cut=False,
                  synthetic=None, ms_cut =None, 
                  synthetic_type = None):
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    slct = (redshift == redshift)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if rs_cut:
        a = get_val(hfiles,'baseDC2/is_on_red_sequence_gr')
        b = get_val(hfiles,'baseDC2/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    if synthetic is not None:
        lc_id = get_val(hfiles,'baseDC2/lightcone_id')
        slct = slct & ((lc_id < 0) == synthetic)
        title = title +'Synth.'
    if synthetic_type is not None:
        halo_id = get_val(hfiles,'baseDC2/halo_id')
        slct = slct & (halo_id == synthetic_type)
        title = title +'halo_id == {}, '.format(synthetic_type) 

    if ms_cut is not None:
        stellar_mass = get_val(hfiles,'totalMassStellar')
        slct = slct & ( stellar_mass > ms_cut)
        title = title + "M* > {:.2e}".format(ms_cut)
    return slct, title
    

def plot_color_z(fname, healpix_pixels, slct, title, filter_type, frame, mag1, mag2,
                 plot_type=None, scatter_color=None,  figsize = None):
    # hgroup = h5py.File(fname,'r')['galaxyProperties']
    # redshift = hgroup['redshift'].value
    # host_mass = hgroup['hostHaloMass'].value
    # mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    # mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    #redshift = get_val(hfiles,'UMachineNative/target_halo_redshift')
    # mag1_val = get_val(hfiles, Mag_trans[mag1].replace('${filter_type}',filter_type))
    # mag2_val = get_val(hfiles, Mag_trans[mag2].replace('${filter_type}',filter_type))
    mag1_val = get_mag(hfiles, filter_type,frame,mag1)
    mag2_val = get_mag(hfiles, filter_type,frame,mag2)
    clr_mag = mag1_val - mag2_val
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(0,1,250)
    print("color_z num: ", np.sum(slct))

    if figsize is None:
        figsize=(7,5)
    plt.figure(figsize=figsize)
    print(plot_type)
    if plot_type == 'hist':
        print(np.min(redshift[slct]), np.max(redshift[slct]))
        print(np.nanmin(clr_mag[slct]), np.nanmax(clr_mag[slct]))
        ybins = np.linspace(-1,2.5,256)
        h,xbins,ybins = np.histogram2d(redshift[slct],clr_mag[slct],bins=(256,ybins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        cb = plt.colorbar()
        cb.set_label('population density')
    elif plot_type == 'scatter':
        cmap = 'coolwarm'
        if scatter_color is not None:
            if scatter_color == 'err Mag_r':
                cat_mag = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = cat_mag - model_mag
                norm = get_norm(colors)
            elif scatter_color == 'err clr g-r rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'g')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magg')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
                norm = get_norm(colors)
            elif scatter_color == 'err clr r-i rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'i')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magi')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
                norm = get_norm(colors)
            elif scatter_color == 'err clr g-r obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'g')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'r')
                model_color = mean_des_red_sequence_gr_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            elif scatter_color == 'err clr r-i obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'r')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'i')
                model_color = mean_des_red_sequence_ri_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            elif scatter_color == 'err clr i-z obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'i')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'z')
                model_color = mean_des_red_sequence_iz_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            else:
                colors = get_val(hfiles,scatter_color)
                cmap='jet'
            sc = plt.scatter(redshift[slct],clr_mag[slct],alpha=.3,c=colors[slct],cmap=cmap,norm=norm,edgecolor='none')
            plt.colorbar(sc,label=scatter_color)
        else:
            plt.plot(redshift[slct],clr_mag[slct],'+',alpha=0.3)
    else:
        raise ValueError
    plt.grid(ls=':')
    plt.xlabel('redshift');plt.ylabel('{} {} {}-{}'.format(frame, filter_type, mag1,mag2))
    plt.title(title)
    plt.tight_layout()


def plot_color_error_z(fname,title,mass_cut,filter_type,mag1, mag2, mag_err1, mag_err2, rs_clr, rs_err = False, central_cut=False):
    hgroup = h5py.File(fname,'r')['galaxyProperties']
    redshift = hgroup['redshift'].value
    host_mass = hgroup['hostHaloMass'].value
    mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    clr_mag = mag1_val - mag2_val
    mag_err1_val = hgroup[Mag_trans[mag_err1].replace('${filter_type}',filter_type)].value
    mag_err2_val = hgroup[um_Mag_trans[mag_err2].replace('${filter_type}',filter_type.lower())].value
    clr_err = mag_err1_val - mag_err2_val
    abs_err = np.max(np.abs(clr_err))
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(0,1,250)
    slct = (mass_cut < host_mass)
    if central_cut:
        central = hgroup['isCentral'].value
        slct = slct & (central == 1)
    print("color_z num: ", np.sum(slct))
    plt.figure()
    plt.scatter(redsfhit, clr_mag, c=clr_err, cmap='coolwarm')
    plt.xlabel('redshift');plt.ylabel('{} {}-{}'.format(filter_type, mag1,mag2))
    title = "{gal} M_halo > {Mh:.2e}"
    if central_cut:
        title=title.format(**{'gal':'centrals galaxies', 'Mh':mass_cut})
    else:
        title=title.format(**{'gal':'member galaxies', 'Mh':mass_cut})
    plt.grid()


def plot_color_mag_z(fname,healpix_pixels, title, mass_cut, filter_type, clr_mag1, clr_mag2, mag, steps,central_cut=True):
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    host_mass = get_val(hfiles, 'hostHaloMass')
    step = get_val(hfiles,'step')
    mag1_val = get_val(hfiles, mag_trans[clr_mag1].replace('${filter_type}',filter_type))
    mag2_val = get_val(hfiles, mag_trans[clr_mag2].replace('${filter_type}',filter_type))
    clr_mag = mag1_val - mag2_val
    mag = get_val(hfiles, mag_trans[clr_mag2].replace('${filter_type}',filter_type))
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(0,1,250)
    slct = (mass_cut < host_mass)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)


def plot_color_color(fname, healpix_pixels, title, filter_type, frame,
                     clr1_mag1, clr1_mag2, clr2_mag1, clr2_mag2,
                     central_cut=False, Mr_cut=None, mr_cut = None, mass_cut=None, rs_cut=False,
                     plot_type=None, scatter_color=None, host_id_cut = False):

    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    host_mass = get_val(hfiles, 'hostHaloMass')
    step = get_val(hfiles,'step')
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    # mag1_val = get_val(hfiles, Mag_trans[mag1].replace('${filter_type}',filter_type))
    # mag2_val = get_val(hfiles, Mag_trans[mag2].replace('${filter_type}',filter_type))
    mag1_val = get_mag(hfiles, filter_type,frame,clr1_mag1)
    mag2_val = get_mag(hfiles, filter_type,frame,clr1_mag2)
    clr1 = mag1_val - mag2_val
    mag1_val = get_mag(hfiles, filter_type,frame,clr2_mag1)
    mag2_val = get_mag(hfiles, filter_type,frame,clr2_mag2)
    clr2 = mag1_val - mag2_val
    slct = (mag1_val == mag1_val)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if rs_cut:
        a = get_val(hfiles,'UMachineNative/is_on_red_sequence_gr')
        b = get_val(hfiles,'UMachineNative/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    if host_id_cut is not None:
        halo_id = get_val(hfiles, 'hostHaloTag')
        slct = slct & (halo_id !=0)
    print("color_z num: ", np.sum(slct))
    print('{} {} {}-{}: min:{} max:{}'.format(frame, filter_type, clr1_mag1, clr1_mag2, np.min(clr1[slct]), np.max(clr1[slct])))
    print('{} {} {}-{}: min:{} max:{}'.format(frame, filter_type, clr2_mag1, clr2_mag2, np.min(clr2[slct]), np.max(clr2[slct])))

    plt.figure(figsize=(7,5))
    if plot_type == 'hist':
        h,xbins,ybins = np.histogram2d(clr1[slct],clr2[slct],bins=(200,200))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.colorbar()
    elif plot_type == 'scatter':
        if scatter_color is not None:
            if scatter_color == 'err Mag_r':
                cat_mag = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = cat_mag - model_mag
            if scatter_color == 'err clr g-r rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'g')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magg')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
            if scatter_color == 'err clr r-i rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'i')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magi')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
            else:
                colors = get_val(hfiles,scatter_color)
            sc = plt.scatter(clr1[slct],clr2[slct],alpha=0.3,c=colors[slct],cmap='jet')
            plt.colorbar(sc,label=scatter_color)
        else:
            plt.plot(clr1[slct],clr2[slct],'+',alpha=0.3)
    else:
        raise ValueError
    plt.grid()
    plt.xlabel('{} {} {}-{}'.format(frame, filter_type, clr1_mag1, clr1_mag2))
    plt.ylabel('{} {} {}-{}'.format(frame, filter_type, clr2_mag1, clr2_mag2))
    plt.title(title)
    plt.tight_layout()
                     

def plot_color_mass(fname,healpix_pixels,title, filter_type, frame, mag1, mag2, redshift_range,
                    central_cut=False, Mr_cut=None, mr_cut = None, mass_cut=None, rs_cut=False,
                    plot_type=None, scatter_color=None):
    # hgroup = h5py.File(fname,'r')['galaxyProperties']
    # redshift = hgroup['redshift'].value
    # host_mass = hgroup['hostHaloMass'].value
    # mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    # mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    log_host_mass = np.log10(get_val(hfiles, 'hostHaloMass'))
    # mag1_val = get_val(hfiles, Mag_trans[mag1].replace('${filter_type}',filter_type))
    # mag2_val = get_val(hfiles, Mag_trans[mag2].replace('${filter_type}',filter_type))
    mag1_val = get_mag(hfiles, filter_type,frame,mag1)
    mag2_val = get_mag(hfiles, filter_type,frame,mag2)
    clr_mag = mag1_val - mag2_val
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(0,1,250)
    slct = (mag1_val == mag1_val)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if rs_cut:
        a = get_val(hfiles,'UMachineNative/is_on_red_sequence_gr')
        b = get_val(hfiles,'UMachineNative/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    slct = (redshift>redshift_range[0]) & (redshift<redshift_range[1])
    
    print("color_z num: ", np.sum(slct))
    plt.figure(figsize=(7,5))
    print(plot_type)
    if plot_type == 'hist':
        h,xbins,ybins = np.histogram2d(log_host_mass[slct],clr_mag[slct],bins=(256,256))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.colorbar()
    elif plot_type == 'scatter':
        cmap = 'coolwarm'
        if scatter_color is not None:
            if scatter_color == 'err Mag_r':
                cat_mag = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = cat_mag - model_mag
                norm = get_norm(colors)
            elif scatter_color == 'err clr g-r rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'g')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magg')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
                norm = get_norm(colors)
            elif scatter_color == 'err clr r-i rest':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'rest', 'r')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'rest', 'i')
                model_mag1 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magr')
                model_mag2 = get_val(hfiles, 'UMachineNative/restframe_extincted_sdss_abs_magi')
                colors = (cat_mag1 - cat_mag2) - (model_mag1 - model_mag2)
                norm = get_norm(colors)
            elif scatter_color == 'err clr g-r obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'g')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'r')
                model_color = mean_des_red_sequence_gr_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            elif scatter_color == 'err clr r-i obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'r')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'i')
                model_color = mean_des_red_sequence_ri_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            elif scatter_color == 'err clr i-z obs':
                cat_mag1 = get_mag(hfiles, 'SDSS', 'obs', 'i')
                cat_mag2 = get_mag(hfiles, 'SDSS', 'obs', 'z')
                model_color = mean_des_red_sequence_iz_color_vs_redshift(redshift)
                colors= (cat_mag1-cat_mag2) - model_color
                norm = get_norm(colors)
            else:
                colors = get_val(hfiles,scatter_color)
                cmap='jet'
            sc = plt.scatter(log_host_mass[slct],clr_mag[slct],alpha=.3,c=colors[slct],cmap=cmap,norm=norm,edgecolor='none')
            plt.colorbar(sc,label=scatter_color)
        else:
            plt.plot(log_host_mass[slct],clr_mag[slct],'+',alpha=0.3)
    else:
        raise ValueError
    plt.grid()
    plt.xlabel('host halo mass');plt.ylabel('{} {} {}-{}'.format(frame, filter_type, mag1,mag2))
    plt.title(title)
    plt.tight_layout()
    

def plot_ellipticity_z(fname,healpix_pixels,title, 
                       central_cut=False, Mr_cut=None, mr_cut = None, mass_cut=None, rs_cut=False,
                       plot_type=None, scatter_color=None,synthetic=None, ms_cut =None,
                       mi_cut = None):
    # hgroup = h5py.File(fname,'r')['galaxyProperties']
    # redshift = hgroup['redshift'].value
    # host_mass = hgroup['hostHaloMass'].value
    # mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    # mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    # mag1_val = get_val(hfiles, Mag_trans[mag1].replace('${filter_type}',filter_type))
    # mag2_val = get_val(hfiles, Mag_trans[mag2].replace('${filter_type}',filter_type))
    slct = (redshift == redshift)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if mi_cut is not None:
        mi = get_mag(hfiles, 'SDSS', 'obs','i')
        slct = slct & (mi < mi_cut)
        title = title+' m_i < {:.1f}'.format(mi_cut)
        
    if rs_cut:
        a = get_val(hfiles,'UMachineNative/is_on_red_sequence_gr')
        b = get_val(hfiles,'UMachineNative/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    if synthetic is not None:
        lc_id = get_val(hfiles,'UMachineNative/lightcone_id')
        slct = slct & ((lc_id < 0) == synthetic)
        title = title +'Synth.'
    if ms_cut is not None:
        stellar_mass = get_val(hfiles,'totalMassStellar')
        slct = slct & ( stellar_mass > ms_cut)

    print("gal num: ", np.sum(slct))
    ellip = get_val(hfiles,'morphology/totalEllipticity')
    ellip_disk = get_val(hfiles,'morphology/diskEllipticity')
    ellip_spheroid = get_val(hfiles,'morphology/spheroidEllipticity')
    
    # Creating figures
    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct],ellip[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.colorbar(label='population density')
    plt.title(title)
    plt.xlabel('redshift')
    plt.ylabel('total ellipticity')
    plt.tight_layout()

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct],ellip_disk[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.colorbar(label='population density')
    plt.title(title)
    plt.xlabel('redshift')
    plt.ylabel('disk ellipticity')
    plt.tight_layout()

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct],ellip_spheroid[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.colorbar(label='population density')
    plt.title(title)
    plt.xlabel('redshift')
    plt.ylabel('bulge ellipticity')
    plt.tight_layout()
    

def plot_size_z(fname,healpix_pixels,slct, title,
                plot_type=None):
    # hgroup = h5py.File(fname,'r')['galaxyProperties']
    # redshift = hgroup['redshift'].value
    # host_mass = hgroup['hostHaloMass'].value
    # mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    # mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    hfiles = get_hfiles(fname, healpix_pixels)
    #redshift = get_val(hfiles, 'redshift')
    redshift = get_val(hfiles, 'baseDC2/redshift')
    print(slct.size, np.sum(slct), np.sum(slct)/slct.size)
    mag_r = get_mag(hfiles, "LSST", "rest", "r")
    shlr = get_val(hfiles,  "morphology/spheroidHalfLightRadius")
    dhlr = get_val(hfiles,  "morphology/spheroidHalfLightRadius")
    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct], mag_r[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    shlr = get_val(hfiles,  "morphology/spheroidHalfLightRadius")[slct]
    dhlr = get_val(hfiles,  "morphology/spheroidHalfLightRadius")[slct]
    mhlr = np.maximum(shlr, dhlr)
    shlr_ac = get_val(hfiles,  "morphology/spheroidHalfLightRadiusArcsec")[slct]
    dhlr_ac = get_val(hfiles,  "morphology/spheroidHalfLightRadiusArcsec")[slct]
    mhlr_ac = np.maximum(shlr_ac, dhlr_ac)

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct], mag_r[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.xlabel('z')
    plt.ylabel('Mag r')

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct], mag_r[slct],bins=100)
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.xlabel('z')
    plt.ylabel('Mag r')

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct], mhlr,bins=(100, np.logspace(-2,1,100)))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    plt.yscale('log')
    plt.xlabel('z')
    plt.ylabel('Size')

    plt.figure()
    h,xbins,ybins = np.histogram2d(redshift[slct], mhlr_ac,bins=(100, np.logspace(-3,2,100)))
    plt.pcolor(xbins, ybins, h.T, cmap='Blues', norm=clr.LogNorm())
    cb = plt.colorbar()
    plt.title(title)
    cb.set_label('Population Density')
    plt.yscale('log')
    plt.xlabel('z')
    plt.ylabel('Size [Arcseconds]')
    
    z_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    plt.figure()
    size_bins = np.logspace(-3,1.5, 100)
    for i in range(0, len(z_bins)-1):
        slct_z = (redshift[slct] > z_bins[i]) & (redshift[slct] < z_bins[i+1])
        h, xbins = np.histogram(mhlr_ac[slct_z], bins = size_bins)
        plt.plot(dtk.bins_avg(xbins), h, c = cm.plasma(i/(len(z_bins)-1)), label="{} < z < {}".format(z_bins[i], z_bins[i+1]))
    plt.legend(loc='best', framealpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('counts')
    plt.xlabel('Galaxy Size [Arcseconds]')
    return


def get_hfiles(fname, healpix_pixels):
    if len(healpix_pixels) == 0:
        healpix_pixels = ['']
    hfiles =[]
    print(fname)
    for healpix_pixel in healpix_pixels:
        print(healpix_pixel)
        if "#z_range#" in fname:
            for z_range in ["0_1", "1_2", "2_3"]:
                ffname = fname.replace('#healpix#',str(healpix_pixel)).replace("#z_range#", z_range)
                hfiles.append(h5py.File(ffname, 'r'))
        else:
            hfiles.append(h5py.File(fname.replace('#healpix#',str(healpix_pixel)),'r'))
    return hfiles


def get_val(hfiles, var_name, remove_nan=None):
    sub_result = []
    for hfile in hfiles:
        sub_result.append(hfile['galaxyProperties/'+var_name].value)
    result = np.concatenate(sub_result)
    if remove_nan is not None:
        result[~np.isfinite(result)]=remove_nan
    return result


def get_mag(hfiles, filter_type, frame, band):
    remove_nan = None
    if frame == 'obs':
        frame = 'observed'
    if filter_type == 'model':
        assert frame=='rest'
        var_name = model_band_filter_frame.format(**{'band': band})
        remove_nan = -14
    else:
        var_name = band_filter_frame.format(**{'filter_type':filter_type,
                                               'frame':frame,
                                               'band':band,})
    print(var_name)
    return get_val(hfiles,var_name, remove_nan = remove_nan)
                                          

def soft_transition(val, trans_start, trans_end):
    from halotools.utils import fuzzy_digitize
    if(trans_start == trans_end):
        return val>trans_start
    elif(trans_start > trans_end):
        raise ValueError('Trans_start value is greater than trans_end')
    else:
        slct_between = (vals>trans_start) & (vals<trans_end)
        bins = fuzzy_digitize(vals[slct_between],[trans_start,trans_end])
        result = np.ones(vals.size, dtype='bool')
        result[val<=trans_start] = False
        result[slct_between] = bins==1
        return result
    

def plot_ra_dec(fname, healpix_pixels, title, 
                 central_cut=False, Mr_cut=None, mr_cut = None, mass_cut=None, rs_cut=False,
                 plot_type=None, scatter_color=None,synthetic=None, ms_cut =None, figsize = None, 
                step_cut = None):
    # hgroup = h5py.File(fname,'r')['galaxyProperties']
    # redshift = hgroup['redshift'].value
    # host_mass = hgroup['hostHaloMass'].value
    # mag1_val = hgroup[mag_trans[mag1].replace('${filter_type}',filter_type)].value
    # mag2_val = hgroup[mag_trans[mag2].replace('${filter_type}',filter_type)].value
    hfiles = get_hfiles(fname, healpix_pixels)
    redshift = get_val(hfiles,'redshift')
    #redshift = get_val(hfiles,'UMachineNative/target_halo_redshift')
    # mag1_val = get_val(hfiles, Mag_trans[mag1].replace('${filter_type}',filter_type))
    # mag2_val = get_val(hfiles, Mag_trans[mag2].replace('${filter_type}',filter_type))
    ybins = np.linspace(-0.5,2,250)
    xbins = np.linspace(0,1,250)
    slct = (redshift == redshift)
    if central_cut:
        central = get_val(hfiles, 'isCentral')
        slct = slct & (central == 1)
        title=title+', central galaxies'
    title=title+'\n'
    if step_cut is not None:
        step = get_val(hfiles, 'step')
        slct = slct & (step == step_cut)
        title = title+"step {}".format(step_cut)
    if mass_cut is not None:
        host_mass = get_val(hfiles, 'hostHaloMass')
        if isinstance(mass_cut, (list,)):
            slct = slct & (mass_cut[0] < host_mass) & (host_mass < mass_cut[1])
            title = title+'{:.0e} <  M_halo < {:.0e}'.format(mass_cut[0],mass_cut[1])
        else:
            slct = slct & (mass_cut < host_mass)
            title = title+'M_halo > {:.0e}'.format(mass_cut)
    if Mr_cut is not None:
        Mr = get_mag(hfiles, 'SDSS', 'rest', 'r')
        if isinstance(Mr_cut, (list,)):
            slct = slct & (Mr_cut[0] < Mr) & (Mr < Mr_cut[1])
            title = title+'  {:.1f} < Mr < {:.1f}'.format(Mr_cut[0],Mr_cut[1])
        else:
            slct = slct & (Mr < Mr_cut)
            title = title+'  Mr < {:.1f}'.format(Mr_cut)
    if mr_cut is not None:
        mr = get_mag(hfiles, 'SDSS', 'obs', 'r')
        if isinstance(mr_cut, (list,)):
            slct = slct & (mr_cut[0] < mr) & (mr < mr_cut[1])
            title = title+'  {:.1f} < mr < {:.1f}'.format(mr_cut[0],mr_cut[1])
        else:
            slct = slct & (mr < mr_cut)
            title = title+'  mr < {:.1f}'.format(mr_cut)
    if rs_cut:
        a = get_val(hfiles,'baseDC2/is_on_red_sequence_gr')
        b = get_val(hfiles,'baseDC2/is_on_red_sequence_ri')
        print(a)
        slct = slct & (a & b)
        title = title+', Red Seq.'
    if synthetic is not None:
        lc_id = get_val(hfiles,'baseDC2/lightcone_id')
        slct = slct & ((lc_id < 0) == synthetic)
        title = title +'Synth.'
    if ms_cut is not None:
        stellar_mass = get_val(hfiles,'totalMassStellar')
        slct = slct & ( stellar_mass > ms_cut)
        title = title + "M* > {:.2e}".format(ms_cut)

        

    ra = get_val(hfiles, 'ra')
    dec = get_val(hfiles, 'dec')
    plt.figure()
    h, xbins, ybins = np.histogram2d(ra[slct], dec[slct], bins=200)
    plt.pcolor(xbins,ybins, h.T)
    plt.title(title)
    plt.tight_layout()


def plot_redshift_distance(fname, healpix_pixels, slct, title):
    hfiles = get_hfiles(fname, healpix_pixels)
    x = get_val(hfiles, "x")[slct]
    y = get_val(hfiles, "y")[slct]
    z = get_val(hfiles, "z")[slct]
    r = np.sqrt(x**2 + y**2 + z**2)
    redshift = get_val(hfiles, 'redshift')[slct]
    redshift_h = get_val(hfiles, 'redshiftHubble')[slct]
    plt.figure()
    plt.plot(redshift, r, ',')
    plt.figure()
    plt.plot(redshift_h, r, ',')
    

if __name__ == "__main__":
    title = sys.argv[1]
    fname = sys.argv[2]
    healpix_pixels = sys.argv[3:]

    #plot_ra_dec(fname, healpix_pixels, title, step_cut = 487, mr_cut = 29)
    # plot_ra_dec(fname, healpix_pixels, title, step_cut = 475, mr_cut = 29)
    # plot_ra_dec(fname, healpix_pixels, title, step_cut = 464, mr_cut = 29)
    # plot_ra_dec(fname, healpix_pixels, title, step_cut = 401, mr_cut = 29)
    plot_ra_dec(fname, healpix_pixels, title, step_cut = 315)
    # plot_ra_dec(fname, healpix_pixels, title, step_cut = 253, mr_cut = 29)



    # plt.show()
    # exit()
    print("healpix_pixels: ", healpix_pixels)
    mass_cut = None
    central_cut = None
    rs_cut = False
    plot_type = 'scatter'
    Mr_cut = None
    mi_cut = 24
    mr_cut = None
    # plot_ellipticity_z(fname,healpix_pixels, title, mass_cut = mass_cut, central_cut = central_cut,
    #                    rs_cut = rs_cut, Mr_cut = Mr_cut,mi_cut = mi_cut)
    figsize = (4,3)
    filter_type = 'LSST'
    scatter_color = None

    frame = 'obs'
    mag1 = 'g'
    mag2 = 'r'
    slct, title_slct = get_selection(fname, healpix_pixels, title, synthetic=True)
    plot_size_z(fname, healpix_pixels, slct, title_slct)
    plot_redshift_distance(fname, healpix_pixels, slct, title_slct)
    # plot_color_mass(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, [0,1], mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color, figsize=figsize)
    # mag1 = 'r'
    # mag2 = 'i'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color, figsize=figsize)
    # frame = 'rest'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # filter_type = 'model'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # mag1 = 'r'
    # mag2 = 'i'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # mag1 = 'i'
    # mag2 = 'z'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='hist',Mr_cut=Mr_cut,scatter_color=scatter_color)

    #########################################
    # Error Plots
    # scatter_color = 'err Mag_r'
    # mag1 = 'i'
    # mag2 = 'z'
    # filter_type='SDSS'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # scatter_color = 'err clr g-r rest'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # scatter_color = 'err clr r-i rest'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # scatter_color = 'err clr g-r obs'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # scatter_color = 'err clr r-i obs'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    # scatter_color = 'err clr i-z obs'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type='scatter',Mr_cut=Mr_cut,scatter_color=scatter_color)
    #########################################
    
    # plot_color_color(fname,healpix_pixels, title, filter_type, frame, 'g', 'r', 'r', 'i', plot_type='hist', host_id_cut = True)
    # plot_color_color(fname,healpix_pixels, title, filter_type, frame, 'r', 'i', 'i', 'z', plot_type='hist', host_id_cut = True)
    # plot_color_color(fname,healpix_pixels, title, filter_type, frame, 'g', 'r', 'i', 'z', plot_type='hist', host_id_cut = True)
    # plot_color_color(fname,healpix_pixels, title, filter_type, frame, 'i', 'z', 'z', 'y', plot_type='hist', host_id_cut = True)
    # plot_color_color(fname,healpix_pixels, title, filter_type, frame, 'g', 'r', 'r', 'i',  mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # mag1 = 'r'
    # mag2 = 'i'
    # Mr_cut = -23
    # plot_type='hist'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut)
    # Mr_cut = [-23,-22]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-22,-21]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-21,-20]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-20,-19]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-19,-7]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)

    # mag1 = 'i'
    # mag2 = 'z'
    # Mr_cut = -23
    # plot_type='hist'
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut)
    # Mr_cut = [-23,-22]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-22,-21]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-21,-20]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-20,-19]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
    # Mr_cut = [-19,-7]
    # plot_color_z(fname,healpix_pixels, title,filter_type, frame, mag1, mag2, mass_cut=mass_cut, central_cut=central_cut, rs_cut=rs_cut,plot_type=plot_type,Mr_cut=Mr_cut,scatter_color=scatter_color)
  

    #################################
    # Basic observed colors
    # mr_cut = None

    frame = 'rest'
    slct, _ = get_selection(fname, healpix_pixels, title)
    plot_color_z(fname, healpix_pixels, slct, title, 'model', frame, 'g', 'r', plot_type='hist')
    # plot_color_z(fname, healpix_pixels, title, 'model', frame, 'r', 'i', plot_type='hist', ms_cut = 1e9)
    # rs_cut = True
    # mass_cut = 1e13
    # frame = 'rest'
    # plot_color_z(fname, healpix_pixels, title, 'model', frame, 'g', 'r', plot_type='hist')
    # plot_color_z(fname, healpix_pixels, title, 'model', frame, 'r', 'i', plot_type='hist')

    # frame = 'rest'
    # plot_color_z(fname, healpix_pixels, title, 'LSST', frame, 'g', 'r', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)
    # plot_color_z(fname, healpix_pixels, title, 'LSST', frame, 'r', 'i', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)

    # plot_color_z(fname, healpix_pixels, title, 'SDSS', frame, 'g', 'r', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)
    # plot_color_z(fname, healpix_pixels, title, 'SDSS', frame, 'r', 'i', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)

    frame = 'obs'
    slct, _ = get_selection(fname, healpix_pixels, title)
    plot_color_z(fname, healpix_pixels, slct, title, 'LSST', frame, 'g', 'r', plot_type='hist')
    # plot_color_z(fname, healpix_pixels, title, 'LSST', frame, 'r', 'i', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)
    # plot_color_z(fname, healpix_pixels, title, 'LSST', frame, 'i', 'z', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)

    # plot_color_z(fname,healpix_pixels, title, 'SDSS', frame, 'g', 'r', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)
    # plot_color_z(fname,healpix_pixels, title, 'SDSS', frame, 'r', 'i', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)
    # plot_color_z(fname,healpix_pixels, title, 'SDSS', frame, 'i', 'z', plot_type='hist', mr_cut = mr_cut, rs_cut=rs_cut, mass_cut = mass_cut)


    # #################################                                 

    # plot_color_z(fname,healpix_pixels, title,mass_cut,'SDSS','rest','r','i', central_cut=True)
    # plot_color_z(fname,healpix_pixels, title,mass_cut,'LSST','rest','r','i', central_cut=True)
    # plot_color_z(fname,healpix_pixels,title,mass_cut,'SDSS','rest','i','z', central_cut=True)
    # plot_color_z(fname,healpix_pixels,title,mass_cut,'LSST','rest','i','z', central_cut=True)

    # plot_color_mag_z(fname,healpix_pixels, title, mass_cut, 'SDSS', 'g','r','r', ['293'],central_cut=True)
    dtk.save_figs(path='figs/'+__file__+"/"+title+"/"+fname.replace("/", "!")+"/")
    plt.show()
    
