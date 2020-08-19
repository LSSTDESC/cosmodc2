#!/usr/bin/env python2.7

import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk 
import h5py
import sys
import time
import pdb

from scipy.stats import binned_statistic_2d
from numpy.random import normal

from cosmodc2.mock_diagnostics import mean_des_red_sequence_gr_color_vs_redshift, mean_des_red_sequence_ri_color_vs_redshift, mean_des_red_sequence_iz_color_vs_redshift

def plot_clr_clr(v3_data, key1, key2, title):
    plt.figure()
    bins = np.linspace(-1,2,100)
    h,xbins,ybins = np.histogram2d(gr,ri,bins=(bins,bins))
    plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
    plt.grid()
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.title(title)


def plot_clr_z_dist(protoDC2, y_key, dist_keys):
    fig, axs = plt.subplots(2,2)
    for i, dist_key in enumerate(dist_keys):
        ax = axs[i//2, i%2]
        m, xbins, ybins, _ = binned_statistic_2d(protoDC2['redshift'], protoDC2[y_key],
                                                 protoDC2[dist_key], bins=250)
        m = np.ma.array(m, mask = ~np.isfinite(m))
        absmax = np.nanmax(np.abs(m))
        im = ax.pcolormesh(xbins, ybins, m.T, cmap='coolwarm', vmin =-0.25, vmax= +0.25)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(dist_key)
        ax.set_xlabel('redshift')
        ax.set_ylabel(y_key)
        ax.grid()
        ax.set_title(dist_key)


def load_protoDC2(fname, cut_small_galaixes):
    t1 = time.time()
    print "loading protodc2...",
    hgroup = h5py.File(fname,'r')['galaxyProperties']
    redshift = hgroup['redshiftHubble'].value
    result  = {}
    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:rest:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:rest:dustAtlas'].value
    mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:rest:dustAtlas'].value
    mag_z = hgroup['SDSS_filters/magnitude:SDSS_z:rest:dustAtlas'].value
    result['g-r rest'] = mag_g - mag_r
    result['r-i rest'] = mag_r - mag_i
    result['i-z rest'] = mag_i - mag_z
    result['mag g rest'] = mag_g
    result['mag r rest'] = mag_r 
    result['mag i rest'] = mag_i
    result['mag z rest'] = mag_z

    mag_g = hgroup['SDSS_filters/magnitude:SDSS_g:observed:dustAtlas'].value
    mag_r = hgroup['SDSS_filters/magnitude:SDSS_r:observed:dustAtlas'].value
    mag_i = hgroup['SDSS_filters/magnitude:SDSS_i:observed:dustAtlas'].value
    mag_z = hgroup['SDSS_filters/magnitude:SDSS_z:observed:dustAtlas'].value
    result['mag g obs'] = mag_g
    result['mag r obs'] = mag_r
    result['mag i obs'] = mag_i
    result['mag z obs'] = mag_z
    result['g-r obs'] = mag_g - mag_r
    result['r-i obs'] = mag_r - mag_i
    result['i-z obs'] = mag_i - mag_z
    result['redshift'] = redshift
    result['dust factor'] = hgroup['dustFactor'].value
    result['sm'] = np.log10(hgroup['totalMassStellar'].value)
    um_mag_r = hgroup['UMachineNative/restframe_extincted_sdss_abs_magr'].value
    um_clr_gr = hgroup['UMachineNative/restframe_extincted_sdss_gr'].value
    um_clr_ri = hgroup['UMachineNative/restframe_extincted_sdss_ri'].value
    um_sm = np.log10(hgroup['UMachineNative/obs_sm'].value)
    result['um mag r rest'] = um_mag_r
    result['um g-r rest'] = um_clr_gr
    result['um r-i rest'] = um_clr_ri
    result['um sm'] = um_sm
    red_seq = hgroup['UMachineNative/is_on_red_sequence_gr'].value & hgroup['UMachineNative/is_on_red_sequence_ri'].value 
    host_halo_mvir = hgroup['UMachineNative/host_halo_mvir'].value > 10**13.5
    result['um clstr red seq'] = red_seq & host_halo_mvir
    if cut_small_galaxies == 0:
        print(um_mag_r.size)
        print(result['mag r rest'].size)
        result['dist r'] = result['mag r rest'] -um_mag_r
        result['dist g-r'] = result['g-r rest'] -  um_clr_gr 
        result['dist r-i'] =  result['r-i rest'] - um_clr_ri
        result['dist sm'] =  result['sm'] - um_sm
        result['dist'] = np.sqrt(result['dist r']**2    + result['dist g-r']**2 
                             + result['dist r-i']**2 + result['dist sm']**2)
    print "\n\ttime: ",time.time() - t1
    return result


def load_umachine(fname):
    hfile = h5py.File(fname,'r')
    result = {}
    result['mag r rest']= hfile['restframe_extincted_sdss_abs_magr'].value
    result['g-r'] = hfile['restframe_extincted_sdss_gr'].value
    result['r-i'] = hfile['restframe_extincted_sdss_ri'].value
    result['redshift'] = hfile['redshift'].value
    return result


def append_dics(dics):
    result = {}
    keys = dics[0].keys()
    for key in keys:
        result[key] = []
    for dic in dics:
        for key in keys:
            result[key].append(dic[key])
    for key in keys:
        result[key] = np.concatenate(result[key])
    return result



if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    lightcone_fname = param.get_string("lightcone_fname")
    output_fname = param.get_string("output_fname")
    steps = param.get_int_list("steps")
    cut_small_galaxies = param.get_bool('cut_small_galaxies')
    v1 = param.get_int("version_major")
    v2 = param.get_int("version_minor")
    v3 = param.get_int("version_minor_minor")
    lc_dics = []

    # for i,step in enumerate(steps):
    #     if i == 0:
    #         continue
    #     lc_dics.append(load_umachine(lightcone_fname.replace("${step}",str(step))))
    # lc_data = append_dics(lc_dics)
    
    # plt.figure()
    # h,xbins,ybins = np.histogram2d(lc_data['redshift'],lc_data['g-r'],bins=(250,250))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.ylabel('g-r rest'); plt.xlabel('redshift')
    # plt.title("UMachine+SDSS Light Cone")
    # plt.grid()

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(lc_data['redshift'],lc_data['r-i'],bins=(250,250))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.ylabel('r-i rest'); plt.xlabel('redshift')
    # plt.title("UMachine+SDSS Light Cone")
    # plt.grid()
    # clrclrbins = np.linspace(-.5,1,250)

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(lc_data['g-r'],lc_data['r-i'],bins=(clrclrbins,clrclrbins))
    # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
    # plt.xlabel('g-r rest'); plt.ylabel('r-i rest')
    # plt.title("UMachine+SDSS Light Cone")
    # plt.grid()

    
    protoDC2 = load_protoDC2(output_fname.replace("${step}","all"), cut_small_galaxies)
    bins = np.linspace(-1,3,250)
    zbins = np.linspace(0,1,250)
    magobsbins = np.linspace(14,39,100)
    magrestbins = np.linspace(-30,-12,100)
    clrclrbins= np.linspace(-1,2,250)
    if True:
        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['mag r rest'],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('Mag r rest'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['mag g rest'],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('Mag g rest'); plt.xlabel('redshift')
        plt.grid()
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))

        # obs clr-z plots
        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['g-r obs'],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('g-r observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['r-i obs'],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('r-i observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['i-z obs'],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('i-z observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        # Cluster red sequence clr-z plots
        clr_bins2 = np.linspace(-0.5, 2, 250)
        slct = protoDC2['um clstr red seq']
        plt.figure()
        h,xbins,ybins = np.histogram2d(
            protoDC2['redshift'][slct],
            protoDC2['g-r obs'][slct],
            bins=(zbins,clr_bins2))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.plot(xbins, 
                 mean_des_red_sequence_gr_color_vs_redshift(xbins),'--k',
                 label='DES fit')
        plt.ylabel('g-r observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.legend(loc='best')
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(
            protoDC2['redshift'][slct],
            protoDC2['r-i obs'][slct],
            bins=(zbins,clr_bins2))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.plot(xbins, 
                 mean_des_red_sequence_ri_color_vs_redshift(xbins),'--k',
                 label='DES fit')
        plt.ylabel('r-i observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.legend(loc='best')
        plt.grid()

        plt.figure()
        plt.plot(protoDC2['redshift'][slct],
                 protoDC2['r-i obs'][slct],
                 'x',alpha=0.3)

        plt.plot(xbins, 
                 mean_des_red_sequence_ri_color_vs_redshift(xbins),'--k',
                 label='DES fit')
        plt.ylabel('r-i observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.legend(loc='best')
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(
            protoDC2['redshift'][slct],
            protoDC2['i-z obs'][slct],
            bins=(zbins,clr_bins2))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.plot(xbins, 
                 mean_des_red_sequence_iz_color_vs_redshift(xbins),'--k',
                 label='DES fit')
        plt.ylabel('i-z observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.legend(loc='best')
        plt.grid()
        

        # Clr-clr rest frame

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['g-r rest'],protoDC2['r-i rest'],bins=(clrclrbins,clrclrbins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.xlabel('g-r rest'); plt.ylabel('r-i rest')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['g-r obs'],protoDC2['r-i obs'],bins=(clrclrbins,clrclrbins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.xlabel('g-r observed'); plt.ylabel('r-i observed')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()


        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['mag r rest'], protoDC2['g-r rest'],bins=(magrestbins,clrclrbins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.xlabel('r rest'); plt.ylabel('g-r rest')
        plt.title("ProtoDC2 v{}.{}.{}".format(v1,v2,v3))
        plt.grid()

        # clr-clr obs frame

        slct = protoDC2['mag r rest']<-18
        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['g-r obs'][slct],protoDC2['r-i obs'][slct],bins=(clrclrbins,clrclrbins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.xlabel('g-r observed'); plt.ylabel('r-i observed')
        plt.title("ProtoDC2 v{}.{}.{}\nMr<-18".format(v1,v2,v3))
        plt.grid()

        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'][slct],protoDC2['g-r obs'][slct],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('g-r observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}\nMr<-18".format(v1,v2,v3))
        plt.grid()
        
        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'][slct],protoDC2['r-i obs'][slct],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('r-i observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}\nMr<-18".format(v1,v2,v3))
        plt.grid()
        
        plt.figure()
        h,xbins,ybins = np.histogram2d(protoDC2['redshift'][slct],protoDC2['i-z obs'][slct],bins=(zbins,bins))
        plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        plt.ylabel('i-z observed'); plt.xlabel('redshift')
        plt.title("ProtoDC2 v{}.{}.{}\nMr<-18".format(v1,v2,v3))
        plt.grid()


    # plt.figure()
    # h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['mag i obs'],bins=(zbins,magobsbins))
    # plt.pcolor(xbins,ybins,h.T,cmap="PuBu",norm=clr.LogNorm())
    # plt.colorbar()
    # plt.plot([0,1],[25.3,25.3],'r',label='gold sample')
    # plt.plot([0,1],[26.8, 26.8], 'r--',label='i<26.8')
    # plt.legend(loc='best',framealpha=0.5)
    # plt.xlabel('redshift');plt.ylabel('mag i observed')
    # plt.grid()
    # plt.xlim([0,1])
    # plt.tight_layout()

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['mag r obs'],bins=(zbins,magobsbins))
    # plt.pcolor(xbins,ybins,h.T,cmap="PuBu",norm=clr.LogNorm())
    # plt.colorbar()
    # plt.legend(loc='best',framealpha=0.5)
    # plt.xlabel('redshift');plt.ylabel('mag r observed')
    # plt.grid()
    # plt.xlim([0,1])
    # plt.tight_layout()

    # plt.figure()
    # h,xbins,ybins = np.histogram2d(protoDC2['redshift'],protoDC2['mag g obs'],bins=(zbins,magobsbins))
    # plt.pcolor(xbins,ybins,h.T,cmap="PuBu",norm=clr.LogNorm())
    # plt.colorbar()
    # plt.legend(loc='best',framealpha=0.5)
    # plt.xlabel('redshift');plt.ylabel('mag g observed')
    # plt.grid()
    # plt.xlim([0,1])
    # plt.tight_layout()

    plt.figure()
    h, xbins = np.histogram(protoDC2['redshift'],bins=250)
    plt.plot(dtk.bins_avg(xbins), h, label='total')

    slct = protoDC2['dust factor'] == 1.0
    h, xbins = np.histogram(protoDC2['redshift'][slct], bins=250)
    plt.plot(dtk.bins_avg(xbins), h, label='dust factor = 1')

    slct = protoDC2['dust factor'] == 3.0
    h, xbins = np.histogram(protoDC2['redshift'][slct], bins=250)
    plt.plot(dtk.bins_avg(xbins), h, label='dust factor = 3')

    slct = protoDC2['dust factor'] == 6.0
    h, xbins = np.histogram(protoDC2['redshift'][slct], bins=250)
    plt.plot(dtk.bins_avg(xbins), h, label='dust factor = 6')

    plt.xlabel('redshift')
    plt.ylabel('count')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.grid()
    if cut_small_galaxies == 0:
        plt.figure()
        h, xbins, ybins = np.histogram2d(protoDC2['um mag r rest'], protoDC2['dist r'], bins = 250)
        plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.xlabel('um mag r rest')
        plt.ylabel('distance r (Galaticus - UMachine)')
        
        plt.figure()
        h, xbins, ybins = np.histogram2d(protoDC2['um g-r rest'], protoDC2['dist g-r'], bins = 250)
        plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.xlabel('um g-r rest')
        plt.ylabel('distance g-r (Galaticus - UMachine)')
        
        plt.figure()
        h, xbins, ybins = np.histogram2d(protoDC2['um r-i rest'], protoDC2['dist r-i'], bins = 250)
        plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.xlabel('um r-i rest')
        plt.ylabel('distance r-i (Galaticus - UMachine)')
        
        plt.figure()
        h, xbins, ybins = np.histogram2d(protoDC2['um sm'], protoDC2['dist sm'], bins = 250)
        plt.pcolor(xbins, ybins, h.T, cmap='PuBu', norm=clr.LogNorm())
        plt.xlabel('um sm')
        plt.ylabel('distance sm (Galaticus - UMachine)')

        dists = ['dist r', 'dist sm', 'dist g-r', 'dist r-i']
    plot_clr_z_dist(protoDC2, 'g-r obs', dists)
    plot_clr_z_dist(protoDC2, 'r-i obs', dists)
    plot_clr_z_dist(protoDC2, 'i-z obs', dists)



    if False:
        cmap = mlp.cm.ScalarMappable(cmap='coolwarm')
        cmap.set_clim(vmin=-0.4,vmax=0.4)
        mat = cmap.to_rgba(m.T)
        print(mat.shape)
        alpha, _, _ = np.histogram2d(protoDC2['redshift'], protoDC2['g-r obs'], bins=250)
        alpha = np.log10(alpha)+1.0
        alpha[~np.isfinite(alpha)] = 0.0
        alpha = alpha/np.max(alpha)
        print(alpha)
        print(np.min(alpha), np.max(alpha))
        plt.figure()
        m, xbins, ybins, _ = binned_statistic_2d(protoDC2['redshift'], protoDC2['g-r obs'],
                                                 protoDC2['dist r'], bins=250)
        m = np.ma.array(m, mask = ~np.isfinite(m))
        absmax = np.nanmax(np.abs(m))
        p = plt.pcolormesh(xbins, ybins, m.T, vmin =-0.25, vmax= +0.25)
        plt.savefig("tmp/tmp.png")
        for fc, a in zip(p.get_facecolors(), alpha.flatten()):
            fc[3]=a
            cb = plt.colorbar()
            cb.set_label('dist r')
            plt.xlabel('redshift')
            plt.ylabel('g-r rest')
            
        plt.figure()
        plt.pcolor(xbins,ybins,alpha.T)
    
        plt.figure()
        plt.hist(alpha.flatten(), bins=250)
        plt.figure()
        mat[:,:,3] = alpha.T
        plt.imshow(mat,origin='lower')


    dtk.save_figs("figs/"+sys.argv[1]+"/"+__file__+"/")
    #plt.show()
    
    #pdb.set_trace()
