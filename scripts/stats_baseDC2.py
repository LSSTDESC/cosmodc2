import os
import sys
import re
import glob
import h5py
import numpy as np
import scipy
import re
import pickle
import psutil
from time import time
import argparse
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from astropy.table import Table, vstack
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


hpxdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/baseDC2_5000_v1.1.1/'
rundir = '/gpfs/mira-home/ekovacs/cosmology/DC2/cosmoDC2/OR_BaseDC2_5000_v1.1.1'
pkldir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/pklfiles'
pdfdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/pdffiles'
pixel_template = 'pixels_check_{}.txt'
hpx_template = 'baseDC2_z_*_cutout_{}.hdf5'
samples = ['gal', 'cen', 'sat', 'all', 'syn_cen', 'syn_sat']
logMhalo_bins = [11, 12, 13, 14, 16]
logMstar_bins = [7, 8, 9, 10, 11, 12.5]
delta_z = 0.25 #4 z bins
S_labels = ['Real Galaxies', 'Centrals', 'Satellites', 'All Galaxies', 'Synthetic Centrals',
                 'Synthetic Satellites']

def make_labels(bins, id):
    labels = []
    for blo, bhi in zip(bins[0:-1], bins[1:]):
        labels.append(' ${} \leq {} < {}$'.format(blo, id, bhi))
    
    return labels
Mh_labels = make_labels(logMhalo_bins, '\log_{10}(M_{halo}/M_\odot)')
Ms_labels = make_labels(logMstar_bins, '\log_{10}(M_{*}/M_\odot)')


def get_table(f): 
    t = Table()
    for k in list(f.keys()):
        t[k] = f[k]
    
    return t

def get_data(fh):
    data = Table()
    keys = list(k for k in fh.keys() if k.isdigit())# and k!='247')
    for s in keys:
        t = get_table(fh[s])
        data = vstack([data, t])
    
    return data
     

def get_stats(data, subsamples=samples, z=0, dz=delta_z,
              logMstar_bins=logMstar_bins, logMhalo_bins=logMhalo_bins):
    stats_keys = [k for k in data.keys() if 'lightcone' not in k and '_id' not in k]
    stats = {}
    counts = {}
    masks = {}
    #setup masks
    masks['all'] = np.ones(len(data), dtype=bool)
    masks['gal'] = data['halo_id'] > 0
    masks['syn_cen'] = data['halo_id'] == -20
    masks['syn_sat'] = data['halo_id'] == -1
    masks['cen'] = (data['upid'] == -1) & masks['gal']
    masks['sat'] = (data['upid'] != -1) & masks['gal']
    logMh = np.log10(data['target_halo_mass'])
    logMs = np.log10(data['obs_sm'])
    for s in subsamples:
        smask = masks[s]
        #mass masks
        for Mlo, Mhi in zip(logMhalo_bins[0:-1], logMhalo_bins[1:]):
            masks['{}:{}<=logMhalo<{}'.format(s, Mlo, Mhi)] = (logMh >= Mlo) & (logMh < Mhi) & smask
        for Mlo, Mhi in zip(logMstar_bins[0:-1], logMstar_bins[1:]):
            masks['{}:{}<=logMstar<{}'.format(s, Mlo, Mhi)] = (logMs >= Mlo) & (logMs < Mhi) & smask
        #zmasks
        zlos = np.arange(float(z), float(z+1), dz)
        for zlo in zlos: 
            mkey = '{}:{:.2f}<=z<{:.2f}'.format(s, zlo, zlo+dz)
            if zlo < zlos[-1]:
                masks[mkey] = (data['redshift'] >= zlo) & (data['redshift'] < zlo + dz) & smask
            else:
                masks[mkey] = (data['redshift'] >= zlo) & smask
            
    #counts
    for s in list(masks.keys()):
        counts[s] = np.count_nonzero(masks[s])

    #stats
    for k in stats_keys:
        stats[k] = {}
        for m in list(masks.keys()):
            if counts[m] > 0:
                stats[k]['mean_' + m] = np.mean(data[k][masks[m]])
                stats[k]['std_' + m] = np.std(data[k][masks[m]])
                stats[k]['min_' + m] = np.min(data[k][masks[m]])
                stats[k]['max_' + m] = np.max(data[k][masks[m]])
                stats[k]['med_' + m] = np.median(data[k][masks[m]])
                
    return stats, counts

def pickle_dict(ddict, pklname='counts', pkldir='./', group=0):
    pklfile = os.path.join(pkldir, '{}_grp{}.pkl'.format(pklname, group))
    with open(pklfile, 'wb') as ph:
        pickle.dump(ddict, ph) #use protocol=2 to get python2 readable pkl
        print('Wrote {}'.format(pklfile))
    return

def unpickle_dict(pklname, pkldir='./', group=0):
    pklfile = os.path.join(pkldir, '{}_grp{}.pkl'.format(pklname, group))
    with open(pklfile, 'rb') as ph:
        ddict = pickle.load(ph) #encoding="latin1" to read python2 pkl with python3
        print('Read {}'.format(pklfile))
    return ddict


def check_pixels(group, pkldir='./', pklname='', nfiles=None):
    pfile = os.path.join(rundir, pixel_template.format(group))
    process = psutil.Process(os.getpid())
    with open(pfile, 'r') as ph:
        contents = ph.read()
        lines = contents.splitlines()
        
    print('Read {} files (hpx {}-{}) in group {}'.format(len(lines),
                                                         lines[0],  lines[-1], group))

    stats = {}
    counts = {}
    nfiles = len(lines) if nfiles is None else nfiles
    start = time()
    for hpx in lines[0:nfiles]:
        timei = time()
        hpx_tmp = os.path.join(hpxdir, hpx_template.format(hpx))
        hpx_files = sorted(glob.glob(hpx_tmp))
        print('Processing {} files matching {}'.format(len(hpx_files), hpx))
        ihpx = int(hpx)
        stats[ihpx] = {}
        counts[ihpx] = {}
        for hpx_file in hpx_files:
            z = int(os.path.basename(hpx_file).split('z_')[-1].split('_')[0])
            fh = h5py.File(hpx_file)
            data = get_data(fh)
            fh.close()
            stats[ihpx][z], counts[ihpx][z] = get_stats(data, z=z)
            #print(stats[hpx][z].keys(), hpx, z)

        print('Healpix run time = {0:.2f} minutes'.format((time() - timei)/60.))
        mem = "Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss/1.e9))
        del data
            
    hpx_keys = sorted(counts.keys())
    zkeys = sorted(counts[hpx_keys[0]].keys())
    pickle_dict(counts, pklname=pklname+'counts', group=group, pkldir=pkldir)
    pickle_dict(stats, pklname=pklname+'stats', group=group, pkldir=pkldir)
    print('Total run time = {0:.2f} minutes'.format((time() - start)/60.))
    
    return stats, counts, hpx_keys, zkeys


def get_dicts_from_pkl(pklname, group=0, pkldir='./'):
    stats = unpickle_dict(pklname+'stats', group=group, pkldir=pkldir)
    counts = unpickle_dict(pklname+'counts', group=group, pkldir=pkldir)
    hpx_keys = sorted(counts.keys())
    zkeys = sorted(counts[hpx_keys[0]].keys())
    print(hpx_keys, zkeys)
    return stats, counts, hpx_keys, zkeys

def make_labels(keys, labeltxt, write=False):
    labels = [re.sub(key.split('<=')[-1].split('<')[0], ' '+labeltxt+' ', key) for key in keys]
    labels = [re.sub(':', ': $', l)+'$' for l in labels]
    labels = [re.sub(l.split(':')[0], S_labels[samples.index(l.split(':')[0])], l) for l in labels]
    if write:
        print(labels)
        
    return labels

def check_keys(hpx, stats, counts, zkeys, sample=None, write=False):
    kk = hpx
    stext = sample if sample is not None else ''
    Mh_keys = [k for k in counts[kk][0].keys() if 'Mhalo' in k and stext in k]
    Ms_keys = [k for k in counts[kk][0].keys() if 'Mstar' in k and stext in k]
    dz_keys = [k for z in zkeys for k in counts[kk][z].keys() if 'z' in k and stext in k]
    if write:
        print(stats[kk][0].keys())
        print(counts[kk][0])
        print(Mh_keys, Ms_keys, dz_keys)
        
    return Mh_keys, Ms_keys, dz_keys


def get_data_vector(q, ddict, ddict_keys, z=0, subkey=None):
    values = []
    hpxs = []
    for hpx in ddict_keys:
        if subkey is None:
            if q in ddict[hpx][z].keys():  #check if key exists (non-zero count)
                values.append(ddict[hpx][z][q])
                hpxs.append(hpx)
        elif subkey in ddict[hpx][z][q].keys(): #check if key exists (non-zero count)
            values.append(ddict[hpx][z][q][subkey])
            hpxs.append(hpx)
        
    return np.asarray(values), hpxs


def check_dicts(stats, counts):
    N = get_data_vector('all', counts, hpx_keys)
    Nsyn = get_data_vector('syn_cen', counts, hpx_keys)
    stellar_mass = get_data_vector('obs_sm', stats, hpx_keys, subkey='mean_cen')
    stellar_mass_syn = get_data_vector('obs_sm', stats, hpx_keys, subkey='mean_syn_cen')
    dsm = get_data_vector('obs_sm', stats, hpx_keys, subkey='max_cen')
    dsmy = get_data_vector('obs_sm', stats, hpx_keys, subkey='max_syn_sat')
    print(N[0:10], Nsyn[0:10], stellar_mass[0:10], stellar_mass_syn[0:10], dsm[0:10], dsmy[0:10])


def plot_vector(ax, hpxs, data_vector, label='', color='r', marker='o', 
                h_offset=0., errors=None):
    xpts = np.arange(len(hpxs)) + h_offset
    if errors is None:
        ax.scatter(xpts, data_vector, c=color, label=label, marker=marker)
    else:
        ax.errorbar(xpts, data_vector, yerr=errors, c=color, label=label, ls='')
        
    return

def save_fig(fig, plotdir, figname, title='', hspace=.05, wspace=.2):
    fig.suptitle(title, size=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    fig.subplots_adjust(top=0.94)
    file_type = '.pdf' if 'pdf' in plotdir else '.png'
    figname = os.path.join(plotdir, re.sub('_+', '_', figname) + file_type)
    print('  Saving {}\n'.format(figname))
    fig.savefig(figname, bbox_inches='tight')
    #plt.close(fig)


def plot_ax_z(ax, q, ddict, ddict_keys, zkeys, Nid=None,
              subkeys=None, hpx_label_stride=10, sample_label='', hpx_int=16,
              colors=['r', 'g', 'blue'], log=False, marker='o'):

    for z, c in zip(zkeys, colors):
        if subkeys is None:
            values, hpxs = get_data_vector(q, ddict, ddict_keys, z=z, subkey=subkeys)
            if len(values) > 0:
                # fix legend for plotting by z keys
                lgnd = ' (${}<z<{}$)'.format(z, z+1) if 'z' not in sample_label else sample_label.split(':')[-1]
                sample_label = sample_label.split(':')[0] if 'z' in sample_label else sample_label
                plot_vector(ax, hpxs, values,
                            color=c, label='${}<z<{}$'.format(z, z+1), marker=marker)
        else:
            values = {}
            hpxs = {}
            for subkey in subkeys:
                values[subkey], hpxs[subkey] = get_data_vector(q, ddict, ddict_keys,
                                                                  z=z, subkey=subkey)
            xpts = np.arange(len(values['mean_'+Nid])) #account for missing values in some hpxs
            if len(xpts) > 0:
                lgnd = ' (${}<z<{}$)'.format(z, z+1) if 'z' not in Nid else '' #add z-label if not z-sample
                plot_vector(ax, hpxs['mean_'+Nid], values['mean_'+Nid], h_offset=z*0.25,
                                color=c, label='mean'+lgnd, marker=marker)
                if 'std_'+Nid in subkeys:     
                    ax.fill_between(xpts + z*0.25, values['mean_'+Nid]+values['std_'+Nid],
                                    values['mean_'+Nid]-values['std_'+Nid], alpha=0.4, color=c, 
                                    label='$\pm 1\sigma$'+lgnd)
                if 'min_'+Nid in subkeys and 'max_'+Nid in subkeys:
                    ax.fill_between(xpts + z*0.25, values['min_'+Nid],
                                    values['max_'+Nid], alpha=0.4, color=c, 
                                    label='min-max'+lgnd)
                
    ax.set_xlabel('Healpixel')
    locs = np.arange(0, len(ddict_keys), hpx_int)
    ax.set_xticks(locs)
    xticklabels = [ddict_keys[int(l)] if l >= 0. and l < len(ddict_keys) else '' for l in locs]
    ax.set_xticklabels(xticklabels, rotation = 90)
    ylabel = 'N' if subkeys is None else re.sub('_', ' ', q)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')
    ax.legend(loc='best', numpoints=1, title=sample_label)


figx=15
figy=12
def plot_data(ddict, hpx_keys, zkeys, samples, sample_labels, q=None, group=0,
              marker='o', sharey=False, sharex=True, plotname='',
              fname='{}_grp{}', log=True, plotdir='./pdffiles',
              subkeys=['mean', 'min', 'max']):
    nrows = 2 if len(samples) <= 6 else 3
    ncols = int(np.ceil(len(samples)/nrows))
    fig, ax_all = plt.subplots(nrows, ncols, figsize=(figx, figy), sharey=sharey,
                               sharex=sharex)
    for s, ax, slabel in zip_longest(samples, ax_all.flatten(), sample_labels):
        if s is None:
            if ax is not None:
                ax.set_visible(False)
            continue
        if q is not None:
            nsubkeys = [ '_'.join([k, s]) for k in subkeys]
            plot_ax_z(ax, q, ddict, hpx_keys, zkeys, Nid=s, log=log, marker=marker,
                      subkeys=nsubkeys, sample_label=slabel)
        else:
            plot_ax_z(ax, s, ddict, hpx_keys, zkeys, log=log, marker=marker, subkeys=None,
                      sample_label=slabel)
    
    sblist = '_'.join(subkeys) if subkeys is not None else ''
    slist = '_'.join([sblist, plotname]) if len(plotname)!=0 else sblist
    qname = '_'. join([q, slist]) if q is not None else '_'. join(['N', slist])
    wspace = 0.0 if sharey else .2
    hspace = 0.0 if sharex else .05
    save_fig(fig, plotdir, fname.format(qname, group), hspace=hspace, wspace=wspace)
    
    return fig


def make_plots(counts, stats, hpx_keys, zkeys, group=0, plotdir='./pdffiles'):
    Mh_keys, Ms_keys, Dz_keys = check_keys(hpx_keys[0], stats, counts, zkeys)
    #Mh_labels = make_labels(Mh_keys, '\\\\log_{10}(M_{halo}/M_\\\\odot)')
    #Ms_labels = make_labels(Ms_keys, '\\\\log_{10}(M_{*}/M_\\\\odot)')
    #Dz_labels = make_labels(dz_keys, 'z')
    fig = plot_data(counts, hpx_keys, zkeys, samples, S_labels, group=group, subkeys=None, plotdir=plotdir)
    fig2 = plot_data(stats, hpx_keys, zkeys, samples, S_labels, group=group, q='obs_sm',
                 subkeys=['mean', 'min', 'max'], plotdir=plotdir)    
    fig3 = plot_data(stats, hpx_keys, zkeys, samples, S_labels, group=group, q='target_halo_mass',
                 subkeys=['mean', 'min', 'max'], plotdir=plotdir)
    # by sample
    for s in ['all']:
        ms_keys = [k for k in Ms_keys if s in k and '_'+s not in k]
        mh_keys = [k for k in Mh_keys if s in k and '_'+s not in k]
        dz_keys = [k for k in Dz_keys if s in k and '_'+s not in k]
        ms_labels = make_labels(ms_keys, '\\\\log_{10}(M_{*}/M_\\\\odot)')
        mh_labels = make_labels(mh_keys, '\\\\log_{10}(M_{halo}/M_\\\\odot)')
        dz_labels = make_labels(dz_keys, 'z')
        fig4 = plot_data(stats, hpx_keys, zkeys, ms_keys, ms_labels, group=group,
                         q='target_halo_axis_A_length', plotname=s+'_Mstar',
                         subkeys=['mean', 'min', 'max'], log=False, plotdir=plotdir)
        fig5 = plot_data(stats, hpx_keys, zkeys, mh_keys, mh_labels, group=group,
                         q='target_halo_ellipticity', plotname=s+'_Mhalo',
                         subkeys=['mean', 'std'], log=False, plotdir=plotdir)
        fig6 = plot_data(stats, hpx_keys, zkeys, dz_keys, dz_labels, group=group,
                         q='restframe_extincted_sdss_ri', sharey=True, plotname=s+'_Dz',
                         subkeys=['mean', 'std'], log=False, plotdir=plotdir)
        #counts by slice
        if 'syn' not in s and 'all' not in s:
            fig7 = plot_data(counts, hpx_keys, zkeys, ms_keys, ms_labels, group=group, plotname=s+'_dzslice',
                             subkeys=None, plotdir=pdfdir, sharey=True)
            fig8 = plot_data(counts, hpx_keys, zkeys, mh_keys, mh_labels, group=group, plotname=s+'_msslice',
                             subkeys=None, plotdir=pdfdir, sharey=True)
            fig9 = plot_data(counts, hpx_keys, zkeys, dz_keys, dz_labels, group=group, plotname=s+'_mhslice',
                             subkeys=None, plotdir=pdfdir, sharey=True)

    return

# Example to run
#stats, counts, hpx_keys, zkeys = check_pixels(5, pklname='memtest_')
#print(hpx_keys, zkeys)
#run_plots(counts, stats, hpx_keys, zkeys, dz_keys, group=5)

def main(argsdict):
    pklname = argsdict['pklname']
    group = argsdict['group']
    nfiles = argsdict['nfiles']
    run_check = argsdict['check']
    run_plots = argsdict['plots']
    print('Processing group {} with pklname {}'.format(group, pklname))
    msg = 'all' if nfiles is None else str(nfiles)
    if run_check:
        print('Running check with {} files'.format(msg))
        stats, counts, hpx_keys, zkeys = check_pixels(group, pklname=pklname, pkldir=pkldir, nfiles=nfiles)
    else:
        print('Reading from pkldir {}'.format(pkldir))
        stats, counts, hpx_keys, zkeys = get_dicts_from_pkl(pklname, group=group, pkldir=pkldir)

    if run_plots:
        print('Running plots')
        make_plots(counts, stats, hpx_keys, zkeys, group=group)
    return


def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Gets hpx stats')
    parser.add_argument('--group', type=int, help='file group to run', default=0)
    parser.add_argument('--nfiles', type=int, help='number of files to run (None runs all files in group)', default=None)
    parser.add_argument('--plots', default=False, help='make plots', action='store_true')
    parser.add_argument('--check', default=False, help='make dicts and pkl', action='store_true')
    parser.add_argument('--pklname', help='extra name for pklfiles', default='')
    args=parser.parse_args()
    argsdict=vars(args)
    print ("Running", sys.argv[0], "with parameters:")
    for arg in argsdict.keys():
        print(arg," = ", argsdict[arg])

    return argsdict

if __name__ == '__main__':    
    argsdict=parse_args(sys.argv)
    main(argsdict)
