import h5py
import re
import os
from collections import defaultdict, OrderedDict
import numpy as np
import pickle
import json
import time
import copy
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from itertools import chain
from itertools import combinations
from astropy.table import Table
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
#display backend
#matplotlib.get_backend()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pylab import *
from matplotlib import rc
#from matplotlib.backends.backend_pdf import PdfPages
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
#my code
import mask_DC2

catfile_cut = '/cosmo/homes/dkorytov/proj/protoDC2/output/ANL_box_v2.1.3_2_mod.hdf5'
catfile = '/cosmo/homes/dkorytov/proj/protoDC2/output/ANL_box_v2.1.3_nocut_mod.hdf5'
pdfdir = '../pdffiles'
pkldir = '../pklfiles'

galaxyProperties = 'galaxyProperties'
galaxyID = 'galaxyID'

sdss_properties = ['obs_sm', 'rmag', 'sdss_petrosian_gr', 'sdss_petrosian_ri']
galacticus_properties_nodust = [['totalMassStellar'],
                                ['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest'],
                                ['SDSS_filters/totalLuminositiesStellar:SDSS_g:rest',
                                 'SDSS_filters/totalLuminositiesStellar:SDSS_r:rest'],
                                ['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest',
                                 'SDSS_filters/totalLuminositiesStellar:SDSS_i:rest'],
                               ]
galacticus_properties = [['totalMassStellar'],
                         ['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest:dustAtlas'],
                         ['SDSS_filters/totalLuminositiesStellar:SDSS_g:rest:dustAtlas', 
                          'SDSS_filters/totalLuminositiesStellar:SDSS_r:rest:dustAtlas'],
                         ['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest:dustAtlas',
                          'SDSS_filters/totalLuminositiesStellar:SDSS_i:rest:dustAtlas'],
                        ]

generic_properties = OrderedDict((
                                 ('M*', {'label':r'$M_*$', 'data_column': 0, 'bins':60, 'label_log':'$\log_{10}M_*$'}),
                                 ('Mr', {'label':r'$M_r$', 'data_column': 1, 'bins':40}),
                                 ('g-r', {'label':'$g-r$', 'data_column': 2, 'bins':50}),
                                 ('r-i', {'label':'$r-i$', 'data_column': 3, 'bins':50}),
                                ))

sdss_info = {'0.0':{
                   'file':'../logsm_9p5_testing_catalog_for_dtk_with_fake_sdss.hdf5',
                   'steps':[499]},
            '0.11':{
                   'file':'../logsm_9p5_testing_catalog_for_dtk_with_fake_sdss.hdf5',
                   #'steps':[464, 453, 442, 432]},
                   'steps':[453]},
            '0.25':{
                   'file':'../logsm_9p5_testing_catalog_for_dtk_with_fake_sdss.hdf5',
                   #'steps':[421, 411, 401, 392]},
                   'steps':[401]},
            }

options_match = {#'basic':{'logm':False, 'norm':False},
                 'log':{'logm':True, 'norm':False, 'dist_type':['raw', 'rescaled']},
                 'norm':{'logm':True, 'norm':True, 'dist_type':['raw', 'unnormed','rescaled']},
                }

def rescale_Mstar(scale_factor, Mstar_match, log_values=True):
    if log_values:
        Mstar_rescaled = Mstar_match + np.log10(scale_factor) 
    return Mstar_rescaled

def rescale_mag(scale_factor, mag_match):

    flux_rescaled = scale_factor*np.power(10, -0.4*mag_match)
    mag_rescaled = -2.5*np.log10(flux_rescaled)
    return mag_rescaled

def extract_Mstar_scale(Mstar_ref, Mstar_match, log_values=True):
    
    if log_values:
        Mstar_scale_factor = np.power(10., Mstar_ref - Mstar_match)
    return Mstar_scale_factor

def extract_mag_scale(mag_ref, mag_match):
    
    return np.power(10., -0.4*(mag_ref - mag_match))

options_rescale = {'rescale':{'0':rescale_Mstar, '1': rescale_mag},
                   'extract':{'0':extract_Mstar_scale, '1':extract_mag_scale}
                  }

#plotting
figx_l = 15
figy_l = 11
default_colors = ['blue', 'r', 'm', 'g', 'navy', 'y', 'purple', 'gray', 'c',\
                  'orange', 'violet', 'coral', 'gold', 'orchid', 'maroon', 'tomato',\
                  'sienna', 'chartreuse', 'firebrick', 'SteelBlue']                        
total_color = 'black'                                                   
default_markers = ['o', 'v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']   


#how to read catalog
def get_galacticus(catfile=catfile, yamlfile=mask_DC2.yamlfile):

    galacticus = h5py.File(catfile, 'r')
    print 'galacticus keys: {}'.format(', '.join(galacticus[galaxyProperties].keys()))
    pklfile =os.path.join(pkldir, os.path.splitext(os.path.split(catfile)[-1])[0]+'_mask_bad.pkl')
    if os.path.exists(pklfile):
        print 'Reading pklfile {}'.format(pklfile)
        mask_bad = pickle.load( open(pklfile, 'rb'))['mask_bad']
    else:    
        selections = mask_DC2.read_selections(yamlfile=yamlfile)
        mask_bad = mask_DC2.mask_cat(galacticus, selections=selections)
        #save
        print 'Saving pklfile {}'.format(pklfile)
        pickle.dump({'mask_bad': mask_bad}, open(pklfile, 'wb'))

    return galacticus, mask_bad                    

def get_sdss_data(sdssfile='', properties=sdss_properties, logm=True):

    try:
        sdss = Table.read(sdssfile, path='data')
        print 'sdss keys: {}\n Assembling array of properties {}'.format(', '.join(sdss.keys()), ', '.join(properties))
        #assemble array of properties
        stacked_array = np.vstack((np.log10(sdss[prop].data) if prop=='obs_sm' and logm else sdss[prop].data for prop in properties)).T
    except:
        print 'Error in reading file {}'.format(sdssfile)
        stacked_array = np.asarray([])

    return stacked_array
  

def get_zmask(galacticus, steps=[]):
    
    mask = np.zeros(len(galacticus[galaxyProperties][galaxyID]), dtype=bool)
    for step in steps:
        mask |= galacticus[galaxyProperties]['step'].value==step

    print 'Found {} matching galaxies for steps {}'.format(np.sum(mask), '+'.join([str(step) for step in steps]))
     
    return mask
                                                           
def get_galacticus_data(galacticus, properties=galacticus_properties, mask=None, logm=True):

    data = OrderedDict()
    mask = mask if mask is not None else np.ones(len(galacticus[galaxyProperties][galaxyID]), dtype=bool)
    for pgroup in properties:
        key = '-'.join(pgroup)
        for n, prop in enumerate(pgroup):
            p_this = -2.5*np.log10(galacticus[galaxyProperties][prop].value[mask]) if 'Luminosities' in prop\
                     else galacticus[galaxyProperties][prop].value[mask]
            if n==0:
                data[key] = p_this
            else: #subtract other props in list
                data[key] -= p_this
                
        #adjust for logm
        if logm and 'MassStellar' in key:
            data[key] = np.log10(data[key])

    #print data.keys()
    stacked_array = np.vstack((data[key] for key in data.keys())).T
                      
    return stacked_array

def get_normalize(galaxy_data, quiet=False):
    normalize={}
    #compute and save remormalizations:
    for n, column_data in enumerate(galaxy_data.T):
        normalize[str(n)] = {'min':np.min(column_data), 'max':np.max(column_data)}
        
    if not quiet:
        print 'Setting up dict for normalizations\n:', json.dumps(normalize, indent=2)
    return normalize


def normalize_data(galaxy_data, normalize={}):

    data = OrderedDict()
    if normalize: 
        for n, column_data in enumerate(galaxy_data.T):
            data[str(n)] = (column_data - normalize[str(n)].get('min'))/(normalize[str(n)].get('max') - normalize[str(n)].get('min'))
            print 'Normalizing column {} data to range {} - {}'.format(n, np.min(data[str(n)]), np.max(data[str(n)]))

    stacked_array = np.vstack((data[key] for key in data.keys())).T
    return stacked_array

def unnormalize_data(galaxy_data, normalize):

    data = OrderedDict()
    if normalize:
        for n, column_data in enumerate(galaxy_data.T):
            data[str(n)] = normalize[str(n)].get('min') + column_data*(normalize[str(n)].get('max') - normalize[str(n)].get('min'))
            print 'Un-normalizing column {} data to range {} - {}'.format(n, np.min(data[str(n)]), np.max(data[str(n)]))

    stacked_array = np.vstack((data[key] for key in data.keys())).T
                      
    return stacked_array

def get_cKDTree(galacticus_data, zkey, opt, speed='slow', read=False, write=False):

    balanced_tree = speed=='slow'
    compact_nodes = speed=='slow'
    fname = os.path.join(pkldir, '{}_build'.format(speed), 'galacticus_tree_{}_{}_{}.pkl'.format(zkey, opt, speed))
    if read and os.path.exists(fname):
        t0 = time.time()
        galacticus_tree = pickle.load(open(fname, 'rb'))
        t1 = time.time()
        print 'Loading pkl file {} (load-time = {})'.format(fname, t1-t0)
    else:
        t0 = time.time()
        galacticus_tree = cKDTree(galacticus_data, balanced_tree=balanced_tree, compact_nodes=compact_nodes)
        t1 = time.time()
        print 'Tree build time = {}'.format(t1-t0)
        if write:
            pickle.dump(galacticus_tree, open(fname, 'wb'))
            print 'Saving pkl file {}'.format(fname)

    return galacticus_tree
        
def rescale_galaxies(sdss_data, matched_data, ref_column=1, rescale_columns=[0,1],\
                     warp_colors=True, warp_columns=[2,3], warp_ref_column=3):

    rescale_factor = options_rescale['extract'].get(str(ref_column))(sdss_data.T[ref_column], matched_data.T[ref_column])
    print 'Rescaling using column {} data with {}'.format(ref_column, str(options_rescale['extract'].get(str(ref_column))))

    #copy dict and rescale column data with specified function
    rescaled_data = {'data':matched_data.copy()}
    for col in rescale_columns:
        rescaled_data['data'].T[col] = options_rescale['rescale'].get(str(col))(rescale_factor, matched_data.T[col])
        print 'Rescaling column {} data with {}'.format(col, str(options_rescale['rescale'].get(str(col))))

    #save rescale factors and columns
    rescaled_data['rescale_factor'] = rescale_factor
    rescaled_data['rescale_columns'] = rescale_columns
    rescaled_data['ref_column'] = ref_column

    #color shift option
    if warp_colors:
        warp_shift = sdss_data.T[warp_ref_column] - matched_data.T[warp_ref_column]
        warp_factor = np.power(10., 0.4*warp_shift) if warp_ref_column==3 else np.power(10., -0.4*warp_shift)
        funcs=[]
        for col in warp_columns:
            if col==warp_ref_column:
                funcs.append('add')
                rescaled_data['data'].T[col] = np.add(matched_data.T[col], warp_shift)
            else:
                funcs.append('subtract')
                rescaled_data['data'].T[col] = np.subtract(matched_data.T[col], warp_shift)

        print 'Warping columns {} with {}'.format(', '.join([str(w) for w in warp_columns]),', '.join(funcs))
        #save
        rescaled_data['warp_shift'] = warp_shift
        rescaled_data['warp_factor'] = warp_factor
        rescaled_data['warp_ref_column'] = warp_ref_column
        rescaled_data['warp_columns'] = warp_columns

    #recompute distances
    distances = get_distances(sdss_data, rescaled_data['data'])

    return rescaled_data, distances


def get_distances(reference_data, matched_data, properties=generic_properties):
    
    distances ={}
    distances['Total'] = np.zeros(len(reference_data.T[0]))
    for n, (col_ref, col_match) in enumerate(zip(reference_data.T, matched_data.T)):
        #get label corresponding to column number
        dkey = next(key for key, value in generic_properties.iteritems() if value.get('data_column')==n)
        distances[dkey]=col_ref - col_match
        distances['Total'] += distances[dkey]**2

    distances['Total'] = np.sqrt(distances['Total'])

    return distances


def make_cat(zkeys=['0.11'], nn=1, sdss_info=sdss_info, catfile=catfile, yamlfile=mask_DC2.yamlfile,\
             sdss_properties=sdss_properties, galacticus_properties=galacticus_properties, options=options_match,\
             rescale_column=0, fast=True, warp_colors=True, warp_columns=[2,3], warp_ref_column=3,\
             read_pkl=True, write_pkl=False, save_pkl=False, read_tree=False, write_tree=False):
    
    if not read_pkl:
        galacticus, mask_bad = get_galacticus(catfile=catfile, yamlfile=yamlfile)
        sdss_data_dict = {}
        galacticus_data_dict = {}
    
    galacticus_tree_dict = {}
    matched_data_dict = {}
    rescaled_data_dict = {}
    matched_stats_dict = {}

    rescaled_label = 'rescale_'+str([d for d in generic_properties if generic_properties[d]['data_column']==rescale_column][0])
    speed = 'fast' if fast else 'slow'
    warp_label = '_warp_'+str([d for d in generic_properties if generic_properties[d]['data_column']==warp_ref_column][0]) if warp_colors else ''

    for zkey in zkeys:
        if not read_pkl:
            zmask = get_zmask(galacticus, steps=sdss_info[zkey].get('steps',[]))
            mask_this = mask_bad & zmask
            print 'Using {} Galacticus galaxies for match'.format(np.sum(mask_this))
        else:
            print 'Reading sdss and galacticus data from pklfiles xxx_data_dict_{}.pkl'.format(str(zkey))
            sdss_data_dict = pickle.load(open(os.path.join(pkldir, 'sdss_data_dict_{}.pkl'.format(str(zkey))),'rb'))
            galacticus_data_dict = pickle.load(open(os.path.join(pkldir, 'galacticus_data_dict_{}.pkl'.format(str(zkey))),'rb'))
                                               
        if read_pkl or any(mask_this):
        #loop over matching options:
            for opt in sorted(options_match.keys()): #'log' before 'norm'
                if not sdss_data_dict or opt not in sdss_data_dict:
                    sdss_data_dict[opt] = get_sdss_data(sdssfile=sdss_info[zkey].get('file',''), properties=sdss_properties,\
                                                    logm=options_match[opt].get('logm',True))
                    #normalize data
                    if options_match[opt].get('norm',False):
                        print 'Normalizing sdss variables'
                        sdss_data_dict[opt] = normalize_data(sdss_data_dict[opt], normalize=normalize)

                normalize = get_normalize(sdss_data_dict['log'])
                plot_ranges = get_normalize(sdss_data_dict[opt], quiet=True) #depend on whether data is normed 

                if not galacticus_data_dict or opt not in galacticus_data_dict:
                    galacticus_data_dict[opt] = get_galacticus_data(galacticus, mask=mask_this, properties=galacticus_properties,\
                                                                logm=options_match[opt].get('logm',True))
                    if options_match[opt].get('norm',False):
                        print 'Normalizing galacticus variables'
                        galacticus_data_dict[opt] = normalize_data(galacticus_data_dict[opt], normalize=normalize)

                galacticus_tree_dict[opt] = get_cKDTree(galacticus_data_dict[opt], zkey, opt, speed=speed, read=read_tree, write=write_tree)
                
                t0 = time.time()
                nn_distances, nn_indices = galacticus_tree_dict[opt].query(sdss_data_dict[opt], k=nn, n_jobs=16)
                print 'Tree query time = {}'.format(time.time()-t0)
                if nn > 1:
                    #random selection from nn nearest neighbors
                    select = np.random.uniform(low=0.,high=float(nn),size=len(nn_indices)).astype(int)                
                    nn_indices = np.asarray([nn_indices[i][select[i]] for i in range(len(select))])
                    nn_distances = np.asarray([nn_distances[i][select[i]] for i in range(len(select))])

                nn_unique, nn_counts = np.unique(nn_indices, return_counts=True)
                matched_data_dict[opt] = {'data': galacticus_data_dict[opt][nn_indices], 'nn_indices':nn_indices}
                distances = get_distances(sdss_data_dict[opt], matched_data_dict[opt]['data']) 
                assert np.allclose(nn_distances, distances.get('Total')), "Distances don't match"

                matched_stats_dict[opt] = {'unique': nn_unique, 'counts': nn_counts, 'distances_raw': distances, 'normalize':normalize}
                if options_match[opt].get('norm',False):
                    #rescale data if normed above
                    print 'Unnormalizing matched variables'
                    matched_data_dict[opt]['data'] = unnormalize_data(matched_data_dict[opt]['data'], normalize)
                    matched_stats_dict[opt]['distances_unnormed'] = get_distances(sdss_data_dict['log'], matched_data_dict[opt]['data'])
                    sdss_key = 'log'
                    plot_ranges = normalize #reset plot_ranges
                else:
                    sdss_key = opt
                rescaled_data_dict[opt], rescaled_distances = rescale_galaxies(sdss_data_dict[sdss_key], matched_data_dict[opt]['data'],\
                                                              ref_column=rescale_column, warp_colors=warp_colors,\
                                                              warp_ref_column=warp_ref_column, warp_columns=warp_columns)
                
                matched_stats_dict[opt]['distances_rescaled'] = rescaled_distances
                
                #make_plots
                plot_distributions(sdss_data_dict[opt], title='UM+SDSS z={} {}'.format(str(zkey), opt),\
                                   pdfid='_'.join(['sdss', zkey, opt]), ranges=plot_ranges)
                plot_distributions(galacticus_data_dict[opt], title='Galacticus z={} {} scales'.format(str(zkey), opt),\
                                   pdfid='_'.join(['galacticus', zkey, opt])) #plot full range of Galacticus data
                plot_distributions(matched_data_dict[opt]['data'], title='{} {} match nn={}'.format(speed, opt, str(nn)),\
                                   pdfid='_'.join(['matched', zkey, 'nn', str(nn), opt, speed]), ranges=plot_ranges)
                plot_distributions(rescaled_data_dict[opt]['data'], title='{} {} match nn={} {}'.format(speed, opt, str(nn), rescaled_label+warp_label),\
                                   pdfid='_'.join([rescaled_label+warp_label, zkey, 'nn', str(nn), opt, speed]), ranges=plot_ranges)
                plot_stats(matched_stats_dict, pdfid='_'.join([rescaled_label+warp_label, zkey, 'nn', str(nn), speed]))


        results = {}
        results['sdss'] = sdss_data_dict
        results['galacticus'] = galacticus_data_dict
        results['matched'] = matched_data_dict
        results['rescaled'] = rescaled_data_dict
        results['matched_stats'] = matched_stats_dict

        #save everything
        if write_pkl or save_pkl:
            for key in results:
                if save_pkl and ('sdss' in key or 'galacticus' in key):
                    fname = os.path.join(pkldir, '_'.join([key, 'data_dict', zkey+'.pkl']))
                elif write_pkl and 'rescaled' in key:
                    fname = os.path.join(pkldir, '_'.join([key, 'data_dict', zkey, rescaled_label, 'nn', str(nn)+'.pkl']))
                elif write_pkl:
                    fname = os.path.join(pkldir, '_'.join([key, 'data_dict', zkey, 'nn', str(nn)+'.pkl']))
                else:
                    continue
                pickle.dump(results[key], open(fname, 'wb'))
                print 'Saving pkl file {}'.format(fname)

        return results

def plot_stats(stats_data, nrows=2, ncolumns=2, pdfid='', usetex=False, dist_cut=10.):

    rc('text', usetex=usetex)
    for key in stats_data:
        fig, ax = plt.subplots(nrows, ncolumns, figsize=(figx_l, figy_l))

        #distances
        for n, (ax_this, dkey) in enumerate(zip_longest(ax.flat, options_match[key].get('dist_type',[]))):
            colors = iter(default_colors)
            if not dkey:
                if n > len(options_match[key].get('dist_type',[])):
                    ax_this.set_visible(False)
                else:
                    counts = stats_data[key]['counts']
                    ax_this.hist(counts, bins=np.logspace(0, np.max(np.log10(counts)),40), color=colors.next(), histtype='step', label='counts')
                    ax_this.set_xscale('log')
                    ax_this.set_yscale('log')
                    ax_this.set_xlabel('Galaxy Counts')
                    ax_this.set_title(key+'-match')
                    ax_this.legend(loc='best', numpoints=1)
            else:
                distances = stats_data[key]['distances_'+dkey]
                for dist in distances:
                    mask = np.abs(distances.get(dist)) < dist_cut
                    if np.sum(mask):
                        if np.sum(mask) < len(distances.get(dist)):
                            print 'Ignoring {} outliers out of total {}'.format(np.sum(mask), len(distances.get(dist)))
                            bins = 250
                        else:
                            bins = 50
                        ax_this.hist(distances.get(dist)[mask], bins=bins, color=colors.next(), histtype='step', label=dist)
                        ax_this.set_title(key+'-match '+dkey)
                        ax_this.set_yscale('log')
                        ax_this.legend(loc='best', numpoints=1)
                    else:
                        print "No distances < {}".format(dist_cut)

        pdffile = os.path.join(pdfdir, '_'.join(['matched_stats',pdfid,key+'.pdf']))
        plt.savefig(pdffile)
        print 'Saving pdf {}'.format(pdffile)
        plt.close(fig)


def plot_distributions(galaxy_data, properties=generic_properties, ncolumns=3, pdfid='', cmap='plasma', title=title, usetex=False, ranges={}):
                      
    rc('text', usetex=usetex)
    #plot all combinations
    plot_variables = [map(str, c) for c in combinations(properties.keys(),2)]
    nplots = len(plot_variables)
    nrows = (nplots+ncolumns-1)//ncolumns                  
    
    fig, ax = plt.subplots(nrows, ncolumns, figsize=(figx_l, figy_l))
    
    for ax_this, keys in zip(ax.flat, plot_variables):
        xdata = galaxy_data.T[properties[keys[0]]['data_column']]
        ydata = galaxy_data.T[properties[keys[-1]]['data_column']]
        ax_this.set_xscale(properties[keys[0]].get('scale', 'linear'))
        ax_this.set_yscale(properties[keys[-1]].get('scale', 'linear'))
        if len(xdata)>0 and len(ydata)>0:
            h = ax_this.hist2d(xdata, ydata, bins=(properties[keys[0]].get('bins', 40), properties[keys[-1]].get('bins', 40)), norm=LogNorm())
            if usetex:
                ax_this.set_xlabel(properties[keys[0]].get('label',keys[0]))
                ax_this.set_ylabel(properties[keys[-1]].get('label',keys[-1]))
            else:
                ax_this.set_xlabel(keys[0])
                ax_this.set_ylabel(keys[-1])
            if ranges:
                ax_this.set_xlim(xmin=ranges.get(str(properties[keys[0]]['data_column'])).get('min'),\
                             xmax=ranges.get(str(properties[keys[0]]['data_column'])).get('max'))
                ax_this.set_ylim(ymin=ranges.get(str(properties[keys[-1]]['data_column'])).get('min'),\
                             ymax=ranges.get(str(properties[keys[-1]]['data_column'])).get('max')) 
            plt.colorbar(h[3], ax=ax_this)
            ax_this.set_title(title)
        else:
            print 'Nothing to plot'

    pdffile = os.path.join(pdfdir, '_'.join(['mstar_color_mag', pdfid+'.pdf']))
    plt.savefig(pdffile)
    print 'Saving pdf {}'.format(pdffile)
    plt.close(fig)
