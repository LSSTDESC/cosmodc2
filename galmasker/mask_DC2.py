import os
import h5py
import yaml
import re
import fnmatch                                                                    
from collections import defaultdict, OrderedDict                                  
import numpy as np
try:                                                                                   
    from itertools import zip_longest                                                  
except ImportError:                                                                    
    from itertools import izip_longest as zip_longest
from itertools import chain

phoenix_file = '/cosmo/homes/dkorytov/proj/protoDC2/output/ANL_box_v2.1.3_2_mod.hdf5'
lowz_lib = '/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/low_z/galaxy_library'
hiz_lib = '/gpfs/mira-fs0/projects/DarkUniverse_esp/dkorytov/data/Galacticus/high_z/galaxy_library'
yamlfile = '../yaml/vet_protoDC2.yaml'

#globals
galaxyProperties = 'galaxyProperties'

func_checks = OrderedDict((
        ('isclose', np.isclose),
    ))                              

value_checks = ['min', 'max']

def flux_to_mag(flux):
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = -2.5*np.log10(flux)
        return mag

derived_quantities = OrderedDict((
        ('flux_to_mag', flux_to_mag),
    ))

def read_library(timestep, lowz=True):
    catfile=os.path.join(lowz_lib,'{}_mod.hdf5'.format(timestep))
    catalog = h5py.File(catfile, 'r')
    return catalog

def read_selections(yamlfile=yamlfile):
    
    with open(yamlfile) as f:
        selections =  yaml.safe_load(f)
    assert all('quantities' in d for d in selections['quantities_to_check']), 'yaml file not correctly specified'
    weights = list(chain.from_iterable([(d.get('weights',[])) for d in  selections['quantities_to_check']]))
    assert all([type(w)==str or type(w)==float or type(w)==int for w in weights]), 'yaml file weights not correctly specified'

    return selections

def mask_cat(catalog, selections={}):

    quantities = list(chain.from_iterable([(d.get('quantities',[''])) for d in  selections['quantities_to_check']]))
    assert all(q in catalog[galaxyProperties] for q in quantities), 'Not all quantities available in catalog'
    weights = list(chain.from_iterable([(d.get('weights',[])) for d in  selections['quantities_to_check']]))
    assert all(w in catalog[galaxyProperties] for w in weights if type(w)==str), 'Not all weights available in catalog'

    mask_len = len(catalog[galaxyProperties]['galaxyID'])
    mask = np.ones(mask_len, dtype=bool)
    print "Setting up mask for {} entries".format(mask_len)
    
    for qdict in selections['quantities_to_check']:
        print 'Vetting quantities: {}'.format(qdict['quantities'])
        checktype = qdict['function'] if 'function' in qdict.keys() else 'values' 
        if qdict.get('weights',None):
            checktype = '{} with weights {}'.format(checktype, str(qdict.get('weights')))
        print 'Removing any outliers in {}'.format(checktype) 

        catalog_data = {}
        key = qdict.get('label','_'.join(qdict['quantities']))
        if qdict.get('group_start_index',[]) and len(qdict.get('group_start_index'))>1:
            group_start_index = qdict.get('group_start_index')
            group_end_index = [group_start_index[g] + group_start_index[g+1] for g in range(len(group_start_index)-1)]
            group_end_index.append(len(qdict.get('weights'))) #must exist
            group = 0 #initialize group count
            print 'Using weighted quantities in {} groups with lengths {}'.format(len(group_start_index),
                                             ' '.join([str(group_end_index[g]-group_start_index[g]) for g in range(len(group_start_index))]))
            grouped_data = {}
            grouped_sum = {} #not needed yet
            grouped = True
        else:
            grouped = False
        #check for quantities and weights
        for n, (q, w) in enumerate(zip_longest(qdict['quantities'], qdict.get('weights',[None]))):
            q_this = catalog[galaxyProperties][q].value
            #check for derived quantities
            if qdict.get('derived','') in derived_quantities.keys():
                print 'Deriving {} from {}'.format(qdict.get('derived'), q)
                q_this = derived_quantities[qdict.get('derived')](q_this)
            if w:
                w_this = catalog[galaxyProperties][w].value if type(w)==str else w 
                q_this = np.multiply(w_this, q_this)
            if grouped: #will skip option for weighted sums - not needed yet
                if n >= group_end_index[group]:
                    group += 1
                if n == group_start_index[group]:
                    grouped_data[str(group)] = q_this
                else:
                    grouped_data[str(group)] += q_this
            else:
                if n==0:
                    catalog_data[key] = q_this 
                    wsum_this = w_this if w and type(w)==str else np.zeros(len(q_this))
                else: 
                    if w and type(w)==str:
                        wsum_this = np.add(wsum_this, w_this)
                    #implement just sum for now
                    if 'sum' in qdict.get('function',''):
                        catalog_data[key] += q_this

        #check for group weighting and post_processing
        if grouped:
            if 'quotient' in qdict.get('post_process', None):
                catalog_data[key] = grouped_data['0']/grouped_data['1']
                wsum_this = np.zeros(len(catalog_data[key]))
            else:
                print 'Warning: no post_processing function for grouped data'

        #normalize if needed
        if any(wsum_this):
            catalog_data[key] = catalog_data[key]/wsum_this

        mask_ok = np.isfinite(catalog_data[key])
        print "Rejecting {} infinite/nan values (fraction = {:.4g})".format(np.sum(~mask_ok), float(np.sum(~mask_ok))/mask_len) 

        #compute checks for each quantity group
        for c in value_checks:
            if qdict.get(c, None) is not None: #explicit test needed in case value is 0
                if c=='min':
                    mask_notok = catalog_data[key] < qdict.get(c)
                elif c=='max':
                    mask_notok = catalog_data[key] > qdict.get(c)
                print "Rejecting {} values failing {} cut = {} (fraction = {:.4g})".format(np.sum(mask_notok), c, qdict.get(c), float(np.sum(mask_notok))/mask_len)
                mask_ok &= ~mask_notok

        #checks for values accounting for machine precision
        for c, func in func_checks.items():
            if qdict.get(c,None) is not None: #(watch out for zeros!)
                #isclose
                if  c=='isclose':
                    #mask_notok = func(catalog_data[key], np.array([qdict[c]]*len(catalog_data[key]))) #create array of values, single value misses cases
                    mask_notok = func(catalog_data[key], qdict[c])  #use scalar value
                print "Rejecting {} values failing {} cut = {} (fraction = {:.4g})".format(np.sum(mask_notok), c, qdict.get(c), float(np.sum(mask_notok))/mask_len)    
                mask_ok &= ~mask_notok

        print 'Total number accepted after {} cuts = {}\n'.format(key, np.sum(mask_ok))

        #update global mask
        mask &= mask_ok

    print 'Total number accepted (& of all cuts) = {} (fraction = {:.4g})\n'.format(np.sum(mask), float(np.sum(mask))/mask_len)

    return mask
                

#how to read catalog
#catalog = h5py.File(catfile, 'r')
