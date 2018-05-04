import h5py
import os
import numpy as np
import get_fof_info
import load_gio_halos
import glob
#import sys
#sys.path.insert(0,'/gpfs/mira-home/ekovacs/cosmology/cosmodc2/cosmodc2')
pkldirname = '/gpfs/mira-home/ekovacs/cosmology/cosmodc2/cosmodc2/'
cutout_dirname = '/projects/DarkUniverse_esp/prlarsen/healpix_cutouts/'
bitmask = 0x0000FFFFFFFFFFFF
output_dirname = '/projects/DarkUniverse_esp/kovacs/LC_Test/healpix_cutouts'

def process_cutouts(nstart=0, nend=-1, cutout_dirname=cutout_dirname):

    error_log = {}
    cutout_fnames = get_cutout_fnames(cutout_dirname=cutout_dirname)
    if nend > 0:
        cutout_fnames = cutout_fnames[0:n]
    if nstart > 0:
        cutout_fnames = cutout_fnames[nstart:len(cutout_fnames)+1]

    for cutout_file in cutout_fnames:
        error_log[os.path.basename(cutout_file)] = get_fof_halo_masses(cutout_file)

    return error_log

def get_cutout_fnames(cutout_dirname=cutout_dirname):
    return glob.glob(os.path.join(cutout_dirname,'*.hdf5'))


def get_fof_halo_masses(cutout_filename):
    
    print('Processing file {}\n'.format(cutout_filename))
    #h5file = os.path.join(cutout_dirname, cutout_filename)
    cutout = h5py.File(cutout_filename, 'r')

    #cutouts may not have consecutive snapshots; need all fof steps
    redshift_strings, snapshots, fof_halos_fname_list = get_fof_info.get_fof_info(pkldirname, sim_name='LC_Test')
    
    fof_masses = {}
    error_log = {}
    for k in sorted(cutout.keys())[::-1]:
        halo_lc_ids = cutout[k]['id'].value
        #fix fragment ids
        mask = halo_lc_ids < 0
        if  np.sum(mask) > 0:
            print('Converting {} fragment halo-ids in step {}'.format(np.sum(mask), k))
            halo_lc_ids[mask] = -halo_lc_ids[mask] & bitmask 
        #get fof properties file
        index =  snapshots.index(int(k))
        filename = fof_halos_fname_list[index - 1]
        print('Matching lc cutout at step {} with fof halos in {}'.format(k, os.path.basename(filename)))
        halo_table = load_gio_halos.load_gio_halo_snapshot(filename, all_properties=False)
        sort_indices = halo_table['fof_halo_tag'].quantity.value.argsort()
        match_indices = sort_indices[np.searchsorted(halo_table['fof_halo_tag'].quantity.value[sort_indices], halo_lc_ids)]
        fof_masses[k] = halo_table['fof_halo_mass'].quantity.value[match_indices]
        #check tags match
        fof_tags = halo_table['fof_halo_tag'].quantity.value[match_indices]
        print('Step {}, halo_lc_ids match fof tags: {}\n'.format(k, np.array_equal(fof_tags, halo_lc_ids)))
        if not np.array_equal(fof_tags, halo_lc_ids):
            error_log[k] = np.setdiff1d(halo_lc_ids, fof_tags)
    
    write_cutout(cutout, cutout_filename, label='fof_halo_mass', new_data=fof_masses, output_dir=output_dirname)
    cutout.close()

    return error_log

def write_cutout(cutout, cutout_filename, label='', new_data={}, output_dir=output_dirname):
    output_filename = os.path.join(output_dirname, os.path.basename(cutout_filename).replace('cutout', '_'.join(['cutout', label])))
    print('Writing {}\n'.format(output_filename))

    hdfFile=h5py.File(output_filename, 'w')
    #copy contents of cutout file to output
    for k in cutout.keys():
        hdfFile.copy(cutout[k], k)

    #create new datasets
    for k, v in new_data.items():
        #create dataset for group k in output file
        dataset = hdfFile[k].create_dataset(label, (len(v),), dtype='f')
        dataset.write_direct(v)

    hdfFile.close()
