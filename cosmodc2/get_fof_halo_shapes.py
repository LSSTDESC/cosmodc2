import os
import h5py
import numpy as np
from astropy.table import Table

shape_file_template = 'shapes_{}_l.hdf5'
evalues = 'eigenvalues_SIT'
evectors = 'eigenvectors_SIT'

def get_halo_shapes(snapshot, hpx_fof_tags, shape_dir, debug=False):
    """
    read file with halo shapes and return shape data for matches
    Parameters
    ----------
    snapshot:  current snapshot
    hpx_fof_tags:  fof halo tags in current healpix snapshot
    shape_dir: directory containing halo shape files
    Returns
    -------
    shapes: output dict containing shape information for matching halo tags
    """
    shapes={}
    fn = os.path.join(shape_dir, shape_file_template.format(snapshot))
    if os.path.isfile(fn):
        with h5py.File(fn) as fh:
            fof_tags = fh['fof_halo_tag'].value
            mask = np.in1d(fof_tags, hpx_fof_tags)
            if np.count_nonzero(mask) > 0:
                print('{} matched for snapshot {}'.format(np.count_nonzero(mask),
                                                      snapshot))
                for k, v in fh.items():
                    if 'RIT' and 'COM' not in k:
                        shapes[k] = v.value[mask]
    else:
        if debug:
            print('Skipping {} (not found)'.format(fn))


    return shapes

def get_matched_shapes(shapes, target_halos):
    """
    modify array of target halo shape information to include 
    host halo shape information if available
    Parameters
    ----------
    shapes: dict of available shape information
    target_halos: astropy table of target halo information

    Returns
    -------
    target_halos: modified table
    """
    # find position of fof_halo_tags in target_halo array
    orig_indices = target_halos['fof_halo_id'].argsort()
    insertions = np.searchsorted(target_halos['fof_halo_id'][orig_indices],
                                 shapes['fof_halo_tag'])
    locations = orig_indices[insertions]
    source = target_halos['fof_halo_id'][locations]
    assert np.array_equal(source, shapes['fof_halo_tag']), "Fof tag arrays don't match"
    
    # get axis lengths, convert to ratios and compute ellipticity and prolaticity
    # see code in triaxial_satellite_distributions/axis_ratio_model.py for definitions
    # may have to reorder
    a = np.sqrt(shapes[evalues][:,0])
    b = np.sqrt(shapes[evalues][:,1])
    c = np.sqrt(shapes[evalues][:,2])
    b_to_a = b/a
    c_to_a = c/a
    s = 1. + b_to_a**2 + c_to_a**2
    e = (1. - c_to_a**2)/2./s
    p = (1. - 2*b_to_a**2 + c_to_a**2)/2./s

    target_halos['axis_A_length'][locations] = a
    target_halos['axis_B_length'][locations] = b
    target_halos['axis_C_length'][locations] = c
    target_halos['halo_ellipticity'][locations] = e
    target_halos['halo_prolaticity'][locations] = p

    # save direction of major axis noting that the z axis must be flipped for the correct octant
    major_axis_evectors = shapes[evectors][:, 0]  #select first evector
    
    assert np.array_equal(target_halos['rep'][locations], shapes['replication']), "Replication mismatch"
    target_halos['axis_A_x'][locations] = major_axis_evectors[:,0]
    target_halos['axis_A_y'][locations] = major_axis_evectors[:,1]
    target_halos['axis_A_z'][locations] = -major_axis_evectors[:,2]

    # check positions
    for q in ['x', 'y', 'z']:
        if 'z' not in q:
            dq  = np.abs(target_halos[q][locations] - shapes['c'+q])
        else:
            dq  = np.abs(target_halos[q][locations] + shapes['c'+q]) #need negative of z position
        print('Min/max for |d{}| = {:.2g}/{:.2g}:'.format(q, np.min(dq), np.max(dq)))
 
    return target_halos


new_col_names = ('axis_A_length', 'axis_B_length', 'axis_C_length',
                 'halo_ellipticity','halo_prolaticity',
                 'axis_A_x', 'axis_A_y','axis_A_z')
def get_halo_table(file_handle):
    t = Table()
    for k in file_handle.keys():
        t[k] = file_handle[k]

    t.rename_column('id', 'fof_halo_id')
    #add test columns
    for k in new_col_names:
        t[k] = np.zeros(len(t['fof_halo_id']))

    return t


def run_shapes(h5, shape_dir):
    shapes = {}
    for k, v in h5.items():
        fof_tags = v['id'].value
        shapes[k] = get_halo_shapes(k, fof_tags, shape_dir, debug=True)
        
    return shapes

healpix_file = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/healpix_cutouts/z_2_3/cutout_9554.hdf5'
shape_dir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/OR_haloshapes'

def run_check(healpix_file, shape_dir):
    fh = h5py.File(healpix_file)

    shapes = run_shapes(fh, shape_dir)
    for snapshot in fh.keys():
        target_halos = get_halo_table(fh[snapshot])
        if shapes[snapshot]:
            print("Processing {}".format(snapshot))
            target_halos = get_matched_shapes(shapes[snapshot], target_halos)
        else:
            print("Skipping: no shape information for {}".format(snapshot))
