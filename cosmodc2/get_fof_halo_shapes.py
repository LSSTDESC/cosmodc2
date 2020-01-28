import os
import h5py
import numpy as np
from astropy.table import Table

shape_file_template = 'shapes_{}_l.hdf5'
evalues = 'eigenvalues_SIT_COM'
evectors = 'eigenvectors_SIT_COM'

def get_halo_shapes(snapshot, hpx_fof_tags, hpx_reps, shape_dir, debug=True):
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
    #check for missing steps
    if snapshot == '347':
        fsnap = '338'
    else:
        fsnap = '253' if snapshot == '259' else snapshot
    fn = os.path.join(shape_dir, shape_file_template.format(fsnap))
    if os.path.isfile(fn):
        with h5py.File(fn) as fh:
            fof_tags = fh['fof_halo_tag'].value
            mask = np.in1d(fof_tags, hpx_fof_tags) # duplicates possible
            nfof = np.count_nonzero(mask)
            if nfof > 0:
                reps = fh['replication'].value
                mask &= np.in1d(reps, hpx_reps) # duplicates possible
                #check foftag/replication pairs to verify they are matched
                mask_locations = np.where(mask==True)[0]
                for mloc, foftag, rep  in zip(mask_locations, fof_tags[mask], reps[mask]):
                    locs = np.where(hpx_fof_tags==foftag)[0]
                    found = False
                    n=0
                    while not found and n < len(locs):
                        found = (hpx_reps[locs[n]] == rep)
                        #print(mloc, foftag, rep, hpx_fof_tags[locs[n]], hpx_reps[locs[n]], n, found, locs[n])
                        n += 1

                    mask[mloc] = found

                print('...Matched {} fof & replication tags (/{} fof tags) for snapshot {}'.format(
                    np.count_nonzero(mask), nfof, snapshot))
                for k, v in fh.items():
                    if 'RIT' not in k and k[-3:] != 'SIT':
                        shapes[k] = v.value[mask]
    else:
        if debug:
            print('...Skipping {} (not found)'.format(fn))


    return shapes

def get_locations(shapes, fof_halo_tags, replications):
    # searchsorted returns location of first occurrence and fails for multiple occurrences
    #orig_indices = target_halos['fof_halo_id'].argsort()
    #insertions = np.searchsorted(target_halos['fof_halo_id'][orig_indices], shapes['fof_halo_tag'])
    #locations = orig_indices[insertions]

    # find position of fof_halo_tags in target_halo array
    locations = []
    for foftag, rep in zip(shapes['fof_halo_tag'], shapes['replication']):
        loc = np.where(fof_halo_tags==foftag)[0]
        if len(loc) > 1:
            idx = np.where(replications[loc]==rep)[0]
            if len(idx) == 1:
                locations.append(loc[idx[0]])
            else:  #duplicate halo
                print('Warning: duplicate entries for fof_tag {} rep {}'.format(foftag, rep))
                locations.append(loc[idx[0]])
        elif len(loc) == 1:
            locations.append(loc[0])
        else:
            print('Error: entry not found for fof_tag {}'.format(foftag))

    return np.asarray(locations)

def get_matched_shapes(shapes, target_halos, check_positions=False, Lbox=3000.):
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
    locations = get_locations(shapes, target_halos['fof_halo_id'], target_halos['rep'])
    assert np.array_equal(target_halos['fof_halo_id'][locations], shapes['fof_halo_tag']), "Fof tag arrays don't match"
    assert np.array_equal(target_halos['rep'][locations], shapes['replication']), "Replication mismatch"
    
    # get axis lengths, convert to ratios and compute ellipticity and prolaticity
    # see code in triaxial_satellite_distributions/axis_ratio_model.py for definitions
    # reorder eigenvalues
    reorder = shapes[evalues].argsort()
    nvals = len(shapes[evalues])
    ordered_evals = np.asarray([shapes[evalues][i][reorder[i]] for i in range(nvals)])
    a = np.sqrt(ordered_evals[:, 2])
    b = np.sqrt(ordered_evals[:, 1])
    c = np.sqrt(ordered_evals[:, 0])
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

    # save direction of major axis; note we transpose the evectors so that xyz components are in rows
    ordered_evecs = np.asarray([shapes[evectors][i].T[reorder[i]] for i in range(nvals)])
    major_axis_evectors = ordered_evecs[:, 2]  #select evector corresponding to largest evalue
    # check that normalization is correct
    norms = np.asarray([np.dot(major_axis_evectors[i], major_axis_evectors[i].T) for i in range(nvals)])
    assert all(np.isclose(norms, 1)), "Major-axis eigenvector has incorrect norm"

    # save axis vector; z direction is already flipped to cosmoDC2 coordinates
    target_halos['axis_A_x'][locations] = major_axis_evectors[:,0]
    target_halos['axis_A_y'][locations] = major_axis_evectors[:,1]
    target_halos['axis_A_z'][locations] = major_axis_evectors[:,2]

    # check positions
    if check_positions:
        for q in ['x', 'y', 'z']:
            if 'z' in q: # flip sign of z component since positions were not flipped
                dq = np.mod(target_halos[q][locations] + shapes['c'+q].flatten(), Lbox)
            else:
                dq = np.mod(target_halos[q][locations] - shapes['c'+q].flatten(), Lbox)
            mask = abs(dq) > Lbox/2
            dq[mask] = dq[mask] - Lbox
            print('...Min/max for |d{}| = {:.2g}/{:.2g}:'.format(q, np.min(dq), np.max(dq)))
 
    return target_halos


new_col_names = ('axis_A_length', 'axis_B_length', 'axis_C_length',
                 'halo_ellipticity','halo_prolaticity',
                 'axis_A_x', 'axis_A_y','axis_A_z')
def get_halo_table(file_handle): #read hpx file into astropy table for testing
    t = Table()
    for k in file_handle.keys():
        t[k] = file_handle[k]

    t.rename_column('id', 'fof_halo_id')
    #add test columns
    for k in new_col_names:
        t[k] = np.zeros(len(t['fof_halo_id']))

    return t


def run_shapes(h5, shape_dir):  #for testing
    shapes = {}
    for k, v in h5.items(): 
        fof_tags = v['id'].value
        reps = v['rep'].value
        # match on fof tags and replication values
        shapes[k] = get_halo_shapes(k, fof_tags, reps, shape_dir, debug=True)
        
    return shapes

# example files for testing
#healpix_file = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/healpix_cutouts/z_2_3/cutout_9554.hdf5'
#shape_dir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/OR_haloshapes'

def run_check(healpix_file, shape_dir): #for testing
    fh = h5py.File(healpix_file)

    shapes = run_shapes(fh, shape_dir)
    for snapshot in fh.keys():
        target_halos = get_halo_table(fh[snapshot])
        if shapes[snapshot]:
            print("Processing {}".format(snapshot))
            target_halos = get_matched_shapes(shapes[snapshot], target_halos, check_positions=True)
        else:
            print("Skipping: no shape information for {}".format(snapshot))
