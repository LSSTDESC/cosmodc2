import sys
import os
import glob
import argparse
import numpy as np
from os.path import expanduser
import h5py
import pickle
from time import time
import subprocess

velocity_bound=10000.

def retrieve_commit_hash(path_to_repo):
    """ Return the commit hash of the git branch currently live in the input path.
    Parameters
    ----------
    path_to_repo : string
    Returns
    -------
    commit_hash : string
    """
    cmd = 'cd {0} && git rev-parse HEAD'.format(path_to_repo)
    return subprocess.check_output(cmd, shell=True).strip()


def velocity_bug_fix(output_snapshot, scalefactor=1.0):
    """
    output_snapshot: dict containing mock data to be modified
    scale: correction factor for velocities
    """
    #overwrite with corrected values
    for d in ['x', 'y', 'z']:
        print(".....Correcting v{}'s".format(d))
        corrected_halo_velocity = output_snapshot['target_halo_v{}'.format(d)]*scalefactor
        host_centric_v = output_snapshot['host_centric_v{}'.format(d)]
        corrected_galaxy_velocity = corrected_halo_velocity + host_centric_v
        output_snapshot['target_halo_v{}'.format(d)] = corrected_halo_velocity
        output_snapshot['v{}'.format(d)] = corrected_galaxy_velocity

    return output_snapshot


def mask_large_velocities(output_snapshot, max_value=velocity_bound):

    total = len(output_snapshot['target_halo_vx'])
    mask = np.ones(total, dtype=bool)
    for d in ['x', 'y', 'z']:
        mask &= np.abs(output_snapshot['target_halo_v{}'.format(d)]) < max_value 
    
    nbad = np.sum(~mask)
    print('.....Masking {} galaxy(ies); total = {}; fraction = {}'.format(nbad, total, nbad/float(total)))

    return mask


def apply_mask(output_snapshot, mask):

    for k in output_snapshot.keys():
        output_snapshot[k] = output_snapshot[k][mask]

    return output_snapshot


def healpix_mock_modify(healpix_filename, commit_hash, functions=None, correction_data=None):
    
    output_mock = {}
    masks_used = {}
    print('Starting correction of {}'.format(os.path.basename(healpix_filename)))
    with h5py.File(healpix_filename, 'r') as fh:
        print('...copying input to output_mock')
        #copy input mock to output mock
        for (k, v) in fh.items():
            if k.isdigit():
                output_snapshot = {}
                for kk, vv  in v.items():
                    output_snapshot[kk] = vv.value

                output_mock[k] = output_snapshot
                
        print('...Keys copied to output mock: {}'.format(', '.join(output_mock.keys())))

        for f in functions:
            corrections = correction_data.get(str(f),{})
            for (k, v) in output_mock.items():
                if k.isdigit():
                    if len(v) == 0:
                        print('Skipping empty snap {}'.format(k))
                        continue
                    print('...Processing snap {} with {} and data-correction value(s) {}'.format(k, str(f), corrections[int(k)]))
                    output_mock[k] = f(v, corrections[int(k)])

                    #apply masks
                    mask = mask_large_velocities(output_mock[k], max_value=velocity_bound)
                    output_mock[k] = apply_mask(output_mock[k], mask)
                    print('...Masked length of arrays in snapshot {} = {}'.format(k, len(output_mock[k]['galaxy_id'])))
                    masks_used['large_velocities'] =  mask_large_velocities

                    del mask

        # copy and correct metaData
        k = 'metaData'
        output_mock[k] = {}
        for tk, v in fh[k].items():
            output_mock[k][tk] = v.value

        output_mock[k]['versionMinorMinor'] += 1
        for n, f in enumerate(functions):
            ckey = 'comment_'+str(n)
            output_mock[k][ckey] = ' '.join(['Corrected with', str(f)])
            print('...Adding metaData comment: {}'.format(output_mock[k][ckey]))

        for tk, v in masks_used.items():
            ckey = 'mask_'+tk
            output_mock[k][ckey] = ' '.join(['Corrected with', str(v)])
            print('...Adding metaData comment: {}'.format(output_mock[k][ckey]))

        output_mock[k]['current_commit_hash'] = commit_hash

    return output_mock


def write_output_mock(output_mock, output_healpix_file):
    hdfFile = h5py.File(output_healpix_file, 'w')

    for k, v in output_mock.items():
        gGroup = hdfFile.create_group(k)
        for tk in v.keys():
            gGroup[tk] = v[tk]

    hdfFile.close()
    print('...Wrote {} to disk'.format(output_healpix_file))
    return


home = expanduser("~")
path_to_cosmodc2 = os.path.join(home, 'cosmology/cosmodc2')
if 'mira-home' in home:
    sys.path.insert(0, '/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages')
sys.path.insert(0, path_to_cosmodc2)

parser = argparse.ArgumentParser()
parser.add_argument("-cutout",
    help="healpix cutout number to modify",
    default='*')
parser.add_argument("-healpix_fname_template",
    help="Template filename of healpix cutout to modify",
    default='baseDC2*cutout_{}.hdf5')
parser.add_argument("-input_master_dirname",
    help="Directory name (relative to home) storing input and output healpix file directories",
    default='cosmology/DC2/OR_Production')
parser.add_argument("-output_mock_dirname",
    help="Directory name (relative to input_master_dirname) storing output mock healpix files",
    default='baseDC2_9.8C_v1.1_velocity_bug_fixes')
#    default='baseDC2_min_9.8_centrals_v0.4.7_velocity_bug_fixes')
parser.add_argument("-input_mock_dirname",
    help="Directory name (relative to input_master_dirname) storing input mock healpix files",
    default='baseDC2_9.8C_v1.1')
#    default='baseDC2_min_9.8_centrals_v0.4.5')
#    default='baseDC2_min_9.8_centrals_v0.4.5_test')
parser.add_argument("-modify_functions",
    help="Functions applied to modify input -> output",
    nargs='+', choices=[velocity_bug_fix],
    default=[velocity_bug_fix])

args = parser.parse_args()
                    
#setup directory names
input_master_dirname = os.path.join(home, args.input_master_dirname)
input_mock_dirname = os.path.join(input_master_dirname, args.input_mock_dirname)
output_mock_dirname = os.path.join(input_master_dirname, args.output_mock_dirname)
healpix_filename = args.healpix_fname_template.format(args.cutout)
function_list = args.modify_functions

print('Reading input from {}\n'.format(input_mock_dirname))
print('Writing output to {}\n'.format(output_mock_dirname))
print('Modifying healpix files matching {} with function(s) {}\n'.format(healpix_filename, function_list))

current_commit_hash = retrieve_commit_hash(path_to_cosmodc2)[0:7]
print('Using current commit hash {}'.format(current_commit_hash))

#load additional data needed for corrections
correction_data = {}
function_names = map(str, function_list)
for f in map(str, function_list):
    if 'velocity_bug_fix' in f:
        datfile = os.path.join(path_to_cosmodc2, 'scripts/velocity_correction_factors.pkl')
        #text file option
        #datfile = os.path.join(path_to_cosmodc2, 'scripts/velocity_correction_factors.txt')
        #ts, sf = np.loadtxt(datfile, unpack=True, usecols=[0, 1])
        #correction_data[f] =dict(zip(ts.astype(int), sf))
        with open(datfile, 'rb') as handle:
            correction_data[f] = pickle.load(handle)
        print('Using correction data input from {}\n'.format(datfile))    

healpix_files = sorted(glob.glob(os.path.join(input_mock_dirname, healpix_filename)))
start_time = time()

for hpx in healpix_files:
    start_file = time()
    output_mock = healpix_mock_modify(hpx, current_commit_hash, functions=function_list, correction_data=correction_data)
    output_healpix_file = os.path.join(output_mock_dirname, os.path.basename(hpx))
    write_output_mock(output_mock, output_healpix_file)
    end_file = time()
    print('Processed {} in {:.2f} minutes\n'.format(os.path.basename(output_healpix_file), (end_file - start_file)/60.))
    
time_stamp = time()
msg = "End-to-end runtime = {0:.2f} minutes\n"
print(msg.format((time_stamp-start_time)/60.))

