import sys
import os
import glob
import argparse
import pickle
import re
import numpy as np
from os.path import expanduser
import subprocess

#halo_snapshot_fname = '03_31_2018.OR.{}.fofproperties#{}'
sim_name='AlphaQ'
pklname='{}_z2ts.pkl'

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


home = expanduser("~")
path_to_cosmodc2 = os.path.join(home, 'cosmology/cosmodc2')
if 'mira-home' in home:
    sys.path.insert(0, '/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages')
sys.path.insert(0, path_to_cosmodc2)
sys.path.insert(0, os.path.join(home, 'cosmology/galsampler/build/lib.linux-x86_64-2.7'))
sys.path.insert(0, os.path.join(home, 'cosmology/halotools/build/lib.linux-x86_64-2.7'))

from cosmodc2.write_umachine_snapshot_mock_to_disk import write_umachine_snapshot_mock_to_disk
                                            
parser = argparse.ArgumentParser()

parser.add_argument("timesteps", nargs='+', metavar='timesteps', default=['247'],
                    choices=['499', '475', '464', '453', '442', '432', '421', '411', '401', '392', '382', '373',
                             '365', '355', '347', '338', '331', '323', '315', '307', '300', '293', '286', '279',
                             '272', '266', '259', '253', '247', '241', '235', '230', '224', '219', '213', '208',
                             '203', '198', '194', '189', '184', '180', '176', '171', '167', '163', '159', '155',
                             '151', '148', '144', '141', '137', '134', '131', '127', '124'],
                    help="Timesteps of snapshots to run; choose from: {%(choices)s}")
#parser.add_argument("commit_hash",
#    help="Commit hash to save in output files")
parser.add_argument("-input_master_dirname",
    help="Directory name (relative to home) storing sub-directories of input and output files",
    default='cosmology/DC2/OR_Snapshots')
parser.add_argument("-input_halo_catalog_dirname",
    help="Directory name (relative to input_master_dirname) storing halo-catalog files",
    default='OR_HaloCatalog')
parser.add_argument("-input_halo_catalog_filename",
    help="Filename for halo-catalog files",
    default='03_31_2018.OR.{}.fofproperties')
parser.add_argument("-um_input_catalogs_dirname",
    help="Directory name (relative to input_master_dirname) storing um input catalogs",
    default='um_snapshots')
parser.add_argument("-output_mock_dirname",
    help="Directory name (relative to input_master_dirname) storing output snapshot files",
    default='baseDC2_snapshots')
parser.add_argument("-pkldirname",
    help="Directory name (relative to home) storing pkl file with snapshot <-> redshift correspondence",
    default='cosmology/cosmodc2/cosmodc2')
parser.add_argument("-blocks",
                    help="block(s) to run [0-255] (eg. 0 1 10-20 200-255 or combinations)", 
                    nargs='+', default='0')                
parser.add_argument("-verbose",
    help="Turn on extra printing",
        action='store_true', default=False)
        
args = parser.parse_args()

#setup directory names
input_master_dirname = os.path.join(home, args.input_master_dirname)
pkldirname = os.path.join(home, args.pkldirname)
input_halo_catalog_dirname = os.path.join(input_master_dirname, args.input_halo_catalog_dirname)
output_mock_dirname = os.path.join(input_master_dirname, args.output_mock_dirname)
timesteps = args.timesteps

commit_hash = retrieve_commit_hash(path_to_cosmodc2)[0:7]
print('Using commit hash {}'.format(commit_hash))

if args.verbose:
    print("paths=", home, path_to_cosmodc2, sys.path)

#determine blocks to be processed
blocks = []
for block in args.blocks:
    if '-' in block:
        r_min, r_max = re.split('-', block)
        rvals = [str(r) for r in range(int(r_min), int(r_max) + 1)]
        blocks = blocks + rvals
    else:
        blocks.append(str(block))
if args.verbose:
    print('Processing blocks {}'.format(', '.join(blocks)))
    

#determine snapshot inputs required
z2ts = pickle.load(open(os.path.join(pkldirname, pklname.format(sim_name)),'rb'))
redshift_strings = [key for key in sorted(z2ts.keys()) if str(z2ts[key]) in timesteps]
redshift_list = [float(z) for z in redshift_strings]
expansion_factors = [1./(1+float(z)) for z in redshift_strings]
if args.verbose:
    print("target z's and a's:", redshift_strings, expansion_factors)

#determine UM inputs available and match them to requested timesteps
umachine_mstar_ssfr_mock_dirname = os.path.join(input_master_dirname, args.um_input_catalogs_dirname)
sfr_files = sorted([os.path.basename(f) for f in glob.glob(umachine_mstar_ssfr_mock_dirname+'/sfr*')])
um_expansion_factors = np.asarray([float(f.split('sfr_catalog_')[-1].split('_value_added.hdf5')[0]) for f in sfr_files])
closest_snapshots = [np.abs(um_expansion_factors - a).argmin() for a in expansion_factors]
if(args.verbose):
    print('index of closest snapshots:',closest_snapshots)
umachine_mstar_ssfr_mock_basename_list = [sfr_files[n] for n in closest_snapshots]
umachine_mstar_ssfr_mock_fname_list = list((os.path.join(umachine_mstar_ssfr_mock_dirname, basename)
                                            for basename in umachine_mstar_ssfr_mock_basename_list))
if args.verbose:
    print('umachine_mstar_ssfr_mock_basename_list:',umachine_mstar_ssfr_mock_basename_list)
umachine_host_halo_dirname = os.path.join(input_master_dirname, args.um_input_catalogs_dirname)
umachine_host_halo_basename_list = [sfr_files[n].replace('sfr', 'halo') for n in closest_snapshots]
umachine_host_halo_fname_list = list((os.path.join(umachine_host_halo_dirname, basename) 
                                      for basename in umachine_host_halo_basename_list))
if args.verbose:
    print('umachine_host_halo_fname_list:',umachine_host_halo_basename_list)

#loop over requested timesteps
for timestep, redshift, um_mstar_ssfr_fname, um_host_halo_fname in zip(timesteps, redshift_list, 
                                                            umachine_mstar_ssfr_mock_fname_list, 
                                                            umachine_host_halo_fname_list):
    #get list of input files for requested blocks 
    input_halo_catalog_fname = os.path.join(input_halo_catalog_dirname,
                                            args.input_halo_catalog_filename.format(timestep))

    print('Processing halo snapshot file {}, blocks {}'.format(input_halo_catalog_fname, ', '.join(blocks)))
    output_snapshot_mock_fname_list = list((os.path.join(output_mock_dirname, 'STEP'+ timestep,
                                                         '_'.join(["baseDC2",'Step'+timestep, 'z'+str(redshift),
                                                                   '#'+block+'.hdf5']))
                                            for block in blocks))
    
    if args.verbose:
        print('output_snapshot_mock_fname_list: {}'.format(', '.join(output_snapshot_mock_fname_list)))

    write_umachine_snapshot_mock_to_disk(
        um_mstar_ssfr_fname, um_host_halo_fname,
        input_halo_catalog_fname, timestep, blocks, output_snapshot_mock_fname_list,
        redshift, commit_hash)
