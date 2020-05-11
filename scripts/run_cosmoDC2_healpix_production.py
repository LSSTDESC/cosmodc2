import sys
import os
import glob
import argparse
import numpy as np
from os.path import expanduser
import subprocess

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

from cosmodc2.write_umachine_healpix_mock_to_disk import write_umachine_healpix_mock_to_disk
from cosmodc2.get_healpix_cutout_info import get_healpix_cutout_info

parser = argparse.ArgumentParser()

parser.add_argument("healpix_fname",
    help="Filename of healpix cutout to run")
#parser.add_argument("commit_hash",
#    help="Commit hash to save in output files")
parser.add_argument("-input_master_dirname",
    help="Directory name (relative to home) storing sub-directories of input files",
    default='cosmology/DC2/OR_Test')
parser.add_argument("-healpix_cutout_dirname",
    help="Directory name (relative to input_master_dirname) storing healpix cutout files",
    default='healpix_cutouts')
parser.add_argument("-um_input_catalogs_dirname",
    help="Directory name (relative to input_master_dirname) storing um input catalogs",
    default='um_snapshots')
parser.add_argument("-output_mock_dirname",
    help="Directory name (relative to input_master_dirname) storing output mock healpix files",
    default='baseDC2_healpix_mocks')
parser.add_argument("-shape_dirname",
    help="Directory name (relative to input_master_dirname) storing halo shape files",
    default='OR_haloshapes')
parser.add_argument("-pkldirname",
    help="Directory name (relative to home) storing pkl file with snapshot <-> redshift correspondence",
    default='cosmology/cosmodc2/cosmodc2')
parser.add_argument("-zrange_value",
    help="z-range to run",
    choices=['0', '1', '2', 'all'],
    default='all')                
parser.add_argument("-synthetic_mass_min",
    help="Value of minimum halo mass for synthetic halos",
                    type=float, default=9.8)
parser.add_argument("-use_satellites",
    help="Use satellite synthetic low-mass galaxies",
        action='store_true', default=False)
parser.add_argument("-verbose",
    help="Turn on extra printing",
        action='store_true', default=False)
parser.add_argument("-ndebug_snaps",
    help="Number of debug snapshots to save",
                    type=int, default=-1)
parser.add_argument("-gaussian_smearing",
    help="Value of gaussian_smearing_real_redshifts",
                    type=float, default=0.)
parser.add_argument("-nzdivs",
    help="Number of sub-steps for CAM color assignment",
                    type=int, default=6)
parser.add_argument("-nside",
    help="Nside used to create healpixels",
                    type=int, default=32)
        
args = parser.parse_args()

#setup directory names
input_master_dirname = os.path.join(home, args.input_master_dirname)
pkldirname = os.path.join(home, args.pkldirname)
healpix_cutout_dirname = os.path.join(input_master_dirname, args.healpix_cutout_dirname)
output_mock_dirname = os.path.join(input_master_dirname, args.output_mock_dirname)
shape_dir = os.path.join(input_master_dirname, args.shape_dirname)

print('Setting master directory to {}'.format(input_master_dirname))
print('Reading inputs from {}'.format(healpix_cutout_dirname))
print('Writing outputs to {}'.format(output_mock_dirname))

commit_hash = retrieve_commit_hash(path_to_cosmodc2)[0:7]
print('Using commit hash {}'.format(commit_hash))
synthetic_halo_minimum_mass = args.synthetic_mass_min
use_centrals = not(args.use_satellites)

if args.verbose:
        print("paths=", home, path_to_cosmodc2, sys.path)

#loop over z-ranges
if args.zrange_value == 'all':
    z_range_dirs = [os.path.basename(d) for d in glob.glob(healpix_cutout_dirname+'/*') if 'z' in d]
else:
    z_range_dirs = [os.path.basename(d) for d in glob.glob(healpix_cutout_dirname+'/z_{}*'.format(args.zrange_value))]


for zdir in z_range_dirs:
    
    #get list of snapshots 
    healpix_cutout_fname = os.path.join(healpix_cutout_dirname, zdir, args.healpix_fname)
    print('Processing healpix cutout {}'.format(healpix_cutout_fname))
    healpix_data, redshift_strings, snapshots, z2ts = get_healpix_cutout_info(pkldirname,
                                                                              healpix_cutout_fname, sim_name='AlphaQ')

    if args.ndebug_snaps > 0:
        
        redshift_strings = redshift_strings[-args.ndebug_snaps:]
        snapshots = snapshots[-args.ndebug_snaps:]
    expansion_factors = [1./(1+float(z)) for z in redshift_strings]
    if args.verbose:
        print("target z's and a's:", redshift_strings, expansion_factors)

    if len(snapshots) > 0:
        umachine_mstar_ssfr_mock_dirname = (
            os.path.join(input_master_dirname, args.um_input_catalogs_dirname))
        sfr_files = sorted([os.path.basename(f) for f in glob.glob(umachine_mstar_ssfr_mock_dirname+'/sfr*')])
        um_expansion_factors = np.asarray([float(f.split('sfr_catalog_')[-1].split('_value_added.hdf5')[0]) for f in sfr_files])
        closest_snapshots = [np.abs(um_expansion_factors - a).argmin() for a in expansion_factors]
        if(args.verbose):
            print('index of closest snapshots:',closest_snapshots)
    
        umachine_mstar_ssfr_mock_basename_list = [sfr_files[n] for n in closest_snapshots]
        umachine_mstar_ssfr_mock_fname_list = list(
            (os.path.join(umachine_mstar_ssfr_mock_dirname, basename)
             for basename in umachine_mstar_ssfr_mock_basename_list))
        if(args.verbose):
            print('umachine_mstar_ssfr_mock_basename_list:',umachine_mstar_ssfr_mock_basename_list)

        umachine_host_halo_dirname = (
            os.path.join(input_master_dirname, args.um_input_catalogs_dirname))
        umachine_host_halo_basename_list = [sfr_files[n].replace('sfr', 'halo') for n in closest_snapshots]
        umachine_host_halo_fname_list = list(
            (os.path.join(umachine_host_halo_dirname, basename)
             for basename in umachine_host_halo_basename_list))
        if(args.verbose):
            print('umachine_host_halo_fname_list:',umachine_host_halo_basename_list)

        healpix_basename = os.path.basename(args.healpix_fname)
        output_mock_basename = ''.join(["base5000_", zdir, '_', healpix_basename.replace('_fof_halo_mass', '')])
        output_healpix_mock_fname = os.path.join(output_mock_dirname, output_mock_basename)
        if(args.verbose):
            print('output_healpix_mock_fname:', output_healpix_mock_fname)

        redshift_list = [float(z) for z in redshift_strings]

        write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            healpix_data, snapshots, output_healpix_mock_fname, shape_dir,
            redshift_list, commit_hash, synthetic_halo_minimum_mass=synthetic_halo_minimum_mass,
            use_centrals=use_centrals, gaussian_smearing_real_redshifts=args.gaussian_smearing, 
            nzdivs=args.nzdivs, Nside_cosmoDC2=args.nside, z2ts=z2ts)

    else:
        print('Skipping empty healpix-cutout file {}'.format(args.healpix_fname))
