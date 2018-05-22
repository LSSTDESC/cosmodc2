import sys
import os
import glob
import argparse
import numpy as np
from os.path import expanduser

home = expanduser("~")
if 'mira-home' in home:
    sys.path.insert(0, '/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages')
sys.path.insert(0, os.path.join(home, 'cosmology/cosmodc2'))
sys.path.insert(0, os.path.join(home, 'cosmology/galsampler/build/lib.linux-x86_64-2.7'))
sys.path.insert(0, os.path.join(home, 'cosmology/halotools/build/lib.linux-x86_64-2.7'))

from cosmodc2.write_umachine_healpix_mock_to_disk import write_umachine_healpix_mock_to_disk
from cosmodc2.get_healpix_cutout_info import get_healpix_cutout_info

parser = argparse.ArgumentParser()

parser.add_argument("healpix_fname",
    help="Filename of healpix cutout to run")
parser.add_argument("commit_hash",
    help="Commit hash to save in output files")
parser.add_argument("-input_master_dirname",
    help="Directory name (relative to home) storing sub-directories of input files",
    default=os.path.join(home, 'cosmology/DC2/LC_Test'))
parser.add_argument("-healpix_cutout_dirname",
    help="Directory name (relative to home) storing healpix cutout files",
    default='healpix_cutouts')
    #default='/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/LC_Test/healpix_cutouts')
parser.add_argument("-um_input_catalogs_dirname",
    help="Directory name (relative to home) storing um input catalogs",
    default='protoDC2_v4_um_sfr_catalogs_and_halos')
parser.add_argument("-output_mock_dirname",
    help="Directory name (relative to home) storing output mock healpix files",
    default='um_healpix_mocks')
    #default='/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/LC_Test/um_healpix_mocks')
parser.add_argument("-pkldirname",
    help="Directory name (relative to home) storing pkl file with snapshot <-> redshift correspondence",
    default='cosmology/cosmodc2/cosmodc2')

parser.add_argument("-verbose",
    help="Turn on extra printing",
        action='store_true', default=False)

args = parser.parse_args()

pkldirname = os.path.join(home, args.pkldirname)
healpix_cutout_dirname = os.path.join(args.input_master_dirname, args.healpix_cutout_dirname)
output_mock_dirname = os.path.join(args.input_master_dirname, args.output_mock_dirname)

#get list of snapshots 
healpix_cutout_fname = os.path.join(healpix_cutout_dirname, args.healpix_fname)
print('Processing healpix cutout {}'.format(healpix_cutout_fname))
healpix_data, redshift_strings, snapshots  = get_healpix_cutout_info(pkldirname, healpix_cutout_fname, sim_name='AlphaQ')
expansion_factors = [1./(1+float(z)) for z in redshift_strings]
if(args.verbose):
    print("target z's and a's:", redshift_strings, expansion_factors)

if len(snapshots) > 0:
    umachine_mstar_ssfr_mock_dirname = (
        os.path.join(args.input_master_dirname, args.um_input_catalogs_dirname))
        #'/projects/DarkUniverse_esp/kovacs/AlphaQ/protoDC2_v4_um_sfr_catalogs_and_halos')
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
        os.path.join(args.input_master_dirname, args.um_input_catalogs_dirname))
        #'/projects/DarkUniverse_esp/kovacs/AlphaQ/protoDC2_v4_um_sfr_catalogs_and_halos')
    umachine_host_halo_basename_list = [sfr_files[n].replace('sfr', 'halo') for n in closest_snapshots]
    umachine_host_halo_fname_list = list(
        (os.path.join(umachine_host_halo_dirname, basename)
         for basename in umachine_host_halo_basename_list))
    if(args.verbose):
        print('umachine_host_halo_fname_list:',umachine_host_halo_basename_list)

    healpix_basename = os.path.basename(args.healpix_fname)
    output_mock_basename = ''.join(["umachine_color_mock_", healpix_basename.replace('_fof_halo_mass', '')])
    output_healpix_mock_fname = os.path.join(output_mock_dirname, output_mock_basename)
    if(args.verbose):
        print('output_healpix_mock_fname:', output_healpix_mock_fname)

    redshift_list = [float(z) for z in redshift_strings]
    commit_hash = args.commit_hash
    Lbox_target_halos = 256.

    write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            healpix_data, snapshots, output_healpix_mock_fname,
            redshift_list, commit_hash, Lbox_target_halos)
else:
    print('Skipping empty healpix-cutout file {}'.format(args.healpix_fname))

          
