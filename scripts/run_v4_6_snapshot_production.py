import sys
import os
import glob
import argparse
import numpy as np

sys.path.insert(0, '/gpfs/mira-home/ekovacs/.local/lib/python2.7/site-packages')
sys.path.insert(0, '/gpfs/mira-home/ekovacs/cosmology/cosmodc2')
sys.path.insert(0, '/gpfs/mira-home/ekovacs/cosmology/galsampler/build/lib.linux-x86_64-2.7')
sys.path.insert(0, '/gpfs/mira-home/ekovacs/cosmology/halotools/build/lib.linux-x86_64-2.7')

from cosmodc2.write_umachine_color_mocks_to_disk import write_snapshot_mocks_to_disk
from cosmodc2.get_fof_info import get_fof_info

parser = argparse.ArgumentParser()

parser.add_argument("commit_hash",
    help="Commit hash to save in output files")
parser.add_argument("-nsnap",
    help="Number of snapshots to loop over. Default is 29.",
        default=29, type=int)
parser.add_argument("-verbose",
    help="Turn on extra printing",
        action='store_true', default=False)


args = parser.parse_args()

pkldirname = "/home/ekovacs/cosmology/cosmodc2/cosmodc2"

#get list of simulation halo files
redshift_strings, snapshots, alphaQ_halos_fname_list = get_fof_info(pkldirname, nsnapshot=args.nsnap)
expansion_factors = [1./(1+float(z)) for z in redshift_strings]
if(args.verbose):
    print("target z's and a's:", redshift_strings, expansion_factors)

umachine_mstar_ssfr_mock_dirname = (
    '/projects/DarkUniverse_esp/kovacs/AlphaQ/protoDC2_v4_um_sfr_catalogs_and_halos')
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
    '/projects/DarkUniverse_esp/kovacs/AlphaQ/protoDC2_v4_um_sfr_catalogs_and_halos')
umachine_host_halo_basename_list = [sfr_files[n].replace('sfr', 'halo') for n in closest_snapshots]
umachine_host_halo_fname_list = list(
    (os.path.join(umachine_host_halo_dirname, basename)
    for basename in umachine_host_halo_basename_list))
if(args.verbose):
    print('umachine_host_halo_fname_list:',umachine_host_halo_basename_list)

target_halo_dirname = "/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/b0168/fofp_new"
target_halo_basename_list = [os.path.basename(q) for q in alphaQ_halos_fname_list]
target_halo_fname_list = list(
    (os.path.join(target_halo_dirname, basename)
    for basename in target_halo_basename_list))
if(args.verbose):
    print('target_halo_basename_list:',target_halo_basename_list)

output_mock_dirname = (
        "/projects/DarkUniverse_esp/kovacs/AlphaQ/galsampler_alphaq_outputs_v4_6")

output_mock_basename_list = [''.join(["umachine_color_mock_v4_", t.replace('fofproperties', ''), 'hdf5']) for t in target_halo_basename_list]
output_color_mock_fname_list = list(
    (os.path.join(output_mock_dirname, basename)
    for basename in output_mock_basename_list))
if(args.verbose):
    print('output_mock_basename_list:', output_mock_basename_list)

redshift_list = [float(z) for z in redshift_strings]
commit_hash = args.commit_hash
target_halo_loader = "gio"
Lbox_target_halos = 256.

write_snapshot_mocks_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list,
            redshift_list, commit_hash, target_halo_loader, Lbox_target_halos)
