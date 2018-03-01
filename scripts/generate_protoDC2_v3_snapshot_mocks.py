"""
"""
import os
from cosmodc2.generate_snapshot_collection import get_filename_lists_of_protoDC2
from cosmodc2.generate_snapshot_collection import write_snapshot_mocks_to_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pkldirname",
    help="Absolute path to pickle file storing snapnum<-->redshift correspondence")
parser.add_argument("halocat_dirname",
    help="Absolute path to directory storing Bolshoi-Planck halo catalogs")
parser.add_argument("um_dirname",
    help="Absolute path to directory storing UniverseMachine galaxy catalogs")
parser.add_argument("umachine_z0p1_color_mock_fname",
    help="Absolute path to file storing UniverseMachine z=0.1 baseline color mock")
parser.add_argument("output_mocks_dirname",
    help="Absolute path to directory storing the output mocks")
parser.add_argument("commit_hash",
    help="Commit hash to save in output files")        
parser.add_argument("-nsnap",
    help="Number of snapshots to loop over. Default is all.",
        default=-1, type=int)

args = parser.parse_args()


_x = get_filename_lists_of_protoDC2(args.pkldirname,
            args.halocat_dirname, args.um_dirname)
(alphaQ_halos_fname_list, umachine_mstar_ssfr_mock_fname_list,
    bpl_halos_fname_list, output_color_mock_basename_list, redshift_list) = _x

output_color_mock_fname_list = list(
    os.path.join(args.output_mocks_dirname, basename)
    for basename in output_color_mock_basename_list)

umachine_z0p1_color_mock_fname = args.umachine_z0p1_color_mock_fname

number_of_snapshots = args.nsnap
if number_of_snapshots != -1:
    alphaQ_halos_fname_list = list(
        alphaQ_halos_fname_list[i] for i in range(number_of_snapshots))
    umachine_mstar_ssfr_mock_fname_list = list(
        umachine_mstar_ssfr_mock_fname_list[i] for i in range(number_of_snapshots))
    bpl_halos_fname_list = list(
        bpl_halos_fname_list[i] for i in range(number_of_snapshots))
    output_color_mock_fname_list = list(
        output_color_mock_fname_list[i] for i in range(number_of_snapshots))
    redshift_list = list(
        redshift_list[i] for i in range(number_of_snapshots))

commit_hash = args.commit_hash
write_snapshot_mocks_to_disk(
            umachine_z0p1_color_mock_fname, alphaQ_halos_fname_list,
            umachine_mstar_ssfr_mock_fname_list, bpl_halos_fname_list,
            output_color_mock_fname_list, redshift_list, commit_hash, overwrite=True)







