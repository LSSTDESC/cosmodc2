import os
from cosmodc2 import write_snapshot_mocks_to_disk

sdss_fname = "/Users/aphearin/Dropbox/protoDC2/SDSS/dr10_mgs_colors_processed.txt"

umachine_mstar_ssfr_mock_dirname = ""
umachine_mstar_ssfr_mock_basename_list = []
umachine_mstar_ssfr_mock_fname_list = list(
    (os.path.join(umachine_mstar_ssfr_mock_dirname, basename)
    for basename in umachine_mstar_ssfr_mock_basename_list))

umachine_host_halo_dirname = ""
umachine_host_halo_basename_list = []
umachine_host_halo_fname_list = list(
    (os.path.join(umachine_host_halo_dirname, basename)
    for basename in umachine_host_halo_basename_list))

target_halo_dirname = ""
target_halo_basename_list = []
target_halo_fname_list = list(
    (os.path.join(target_halo_dirname, basename)
    for basename in target_halo_basename_list))

output_mock_dirname = ""
output_mock_basename_list = []
output_color_mock_fname_list = list(
    (os.path.join(output_mock_dirname, basename)
    for basename in output_mock_basename_list))

redshift_list = []
commit_hash = ""
target_halo_loader = "hdf5"
Lbox_target_halos = 256.

write_snapshot_mocks_to_disk(sdss_fname,
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list,
            redshift_list, commit_hash, target_halo_loader, Lbox_target_halos)
