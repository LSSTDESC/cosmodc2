import sys
sys.path.insert(0, "/Users/aphearin/work/repositories/python/cosmodc2/build/lib")
# sys.path.insert(0, "/Users/aphearin/work/repositories/python/halotools/build/lib.macosx-10.9-x86_64-3.6")
import os
from cosmodc2.write_umachine_color_mocks_to_disk import write_snapshot_mocks_to_disk


umachine_mstar_ssfr_mock_dirname = (
    "/Volumes/NbodyDisk1/UniverseMachine/protoDC2_v4_mocks/value_added_catalogs")
umachine_mstar_ssfr_mock_basename_list = list(
    ("sfr_catalog_1.000000_value_added.hdf5", "sfr_catalog_0.270600_value_added.hdf5"))
umachine_mstar_ssfr_mock_fname_list = list(
    (os.path.join(umachine_mstar_ssfr_mock_dirname, basename)
    for basename in umachine_mstar_ssfr_mock_basename_list))

umachine_host_halo_dirname = (
    "/Volumes/NbodyDisk1/UniverseMachine/protoDC2_v4_mocks/value_added_catalogs")
umachine_host_halo_basename_list = list(
    ("halo_catalog_1.000000_value_added.hdf5", "halo_catalog_0.270600_value_added.hdf5"))
umachine_host_halo_fname_list = list(
    (os.path.join(umachine_host_halo_dirname, basename)
    for basename in umachine_host_halo_basename_list))

target_halo_dirname = "/Volumes/NbodyDisk1/UniverseMachine/protoDC2_v3_mocks/fof_halos"
target_halo_basename_list = list(
    ("fof_halos_a1.00.hdf5", "fof_halos_a0.334060.hdf5"))
target_halo_fname_list = list(
    (os.path.join(target_halo_dirname, basename)
    for basename in target_halo_basename_list))

output_mock_dirname = (
    "/Volumes/NbodyDisk1/UniverseMachine/protoDC2_v4_mocks/galsampler_alphaq_outputs")
output_mock_basename_list = list(
    ("umachine_color_mock_1.000000.hdf5", "umachine_color_mock_0.270600.hdf5"))
output_color_mock_fname_list = list(
    (os.path.join(output_mock_dirname, basename)
    for basename in output_mock_basename_list))

redshift_list = [0., 2.]
commit_hash = "dummy_hash"
target_halo_loader = "hdf5"
Lbox_target_halos = 256.

write_snapshot_mocks_to_disk(
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list,
            redshift_list, commit_hash, target_halo_loader, Lbox_target_halos)
