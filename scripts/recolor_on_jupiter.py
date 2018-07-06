"""
"""
import sys
import os
import fnmatch
import numpy as np
from astropy.table import Table, vstack
from time import time

sys.path.insert(0, '/homes/ahearin/repositories/halotools/build/lib.linux-x86_64-2.7')
sys.path.insert(0, '/homes/ahearin/repositories/cosmodc2')

from cosmodc2.sdss_colors import v4_paint_colors_onto_umachine_snaps
from cosmodc2.synthetic_subhalos import model_extended_mpeak, map_mstar_onto_lowmass_extension
from cosmodc2.synthetic_subhalos import create_synthetic_mock, create_synthetic_cluster_satellites


input_dirname = "/homes/ahearin/protoDC2/baseline_umachine_snapshot_mocks_v4.6"
output_dirname = "/homes/ahearin/protoDC2/recolored_mocks_v4p15"


def fname_generator(root_dirname, basename_filepat):
    """ Yield the absolute path of all files in the directory tree of ``root_dirname``
    with a basename matching the input pattern
    """

    for path, dirlist, filelist in os.walk(root_dirname):
        for filename in fnmatch.filter(filelist, basename_filepat):
            yield os.path.join(path, filename)


basename_pattern = "umachine_color_mock_v4_m000-*.hdf5"
matching_filenames = list(fname_generator(input_dirname, basename_pattern))
matching_basenames = list(set([os.path.basename(fname) for fname in matching_filenames if 'lightcone' not in fname]))
input_fnames = [os.path.join(input_dirname, fname) for fname in matching_basenames]
input_fnames = sorted(matching_basenames)[::-1]
if len(input_fnames) < 29:
    raise ValueError("Bad basename_pattern = {0}".format(basename_pattern))

snapnum_data_fname = '/homes/ahearin/repositories/cosmodc2/cosmodc2/data/z2ts.txt'
X = np.loadtxt(snapnum_data_fname, delimiter=',')
snapnums = X[:, 1].astype(int)
redshifts = X[:, 0].astype(float)
redshift_dict = {str(snapnum).zfill(3): float(redshift) for snapnum, redshift in zip(snapnums, redshifts)}

basename_string = "umachine_color_mock_v4_m000-{0}.hdf5"

start = time()
for fname in input_fnames:
    snapstart = time()
    snapnum = fname[-8:-5]
    redshift = redshift_dict[snapnum]
    basename = basename_string.format(str(snapnum).zfill(3))
    fname = os.path.join(input_dirname, basename)
    assert os.path.isfile(fname), "{0} is not a file".format(fname)

    msg = "...reading snapnum {0} baseline UniverseMachine mock at z = {1:.2f}"
    print(msg.format(snapnum, redshift))
    mock = Table.read(fname, path='data')

    corrected_mpeak, mpeak_synthetic = model_extended_mpeak(mock['mpeak'], 10)
    mock.rename_column('mpeak', '_mpeak_orig_um_snap')
    mock['mpeak'] = corrected_mpeak

    #  Add call to map_mstar_onto_lowmass_extension function after pre-determining low-mass slope
    new_mstar_real, mstar_synthetic = map_mstar_onto_lowmass_extension(
        corrected_mpeak, mock['obs_sm'], mpeak_synthetic)
    mock['obs_sm'] = new_mstar_real

    print("...generating synthetic cluster satellites")
    fake_cluster_sats = create_synthetic_cluster_satellites(mock)
    if len(fake_cluster_sats) > 0:
        mock = vstack((mock, fake_cluster_sats))

    print("...stacking tables")
    mock = vstack((mock, create_synthetic_mock(mpeak_synthetic, mstar_synthetic, 256.)))

    print("...recoloring")
    result = v4_paint_colors_onto_umachine_snaps(
            mock['mpeak'], mock['obs_sm'], mock['upid'],
            redshift, mock['sfr_percentile'], mock['host_halo_mvir'])
    new_mstar, new_magr_rest, gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = result
    mock['obs_sm'] = new_mstar
    mock['restframe_extincted_sdss_abs_magr'] = new_magr_rest
    mock['restframe_extincted_sdss_gr'] = gr_mock
    mock['restframe_extincted_sdss_ri'] = ri_mock
    mock['is_on_red_sequence_ri'] = is_red_ri_mock
    mock['is_on_red_sequence_gr'] = is_red_gr_mock

    mock['redshift'] = redshift
    mock['lightcone_id'] = np.arange(len(mock)).astype(long)

    outbase = 'recolored_' + basename.replace('_v4_', '_v4.15_')
    outname = os.path.join(output_dirname, outbase)
    msg = "...writing recolored mock to the following path on disk:\n{0}"
    print(msg.format(outname))

    mock.write(outname, path='data', overwrite=True)
    snapend = time()
    snap_runtime = (snapend-snapstart)/60.
    print("Total runtime for snapnum {0} = {1:.1f} minutes\n".format(snapnum, snap_runtime))


end = time()
runtime = (end-start)/60.
print("Total runtime to recolor {0} snapshots = {1:.1f} minutes".format(
    len(input_fnames), runtime))



