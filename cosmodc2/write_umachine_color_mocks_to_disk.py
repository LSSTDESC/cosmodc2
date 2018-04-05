"""
"""
import numpy as np
from time import time
from astropy.table import Table
from cosmodc2.sdss_colors import load_umachine_processed_sdss_catalog
from cosmodc2.sdss_colors import assign_restframe_sdss_gri
from galsampler import halo_bin_indices, source_halo_index_selection
from galsampler.cython_kernels import galaxy_selection_kernel
from cosmodc2.load_gio_halos import load_gio_halo_snapshot


def write_snapshot_mocks_to_disk(sdss_fname,
            umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list,
            redshift_list, commit_hash, target_halo_loader):
    """
    """

    gen = zip(umachine_mstar_ssfr_mock_fname_list, umachine_host_halo_fname_list,
            target_halo_fname_list, output_color_mock_fname_list, redshift_list)

    for a, b, c, d, e in gen:
        umachine_mock_fname = a
        umachine_halos_fname = b
        target_halo_fname = c
        output_color_mock_fname = d
        redshift = e

        new_time_stamp = time()
        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))

        mock = Table.read(umachine_mock_fname, path='data')
        sdss = load_umachine_processed_sdss_catalog(sdss_fname)

        upid_mock = mock['upid']
        mstar_mock = mock['obs_sm']
        sfr_percentile_mock = mock['sfr_percentile']
        host_halo_mvir_mock = mock['host_halo_mvir']
        redshift_mock = np.zeros(len(mock)) + 0.0

        logmstar_sdss = sdss['sm']
        sfr_percentile_sdss = sdss['sfr_percentile_fixed_sm']
        sdss_magr = sdss['restframe_extincted_sdss_abs_magr']
        sdss_redshift = sdss['z']

        magr, gr_mock, ri_mock = assign_restframe_sdss_gri(
            upid_mock, mstar_mock, sfr_percentile_mock, host_halo_mvir_mock, redshift_mock,
            logmstar_sdss, sfr_percentile_sdss, sdss_magr, sdss_redshift)
        mock['restframe_extincted_sdss_abs_magr'] = magr
        mock['restframe_extincted_sdss_gr'] = gr_mock
        mock['restframe_extincted_sdss_ri'] = ri_mock

        ###  GalSampler
        print("\n...loading z = {0:.2f} halo catalogs into memory".format(redshift))
        source_halos = Table.read(umachine_halos_fname, path='data')
        target_halos = load_alphaQ_halos(target_halo_fname, target_halo_loader)

        #  Bin the halos in each simulation by mass
        dlogM = 0.15
        mass_bins = 10.**np.arange(10.5, 14.5+dlogM, dlogM)
        source_halos['mass_bin'] = halo_bin_indices(
            mass=(source_halos['mvir'], mass_bins))
        target_halos['mass_bin'] = halo_bin_indices(
            mass=(target_halos['fof_halo_mass'], mass_bins))

        #  Randomly draw halos from corresponding mass bins
        nhalo_min = 10
        source_halo_bin_numbers = source_halos['mass_bin']
        target_halo_bin_numbers = target_halos['mass_bin']
        target_halo_ids = target_halos['halo_id']
        _result = source_halo_index_selection(source_halo_bin_numbers,
            target_halo_bin_numbers, target_halo_ids, nhalo_min, mass_bins)
        source_halo_indx, matching_target_halo_ids = _result

        #  Transfer quantities from the source halos to the corresponding target halo
        target_halos['source_halo_id'] = source_halos['halo_id'][source_halo_indx]
        target_halos['matching_mvir'] = source_halos['mvir'][source_halo_indx]
        target_halos['richness'] = source_halos['richness'][source_halo_indx]
        target_halos['first_galaxy_index'] = source_halos['first_galaxy_index'][source_halo_indx]

        ################################################################################
        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        ################################################################################
        print("          Mapping z={0:.2f} galaxies to AlphaQ halos".format(redshift))

        source_galaxy_indx = np.array(galaxy_selection_kernel(
            target_halos['first_galaxy_index'].astype('i8'),
            target_halos['richness'].astype('i4'), target_halos['richness'].sum()))

        ########################################################################
        #  Assemble the output protoDC2 mock
        ########################################################################

        raise NotImplementedError("Still need to GalSample into AlphaQ")

        print("          Writing to disk using commit hash {}".format(commit_hash))
        mock.meta['cosmodc2_commit_hash'] = commit_hash
        mock.write(output_color_mock_fname, path='data', overwrite=True)

        time_stamp = time()
        msg = "End-to-end runtime for redshift {0:.1f} = {1:.2f} minutes"
        print(msg.format(redshift, (new_time_stamp-time_stamp)/60.))


def load_alphaQ_halos(fname, target_halo_loader):
    """
    """
    if target_halo_loader == 'hdf5':
        t = Table.read(fname, path='data')
    elif target_halo_loader == 'gio':
        t = load_gio_halo_snapshot(fname)
    else:
        raise ValueError("Options for ``loader`` are ``hdf5`` or ``gio``")

    t.rename_column('fof_halo_tag', 'halo_id')

    t.rename_column('fof_halo_center_x', 'x')
    t.rename_column('fof_halo_center_y', 'y')
    t.rename_column('fof_halo_center_z', 'z')

    t.rename_column('fof_halo_mean_vx', 'vx')
    t.rename_column('fof_halo_mean_vy', 'vy')
    t.rename_column('fof_halo_mean_vz', 'vz')

    return t
