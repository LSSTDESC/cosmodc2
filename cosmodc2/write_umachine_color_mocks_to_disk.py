"""
"""
import numpy as np
from time import time
from astropy.table import Table
from cosmodc2.sdss_colors import load_umachine_processed_sdss_catalog
from cosmodc2.sdss_colors import assign_restframe_sdss_gri


def write_snapshot_mocks_to_disk(sdss_fname,
            umachine_mstar_ssfr_mock_fname_list,
            output_color_mock_fname_list, redshift_list, commit_hash):
    """
    """

    gen = zip(umachine_mstar_ssfr_mock_fname_list,
            output_color_mock_fname_list, redshift_list)

    for umachine_mock_fname, output_color_mock_fname, redshift in gen:
        new_time_stamp = time()

        print("\n...loading z = {0:.3f} catalogs into memory".format(redshift))

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

        raise NotImplementedError("Still need to GalSample into AlphaQ")

        print("          Writing to disk using commit hash {}".format(commit_hash))
        mock.meta['cosmodc2_commit_hash'] = commit_hash
        mock.write(output_color_mock_fname, path='data', overwrite=True)

        time_stamp = time()
        msg = "End-to-end runtime for redshift {0:.1f} = {1:.2f} minutes"
        print(msg.format(redshift, (new_time_stamp-time_stamp)/60.))
