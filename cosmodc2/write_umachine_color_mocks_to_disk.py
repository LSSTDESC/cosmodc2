"""
"""
import numpy as np
from time import time
from astropy.table import Table
from cosmodc2.sdss_colors import load_umachine_processed_sdss_catalog
from cosmodc2.sdss_colors import mock_magr, gr_ri_monte_carlo, remap_cluster_bcg_gr_ri_color
from cosmodc2.size_modeling import mc_size_vs_luminosity_early_type, mc_size_vs_luminosity_late_type


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

        #  Assign r-band magnitude
        mock['restframe_extincted_sdss_abs_magr'] = mock_magr(
                    mock['upid'], mock['obs_sm'], mock['sfr_percentile'],
                    mock['host_halo_mvir'], sdss['sm'], sdss['sfr_percentile_fixed_sm'],
                    sdss['restframe_extincted_sdss_abs_magr'], sdss['z'])

        #  Assign g-r & r-i colors
        magr = mock['restframe_extincted_sdss_abs_magr']
        percentile = mock['sfr_percentile']
        redshift = np.zeros_like(magr)

        gr, ri, is_red_ri, is_red_gr = gr_ri_monte_carlo(
            magr, percentile, redshift, local_random_scale=0.1)

        mock['restframe_extincted_sdss_gr'] = gr
        mock['restframe_extincted_sdss_ri'] = ri
        mock['is_red_gr'] = is_red_gr
        mock['is_red_ri'] = is_red_ri

        _result = remap_cluster_bcg_gr_ri_color(
                mock['upid'], mock['host_halo_mvir'],
                np.copy(mock['restframe_extincted_sdss_gr']),
                np.copy(mock['restframe_extincted_sdss_ri']),
                mock['is_red_gr'], mock['is_red_ri'])
        gr_remapped, ri_remapped, is_red_gr_remapped, is_red_ri_remapped = _result

        mock['_gr_no_remap'] = np.copy(mock['restframe_extincted_sdss_gr'])
        mock['_ri_no_remap'] = np.copy(mock['restframe_extincted_sdss_ri'])
        mock['restframe_extincted_sdss_gr'] = gr_remapped
        mock['restframe_extincted_sdss_ri'] = ri_remapped
        mock['_is_red_ri_no_remap'] = np.copy(is_red_ri)
        mock['_is_red_gr_no_remap'] = np.copy(is_red_gr)
        mock['is_red_gr'] = is_red_gr_remapped
        mock['is_red_ri'] = is_red_ri_remapped

        #  Assign bulge and disk sizes
        mock['bulge_size'] = mc_size_vs_luminosity_early_type(
                mock['restframe_extincted_sdss_abs_magr'], redshift)
        mock['disk_size'] = mc_size_vs_luminosity_late_type(
                mock['restframe_extincted_sdss_abs_magr'], redshift)

        print("          Writing to disk using commit hash {}".format(commit_hash))
        mock.meta['cosmodc2_commit_hash'] = commit_hash
        mock.write(output_color_mock_fname, path='data', overwrite=True)

        time_stamp = time()
        msg = "End-to-end runtime for redshift {0:.1f} = {1:.2f} minutes"
        print(msg.format(redshift, (new_time_stamp-time_stamp)/60.))
