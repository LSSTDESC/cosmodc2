"""
"""
from cosmodc2.sdss_colors import mock_magr
from cosmodc2.sdss_colors import mc_sdss_gr_ri


__all__ = ('write_z0p1_color_mock_to_disk', )


def write_z0p1_color_mock_to_disk(sdss, um_mock, outname, overwrite=False):
    """
    """
    um_mock['restframe_extincted_sdss_abs_magr'] = mock_magr(
            um_mock['obs_sm'], um_mock['sfr_percentile_fixed_sm'],
            sdss['sm'], sdss['sfr_percentile_fixed_sm'],
            sdss['restframe_extincted_sdss_abs_magr'], sdss['z'])

    mock_rmag = um_mock['restframe_extincted_sdss_abs_magr']
    mock_mstar = um_mock['obs_sm']
    mock_sfr_percentile = um_mock['sfr_percentile_fixed_sm']

    sdss_redshift = sdss['z']
    sdss_magr = sdss['restframe_extincted_sdss_abs_magr']
    sdss_gr = sdss['restframe_extincted_sdss_gr']
    sdss_ri = sdss['restframe_extincted_sdss_ri']

    gr, ri = mc_sdss_gr_ri(
            mock_rmag, mock_mstar, mock_sfr_percentile,
            sdss_redshift, sdss_magr, sdss_gr, sdss_ri, k=10)
    um_mock['restframe_extincted_sdss_gr'] = gr
    um_mock['restframe_extincted_sdss_ri'] = ri

    um_mock.write(outname, path='data', overwrite=overwrite)
