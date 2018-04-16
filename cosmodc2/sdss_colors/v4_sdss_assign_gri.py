"""
"""
import numpy as np
from ..sdss_colors import magr_monte_carlo, gr_ri_monte_carlo
from ..sdss_colors import remap_cluster_bcg_gr_ri_color, remap_cluster_satellite_gr_ri_color


__all__ = ('assign_restframe_sdss_gri', )


def assign_restframe_sdss_gri(upid_mock, mstar_mock, sfr_percentile_mock,
            host_halo_mvir_mock, redshift_mock):
    """
    """
    ngals = len(upid_mock)

    redshift_mock = np.atleast_1d(redshift_mock)
    if len(redshift_mock) == 1:
        redshift_mock = np.zeros(ngals).astype('f4') + redshift_mock[0]

    magr = magr_monte_carlo(mstar_mock, redshift_mock)

    gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = gr_ri_monte_carlo(
        magr, sfr_percentile_mock, redshift_mock, local_random_scale=0.1)

    _result = remap_cluster_bcg_gr_ri_color(
            upid_mock, host_halo_mvir_mock,
            np.copy(gr_mock), np.copy(ri_mock),
            is_red_gr_mock, is_red_ri_mock)
    gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = _result

    gr_mock, ri_mock = remap_cluster_satellite_gr_ri_color(
            upid_mock, mstar_mock, host_halo_mvir_mock, magr, gr_mock, ri_mock)

    return magr, gr_mock, ri_mock

