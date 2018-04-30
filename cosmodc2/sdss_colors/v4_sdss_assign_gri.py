""" Module storing the primary function used to paint g, r, i SDSS flux
onto model galaxies.
"""
import numpy as np
from .analytical_magr import magr_monte_carlo
from ..stellar_mass_remapping import lift_high_mass_mstar
from .analytical_colors import gr_ri_monte_carlo
from .fix_cluster_lf import remap_cluster_bcg_gr_ri_color, remap_cluster_satellite_gr_ri_color


__all__ = ('assign_restframe_sdss_gri', )


def assign_restframe_sdss_gri(upid_mock, mstar_mock, sfr_percentile_mock,
            host_halo_mvir_mock, redshift_mock, **kwargs):
    """ Primary function used to paint g, r, i SDSS flux onto UniverseMachine galaxies.

    Parameters
    ----------
    upid_mock : ndarray
        Numpy integer array of shape (ngals, ) storing the upid column

    mstar_mock : ndarray
        Numpy array of shape (ngals, ) storing stellar mass in
        units of Msun assuming h=0.7

    sfr_percentile_mock : ndarray
        Numpy array of shape (ngals, ) storing the SFR-percentile
        of each galaxy, Prob(< SFR | M*). This quantity can be calculated using
        the halotools.utils.sliding_conditional_percentile function.

    host_halo_mvir_mock : ndarray
        Numpy array of shape (ngals, ) storing the host halo mass in
        units of Msun assuming h=0.7

    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of every galaxy

    Returns
    -------
    magr : ndarray
        Numpy array of shape (ngals, ) storing restframe r-band absolute magnitude

    gr_mock : ndarray
        Numpy array of shape (ngals, ) storing restframe g-r color

    ri_mock : ndarray
        Numpy array of shape (ngals, ) storing restframe r-i color
    """
    ngals = len(upid_mock)

    redshift_mock = np.atleast_1d(redshift_mock)
    if len(redshift_mock) == 1:
        redshift_mock = np.zeros(ngals).astype('f4') + redshift_mock[0]

    #  Calculate model values of Mr
    magr = magr_monte_carlo(mstar_mock, redshift_mock, **kwargs)

    #  Calculate model values of (g-r) and (r-i)
    gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = gr_ri_monte_carlo(
        magr, sfr_percentile_mock, redshift_mock, local_random_scale=0.1)

    #  Redden the centrals of cluster-mass halos
    _result = remap_cluster_bcg_gr_ri_color(
            upid_mock, host_halo_mvir_mock,
            np.copy(gr_mock), np.copy(ri_mock),
            is_red_gr_mock, is_red_ri_mock)
    gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = _result

    #  Redden the satellites of cluster-mass halos
    gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = remap_cluster_satellite_gr_ri_color(
            upid_mock, mstar_mock, host_halo_mvir_mock, magr, gr_mock, ri_mock,
            is_red_gr_mock, is_red_ri_mock)

    return magr, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock
