""" Module storing the primary function used to paint g, r, i SDSS flux
onto model galaxies.
"""
import numpy as np
from .sigmoid_magr_model import magr_monte_carlo
from .analytical_gr_ri import gr_ri_monte_carlo_substeps
from .analytical_gr_ri import gr_ri_monte_carlo
from .cluster_color_modeling import remap_cluster_bcg_gr_ri_color, remap_cluster_satellite_gr_ri_color


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
    use_substeps = kwargs.get('use_substeps', False)
    seed = kwargs.get('seed', 43)

    redshift_mock = np.atleast_1d(redshift_mock)
    if len(redshift_mock) == 1:
        redshift_mock = np.zeros(ngals).astype('f4') + redshift_mock[0]

    #  Calculate model values of Mr
    magr = magr_monte_carlo(mstar_mock, upid_mock, redshift_mock, **kwargs)

    #  Calculate model values of (g-r) and (r-i)
    if use_substeps:
        print('.....Check: calling gr_ri_monte_carlo_substeps with seed {}'.format(seed))
        gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = gr_ri_monte_carlo_substeps(
            magr, sfr_percentile_mock, redshift_mock, **kwargs)
    else:
        print('.....Check: calling gr_ri_monte_carlo with seed {}'.format(seed))
        gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = gr_ri_monte_carlo(
            magr, sfr_percentile_mock, redshift_mock, **kwargs)

    #  Redden the centrals of cluster-mass halos
    _result = remap_cluster_bcg_gr_ri_color(upid_mock, host_halo_mvir_mock,
                magr, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock, redshift_mock, **kwargs)
    gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = _result

    #  Redden the satellites of cluster-mass halos
    gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = remap_cluster_satellite_gr_ri_color(
            upid_mock, mstar_mock, host_halo_mvir_mock, magr, gr_mock, ri_mock,
            is_red_gr_mock, is_red_ri_mock, redshift_mock, **kwargs)

    return magr, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock
