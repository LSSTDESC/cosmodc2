"""
"""
import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm, binned_statistic
from halotools.empirical_models import conditional_abunmatch

from .analytical_g_minus_r import red_sequence_peak_gr, default_red_peak_gr, default_red_peak_gr_zevol
from .analytical_g_minus_r import default_red_scatter_gr, default_red_scatter_gr_zevol_table

from .analytical_r_minus_i import red_sequence_peak_ri, default_red_peak_ri, default_red_peak_ri_zevol
from .analytical_r_minus_i import default_red_scatter_ri, default_red_scatter_ri_zevol_table

from .analytical_g_minus_r import red_sequence_width_gr
from .analytical_r_minus_i import red_sequence_width_ri


__all__ = ('calculate_cluster_clf_powerlaw_coeffs',
        'remap_cluster_bcg_gr_ri_color', 'remap_cluster_satellite_gr_ri_color')


def calculate_cluster_clf_powerlaw_coeffs(mstar, magr, upid):
    """ Fit the M*-Mr relation over the reliable range with a powerlaw:

    Mr = c0 + c1*np.log10(mstar)

    Return the coefficients c0, c1.
    """
    cenmask = upid == -1

    sm_bins = np.logspace(10, 11.5, 30)
    logsm_bins = np.log10(sm_bins)
    sm_mids = 10**(0.5*(logsm_bins[:-1] + logsm_bins[1:]))
    logsm_mids = np.log10(sm_mids)

    median_magr, __, __ = binned_statistic(mstar[cenmask], magr[cenmask],
        bins=sm_bins, statistic='median')
    c1, c0 = np.polyfit(logsm_mids, median_magr, deg=1)

    return c0, c1


def cluster_bcg_red_sequence(ngals, red_sequence_median, red_sequence_scatter):
    """
    """
    return np.random.normal(loc=red_sequence_median, scale=red_sequence_scatter, size=ngals)


def prob_remap_cluster_bcg(upid, host_halo_mvir, host_mass_table, prob_remap_table):
    """
    """
    cenmask = upid == -1
    prob_remap = np.interp(np.log10(host_halo_mvir), host_mass_table, prob_remap_table)
    host_halo_mask = np.random.rand(len(upid)) < prob_remap
    remapping_mask = host_halo_mask & cenmask
    return remapping_mask


def correlated_gr_ri(num_samples, gr_median, ri_median, scatter):
    """
    """
    cov = np.array([
            [5.50, 1.50],
            [1.50,  1.25]
        ])
    cov = (scatter**2)*cov/np.linalg.det(cov)

    # We need a matrix `c` for which `c*c^T = cov`.
    # Construct c, so c*c^T = cov.
    evals, evecs = eigh(cov)
    c = np.dot(evecs, np.diag(np.sqrt(evals)))

    # Generate samples from independent normally distributed random
    # variables (with mean 0 and std. dev. 1).
    x = norm.rvs(size=(2, num_samples))

    # Convert the data to correlated random variables.
    y = np.dot(c, x)
    gr, ri = y[0] + gr_median, y[1] + ri_median
    return gr, ri


def remap_cluster_bcg_gr_ri_color(upid, host_halo_mvir, magr, gr, ri,
        is_on_red_sequence_gr, is_on_red_sequence_ri, redshift,
        host_mass_table=(13.25, 13.5, 13.75, 14, 14.25), prob_remap_table=(0, 0.35, 0.5, 0.75, 1),
        gr_red_sequence_scatter=0.015, ri_red_sequence_scatter=0.01,
        red_peak_gr=default_red_peak_gr, red_peak_gr_zevol_shift_table=default_red_peak_gr_zevol,
        red_peak_ri=default_red_peak_ri, red_peak_ri_zevol_shift_table=default_red_peak_ri_zevol,
        **kwargs):
    """ Redden centrals in cluster-mass halos

    Parameters
    ----------
    upid : ndarray
        Numpy integer array of shape (ngals, ) storing the upid column

    host_halo_mvir : ndarray
        Numpy array of shape (ngals, ) storing the host halo mass in
        units of Msun assuming h=0.7

    gr : ndarray
        Numpy array of shape (ngals, ) storing restframe g-r color

    ri : ndarray
        Numpy array of shape (ngals, ) storing restframe r-i color

    is_on_red_sequence_gr : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the g-r red sequence

    is_on_red_sequence_ri : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the r-i red sequence

    Returns
    -------
    gr_new : ndarray
        Numpy array of shape (ngals, ) storing g-r restframe color for every galaxy,
        reddened for centrals of cluster halos

    ri_new : ndarray
        Numpy array of shape (ngals, ) storing r-i restframe color for every galaxy,
        reddened for centrals of cluster halos

    is_quiescent_gr_new : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the g-r red sequence

    is_quiescent_ri_new : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the r-i red sequence

    """
    remapping_mask = prob_remap_cluster_bcg(
        upid, host_halo_mvir, host_mass_table, prob_remap_table)
    num_to_remap = np.count_nonzero(remapping_mask)

    if num_to_remap > 0:
        is_on_red_sequence_gr[remapping_mask] = True
        is_on_red_sequence_ri[remapping_mask] = True

        gr_red_sequence_median = red_sequence_peak_gr(magr[remapping_mask], red_peak_gr,
            redshift[remapping_mask], red_peak_gr_zevol_shift_table)
        ri_red_sequence_median = red_sequence_peak_ri(magr[remapping_mask], red_peak_ri,
            redshift[remapping_mask], red_peak_ri_zevol_shift_table)

        bcg_red_sequence_gr, bcg_red_sequence_ri = correlated_gr_ri(
            num_to_remap, gr_red_sequence_median,
            ri_red_sequence_median, gr_red_sequence_scatter)

        gr[remapping_mask] = bcg_red_sequence_gr
        ri[remapping_mask] = bcg_red_sequence_ri
    return gr, ri, is_on_red_sequence_gr, is_on_red_sequence_ri


def prob_remap_cluster_satellite(upid, mstar, host_halo_mvir,
            mstar_sat_prob_remap_abscissa=(9, 9.75, 10.25, 11),
            mstar_sat_prob_remap=(1., 1., 0.5, 0.),
            mhalo_sat_prob_remap_abscissa=(13.25, 13.5, 13.75, 14, 14.5),
            mhalo_sat_prob_remap=(0.0, 0.35, 0.5, 0.65, 0.85),
            **kwargs):
    """
    """
    ngals = len(mstar)
    satmask = upid != -1
    mstar_prob = np.interp(np.log10(mstar), mstar_sat_prob_remap_abscissa, mstar_sat_prob_remap)
    mhost_prob = np.interp(np.log10(host_halo_mvir), mhalo_sat_prob_remap_abscissa, mhalo_sat_prob_remap)
    mstar_mask = np.random.rand(ngals) < mstar_prob
    mhost_mask = np.random.rand(ngals) < mhost_prob
    remapping_mask = mstar_mask & mhost_mask & satmask
    return remapping_mask


def remap_satellites(mstar, gr, ri,
            gr_red_sequence_median, ri_red_sequence_median,
            gr_red_sequence_scatter, ri_red_sequence_scatter, nwin=301):
    """
    """
    num_to_remap = len(mstar)
    bcg_red_sequence_gr, bcg_red_sequence_ri = correlated_gr_ri(
        num_to_remap, gr_red_sequence_median,
        ri_red_sequence_median, gr_red_sequence_scatter)

    input_gr = gr
    desired_gr = bcg_red_sequence_gr
    output_gr = conditional_abunmatch(
        mstar, input_gr, mstar, desired_gr, nwin)

    desired_ri = bcg_red_sequence_ri
    noisy_input_gr = np.random.normal(loc=output_gr, scale=0.1)
    output_ri = conditional_abunmatch(
        mstar, noisy_input_gr, mstar, desired_ri, nwin)
    return output_gr, output_ri


def remap_cluster_satellite_gr_ri_color(upid, mstar, host_halo_mvir, magr, gr, ri,
            is_on_red_sequence_gr, is_on_red_sequence_ri, redshift, scatter=0.03, **kwargs):
    """ Redden satellites in cluster-mass halos

    Parameters
    ----------
    upid : ndarray
        Numpy integer array of shape (ngals, ) storing the upid column

    mstar : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass in
        units of Msun assuming h=0.7

    host_halo_mvir : ndarray
        Numpy array of shape (ngals, ) storing the host halo mass in
        units of Msun assuming h=0.7

    magr : ndarray
        Numpy array of shape (ngals, ) storing restframe r-band absolute magnitude

    gr : ndarray
        Numpy array of shape (ngals, ) storing restframe g-r color

    ri : ndarray
        Numpy array of shape (ngals, ) storing restframe r-i color

    Returns
    -------
    gr_new : ndarray
        Numpy array of shape (ngals, ) storing g-r restframe color for every galaxy,
        reddened for satellites of cluster halos

    ri_new : ndarray
        Numpy array of shape (ngals, ) storing r-i restframe color for every galaxy,
        reddened for satellites of cluster halos
    """
    remapping_mask = prob_remap_cluster_satellite(upid, mstar, host_halo_mvir, **kwargs)
    is_on_red_sequence_gr[remapping_mask] = True
    is_on_red_sequence_ri[remapping_mask] = True

    gr_peak_sats_to_remap = red_sequence_peak_gr(magr[remapping_mask], default_red_peak_gr,
            redshift[remapping_mask], default_red_peak_gr_zevol)
    ri_peak_sats_to_remap = red_sequence_peak_ri(magr[remapping_mask], default_red_peak_ri,
            redshift[remapping_mask], default_red_peak_ri_zevol)

    gr_scatter_sats_to_remap = red_sequence_width_gr(
                magr[remapping_mask], default_red_scatter_gr,
                redshift[remapping_mask], default_red_scatter_gr_zevol_table)
    ri_scatter_sats_to_remap = red_sequence_width_ri(
                magr[remapping_mask], default_red_scatter_ri,
                redshift[remapping_mask], default_red_scatter_ri_zevol_table)

    remapped_cluster_satellite_gr = np.random.normal(
        loc=gr_peak_sats_to_remap, scale=gr_scatter_sats_to_remap)
    remapped_cluster_satellite_ri = np.random.normal(
        loc=ri_peak_sats_to_remap, scale=ri_scatter_sats_to_remap)

    gr[remapping_mask] = remapped_cluster_satellite_gr
    ri[remapping_mask] = remapped_cluster_satellite_ri

    return gr, ri, is_on_red_sequence_gr, is_on_red_sequence_ri

