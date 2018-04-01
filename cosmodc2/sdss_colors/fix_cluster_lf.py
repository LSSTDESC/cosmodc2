"""
"""
import numpy as np
from scipy.stats import binned_statistic
from halotools.empirical_models import conditional_abunmatch


__all__ = ('calculate_cluster_clf_powerlaw_coeffs', 'remap_cluster_bcg_gr_color')


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


def cluster_bcg_red_sequence_gr(ngals, red_sequence_median, red_sequence_scatter):
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


def remap_cluster_bcg_gr_color(upid, host_halo_mvir, gr,
        host_mass_table=(13.5, 13.75, 14, 14.25), prob_remap_table=(0, 0.25, 0.75, 1),
        red_sequence_median=0.95, red_sequence_scatter=0.04, nwin=101):
    """
    """
    remapping_mask = prob_remap_cluster_bcg(
        upid, host_halo_mvir, host_mass_table, prob_remap_table)
    num_to_remap = np.count_nonzero(remapping_mask)

    if num_to_remap > 2:
        bcg_red_sequence = cluster_bcg_red_sequence_gr(
            num_to_remap, red_sequence_median, red_sequence_scatter)

        x1 = host_halo_mvir[remapping_mask]
        y1 = gr[remapping_mask]
        x2 = host_halo_mvir[remapping_mask]
        y2 = bcg_red_sequence
        nwin = min(num_to_remap, nwin)
        if nwin % 2 == 0:
            nwin -= 1
        remapped_red_sequence = conditional_abunmatch(x1, y1, x2, y2, nwin)
        gr[remapping_mask] = remapped_red_sequence

    return gr















