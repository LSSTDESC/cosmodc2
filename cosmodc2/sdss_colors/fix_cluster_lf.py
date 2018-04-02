"""
"""
import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm, binned_statistic
from halotools.empirical_models import conditional_abunmatch


__all__ = ('calculate_cluster_clf_powerlaw_coeffs', 'remap_cluster_bcg_gr_color',
        'remap_cluster_bcg_gr_ri_color')


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


def twodim_cluster_bcg_red_sequence(ngals, gr_median, gr_scatter, ri_median, ri_scatter):
    """
    """
    gr = cluster_bcg_red_sequence(ngals, gr_median, gr_scatter)
    ri = cluster_bcg_red_sequence(ngals, ri_median, ri_scatter)



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
        bcg_red_sequence = cluster_bcg_red_sequence(
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


def cluster_bcg_red_sequence_gr_ri(num_samples, gr_median, ri_median, scatter):
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


def remap_cluster_bcg_gr_ri_color(upid, host_halo_mvir, gr, ri,
        is_on_red_sequence_gr, is_on_red_sequence_ri,
        host_mass_table=(13.5, 13.75, 14, 14.25), prob_remap_table=(0, 0.25, 0.75, 1),
        gr_red_sequence_median=0.95, gr_red_sequence_scatter=0.04,
        ri_red_sequence_median=0.44, ri_red_sequence_scatter=0.03, nwin=101):
    """
    """
    remapping_mask = prob_remap_cluster_bcg(
        upid, host_halo_mvir, host_mass_table, prob_remap_table)
    is_on_red_sequence_gr[remapping_mask] = True
    is_on_red_sequence_ri[remapping_mask] = True

    num_to_remap = np.count_nonzero(remapping_mask)
    nwin = min(num_to_remap, nwin)
    if nwin % 2 == 0:
        nwin -= 1

    if num_to_remap > 2:

        bcg_red_sequence_gr, bcg_red_sequence_ri = cluster_bcg_red_sequence_gr_ri(
            num_to_remap, gr_red_sequence_median,
            ri_red_sequence_median, gr_red_sequence_scatter)

        # bcg_red_sequence_gr = cluster_bcg_red_sequence(
        #     num_to_remap, gr_red_sequence_median, gr_red_sequence_scatter)

        halo_mass = host_halo_mvir[remapping_mask]

        input_gr = gr[remapping_mask]
        desired_gr = bcg_red_sequence_gr
        remapped_red_sequence_gr = conditional_abunmatch(
            halo_mass, input_gr, halo_mass, desired_gr, nwin)
        gr[remapping_mask] = remapped_red_sequence_gr

        desired_ri = bcg_red_sequence_ri
        noisy_input_gr = np.random.normal(loc=remapped_red_sequence_gr, scale=0.1)
        remapped_red_sequence_ri = conditional_abunmatch(
            halo_mass, noisy_input_gr, halo_mass, desired_ri, nwin)
        ri[remapping_mask] = remapped_red_sequence_ri

    return gr, ri, is_on_red_sequence_gr, is_on_red_sequence_ri














