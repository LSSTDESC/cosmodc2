"""
"""
import numpy as np


__all__ = ('lift_high_mass_mstar', 'remap_stellar_mass_in_snapshot')


def remap_stellar_mass_in_snapshot(snapshot_redshift, mpeak, mstar,
        z_table=[0.25, 0.5, 0.75, 1],
        slope_table=[0.45, 0.5, 0.65, 0.7], pivot_table=[13.5, 13.25, 13, 12.75]):
    """
    """
    logmpeak_pivot = np.interp(snapshot_redshift, z_table, pivot_table)
    slope = np.interp(snapshot_redshift, z_table, slope_table)
    logmpeak_low, logmpeak_high = logmpeak_pivot - 0.05, logmpeak_pivot + 0.05

    prob_remap = np.interp(
        np.log10(mpeak), [logmpeak_low, logmpeak_high], [0, 1])
    remapping_mask = np.random.rand(len(mpeak)) < prob_remap

    pivot_mask = (mpeak > 10**logmpeak_low) & (mpeak < 10**logmpeak_high)
    logsm_pivot = np.median(np.log10(mstar[pivot_mask]))

    new_median_logsm = slope*(np.log10(mpeak[remapping_mask]) - logmpeak_pivot) + logsm_pivot

    new_mstar = 10**np.random.normal(loc=new_median_logsm, scale=0.15)
    result = np.zeros_like(mstar)
    result[remapping_mask] = new_mstar
    result[~remapping_mask] = mstar[~remapping_mask]

    return np.where(result < mstar, mstar, result)


def redshift_lifting_probability(redshift, z_low=0.25, z_high=0.5):
    """
    """
    return np.interp(redshift, [z_low, z_high], [0., 1.])


def lift_high_mass_mstar(mpeak, mstar, upid, redshift,
        z_table=[0.25, 0.5, 1], slope_table=[0.5, 0.65, 0.7], pivot_table=[13.25, 13, 12.5]):
    """
    """
    lifting_probability = redshift_lifting_probability(redshift)
    lifting_probability[upid != -1] = 0.
    cenmask = upid == -1
    cluster_bcg_outlier_mask = mpeak > 10**13.5
    cluster_bcg_outlier_mask *= mstar < 10**10
    lifting_probability[cluster_bcg_outlier_mask & cenmask] = 1.

    lifting_mask = np.random.rand(len(lifting_probability)) < lifting_probability

    logmpeak_pivot = calculate_logmpeak_pivot(redshift, z_table, pivot_table)
    boosted_high_mass_slope = calculate_high_mass_slope(redshift, z_table, slope_table)

    high_mass_slope, high_mass_intercept = fit_high_mass_smhm(mpeak, mstar)

    boosted_high_mass_intercept = calculate_boosted_new_intercept(
            high_mass_intercept, high_mass_slope, boosted_high_mass_slope, logmpeak_pivot)

    lifted_mstar = new_mstar(mpeak, mstar, logmpeak_pivot,
                boosted_high_mass_intercept, boosted_high_mass_slope)

    result = np.copy(mstar)
    result[lifting_mask] = lifted_mstar[lifting_mask]
    return result


def calculate_high_mass_slope(redshift, z_table, slope_table):
    """
    """
    return np.interp(redshift, z_table, slope_table)


def calculate_logmpeak_pivot(redshift, z_table, pivot_table):
    """
    """
    return np.interp(redshift, z_table, pivot_table)


def fit_high_mass_smhm(mpeak, mstar, logmpeak_low=12.5, logmpeak_high=14.5):
    """
    """
    from scipy.stats import binned_statistic

    median_logsm, logmpeak_bins, __ = binned_statistic(
        np.log10(mpeak), np.log10(mstar), bins=25)

    logmpeak_mids = 0.5*(logmpeak_bins[:-1] + logmpeak_bins[1:])

    high_mass_mask = (logmpeak_mids >= logmpeak_low) & (logmpeak_mids < logmpeak_high)
    logmpeak_high_mass_end = logmpeak_mids[high_mass_mask]
    logsm_high_mass_end = median_logsm[high_mass_mask]
    high_mass_slope, high_mass_intercept = np.polyfit(
        logmpeak_high_mass_end, logsm_high_mass_end, 1)

    return high_mass_slope, high_mass_intercept

def calculate_boosted_new_intercept(
        high_mass_intercept, high_mass_slope, boosted_high_mass_slope, logmpeak_pivot):
    orig_pivot_value = high_mass_intercept + logmpeak_pivot*high_mass_slope
    new_pivot_value = high_mass_intercept + logmpeak_pivot*boosted_high_mass_slope
    delta_pivot = new_pivot_value - orig_pivot_value
    boosted_high_mass_intercept = high_mass_intercept - delta_pivot
    return boosted_high_mass_intercept

def new_mstar(mpeak, mstar, logmpeak_pivot, intercept, slope):
    new_median_logsm = (intercept + slope*np.log10(mpeak))
    new_logsm = np.copy(np.log10(mstar))
    new_logsm_mask = np.log10(mpeak) > logmpeak_pivot
    new_logsm[new_logsm_mask] = np.random.normal(
        loc=new_median_logsm[new_logsm_mask], scale=0.25)
    return 10**new_logsm
