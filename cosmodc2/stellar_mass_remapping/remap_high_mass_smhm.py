"""
"""
import numpy as np
from scipy.stats import binned_statistic


__all__ = ('lift_high_mass_mstar', )


def lift_high_mass_mstar(mpeak, mstar, boosted_high_mass_slope, logmpeak_pivot):
    """
    """
    high_mass_slope, high_mass_intercept = fit_high_mass_smhm(mpeak, mstar)

    boosted_high_mass_intercept = calculate_boosted_new_intercept(
            high_mass_intercept, high_mass_slope, boosted_high_mass_slope, logmpeak_pivot)

    return new_mstar(mpeak, mstar, logmpeak_pivot,
                boosted_high_mass_intercept, boosted_high_mass_slope)


def fit_high_mass_smhm(mpeak, mstar, logmpeak_low=12.5, logmpeak_high=14.5):
    """
    """
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
