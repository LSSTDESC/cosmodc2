"""
"""
import numpy as np


__all__ = ('remap_stellar_mass_in_snapshot', )


def remap_stellar_mass_in_snapshot(snapshot_redshift, mpeak, mstar,
        z_table=[0.1, 0.2, 0.5, 0.65, 1],
        slope_table=[0.5, 0.7, 0.7, 0.75, 0.75], pivot_table=[13., 13., 13., 12.75, 12.75]):
    """
    """
    mpeak = np.atleast_1d(mpeak)
    mstar = np.atleast_1d(mstar)

    msg = "Input snapshot_redshift must be a single float.\nReceived an array of shape {0}"
    _x = np.atleast_1d(snapshot_redshift)
    assert len(_x) == 1, msg.format(_x.shape)
    snapshot_redshift = _x[0]

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
