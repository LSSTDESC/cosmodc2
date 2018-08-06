"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
default_desired_logm_completeness = 9.75


__all__ = ('synthetic_logmpeak',)


def synthetic_logmpeak(mpeak_orig, desired_logm_completeness=default_desired_logm_completeness,
            nout_max=np.inf, c0=5.5, c1=-0.15,
            lowm_fit=11.5, highm_fit=12.25, dlogm_fit=0.1, return_fit=False, seed=43):
    """
    """
    logmbins_fit = np.arange(lowm_fit, highm_fit+dlogm_fit, dlogm_fit)
    logmmids_fit = 0.5*(logmbins_fit[:-1] + logmbins_fit[1:])

    logmpeak_orig = np.log10(mpeak_orig)
    counts_fit, __ = np.histogram(logmpeak_orig, bins=logmbins_fit)

    zero_mask = counts_fit > 0.
    if np.count_nonzero(zero_mask) > 3:
        c1, c0 = np.polyfit(logmmids_fit[zero_mask], np.log10(counts_fit[zero_mask]), deg=1)

    logmbins_extrap = np.arange(desired_logm_completeness-dlogm_fit, highm_fit+dlogm_fit, dlogm_fit)
    logmmids_extrap = 0.5*(logmbins_extrap[:-1] + logmbins_extrap[1:])
    logcounts_extrap = c0 + c1*logmmids_extrap
    model_counts_extrap = np.array(10**logcounts_extrap).astype(int)
    actual_counts_extrap, __ = np.histogram(logmpeak_orig, bins=logmbins_extrap)
    delta_counts_extrap = np.maximum(model_counts_extrap - actual_counts_extrap, 0)
    delta_counts_cumprob = np.cumsum(delta_counts_extrap)/float(delta_counts_extrap.sum()+1)

    npts_out = int(min(nout_max, delta_counts_extrap.sum()))
    with NumpyRNGContext(seed):
        uran = np.random.rand(npts_out)
    mc_logm = np.interp(uran, delta_counts_cumprob, logmmids_extrap)
    outmask = mc_logm > desired_logm_completeness
    return mc_logm[outmask]

