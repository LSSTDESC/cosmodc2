"""
"""
import numpy as np
from halotools.utils import unsorting_indices


__all__ = ('model_extended_mpeak', )


def model_extended_mpeak(mpeak, desired_logm_completeness,
            logm_min_fit=11.75, logm_max_fit=12.25, Lbox=256.):
    """ Given an input set of subhalo mpeak values, and a desired completeness limit,
    fit the input distribution with a power law at the low mass end,
    extrapolate subhalo abundance to lower masses, and return a set of subhalos
    whose abundance obeys the best-fit power law down to the desired completeness limit.

    Parameters
    ----------
    mpeak : ndarray
        Numpy array of shape (nsubs_orig, )

    new_logmpeak_low : float
        Desired completeness limit in log10 units

    Returns
    -------
    corrected_mpeak : ndarray
        Numpy array of shape (nsubs_orig, ). Values greater than the midpoint
        of the fitted range will be unchanged from the original mpeak. Values less than this
        will be altered to correct for the power law departure

    mpeak_extension : ndarray
        Numpy array of shape (num_new_subs, ) storing values of Mpeak for synthetic subhalos

    Examples
    --------
    >>> from scipy.stats import powerlaw
    >>> mpeak = 10**(5*(1-powerlaw.rvs(2, size=40000)) + 10.)
    >>> desired_logm_completeness = 9.5
    >>> corrected_mpeak, mpeak_extension = model_extended_mpeak(mpeak, desired_logm_completeness)
    """
    logmpeak = np.log10(mpeak)
    idx_sorted = np.argsort(logmpeak)[::-1]
    sorted_logmpeak = logmpeak[idx_sorted]

    Vbox = Lbox**3
    npts_total = len(logmpeak)
    logndarr = np.log10(np.arange(1, 1 + npts_total)/Vbox)

    logm_mid = 0.5*(logm_min_fit + logm_max_fit)

    mask = sorted_logmpeak >= logm_min_fit
    mask &= sorted_logmpeak < logm_max_fit

    c1, c0 = np.polyfit(sorted_logmpeak[mask][::100], logndarr[mask][::100], deg=1)
    model_lognd = c0 + c1*sorted_logmpeak

    model_logmpeak = np.interp(logndarr, model_lognd, sorted_logmpeak)
    model_logmpeak[sorted_logmpeak > logm_mid] = sorted_logmpeak[sorted_logmpeak > logm_mid]

    lognd_extension_max = c0 + c1*desired_logm_completeness
    new_ngals_tot = int((10**lognd_extension_max)*Vbox)
    logndarr_extension = np.log10(np.arange(1 + npts_total, new_ngals_tot)/Vbox)
    logmpeak_extension = (logndarr_extension - c0)/c1
    mpeak_extension = 10**logmpeak_extension

    corrected_mpeak = 10**model_logmpeak[unsorting_indices(idx_sorted)]
    return corrected_mpeak, mpeak_extension

