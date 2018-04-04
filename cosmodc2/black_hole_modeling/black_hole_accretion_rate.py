"""
"""
import numpy as np
from halotools.utils import monte_carlo_from_cdf_lookup


A = 10**-3.15
gamma_e = -0.65
gamma_z = 3.47
z0 = 0.6


__all__ = ('monte_carlo_specific_bh_acc_rate', )


def eddington_ratio_distribution(redshift, npts=1000):
    """
    """
    rate_table = np.logspace(-4, 0, npts)
    return rate_table, A*(((1. + redshift)/(1. + z0))**gamma_z)*rate_table**gamma_e


def monte_carlo_specific_bh_acc_rate(redshift, sfr_percentile):
    """
    Parameters
    ----------
    redshift : float
        Median redshift of the sample

    sfr_percentile : ndarray
        Numpy array of shape (ngals, ) storing the SFR rank-order percentile
        at fixed stellar mass.

    Returns
    -------
    specific_bh_acc_rate : ndarray
        Numpy array of shape (ngals, ) storing log10(dM_bh/dt / Mbh)

    Examples
    --------
    >>> redshift = 0.4
    >>> sfr_percentile = np.random.uniform(0, 1, 10000)
    >>> specific_bh_acc_rate = monte_carlo_specific_bh_acc_rate(redshift, sfr_percentile)

    """
    redshift = np.atleast_1d(redshift)
    msg = ("monte_carlo_specific_bh_acc_rate only accepts "
        "a single float for ``redshift`` argument")
    assert len(redshift) == 1, msg

    rate_table, prob_table = eddington_ratio_distribution(redshift[0])
    return monte_carlo_from_cdf_lookup(
        rate_table, prob_table, mc_input=1-sfr_percentile)

