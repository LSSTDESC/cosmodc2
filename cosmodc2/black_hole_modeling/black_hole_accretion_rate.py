"""
"""
import numpy as np
from halotools.utils import monte_carlo_from_cdf_lookup
from halotools.empirical_models import conditional_abunmatch


A = 10**-3.15
gamma_e = -0.65
gamma_z = 3.47
z0 = 0.6


__all__ = ('monte_carlo_specific_bh_acc_rate', )


def specific_bh_acc_rate_distribution_table(redshift, npts=1000):
    """
    """
    rate_table = np.logspace(-1, 0, npts)
    return rate_table, A*(((1. + redshift)/(1. + z0))**gamma_z)*rate_table**gamma_e


def monte_carlo_specific_bh_acc_rate(redshift, mstar, sfr):
    """
    """
    ngals = len(mstar)
    rate_table, prob_table = specific_bh_acc_rate_distribution_table(redshift)
    mc_specific_acc_rates = monte_carlo_from_cdf_lookup(
        rate_table, prob_table, num_draws=ngals)

    nwin = 101
    correlated_specific_acc_rates = conditional_abunmatch(
        mstar, sfr, mstar, mc_specific_acc_rates, nwin)
    raise NotImplementedError()
