"""
"""
from astropy.utils.misc import NumpyRNGContext
import numpy as np


__all__ = ('bh_mass_from_bulge_mass', 'monte_carlo_black_hole_mass')
fixed_seed = 43


def bh_mass_from_bulge_mass(bulge_mass):
    """
    Kormendy & Ho (2013) fitting function for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass of the bulge
        in units of solar mass assuming h=0.7

    Returns
    -------
    bh_mass : ndarray
        Numpy array of shape (ngals, ) storing black hole mass

    Examples
    --------
    >>> ngals = int(1e4)
    >>> bulge_mass = np.logspace(8, 12, ngals)
    >>> bh_mass = bh_mass_from_bulge_mass(bulge_mass)
    """
    prefactor = 0.49*(bulge_mass/100.)
    return prefactor*(bulge_mass/1e11)**0.15


def monte_carlo_black_hole_mass(bulge_mass, seed=fixed_seed):
    """
    Monte Carlo realization of the Kormendy & Ho (2013) fitting function
    for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass of the bulge
        in units of Msun assuming h=0.7

    seed : int, optional
        Random number seed in the Monte Carlo. Default is 43.

    Returns
    -------
    bh_mass : ndarray
        Numpy array of shape (ngals, ) storing black hole mass in units of
        Msun assuming h=0.7

    Examples
    --------
    >>> ngals = int(1e4)
    >>> bulge_mass = np.logspace(8, 12, ngals)
    >>> bh_mass = bh_mass_from_bulge_mass(bulge_mass)

    """
    loc = np.log10(bh_mass_from_bulge_mass(bulge_mass))
    with NumpyRNGContext(seed):
        return 10**np.random.normal(loc=loc, scale=0.28)

