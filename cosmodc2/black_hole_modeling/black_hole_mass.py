"""
"""
import numpy as np


__all__ = ('bh_mass_from_bulge_mass', )


def bh_mass_from_bulge_mass(bulge_mass):
    """
    Kormendy & Ho (2013) fitting function for the Mbh--Mbulge power law relation.

    Parameters
    ----------
    bulge_mass : ndarray
        Numpy array of shape (ngals, ) storing the mass of the bulge in
        solar masses assuming h=0.7

    Returns
    -------
    bh_mass : ndarray
        Numpy array of shape (ngals, ) storing black hole mass
    """
    prefactor = 0.49*(bulge_mass/100.)
    return prefactor*(bulge_mass/1e11)**0.15
