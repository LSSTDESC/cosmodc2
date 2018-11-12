""" Functions used to generate uniform random points distributed in a lightcone
"""
import numpy as np
import healpy as hp
from astropy.utils.misc import NumpyRNGContext


def generate_healpixel_randoms(npts, *args, **kwargs):
    """
    Generate a Monte Carlo realization of points randomly distributed throughout a healpixel.
    The redshift distribution of the returned points will be uniform random in cosmological volume

    Parameters
    ----------
    npts : int

    Returns
    -------
    ra, dec, z : ndarrays
        Numpy arrays of shape (npts, ) storing the random points
    """
    raise NotImplementedError()



