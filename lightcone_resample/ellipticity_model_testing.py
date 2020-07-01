"""
"""
import numpy as np
from scipy.stats import johnsonsb
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch


def calculate_johnsonsb_params_disk(
        magr, johnsonsb_disk_table_abscissa=[-19, -21],
        johnsonsb_disk_table=[0.15, -0.15], **kwargs):
    return np.interp(magr, johnsonsb_disk_table_abscissa, johnsonsb_disk_table)


def calculate_johnsonsb_params_bulge(
        magr, johnsonsb_bulge_table_abscissa=[-19, -21],
        johnsonsb_bulge_table=[1.5, 0.6], **kwargs):
    return np.interp(magr, johnsonsb_bulge_table_abscissa, johnsonsb_bulge_table)


def monte_carlo_ellipticity_disk(magr, inclination = None, seed=None, **kwargs):
    """
    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, )

    inclination : ndarray
        Numpy array of shape (ngals, )

    Returns
    -------
    ellipticity_realization : ndarray
    """

    magr = np.atleast_1d(magr)
    inclination = np.atleast_1d(inclination)

    a = calculate_johnsonsb_params_disk(magr, **kwargs)
    b = np.ones_like(a)

    with NumpyRNGContext(seed):
        ellipticity_realization = johnsonsb.rvs(a, b)

    nwin = 101
    if inclination is None:
        inclination_correlated_ellipticity = conditional_abunmatch(
            magr, inclination, magr, ellipticity_realization, nwin)
        return inclination_correlated_ellipticity
    else:
        return ellipticity_realization


def monte_carlo_ellipticity_bulge(magr, seed=None, **kwargs):
    """
    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, )

    Returns
    -------
    ellipticity_realization : ndarray
    """
    magr = np.atleast_1d(magr)

    a = calculate_johnsonsb_params_bulge(magr, **kwargs)
    b = np.ones_like(a)

    with NumpyRNGContext(seed):
        ellipticity_realization = johnsonsb.rvs(a, b)
    return ellipticity_realization

