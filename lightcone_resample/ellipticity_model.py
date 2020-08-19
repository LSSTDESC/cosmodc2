"""
"""
import numpy as np
from scipy.stats import johnsonsb
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch
from halotools.utils import rank_order_percentile

# def calculate_johnsonsb_params_disk(
#         magr, johnsonsb_disk_table_abscissa=[-21,-19],
#         johnsonsb_disk_table=[-0.15, 0.15], **kwargs):
#     return np.interp(magr, johnsonsb_disk_table_abscissa, johnsonsb_disk_table)


# def calculate_johnsonsb_params_bulge(
#         magr, johnsonsb_bulge_table_abscissa=[-21, -19],
#         johnsonsb_bulge_table=[0.6, 1.5], **kwargs):
#     return np.interp(magr, johnsonsb_bulge_table_abscissa, johnsonsb_bulge_table)


# def monte_carlo_ellipticity_disk(magr, inclination = None, seed=None, **kwargs):
#     """
#     Parameters
#     ----------
#     magr : ndarray
#         Numpy array of shape (ngals, )

#     inclination : ndarray
#         Numpy array of shape (ngals, )

#     Returns
#     -------
#     ellipticity_realization : ndarray
#     """
    
    
#     magr = np.atleast_1d(magr)
#     inclination = np.atleast_1d(inclination)

#     a = calculate_johnsonsb_params_disk(magr, **kwargs)
#     b = np.ones_like(a)

#     with NumpyRNGContext(seed):
#         ellipticity_realization = johnsonsb.rvs(a, b)

#     nwin = 101
#     if inclination is None:
#         inclination_correlated_ellipticity = conditional_abunmatch(
#             magr, inclination, magr, ellipticity_realization, nwin)
#         return inclination_correlated_ellipticity
#     else:
#         return ellipticity_realization


# def monte_carlo_ellipticity_bulge(magr, seed=None, **kwargs):
#     """
#     Parameters
#     ----------
#     magr : ndarray
#         Numpy array of shape (ngals, )

#     Returns
#     -------
#     ellipticity_realization : ndarray
#     """
#     magr = np.atleast_1d(magr)

#     a = calculate_johnsonsb_params_bulge(magr, **kwargs)
#     b = np.ones_like(a)

#     with NumpyRNGContext(seed):
#         ellipticity_realization = johnsonsb.rvs(a, b)
#     return ellipticity_realization


def monte_carlo_ellipticity_bulge_disk(magr, seed=None):
    
    magr = np.atleast_1d(magr)

    a_disk =  np.interp(magr, [-21,-19],[-0.4,-0.4])
    #a_disk = calculate_johnsonsb_params_disk(mag_r)
    b_disk = np.ones_like(a_disk)*0.7

    a_bulge = np.interp(magr, [-21,-19,-17],[.6,1.0,1.6])
    #a_bulge = calculate_johnsonsb_params_bulge(mag_r)
    b_bulge = np.interp(magr, [-19,-17],[1.0,1.0])
    #b_bulge = np.ones_like(a_bulge)

    with NumpyRNGContext(seed):
        urand = np.random.uniform(size=magr.size)
        urand2 = rank_order_percentile(1*urand + 0.6*np.random.uniform(size=magr.size))
        ellip_bulge = johnsonsb.isf(urand, a_bulge, b_bulge)
        ellip_disk =  johnsonsb.isf(urand2, a_disk, b_disk)
    return ellip_bulge, ellip_disk
    
