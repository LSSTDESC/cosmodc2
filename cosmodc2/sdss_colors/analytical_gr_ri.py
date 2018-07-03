"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch
from .sigmoid_g_minus_r import g_minus_r
from .sigmoid_r_minus_i import r_minus_i

from .sigmoid_g_minus_r import quiescent_fraction_gr
from .sigmoid_r_minus_i import quiescent_fraction_ri


__all__ = ('gr_ri_monte_carlo',)
fixed_seed = 43


def gr_ri_monte_carlo(magr, sfr_percentile, redshift,
            local_random_scale=0.1, nwin=301, seed=fixed_seed, **kwargs):
    """ Generate a Monte Carlo realization of (g-r) and (r-i) restframe colors.

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, ) storing the
        r-band absolute restframe magnitude of every galaxy

    sfr_percentile : ndarray
        Numpy array of shape (ngals, ) storing the SFR-percentile
        of each galaxy, Prob(< SFR | M*). This quantity can be calculated using
        the halotools.utils.sliding_conditional_percentile function.

    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of each galaxy

    Returns
    -------
    gr : ndarray
        Numpy array of shape (ngals, ) storing g-r restframe color for every galaxy

    ri : ndarray
        Numpy array of shape (ngals, ) storing r-i restframe color for every galaxy

    is_quiescent_ri : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the r-i red sequence

    is_quiescent_gr : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the g-r red sequence

    """
    with NumpyRNGContext(seed):
        p = np.random.normal(loc=1-sfr_percentile, scale=local_random_scale)

    ri_orig, is_quiescent_ri = r_minus_i(magr, redshift, seed=seed, **kwargs)
    gr_orig, is_quiescent_gr = g_minus_r(magr, redshift, seed=seed, **kwargs)
    gr = conditional_abunmatch(magr, p, magr, np.copy(gr_orig), nwin)

    with NumpyRNGContext(seed):
        noisy_gr = np.random.normal(loc=gr, scale=0.1)
    ri = conditional_abunmatch(magr, noisy_gr, magr, np.copy(ri_orig), nwin)

    fq_gr_model = quiescent_fraction_gr(magr, redshift)
    fq_ri_model = quiescent_fraction_ri(magr, redshift)
    is_quiescent_gr = sfr_percentile < fq_gr_model
    is_quiescent_ri = sfr_percentile < fq_ri_model

    return gr, ri, is_quiescent_ri, is_quiescent_gr
