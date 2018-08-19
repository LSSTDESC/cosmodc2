"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch
from .probabilistic_binning import fuzzy_digitize

from .sigmoid_g_minus_r import g_minus_r
from .sigmoid_r_minus_i import r_minus_i

from .sigmoid_g_minus_r import quiescent_fraction_gr
from .sigmoid_r_minus_i import quiescent_fraction_ri


__all__ = ('gr_ri_monte_carlo',)
fixed_seed = 43


def gr_ri_monte_carlo(magr, sfr_percentile, redshift,
            local_random_scale=0.1, nwin=201, seed=fixed_seed, **kwargs):
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
    print('.....Check: in gr_ri_monte_carlo using seed {} and {} galaxies'.format(seed, len(redshift)))

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


def gr_ri_monte_carlo_substeps(magr, sfr_percentile, redshift, nzdivs=6,
            nwin=201, nwin_min=21, **kwargs):
    """
    """
    print('.....Check: in gr_ri_monte_carlo_substeps with supplied kwarg seed {}'.format(kwargs.get('seed', -1)))
    print("\n...running gr_ri_monte_carlo_substeps with nzdivs = {0}".format(nzdivs))
    ngals = len(magr)
    sorted_redshift = np.sort(redshift)
    _zbins = sorted_redshift[np.arange(nzdivs+4)*int(ngals/float(nzdivs+4))]
    _zbins[0] = redshift.min() - 0.001
    _zbins[-1] = redshift.max() + 0.001
    zbin_edges = np.fromiter(
        (z for i, z in enumerate(_zbins)
            if i not in (1, 2, len(_zbins)-3, len(_zbins)-2)), dtype=float)
    idx = fuzzy_digitize(redshift, zbin_edges, min_counts=nwin_min + 1)

    gr_substeps = np.zeros_like(idx).astype('f4')
    ri_substeps = np.zeros_like(idx).astype('f4')
    is_quiescent_ri_substeps = np.zeros_like(idx).astype(bool)
    is_quiescent_gr_substeps = np.zeros_like(idx).astype(bool)
    for i in np.unique(idx):
        binmask = idx == i
        num_ibin = np.count_nonzero(binmask)
        if nwin_min <= num_ibin < nwin:
            print("...decreasing nwin from {0} to {1}".format((nwin, num_ibin)))
            nwin = max(num_ibin, nwin_min)
            nwin = 2*(nwin/2) + 1
        elif num_ibin < nwin_min:
            msg = ("Why are there only {0} galaxies in this call "
                "to gr_ri_monte_carlo_substeps?\n"
                "i = {1}; zbin_edges[i] = {2}\n"
                "zbin_edges = {3}\n"
                "redshift.min() = {4}\n"
                "redshift.max() = {5}")
            raise ValueError(msg.format((num_ibin, i, zbin_edges[i], zbin_edges,
                redshift.min(), redshift.max())))
        gr_temp, ri_temp, is_quiescent_ri_temp, is_quiescent_gr_temp = gr_ri_monte_carlo(
            magr[binmask], sfr_percentile[binmask], redshift[binmask], nwin=nwin, **kwargs)
        gr_substeps[binmask] = gr_temp
        ri_substeps[binmask] = ri_temp

        is_quiescent_ri_substeps[binmask] = is_quiescent_ri_temp
        is_quiescent_gr_substeps[binmask] = is_quiescent_gr_temp

    return gr_substeps, ri_substeps, is_quiescent_ri_substeps, is_quiescent_gr_substeps
