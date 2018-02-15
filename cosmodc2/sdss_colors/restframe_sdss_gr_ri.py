""" Functions used to assign restframe g-r and r-i SDSS colors to mock galaxies.
"""
import numpy as np
from .sawtooth_binning import sawtooth_bin_indices
from astropy.utils.misc import NumpyRNGContext


__all__ = ('fuzzy_sawtooth_magr_binning', )


def fuzzy_sawtooth_magr_binning(mock_rmag, data_source, magr_bins=None, seed=None):
    """ Assign galaxies to overlapping bins based on their restframe absolute r-band magnitude.

    Binning will be done separately for mock galaxies above and below the SDSS completeness limit.

    Parameters
    ----------
    mock_rmag : ndarray
        Numpy array of shape (ngals, ) storing the
        restframe absolute r-band magnitude of mock galaxies

    data_source : ndarray
        Numpy array of shape (ngals, ) storing an integer indicating the data source
        from which colors will be drawn (zero for real data, one for fake data).

    magr_bins : ndarray, optional
        Numpy array of shape (nbins, ) storing the bin boundaries.
        Must strictly encompass the range spanned by the mock.
        Default is to use 25 bins linearly spaced in Magr covering a range that
        just beyond the boundaries of the data.

    seed : int, optional
        Random number seed. Default is None, for stochastic results.

    Returns
    -------
    rmag_bin_number : ndarray
        Numpy integer array of shape (ngals, ) storing the bin number of each mock galaxy
    """
    if magr_bins is None:
        epsilon = 0.01
        rmin, rmax, dr = mock_rmag.min()-epsilon, mock_rmag.max()+epsilon, 0.25
        magr_bins = np.arange(rmin, rmax+dr, dr)

    source0_mask = data_source == 0
    rmag_bin_number = -np.ones_like(mock_rmag).astype('i4')
    rmag_bin_number[source0_mask] = sawtooth_bin_indices(mock_rmag[source0_mask], magr_bins, seed=seed)
    rmag_bin_number[~source0_mask] = sawtooth_bin_indices(mock_rmag[~source0_mask], magr_bins, seed=seed)
    return rmag_bin_number


def shift_gr_ri_colors_at_high_redshift(gr, ri, redshift):
    """ Apply a simple multiplicative shift to the g-r and r-i color distributions
    to crudely mock up redshift evolution in the colors.

    Parameters
    ----------
    gr : ndarray
        Array of shape (ngals, ) storing the g-r colors

    ri : ndarray
        Array of shape (ngals, ) storing the r-i colors

    redshift : float
        Redshift of the snapshot

    Returns
    -------
    gr_new : ndarray
        Array of shape (ngals, ) storing the shifted g-r colors

    ri_new : ndarray
        Array of shape (ngals, ) storing the shifted r-i colors

    Examples
    --------
    >>> gr = np.random.uniform(0, 1.25, 1000)
    >>> ri = np.random.uniform(0.25, 0.75, 1000)
    >>> gr_new, ri_new = shift_gr_ri_colors_at_high_redshift(gr, ri, 0.8)
    >>> gr_new, ri_new = shift_gr_ri_colors_at_high_redshift(gr, ri, 8.)
    """
    gr_shift = np.interp(redshift, [0, 0.3, 1], [1., 1.15, 1.3])
    ri_shift = np.interp(redshift, [0, 0.3, 1], [1., 1.05, 1.1])
    gr_new = gr/gr_shift
    ri_new = ri/ri_shift
    return gr_new, ri_new
