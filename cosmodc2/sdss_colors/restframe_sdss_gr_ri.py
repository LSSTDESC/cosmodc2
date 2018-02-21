""" Functions used to assign restframe g-r and r-i SDSS colors to mock galaxies.
"""
import numpy as np
from .sawtooth_binning import sawtooth_bin_indices
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch
from scipy.spatial import cKDTree
from halotools.empirical_models import polynomial_from_table
from ..sdss_colors.sdss_completeness_model import retrieve_sdss_sample_mask


__all__ = ('mc_sdss_gr_ri', )


def assign_data_source(mock_logsm, table_abscissa=np.array([8.5, 9, 9.5, 10]),
            table_ordinates=np.array([1, 0.8, 0.35, 0])):
    """
    Determine the source of observational data that will be used
    to map colors onto mock galaxies.

    For entries in the returned ndarray equal to zero, real SDSS objects
    will be used to map colors onto mock galaxies.

    For entries in the returned ndarray equal to one, fake SDSS objects will be used.

    Parameters
    ----------
    mock_logsm : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass of mock galaxies

    table_abscissa : ndarray, optional
        Control points in log10 stellar mass at which the data source probability is defined

    table_ordinates : ndarray, optional
        Control points defining the probability that
        a mock galaxy will be assigned to data source one.

    Returns
    -------
    data_source : ndarray
        Numpy array of shape (ngals, ) storing an integer indicating the data source
        from which colors will be drawn
    """
    input_abscissa = np.linspace(table_abscissa.min(), table_abscissa.max(), 100)
    ordinates = polynomial_from_table(table_abscissa, table_ordinates, input_abscissa)

    prob_fake = np.interp(mock_logsm, input_abscissa, ordinates)
    fake_mask = np.random.rand(len(mock_logsm)) < prob_fake
    data_source = np.zeros_like(fake_mask).astype(int)
    data_source[fake_mask] = 1

    return data_source


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
    gr_shift = np.interp(redshift, [0.1, 0.3, 1], [1., 1.15, 1.3])
    ri_shift = np.interp(redshift, [0.1, 0.3, 1], [1., 1.05, 1.1])
    gr_new = gr/gr_shift
    ri_new = ri/ri_shift
    return gr_new, ri_new


def mc_true_sdss_gr_ri(sdss_redshift, sdss_magr, sdss_gr, sdss_ri,
        mock_magr_bin_number, mock_magr, mock_sfr_percentile, sigma=0.):
    """
    """
    mock_gr = np.zeros_like(mock_magr)
    mock_ri = np.zeros_like(mock_magr)

    true_color_bin_numbers = list(set(mock_magr_bin_number))
    for bin_number in true_color_bin_numbers:

        mock_magr_mask = mock_magr_bin_number == bin_number
        mock_magr_bin = mock_magr[mock_magr_mask]
        mock_sfr_percentile_bin = 1.-mock_sfr_percentile[mock_magr_mask]

        Mr_min, Mr_max = mock_magr_bin.min(), mock_magr_bin.max()
        sdss_mask = retrieve_sdss_sample_mask(sdss_redshift, sdss_magr, Mr_min, Mr_max)
        sdss_rmag_bin = sdss_magr[sdss_mask]
        sdss_gr_bin = sdss_gr[sdss_mask]
        sdss_ri_bin = sdss_ri[sdss_mask]

        mock_gr_bin = conditional_abunmatch(
            mock_sfr_percentile_bin, sdss_gr_bin,
            sigma=sigma, npts_lookup_table=np.count_nonzero(sdss_mask))

        sdss_tree = cKDTree(np.vstack((sdss_rmag_bin, sdss_gr_bin)).T)
        d, idx = sdss_tree.query(np.vstack((mock_magr_bin, mock_gr_bin)).T, k=1)

        mock_gr[mock_magr_mask] = mock_gr_bin
        mock_ri[mock_magr_mask] = sdss_ri_bin[idx]

    return mock_gr, mock_ri


def mc_fake_sdss_gr_ri(sdss_gr, sdss_ri, rmag_bin_number, mock_log10_mstar):
    """
    """
    gr_center = np.median(sdss_gr)-0.35
    ri_center = np.median(sdss_ri)-0.15
    median_gr = np.interp(mock_log10_mstar, [6, 9], [gr_center-0.2, gr_center])
    median_ri = np.interp(mock_log10_mstar, [6, 9], [ri_center-0.3, ri_center])
    median_array = np.vstack((median_gr, median_ri)).T

    ngals_mock = len(mock_log10_mstar)
    X = np.vstack((sdss_gr, sdss_ri))
    cov = np.cov(X)/2.

    Z = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=ngals_mock) + median_array
    mock_gr, mock_ri = Z[:, 0], Z[:, 1]
    return mock_gr, mock_ri


def mc_sdss_gr_ri(mock_rmag, mock_mstar, mock_sfr_percentile,
            sdss_redshift, sdss_magr, sdss_gr, sdss_ri):
    """
    """
    mock_data_source = assign_data_source(np.log10(mock_mstar))
    mock_rmag_bin_number = fuzzy_sawtooth_magr_binning(mock_rmag, mock_data_source)

    source0_mask = mock_data_source == 0

    mock_source0_gr, mock_source0_ri = mc_true_sdss_gr_ri(
        sdss_redshift, sdss_magr, sdss_gr, sdss_ri,
        mock_rmag_bin_number[source0_mask], mock_rmag[source0_mask],
        mock_sfr_percentile[source0_mask])

    mock_source1_gr, mock_source1_ri = mc_fake_sdss_gr_ri(sdss_gr, sdss_ri,
            mock_rmag_bin_number[~source0_mask], np.log10(mock_mstar[~source0_mask]))

    output_mock_gr = np.zeros_like(mock_rmag) - 999.
    output_mock_ri = np.zeros_like(mock_rmag) - 999.
    output_mock_gr[source0_mask] = mock_source0_gr
    output_mock_ri[source0_mask] = mock_source0_ri
    output_mock_gr[~source0_mask] = mock_source1_gr
    output_mock_ri[~source0_mask] = mock_source1_ri

    return output_mock_gr, output_mock_ri




