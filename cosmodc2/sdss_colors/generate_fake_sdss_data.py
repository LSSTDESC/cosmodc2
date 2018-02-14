"""
"""
import numpy as np
from scipy.spatial import cKDTree
from halotools.empirical_models import polynomial_from_table


__all__ = ('mock_magr', )


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


def sdss_selection_indices(mstar_mock, sfr_percentile_mock, logsm_sdss, sfr_percentile_sdss):
    """ Use scipy cKDTree to find nearest matching SDSS galaxies
    """
    sdss_tree = cKDTree(np.vstack((logsm_sdss, sfr_percentile_sdss)).T)

    nn_distinces, nn_indices = sdss_tree.query(
        np.vstack((np.log10(mstar_mock), sfr_percentile_mock)).T, k=1)

    return nn_distinces, nn_indices


def extrapolate_faint_end_magr(mstar_mock, p0, p1, magr_scatter):
    """ Power law extrapolation for the median restframe absolute r-band magnitude
    as a function of stellar mass.
    """
    median_magr = p0 + p1*np.log10(mstar_mock)
    return np.random.normal(loc=median_magr, scale=magr_scatter)


def mock_magr(mstar_mock, sfr_percentile_mock,
            logsm_sdss, sfr_percentile_sdss, magr_sdss, redshift_sdss,
            p0=0.6, p1=-2.02, scatter1=0.5, scatter2=0.5, z0=0.05):
    """ Generate a Monte Carlo realization of r-band Absolute magnitude
    """
    data_source = assign_data_source(np.log10(mstar_mock))

    mock_mask0 = data_source == 0
    mstar_mock0 = mstar_mock[mock_mask0]
    sfr_percentile_mock0 = sfr_percentile_mock[mock_mask0]
    sdss_mask0 = redshift_sdss < z0
    data_source0_dist, data_source0_idx = sdss_selection_indices(
        mstar_mock0, sfr_percentile_mock0, logsm_sdss[sdss_mask0], sfr_percentile_sdss[sdss_mask0])

    mock_mask1 = data_source == 1
    mstar_mock1 = mstar_mock[mock_mask1]
    magr_scatter = np.interp(np.log10(mstar_mock), [6, 8.5], [scatter1, scatter2])
    magr_scatter1 = magr_scatter[mock_mask1]
    extrapolated_magr = extrapolate_faint_end_magr(mstar_mock1, p0, p1, magr_scatter1)

    magr = np.zeros_like(mstar_mock)
    magr[mock_mask0] = magr_sdss[sdss_mask0][data_source0_idx]
    magr[mock_mask1] = extrapolated_magr

    return magr
