"""
Module storing the analytical model for SDSS restframe r-band absolute magnitude
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


__all__ = ('magr_monte_carlo', )

fixed_seed = 43


def redshift_boost_median_magr(redshift,
            z_table=[0, 0.25, 0.5, 1], boost_table=[0, -0.2, -0.5, -1.], **kwargs):
    """
    Calculate the redshift-dependent multiplicative factor by which the
    median <Mr | M*>(z) relation should be boosted relative to the z=0 relation.

    The boost factor model is defined via linear interpolation from a set of
    control points in redshift.

    Parameters
    ----------
    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of every galaxy in the sample

    z_table : sequence, optional
        Abscissa of the interpolation table defining the boost factor.
        Default is [0, 0.25, 0.5, 1].

    boost_table : sequence, optional
        Ordinates of the interpolation table defining the boost factor.

    Returns
    -------
    boost_factor : ndarray
        Numpy array of shape (ngals, ) storing the multiplicative boost factor

    Examples
    --------
    >>> ngals = int(1e4)
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> boost_factor = redshift_boost_median_magr(redshift)

    """
    return np.interp(redshift, z_table, boost_table)


def magr_at_m1_vs_redshift(redshift, magr_at_m1_z0, z_table, boost_table):
    """
    """
    return magr_at_m1_z0 + np.interp(redshift, z_table, boost_table)


def high_mass_slope_vs_redshift(redshift, beta_z0, slope_z_table, slope_boost_table):
    """
    """
    return beta_z0 + np.interp(redshift, slope_z_table, slope_boost_table)


def median_magr_from_mstar(mstar, upid, redshift,
            beta_z0=2.85, magr_at_m1_z0=-20.2, gamma=2.25, m1=10., beta_z0_satellites=2.7,
            slope_z_table=[0.25, 0.5, 1], slope_boost_table=[0, 0., 0.],
            z_table=[0, 0.25, 0.5, 1], boost_table=[0, -0.5, -1, -1.25], **kwargs):
    """ Double power-law model for the median of the scaling relation <Mr | M*>(z).

    Parameters
    ----------
    mstar : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass
         of every galaxy in the sample in units of Msun assuming h=0.7

    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of every galaxy in the sample

    beta : float, optional
        First power law index

    gamma : float, optional
        Second power law index

    m1 : float, optional
        Inflection point of the double power law

    magr_at_m1 : float, optional
        Normalization of the double power law defined by the median Mr evaluated
        at 10**m1.

    Returns
    -------
    median_magr : ndarray
        Numpy array of shape (ngals, ) storing the median restframe
        r-band absolute magnitude

    Examples
    --------
    >>> ngals = int(1e4)
    >>> mstar = 10**np.random.uniform(8, 12, ngals)
    >>> upid = np.zeros_like(mstar) - 1.
    >>> upid[-100:] = 100
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> median_magr = median_magr_from_mstar(mstar, upid, redshift)

    """
    m_by_m1 = mstar/10.**m1

    cenmask = upid == -1
    beta_cens = high_mass_slope_vs_redshift(
        redshift, beta_z0, slope_z_table, slope_boost_table)
    beta_sats = high_mass_slope_vs_redshift(
        redshift, beta_z0_satellites, slope_z_table, slope_boost_table)
    beta = beta_cens
    num_sats = np.count_nonzero(~cenmask)
    if num_sats > 0:
        beta[~cenmask] = beta_sats[~cenmask]

    denom_term1 = m_by_m1**beta
    denom_term2 = m_by_m1**gamma
    result = 1. / (denom_term1 + denom_term2)
    magr_at_m1 = magr_at_m1_vs_redshift(redshift, magr_at_m1_z0, z_table, boost_table)
    return np.log10(result*mstar) - np.log10(0.5*10**m1) + magr_at_m1


def scatter_magr_from_mstar(mstar, logsm_abscissa=[6, 8, 9],
                            scatter_ordinates=[0.5, 0.3, 0.15], **kwargs):
    """ Scatter about the median scaling relation <Mr | M*>(z).

    The scatter model is defined via linear interpolation from a set of
    control points in redshift.

    Parameters
    ----------
    mstar : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass
         of every galaxy in the sample in units of Msun assuming h=0.7

    logsm_abscissa : sequence, optional
        Abscissa of the interpolation table defining the scatter.
        Default is [6, 8, 9].

    scatter_ordinates : sequence, optional
        Ordinates of the interpolation table defining the scatter.

    Returns
    -------
    scatter_magr : ndarray
        Numpy array of shape (ngals, ) storing the Magr scatter

    Examples
    --------
    >>> ngals = int(1e4)
    >>> mstar = 10**np.random.uniform(8, 12, ngals)
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> scatter_magr = scatter_magr_from_mstar(mstar)
    """
    return np.interp(np.log10(mstar), logsm_abscissa, scatter_ordinates)


def magr_monte_carlo(mstar, upid, redshift, seed=fixed_seed, **kwargs):
    """ Monte Carlo realization of the scaling relation <Mr | M*>(z).

    Parameters
    ----------
    mstar : ndarray
        Numpy array of shape (ngals, ) storing the stellar mass
         of every galaxy in the sample in units of Msun assuming h=0.7

    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of every galaxy in the sample

    beta : float, optional
        First power law index

    gamma : float, optional
        Second power law index

    m1 : float, optional
        Inflection point of the double power law

    magr_at_m1 : float, optional
        Normalization of the double power law defined by the median Mr evaluated
        at 10**m1.

    seed : int, optional
        Random number seed. Default is 43.

    Returns
    -------
    magr : ndarray
        Numpy array of shape (ngals, ) storing the median restframe
        r-band absolute magnitude

    Notes
    -----
    The behavior of the magr_monte_carlo function is determined by
    the median <Mr | M*>(z) and its scatter, defined by the
    median_magr_from_mstar and scatter_magr_from_mstar functions, respectively.

    Examples
    --------
    >>> ngals = int(1e4)
    >>> mstar = 10**np.random.uniform(8, 12, ngals)
    >>> upid = np.zeros_like(mstar) - 1.
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> magr = magr_monte_carlo(mstar, upid, redshift)

    """
    median_magr = median_magr_from_mstar(mstar, upid, redshift, **kwargs)
    scatter_magr = scatter_magr_from_mstar(mstar, **kwargs)

    with NumpyRNGContext(seed):
        return np.random.normal(loc=median_magr, scale=scatter_magr)
