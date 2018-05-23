"""
"""
import numpy as np
from halotools.empirical_models import polynomial_from_table
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch


__all__ = ('mock_magr', 'assign_data_source',
        'median_magr_from_mstar', 'dim_satellites', 'mock_magr_from_sdss_draws')

default_seed = 43


def median_magr_from_mstar(log_mstar,
        x_table=[9.5, 10.25, 11.5], y_table=[-18.8, -20.1, -22.6],
        faint_end_x=6, faint_end_y=-15):
    """
    """
    result = polynomial_from_table(x_table, y_table, log_mstar)
    mask = log_mstar < x_table[0]
    faint_end_result = np.interp(log_mstar,
        [faint_end_x, x_table[0]], [faint_end_y, y_table[0]])
    result[mask] = faint_end_result[mask]
    return result


def satellite_dimming_selection(log_mhost, upid, log_mhost_table, dimprob):
    """
    """
    prob = np.ones_like(log_mhost)
    satmask = upid != -1
    prob[satmask] = np.interp(log_mhost[satmask], log_mhost_table, dimprob)
    return np.random.rand(len(upid)) < prob


def satellite_dimming_factor(log_mpeak, upid, mpeak_table, dimming_mag_table):
    """
    """
    mask = upid != -1
    dimming_factor = np.zeros_like(log_mpeak)
    dimming_factor[mask] = np.interp(
        log_mpeak[mask], mpeak_table, dimming_mag_table)
    return dimming_factor


def dim_satellites(magr, log_mpeak, log_mhost, upid,
        log_mhost_table=[12, 13.5, 15], dimprob=[0, 0.25, 1.],
        mpeak_table=[11.5, 12.25, 13], dimming_mag_table=[0.75, 0.35, 0.]):
    """
    """
    dimming_mag = satellite_dimming_factor(
            log_mpeak, upid, mpeak_table, dimming_mag_table)
    dimming_mask = satellite_dimming_selection(log_mhost, upid, log_mhost_table, dimprob)

    magr[dimming_mask] += dimming_mag[dimming_mask]
    return magr


def assign_data_source(mock_logsm, table_abscissa=np.array([8.5, 9, 9.5, 10]),
            table_ordinates=np.array([1, 0.8, 0.35, 0]), seed=default_seed):
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

    seed : int, optional
        Random number seed. Default is default_seed, set at the top of
        the module where the function is defined.

    Returns
    -------
    data_source : ndarray
        Numpy array of shape (ngals, ) storing an integer indicating the data source
        from which colors will be drawn
    """
    input_abscissa = np.linspace(table_abscissa.min(), table_abscissa.max(), 100)
    ordinates = polynomial_from_table(table_abscissa, table_ordinates, input_abscissa)

    prob_fake = np.interp(mock_logsm, input_abscissa, ordinates)
    with NumpyRNGContext(seed):
        fake_mask = np.random.rand(len(mock_logsm)) < prob_fake
    data_source = np.zeros_like(fake_mask).astype(int)
    data_source[fake_mask] = 1

    return data_source


def sdss_selection_indices(mstar_mock, sfr_percentile_mock, logsm_sdss, sfr_percentile_sdss):
    """ For every galaxy in the mock, find the nearest matching SDSS galaxy

    Parameters
    ----------
    mstar_mock : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy stellar mass
        (in linear units assuming h=0.7)

    sfr_percentile_mock : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy conditional
        cumulative distribution Prob(< SFR | M*).
        Can be computed using the SlidingPercentile package.

    logsm_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy stellar mass
        (in log10 units assuming h=0.7)

    sfr_percentile_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy conditional
        cumulative distribution Prob(< SFR | M*).
        Can be computed using the SlidingPercentile package.

    Returns
    -------
    nn_distinces : ndarray
        Numpy array of shape (ngals_mock, ) storing the Euclidean distance
        to the nearest SDSS galaxy

    nn_indices : ndarray
        Numpy integer array of shape (ngals_mock, ) storing the indices of
        the nearest SDSS galaxies

    """
    from scipy.spatial import cKDTree

    sdss_tree = cKDTree(np.vstack((logsm_sdss, sfr_percentile_sdss)).T)

    nn_distinces, nn_indices = sdss_tree.query(
        np.vstack((np.log10(mstar_mock), sfr_percentile_mock)).T, k=1)

    return nn_distinces, nn_indices


def extrapolate_faint_end_magr(mstar_mock, p0, p1, magr_scatter, seed=default_seed):
    """ Power law extrapolation for the median restframe absolute r-band magnitude
    as a function of stellar mass.

    Parameters
    ----------
    mstar_mock : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy stellar mass
        (in linear units assuming h=0.7)

    p0 : float
        Intercept in the power law relation between M* and r-band luminosity.

    p1 : float
        Index in the power law relation between M* and r-band luminosity.

    magr_scatter : float or ndarray
        Float or Numpy array of shape (ngals_mock, ) storing
        level of scatter in the log-normal relation Prob(Mr | M*).

    seed : int, optional
        Random number seed. Default is default_seed, set at the top of
        the module where the function is defined.

    Returns
    -------
    magr : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy restframe
        r-band absolute magnitude
    """
    median_magr = p0 + p1*np.log10(mstar_mock)
    with NumpyRNGContext(seed):
        return np.random.normal(loc=median_magr, scale=magr_scatter)


def mock_magr_from_sdss_draws(mstar_mock, sfr_percentile_mock,
            logsm_sdss, sfr_percentile_sdss, magr_sdss, redshift_sdss,
            p0=0.6, p1=-2.02, scatter1=0.5, scatter2=0.5, z0=0.05, seed=default_seed):
    """ Generate a Monte Carlo realization of r-band Absolute magnitude by
    empirically drawing from SDSS data (with an extrapolation at the faint end)

    Parameters
    ----------
    mstar_mock : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy stellar mass
        (in linear units assuming h=0.7)

    sfr_percentile_mock : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy conditional
        cumulative distribution Prob(< SFR | M*).
        Can be computed using the SlidingPercentile package.

    logsm_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy stellar mass
        (in log10 units assuming h=0.7)

    sfr_percentile_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy conditional
        cumulative distribution Prob(< SFR | M*).
        Can be computed using the SlidingPercentile package.

    magr_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy restframe r-band
        absolute magnitude (in units assuming h=0.7)

    redshift_sdss : ndarray
        Numpy array of shape (ngals_sdss, ) storing SDSS galaxy redshifts.
        This is used to make a completeness cut z < z0.

    p0 : float, optional
        Intercept in the power law relation between M* and r-band luminosity.
        Default is 0.6, which has been hand-tuned through visual inspection of SDSS data.

    p1 : float, optional
        Index in the power law relation between M* and r-band luminosity.
        Default is 2.02, which has been hand-tuned through visual inspection of SDSS data.

    scatter1 : float, optional
        Level of scatter in the log-normal relation Prob(Mr | M*=10**6).
        Default is 0.5 dex, which has been hand-tuned through visual inspection of SDSS data.

    scatter2 : float, optional
        Level of scatter in the log-normal relation Prob(Mr | M*=10**8.5).
        Default is 0.5 dex, which has been hand-tuned through visual inspection of SDSS data.

    z0 : float, optional
        Redshift used to mask SDSS galaxies. Default is 0.05.

    seed : int, optional
        Random number seed. Default is default_seed, set at the top of
        the module where the function is defined.

    Returns
    -------
    mock_magr : ndarray
        Numpy array of shape (ngals_mock, ) storing mock galaxy restframe
        SDSS r-band absolute magnitude (assuming h=0.7).
    """
    data_source = assign_data_source(np.log10(mstar_mock), seed=seed)

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
    extrapolated_magr = extrapolate_faint_end_magr(mstar_mock1, p0, p1, magr_scatter1, seed=seed)

    magr = np.zeros_like(mstar_mock)
    magr[mock_mask0] = magr_sdss[sdss_mask0][data_source0_idx]
    magr[mock_mask1] = extrapolated_magr

    return magr


def prob_remap_bcg(upid, host_halo_mass,
            mhalo_table=(13.5, 13.75, 14, 15), prob_table=(0, 0.1, 0.5, 1)):
    """
    """
    ngals = len(upid)

    prob_remap = np.interp(np.log10(host_halo_mass), mhalo_table, prob_table)
    uran = np.random.rand(ngals)
    uran[upid != -1] = 1.0
    return uran < prob_remap


def prob_remap_highsm(upid, logsm,
            logsm_table=(11, 11.25, 11.5), prob_table=(0, 0.5, 1)):
    """
    """
    ngals = len(upid)

    prob_remap = np.interp(logsm, logsm_table, prob_table)
    uran = np.random.rand(ngals)
    return uran < prob_remap


def remap_bcg_and_high_mass(upid, host_halo_mvir, logsm, magr):
    """
    """

    remap_bcg = prob_remap_bcg(upid, host_halo_mvir)
    remap_highsm = prob_remap_highsm(upid, logsm)
    remapping_mask = remap_bcg | remap_highsm
    num_to_remap = np.count_nonzero(remapping_mask)

    c0, c1 = (-0.178, -1.935)
    median_new_magr2 = c0 + c1*logsm

    new_magr = magr
    new_magr[remapping_mask] = np.random.normal(
        loc=median_new_magr2[remapping_mask], scale=0.35, size=num_to_remap)

    return new_magr


def cam_rematch_magr(mstar, old_magr, new_magr, nwin=301):
    """
    """
    return conditional_abunmatch(mstar, old_magr, mstar, new_magr, nwin)


def mock_magr(upid_mock, mstar_mock, sfr_percentile_mock, host_halo_mvir_mock,
            logsm_sdss, sfr_percentile_sdss, magr_sdss, redshift_sdss):
    """
    """
    magr_mock_from_data = mock_magr_from_sdss_draws(mstar_mock, sfr_percentile_mock,
                logsm_sdss, sfr_percentile_sdss, magr_sdss, redshift_sdss)

    logsm_mock = np.log10(mstar_mock)
    remapped_magr_mock = remap_bcg_and_high_mass(
        upid_mock, host_halo_mvir_mock, logsm_mock, magr_mock_from_data)

    cam_smoothed_magr_mock = cam_rematch_magr(
        mstar_mock, magr_mock_from_data, remapped_magr_mock)

    return cam_smoothed_magr_mock
