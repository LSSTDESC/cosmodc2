"""Module implementing the monte_carlo_halo_shapes function that generates
distributions of halo axis ratios that agree reasonably well with simulations.
See demo_shape_model.ipynb for some validation plots.
"""
import numpy as np
from scipy.stats import gengamma


emin, emax = 0, 0.5
pmin, pmax = -0.25, 0.5
c0, c1 = -0.18, 1.35


def monte_carlo_halo_shapes(logmhalo, floor=0.05):
    """Generate axis ratio distributions as a function of halo mass.

    Parameters
    ----------
    logmhalo : ndarray
        Array of shape (npts, ) storing log halo mass

    floor : float, optional
        Floor to impose on the axis ratios. Default is 0.05.

    Returns
    -------
    b_to_a : ndarray
        Array of shape (npts, ) storing halo B/A

    c_to_a : ndarray
        Array of shape (npts, ) storing halo C/A

    e : ndarray
        Array of shape (npts, ) storing halo ellipticity
        e = (1 - c**2)/2L, where L = 1 + b**2 + c**2
        Defined according to Equation 9 of https://arxiv.org/abs/1109.3709

    p : ndarray
        Array of shape (npts, ) storing halo prolaticity
        p = (1 - 2b**2 + c**2)/2L, where L = 1 + b**2 + c**2
        Defined according to Equation 9 of https://arxiv.org/abs/1109.3709

    """
    b_to_a, c_to_a = monte_carlo_axis_ratios(logmhalo, floor=floor)
    s = 1. + b_to_a**2 + c_to_a**2
    e = (1. - c_to_a**2)/2./s
    p = (1. - 2*b_to_a**2 + c_to_a**2)/2./s
    return b_to_a, c_to_a, e, p


def _sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))


def monte_carlo_b_to_a(logmhalo):
    a, c = _get_gengamma_b_to_a_params(logmhalo)
    return gengamma_b_to_a(a, c)


def gengamma_b_to_a(a, c):
    r = 1-1./(1 + gengamma.rvs(a, c))
    return r


def _get_gengamma_b_to_a_params(logmhalo):
    a = _sigmoid(logmhalo, x0=12.5, k=2, ymin=3.25, ymax=2.25)
    c = _sigmoid(logmhalo, x0=12.5, k=2, ymin=0.5, ymax=1.25)
    return a, c


def _get_gengamma_c_to_b_params(logmhalo):
    a = _sigmoid(logmhalo, x0=12.5, k=2, ymin=3, ymax=2)
    c = _sigmoid(logmhalo, x0=12.25, k=2, ymin=-5, ymax=-6)
    return a, c


def monte_carlo_c_to_b(logmhalo):
    a, c = _get_gengamma_c_to_b_params(logmhalo)
    return 1.65-gengamma.rvs(a, c)


def monte_carlo_axis_ratios(logmhalo, floor=0.05):
    """
    """
    b_to_a = monte_carlo_b_to_a(logmhalo)
    b_to_a = np.where(b_to_a < floor, floor, b_to_a)
    c_to_b = monte_carlo_c_to_b(logmhalo)
    c_to_b = np.where(c_to_b > 1, 1, c_to_b)
    c_to_a = c_to_b*b_to_a
    c_to_a = np.where(c_to_a < floor, floor, c_to_a)
    c_to_a = np.where(c_to_a > b_to_a, b_to_a, c_to_a)
    return b_to_a, c_to_a


def calculate_ellipticity_prolaticity_from_axis_ratios(b, c):
    """
    """
    b = np.atleast_1d(b)
    c = np.atleast_1d(c)
    assert np.all(b > 0), "b must be strictly positive"
    assert np.all(b <= 1), "b cannot exceed unity"
    assert np.all(c > 0), "c must be strictly positive"
    assert np.all(b >= c), "c cannot exceed b"

    lam = 1. + b**2 + c**2
    num = 1. - c**2
    denom = 2*lam
    e = num/denom
    p = (1. - 2*b**2 + c**2)/denom
    return e, p


def calculate_axis_ratios_from_ellipticity_prolaticity(e, p):
    """
    """
    e = np.atleast_1d(e)
    p = np.atleast_1d(p)

    zero_ellipticity_mask = e == 0

    num1 = (p+1)*(2*e-1)
    num1[~zero_ellipticity_mask] = num1[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]
    num1[zero_ellipticity_mask] = 0.

    num2 = 2*p-1.
    num = num1 - num2
    num[zero_ellipticity_mask] = 0.

    denom1 = 2*p-1
    denom2 = (p+1.)*(2.*e+1)
    denom2[~zero_ellipticity_mask] = denom2[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]

    denom = denom1 - denom2
    csq = num/denom
    csq[zero_ellipticity_mask] = 1.

    prefactor = np.zeros_like(e) - 1/2.
    prefactor[~zero_ellipticity_mask] = prefactor[~zero_ellipticity_mask]/e[~zero_ellipticity_mask]

    term1 = 2*e-1
    term2 = (2*e+1)*csq
    bsq = prefactor*(term1 + term2)
    bsq[zero_ellipticity_mask] = 1.

    return np.sqrt(bsq), np.sqrt(csq)

