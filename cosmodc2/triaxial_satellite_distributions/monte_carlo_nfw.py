"""See https://arxiv.org/abs/1805.09550
"""
import numpy as np
from scipy import special
from scipy.integrate import quad as quad_integration


def nfw_profile_realization(conc, seed=43):
    """Generate a random realization of a dimensionless NFW profile
    according to the input concentration.

    Parameters
    ----------
    conc : ndarray
        Array of shape (npts, ) storing NFW concentration of the host halo

    Returns
    -------
    r : ndarray
        Array of shape (npts, ) storing radial distances, 0 < r < 1
    """
    conc = np.atleast_1d(conc)
    n = int(conc.size)
    rng = np.random.RandomState(seed)
    uran = rng.rand(n)
    return _qnfw(uran, conc=conc)


def _pnfwunorm(q, conc):
    """
    """
    y = q*conc
    return np.log(1.0 + y)-y/(1.0 + y)


def _qnfw(p, conc, logp=False):
    """
    """
    p[p>1] = 1
    p[p<=0] = 0
    p *= _pnfwunorm(1, conc)
    return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/conc


def _jeans_integrand_term1(y):
    r"""
    """
    return np.log(1+y)/(y**3*(1+y)**2)


def _jeans_integrand_term2(y):
    r"""
    """
    return 1/(y**2*(1+y)**3)


def _g_integral(x):
    """
    """
    x = np.atleast_1d(x).astype(np.float64)
    return np.log(1.0+x) - (x/(1.0+x))


def _nfw_velocity_dispersion_table(scaled_radius_table, conc, tol=1e-5):
    """
    """
    x = np.atleast_1d(scaled_radius_table).astype(np.float64)
    result = np.zeros_like(x)

    prefactor = conc*(conc*x)*(1. + conc*x)**2/_g_integral(conc)

    lower_limit = conc*x
    upper_limit = float("inf")
    for i in range(len(x)):
        term1, __ = quad_integration(_jeans_integrand_term1,
            lower_limit[i], upper_limit, epsrel=tol)
        term2, __ = quad_integration(_jeans_integrand_term2,
            lower_limit[i], upper_limit, epsrel=tol)
        result[i] = term1 - term2

    dimless_velocity_table = np.sqrt(result*prefactor)
    return dimless_velocity_table
