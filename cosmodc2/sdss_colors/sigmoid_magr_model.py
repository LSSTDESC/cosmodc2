"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


default_logsm_table = np.array((6, 7, 8, 9, 10, 11, 12)).astype('f4')
default_magr_at_m0_ztable = [0.1, 0.25, 0.5, 1]
default_magr_at_m0_table = [-20.1, -20.55, -20.7, -20.7]
fixed_seed = 43

__all__ = ('magr_monte_carlo', )


def sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))


def magr_monte_carlo(mstar, upid, redshift, scatter=0.15, **kwargs):
    """
    """
    mstar = np.atleast_1d(mstar)
    upid = np.atleast_1d(upid)
    redshift = np.atleast_1d(redshift)

    median_magr = median_magr_from_mstar(mstar, upid, redshift, **kwargs)
    with NumpyRNGContext(fixed_seed):
        result = np.random.normal(loc=median_magr, scale=scatter)
    return result


def median_magr_from_mstar(mstar, upid, redshift,
            magr_at_m0_ztable=default_magr_at_m0_ztable,
            magr_at_m0_table=default_magr_at_m0_table, **kwargs):
    """
    """
    logsm = np.log10(mstar)
    magr_at_m0 = np.interp(redshift, magr_at_m0_ztable, magr_at_m0_table)
    __ = kwargs.setdefault('magr_at_m0', magr_at_m0)
    magr_z0 = median_magr_from_mstar_z0(logsm, **kwargs)
    delta_magr = delta_magr_vs_mstar_redshift(logsm, redshift, **kwargs)
    return magr_z0 + delta_magr


def median_magr_from_mstar_z0(logsm, logm0=10., magr_at_m0=-20.1,
            low_mass_slope=1.75, high_mass_slope=1.8, logsm_k=2.5, **kwargs):
    """
    """
    x = np.atleast_1d(logsm) - logm0
    slope = sigmoid(x, x0=0, ymin=low_mass_slope, ymax=high_mass_slope, k=logsm_k)
    return magr_at_m0 - slope*x


def delta_magr_vs_mstar_redshift(logsm, redshift, dmin=-6.5, **kwargs):
    dmagr_highz = delta_magr_highz_vs_mstar(logsm, **kwargs)
    z_crit = z_crit_vs_mstar(logsm, **kwargs)
    delta_magr = delta_magr_vs_redshift(redshift, z_crit, dmagr_highz)
    delta_magr = np.where(delta_magr > 0, 0., delta_magr)
    delta_magr = np.where(delta_magr < dmin, dmin, delta_magr)
    return delta_magr


def delta_magr_vs_redshift(z, z_crit, dmagr_highz, redshift_k=10, **kwargs):
    return sigmoid(z, x0=z_crit, ymin=0, ymax=dmagr_highz, k=redshift_k)


def z_crit_vs_mstar(
        logsm, z_crit_table=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], **kwargs):
    x = np.append(default_logsm_table, [13, 14, 15])
    zhigh = z_crit_table[-1]
    y = np.append(z_crit_table, [zhigh, zhigh, zhigh])
    c3, c2, c1, c0 = np.polyfit(x, y, deg=3)
    return c0 + c1*logsm + c2*logsm**2 + c3*logsm**3


def delta_magr_highz_vs_mstar(
        logsm, delta_magr_highz_table=[-2, -2, -1.5, -1, -0.5, -0.25, 0.], **kwargs):
    x = np.append(default_logsm_table, [13, 14, 15])
    zhigh = delta_magr_highz_table[-1]
    y = np.append(delta_magr_highz_table, [zhigh, zhigh, zhigh])
    c3, c2, c1, c0 = np.polyfit(x, y, deg=3)
    return c0 + c1*logsm + c2*logsm**2 + c3*logsm**3
