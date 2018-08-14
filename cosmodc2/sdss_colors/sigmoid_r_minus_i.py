"""
Module storing the analytical model for SDSS restframe colors
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


#  Control points in r-band luminosity at which the (r-i) sequence peaks are defined
ms_peak_ri_abscissa = [-22.5, -21, -20, -19, -18, -15]
red_peak_ri_abscissa = ms_peak_ri_abscissa

#  Control points defining the locus of the (r-i) main sequence
default_ms_peak_ri = [0.4, 0.35, 0.3, 0.24, 0.2, 0.185]

#  Control points defining the locus of the (r-i) red sequence
default_red_peak_ri = [0.41, 0.41, 0.4, 0.385, 0.375, 0.35]

#  Control points defining the z-dependent blueshift of the locus of the (r-i) sequences
peak_shift_factor_z_table = [0.1, 0.35, 0.65, 1.0]
default_red_peak_ri_zevol = [0, -0.02, -0.05, -0.2]
default_ms_peak_ri_zevol = [0.0, -0.02, -0.1, -0.25]

#  Control points defining the magr-dependent (r-i) scatter at z = 0
default_ms_scatter_ri = [0.02, 0.05, 0.05, 0.05, 0.05, 0.05]
default_red_scatter_ri = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

#  Control points defining the z-evolution lf (r-i) scatter
default_red_scatter_ri_zevol_table = [1., 1., 1., 1.]
default_ms_scatter_ri_zevol_table = [1., 1., 1., 1.]
scatter_zevol_z_table = peak_shift_factor_z_table


default_mr_z0_fq_ri_pivot = -20
default_mr_z0_fq_ri_k = 1
default_fq_ri_z0_floor = 0.2
default_fq_ri_z0_ceil = 0.85

default_fq_ri_z1_floor = 0.2
default_fq_ri_z1_ceil = 0.25


def sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))


def fq_ri_z0_vs_magr(magr, mr_z0_fq_ri_pivot=default_mr_z0_fq_ri_pivot,
            mr_z0_fq_ri_k=default_mr_z0_fq_ri_k, fq_ri_z0_floor=default_fq_ri_z0_floor,
            fq_ri_z0_ceil=default_fq_ri_z0_ceil):
    return sigmoid(magr, x0=mr_z0_fq_ri_pivot,
                k=mr_z0_fq_ri_k, ymin=fq_ri_z0_ceil, ymax=fq_ri_z0_floor)


def fq_ri_z1_vs_magr(magr, mr_z1_fq_ri_pivot=default_mr_z0_fq_ri_pivot,
            mr_z1_fq_ri_k=default_mr_z0_fq_ri_k, fq_ri_z1_floor=default_fq_ri_z1_floor,
            fq_ri_z1_ceil=default_fq_ri_z1_ceil):
    return sigmoid(magr, x0=mr_z1_fq_ri_pivot,
                k=mr_z1_fq_ri_k, ymin=fq_ri_z1_ceil, ymax=fq_ri_z1_floor)


def quiescent_fraction_ri(magr, redshift):
    """
    Fraction of galaxies on the r-i red sequence as a function of Mr.
    """
    fq_z0 = fq_ri_z0_vs_magr(magr)
    fq_z1 = fq_ri_z1_vs_magr(magr)
    fq_at_z = sigmoid(redshift, x0=0.5, k=12, ymin=fq_z0, ymax=fq_z1)
    return fq_at_z


def red_sequence_width_ri(magr, red_scatter_ri, redshift, red_ri_scatter_zevol_table,
            x=red_peak_ri_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, red_scatter_ri)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, red_ri_scatter_zevol_table)
    return z0_scatter*zevol_factor


def main_sequence_ri_zevol_sigmoid_params(r):
    """
    """
    ymin = sigmoid(r, x0=-21, ymin=0.46, ymax=0.2, k=0.8)
    ymax = sigmoid(r, x0=-21., ymin=0.18, ymax=-0.04, k=1)
    return ymin, ymax


def red_sequence_ri_zevol_sigmoid_params(r):
    """
    """
    c0_ymin, c1_ymin = 0.2646, -0.0063
    ymin = c0_ymin + c1_ymin*r

    c0_ymax, c1_ymax = 0.0419, -0.00695
    ymax = c0_ymax + c1_ymax*r

    return ymin, ymax


def main_sequence_peak_ri(magr, redshift):
    """
    """
    ymin, ymax = main_sequence_ri_zevol_sigmoid_params(magr)
    return sigmoid(redshift, x0=0.7, k=7, ymin=ymin, ymax=ymax)


def red_sequence_peak_ri(magr, redshift):
    """
    """
    ymin, ymax = red_sequence_ri_zevol_sigmoid_params(magr)
    return sigmoid(redshift, x0=0.7, k=7, ymin=ymin, ymax=ymax)


def main_sequence_width_ri(magr, ms_scatter_ri, redshift, ms_ri_scatter_zevol_table,
            x=ms_peak_ri_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, ms_scatter_ri)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, ms_ri_scatter_zevol_table)
    return z0_scatter*zevol_factor


def r_minus_i(magr, redshift, seed=None,
            red_peak_ri=default_red_peak_ri,
            red_peak_ri_zevol_shift_table=default_red_peak_ri_zevol,
            red_scatter_ri=default_red_scatter_ri,
            red_scatter_ri_zevol_table=default_red_scatter_ri_zevol_table,
            ms_peak_ri=default_ms_peak_ri,
            ms_peak_ri_zevol_shift_table=default_ms_peak_ri_zevol,
            ms_scatter_ri=default_ms_scatter_ri,
            ms_scatter_ri_zevol_table=default_ms_scatter_ri_zevol_table, **kwargs):
    """ Generate a Monte Carlo realization of g-r restframe color.

    Parameters
    ----------
    magr : ndarray
        Numpy array of shape (ngals, ) storing the
        r-band absolute restframe magnitude of every galaxy

    redshift : ndarray
        Numpy array of shape (ngals, ) storing the redshift of each galaxy

    Returns
    -------
    ri : ndarray
        Numpy array of shape (ngals, ) storing g-r restframe color for every galaxy

    is_quiescent : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the g-r red sequence

    Examples
    --------
    >>> ngals = int(1e4)
    >>> magr = np.random.uniform(-25, -12, ngals)
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> ri, is_red = r_minus_i(magr, redshift)
    """
    magr = np.atleast_1d(magr)

    ngals = len(magr)
    with NumpyRNGContext(seed):
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_ri(magr, redshift)

    red_sequence_loc = red_sequence_peak_ri(magr[is_quiescent], redshift[is_quiescent])
    red_sequence_scatter = red_sequence_width_ri(magr[is_quiescent],
            red_scatter_ri, redshift[is_quiescent], red_scatter_ri_zevol_table)
    red_sequence = np.random.normal(loc=red_sequence_loc, scale=red_sequence_scatter)

    ms_sequence_loc = main_sequence_peak_ri(magr[~is_quiescent], redshift[~is_quiescent])
    ms_sequence_scatter = main_sequence_width_ri(magr[~is_quiescent],
            ms_scatter_ri, redshift[~is_quiescent], ms_scatter_ri_zevol_table)
    main_sequence = np.random.normal(loc=ms_sequence_loc, scale=ms_sequence_scatter)

    ri = np.zeros(ngals).astype('f4')
    ri[is_quiescent] = red_sequence
    ri[~is_quiescent] = main_sequence
    return ri, is_quiescent


def _sequence_peak(magr, x, y):
    """ Numpy kernel used to fit a 2-degree polynomial to an input
    data table storing the median value <color | Mr>.
    """
    c2, c1, c0 = np.polyfit(x, np.log(y), deg=2)
    result = np.exp(c0 + c1*magr + c2*magr**2)

    #  Now clip endpoints
    xmin, xmax = np.min(x), np.max(x)
    faint_edge_value = np.exp(c0 + c1*xmax + c2*xmax**2)
    bright_edge_value = np.exp(c0 + c1*xmin + c2*xmin**2)
    result = np.where(magr > xmax, faint_edge_value, result)
    result = np.where(magr < xmin, bright_edge_value, result)
    return result


def _sequence_width(magr, x, y):
    """ Numpy kernel used to fit a 2-degree polynomial to an input
    data table storing the level of scatter in Mr at fixed Mr.
    """
    c2, c1, c0 = np.polyfit(x, y, deg=2)
    result = c0 + c1*magr + c2*magr**2

    #  Now clip endpoints
    xmin, xmax = np.min(x), np.max(x)
    faint_edge_value = c0 + c1*xmax + c2*xmax**2
    bright_edge_value = c0 + c1*xmin + c2*xmin**2
    result = np.where(magr > xmax, faint_edge_value, result)
    result = np.where(magr < xmin, bright_edge_value, result)
    return result


def _peak_zevol_factor(redshift, shift_table, z_table=peak_shift_factor_z_table):
    """
    """
    return np.interp(redshift, z_table, shift_table)


def _scatter_zevol_factor(redshift, z_table, scatter_zevol_table):
    return np.interp(redshift, z_table, scatter_zevol_table)

