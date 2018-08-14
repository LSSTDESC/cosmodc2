"""
Module storing the analytical model for SDSS restframe colors
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


#  Control points in r-band luminosity at which the (g-r) sequence peaks are defined
ms_peak_gr_abscissa = [-22.5, -21, -20, -19, -18, -15]
red_peak_gr_abscissa = ms_peak_gr_abscissa

#  Control points defining the locus of the (g-r) main sequence
default_ms_peak_gr = [0.65, 0.65, 0.6, 0.4, 0.4, 0.35]

#  Control points defining the locus of the (g-r) red sequence
default_red_peak_gr = [0.9, 0.85, 0.8, 0.7, 0.7, 0.7]

#  Control points defining the z-dependent blueshift of the locus of the (g-r) red sequence
peak_shift_factor_z_table = [0.1, 0.35, 0.65, 1.0]
default_red_peak_gr_zevol = [0, -0.03, -0.08, -0.25]
default_ms_peak_gr_zevol = [0.0, -0.03, -0.12, -0.35]

#  Control points defining the magr-dependent (g-r) scatter at z = 0
default_ms_scatter_gr = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
default_red_scatter_gr = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04]

#  Control points defining the z-evolution lf (g-r) scatter
default_red_scatter_gr_zevol_table = [1., 1., 1., 1.]
default_ms_scatter_gr_zevol_table = [1., 1., 1.1, 1.2]
scatter_zevol_z_table = peak_shift_factor_z_table


default_mr_z0_fq_gr_pivot = -20.25
default_mr_z0_fq_gr_k = 1.
default_fq_gr_z0_floor = 0.2
default_fq_gr_z0_ceil = 0.95

default_fq_gr_z1_floor = 0.2
default_fq_gr_z1_ceil = 0.25


def sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    """
    """
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))


def fq_gr_z0_vs_magr(magr, mr_z0_fq_gr_pivot=default_mr_z0_fq_gr_pivot,
            mr_z0_fq_gr_k=default_mr_z0_fq_gr_k, fq_gr_z0_floor=default_fq_gr_z0_floor,
            fq_gr_z0_ceil=default_fq_gr_z0_ceil):
    return sigmoid(magr, x0=mr_z0_fq_gr_pivot,
                k=mr_z0_fq_gr_k, ymin=fq_gr_z0_ceil, ymax=fq_gr_z0_floor)


def fq_gr_z1_vs_magr(magr, mr_z1_fq_gr_pivot=default_mr_z0_fq_gr_pivot,
            mr_z1_fq_gr_k=default_mr_z0_fq_gr_k, fq_gr_z1_floor=default_fq_gr_z1_floor,
            fq_gr_z1_ceil=default_fq_gr_z1_ceil):
    return sigmoid(magr, x0=mr_z1_fq_gr_pivot,
                k=mr_z1_fq_gr_k, ymin=fq_gr_z1_ceil, ymax=fq_gr_z1_floor)


def quiescent_fraction_gr(magr, redshift):
    """
    Fraction of galaxies on the g-r red sequence as a function of Mr.
    """
    fq_z0 = fq_gr_z0_vs_magr(magr)
    fq_z1 = fq_gr_z1_vs_magr(magr)
    fq_at_z = sigmoid(redshift, x0=0.5, k=12, ymin=fq_z0, ymax=fq_z1)
    return fq_at_z


def main_sequence_gr_zevol_sigmoid_params(r):
    """
    """
    ymin = sigmoid(r, x0=-20.5, ymin=0.75, ymax=0.375, k=0.7)
    ymax = sigmoid(r, x0=-20.5, ymin=0.385, ymax=0.02, k=0.7)
    return ymin, ymax


def red_sequence_gr_zevol_sigmoid_params(r):
    """
    """
    ymin = sigmoid(r, x0=-20.85, ymin=0.94, ymax=0.695, k=1)
    ymax = sigmoid(r, x0=-20.5, ymin=0.665, ymax=0.385, k=0.8)
    return ymin, ymax


def main_sequence_peak_gr(magr, redshift):
    """
    """
    ymin, ymax = main_sequence_gr_zevol_sigmoid_params(magr)
    return sigmoid(redshift, x0=0.7, k=7, ymin=ymin, ymax=ymax)


def red_sequence_peak_gr(magr, redshift):
    """
    """
    ymin, ymax = red_sequence_gr_zevol_sigmoid_params(magr)
    return sigmoid(redshift, x0=0.7, k=7, ymin=ymin, ymax=ymax)


def red_sequence_width_gr(magr, red_scatter_gr, redshift, red_gr_scatter_zevol_table,
            x=red_peak_gr_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, red_scatter_gr)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, red_gr_scatter_zevol_table)
    return z0_scatter*zevol_factor


def main_sequence_width_gr(magr, ms_scatter_gr, redshift, ms_gr_scatter_zevol_table,
            x=ms_peak_gr_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, ms_scatter_gr)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, ms_gr_scatter_zevol_table)
    return z0_scatter*zevol_factor


def g_minus_r(magr, redshift, seed=None,
            red_peak_gr=default_red_peak_gr,
            red_peak_gr_zevol_shift_table=default_red_peak_gr_zevol,
            red_scatter_gr=default_red_scatter_gr,
            red_scatter_gr_zevol_table=default_red_scatter_gr_zevol_table,
            ms_peak_gr=default_ms_peak_gr,
            ms_peak_gr_zevol_shift_table=default_ms_peak_gr_zevol,
            ms_scatter_gr=default_ms_scatter_gr,
            ms_scatter_gr_zevol_table=default_ms_scatter_gr_zevol_table, **kwargs):
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
    gr : ndarray
        Numpy array of shape (ngals, ) storing g-r restframe color for every galaxy

    is_quiescent : ndarray
        Numpy boolean array of shape (ngals, ) storing whether or not
        the galaxy is on the g-r red sequence

    Examples
    --------
    >>> ngals = int(1e4)
    >>> magr = np.random.uniform(-25, -12, ngals)
    >>> redshift = np.random.uniform(0, 3, ngals)
    >>> gr, is_red = g_minus_r(magr, redshift)
    """
    magr = np.atleast_1d(magr)

    ngals = len(magr)
    with NumpyRNGContext(seed):
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_gr(magr, redshift)

    red_sequence_loc = red_sequence_peak_gr(magr[is_quiescent], redshift[is_quiescent])
    red_sequence_scatter = red_sequence_width_gr(magr[is_quiescent],
            red_scatter_gr, redshift[is_quiescent], red_scatter_gr_zevol_table)
    red_sequence = np.random.normal(loc=red_sequence_loc, scale=red_sequence_scatter)

    ms_sequence_loc = main_sequence_peak_gr(magr[~is_quiescent], redshift[~is_quiescent])
    ms_sequence_scatter = main_sequence_width_gr(magr[~is_quiescent],
            ms_scatter_gr, redshift[~is_quiescent], ms_scatter_gr_zevol_table)
    main_sequence = np.random.normal(loc=ms_sequence_loc, scale=ms_sequence_scatter)

    gr = np.zeros(ngals).astype('f4')
    gr[is_quiescent] = red_sequence
    gr[~is_quiescent] = main_sequence
    return gr, is_quiescent


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

