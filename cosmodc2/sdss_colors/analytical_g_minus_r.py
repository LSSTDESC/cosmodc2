"""
Module storing the analytical model for SDSS restframe colors
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


#  Control points defining the (g-r) red fraction as a function of r-band luminosity
fq_gr_abscissa = [-22.5, -22., -21, -20, -19.5, -19, -18.5, -18, -15]
default_fq_gr = [0.9, 0.775, 0.6, 0.55, 0.525, 0.50, 0.25, 0.2, 0.1]
default_fq_gr_floor_table = [0.25, 0.25, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15]

#  Define how the red fraction evolves with redshift for g-r color
default_blueshift_fq_gr_z_table = [0.3, 0.55, 0.75, 1.0, 2]
default_fq_gr_blueshift_table = (1., 1.5, 3.5, 5., 10)

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


def redshift_dependent_fq_gr_floor(magr, fq_gr_floor_table):
    """
    """
    return np.interp(magr, fq_gr_abscissa, fq_gr_floor_table)


def apply_z_evolution_of_fq(fq_z0p0, redshift, z_table, reduction_factor):
    """
    """
    reduction_factor = np.interp(redshift, z_table, reduction_factor)
    result = fq_z0p0/reduction_factor
    return np.where(result < 0, 0., result)


def quiescent_fraction_gr(magr, redshift, fq_gr, blueshift_factor, fq_gr_floor_table,
            blueshift_fq_gr_z_table=default_blueshift_fq_gr_z_table):
    """
    Fraction of galaxies on the g-r red sequence as a function of Mr.
    """
    fq_z0 = np.interp(magr, fq_gr_abscissa, fq_gr)
    fq_at_z = apply_z_evolution_of_fq(fq_z0, redshift, blueshift_fq_gr_z_table, blueshift_factor)
    fq_gr_floor = redshift_dependent_fq_gr_floor(magr, fq_gr_floor_table)
    return np.where(fq_at_z <= fq_gr_floor, fq_gr_floor, fq_at_z)


def red_sequence_peak_gr(magr, red_peak_gr, redshift, red_peak_gr_zevol_shift_table,
            x=red_peak_gr_abscissa):
    """
    Location of the median value of <g-r | Mr> for quiescent galaxies.
    """
    z0_peak = _sequence_peak(magr, x, red_peak_gr)
    zevol_factor = _peak_zevol_factor(redshift, red_peak_gr_zevol_shift_table)
    return z0_peak + zevol_factor


def red_sequence_width_gr(magr, red_scatter_gr, redshift, red_gr_scatter_zevol_table,
            x=red_peak_gr_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, red_scatter_gr)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, red_gr_scatter_zevol_table)
    return z0_scatter*zevol_factor


def main_sequence_peak_gr(magr, ms_peak_gr, redshift, ms_peak_gr_zevol_shift_table,
            x=ms_peak_gr_abscissa):
    """
    Location of the median value of <g-r | Mr> for star-forming galaxies.
    """
    z0_peak = _sequence_peak(magr, x, ms_peak_gr)
    zevol_factor = _peak_zevol_factor(redshift, ms_peak_gr_zevol_shift_table)
    return z0_peak + zevol_factor


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
            fq_gr=default_fq_gr, fq_gr_blueshift_factor_table=default_fq_gr_blueshift_table,
            red_peak_gr=default_red_peak_gr,
            red_peak_gr_zevol_shift_table=default_red_peak_gr_zevol,
            red_scatter_gr=default_red_scatter_gr,
            red_scatter_gr_zevol_table=default_red_scatter_gr_zevol_table,
            ms_peak_gr=default_ms_peak_gr,
            ms_peak_gr_zevol_shift_table=default_ms_peak_gr_zevol,
            ms_scatter_gr=default_ms_scatter_gr,
            ms_scatter_gr_zevol_table=default_ms_scatter_gr_zevol_table,
            fq_gr_floor_table=default_fq_gr_floor_table,
            blueshift_fq_gr_z_table=default_blueshift_fq_gr_z_table, **kwargs):
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
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_gr(
            magr, redshift, fq_gr, fq_gr_blueshift_factor_table, fq_gr_floor_table,
            blueshift_fq_gr_z_table=blueshift_fq_gr_z_table)

    red_sequence_loc = red_sequence_peak_gr(
        magr[is_quiescent], red_peak_gr, redshift[is_quiescent], red_peak_gr_zevol_shift_table)
    red_sequence_scatter = red_sequence_width_gr(magr[is_quiescent],
            red_scatter_gr, redshift[is_quiescent], red_scatter_gr_zevol_table)
    red_sequence = np.random.normal(loc=red_sequence_loc, scale=red_sequence_scatter)

    ms_sequence_loc = main_sequence_peak_gr(
        magr[~is_quiescent], ms_peak_gr, redshift[~is_quiescent], ms_peak_gr_zevol_shift_table)
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
    """ Function responsible for modulating the redshift-dependence
    of the relation <Mr | M*>(z) by an overall multiplicative factor.
    The multiplicative factor is determined by linearly interpolating from
    the set of input points in redshift.
    """
    return np.interp(redshift, z_table, shift_table)


def _scatter_zevol_factor(redshift, z_table, scatter_zevol_table):
    return np.interp(redshift, z_table, scatter_zevol_table)

