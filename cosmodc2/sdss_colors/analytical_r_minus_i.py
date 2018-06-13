"""
Module storing the analytical model for SDSS restframe colors
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


#  Control points defining the (g-r) red fraction as a function of r-band luminosity
fq_ri_abscissa = [-25, -22.5, -21, -20, -19.5, -19, -18.5, -18, -15]
default_fq_ri = [0.9, 0.8, 0.65, 0.60, 0.465, 0.35, 0.2, 0.1, 0.1]
default_fq_ri_floor_table = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08]

#  Define how the red fraction evolves with redshift for g-r color
default_blueshift_fq_ri_z_table = [0.25, 0.5, 1.0]
default_fq_ri_blueshift_table = (1., 2.5, 10.)

#  Control points in r-band luminosity at which the (r-i) sequence peaks are defined
ms_peak_ri_abscissa = [-22.5, -21, -20, -19, -18, -15]
red_peak_ri_abscissa = ms_peak_ri_abscissa

#  Control points defining the locus of the (r-i) main sequence
default_ms_peak_ri = [0.4, 0.35, 0.3, 0.24, 0.2, 0.185]

#  Control points defining the locus of the (r-i) red sequence
default_red_peak_ri = [0.41, 0.41, 0.4, 0.385, 0.375, 0.35]

#  Control points defining the z-dependent blueshift of the locus of the (r-i) sequences
peak_shift_factor_z_table = [0.1, 0.25, 0.50, 1.0]
default_red_peak_ri_zevol = [0, -0.04, -0.1, -0.2]
default_ms_peak_ri_zevol = [0.0, -0.05, -0.15, -0.3]

#  Control points defining the magr-dependent (g-r) scatter at z = 0
default_ms_scatter_ri = [0.02, 0.05, 0.05, 0.05, 0.05, 0.05]
default_red_scatter_ri = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02]

#  Control points defining the z-evolution lf (g-r) scatter
default_red_scatter_ri_zevol_table = [1., 1., 1., 1.]
default_ms_scatter_ri_zevol_table = [1., 1., 1., 1.]
scatter_zevol_z_table = peak_shift_factor_z_table


def redshift_dependent_fq_ri_floor(magr, fq_ri_floor_table):
    """
    """
    return np.interp(magr, fq_ri_abscissa, fq_ri_floor_table)


def apply_z_evolution_of_fq(fq_z0p0, redshift, z_table, reduction_factor):
    """
    """
    reduction_factor = np.interp(redshift, z_table, reduction_factor)
    result = fq_z0p0/reduction_factor
    return np.where(result < 0, 0., result)


def quiescent_fraction_ri(magr, redshift, fq_ri, blueshift_factor, fq_ri_floor_table,
            blueshift_fq_ri_z_table=default_blueshift_fq_ri_z_table):
    """
    Fraction of galaxies on the g-r red sequence as a function of Mr.
    """
    fq_z0 = np.interp(magr, fq_ri_abscissa, fq_ri)
    fq_at_z = apply_z_evolution_of_fq(fq_z0, redshift, blueshift_fq_ri_z_table, blueshift_factor)
    fq_ri_floor = redshift_dependent_fq_ri_floor(magr, fq_ri_floor_table)
    return np.where(fq_at_z <= fq_ri_floor, fq_ri_floor, fq_at_z)


def red_sequence_peak_ri(magr, red_peak_ri, redshift, red_peak_ri_zevol_shift_table,
            x=red_peak_ri_abscissa):
    """
    Location of the median value of <g-r | Mr> for quiescent galaxies.
    """
    z0_peak = _sequence_peak(magr, x, red_peak_ri)
    zevol_factor = _peak_zevol_factor(redshift, red_peak_ri_zevol_shift_table)
    return z0_peak + zevol_factor


def red_sequence_width_ri(magr, red_scatter_ri, redshift, red_ri_scatter_zevol_table,
            x=red_peak_ri_abscissa):
    """
    Level of scatter about the median value of <g-r | Mr> for red sequence galaxies.
    """
    z0_scatter = _sequence_width(magr, x, red_scatter_ri)
    zevol_factor = _scatter_zevol_factor(
            redshift, scatter_zevol_z_table, red_ri_scatter_zevol_table)
    return z0_scatter*zevol_factor


def main_sequence_peak_ri(magr, ms_peak_ri, redshift, ms_peak_ri_zevol_shift_table,
            x=ms_peak_ri_abscissa):
    """
    Location of the median value of <g-r | Mr> for star-forming galaxies.
    """
    z0_peak = _sequence_peak(magr, x, ms_peak_ri)
    zevol_factor = _peak_zevol_factor(redshift, ms_peak_ri_zevol_shift_table)
    return z0_peak + zevol_factor


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
            fq_ri=default_fq_ri, fq_ri_blueshift_factor_table=default_fq_ri_blueshift_table,
            red_peak_ri=default_red_peak_ri,
            red_peak_ri_zevol_shift_table=default_red_peak_ri_zevol,
            red_scatter_ri=default_red_scatter_ri,
            red_scatter_ri_zevol_table=default_red_scatter_ri_zevol_table,
            ms_peak_ri=default_ms_peak_ri,
            ms_peak_ri_zevol_shift_table=default_ms_peak_ri_zevol,
            ms_scatter_ri=default_ms_scatter_ri,
            ms_scatter_ri_zevol_table=default_ms_scatter_ri_zevol_table,
            fq_ri_floor_table=default_fq_ri_floor_table,
            blueshift_fq_ri_z_table=default_blueshift_fq_ri_z_table, **kwargs):
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
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_ri(
            magr, redshift, fq_ri, fq_ri_blueshift_factor_table, fq_ri_floor_table,
            blueshift_fq_ri_z_table=blueshift_fq_ri_z_table)

    red_sequence_loc = red_sequence_peak_ri(
        magr[is_quiescent], red_peak_ri, redshift[is_quiescent], red_peak_ri_zevol_shift_table)
    red_sequence_scatter = red_sequence_width_ri(magr[is_quiescent],
            red_scatter_ri, redshift[is_quiescent], red_scatter_ri_zevol_table)
    red_sequence = np.random.normal(loc=red_sequence_loc, scale=red_sequence_scatter)

    ms_sequence_loc = main_sequence_peak_ri(
        magr[~is_quiescent], ms_peak_ri, redshift[~is_quiescent], ms_peak_ri_zevol_shift_table)
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
    """ Function responsible for modulating the redshift-dependence
    of the relation <Mr | M*>(z) by an overall multiplicative factor.
    The multiplicative factor is determined by linearly interpolating from
    the set of input points in redshift.
    """
    return np.interp(redshift, z_table, shift_table)


def _scatter_zevol_factor(redshift, z_table, scatter_zevol_table):
    return np.interp(redshift, z_table, scatter_zevol_table)

