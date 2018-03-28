"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import conditional_abunmatch


def sequence_width(magr, x, y):
    """
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


def sequence_peak(magr, x, y):
    """
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


def red_sequence_width_gr(magr,
        x=[-22.5, -21, -20, -18, -15],
        y=[0.05, 0.06, 0.065, 0.06, 0.06]):
    return sequence_width(magr, x, y)


def main_sequence_width_gr(magr,
        x=[-22.5, -21, -20, -18, -15],
        y=[0.1, 0.1, 0.1, 0.1, 0.1]):
    return sequence_width(magr, x, y)


def red_sequence_peak_gr(magr,
        x=[-25, -21, -20, -19, -18, -15],
        y=[1.1, 0.95, 0.8, 0.7, 0.7, 0.7]):
    return sequence_peak(magr, x, y)


def main_sequence_peak_gr(magr,
        x=[-25, -21, -20, -19, -18, -15],
        y=[0.65, 0.65, 0.6, 0.4, 0.4, 0.35]):
    return sequence_peak(magr, x, y)


def quiescent_fraction_gr(magr,
        x=[-22.5, -21, -20, -19.5, -19, -18, -15],
        y=[0.85, 0.65, 0.6, 0.50, 0.50, 0.2, 0.1]):
    c2, c1, c0 = np.polyfit(x, np.log(y), deg=2)
    return np.exp(c0 + c1*magr + c2*magr**2)


def g_minus_r(magr, seed=None):
    magr = np.atleast_1d(magr)

    ngals = len(magr)
    with NumpyRNGContext(seed):
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_gr(magr)

    red_sequence = np.random.normal(
        loc=red_sequence_peak_gr(magr[is_quiescent]),
        scale=red_sequence_width_gr(magr[is_quiescent]))
    star_forming_sequence = np.random.normal(
        loc=main_sequence_peak_gr(magr[~is_quiescent]),
        scale=main_sequence_width_gr(magr[~is_quiescent]))

    result = np.zeros(ngals).astype('f4')
    result[is_quiescent] = red_sequence
    result[~is_quiescent] = star_forming_sequence
    return result


def red_sequence_width_ri(magr,
        x=[-22.5, -21, -20, -18, -15],
        y=[0.025, 0.03, 0.03, 0.025, 0.025]):
    return sequence_width(magr, x, y)


def main_sequence_width_ri(magr,
        x=[-22.5, -21, -20, -18, -15],
        y=[0.025, 0.065, 0.065, 0.06, 0.06]):
    return sequence_width(magr, x, y)


def red_sequence_peak_ri(magr,
        x=[-23, -21, -20, -19.5, -19, -18, -15],
        y=[0.435, 0.41, 0.4, 0.385, 0.375, 0.35, 0.31]):
    return sequence_peak(magr, x, y)


def main_sequence_peak_ri(magr,
        x=[-25, -21, -20, -19, -18, -15],
        y=[0.4, 0.35, 0.3, 0.25, 0.215, 0.185]):
    return sequence_peak(magr, x, y)


def quiescent_fraction_ri(magr,
        x=[-22.5, -21, -20, -19.5, -19, -18.5, -18, -15],
        y=[0.8, 0.65, 0.60, 0.465, 0.42, 0.35, 0.15, 0.1]):
    return np.interp(magr, x, y)


def r_minus_i(magr, seed=None):
    magr = np.atleast_1d(magr)

    ngals = len(magr)
    with NumpyRNGContext(seed):
        is_quiescent = np.random.rand(ngals) < quiescent_fraction_ri(magr)

    red_sequence = np.random.normal(
        loc=red_sequence_peak_ri(magr[is_quiescent]),
        scale=red_sequence_width_ri(magr[is_quiescent]))
    star_forming_sequence = np.random.normal(
        loc=main_sequence_peak_ri(magr[~is_quiescent]),
        scale=main_sequence_width_ri(magr[~is_quiescent]))

    result = np.zeros(ngals).astype('f4')
    result[is_quiescent] = red_sequence
    result[~is_quiescent] = star_forming_sequence
    return result


def gr_ri_monte_carlo(magr, percentile,
            local_random_scale=0.1, nonlocal_random_fraction=0.05, nwin=301):
    """
    """
    ngals = len(magr)

    p1 = np.where(np.random.rand(ngals) > 0.05,
        np.random.normal(loc=percentile, scale=local_random_scale), np.random.rand(ngals))
    p2 = np.where(np.random.rand(ngals) > 0.05,
        np.random.normal(loc=percentile, scale=local_random_scale), np.random.rand(ngals))

    gr = conditional_abunmatch(magr, p1, magr, g_minus_r(magr), nwin)
    ri = conditional_abunmatch(magr, p2, magr, r_minus_i(magr), nwin)

    return gr, ri

