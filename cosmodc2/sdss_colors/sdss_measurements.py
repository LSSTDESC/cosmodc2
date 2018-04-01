"""
"""
import os
import numpy as np


rp = np.array((0.17, 0.27, 0.42, 0.67, 1.1, 1.7,
    2.7, 4.2, 6.7, 10.6, 16.9, 26.8))
lumthresh = np.array((-18, -18.5, -19, -19.5, -20, -20.5, -21, -21.5, -22.0))


__all__ = ('zehavi11_cumulative_nd', 'zehavi11_clustering')


def zehavi11_cumulative_nd(magr_h=0.7):
    """ Cumulative number densities in units of (h/Mpc)**3
    taken from Table 2 of Zehavi+11, https://arxiv.org/abs/1005.2413.

    Their quoted values for luminosity and distance are assuming h=1.
    The scaling of Mr with little h is given by Mr + 5 log h.
    The sign convention for this conversion is made explicit below.

    To convert Magr to other little h conventions to permit direct comparison
    to zehavi11_cumulative_nd, proceed as follows.
    Let's use the notation ``Mr[h=1]`` to denote the numerical value
    of the r-band absolute magnitude of a galaxy when one assumes H0=100km/s/Mpc;
    similarly, ``Mr[h=0.7]`` denotes the numerical value of r-band absolute magnitude
    of the very same galaxy when one assumes H0=70/s/Mpc.
    To convert between these two values:

    Mr[h=0.7] = Mr[h=1] + 5*log_10(0.7)

    So when converting to values quoted for h=0.7,
    the value of Mr gets smaller, more negative, corresponding to a brighter value,
    relative to the h=1 quoted value.

    Distance scales as R/h, but the simulation coordinates also use h=1
    so it's easier not to convert.
    """
    cumnd = np.array((3.030, 2.311, 1.676, 1.12, 0.656, 0.318, 0.116, 0.028, 0.005))/100.
    return lumthresh, cumnd


def zehavi11_clustering(magr_thresh, subsample='all',
        data_dirname="/Users/aphearin/Dropbox/sham_colors/DATA/Zehavi_wp_lumthresh_colorsplit"):
    """"
    """
    thresh_18p0 = np.array((
        294.3, 221.5, 161.4, 114.7, 75.5, 48.6,
        32.4, 19.7, 10.8, 6.35, 3.62, 2.14))
    thresh_18p5 = np.array((313.3, 230.2, 165.4, 118.3, 79.7,
        53.8, 37.4, 25.9, 17.4, 10.6, 5.31, 3.56))
    thresh_19p0 = np.array((322.5, 231.1, 162.4, 114.6, 75.5, 50.6,
        35.0, 24.2, 15.3, 9.2, 4.11, 1.81))
    thresh_19p5 = np.array((307.0, 228.5, 159.3, 110.4, 72.9, 49.8, 34.6,
        24.6, 16.7, 10.7, 5.73, 2.82))
    thresh_20p0 = np.array((366.1, 264.3, 184.0, 128.6, 84.7, 59.4,
        42.9, 30.9, 21.9, 14.6, 8.24, 4.88))
    thresh_20p5 = np.array((455.7, 296.9, 197.0, 134.1, 89.4, 61.1, 44.0,
        31.2, 21.3, 13.7, 7.65, 4.09))
    thresh_21p0 = np.array((586.2, 402.9, 258.7, 163.2, 105.5, 68.9, 50.2, 35.5,
        24.5, 15.3, 8.54, 4.11))
    thresh_21p5 = np.array((1028.0, 731.7, 392.6, 228.6, 144.6, 94.3, 70.5, 48.6,
        33.1, 20.9, 11.6, 6.04))

    if subsample == 'all':
        if magr_thresh == -18:
            return thresh_18p0
        elif magr_thresh == -18.5:
            return thresh_18p5
        elif magr_thresh == -19.0:
            return thresh_19p0
        elif magr_thresh == -19.5:
            return thresh_19p5
        elif magr_thresh == -20.0:
            return thresh_20p0
        elif magr_thresh == -20.5:
            return thresh_20p5
        elif magr_thresh == -21.0:
            return thresh_21p0
        elif magr_thresh == -21.5:
            return thresh_21p5
        else:
            return ValueError("unsupported value of magr_thresh = {0}".format(magr_thresh))
    else:
        msg = "``subsample`` argument = {0}\n The available options are ``all``, ``red``, ``blue``"
        assert subsample in ('blue', 'red'), msg.format(subsample)
        magr_thresh_string = "{0:.1f}".format(abs(magr_thresh))
        basename = "wp_" + magr_thresh_string + "_m" + subsample + ".dat"
        fname = os.path.join(data_dirname, basename)
        X = np.loadtxt(fname)
        return X[:-1, 1], X[:-1, 2]





