"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


def sigmoid_downsampling_mask(mag, cut, k=5, seed=None):
    """ Function calculates a boolean array to mask out faint galaxies.

    Downsampling is done gradually as magnitude exceeds the input cut.

    For plots of the behavior, see
    https://gist.github.com/aphearin/d6d12fd0c759a931748c4435ebf4f84c

    Parameters
    ----------
    mag : ndarray
        ndarray of shape (ngals, ) storing apparent magnitude

    cut : float
        Parameter specifying the soft cut magnitude.

        Values of mag larger than the cut will be masked,
        with the soft transition implemented by a sigmoid function.

    k : int, optional
        Slope of the sigmoid function controlling the
        gradual selection probability. Default is 5.

    seed : int, optional
        Default is None, for stochastic results

    Returns
    -------
    mask : ndarray
        ndarray of shape (ngals, ) with boolean dtype
    """
    ntot = len(mag)
    with NumpyRNGContext(seed):
        uran = np.random.rand(ntot)
    p = sigmoid(mag, x0=cut, k=k)
    mask = p < uran
    return mask


def sigmoid(x, x0=0, k=1, ymin=0, ymax=1):
    height_diff = ymax-ymin
    return ymin + height_diff/(1 + np.exp(-k*(x-x0)))
