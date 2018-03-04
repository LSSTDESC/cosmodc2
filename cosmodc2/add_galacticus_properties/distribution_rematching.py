"""
"""
import numpy as np
from halotools.utils import distribution_matching_indices


def resample_x_to_match_y(x, y, bins):
    """ Return the indices that resample `x` (with replacement) so that the
    resampled distribution matches the histogram of `y`. The returned indexing array
    will be sorted so that the i^th element of x[idx] is as close as possible to the
    i^th value of x, subject to the the constraint that x[idx] matches y.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (nx, )

    y : ndarray
        Numpy array of shape (ny, )

    bins : ndarray
        Numpy array of shape (nbins, )

    Returns
    -------
    indices : ndarray
        Numpy array of shape (nx, )
    """
    nselect = len(x)
    idx = distribution_matching_indices(x, y, nselect, bins)
    xnew = x[idx]
    idx_sorted_xnew = np.argsort(xnew)
    idx_sorted_x = np.argsort(x)
    indices = np.empty_like(x).astype(int)
    indices[idx_sorted_x] = idx[idx_sorted_xnew]
    return indices
