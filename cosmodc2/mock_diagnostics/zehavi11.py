"""
"""
import numpy as np


__all__ = ('cumulative_nd', )


def cumulative_nd(magr, volume, assumed_littleh_for_magr, lumthresh_h1p0):
    """
    """
    magr_h1p0 = magr - 5.*np.log10(assumed_littleh_for_magr)
    counts = np.fromiter((np.count_nonzero(magr_h1p0 < lum)
        for lum in lumthresh_h1p0), dtype=int)
    return counts/float(volume)
