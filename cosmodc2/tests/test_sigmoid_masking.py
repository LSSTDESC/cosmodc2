"""
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext
from ..sigmoid_masking import sigmoid_downsampling_mask


fixed_seed = 43
npts = int(1e4)
with NumpyRNGContext(fixed_seed):
    mag = np.random.uniform(20, 31, npts)


def test1():
    """ Enforce that the sigmoid mask does not throw away bright galaxies
    """
    cut = 28
    magmask = mag < cut-3
    sigmoid_mask = sigmoid_downsampling_mask(mag, cut)
    nbright = np.count_nonzero(magmask)
    nbright_masked = np.count_nonzero(sigmoid_mask & magmask)
    assert nbright == nbright_masked


def test2():
    """ Enforce that the magnitude cut is soft.
    At the magnitude cut, there should be some masked galaxies, but not all.
    """
    cut = 28
    magmask = (mag > cut - 0.1) & (mag < cut + 0.1)
    sigmoid_mask = sigmoid_downsampling_mask(mag, cut)
    ngals_at_cut = np.count_nonzero(magmask)
    ngals_at_cut_masked = np.count_nonzero(mag[magmask & sigmoid_mask])
    assert 0.1*ngals_at_cut < ngals_at_cut_masked < 0.9*ngals_at_cut


def test3():
    """ Enforce that the mask throws out 100% of galaxies
    that are much fainter than the cut.
    """
    cut = 27
    sigmoid_mask = sigmoid_downsampling_mask(mag, cut)
    magmask = mag > cut + 2.5
    nfaint = np.count_nonzero(mag[magmask])
    nfaint_masked = np.count_nonzero(mag[magmask & sigmoid_mask])
    assert nfaint > 0
    assert nfaint_masked == 0


def test4():
    """ Enforce that the masked sample is dimmer and smaller than the original
    by roughly the expected amounts for the input cut.
    """
    cut = 28
    sigmoid_mask = sigmoid_downsampling_mask(mag, cut)
    assert 0.7 < np.mean(sigmoid_mask) < 0.8
    assert np.mean(mag[sigmoid_mask]) < np.mean(mag) - 1
