"""
"""
import numpy as np

from ..sawtooth_binning import sawtooth_bin_indices

__all__ = ('test1', )


def test1():
    """ Enforce bins span the sensible range
    """
    npts = int(1e5)
    x = np.random.uniform(-1, 1, npts)
    nbins = 100
    bin_edges = np.linspace(-2, 2, nbins)

    bin_numbers = sawtooth_bin_indices(x, bin_edges)
    assert bin_numbers.shape == (npts, )
    assert np.all(bin_numbers >= 0)
    assert np.all(bin_numbers <= nbins-1)


def test2():
    """ Enforce no bin has fewer than min_counts elements
    """
    npts = 100
    x = np.random.uniform(-1, 1, npts)
    nbins = 100
    bin_edges = np.linspace(-2, 2, nbins)

    bin_numbers = sawtooth_bin_indices(x, bin_edges, min_counts=2)
    uvals, counts = np.unique(bin_numbers, return_counts=True)
    assert np.all(counts >= 2)


def test3():
    """ Enforce bin assignment preferentially assigns membership to bins that are nearby.
    """
    npts = int(1e6)
    x = np.random.uniform(-1, 1, npts)
    nbins = 10
    bin_edges = np.linspace(-1.01, 1.01, nbins)

    bin_numbers = sawtooth_bin_indices(x, bin_edges, min_counts=2, seed=43)

    itest = 4
    test_mask = (x >= bin_edges[itest]) & (x < bin_edges[itest+1])
    assert set(bin_numbers[test_mask]) == set((itest, itest+1))
    assert np.allclose(np.mean(bin_numbers[test_mask]), itest+0.5, rtol=0.1)

    dx_bin = bin_edges[itest+1] - bin_edges[itest]
    test_mask2 = test_mask & (x < bin_edges[itest] + dx_bin/10.)
    assert np.mean(bin_numbers[test_mask2]) < itest + 0.25
    test_mask3 = test_mask & (x > bin_edges[itest+1] - dx_bin/10.)
    assert np.mean(bin_numbers[test_mask3]) > itest + 1 - 0.25
