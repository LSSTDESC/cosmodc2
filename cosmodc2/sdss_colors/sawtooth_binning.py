""" Functions used to set up overlapping bin boundaries
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


__all__ = ('sawtooth_bin_indices', )

default_seed = 43


def sawtooth_bin_indices(x, bin_edges, min_counts=2, seed=default_seed):
    """ Function assigns each element of the input array `x` to a particular bin number.

    The bin boundaries have hard edges, but bin-assignment is probabilistic, such that
    when a point in `x` is halfway between two edges, is equally likely to be assigned
    to the bin to its left or right.

    The `sawtooth_bin_indices` function optionally enforces that elements of very sparsely
    populated bins are remapped to the nearest bin with more than `min_counts` elements.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, ) storing the values to be binned

    bin_edges : ndarray
        Numpy array of shape (nbins, ) defining the binning scheme.
        The values of `bin_edges` must strictly encompass the range of values spanned by `x`.

    min_counts : int, optional
        Minimum required number of elements in a bin. For those bins not satisfying this requirement,
        all their elements will be reassigned to the nearest sufficiently populated bin.
        Default is two.

    seed : int, optional
        Random number seed. Default is default_seed, set at the top of
        the module where the function is defined.

    Returns
    -------
    bin_indices : ndarray
        Numpy integer array of shape (npts, ) storing the bin number to which elements of `x`
        are assigned. All values of `bin_indices` will be between 0 and nbins-1, inclusive.

    Examples
    --------
    >>> x = np.random.uniform(0, 1, 1000)
    >>> bin_edges = np.linspace(-1, 2, 25)
    >>> bin_indices = sawtooth_bin_indices(x, bin_edges)
    """
    assert bin_edges[0] < x.min(), "smallest bin must be less than smallest element in x"
    assert bin_edges[-1] > x.max(), "largest bin must be less than largest element in x"

    npts_x = len(x)
    num_bin_edges = len(bin_edges)
    bin_indices = np.zeros_like(x).astype(int)-999

    with NumpyRNGContext(seed):
        uran = np.random.rand(npts_x)

    for i, low, high in zip(np.arange(num_bin_edges).astype(int), bin_edges[:-1], bin_edges[1:]):
        bin_mask = (x >= low) & (x < high)

        npts_bin = np.count_nonzero(bin_mask)
        if npts_bin > 0:
            x_in_bin = x[bin_mask]
            dx_bin = high - low
            x_in_bin_rescaled = (x_in_bin - low)/float(dx_bin)

            high_bin_selection = (x_in_bin_rescaled > uran[bin_mask])
            bin_assignment = np.zeros(npts_bin).astype(int) + i
            bin_assignment[high_bin_selection] = i + 1
            bin_indices[bin_mask] = bin_assignment

    bin_indices[bin_indices == -999] = 0

    return enforce_bin_counts(bin_indices, min_counts)


def enforce_bin_counts(bin_indices, min_counts):
    """ Function enforces that each entry of `bin_indices` appears at least `min_counts` times.
    For entries not satisfying this requirement, the nearest index of a sufficiently populated bin
    will be used as a replacement.

    Parameters
    ----------
    bin_indices : ndarray
        Numpy integer array storing bin numbers

    min_counts : int
        Minimum acceptable number of elements per bin

    Returns
    -------
    output_bin_inidices : ndarray
        Numpy integer array storing bin numbers after enforcing the population requirement.

    Examples
    --------
    >>> bin_indices = np.random.randint(0, 1000, 1000)
    >>> min_counts = 3
    >>> output_bin_indices = enforce_bin_counts(bin_indices, min_counts)
    """
    output_bin_indices = np.copy(bin_indices)
    unique_bin_numbers, counts = np.unique(bin_indices, return_counts=True)
    for i, bin_number, count in zip(np.arange(len(counts)), unique_bin_numbers, counts):
        new_bin_number = _find_nearest_populated_bin_number(
            counts, unique_bin_numbers, i, min_counts)
        if new_bin_number != bin_number:
            output_bin_indices[bin_indices==bin_number] = new_bin_number
    return output_bin_indices


def _find_nearest_populated_bin_number(counts, bin_numbers, bin_index, min_counts):
    """ Helper function used by the `enforce_bin_counts` function.
    """
    bin_numbers = np.atleast_1d(bin_numbers)
    bin_indices = np.arange(len(bin_numbers))
    counts = np.atleast_1d(counts)
    msg = "Must have at least one bin with greater than {0} elements"
    assert np.any(counts >= min_counts), msg.format(min_counts)

    counts_mask = counts >= min_counts
    available_bin_numbers = bin_numbers[counts_mask]
    available_indices = bin_indices[counts_mask]

    return available_bin_numbers[np.argmin(np.abs(available_indices - bin_index))]


