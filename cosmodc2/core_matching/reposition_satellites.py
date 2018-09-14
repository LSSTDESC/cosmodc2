""" Module stores functions used to reposition cosmoDC2 satellites on top of cores.
"""
import numpy as np


def core_satellite_matching_indices(sorted_sat_hostid, sorted_core_hostid):
    """ Calculate an indexing array providing a correspondence between cores and satellites.
    Each of the input arrays must be sorted in ascending order.
    Within groups of a common host halo, the cores and satellites should be sorted
    such that the first satellite in the halo will be assigned to the first core in the halo,
    the second to the second, etc. Satellites with no matching core will be assigned -1.

    Parameters
    ----------
    sorted_sat_hostid : ndarray
        Integer array of shape (nsats, ) storing the unique ID
        of the Outer Rim parent halo of each DC2 satellite galaxy

    sorted_core_hostid : ndarray
        Integer array of shape (ncores, ) storing the unique ID
        of the Outer Rim parent halo of each core satellite

    Returns
    -------
    matching_indices : ndarray
        Integer array of shape (nsats, ) storing the index
        of the matching core satellite. Equals -1 for satellites with no matching core.

    Examples
    --------
    >>> sorted_sat_hostid = [3, 4, 9, 10]
    >>> sorted_core_hostid = [2, 3, 8, 8, 9, 10, 11]
    >>> matching_indices = core_satellite_matching_indices(sorted_sat_hostid, sorted_core_hostid)
    """
    sorted_sat_hostid = np.asarray(sorted_sat_hostid)
    sorted_core_hostid = np.asarray(sorted_core_hostid)
    nsats = sorted_sat_hostid.size
    ncores = sorted_core_hostid.size

    matching_indices = np.zeros(nsats).astype(int) - 1

    last_icore = 0
    for isat in range(nsats):
        satid = sorted_sat_hostid[isat]

        for icore in range(last_icore, ncores):
            coreid = sorted_core_hostid[icore]
            if satid > coreid:
                continue
            elif satid == coreid:
                matching_indices[isat] = icore
                last_icore = icore + 1
                break
            else:
                break

    return matching_indices
