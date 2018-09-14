"""
"""
import numpy as np
from ..reposition_satellites import core_satellite_matching_indices


def test1():
    """ Test a specific example where each satellite has a match,
    the core hostIDs span the range of satellite host IDs,
    and all multiplicities are unity.
    """
    satellite_hostid = [4, 6]
    core_hostid = [2, 3, 4, 5, 6, 7]

    result = core_satellite_matching_indices(satellite_hostid, core_hostid)
    correct_result = (2, 4)
    assert np.all(result == correct_result)


def test2():
    """ Test a specific example where each satellite has a match,
    the core hostIDs span the range of satellite host IDs,
    and all nontrivial multiplicities match.
    """
    satellite_hostid = [4, 6, 6, 6]
    core_hostid = [2, 3, 4, 5, 6, 6, 6, 7]

    result = core_satellite_matching_indices(satellite_hostid, core_hostid)
    correct_result = (2, 4, 5, 6)
    assert np.all(result == correct_result)


def test3():
    """ Test a specific example where the final three satellites have no match
    because the core host IDs extend beyond the range of the satellite host IDs.
    """
    satellite_hostid = [4, 6, 6, 6, 8, 100]
    core_hostid = [2, 3, 4, 5, 6, 6, 7, 7]

    result = core_satellite_matching_indices(satellite_hostid, core_hostid)
    correct_result = (2, 4, 5, -1, -1, -1)
    assert np.all(result == correct_result)


def test4():
    """ Test a specific example where the first satellites has no match
    because the first satellite host IDs is smaller than the first core host ID.
    """
    satellite_hostid = [1, 6, 6, 6]
    core_hostid = [2, 3, 4, 5, 6, 6, 6, 7]

    result = core_satellite_matching_indices(satellite_hostid, core_hostid)
    correct_result = (-1, 4, 5, 6)
    assert np.all(result == correct_result)


def test5():
    """ Test a specific example where the first and last satellites have no match.
    """
    satellite_hostid = [1, 6, 6, 8]
    core_hostid = [2, 6, 6, 7, 100]

    result = core_satellite_matching_indices(satellite_hostid, core_hostid)
    correct_result = (-1, 1, 2, -1)
    assert np.all(result == correct_result)


def test_brute_force():
    """ Run a few hundred tests of randomly selected host IDs for cores and satellites.
    Enforce that the returned indices always give corresponding host IDs.
    """
    ntests = 200

    for itest in range(ntests):
        nsats = np.random.randint(1000, 2000)
        ncores = np.random.randint(nsats-100, nsats+100)
        nhosts = np.random.randint(min(nsats, ncores)-10, max(nsats, ncores)+10)

        satellite_hostid = np.sort(np.random.randint(0, nhosts, nsats))
        core_hostid = np.sort(np.random.randint(0, nhosts, ncores))

        result = core_satellite_matching_indices(satellite_hostid, core_hostid)
        assert result.size == nsats
        assert np.all(result < ncores)

        matched_mask = result != -1
        assert np.all(core_hostid[result[matched_mask]] == satellite_hostid[matched_mask])
