"""
"""
import pytest
import numpy as np
from ..v4_sdss_assign_gri import assign_restframe_sdss_gri


@pytest.mark.xfail
def test1():
    """
    """

    ngals = int(1e4)
    satmask = np.random.rand(ngals) < 0.3
    num_sats = np.count_nonzero(satmask)
    upid = np.zeros(ngals, dtype=int) - 1
    upid[satmask] = np.random.randint(100, 500, num_sats)
    mstar = 10**np.random.uniform(8, 12, ngals)
    sfr_percentile = np.random.rand(ngals)
    mhalo = 10**np.random.uniform(10, 15, ngals)
    z = np.random.uniform(0, 3, ngals)
    result = assign_restframe_sdss_gri(upid, mstar, sfr_percentile, mhalo, z)
    magr, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = result
