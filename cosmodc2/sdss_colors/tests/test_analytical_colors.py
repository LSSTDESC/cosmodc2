"""
"""
import pytest
import numpy as np
from ..analytical_colors import gr_ri_monte_carlo


@pytest.mark.xfail
def test1():
    """
    """

    ngals = int(1e4)
    magr = np.random.uniform(-24, -12, ngals)
    sfr_percentile = np.random.rand(ngals)
    redshift = np.random.uniform(0, 3, ngals)
    result = gr_ri_monte_carlo(magr, sfr_percentile, redshift)


    # satmask = np.random.rand(ngals) < 0.3
    # num_sats = np.count_nonzero(satmask)
    # upid = np.zeros(ngals, dtype=int) - 1
    # upid[satmask] = np.random.randint(100, 500, num_sats)
    # mstar = 10**np.random.uniform(8, 12, ngals)
    # mhalo = 10**np.random.uniform(10, 15, ngals)
    # z = np.random.uniform(0, 3, ngals)
    # result = assign_restframe_sdss_gri(upid, mstar, sfr_percentile, mhalo, z)
    # magr, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = result


