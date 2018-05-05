"""
"""
import pytest
import numpy as np
from ..analytical_gr_ri import gr_ri_monte_carlo


@pytest.mark.xfail
def test1():
    """
    """

    ngals = int(1e4)
    magr = np.random.uniform(-24, -12, ngals)
    sfr_percentile = np.random.rand(ngals)
    redshift = np.random.uniform(0, 3, ngals)
    result = gr_ri_monte_carlo(magr, sfr_percentile, redshift)
