"""
"""
import numpy as np
from ..monte_carlo_nfw import nfw_profile_realization


def test1():
    """Enforce boundary conditions on r
    """
    npts = int(1e3)
    conc = np.zeros(npts) + 5.
    r = nfw_profile_realization(conc)
    assert np.all(r > 0)
    assert np.all(r < 1)


def test2():
    """Enforce high-concentration halos actually have highly concentrated distributions
    """
    npts = int(1e3)
    conc = np.zeros(npts) + 5.
    r = nfw_profile_realization(conc)
    r2 = nfw_profile_realization(conc*10)
    assert r.mean() > r2.mean()


def test3():
    """Enforce controllable stochasticity
    """
    npts = int(1e3)
    conc = np.zeros(npts) + 5.
    r = nfw_profile_realization(conc, seed=43)
    r2 = nfw_profile_realization(conc, seed=43)
    r3 = nfw_profile_realization(conc, seed=44)
    assert np.allclose(r, r2)
    assert not np.allclose(r, r3)

