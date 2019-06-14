"""
"""
import numpy as np
from ..axis_ratio_model import monte_carlo_halo_shapes


def _enforce_constraints(b_to_a, c_to_a, e, p):
    assert np.all(b_to_a > 0), "All elements of b_to_a must be strictly positive"
    assert np.all(c_to_a > 0), "All elements of c_to_a must be strictly positive"
    assert np.all(b_to_a <= 1), "No element of b_to_a can exceed unity"
    assert np.all(c_to_a <= 1), "No element of c_to_a can exceed unity"
    assert np.all(b_to_a >= c_to_a), "No element in c_to_a can exceed the corresponding b_to_a"
    assert np.all(e <= 0.5), "No element of ellipticity can exceed 0.5"
    assert np.all(p <= 0.5), "No element of prolaticity can exceed 0.5"
    assert np.all(e >= 0.), "ellipticity must be non-negative"
    assert np.all(p >= -0.25), "prolaticity cannot exceed -0.25"


def test1():
    """Enforce monte_carlo_halo_shapes doesn't crash when given crazy halo masses
    """
    npts = int(1e5)
    logmhalo = np.linspace(-10, 20, npts)
    b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(logmhalo)
    _enforce_constraints(b_to_a, c_to_a, e, p)


def test2():
    """Enforce expected scaling with halo mass
    """
    npts = int(1e4)
    b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(np.zeros(npts) + 11)
    b_to_a2, c_to_a2, e2, p2 = monte_carlo_halo_shapes(np.zeros(npts) + 15)
    assert e.mean() < e2.mean(), "Higher-mass halos should be more elliptical"
    assert p.mean() < p2.mean(), "Higher-mass halos should be more prolate"
    assert b_to_a.mean() > b_to_a2.mean(), "Higher-mass halos should have more elongated axes"
    assert c_to_a.mean() > c_to_a2.mean(), "Higher-mass halos should have more elongated axes"


def test3():
    """Enforce reasonable correlation coefficient between
    axis ratios and ellipticity and prolaticity
    """
    npts = int(1e4)
    b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(np.zeros(npts) + 11)

    r = np.corrcoef(b_to_a, c_to_a)[0, 1]
    assert r > 0.5, "b_to_a and c_to_a should be highly correlated"

    r = np.corrcoef(e, p)[0, 1]
    assert r > 0.5, "ellipticity and prolaticity should be highly correlated"
