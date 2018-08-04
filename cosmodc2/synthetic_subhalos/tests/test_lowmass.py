"""
"""
import numpy as np
from scipy.stats import powerlaw

from ..synthetic_lowmass_subhalos import synthetic_logmpeak


def test1():

    _norig = int(2e5)
    index = 2.
    lowm_orig, highm_orig = 10., 15.
    orig_width = highm_orig - lowm_orig
    _logmpeak_orig = (orig_width*(1.-powerlaw.rvs(index, size=_norig)) + lowm_orig)
    prob_reject = np.interp(_logmpeak_orig, [10, 10.75], [0.95, 0.])
    mask_reject = np.random.rand(_norig) < prob_reject

    logmpeak_orig = _logmpeak_orig[~mask_reject]
    mpeak_orig = 10.**logmpeak_orig

    desired_logm_completeness = 9.
    fake_logm = synthetic_logmpeak(mpeak_orig, desired_logm_completeness)
    assert np.any(fake_logm < logmpeak_orig.min())
    assert np.all(fake_logm >= desired_logm_completeness)
