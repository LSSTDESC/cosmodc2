"""
"""
import numpy as np
from scipy.stats import powerlaw
from ..extend_subhalo_mpeak_range import model_extended_mpeak


def test1():
    mpeak = 10**(5*(1-powerlaw.rvs(2, size=40000)) + 10.)
    desired_logm_completeness = 9.5
    corrected_mpeak, mpeak_extension = model_extended_mpeak(mpeak, desired_logm_completeness)
    assert np.all(mpeak_extension < corrected_mpeak.min())
    assert np.all(mpeak_extension >= 10**desired_logm_completeness)
    assert np.any(mpeak_extension < 10**(desired_logm_completeness+0.02))
