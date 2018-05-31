"""
"""
import numpy as np
from ..stellar_mass_remapping import remap_stellar_mass_in_snapshot
from .v4_sdss_assign_gri import assign_restframe_sdss_gri


__all__ = ('v4_paint_colors_onto_umachine_snaps', )


def v4_paint_colors_onto_umachine_snaps(
        mpeak, mstar, upid, redshift, sfr_percentile, host_halo_mvir, **kwargs):
    """
    """
    msg = "Input snapshot_redshift must be a single float.\nReceived an array of shape {0}"
    _x = np.atleast_1d(redshift)
    assert len(_x) == 1, msg.format(_x.shape)
    new_mstar = remap_stellar_mass_in_snapshot(redshift, mpeak, mstar)

    result = assign_restframe_sdss_gri(upid, new_mstar, sfr_percentile,
                host_halo_mvir, redshift, **kwargs)
    new_magr_rest, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = result

    return new_mstar, new_magr_rest, gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock
