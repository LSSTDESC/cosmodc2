"""
"""
from ..stellar_mass_remapping import lift_high_mass_mstar
from .v4_sdss_assign_gri import assign_restframe_sdss_gri


__all__ = ('v4_paint_colors_onto_umachine_snaps', )


def v4_paint_colors_onto_umachine_snaps(
        mpeak, mstar, upid, redshift, sfr_percentile, host_halo_mvir, **kwargs):
    """
    """
    new_mstar = lift_high_mass_mstar(mpeak, mstar, upid, redshift)

    result = assign_restframe_sdss_gri(upid, new_mstar, sfr_percentile,
                host_halo_mvir, redshift, **kwargs)
    new_magr_rest, gr_mock, ri_mock, is_red_gr_mock, is_red_ri_mock = result

    return new_mstar, new_magr_rest, gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock
