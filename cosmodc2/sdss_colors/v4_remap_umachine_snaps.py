"""
"""
from ..stellar_mass_remapping import lift_high_mass_mstar
from .analytical_magr import magr_monte_carlo
from .analytical_colors import gr_ri_monte_carlo


__all__ = ('v4_paint_colors_onto_umachine_snaps', )


def v4_paint_colors_onto_umachine_snaps(mpeak, mstar, upid, redshift, sfr_percentile):
    """
    """
    new_mstar = lift_high_mass_mstar(mpeak, mstar, upid,  redshift)
    new_mstar = new_mstar

    new_magr_rest = magr_monte_carlo(new_mstar, upid, redshift)

    gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock = gr_ri_monte_carlo(
        new_magr_rest, sfr_percentile, redshift, local_random_scale=0.1)

    return new_mstar, new_magr_rest, gr_mock, ri_mock, is_red_ri_mock, is_red_gr_mock
