import numpy as np
from astropy.table import Table
from halotools.empirical_models import NFWPhaseSpace, Moster13SmHm
from scipy.stats import powerlaw
from ..sdss_colors.sigmoid_magr_model import magr_monte_carlo
from ..sdss_colors.sigmoid_g_minus_r import red_sequence_peak_gr
from ..sdss_colors.sigmoid_r_minus_i import red_sequence_peak_ri

__all__ = ('calculate_synthetic_richness', 'model_synthetic_cluster_satellites')


def calculate_synthetic_richness(halo_richness, logmhalo, logmhalo_source,
        cluster_satboost_logm_table, cluster_satboost_table, logm_outer_rim_effect=14.75):
    """
    Parameters
    ----------
    halo_richness : ndarray
        Numpy array of shape (nhalos, ) storing the richness of each halo

    logmhalo : ndarray
        Numpy array of shape (nhalos, ) storing the log mass of each halo

    Returns
    -------
    synthetic_richness : ndarray
        Numpy integer array of shape (nhalos, ) storing the synthetic richness
    """
    boost_factor = np.interp(logmhalo, cluster_satboost_logm_table, cluster_satboost_table)
    dlogm = logmhalo - logmhalo_source
    outer_rim_boost_factor = 10.**dlogm
    low, high = logm_outer_rim_effect - 0.25, logm_outer_rim_effect + 0.25
    logm_outer_rim_effect = np.random.uniform(low, high, len(boost_factor))
    highmass_mask = logmhalo > logm_outer_rim_effect
    boost_factor = np.where(highmass_mask, outer_rim_boost_factor*boost_factor, boost_factor)
    return np.array(halo_richness*boost_factor, dtype=int)


def model_synthetic_cluster_satellites(mock, Lbox=256.,
        cluster_satboost_logm_table=[13.5, 13.75, 14],
        cluster_satboost_table=[0., 0.15, 0.2], **kwargs):
    """
    """
    #  Calculate the mass and richness of every target halo
    host_halo_id, idx, counts = np.unique(
        mock['target_halo_id'], return_counts=True, return_index=True)
    host_mass = mock['target_halo_mass'][idx]
    host_redshift = mock['target_halo_redshift'][idx]
    host_x = mock['target_halo_x'][idx]
    host_y = mock['target_halo_y'][idx]
    host_z = mock['target_halo_z'][idx]
    host_vx = mock['target_halo_vx'][idx]
    host_vy = mock['target_halo_vy'][idx]
    host_vz = mock['target_halo_vz'][idx]
    source_halo_mvir = mock['source_halo_mvir'][idx]
    target_halo_id = mock['target_halo_id'][idx]
    target_halo_fof_halo_id = mock['target_halo_fof_halo_id'][idx]
    target_halo_lightcone_replication = mock['lightcone_replication'][idx]
    target_halo_lightcone_rotation = mock['lightcone_rotation'][idx]

    host_richness = counts

    #  For each target halo, calculate the number of synthetic satellites we need to add
    synthetic_richness = calculate_synthetic_richness(
        host_richness, np.log10(host_mass), np.log10(source_halo_mvir),
        cluster_satboost_logm_table, cluster_satboost_table)

    if np.sum(synthetic_richness) <= 1:
        return Table()
    else:
        sats = Table()

        #  For every synthetic galaxy, calculate the mass, redshift, position, and velocity of the host halo
        sats['target_halo_mass'] = np.repeat(host_mass, synthetic_richness)
        sats['host_halo_mvir'] = sats['target_halo_mass']
        sats['target_halo_redshift'] = np.repeat(host_redshift, synthetic_richness)
        sats['target_halo_x'] = np.repeat(host_x, synthetic_richness)
        sats['target_halo_y'] = np.repeat(host_y, synthetic_richness)
        sats['target_halo_z'] = np.repeat(host_z, synthetic_richness)
        sats['target_halo_vx'] = np.repeat(host_vx, synthetic_richness)
        sats['target_halo_vy'] = np.repeat(host_vy, synthetic_richness)
        sats['target_halo_vz'] = np.repeat(host_vz, synthetic_richness)
        sats['target_halo_id'] = np.repeat(target_halo_id, synthetic_richness)
        sats['target_halo_fof_halo_id'] = np.repeat(target_halo_fof_halo_id, synthetic_richness)
        sats['lightcone_replication'] = np.repeat(target_halo_lightcone_replication, synthetic_richness)
        sats['lightcone_rotation'] = np.repeat(target_halo_lightcone_rotation, synthetic_richness)

        sats['upid'] = sats['target_halo_id']

        #  Use Halotools to generate halo-centric positions and velocities according to NFW
        nfw = NFWPhaseSpace()
        nfw_sats = nfw.mc_generate_nfw_phase_space_points(mass=sats['target_halo_mass'])

        sats['host_centric_x'] = nfw_sats['x']
        sats['host_centric_y'] = nfw_sats['y']
        sats['host_centric_z'] = nfw_sats['z']
        sats['host_centric_vx'] = nfw_sats['vx']
        sats['host_centric_vy'] = nfw_sats['vy']
        sats['host_centric_vz'] = nfw_sats['vz']

        #  Add host-centric pos/vel to target halo pos/vel
        sats['x'] = sats['target_halo_x'] + nfw_sats['x']
        sats['y'] = sats['target_halo_y'] + nfw_sats['y']
        sats['z'] = sats['target_halo_z'] + nfw_sats['z']
        sats['vx'] = sats['target_halo_vx'] + nfw_sats['vx']
        sats['vy'] = sats['target_halo_vy'] + nfw_sats['vy']
        sats['vz'] = sats['target_halo_vz'] + nfw_sats['vz']

        if Lbox > 0.:    # enforce periodicity
            sats['x'] = np.mod(sats['x'], Lbox)
            sats['y'] = np.mod(sats['y'], Lbox)
            sats['z'] = np.mod(sats['z'], Lbox)

        #  Assign synthetic subhalo mass according to a power law
        #  Maximum allowed value of the subhalo mass is the host halo mass
        #  Power law distribution in subhalo mass spans [11, logMhost]
        alpha = np.zeros_like(sats['target_halo_mass']) + 2.0  #  power-law slope
        sats['mpeak'] = 10**(np.log10(sats['target_halo_mass']) - 4*powerlaw.rvs(alpha))

        sats['halo_id'] = -1
        sats['source_halo_id'] = -1
        sats['source_halo_mvir'] = sats['target_halo_mass']

        #  Assign M* according to the Halotools implementation
        #  of the Moster+13 stellar-to-halo-mass relation
        moster_model = Moster13SmHm()
        sats['obs_sm'] = moster_model.mc_stellar_mass(
            prim_haloprop=sats['mpeak'], redshift=sats['target_halo_redshift'])
        sats['obs_sfr'] = np.random.normal(loc=-12, scale=0.2, size=len(sats))*sats['obs_sm']
        sats['sfr_percentile'] = np.random.uniform(0, 0.33, len(sats))
        sats['_obs_sm_orig_um_snap'] = sats['obs_sm']

        #  We will only boost the richness of red sequence cluster members
        sats['is_on_red_sequence_gr'] = True
        sats['is_on_red_sequence_ri'] = True

        #  Model r-band magnitude just like regular galaxies
        sats['restframe_extincted_sdss_abs_magr'] = magr_monte_carlo(
            sats['obs_sm'], sats['upid'], sats['target_halo_redshift'])

        #  Model g-r and r-i color just like regular red sequence galaxies
        red_sequence_loc_gr = red_sequence_peak_gr(
            sats['restframe_extincted_sdss_abs_magr'], sats['target_halo_redshift'])
        red_sequence_loc_ri = red_sequence_peak_ri(
            sats['restframe_extincted_sdss_abs_magr'], sats['target_halo_redshift'])

        sats['restframe_extincted_sdss_gr'] = np.random.normal(loc=red_sequence_loc_gr, scale=0.05)
        sats['restframe_extincted_sdss_ri'] = np.random.normal(loc=red_sequence_loc_ri, scale=0.02)

        #  It is important to insure that `sats` the `dc2` mock have the exact same columns,
        #  since these two get combined by a call to `astropy.table.vstack`. Here we enforce this:
        msg = ("The synthetic satellites columns must be the same as the regular mock\n"
            "sats keys = {0}\nmock keys = {1}")
        satkeys = list(sats.keys())
        dc2keys = list(mock.keys())
        assert set(satkeys) == set(dc2keys), msg.format(satkeys, dc2keys)

        return sats
