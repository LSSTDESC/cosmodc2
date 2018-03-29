"""
"""
"""
"""
import numpy as np
import os
import argparse
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.mock_observables import relative_positions_and_velocities
from cosmodc2.umachine_processing import load_um_binary_sfr_catalog


parser = argparse.ArgumentParser()
parser.add_argument("umachine_sfr_catalog_fname",
    help="Absolute path to .bin file storing UniverseMachine output")
parser.add_argument("bpl_halo_catalog_fname",
    help="Absolute path to .bin file storing UniverseMachine output")

args = parser.parse_args()


#  Load both catalogs into memory
t['x'] = enforce_periodicity_of_box(t['x'], Lbox)
t['y'] = enforce_periodicity_of_box(t['y'], Lbox)
t['z'] = enforce_periodicity_of_box(t['z'], Lbox)

#  Correct for edge case where position is exactly on box boundary
epsilon = 0.0001
t['x'][t['x'] == Lbox] = Lbox-epsilon
t['y'][t['y'] == Lbox] = Lbox-epsilon
t['z'][t['z'] == Lbox] = Lbox-epsilon

umachine_mock = load_umachine_mstar_ssfr_mock(args.umachine_sfr_catalog_fname)


bpl_halos = load_bpl_halos(args.bpl_halo_catalog_fname)


#  Throw out the small number of galaxies for which there is no matching host
idxA, idxB = crossmatch(umachine_mock['hostid'], bpl_halos['halo_id'])
umachine_mock = umachine_mock[idxA]


#  Compute host halo position for every UniverseMachine galaxy
idxA, idxB = crossmatch(umachine_mock['hostid'], bpl_halos['halo_id'])

umachine_mock['host_halo_x'] = np.nan
umachine_mock['host_halo_y'] = np.nan
umachine_mock['host_halo_z'] = np.nan
umachine_mock['host_halo_vx'] = np.nan
umachine_mock['host_halo_vy'] = np.nan
umachine_mock['host_halo_vz'] = np.nan
umachine_mock['host_halo_mvir'] = np.nan

umachine_mock['host_halo_x'][idxA] = bpl_halos['x'][idxB]
umachine_mock['host_halo_y'][idxA] = bpl_halos['y'][idxB]
umachine_mock['host_halo_z'][idxA] = bpl_halos['z'][idxB]
umachine_mock['host_halo_vx'][idxA] = bpl_halos['vx'][idxB]
umachine_mock['host_halo_vy'][idxA] = bpl_halos['vy'][idxB]
umachine_mock['host_halo_vz'][idxA] = bpl_halos['vz'][idxB]
umachine_mock['host_halo_mvir'][idxA] = bpl_halos['mvir'][idxB]


#  Compute halo-centric position for every UniverseMachine galaxy
result = calculate_host_centric_position_velocity(umachine_mock)
xrel, yrel, zrel, vxrel, vyrel, vzrel = result
umachine_mock['host_centric_x'] = xrel
umachine_mock['host_centric_y'] = yrel
umachine_mock['host_centric_z'] = zrel
umachine_mock['host_centric_vx'] = vxrel
umachine_mock['host_centric_vy'] = vyrel
umachine_mock['host_centric_vz'] = vzrel






print("          Computing galaxy--halo correspondence for UniverseMachine galaxies/halos\n")

#  Sort the mock by host halo ID, putting centrals first within each grouping
umachine_mock.sort(('hostid', 'upid'))

#  Compute the number of galaxies in each Bolshoi-Planck halo
bpl_halos['richness'] = compute_richness(
    bpl_halos['halo_id'], umachine_mock['hostid'])

#  For every Bolshoi-Planck halo, compute the index of the galaxy table
#  storing the central galaxy, reserving -1 for unoccupied halos
bpl_halos['first_galaxy_index'] = _galaxy_table_indices(
    bpl_halos['halo_id'], umachine_mock['hostid'])
#  Because of the particular sorting order of umachine_mock,
#  knowledge of `first_galaxy_index` and `richness` gives sufficient
#  information to map the correct members of umachine_mock to each halo

