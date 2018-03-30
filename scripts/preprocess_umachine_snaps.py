"""
"""
import sys
sys.path.insert(0, "/scratch2/scratchdirs/aphearin/galsampler/build/lib.linux-x86_64-2.7")
sys.path.insert(0, "/scratch2/scratchdirs/aphearin/halotools/build/lib.linux-x86_64-2.7")
sys.path.insert(0, "/scratch2/scratchdirs/aphearin/cosmodc2")

import numpy as np
import os
import argparse
from halotools.utils import crossmatch
from halotools.empirical_models import enforce_periodicity_of_box
from halotools.mock_observables import relative_positions_and_velocities
from cosmodc2.umachine_processing import reformat_umachine_binary_output
from cosmodc2.umachine_processing import calculate_host_centric_position_velocity
from galsampler.utils import compute_richness
from galsampler.source_galaxy_selection import _galaxy_table_indices
from halotools.utils import sliding_conditional_percentile
from cosmodc2.generate_snapshot_collection import load_bpl_halos


parser = argparse.ArgumentParser()
parser.add_argument("umachine_sfr_catalog_fname",
    help="Absolute path to .bin file storing UniverseMachine output")
parser.add_argument("bpl_halo_catalog_fname",
    help="Absolute path to .bin file storing UniverseMachine output")
parser.add_argument("umachine_catalog_outname",
    help="Absolute path to hdf5 file storing value-added UniverseMachine mock")
parser.add_argument("bpl_halo_catalog_outname",
    help="Absolute path to hdf5 file storing value-added halo catalog")

args = parser.parse_args()

umachine_mock = reformat_umachine_binary_output(args.umachine_sfr_catalog_fname)
umachine_mock = umachine_mock[umachine_mock['obs_sm'] > 10**6]
Lbox = 250.

#  Load both catalogs into memory
umachine_mock['x'] = enforce_periodicity_of_box(umachine_mock['x'], Lbox)
umachine_mock['y'] = enforce_periodicity_of_box(umachine_mock['y'], Lbox)
umachine_mock['z'] = enforce_periodicity_of_box(umachine_mock['z'], Lbox)

#  Correct for edge case where position is exactly on box boundary
epsilon = 0.0001
umachine_mock['x'][umachine_mock['x'] == Lbox] = Lbox-epsilon
umachine_mock['y'][umachine_mock['y'] == Lbox] = Lbox-epsilon
umachine_mock['z'][umachine_mock['z'] == Lbox] = Lbox-epsilon


bpl_halos = load_bpl_halos(args.bpl_halo_catalog_fname)
print("bpl_halos has been loaded")

#  Throw out the small number of galaxies for which there is no matching host
umachine_mock['hostid'] = umachine_mock['upid']
host_halo_mask = umachine_mock['upid'] == -1
umachine_mock['hostid'][host_halo_mask] = umachine_mock['id'][host_halo_mask]

idxA, idxB = crossmatch(umachine_mock['hostid'], bpl_halos['halo_id'])
umachine_mock = umachine_mock[idxA]



#  Compute host halo position for every UniverseMachine galaxy
print("...computing host halo properties ")
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
print("...computing host halo-centric positions and velocities")
result = calculate_host_centric_position_velocity(umachine_mock, Lbox)
xrel, yrel, zrel, vxrel, vyrel, vzrel = result
umachine_mock['host_centric_x'] = xrel
umachine_mock['host_centric_y'] = yrel
umachine_mock['host_centric_z'] = zrel
umachine_mock['host_centric_vx'] = vxrel
umachine_mock['host_centric_vy'] = vyrel
umachine_mock['host_centric_vz'] = vzrel


#  Add column for sfr_percentile
print("...computing SFR percentile")
nwin = 501
x = umachine_mock['obs_sm']
y = umachine_mock['obs_sfr']
umachine_mock['sfr_percentile'] = sliding_conditional_percentile(x, y, nwin)


#  Sort the mock by host halo ID, putting centrals first within each grouping
umachine_mock.sort(('hostid', 'upid'))

#  Compute the number of galaxies in each Bolshoi-Planck halo
print("...computing richness")
bpl_halos['richness'] = compute_richness(
    bpl_halos['halo_id'], umachine_mock['hostid'])

#  For every Bolshoi-Planck halo, compute the index of the galaxy table
#  storing the central galaxy, reserving -1 for unoccupied halos
print("...computing galaxy selection indices")
bpl_halos['first_galaxy_index'] = _galaxy_table_indices(
    bpl_halos['halo_id'], umachine_mock['hostid'])
#  Because of the particular sorting order of umachine_mock,
#  knowledge of `first_galaxy_index` and `richness` gives sufficient
#  information to map the correct members of umachine_mock to each halo



#  Write results to disk
print("...writing to disk")
bpl_halos.write(args.bpl_halo_catalog_outname, path='data', overwrite=True)
umachine_mock.write(args.umachine_catalog_outname, path='data', overwrite=True)

