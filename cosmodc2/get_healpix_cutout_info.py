import pickle
import os
import h5py

def get_healpix_cutout_info(pkldirname, infile, sim_name='AlphaQ', pklname='{}_z2ts.pkl'):
    z2ts = pickle.load(open(os.path.join(pkldirname, pklname.format(sim_name)),'rb'))

    snapshots = []
    redshifts = []
    h5file = None
    if os.path.exists(infile):
        h5file = h5py.File(infile, 'r')
        snapshots = [str(k) for k in sorted(h5file.keys())][::-1]  #reverse order
        redshifts = [key for key in sorted(z2ts.keys()) if str(z2ts[key]) in snapshots]
        available_snapshots = [str(z2ts[key]) for key in sorted(z2ts.keys()) if (key <= max(redshifts) and key >= min(redshifts))]
        missing_snapshots = [s for s in available_snapshots if not s in snapshots]
        if missing_snapshots:
            print('{} is missing snapshots {}'.format(os.path.basename(infile), ', '.join(missing_snapshots)))
    else:
        print('{} not found'.format(infile))

    return h5file, redshifts, snapshots, z2ts

def get_snap_redshift_min(z2ts, snapshots):
    assert (len(z2ts) > 0), 'z2ts data NOT supplied'
    all_snapshots =  sorted([v for v in z2ts.values()])[::-1]
    # find redshift of preceding snapshot (499 is at index 0 and is never in snapshots list)
    previous_snap = all_snapshots[all_snapshots.index(int(max(snapshots))) - 1]
    previous_redshift = list(z2ts.keys())[z2ts.values().index(previous_snap)]
    
    return previous_redshift
