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

    return h5file, redshifts, snapshots
