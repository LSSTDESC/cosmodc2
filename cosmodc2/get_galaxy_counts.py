import numpy as np
import h5py
import os
import glob
from os.path import expanduser
import argparse

home = expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument("-um_healpix_mocks_dirname",
    help="Directory name (relative to home) storing um healpix mocks",
    default='cosmology/DC2/OR_Test/um_healpix_mocks')

args = parser.parse_args()

#setup directory names
healpix_mocks_dirname = os.path.join(home, args.um_healpix_mocks_dirname)


um_files = glob.glob(os.path.join(healpix_mocks_dirname, '*.hdf5'))

Ngal_max = 0

outfile = os.path.join('./', 'Ngalaxies.txt')
f = open(outfile, 'w')

for uf in um_files:
    udata = h5py.File(uf, 'r')
    lc_keys = [k for k in udata.keys() if k.isdigit()]
    Ngals = 0
    for k in lc_keys:
        Ngals = Ngals + len(udata[k]['halo_id'])

    f.write('{}: Ngals = {}\n'.format(os.path.basename(uf), Ngals))
    Ngal_max = max(Ngal_max, Ngals)

f.write('Ngal_max = {}'.format(Ngal_max))
f.close()
