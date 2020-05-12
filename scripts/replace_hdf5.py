from __future__ import print_function, division
import os
import numpy as np
import h5py
from time import time
import sys
import re
import glob
import argparse

hpxdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/baseDC2_5000_v1.1.1/'
#hpxdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/baseDC2_5000_v1.1.1_zfix'
rundir = '/gpfs/mira-home/ekovacs/cosmology/DC2/cosmoDC2/OR_BaseDC2_5000_v1.1.1'
hpx_template = 'baseDC2_z_{}*_cutout_*.hdf5'
Nside = '32'
cutout_id_offset = int(1e9)
z_offsets =  {'32':[0, 1e8, 2e8, 3e8]}

def rewrite_pixels(zgroup=1, nfiles=None, quantity='galaxy_id'):
   
    hpx_tmp = os.path.join(hpxdir, hpx_template.format(zgroup))
    print(hpx_tmp)
    hpx_files = sorted(glob.glob(hpx_tmp))
    print('Processing {} files for z-range {}'.format(len(hpx_files), zgroup))
    nfiles = len(hpx_files) if nfiles is None else nfiles
    start = time()
    for hpx_file in hpx_files[0:nfiles]:
        timei = time()
        fh = h5py.File(hpx_file, 'a') #open in append mode
        hpx = int(os.path.basename(hpx_file).split('cutout_')[-1].split('.hdf5')[0])
        galaxy_id_offset = int(int(hpx)*cutout_id_offset + z_offsets[Nside][zgroup])
        ngals = []
        for k in sorted(fh.keys()):
            if 'metaData' in k:
                continue
            hgroup = fh[k]
            ngal = len(hgroup['galaxy_id'])
            ngals.append(ngal)
            imin = np.min(hgroup['galaxy_id'])
            imax = np.max(hgroup['galaxy_id'])
            print('Overwriting healpix {} step {} with {} galaxy-ids:'.format(hpx, k, ngals[-1]))
            print('...Old values range from {} to {}'.format(imin, imax))
            nmax = galaxy_id_offset + ngal
            hgroup['galaxy_id'][:]  = np.arange(galaxy_id_offset, nmax)
            print('...New values range from {} to {}'.format(galaxy_id_offset, nmax - 1))
            if nmax > int(hpx)*cutout_id_offset + z_offsets[Nside][zgroup + 1]:
                print('Warning: allowed range exceeded with value {} for step {}'.format(nmax, k))
            #update offset
            galaxy_id_offset = nmax
            
        # now close and reopen to check contents
        fh.close()
        fh = h5py.File(hpx_file, 'r')
        print('Re-opening hpx {} for check'.format(hpx))
        galaxy_id_offset = int(int(hpx)*cutout_id_offset + z_offsets[Nside][zgroup])
        for k, ngal in zip(sorted(fh.keys()), ngals):
            hgroup = fh[k]
            if len(hgroup['galaxy_id']) != ngal:
                print('Warning: check new: mismatch in expected number of galaxies for step {}'.format(k))
            if np.min(hgroup['galaxy_id']) != galaxy_id_offset:
                print('Warning: check new: mismatch in expected min value for step {}'.format(k))
            if np.max(hgroup['galaxy_id']) != galaxy_id_offset + ngal - 1:
                print('Warning: check new: mismatch in expected max value for step {}'.format(k))
            galaxy_id_offset += ngal
            
        fh.close()
        
        #update offset
        galaxy_id_offset = nmax
        print('Healpix run time = {0:.4f} minutes'.format((time() - timei)/60.))
            
    print('Total run time = {0:.4f} minutes'.format((time() - start)/60.))
    
    return

def main(argsdict):
    zgroup = argsdict['zgroup']
    nfiles = argsdict['nfiles']
    rewrite_pixels(zgroup=zgroup, nfiles=nfiles)
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Replaces hdf5 column')
    parser.add_argument('--zgroup', type=int, help='zgroup to run', default=0)
    parser.add_argument('--nfiles', type=int, help='number of files to run (None runs all files in group)', default=None)
    args=parser.parse_args()
    argsdict=vars(args)
    print ("Running", sys.argv[0], "with parameters:")
    for arg in argsdict.keys():
        print(arg," = ", argsdict[arg])

    return argsdict

if __name__ == '__main__':    
    argsdict=parse_args(sys.argv)
    main(argsdict)

