import sys
import glob
import os
import re
import argparse
import numpy as np

hpxdir = '/gpfs/mira-home/*/healpix_cutouts/z_0_1/*'
hpxfiles = './hpx_z_0.txt'
file_template = 'pixels_{}.txt'

def get_hpxlist(hpxdir, hpxfiles):
    #filelist = sorted(glob.glob(hpxdir))
    hpxlist = []
    with open(hpxfiles, 'r') as fh:
        contents = fh.read()
        lines = contents.splitlines()

    hpxlist = sorted([int(re.split('.hdf5', re.split('cutout_', l)[-1])[0]) for l in lines])
    return hpxlist


def main(argsdict):
    nfiles = argsdict['nfiles']
    total = argsdict['total']
    hpx_per_file = argsdict['stride']
    #hpx_per_file = int(np.ceil(float(total)/float(nfiles)))
    print('# per file = {}'.format(hpx_per_file))

    hpxlist = get_hpxlist(hpxdir, hpxfiles)
    for nf in range(nfiles):
        outfile = file_template.format(nf)
        with open(outfile, 'w') as fh:
            for hpxn in hpxlist[nf*hpx_per_file:min((nf+1)*hpx_per_file, total)]:
                fh.write('{}\n'.format(hpxn))


def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Makes hpx lists')
    parser.add_argument('--nfiles',help='Number of hpx file lists (=number of jobs)', default=17)
    parser.add_argument('--total',type=float,help='Total number of files', default=2122)
    parser.add_argument('--stride',type=float,help='Stride for file groups', default=128)
    args=parser.parse_args()
    argsdict=vars(args)
    print ("Running", sys.argv[0], "with parameters:")
    for arg in argsdict.keys():
        print(arg," = ", argsdict[arg])

    return argsdict

if __name__ == '__main__':    
    argsdict=parse_args(sys.argv)
    main(argsdict)
