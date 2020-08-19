import sys
import glob
import os
import re
import argparse
import numpy as np

hpxdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000/auxiliary_files'
hpx_template = 'healpix_cutouts/z_{}_{}/cutout_*.hdf5'
hpxfiles = 'hpx_z_{}_{}.txt'
file_template = 'pixels_{}_{}.txt'
hpxfiles_all = 'hpx_z_{}.txt'
size_min = 500000

def get_hpxlist(hpxdir, hpxfiles):
    #filelist = sorted(glob.glob(hpxdir))
    hpxlist = []
    with open(os.path.join(hpxdir, hpxfiles), 'r') as fh:
        contents = fh.read()
        lines = contents.splitlines()

    hpxlist = sorted([int(re.split('.hdf5', re.split('cutout_', l)[-1])[0]) for l in lines])
    return hpxlist

def check_file_sizes(hpxdir, z):
    fn = os.path.join(hpxdir, hpx_template.format(z, z+1))
    filelist = sorted(glob.glob(fn))
    sizes = np.asarray([os.stat(f).st_size for f in filelist])
    fnf = os.path.join(hpxdir, hpxfiles.format(z,'full'))
    fne = os.path.join(hpxdir, hpxfiles.format(z,'empty'))
    fhf = open(fnf, 'w')
    fhe = open(fne, 'w')
    mask = (sizes < size_min)
    print('Found {} empty pixels'.format(np.count_nonzero(mask)))

    for s, f in zip(sizes, filelist):
        if s < size_min:
            fhe.write('{}\n'.format(os.path.basename(f)))
        else:
            fhf.write('{}\n'.format(os.path.basename(f)))

    fhf.close()
    fhf.close()
    print('Wrote {}\n      {}'.format(fnf, fne))
    return

def main(argsdict):
    hpx_per_file = argsdict['stride']
    print('# per file = {}'.format(hpx_per_file))
    list_format = argsdict['list_format']
    if list_format:
        print('Output in list format')
    run_format = argsdict['run_format']
    if run_format:
        print('Output in run-list format')
    xname = argsdict['name']

    #first make list of full and empty pixels
    #for z in range(3):
    #    check_file_sizes(hpxdir, z)
    hpxfile = hpxfiles.format(0, 'full')
    hpxlist = get_hpxlist(hpxdir, hpxfile)
    total = len(hpxlist)
    nfiles = int(np.ceil(total/float(hpx_per_file)))
    
    for nf in range(nfiles):
        outfile = file_template.format(xname, nf)
        with open(outfile, 'w') as fh:
            for hpxn in hpxlist[nf*hpx_per_file:min((nf+1)*hpx_per_file, total)]:
                if list_format:
                    fh.write('{}, '.format(hpxn))
                elif run_format:
                    fh.write('{} '.format(hpxn))
                else:
                    fh.write('{}\n'.format(hpxn))


def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description='Makes hpx lists')
    parser.add_argument('--stride', type=int, help='Stride for file groups', default=128)
    parser.add_argument('--list_format', default=False, help='output in list format', action='store_true')
    parser.add_argument('--run_format', default=False, help='output in run list format', action='store_true')
    parser.add_argument('--name', help='extra name for pixels_{}_{}.txt', default='')
    args=parser.parse_args()
    argsdict=vars(args)
    print ("Running", sys.argv[0], "with parameters:")
    for arg in argsdict.keys():
        print(arg," = ", argsdict[arg])

    return argsdict

if __name__ == '__main__':    
    argsdict=parse_args(sys.argv)
    main(argsdict)
