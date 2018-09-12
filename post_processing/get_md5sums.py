import glob
import re
import numpy as np
import subprocess as SP
import os
master_dir='/global/projecta/projectdirs/lsst/groups/CS/cosmoDC2/'
#filedir = cosmoDC2_v1.0.0_full_highres'
#version = '1.0.0'

def write_md5sums(filedir, version):
    filedir = os.path.join(master_dir, filedir)
    filelist = sorted(glob.glob(filedir+'/*.hdf5'))

    outfile = '_cosmoDC2_check_md5.yaml'
    print('Opening {}'.format(outfile))
    with open(outfile, 'w') as fh:
        fh.write("'{}' :\n".format(version))
        fh.write('  md5 :\n')

        for f in filelist:
            fname = os.path.basename(f)
            md5sum = re.split(' ', SP.check_output(['md5sum', f]).decode("utf-8"))[0]
            print('Processed {} with md5sum {}'.format(fname, md5sum))            
            fh.write('    {} : {}\n'.format(fname, md5sum))

        fh.close

    return
