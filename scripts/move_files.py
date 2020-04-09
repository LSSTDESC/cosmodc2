import sys
import glob
import os

hpxdir = '/gpfs/mira-fs0/projects/DarkUniverse_esp/kovacs/OR_5000'
hpxlist = 'hpx_z_0_empty.txt'
base_dir = 'baseDC2_5000_v1.1.1'
new_dir = 'baseDC2_5000)v1.1.1_empty'

def read_list(fn):
    with open(os.path.join(hpxdir, hpxlist), 'r') as fh:
        contents = fh.read()
        lines = contents.splitlines()

    return lines

def move_files(cutout_list, base_dir):
    nmoved = 0
    for hpx in cutout_list:
        fn = os.path.join(hpxdir, base_dir,'*'+ hpx)
        found_files = glob.glob(fn)
        for f in found_files:
            new_name = os.path.join(hpxdir, new_dir, os.path.basename(f))
            os.rename(f, new_name)
            nmoved += 1

    print('Moved {} files to {}'.format(nmoved, new_dir))

    moved_files = os.listdir(os.path.join(hpxdir, new_dir))
    print('Total of {} files in {}'.format(len(moved_files), new_dir))
