import numpy as np
import sys
import pickle
import os

halo_dir_template = {'AlphaQ': '/projects/DarkUniverse_esp/heitmann/OuterRim/M000/L360/HACC001/analysis/Halos/b0168/fofp_new',
                     'LC_Test': '/projects/SkySurvey/rangel/lc_test_data/b0168/STEP{}/',
           }
halo_file_template = {'AlphaQ': 'm000-{}.fofproperties',
                      'LC_Test':'m000-{}.fofproperties',
                     } 

def get_fof_info(pkldirname, nsnapshot=29, sim_name='AlphaQ', pklname='{}_z2ts.pkl'):
    z2ts = pickle.load(open(os.path.join(pkldirname, pklname.format(sim_name)),'rb'))
    redshifts = [key for i, key in enumerate(sorted(z2ts.keys())) if i <= nsnapshot] 
    snapshots = [z2ts[z] for z in redshifts]
    filename_list = [os.path.join(halo_dir_template[sim_name].format(str(snap)), halo_file_template[sim_name].format(str(snap))) for snap in snapshots]

    return redshifts, snapshots, filename_list
