import numpy as np
import h5py

lc_block_num_offset = np.int64(1e15)
lc_step_num_offset  = np.int64(1e12)

def add_lightcone_id_block_num(block_num, lc_id):
    return lc_id + lc_block_num_offset*block_num;
    
def add_lightcone_id_step_num(step_num, lc_id):
    return lc_id + lc_step_num_offset*step_num;

def extract_lightcone_id_block_num(lc_id):
    return (lc_id/lc_block_num_offset).astype(int)
    
def extract_lightcone_id_step_num(lc_id):
    return ((lc_id%lc_block_num_offset)/lc_step_num_offset).astype(int)
    
def extract_lightcone_id(lc_id):
    return lc_id%lc_block_num_offset%lc_step_num_offset

def validate_lightcone_ids(block_num, step_num, max_lc_id):
    """Makes sure that that block_num, step_num and ids do not overlap
    and that all values are recoverable. 

    """
    assert(lc_block_num_offset > lc_step_num_offset)
    assert(lc_step_num_offset > max_lc_id)
    max_step = np.int(lc_block_num_offset/lc_step_num_offset)
    assert(step_num < max_step)
    max_int64 = np.iinfo(np.int64).max #Max int64 value
    max_block = np.int(max_int64/lc_block_num_offset)
    assert(max_block > block_num)

def append_lightcone_id(block_num, step_num, tbl):
    """Assigns a unique ID to each row in the astropy table, with block
    and step embedded into the id. The id will read as
    XXXYYYZZZZZZZZZZ (decimal), were xxx is the block number, yyy is
    the step number and zzzzzzzzzz is unique id for this block/step
    combination. The exact size of xxx and yyy are specified by
    lc_block_num_offset and lc_step_num_offset. The left over space
    in np.int64 is for zzz. 

    """
    keys = tbl.keys()
    table_size = tbl[keys[0]].quantity.shape[0]
    lightcone_id = np.arange(table_size,dtype=np.int64)
    max_id= np.max(lightcone_id)
    validate_lightcone_ids(block_num, step_num, max_id)
    lightcone_id_b = add_lightcone_id_block_num(block_num, lightcone_id)
    lightcone_id_bs = add_lightcone_id_step_num(step_num, lightcone_id_b)
    tbl['lightcone_id'] = lightcone_id_bs
    

def astropy_table_to_lightcone_hdf5(tbl, hdf5_fname, commit_hash):
    """
    Takes in an astropy Table object and writes it an easy to 
    read hdf5 file with only the data that's needed for the light
    cone. 
    """
    hfile = h5py.File(hdf5_fname,'w')
    hfile['x'] = tbl['x'].quantity
    hfile['y'] = tbl['y'].quantity
    hfile['z'] = tbl['z'].quantity
    hfile['vx'] = tbl['vx'].quantity
    hfile['vy'] = tbl['vy'].quantity
    hfile['vz'] = tbl['vz'].quantity
    hfile['id'] = tbl['lightcone_id'].quantity
    hfile.attrs.create('cosmodc2_commit', commit_hash)
    hfile.close()
    return
