#!/usr/bin/env python2.7

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import dtk
import h5py
import time
import sys
#from mpi4py import MPI
from multiprocessing import Process
from scipy.interpolate import interp1d 


class MTreeObj:
    def __init__(self):
        self.nodeIndex_list = []
        self.descnIndex_list = []


    def load_mtree(self,mtree_fname,verbose=False):
        if verbose:
            t1 = time.time()
            print("\tLoading file {}".format(mtree_fname))
        hfile = h5py.File(mtree_fname,'r')
        nodeIndex = hfile['forestHalos/nodeIndex'].value
        descnIndex = hfile['forestHalos/descendentIndex'].value
        self.nodeIndex_list.append(nodeIndex)
        self.descnIndex_list.append(descnIndex)
        if verbose:
            print("\t\tdone. {:.2f}".format(time.time()-t1))


    def load_mtrees(self,mtree_fname_ptrn,num,verbose=False):
        if verbose:
            t1 = time.time()
            print("\tLoading all files...")
        for i in range(0,num):
            self.load_mtree(mtree_fname_ptrn.replace("${num}",str(i)),verbose=verbose)
        if verbose:
            t2 = time.time()
            print("\t\tDone. {:.2f}".format(t2-t1))
            print("\tSorting....")
        self.nodeIndex = np.concatenate(self.nodeIndex_list)
        self.descnIndex = np.concatenate(self.descnIndex_list)
        self.srt = np.argsort(self.nodeIndex)
        if verbose:
            print("\t\tDone. {:.2f}".format(time.time()-t2))


    def get_descn(self,nodeIndex,verbose=False):
        if verbose:
            t1 = time.time()
            print("\tFinding descendents...")
        indx = dtk.search_sorted(self.nodeIndex,nodeIndex,sorter=self.srt)
        descn_index = -np.ones_like(indx)
        slct = indx != -1
        descn_index[slct] = self.descnIndex[indx[slct]]
        if verbose:
            print("\t\tdone. {:.2f}".format(time.time()-t1))
        return descn_index

        
    def save(self, fname, verbose):
        t1 = time.time()
        hfile = h5py.File(fname,'w')
        hfile['nodeIndex'] = self.nodeIndex
        hfile['descnIndex'] = self.descnIndex
        hfile['srt'] = self.srt
        if verbose:
            print("done saving. {:.2f}".format(time.time()-t1))


    def load(self, fname, verbose):
        t1 = time.time()
        hfile = h5py.File(fname,'r')
        self.nodeIndex = hfile['nodeIndex'].value
        self.descnIndex = hfile['descnIndex'].value
        self.srt = hfile['srt'].value
        if verbose:
            print("done loading. {:.2f}".format(time.time()-t1))

def get_keys(hgroup):
    keys = []
    def _collect_keys(name, obj):
        if isinstance(obj, h5py.Dataset): 
            keys.append(name)
    hgroup.visititems(_collect_keys)
    return keys



def match_index(gltcs_snapshot_ptrn, step1, step2, mtrees, output_file,verbose=False):
    """Load two ajdacent galacticus snapshots (step 1 going to step
    2). Idenitify the same galaxies in the two snapshots either
    through having the same nodeIndex for satellites, or finding the
    descendentIndex through the merger trees. Once identified, calculate
    dflux/da for each filter and write out to a file.

    """
    if verbose:
        t1 = time.time()
        print("loading node index")
    hfile1 = h5py.File(gltcs_snapshot_ptrn.replace("${step}",str(step1)),'r')
    hfile2 = h5py.File(gltcs_snapshot_ptrn.replace("${step}",str(step2)),'r')
    nodeIndex1 = hfile1['galaxyProperties/infallIndex'].value
    nodeIndex2 = hfile2['galaxyProperties/infallIndex'].value
    if verbose:
        t2 = time.time()
        print("\t done {:.2f}".format(t2-t1))
    srt = np.argsort(nodeIndex2)
    if verbose:
        t3 = time.time()
        print("\t done sorting {:.2f}".format(t3-t2))
    match_2to1 = dtk.search_sorted(nodeIndex2,nodeIndex1,sorter=srt)
    if verbose:
        t4 = time.time()
        print("\t done getting satellte indexes {:.2f}".format(t4-t3))
        slct = match_2to1 != -1
        print(np.sum(nodeIndex1[slct]==nodeIndex2[match_2to1[slct]]), np.sum(slct))
    descnIndex  = mtrees.get_descn(nodeIndex1,verbose)
    central_2to1 = dtk.search_sorted(nodeIndex2,descnIndex,sorter=srt)
    slct = match_2to1 == -1
    match_2to1[slct] = central_2to1[slct]
    print("centrals required: ", np.sum(slct))
    print("central match:", np.sum(central_2to1!=-1))
    print("used central = -1: ",np.sum(central_2to1[slct]==-1))
    if verbose:
        t5 = time.time()
        slct = match_2to1 !=-1
        num_match = np.sum(slct)
        num_mismatch = slct.size - num_match 
        print(np.sum(nodeIndex1==nodeIndex2[match_2to1]),nodeIndex1.size)
        print("result: \n\tMatched: {}, no match: {}".format(num_match, num_mismatch))
        print("\t done getting central indexes {:.2f}".format(t5-t4))
    #Now we have found all galaxies from step1 in step2--stored in match_1to2
    #Next is to iterrate over all the filters and calculate the rate of change going from
    #step1 to step2
    stepZ = dtk.StepZ(sim_name = "AlphaQ")
    da = stepZ.get_a(step2)-stepZ.get_a(step1)
    print("da: {}".format(da))
    #get all keys
    keys = get_keys(hfile1['galaxyProperties'])
    hgroup_out = h5py.File(output_file,'r+').require_group('galaxyProperties')
    keys_done = get_keys(hgroup_out)
    for key in keys:
        t1 = time.time()
        print("\t {} ".format(key),end='')
        if key in keys_done:
            print("skipping.")
            continue
        val1 = hfile1['galaxyProperties'][key].value
        val2 = hfile2['galaxyProperties'][key].value[match_2to1]
        slct = match_2to1 == -1 #for any galaxy we didn't find a match, we just assume
        # a zero slope. Galacticus galaxies merge, so some loss fraction is expected. I'm
        #seeing ~ 1% unmatched. 
        val2[slct] = val1[slct]
        dval_da = (val2-val1)/da
        hgroup_out[key] = dval_da
        # print( val1)
        # print( val2)
        # print( da)
        # print( dval_da)
        # print("dval/da: min:{:.2f} avg{:.2f} max{:.2f}".format(np.min(dval_da),np.average(dval_da),np.max(dval_da)))
        print("time:{:.2f}".format( time.time()-t1))
        # plt.figure()
        # slct = val1>0
        # h,xbins = np.histogram(np.log10(val1[slct]),bins = 100)
        # plt.plot(dtk.bins_avg(xbins),h,label='step1 values')
        # slct = val2>0
        # h,xbins = np.histogram(np.log10(val2[slct]),bins = 100)
        # plt.plot(dtk.bins_avg(xbins),h,label='step2 values')
        # plt.title(key)
        # plt.grid()
        # plt.xlabel('val')
        # plt.ylabel('cnt')
        # plt.figure()
        # dval = val2-val1
        # slct =dval>0
        # h,xbins = np.histogram(np.log10(dval[slct]),bins=100)
        # plt.plot(dtk.bins_avg(xbins),h,label='pos')
        # slct = dval < 0
        # h,xbins = np.histogram(np.log10(-dval[slct]),bins=100)
        # plt.plot(dtk.bins_avg(xbins),h,label='neg')
        # plt.grid()
        # plt.xlabel('log10(dval)')
        # plt.ylabel('cnt')
        # plt.show()

if __name__ == "__main__2":
    print("finding the k-corr for glctcs")
    param = dtk.Param(sys.argv[1])
    gltcs_snapshots_ptrn = param.get_string("gltcs_snapshots_ptrn")
    steps  = param.get_int_list("steps")
    mtree_ptrn = param.get_string("mtree_ptrn")
    mtree_num  = param.get_int("mtree_num")
    output_ptrn = param.get_string("output_ptrn")
    mto = MTreeObj()
    s = mtree_ptrn.replace("${num}",str(0))
    verbose =True
    #mto.load_mtrees(mtree_ptrn,mtree_num,verbose=verbose)
    #mto.save("tmp/mto.hdf5",verbose=verbose)
    mto.load("tmp/mto.hdf5",verbose=verbose)
    ps = []
    for i in range(0,len(steps)-1):
        step2 = steps[i] #steps are in revervse chronological order
        step1 = steps[i+1]
        # match_index(gltcs_snapshots_ptrn, step1, step2, mto,
        #             output_ptrn.replace("${num}", str(step1)),
        #             verbose=True)
        p = Process(target=match_index,args=(gltcs_snapshots_ptrn, 
                                             step1, 
                                             step2, 
                                             mto,
                                             output_ptrn.replace("${num}", str(step1)),
                                             True))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()



#Old MPI Way
# if __name__ == "__main__":
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     nproc = comm.Get_size()
#     print("rank: ",rank)
#     param = dtk.Param(sys.argv[1])
#     gltcs_snapshots_ptrn = param.get_string("gltcs_snapshots_ptrn")
#     steps  = param.get_int_list("steps")
#     mtree_ptrn = param.get_string("mtree_ptrn")
#     mtree_num  = param.get_int("mtree_num")
#     output_ptrn = param.get_string("output_ptrn")
#     mto = MTreeObj()
#     verbose = True
#     mto.load("tmp/mto.hdf5",verbose=verbose)
#     for i in range(0,len(steps)-1):
#         print(i,nproc,rank)
#         if(i%nproc == rank):
#             step2 = steps[i] #steps are in revervse chronological order
#             step1 = steps[i+1]
#             print("rank: {}. Working on {} -> {}".format(rank,step1,step2))
#             match_index(gltcs_snapshots_ptrn, step1, step2, mto,
#                         output_ptrn.replace("${step}", str(step1)),
#                         verbose=True)

if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    gltcs_snapshots_ptrn = param.get_string("gltcs_snapshots_ptrn")
    steps  = param.get_int_list("steps")
    mtree_ptrn = param.get_string("mtree_ptrn")
    mtree_num  = param.get_int("mtree_num")
    output_ptrn = param.get_string("output_ptrn")
    mto = MTreeObj()
    verbose = True
    mto.load("tmp/mto.hdf5",verbose=verbose)
    for i in range(0,len(steps)-1):
        step2 = steps[i] #steps are in revervse chronological order
        step1 = steps[i+1]
        print("rank: {}. Working on {} -> {}".format("?",step1,step2))
        match_index(gltcs_snapshots_ptrn, step1, step2, mto,
                    output_ptrn.replace("${step}", str(step1)),
                    verbose=True)


