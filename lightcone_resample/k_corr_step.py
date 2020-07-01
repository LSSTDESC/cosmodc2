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
#from multiprocessing import Process
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

    def load_mtree_list(self, mtree_fname_list, verbose=False):
        if verbose:
            t1 = time.time()
            print("\tLoading all files...")
        for i, fname in enumerate(mtree_fname_list):
            print("\t{}/{}".format(i,len(mtree_fname_list)))
            self.load_mtree(fname,verbose=verbose)
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



def match_index(gltcs_snapshot_ptrn, step1, step2, mtrees, output_file, output_index_only = False, verbose=False):
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
    # print(np.unique(nodeIndex1).size, nodeIndex1.size)
    # print(np.unique(nodeIndex2).size, nodeIndex2.size)
    # print(nodeIndex1)
    # print(nodeIndex2)
    # for i in range(0,25):
    #     find = nodeIndex2 == nodeIndex1[i]
    #     print(nodeIndex1[i], np.sum(find), np.where(find))
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
    slct_cnt = match_2to1 == -1
    match_2to1[slct_cnt] = central_2to1[slct_cnt]
    print("centrals required: ", np.sum(slct_cnt))
    print("central match:", np.sum(central_2to1!=-1))
    print("used central = -1: ",np.sum(central_2to1[slct_cnt]==-1))
    if verbose:
        t5 = time.time()
        slct = match_2to1 !=-1
        num_match = np.sum(slct)
        num_mismatch = slct.size - num_match 
        print(np.sum(nodeIndex1==nodeIndex2[match_2to1]),nodeIndex1.size)
        print("result: \n\tMatched: {}, no match: {}".format(num_match, num_mismatch))
        print("\t done getting central indexes {:.2f}".format(t5-t4))
    if output_index_only:
        t6 = time.time()
        hfile_out = h5py.File(output_file,'w')
        hfile_out['match_2to1'] = match_2to1
        if verbose:
            print("wrote index to file. time: {:.2f}".format(time.time()-t6))
            print("num matches: {:.2e}".format(np.sum(match_2to1 != -1)))
            print("Step done. Time: {:.2f}".format(time.time()-t1))
        return
    #Now we have found all galaxies from step1 in step2--stored in match_1to2
    #Next is to iterrate over all the filters and calculate the rate of change going from
    #step1 to step2
    stepZ = dtk.StepZ(sim_name = "AlphaQ")
    da = stepZ.get_a(step2)-stepZ.get_a(step1)
    print("\tda: {}".format(da))
    #get all keys
    keys = get_keys(hfile1['galaxyProperties'])
    hgroup_out = h5py.File(output_file,'w').create_group('galaxyProperties')
    # magr1 = hfile1['galaxyProperties']['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest'].value
    # magr2 = hfile2['galaxyProperties']['SDSS_filters/totalLuminositiesStellar:SDSS_r:rest'].value[match_2to1]
    # mstar1 = hfile1['galaxyProperties']['totalMassStellar'].value
    # mstar2 = hfile2['galaxyProperties']['totalMassStellar'].value[match_2to1]
    # log_del = np.log(mstar2/mstar1)
    # slct_mstar = (-1 < log_del) & (log_del < +1)
    for key in keys:
        t1 = time.time()
        print("\t {} ".format(key),end='')
        val1 = hfile1['galaxyProperties'][key].value
        val2 = hfile2['galaxyProperties'][key].value[match_2to1]
        # for k in range(0,10):
        #     print("\n{} => {}\n{} => {}".format(val_1[k],np.log(val_1[k]),val_2[k],np.log(val_2[k])))
        # val1 = np.log(magr1)-np.log(val_1)
        # val2 = np.log(magr2)-np.log(val_2)
        # print("============")
        # for k in range(0,10):
        #     print("\n{}-{} => {}\n{}-{} => {}".format(np.log(magr1[k]),np.log(val_1[k]),val1[k],
        #                                               np.log(magr2[k]),np.log(val_2[k]),val2[k]))
            
        slct_nomatch = match_2to1 == -1 #for any galaxy we didn't find a match, we just assume
        # a zero slope. Galacticus galaxies merge, so some loss fraction is expected. I'm
        #seeing ~ 1% unmatched. 
        val2[slct_nomatch] = val1[slct_nomatch]
        dval_da = (val2-val1)/da
        hgroup_out[key] = dval_da
        # print( val1)
        # print( val2)
        # print( da)
        # print( dval_da)
        # print("dval/da: min:{:.2f} avg{:.2f} max{:.2f}".format(np.min(dval_da),np.average(dval_da),np.max(dval_da)))
        # print("time:{:.2f}".format( time.time()-t1))
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
        # plt.legend()
        
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
        # plt.legend(loc='best')
        
        # log = True
        # bins = np.logspace(1,14,100)

        # plt.figure()
        # h,xbins,ybins = np.histogram2d(val1,val2,bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nAll")
        # plt.grid()

        # plt.figure()
        # slct = slct_cnt
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nCentrals {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.figure()
        # slct = ~slct_cnt 
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nNon central {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.figure()
        # slct = ~slct_cnt & ~slct_nomatch
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nSatellites {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.figure()
        # slct = slct_nomatch
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nNo Descn.fount {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.figure()
        # slct = ~slct_mstar
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nSmall M* change {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.figure()
        # slct = slct_mstar
        # h,xbins,ybins = np.histogram2d(val1[slct],val2[slct],bins=(bins,bins))
        # plt.pcolor(xbins,ybins,h.T,cmap='PuBu',norm=clr.LogNorm())
        # plt.xlabel('step {}'.format(step1))
        # plt.ylabel('step {}'.format(step2))
        # if log:
        #     plt.yscale('log')
        #     plt.xscale('log')
        # plt.title(key+"\nBig M* change {}".format(np.float(np.sum(slct))/np.float(slct.size)))
        # plt.grid()

        # plt.show()
    return 


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
    


    
if __name__ == "__main__mpi":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    print("rank: ",rank)
    param = dtk.Param(sys.argv[1])
    gltcs_snapshots_ptrn = param.get_string("gltcs_snapshots_ptrn")
    steps  = param.get_int_list("steps")
    mtree_ptrn = param.get_string("mtree_ptrn")
    mtree_num  = param.get_int("mtree_num")
    output_index_only = param.get_bool("output_index_only")
    output_ptrn = param.get_string("output_ptrn")
    mto = MTreeObj()
    verbose = True
    mto.load("tmp/mto.hdf5",verbose=verbose)
    for i in range(0,len(steps)-1):
        print(i,nproc,rank)
        if(i%nproc == rank):
            step2 = steps[i] #steps are in revervse chronological order
            step1 = steps[i+1]
            print("rank: {}. Working on {} -> {}".format(rank,step1,step2))
            match_index(gltcs_snapshots_ptrn, step1, step2, mto,
                        output_ptrn.replace("${step}", str(step1)), 
                        output_index_only = output_index_only,
                        verbose=True)

if __name__ == "__main__":
    param = dtk.Param(sys.argv[1])
    gltcs_snapshots_ptrn = param.get_string("gltcs_snapshots_ptrn")
    steps  = param.get_int_list("steps")
    mtree_ptrn = param.get_string("mtree_ptrn")
    mtree_num  = param.get_int("mtree_num")
    output_index_only = param.get_bool("output_index_only")
    output_ptrn = param.get_string("output_ptrn")
    verbose = True
    mto = MTreeObj()
    # if "mtree_list" in param:
    #     mtree_list = param.get_string_list('mtree_list')
    #     mto.load_mtree_list(mtree_list,verbose=verbose)
    # else:
    #     mto.load_mtrees(mtree_ptrn,mtree_num,verbose=verbose)
    # mto.save("tmp/mto.hdf5",verbose=verbose)
    mto.load("tmp/mto.hdf5",verbose=verbose)
    for i in range(0,len(steps)-1):
        step2 = steps[i] #steps are in revervse chronological order
        step1 = steps[i+1]
        print("Working on {} -> {}".format(step1,step2))
        match_index(gltcs_snapshots_ptrn, step1, step2, mto,
                    output_ptrn.replace("${step}", str(step1)), 
                    output_index_only = output_index_only,
                    verbose=True)

