#!/usr/bin/env python2.7
import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import dtk
import h5py 

def frag_to_real(htags):
    return np.abs(htags) & 0x0000ffffffffffff

def major_frag_to_real(htags):
    return np.abs(htags) 


class Catalog:
    
    def __init__(self,file_loc=None,step_string=None):
        self.step_data = {}
        self.in_file_steps = {}
        self.var_names = []
        self.var_infile_name = {}
        self.var_index = {}
        self.file_source_ = ""
        self.step_string_ = ""
        self.srt = None
        self.subfiles = []
        self.file_list = None
        self.var_step_replace = []
        if(file_loc != None):
            if(step_string != None):
                self.set_file(file_loc,step_string)
            else:
                self.set_file(file_loc)
                
    def set_file(self,file_source,step_string="${step}",subfile_string="${subfile}"):
        self.file_source_ = file_source
        self.step_string_ = step_string
        self.subfile_string_ = subfile_string

    def set_explicit_files(self,file_list):
        self.file_list = file_list

    def add_steps(self,steps,in_file_steps=None):
        if(in_file_steps == None):
            for step in steps:
                self.in_file_steps[step]   = step
                self.step_data[step] = {}
        else:
            for i in range(0,steps.size):
                self.in_file_steps[steps[i]] = in_file_steps[i]
                self.step_data[steps[i]] = {}

    def add_step(self,step,in_file_step=None):
        if(in_file_step == None):
            self.in_file_steps[step] = step;
            self.step_data[step]={}
        else:
            self.in_file_steps[step]=in_file_step
            self.step_data[step]={}

    def add_subfiles(self,subfiles):
        self.subfiles = self.subfiles + subfiles

    def add_subfile(self,subfile):
        self.subfiles.append(subfile)

    def clear_subfiles(self):
        self.subfiles = []

    def get_steps(self):
        return self.step_data.keys()
        
    def add_var(self, name,as_name=None,index=None):
        if(as_name == None):
            self.var_names.append(name)
            self.var_infile_name[name]=name
            self.var_index[name]=index
        else:
            self.var_names.append(as_name)
            self.var_infile_name[as_name]=name
            self.var_index[as_name]=index

    def delete_var(self,var):
        if(var in self.var_names):
            self.var_names.remove(var)
            del self.var_infile_name[var]
            del self.var_index[var]
        while(var in self.var_names):
            self.var_names.remove(var)
    def delete_vars(self,var_list):
        for var in var_list:
            self.delete_var(var)

    def add_vars(self,var_names, as_names=None, indexs=None):
        size = len(var_names)
        for i in range(0,size):
            var = var_names[i]
            if(as_names != None):
                as_name = as_names[i]
            else:
                as_name = None
            if(indexs != None):
                index = indexs[i]
            else:
                index = None
            self.add_var(var,as_name,index)

    def refresh_vars(self):
        new_vars = []
        #Check if there are new variables#
        for step in self.step_data.keys():
            for var in self.step_data[step].keys():
                if var not in self.var_names:
                    new_vars.append(var)
        #Check that all steps have the new variables. Otherwise 
        #throw an error
        for step in self.step_data.keys():
            print "step: ",step
            for new_var in new_vars:
                if new_var not in self.step_data[step].keys():
                    raise KeyError(new_var+" not found in step "+str(step)+". Cannot add new variables until that variable is set for all catalog timesteps")
        for new_var in new_vars:
            self.add_var(new_var)

    def get_vars(self):
        return self.var_names
        
    def add_var_step_replace(self,key_string,steps,new_strings):
        # key string and new strings MUST be stings
        if(len(steps) != len(new_strings)):
            print "Input steps and new strings must be equal..."
            raise AssertionError()
        
        dic = {}
        for i in range(0,len(steps)):
            dic[steps[i]]=new_strings[i]
        
        self.var_step_replace.append([key_string,dic])

    def rename(self,var_old_name, var_new_name):
        #print "in renaming...",var_old_name,"->",var_new_name
        #print self.var_names
        #print "doen printing var names..."
        self.var_names.append(var_new_name)
        self.var_names.remove(var_old_name)
        self.var_infile_name[var_new_name]=self.var_infile_name[var_old_name]
        for step in self.step_data.keys():
            self.step_data[step][var_new_name]=self.step_data[step][var_old_name]
            del self.step_data[step][var_old_name]
        
    def read_gio(self,verbose=False):
        if(verbose):
            print "from file",self.file_source_
        for step in self.step_data.keys():
            if(verbose):
                print "\treading in step: ",step
            in_file_step = self.in_file_steps[step]
            file_name = self.file_source_.replace(self.step_string_,"%d"%in_file_step)
            for name in self.var_names:
                if(verbose):
                    print "\treading in ",name
                data = dtk.gio_read(file_name,
                                    self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step))
                self.step_data[step][name] = data

    def read_none(self,verbose=False):
        if(verbose):
            print "sort of from file",self.file_source_
        for step in self.step_data.keys():
            if(verbose):
                print "fake reading of step: ",step
            for name in self.var_names:
                if(verbose):
                    print "\tfaking in ",name
                self.step_data[step][name] = np.zeros(0,dtype='f')

    def read_hdf5(self,verbose=False,index=None):
        print "from file",self.file_source_
        for step in self.step_data.keys():
            if(verbose):
                print "\treading in step: ",step
            in_file_step = self.in_file_steps[step]
            file_name = self.file_source_.replace(self.step_string_,"%d"%in_file_step)
            for name in self.var_names:
                if(verbose):
                    print "\treading in ",name
                subfile_data = []
                if(len(self.subfiles)==0): #if no subfiles are listed, 
                    self.subfiles.append(0)#add a single dummy subfile
                if(self.file_list == None):
                    for subfile in self.subfiles:
                        hfile = h5py.File(file_name.replace(self.subfile_string_,"%d"%subfile),'r')
                        var = self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step)
                        if(self.var_index[name] == None):
                            subfile_data.append(hfile[var][:])
                        else:
                            subfile_data.append(hfile[var][:,self.var_index[name]])
                else:
                    for subfile in self.file_list:
                        print "reading subfile: ",subfile
                        hfile = h5py.File(subfile,'r')
                        var = self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step)
                        if(self.var_index[name] == None):
                            subfile_data.append(hfile[var][:])
                        else:
                            subfile_data.append(hfile[var][:,self.var_index[name]])
                data = np.concatenate(subfile_data)
                self.step_data[step][name]=data

    def read_hdf5_no_step_file(self,verbose=False,index=None):
        print "from file",self.file_source_
        if(len(self.subfiles)==0): #if no subfiles are listed, 
            self.subfiles.append(0)#add a single dummy subfile
        tmp_data = {}
        for step in self.step_data.keys():
            tmp_data[step] = {}
            for name in self.var_names:
                tmp_data[step][name] = []
        if self.file_list == None:
            for subfile in self.subfiles:
                if(verbose):
                    print "\t from subfile",subfile
                hfile = h5py.File(self.file_source_.replace(self.subfile_string_,"%d"%subfile),'r')
                for step in self.step_data.keys():
                    if(verbose):
                        print "\treading in step: ",step
                    in_file_step = self.in_file_steps[step]
                    for name in self.var_names:
                        if(verbose):
                            print "\treading in ",name
                        var = self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step)
                        for rep_rule in self.var_step_replace:
                            if(step in rep_rule[1]):
                                var = var.replace(rep_rule[0],rep_rule[1][step])
                        if(self.var_index[name] == None):
                            data=hfile[var].value
                        else:
                            data=hfile[var][:,self.var_index[name]]
                        tmp_data[step][name].append(data)
        else:
            for subfile in self.file_list:
                if(verbose):
                    print "\t from subfile",subfile
                hfile = h5py.File(subfile,'r')
                for step in self.step_data.keys():
                    if(verbose):
                        print "\treading in step: ",step
                    in_file_step = self.in_file_steps[step]
                    for name in self.var_names:
                        if(verbose):
                            print "\treading in ",name
                        var = self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step)
                        for rep_rule in self.var_step_replace:
                            if(step in rep_rule[1]):
                                var = var.replace(rep_rule[0],rep_rule[1][step])
                        if(self.var_index[name] == None):
                            data=hfile[var].value
                        else:
                            data=hfile[var][:,self.var_index[name]]
                        tmp_data[step][name].append(data)

        # for subfile in self.subfiles:
        #     if(verbose):
        #         print "\t from subfile",subfile
        #     hfile = h5py.File(self.file_source_.replace(self.subfile_string_,"%d"%subfile),'r')
        #     for step in self.step_data.keys():
        #         if(verbose):
        #             print "\treading in step: ",step
        #         in_file_step = self.in_file_steps[step]
        #         for name in self.var_names:
        #             if(verbose):
        #                 print "\treading in ",name
        #             var = self.var_infile_name[name].replace(self.step_string_,"%d"%in_file_step)
        #             for rep_rule in self.var_step_replace:
        #                 if(step in rep_rule[1]):
        #                     var = var.replace(rep_rule[0],rep_rule[1][step])
        #             if(self.var_index[name] == None):
        #                 data=hfile[var].value
        #             else:
        #                 data=hfile[var][:,self.var_index[name]]
        #             tmp_data[step][name].append(data)
        for step in self.step_data.keys():
            for name in self.var_names:
                self.step_data[step][name]=np.concatenate(tmp_data[step][name])




    def __getitem__(self,key):
        return self.step_data[key]

    def select(self,catalog1,slct,remove_slct=True):
        for step in catalog1.step_data.keys():
            self.step_data[step]= {}
            for name in catalog1.var_names:
                self.var_names.append(name)
                self.step_data[step][name] = catalog1[step][name][slct]
                if(remove_slct):
                    catalog1[step][name] = catalog1[step][name][slct==0]
                    
    def join(self,catalog1,catalog2,join_on='fof_halo_tag',req_also=None,
             verbose=False,many_to_one=False,random_to_one=False,
             remove_matched=True,remove_matched_1=False,remove_matched_2=False,
             also_require=None):
             
        #only merger if they are on the same timestep
        if(verbose):
            print "merging catalogs"

        for step in catalog1.step_data.keys():
            if step in catalog2.step_data.keys():
                if(verbose):
                    print "Both have step: ",step
                self.step_data[step]= {}
                srt1 = np.argsort(catalog1.step_data[step][join_on])
                srt2 = np.argsort(catalog2.step_data[step][join_on])
                if(verbose):
                    print "cat1 size: ", srt1.size
                    print "cat2 size: ", srt2.size
                i1 = 0
                i2 = 0
                i1_max = srt1.size
                i2_max = srt2.size
                match1 =[]
                match2 =[]
                unmatched1 = []
                unmatched2 = []
                if(verbose):
                    print "sorting"
                while(i1<i1_max and i2<i2_max):
                    val1 = catalog1.step_data[step][join_on][srt1[i1]]

                    val2 = catalog2.step_data[step][join_on][srt2[i2]]
                    if(req_also != None):
                        valrq1 =  catalog1.step_data[step][req_also][srt1[i1]]
                        valrq2 =  catalog2.step_data[step][req_also][srt2[i2]]
                    else:
                        valrq1 = None
                        valrq2 = None
                    have1 = False
                    if(val1 == val2 and valrq1 == valrq2):
                        if(random_to_one):
                            random_matches1 = []
                            ir1 = i1
                            while(ir1<i1_max):
                                valr1 = catalog1.step_data[step][join_on][srt1[ir1]]
                                if(valr1 == val2):
                                    random_matches1.append(srt1[ir1])
                                    ir1 +=1
                                else:
                                    break
                            random_match = random_matches1[np.random.randint(0,len(random_matches1))]
                            match1.append(random_match)
                        else:
                            if(not have1):
                                match1.append(srt1[i1])
                        match2.append(srt2[i2])
                        if(not random_to_one):
                            i1+=1
                        if(not many_to_one):
                            i2+=1
                            have1 = True
                    elif(val1 > val2):
                        if(not have1):
                            unmatched2.append(srt2[i2])
                        have1 = False
                        i2+=1
                    else: #(val1 < val2):
                        unmatched1.append(srt1[i1])
                        i1+=1
                while(i1<i1_max):
                    unmatched1.append(srt1[i1])
                    i1+=1
                while(i2<i2_max):
                    unmatched2.append(srt2[i2])
                    i2+=1
                match1 = np.atleast_1d(np.array(match1).astype(int))
                match2 = np.atleast_1d(np.array(match2).astype(int))
                unmatched1 = np.array(unmatched1)
                unmatched2 = np.array(unmatched2)
                if(verbose):
                    print "match1: ", match1.size
                    print "match2: ", match2.size
                    print "unmatched1: ", unmatched1.size
                    print "unmatched2: ", unmatched2.size

                    print "making merged catalog"

                #copy over the matched fields to this catalog
                #and only leave the unmatched data rows in the
                #original catalog
                names_this_step = []
                for name in catalog1.var_names:
                    self.step_data[step][name] = catalog1[step][name][match1]
                    if(verbose):
                        print step,name,"size:",self.step_data[step][name].size
                    names_this_step.append(name)
                    if(name not in self.var_names):
                        self.var_names.append(name)
                    if(remove_matched or remove_matched_1):
                        if(unmatched1.size != 0):
                            catalog1[step][name]=catalog1[step][name][unmatched1]
                        else:
                            catalog1[step][name]=np.zeros(0,dtype=catalog1[step][name].dtype)
                for name in catalog2.var_names:
                    print "name:",name
                    if(name not in names_this_step): #not to add the same column multiple times
                        self.step_data[step][name] = catalog2[step][name][match2]
                    if(name not in self.var_names):
                        self.var_names.append(name)
                    if(remove_matched or remove_matched_2):
                        if(unmatched2.size != 0):
                            catalog2[step][name]= catalog2[step][name][unmatched2]
                        else:
                            catalog2[step][name]=np.zeros(0,dtype=catalog2[step][name].dtype)
                if(verbose):
                    print "step: ", step, "vars: ", self.step_data[step].keys()
        if(verbose):
            print "\n\n"
            for step in self.step_data.keys():
                print "step: ",step,"vars: ", self.step_data[step].keys()

    #only one way joining catalog1 -> catalog2. Catalog2 does not change.
    def quick_join(self,catalog1,catalog2,join_on,one_to_random=False,remove_matched_1=True,sorter2=None,req_also=None,verbose=False):
        for step in catalog1.step_data.keys():
            self.step_data[step] = {}
            if(sorter2==None):
                if(verbose):
                    print "sorting catalog2"
                srt2 = np.argsort(catalog2[step][join_on])
            else:
                srt2 = sorter2
            if(verbose):
                print "searching A"
            pos_start = np.searchsorted(catalog2[step][join_on],catalog1[step][join_on],sorter=srt2,side='left')
            if(verbose):
                print "searching B"
            pos_end   = np.searchsorted(catalog2[step][join_on],catalog1[step][join_on],sorter=srt2,side='right')
            pos_len = pos_end-pos_start
            if(one_to_random):
                pos_start = pos_start + (pos_len*np.random.random(size=pos_len.size)).astype(int)

            matched = pos_start != pos_end  
            pos_start = srt2[pos_start.clip(max=srt2.size-1)]
            pos_end = srt2[pos_end.clip(max=srt2.size-1)]
            if(req_also):
                if(verbose):
                    print "req_also: ",req_also
                matched2= catalog1[step][req_also]==catalog2[step][req_also][pos_start]
                matched = matched & matched2
            unmatched = np.logical_not(matched)
            matched_indx = pos_start[matched]
            check = catalog1[step][join_on][matched] == catalog2[step][join_on][matched_indx]
            if(verbose):
                print "check: ",np.sum(check),'/',check.size
                print "matched: ",np.sum(matched)
                print "unmatched: ",np.sum(unmatched)
            names_this_step = []
            for name in catalog1.var_names:
                self.step_data[step][name] = catalog1[step][name][matched]
                if(name not in names_this_step):
                    names_this_step.append(name)
                if(remove_matched_1):
                    if(np.sum(unmatched)!=0):
                        catalog1[step][name] = catalog1[step][name][unmatched]
                    else:
                        catalog1[step][name] = np.zeros((0),dtype=catalog1[step][name].dtype)
            for name in catalog2.var_names:
                if(name not in names_this_step):
                    self.step_data[step][name] = catalog2[step][name][matched_indx]
                    names_this_step.append(name)
            self.var_names = names_this_step

    def apply_function(self,var_name,function,*args):
        for step in self.step_data.keys():
            self.step_data[step][var_name] = function(self.step_data[step][var_name],*args)

    def sort(self,sort_by):
        self.srt = {}
        self.srt_var = sort_by
        for step in self.step_data.keys():
            self.srt[step] = np.argsort(self.step_data[step][sort_by])
        
    def find(self,step,val):
        if(self.srt==None):
            print "Not sorted yet"
            raise Exception('Not Sorted yet!')
        srt_indx = np.searchsorted(self.step_data[step][self.srt_var],val,sorter=self.srt[step])
        if(srt_indx >=0 and srt_indx < self.srt[step].size):
            indx = self.srt[step][srt_indx]
            if(self.step_data[step][self.srt_var][indx] == val):
                return indx
            else:
                return -1
        else:
            return -1
    
    def find_all(self,step,val):
        if(self.srt==None):
            print "Not sorted yet"
            raise Exception('Not Sorted yet!')
        srt_indx_start = np.searchsorted(self.step_data[step][self.srt_var],val,sorter=self.srt[step],side='left')
        srt_indx_end = np.searchsorted(self.step_data[step][self.srt_var],val,sorter=self.srt[step],side='right')-1
        if(np.isfinite(srt_indx_start) and np.isfinite(srt_indx_end)):
            val1 = self.step_data[step][self.srt_var][self.srt[step][srt_indx_start]]
            val2 = self.step_data[step][self.srt_var][self.srt[step][srt_indx_end]]
            if(val1 == val and val2 == val):
                result = []
                for srt_indx in range(srt_indx_start,srt_indx_end+1,1):
                    result.append(self.srt[step][srt_indx])
                return np.array(result)
            else:
                return np.atleast_1d(np.array((),dtype=int))
        return  np.atleast_1d(np.array((),dtype=int))

    def cut_box(self,step,x0,y0,z0,x_lim,y_lim,z_lim,x_wrap,y_wrap,z_wrap,var_x='x',var_y='y',var_z='z',ignore_periodic=False):
        slct_x = self.cut_dim(step,x0,x_lim,x_wrap,var_x=var_x,ignore_periodic=ignore_periodic)
        slct_y = self.cut_dim(step,y0,y_lim,y_wrap,var_x=var_y,ignore_periodic=ignore_periodic)
        slct_z = self.cut_dim(step,z0,z_lim,z_wrap,var_x=var_z,ignore_periodic=ignore_periodic)
        slct= slct_x & slct_y & slct_z
        return slct

    def cut_dim(self,step,x0,x_lim,x_wrap,var_x='x',ignore_periodic=False):
        if(ignore_periodic):
            x_shifted = self.step_data[step][var_x]
        else:
            x_shifted=self.shift_x(x0,self.step_data[step][var_x],x_wrap)
        dist = x_shifted - x0
        slct = (dist<x_lim) & (dist>-x_lim)
        return slct

    def shift_x(self,x0,x,x_wrap):
        x1 = np.copy(x)
        x_pos = x1+x_wrap
        x_neg = x1-x_wrap
        dist = x-x0
        result = np.where(dist>x_wrap/2.0,x_neg,x1)
        result2 = np.where(dist<-x_wrap/2.0,x_pos,result)
        return result2
        
    def make_kdtree(self,var_x='x',var_y='y',var_z='z'):
        self.kd_data = {}
        self.kdtree = {}
        for step in self.step_data.keys():
            self.kd_data[step]  = np.zeros((self.step_data[step][var_x].size,3))
            self.kd_data[step][:,0] = self.step_data[step][var_x]
            self.kd_data[step][:,1] = self.step_data[step][var_y]
            self.kd_data[step][:,2] = self.step_data[step][var_z]
            self.kdtree[step]       = sp.spatial.KDTree(self.kd_data[step])

    def cut_box_kdtree(self,step,x0,y0,z0,r0):
        #the box diagonal is a factor sqrt(3) longer than the length
        indxs = self.kdtree[step].query_ball_point((x0,y0,z0),r0*np.sqrt(3))
        #print "cut box indxs: ",indxs
        x = self.kd_data[step][indxs,0]
        y = self.kd_data[step][indxs,1]
        z = self.kd_data[step][indxs,2]
        r = self.step_data[step]['radius'][indxs]
        slct_x = ((x-x0)<r0) & ((x-x0)>-r0)
        slct_y = ((y-y0)<r0) & ((y-y0)>-r0)
        slct_z = ((z-z0)<r0) & ((z-z0)>-r0)
        slct = slct_x & slct_y & slct_z
        return x[slct],y[slct],z[slct],r[slct]

def frag_to_real(htags):
    return np.abs(htags) & 0x0000ffffffffffff
