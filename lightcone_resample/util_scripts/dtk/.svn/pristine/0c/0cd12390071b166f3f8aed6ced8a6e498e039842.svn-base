import numpy as np

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import param as prm

param = prm.Param("test.param")
print param.get_float("a")
print param.get_float_list("a")
print param.get_int64("b")

cparam = prm.CosmoParam("indat.params")

print "%E" % cparam.get_particle_mass()
print "%E" % cparam.get_rho_crit()
print cparam.get_z(499)
print cparam.get_z(1)
print cparam.get_step(0)
print cparam.get_step(1)

