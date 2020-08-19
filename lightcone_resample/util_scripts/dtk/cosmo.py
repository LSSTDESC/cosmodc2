import numpy as np


#converts simulations steps into z/a and
#vice versa
class StepZ:
    def __init__(self,start_z=None,end_z=None,num_steps=None,sim_name=None):
        if(sim_name != None):
            if sim_name == 'AlphaQ':
                start_z = 200
                end_z = 0
                num_steps = 500
            else:
                raise 
        self.z_in = float(start_z)
        self.z_out = float(end_z)
        self.num_steps = float(num_steps)
        self.a_in = 1./(1.+start_z)
        self.a_out = 1./(1.+end_z)
        self.a_del = (self.a_out- self.a_in)/(self.num_steps)
        self.offset = 1.0

    def get_z(self,step):
        #to get rid of annoying rounding errors on z=0 (or other ending values)
        #if(step == self.num_steps-1):
        #    return 1./self.a_out -1. 
        a = self.a_in+(step+self.offset)*self.a_del
        return 1./a-1.

    def get_step(self,z):
        a = 1./(z+1.)
        return (a-self.a_in)/self.a_del-1

    def get_a(self,step):
        return 1./(self.get_z(step+self.offset)+1.)

def z_from_a(a):
    return 1.0/a-1.0

def a_from_z(z):
    return 1.0/(z+1.0)

b=0.168
Hubble=0.71
Omega_DM = 0.22
Omega_BM = 0.02258/Hubble**2
Omega_M  = Omega_DM+Omega_BM
Omega_L = 1-Omega_M
prtcl_mass = 1491.**3/3200.**3*(Omega_M*2.77536627e11)

def get_rho_crit_z0():
    return 2.77536627e11 #(h^-1 Msun)/(h^-3 Mpc^3)

def get_rho_crit(z=None,a=None):
    if(z==None and a==None):
        raise Exception("rho_crit: z or a must be defined")
    elif(z!=None and a!=None):
        raise Exception("rho_crit: both z and a can't be defined")
    if(z!=None):
        a= a_from_z(z)
    r0 = get_rho_crit_z0()
    return r0*(Omega_M/a**3+Omega_L)
    
