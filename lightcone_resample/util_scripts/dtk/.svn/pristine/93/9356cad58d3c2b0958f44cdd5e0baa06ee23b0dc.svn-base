

#include "dtk/cosmo.hpp"

namespace dtk{
  StepRedshift::StepRedshift(float z_start,float z_end,int num_steps):
    z_in(z_start),z_out(z_end),num_steps(num_steps){
    a_in = a_from_z(z_start);
    a_out = a_from_z(z_end);
    a_del = (a_out - a_in)/num_steps;
  }
  float StepRedshift::step_from_z(float z){
    float a = a_from_z(z);
    return (a-a_in)/a_del;
  }
  float StepRedshift::z_from_step(float step){
    if(step == num_steps-1)
      return z_out; //to avoid the roundoff in a_del*steps
    float a = a_del*step;
    return z_from_a(a);
  }


}
