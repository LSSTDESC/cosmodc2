#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <math.h>

#include "dtk/param.hpp"


namespace dtk{
  std::vector<std::string> split(const char *str){
    std::vector<std::string> result;
    do{
      const char *begin = str;
      while(*str != ' ' && *str != '\t' && *str !='\n' && *str != '\0' && *str !='\r')
	str++;   
      if(begin!=str){
	result.push_back(std::string(begin, str));
      }
      //str++;
    } while (0 != *str++);
    return result;
  }


  void Param::load(std::string file){
    load(file.c_str());
  }
  void Param::load(const char* file){
    parameters.clear();
    accessed_parameters.clear();
    append(file);
  }
  void Param::append(std::string file){
    append(file.c_str());
  }
  void Param::append(const char* file){
    file_name = file;
    std::ifstream param_file;
    param_file.open(file);
    if(param_file.is_open()){
      std::stringstream ss;
      ss<< param_file.rdbuf();
      parse_string(ss.str());
    }
    else{
      std::cout<<"File not found: "<<file<<std::endl;
      throw;
    }
  }
  void Param::parse_string(std::string params){
    std::stringstream ss(params);
     std::string line,name;
      std::vector<std::string> line2;
      while(std::getline(ss,line)){
	//split the line into space space delimited elements
	line2 = split(line.c_str());
	//ignore empty lines
	if(line2.size() == 0)
	  continue;
	// check if it's a comment line 
	else if (line2[0][0] == '#')
	  continue;
	// or too short to contain a value
	/*	else if(line2.size() == 1) 
		throw ParameterHasNoValue(line2[0],file_name);*/
	else{
	  name = line2[0];
	  line2.erase(line2.begin());
	  if(parameters.find(name) == parameters.end()){
	    std::vector<std::string> line2_clean;
	    for(uint i =0;i<line2.size();++i){
	      if(line2[i] != "")
		line2_clean.push_back(line2[i]);
	    }
	    parameters.insert(std::make_pair(name,line2_clean)); 
	  }
	  else
	    throw DuplicateParameter(name,file_name);
	}
      }
  }
  std::string Param::stringify(){
    std::stringstream ss;
    for( std::map<std::string,std::vector<std::string> >::iterator it = parameters.begin();
	 it != parameters.end();
	 ++it){
      std::string key = it->first;
      std::vector<std::string> values = it->second;
      ss<<key;
      for(uint i =0;i<values.size();i++)
	ss<<"\t"<<values[i];
      ss<<"\n"<<std::endl;
    }
    return ss.str();
    
  }
  bool Param::has(std::string name){
    return parameters.find(name) != parameters.end();
  }
  int Param::get_length(std::string name){
    std::vector<std::string> result = parameters[name];
    if(result.size() == 0){
      std::cout<<"Variable \""<<name<<"\" not found"<<std::endl;
      throw 1;
    }
    else{
      return result.size();
    }
  }

  std::vector<std::string> Param::get_parameters(){
    std::vector<std::string> result;
    for( std::map<std::string,std::vector<std::string> >::iterator it = parameters.begin();
	 it != parameters.end();
	 ++it){
      std::string key = it->first;
      result.push_back(key);
    }
    return result;
  }
  std::vector<std::string> Param::get_accessed_parameters(){

    std::vector<std::string> result;
    for(std::set<std::string>::iterator it = accessed_parameters.begin();
	it != accessed_parameters.end();
	++it){
      result.push_back(*it);
    }
    return result;
  }

  std::vector<std::string> Param::get_unaccessed_parameters(){
    std::vector<std::string> unac_param;
    std::vector<std::string> all_params = get_parameters();
    for(std::vector<std::string>::iterator it = all_params.begin();
	it != all_params.end();
	++it){
      if(accessed_parameters.find(*it) == accessed_parameters.end())
	unac_param.push_back(*it);
    }
    return unac_param;
  }
  Param::Param(std::string file){
    load(file.c_str());
  }
  Param::Param(const char* file){
    load(file);
  }

  Param::Param(){}
  std::string Param::get_file_name(){
    return file_name;
  }
  template<>
  bool Param::convert<bool>(std::string s){
    if(s == "true" || s == "True" || s == "TRUE" || s == "T" || s == "t" || s=="1")
      return true;
    else if( s == "false" || s == "False" || s == "FALSE" || s == "F" || s == "f" || s=="0")
      return false;
    else{
      throw ValueDoesNotConvertSimple<bool>(s);
    }
  
  }

  void Param::access_parameter(std::string name){
    accessed_parameters.insert(name);
  }
  
  CosmoParam::CosmoParam(const char* file){
    load(file);
  }
  CosmoParam::CosmoParam(std::string file){
    load(file);
  }
  void CosmoParam::load(std::string file){
    load(file.c_str());
  }
  void CosmoParam::load(const char* file){
    Param::load(file); //load the parameters as before
    //Simulation steps & z-shift
    double z_in = get<double>("Z_IN");
    double z_fin= get<double>("Z_FIN");
    this->num_steps=get<float>("N_STEPS");
    this->a_in = 1./(1.+z_in);
    this->a_out= 1./(1.+z_fin);
    this->a_delta=(a_out-a_in)/(num_steps-1);
    
    //Rho critical at z= 0
    float h = get<float>("HUBBLE"); 
    float Hubble = h*100; // km/s /Mpc
    double G = 6.67408e-11*1e-9; //Grav const // km^3 kg^-1 s^-2
    double kg2msun = 5.02749e-31;  // Msun/kg
    double km2mpc  = 3.24078e-20;  // Mpc/km
    double rho_crit = 3*Hubble*Hubble/(8*M_PI*G);
    rho_crit =  kg2msun/km2mpc*rho_crit;
    this->rho_crit_0 = rho_crit/h/h;
    
    // Particle Mass
    float RL = get<float>("RL");
    float NP = get<float>("NP");
    float Omega_DM = get<float>("OMEGA_CDM");  // CDM density
    float Omega_BM = get<float>("DEUT")/(h*h); // baryon density
    this->Omega_M  = Omega_DM+Omega_BM; // matter denisty
    
    this->prtcl_mass = RL*RL*RL/NP/NP/NP*Omega_M*rho_crit_0;
    return;
  }
  float CosmoParam::get_rho_crit(double z){
    float b = z + 1.;
    return this->rho_crit_0*((b*b*b*this->Omega_M+(1-this->Omega_M))/(b*b*b));
  }
  float  CosmoParam::get_step(double z){
    //there will be a small round off error at step =0
    double a = 1./(1.+z);
    return (a - this->a_in)/this->a_delta;
  }
  float CosmoParam::get_z(double step){
    //the strange way of writing this function is to get rid of an 
    // annoying rounding errors at z=0
    // which give some z=1e-16 instead of 0.0
    return 1./(this->a_out - ((double)this->num_steps-step-1)*this->a_delta) -1.;
  }
  float CosmoParam::get_z(int step){
    return get_z(static_cast<double>(step));
  }
  float CosmoParam::get_particle_mass(){
    return this->prtcl_mass;
  }
}
