#ifndef DTK_PARAM_HPP
#define DTK_PARAM_HPP

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <stdint.h>
#include <sstream>
#include <typeinfo>
namespace dtk{
  // This class interprets a parameter file and extract values 
  // assigned to parameters. The parameter file is a simple ascii
  // file. The first word on a line is the parameter name and 
  // every other word afterward are the values of the parameter. 
  // Using the parameter name, the value defined in the file is 
  // exacted to the data type specified. Exceptions are thrown if
  // asked for parameter name is not found, or the specified value
  // can not be converted into the specified data type (e.g. "a" into 
  // a float, or "3.14" into an int).
  // Lines that start with a "#" is a considered to be comment 
  // and is ignored. Blank lines are ignored. 

  //Example  parameter file:
  /* beginning of file

    #this is a comment
              # also a comment

    # defined variables
    a 3
    b 5.2 
    c 5.2 6.0

    # param.get<int>("a") returns an int = 3
    # param.get<float>("a") returns a float = 3.0
    # param.get<int>("b") fails because it has a decimal point
    # param.get<float>("b") returns a float = 5.2
    # param.get<float>("c") fails fails because there is a list of values
    # param.get_vector<float>("c") returns a std::vector<float> = [5.2,6.0]
    # param.get_vector<float>("a") returns a std::vector<float> = [3.0]

    file_loc ~/projects/example.txt
    # param.get<std::string>("file_loc")

    pi 3.14
    strange a 3.523 dade  122
    # param.get_vector<std::string>("strange") is the only thing that will work
    # above is legal but would be only be read in as strings.

    bool_false  0 f F false False FALSE  
    bool_true   1 t T true  True  TRUE
    # all legal values for param.get_vector<bool>("bool_xxx")

    repeat 2
    repeat 3
    # repeating parameters will throw an error.

  */ //end of file
  class Param{
  public:
    
    // Initializes the parameters from the specified file. 
    Param(std::string file);
    Param( const char* file);

    // Initializes the class without a parameter file. A file can be 
    // loaded later with the load(file) function.
    Param();

    // Loads a new parameter file and removes any parameters
    // from any previous file.
    void load(const char* file);
    void load(std::string file);  


    // Loads a new parameter file but keeps parameters from the
    // previous. Note: Duplicate parameters are not allowed 
    // between the sperate files.
    void append(const char* file);
    void append(std::string file);

    // Converts the parameters stored into a single string that can
    // be shared to be read by other Param objects.
    std::string stringify();
    // parses the string so values can be queried
    void parse_string(std::string params);
    // Returns true if parameter is present 
    bool has(std::string name);
    
    // returns the number of values for a parameter. Zero is given
    // the parameter is not found in the file.
    int get_length(std::string name);

    // Returns the number of parameters stored
    int get_param_count(){return parameters.size();}

    // Returns a vector with all the names of the parameters
    // stored
    std::vector<std::string> get_parameters();
    
    // Returns all accessed parameters
    std::vector<std::string> get_accessed_parameters();

    // Returns all parameters that were not used
    std::vector<std::string> get_unaccessed_parameters();

    // Returns the value of a defined parameter, as the template
    // type. If the parameter is defined as a list, it will return
    // the first value, unless a different position specified. 
    // It throws a exception if the parameters
    // is not found in the file, or does not convert into tthe expected type.
    template<typename T>
    T get(std::string name){
      if(!has(name))
	throw ParameterNotFound(name,file_name);
      std::vector<std::string> result = parameters[name];
      if(result.size() == 0){
	throw ParameterHasNoValue(name,file_name);
      }
      else if(result.size() != 1){
	throw ValueIsNotSingular(name,file_name);
      }
      try{
	T ret_val = convert<T>(result[0]);
	access_parameter(name);
	return ret_val;
      }
      catch (ValueDoesNotConvertSimple<T> e) {
	throw ValueDoesNotConvert<T>(e.get_value(),name,file_name);
      }
    }
    // Loads a preallocated array with the values of the parameter specificed.
    // The array needs to preallocated using the size from get_len(name) function. 
    template<typename T>
    void get_array(std::string name, T* array){
      std::vector<std::string> result = parameters[name];
      if(result.size() == 0){
	throw ParameterNotFound(name,file_name);
      }
      else{

	for(unsigned int i =0;i<result.size();++i){
	  try{
	    array[i] = convert<T>(result[i]);
	    //std::cout<<"array["<<i<<"]: "<<array[i]<<std::endl;
	  }
	  catch (ValueDoesNotConvertSimple<T> e) {
	    throw ValueDoesNotConvert<T>(e.get_value(),name,file_name);
	  }
	}
	access_parameter(name);
      }
    }

    // Returns an vector with all the values of a parameter. It throws
    // the same exceptions as the get() function. 
    template<typename T>
    std::vector<T> get_vector(std::string name){
      std::vector<std::string> result = parameters[name];
      std::vector<T> output;
      if(result.size() == 0){
	throw ParameterNotFound(name,file_name);
      }
      else{
	output.reserve(result.size());
	for(unsigned int i =0;i<result.size();++i){
	  if(result[i].size()>0)
	    try{
	      output.push_back(convert<T>(result[i])); 
	    }
	    catch (ValueDoesNotConvertSimple<T> e) {
	      throw ValueDoesNotConvert<T>(e.get_value(),name,file_name);
	    }
	}
      }
      access_parameter(name);
      return output;
    }

    std::string get_file_name();

    // Exception when a parameter name is not found in a parameter 
    // file. 
    class ParameterNotFound : public std::exception{
      std::string param_name;
      std::string file_name;
      std::string what_str;
    public:
      ParameterNotFound(std::string param_name,std::string file_name): 
	param_name(param_name),file_name(file_name),what_str(""){
	what_str+="Parameter \""+param_name+"\" not found in file \""+file_name+"\".";
      }
      virtual ~ParameterNotFound() throw() {}
      virtual const char* what() const throw(){
	return what_str.c_str();
      }
    };
    // Exception when a parameter has a list of values but singular value
    // is call (i.e. get<T> instead of get_vector<T>/get_array<T>).
    class ValueIsNotSingular: public std::exception{
      std::string param_name;
      std::string file_name;
      std::string what_str;
    public:
      ValueIsNotSingular(std::string param_name,std::string file_name): 
	param_name(param_name),file_name(file_name),what_str(""){
	what_str+= "Parameter \""+param_name+"\"'s value is not signular value but a list of values) in file \""+file_name+"\".";
      }
      virtual ~ValueIsNotSingular() throw() {}
      virtual const char* what() const throw(){
	return what_str.c_str();
      }
    };
    // Exception when a value in a parameter cannot be converted to the specified 
    // type. (e.g. "a" into an float, "3.14" into a int)
    template<typename T>
    class ValueDoesNotConvert : public std::exception{

      std::string value;
      std::string param_name;
      std::string file_name;
      std::string what_str;
    public:
      ValueDoesNotConvert(std::string value, std::string param_name,std::string file_name): 
	value(value),param_name(param_name),file_name(file_name),what_str(""){
	what_str=" Parameter \""+param_name+"\"'s value \""+value+"\" does not convert to "+typeid(T).name()+", in file \""+file_name+"\".";
      }

      virtual ~ValueDoesNotConvert() throw() {}
      virtual const char* what() const throw(){
	return what_str.c_str();
      }
    };
    class DuplicateParameter : public std::exception{
      std::string param_name;
      std::string file_name;
      std::string what_str;
    public:
      DuplicateParameter(std::string param_name,std::string file_name):
	param_name(param_name),file_name(file_name),what_str(""){
	what_str+= "Parameter \""+param_name+"\" has been defined multiple times, in file \""+file_name+"\".";
      }
      virtual ~DuplicateParameter() throw(){}
      virtual const char* what() const throw(){
	return what_str.c_str();
      }
    };
    class ParameterHasNoValue : public std::exception{
      std::string param_name;
      std::string file_name;
      std::string what_str;
    public:
      ParameterHasNoValue(std::string param_name,std::string file_name):
	param_name(param_name),file_name(file_name),what_str(""){
	what_str+="Parameter \""+param_name+"\" does not have a value, in file \""+file_name+"\".";
      }
      virtual ~ParameterHasNoValue() throw(){}
      virtual const char* what() const throw(){
	return what_str.c_str();
      }
    };
  private:
    template<typename T>
    class ValueDoesNotConvertSimple: public std::exception{
      std::string value;
    public:
      ValueDoesNotConvertSimple(std::string value):value(value){}
      virtual ~ValueDoesNotConvertSimple() throw() {}
      virtual const char* what() const throw(){
	std::string s = "Value \""+value+"\" does not convert to "+typeid(T).name()+".";
	return s.c_str();
      }
      std::string get_value(){
	return value;
      }
      
    };
    std::map<std::string, std::vector<std::string> > parameters;
    std::set<std::string > accessed_parameters;
    std::stringstream ss;
    std::string file_name;
    template<typename T>
    T convert(std::string value){
      ss.str("");
      ss.clear();
      ss<<value;
      //      std::cout<<"ss: "<<ss.str()<<" ";
      T output;
      ss>>output;
      //      std::cout<<output<<std::endl;
      if(ss.fail() || !ss.eof()){
	throw ValueDoesNotConvertSimple<T>(value);
      }
      return output;
    }
    //keep track of what parameters have been called
    void access_parameter(std::string name);
    
    
  };
  //this specializated template must be declared outside the class :( 
  template<>
  bool Param::convert<bool>(std::string s);

  
  class CosmoParam: public Param{
  public:
    int num_steps;
    double a_in,a_delta,a_out;
    double rho_crit_0;
    double Omega_M;
    double prtcl_mass;
  
    CosmoParam(const char* file);
    CosmoParam(std::string file);
    void load(const char* file);
    void load(std::string file);
    float get_rho_crit(double z=0);
    float get_step(double z);
    float get_z(double step);
    float get_z(  int step);
    float get_particle_mass();
     
  };
  

} //end namespace dtk

#endif //end ifndef DTK_PARAM_HPP
