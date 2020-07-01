#ifndef UTIL_HPP_
#define UTIL_HPP_
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdarg>
#include "H5Cpp.h"

namespace dtk{
  template <class T,class U>
  void read_binary(std::string file_name, T*& data, U& size);
  template <class T,class U> 
  void read_binary(const char* file_name,T*& data, U& size);


  template <class T>
  void write_binary(std::string file_name,std::vector<T>& data){
    std::ofstream file(file_name.c_str(), std::ios::out | std::ofstream::binary);
    file.write((const char*)&data[0],data.size()*sizeof(T));
    file.close();
  }

  template <class T>
  void write_binary(std::string file_name,T* data,int size){
    std::ofstream file(file_name.c_str(), std::ios::out | std::ofstream::binary);
    file.write((const char*)&data[0],size*sizeof(T));
    file.close();
  }
  template <class T,class U>
  void read_binary(std::string file_name, T*& data, U& size){
    read_binary(file_name.c_str(),data,size);
  }

  template <class T,class U> 
  void read_binary(const char* file_name,T*& data, U& size){
    std::ifstream file(file_name,std::ios::binary|std::ios::in|std::ios::ate);
    if(file.fail()){
      std::stringstream ss;
      ss<<"Can't open file "<<file_name;
      throw std::runtime_error(ss.str().c_str()); 
    }
    size_t size_bytes = file.tellg();
    std::cout<<file_name<<" size_bytes: "<<size_bytes<<std::endl;
    size = size_bytes/sizeof(T);
    data = new T[size];
    file.seekg(0,std::ios::beg);
    file.read((char*)data,size_bytes);
    file.close();
    
  }
  template <class T>
  void read_binary(const char* file_name, std::vector<T>& out){
    T* data;
    int size;
    read_binary(file_name,data,size);
    out.resize(size);
    out.assign(data,data+size);
    delete data;
  }
  template <class T>
  void read_binary(std::string file_name,std::vector<T>& out){
    read_bin(file_name.c_str(),out);
  }


  /*  template <class T,class U>
  void read_hdf5(std::string file_name,std::string var_name,T*& data, U& size){
    H5::H5File file(file_name,H5F_ACC_RDONLY);
    H5::Dataset data = file.openDataSet(var_name);
    H5::DataSpace space = data.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    if(rank > 1){
      std::cout<<"Data isn't a column of data but some matrix"<<std::endl;
    }
    
    }*/

  template <class T>
  std::vector<std::string> replace_strings(std::vector<std::string > file_base, std::string target, T value){
    std::vector<std::string> result;
    std::stringstream ss;
    for(uint i=0;i<file_base.size();++i){
      if(file_base[i]==target){
	ss.str("");
	ss<<value;
	result.push_back(ss.str());
      }
      else
	result.push_back(file_base[i]);
    }
    return result;
  }
  std::string cat_strings(std::vector<std::string> strings);

  template <class T>
  std::string cat_replace_strings(std::vector<std::string> strings, std::string target, T new_val){
    return cat_strings(replace_strings(strings,target,new_val));
  }
  template <class T,class U>
  double fract(T num, U denum){
    double num_d = static_cast<double>(num);
    double denum_d = static_cast<double>(denum);
    return num_d/denum_d;
  }
  template <class T>
  std::string rep_str(std::string str, const std::string target, const T new_val){
    std::stringstream ss;
    ss<<new_val;
    std::string new_str = ss.str();
    if(target == new_str)
      return str;
    size_t target_size = target.size();
    size_t found = str.find(target);
    while(found != std::string::npos){
      str.replace(found,target_size,new_str);
      found = str.find(target);
    }
    return str;
  }

  
  // Makes the directory(tree) specified 
  // if it doesn't exist. This only works on
  // unix systems. 
  bool ensure_dir(std::string dir);

  // Makes the  directory (tree) for
  // the specified file if that directory (tree)
  // doesn't exist
  bool ensure_file_path(std::string file);
  
  void pause();
  bool ask_bool();
  bool ask_continue();
  template<typename T,typename U>
  std::string make_str(T d1,U d2){
    std::stringstream ss;
    ss<<d1<<d2;
    return ss.str();
  }
  template<typename T1,typename T2,typename T3>
  std::string make_str(T1 d1,T2 d2, T3 d3){
    std::stringstream ss;
    ss<<d1<<d2<<d3;
    return ss.str();
  }
  template<typename T1,typename T2,typename T3,typename T4>
  std::string make_str(T1 d1,T2 d2, T3 d3, T4 d4){
    std::stringstream ss;
    ss<<d1<<d2<<d3<<d4;
    return ss.str();
  }
  template<typename T1,typename T2,typename T3,typename T4,typename T5>
  std::string make_str(T1 d1,T2 d2, T3 d3, T4 d4,T5 d5){
    std::stringstream ss;
    ss<<d1<<d2<<d3<<d4<<d5;
    return ss.str();
  }
  template<typename T1, typename T2>
  float fraction(T1 d1, T2 d2){
    return static_cast<float>(d1)/static_cast<float>(d2);
  }
  template<typename T, typename T2>
  void copy_n(T* input, T2 num, T* output){
    std::copy(input, input+num, output);
  }
}  
#endif // UTIL_HPP_
 
