 

#include "gio.hpp"
#include <GenericIO.h>
#include <iostream>
void read_gio_float (char* file_name, char* var_name, float*   data, int only_rank){
  // std::cout<<"Reading in float data..."<<only_rank<<std::endl;
  read_gio<float>(file_name,var_name,data,only_rank);
}
void read_gio_double(char* file_name, char* var_name, double*  data, int only_rank){
  read_gio<double>(file_name,var_name,data,only_rank);
}
void read_gio_int8  (char* file_name, char* var_name, int8_t*  data, int only_rank){
   read_gio<int8_t>(file_name,var_name,data,only_rank);
}
void read_gio_int32 (char* file_name, char* var_name, int*     data, int only_rank){
  read_gio<int>(file_name,var_name,data,only_rank);
}
void read_gio_int64 (char* file_name, char* var_name, int64_t* data, int only_rank){
  read_gio<int64_t>(file_name,var_name,data,only_rank);
}
void read_gio_uint8 (char* file_name, char* var_name, uint8_t* data, int only_rank){
  std::cout<<"we are trying to read"<<std::endl;
   read_gio<uint8_t>(file_name,var_name,data,only_rank);
}
  
int64_t get_elem_num(char* file_name){
  // std::cout<<"Getting elem num"<<std::endl;
  gio::GenericIO reader(file_name);
  reader.openAndReadHeader(gio::GenericIO::MismatchAllowed);
  int num_ranks = reader.readNRanks();
  uint64_t size = 0;
  // std::cout<<"num_ranks"<<num_ranks<<std::endl;
  for(int i =0;i<num_ranks;++i){
    int num = reader.readNumElems(i);
    // std::cout<<"reader.readNumElems(i): "<<num<<std::endl;
    size +=reader.readNumElems(i);
  }
  reader.close();
  //std::cout<<size<<std::endl;
  return size;
}

var_type get_variable_type(char* file_name,char* var_name){
  // std::cout<<"Getting variable type"<<std::endl;
  gio::GenericIO reader(file_name);
  std::vector<gio::GenericIO::VariableInfo> VI;
  reader.openAndReadHeader(gio::GenericIO::MismatchAllowed);
  reader.getVariableInfo(VI);

  int num =VI.size();
  for(int i =0;i<num;++i){
    gio::GenericIO::VariableInfo vinfo = VI[i];
    if(vinfo.Name == var_name){
      if(vinfo.IsFloat && vinfo.Size == 4) //float
	return float_type;
      else if(vinfo.IsFloat && vinfo.Size == 8) //double
	return double_type;
      else if(!vinfo.IsFloat && vinfo.IsSigned && vinfo.Size == 4 ) //int32
	return int32_type;
      else if(!vinfo.IsFloat && vinfo.IsSigned && vinfo.Size == 8) //int64
	return int64_type;
      else if(!vinfo.IsFloat && vinfo.IsSigned && vinfo.Size == 1) //int8
	return int8_type;
      else if(!vinfo.IsFloat && !vinfo.IsSigned && vinfo.Size == 1) //uint8
	return uint8_type;
      else
	return type_not_found;
    }
  }
  return var_not_found;
      
}

extern "C" void inspect_gio(char* file_name){
  int64_t size = get_elem_num(file_name);
  gio::GenericIO reader(file_name);
  std::vector<gio::GenericIO::VariableInfo> VI;
  reader.openAndReadHeader(gio::GenericIO::MismatchAllowed);
  reader.getVariableInfo(VI);
  std::cout<<"Number of Elements: "<<size<<std::endl;
  int num =VI.size();
  std::cout<<"[data type] Variable name"<<std::endl;
  std::cout<<"---------------------------------------------"<<std::endl;
  for(int i =0;i<num;++i){
    gio::GenericIO::VariableInfo vinfo = VI[i];

    if(vinfo.IsFloat)
      std::cout<<"[f";
    else
      std::cout<<"[i";
    std::cout<<" "<<vinfo.Size*8<<"] ";
    std::cout<<vinfo.Name<<std::endl;
  }
  std::cout<<"\n(i=integer,f=floating point, number bits size)"<<std::endl;
}

