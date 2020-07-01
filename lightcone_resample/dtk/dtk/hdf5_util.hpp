

#ifndef DTK_HDF5_HPP
#define DTK_HDF5_HPP
#include "H5Cpp.h"
#include <iostream>
#include <vector>
namespace dtk{
  template <typename T> H5::PredType hdf5_type();
  template <> H5::PredType hdf5_type<float>();
  template <> H5::PredType hdf5_type<int>();
  template <> H5::PredType hdf5_type<double>();
  template <> H5::PredType hdf5_type<int64_t>();
  template <> H5::PredType hdf5_type<uint64_t>();
  template <> H5::PredType hdf5_type<unsigned int>();
  template <> H5::PredType hdf5_type<bool>();

  void hdf5_clear_file(std::string file_loc);
  void hdf5_ensure_variable_path(H5::H5File& file, std::string var_name);
  void hdf5_ensure_group(H5::Group& group, std::string path);
  size_t hdf5_num_elems(H5::H5File,std::string var_name);
  size_t hdf5_num_elems(std::string file_name,std::string var_name);

  template <class T>
  void read_hdf5(std::string file_name,std::string var_name,std::vector<T>& out){
    H5::H5File file(file_name,H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(var_name);
    H5::DataSpace ds = dataset.getSpace();
    hsize_t datasize = ds.getSimpleExtentNpoints();
    out.resize(datasize);
    dataset.read(&out[0],hdf5_type<T>(),ds);
  }
  template <class T>
  void read_hdf5(H5::H5File& file,std::string var_name,std::vector<T>& out){
    H5::DataSet dataset = file.openDataSet(var_name);
    H5::DataSpace ds = dataset.getSpace();
    hsize_t datasize = ds.getSimpleExtentNpoints(); 
    out.resize(datasize);
    dataset.read(&out[0],hdf5_type<T>(),ds);
  }
  template <class T,class U>
  void read_hdf5(std::string file_name,std::string var_name, T*& out, U& size,bool alloc=false){
    H5::H5File file(file_name,H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(var_name);
    H5::DataSpace ds = dataset.getSpace();
    if(alloc){
      hsize_t datasize = ds.getSimpleExtentNpoints(); 
      out = new T[datasize];
    }
    dataset.read(&out[0],hdf5_type<T>(),ds);

  }
  template <class T>
  void read_hdf5(std::string file_name,std::string var_name, T*& out, bool alloc=false){
    size_t size;
    read_hdf5(file_name,var_name,out,size,alloc);
  }
  template <class T,class U>
  void read_hdf5(H5::H5File& file, std::string var_name,T*& out, U& size, bool alloc=false){
    H5::DataSet dataset = file.openDataSet(var_name);
    H5::DataSpace ds = dataset.getSpace();
    if(alloc){
      hsize_t datasize = ds.getSimpleExtentNpoints(); 
      out = new T[datasize];
    }
    dataset.read(&out[0],hdf5_type<T>(),ds);

  }
  template <class T>
  void read_hdf5(H5::H5File& file,std::string var_name,T*& out,bool alloc=false){
    hsize_t size;
    read_hdf5(file,var_name,out,size,alloc);
  }
  template <class T>
  void read_hdf5(H5::H5File& file,std::string var_name,T* out){
    hsize_t size; //this function is to avoid passing a reference to a pointer 
    read_hdf5(file,var_name,out,size);//which doesn't work when you pass a temporary 
    //variable such as data+offset.
  }

  template <class T>
  void read_hdf5(H5::H5File& file, std::string var_name,T& out){
    T* data = &out;
    read_hdf5(file,var_name,data,false); //false -> no alloc for T* data: it's already point to the data
  }
  template <class T>
  void read_hdf5(std::string file_loc, std::string var_name,T& out){
    read_hdf5(file_loc,var_name,&out);
  }

  template <class T> 
  void write_hdf5(H5::H5File& file, std::string var_name, T* data,hsize_t size){
    //std::cout<<"writing hdf5 data to "<<var_name<<std::endl;
    hdf5_ensure_variable_path(file,var_name);
    H5::DataSpace dataspace(1,&size);
    H5::DataSet dataset = file.createDataSet(var_name,hdf5_type<T>(),dataspace);
    dataset.write(data,hdf5_type<T>());
  }
  template <class T>
  void write_hdf5(std::string file_loc, std::string var_name, T* data,hsize_t size,bool append=true){
    H5::H5File file;
    if(append)
      file.openFile(file_loc,H5F_ACC_RDWR);
    else
      file.openFile(file_loc,H5F_ACC_TRUNC);
    write_hdf5(file,var_name,data,size);
  }
  template <class T>
  void write_hdf5(std::string file_loc, std::string var_name, std::vector<T> data,bool append=true){
    write_hdf5(file_loc,var_name,&data[0],data.size(),append);
  }
  template <class T>
  void write_hdf5(H5::H5File& file, std::string var_name, std::vector<T> data){
    write_hdf5(file,var_name,&data[0],data.size());
  }
  template <class T>
  void write_hdf5(H5::H5File& file, std::string var_name, T data){
    //std::cout<<"writing single value hdf5 to "<<var_name<<std::endl;
    write_hdf5(file,var_name,&data,1);
  }
  template <class T>
    void write_hdf5(std::string file_loc, std::string var_name, T data,bool append=true){
    write_hdf5(file_loc,var_name,&data,1,append);
  }


}

#endif //DTK_HDF5_HPP
