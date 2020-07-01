#ifndef GIO_UTIL_HPP_
#define GIO_UTIL_HPP_

#include <vector>
#include "GenericIOMPIReader.h"
#include "GenericIOPosixReader.h"
#include "GenericIO.h"
#include "mpi.h"
namespace dtk{

  template <class T>
  void read_gio_quick(std::string file_name,std::string var_name, T*& data, int64_t& size){
#ifndef  GENERICIO_NO_MPI
    std::cout<<"mpi"<<std::endl;
    gio::GenericIO reader(MPI_COMM_WORLD, file_name);
#else
    std::cout<<"no mpi"<<std::endl;
    gio::GenericIO reader(file_name);
#endif
    std::cout<<"open and read"<<std::endl;
    reader.openAndReadHeader(gio::GenericIO::MismatchRedistribute);
    std::cout<<"size: ";
    size  = reader.readNumElems();
    std::cout<<size<<std::endl;
    
    int pad = reader.requestedExtraSpace()/sizeof(T);
    data = new T[size+pad];
    reader.addVariable(var_name,data,true);
    reader.readData();
    reader.close();
    /*    gio::GenericIOMPIReader reader;
    std::stringstream ss;
    reader.SetFileName(file_name);
    reader.SetCommunicator(MPI_COMM_WORLD);
    reader.SetBlockAssignmentStrategy(gio::RCB_BLOCK_ASSIGNMENT);
    reader.OpenAndReadHeader();
    size = reader.GetNumberOfElements();
    int pad = gio::CRCSize/sizeof(T);
    data = new T[size+pad];
    reader.AddVariable(var_name,data,gio::GenericIOBase::ValueHasExtraSpace);
    reader.ReadData();
    reader.Close();*/
    /*gio::GenericIOPosixReader reader;
    std::stringstream ss;
    reader.SetFileName(file_name);
    reader.SetBlockAssignmentStrategy(gio::RCB_BLOCK_ASSIGNMENT);
    reader.OpenAndReadHeader();
    size = reader.GetNumberOfElements();
    int pad = gio::CRCSize/sizeof(T);
    data = new T[size+pad];
    reader.AddVariable(var_name,data,gio::GenericIOBase::ValueHasExtraSpace);
    reader.ReadData();
    reader.Close();*/
    
  }
  template <class T>
  void read_gio_quick(std::string file_name,std::string var_name, std::vector<T>& data){
    /*#ifndef  GENERICIO_NO_MPI
    gio::GenericIO reader(MPI_COMM_WORLD, file_name);
#else
    gio::GenericIO reader(file_name);
#endif
    reader.openAndReadHeader();
    int64_t size  = reader.readNumElems();
    int pad = reader.requestedExtraSpace()/sizeof(T);
    data.resize(size+pad);
    reader.addVariable(var_name,data,true);
    reader.readData();
    data.resize(size);
    reader.close();
    */
    gio::GenericIOMPIReader reader;
    std::stringstream ss;
    T* data_array;
    reader.SetFileName(file_name);
    reader.SetCommunicator(MPI_COMM_WORLD);
    reader.SetBlockAssignmentStrategy(gio::RCB_BLOCK_ASSIGNMENT);
    reader.OpenAndReadHeader();
    int64_t size = reader.GetNumberOfElements();
    int pad = gio::CRCSize/sizeof(T);
    data_array = new T[size+pad];
    reader.AddVariable(var_name,data,gio::GenericIOBase::ValueHasExtraSpace);
    reader.ReadData();
    reader.Close();
    std::vector<T> new_data(data_array,data_array+size);
    data.swap(new_data);
    delete data_array;
  }
  


} 



#endif //GIO_UTIL_HPP_
