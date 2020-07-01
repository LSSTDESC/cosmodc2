
#include <iostream>
#include <ostream>
#include <fstream>
#include "util.hpp"
#include "param.hpp"
#include "timer.hpp"
#include "sort.hpp"
#include "power_spectrum.hpp"
#include "power_spectrum.hpp"

#include <unistd.h>
#include <fftw3.h>

#include <cmath>  
std::string bool2str(bool val){
  if(val)
    return "Success";
  else
    return "Failure";
  
}
bool test_binary_read_write();


bool test_sorter();
bool test_sorted_index();
bool test_subsorter();
bool test_param();
void test_cparam();
void test_timer();
bool test_multisort();
void test_power_spectrum();
void test_correlation();
int main(int argc, char** argv){
  bool all_success = true;
  bool result;
  std::cout<<"Testing of Dan's Tool Kit"<<std::endl;
  std::cout<<"Testing binary read and write: ";
  result = test_binary_read_write();
  std::cout<<bool2str(result)<<std::endl;;
  all_success &= result;
  std::cout<<"Testing Param: ";
  result = test_param();
  std::cout<<bool2str(result)<<std::endl;
  all_success &= result;
  if(all_success)
    std::cout<<"All automatic tests worked :)"<<std::endl;
  test_cparam();
  test_timer();
  test_power_spectrum();  
  test_correlation();  
  //  test_multisort();
}
bool test_multisort(){
  int size = 200;
  float* data = new float[size];
  double* data2 = new double[size];
  int*    data3 = new int[size];
  std::cout<<"MultiSort, initial data:"<<std::endl;
  for(int i =0;i<size;++i){
    data[i] = (float) rand()/(float) RAND_MAX;
    data2[i] = data[i]*10;
    data3[i] = data2[i]*10;
    std::cout<<data[i]<<"  \t"<<data2[i]<<"  \t"<<data3[i]<<std::endl;
  }
  dtk::MultiSort<float> msort;
  msort.add_sort_data(data,size);
  msort.add_data(data2);
  msort.add_data(data3);
  msort.sort();
  std::cout<<"MultiSort, final data:"<<std::endl;
  for(int i =0;i<size;++i){
    std::cout<<data[i]<<"  \t"<<data2[i]<<"  \t"<<data3[i]<<std::endl;
  }
  return true;
}
void test_timer(){
  dtk::Timer a;
  a.start();
  usleep(300000);
  a.stop();
  std::cout<<"Timer: "<<a.get_seconds()
	   <<"\t"<<     a.get_mseconds()
	   <<"\t"<<     a.get_useconds()
	   <<std::endl;
}
bool test_param(){
  dtk::Param param("test.param");
  std::vector<std::string> all_param = param.get_parameters();
  std::cout<<"all param: "<<std::endl;
  for(uint i =0;i<all_param.size();++i){
    std::cout<<i<<": \""<<all_param[i]<<"\"--";
    std::vector<std::string> val = param.get_vector<std::string>(all_param[i]);

    for(uint j=0;j<val.size();++j){
     std::cout<<" +"<<val[j];
    }
    std::cout<<std::endl;
  }
  std::vector<std::string> unac_param = param.get_unaccessed_parameters();
  std::cout<<"unac param: "<<std::endl;
  for(uint i =0;i<unac_param.size();++i){
    std::cout<<i<<": \""<<unac_param[i]<<"\""<<std::endl;
  }
  std::vector<std::string> ac_param = param.get_accessed_parameters();
  std::cout<<"ac param: "<<std::endl;
  for(uint i =0;i<ac_param.size();++i){
    std::cout<<i<<": \""<<ac_param[i]<<"\""<<std::endl;
  }
  int   foo = param.get<int>("a");
  float bar = param.get<float>("c");
  std::cout<<"a: "<<foo<<" c: "<<bar<<std::endl;
  std::vector<double> list = param.get_vector<double>("list");

  int list_size = param.get_length("list");
  double* list2 = new double[list_size];
  param.get_array<double>("list", list2);

  bool result = true;
  result &= (param.get<int>("a") == 1);
  result &= (param.get<double>("c") == 1.41);
  std::cout<<"initial: "<<result<<std::endl;
  result &= (list[0] == 1.0  && list[1] == 2.5 && list[2] == 3.4);
  std::cout<<list[0]<<" "<<list[1]<<" "<<list[3]<<std::endl;
  result &= (list2[0] == 1.0  && list2[1] == 2.5 && list2[2] == 3.4);
  result &= (list_size =3);
  std::cout<<list2[0]<<" "<<list2[1]<<" "<<list2[3]<<"  size: "<<list_size<<std::endl;
  result &= (param.get<bool>("t"));
  result &= (!param.get<bool>("f"));

  std::cout<<"t: "<<param.get<bool>("t")<<"  f: "<<param.get<bool>("f")<<std::endl;
  unac_param = param.get_unaccessed_parameters();
  std::cout<<"unac param: "<<std::endl;
  for(uint i =0;i<unac_param.size();++i){
    std::cout<<i<<": \""<<unac_param[i]<<"\""<<std::endl;
  }
  ac_param = param.get_accessed_parameters();
  std::cout<<"ac param: "<<std::endl;
  for(uint i =0;i<ac_param.size();++i){
    std::cout<<i<<": \""<<ac_param[i]<<"\""<<std::endl;
  }
  
  return result;
}
void test_cparam(){
  dtk::Param param("test.param");
  dtk::CosmoParam cparam(param.get<std::string>("indat.params"));
  std::cout<<"Do these values make sense for the parameters given?"<<std::endl;
  std::cout<<"rho crit: "  <<cparam.get_rho_crit()<<std::endl;
  std::cout<<"prtcl mass: "<<cparam.get_particle_mass()<<std::endl;
  std::cout<<"step 0: z="  <<cparam.get_z(0.0)<<std::endl;
  std::cout<<"step 100: z="<<cparam.get_z(100.0)<<std::endl;
  std::cout<<"step 200: z="<<cparam.get_z(200.0)<<std::endl;
  std::cout<<"step 300: z="<<cparam.get_z(300.0)<<std::endl;
  std::cout<<"step 400: z="<<cparam.get_z(400.0)<<std::endl;
  std::cout<<"step 499: z="<<cparam.get_z(499)<<std::endl;
  std::cout<<"z=0: step "  <<cparam.get_step(0)<<std::endl;
  std::cout<<"z=0.5: step "<<cparam.get_step(0.5)<<std::endl;
  std::cout<<"z=200: step "<<cparam.get_step(200.0)<<std::endl;
  return;
}
void example(){
  dtk::CosmoParam cparam("indat.params");
  
  float pm  = cparam.get_particle_mass();

  float rho = cparam.get_rho_crit(); // z=0
  rho       = cparam.get_rho_crit(1.0); // z = 1
  
  float z   = cparam.get_z(499); 
  int step  = cparam.get_step(0.5);
}

void example2(std::vector<float> data, float* data2, int size){
  dtk::write_binary<float>("a.bin",data);
  dtk::write_binary<float>("b.bin",data2,size);

  std::vector<float> read_data;
  dtk::read_binary<float>("a.bin",read_data);

  float* read_data2;
  int size2;
  dtk::read_binary<float>("b.bin",read_data2,size2);
  delete [] read_data2;
}

void example3(std::string file_name,std::string var_name,
	      float* data, int64_t& size){
  //  read_gio_quick(file_name,var_name,data,size); 
}

template <class T>
bool test_binary_read_write(std::vector<T>& vec, T* array){
  bool result = true;
  dtk::write_binary<T>("a.bin",vec);
  std::vector<T> vec2;
  dtk::read_binary<T>("a.bin",vec2);
  int a =10;
  dtk::read_binary<T>("a.bin",array,a);
		      
  if(vec.size() != vec2.size()){
    std::cout<<"Read and write vector size do not match: "<<vec.size()<<"!= "<<vec2.size()<<std::endl;
    result = false;
  }

  for(uint i =0;i<vec.size();++i){
    if(vec[i] != vec2[i]){
      std::cout<<"vector values do not match up: "<<vec[i]<<" "<<vec2[i]<<std::endl;
      result = false;
    }
    if(vec[i] != array[i]){
      std::cout<<"vector/array values do not match up: "<<vec[i]<<" "<<array[i]<<std::endl;
      result = false;
    }
    else{
    }

  }
  return result;
}
bool test_binary_read_write(){
  int vec_a[6] = {1,2,3,45,343,2342};
  int* vec_b = new int[6];
  std::vector<int> vec_i (&vec_a[0],&vec_a[0]+6);

  int64_t  vec_a64[7] = {2,4,12,513432,3422,3532,3521};
  int64_t* vec_b64 = new int64_t[7];
  std::vector<int64_t> vec_i64(&vec_a64[0],&vec_a64[0]+7);
  
  float vec_af[7] = {1.20,213.0,1241231,.00023,1232,64346.43463};
  float* vec_bf = new float[7];
  std::vector<float> vec_f (&vec_af[0],&vec_af[0]+7);

  double vec_ad[9] = {.0656,2365.254,214,5447,.005454,546.04547,99978,4454};
  double* vec_bd = new double[9];
  std::vector<double> vec_d (&vec_ad[0],&vec_ad[0]+9);
  bool result = true;
  result &= test_binary_read_write<int>(vec_i,vec_b);
  result &= test_binary_read_write<int64_t>(vec_i64,vec_b64);
  result &= test_binary_read_write<float>(vec_f,vec_bf);
  result &= test_binary_read_write<double>(vec_d,vec_bd);

  return result;
}
void test_power_spectrum(){
  dtk::Param param("ps.param");
  int n = param.get<int>("n_size");
  float domain = param.get<float>("domain");
  float period = param.get<float>("period");
  fftwf_complex* grid =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
  for(int i =0;i<n;++i){
    for(int j=0;j<n;++j){
      for(int k=0;k<n;++k){
	int index = i*n*n+j*n+k;
	double value = std::cos((double)i/(double) n*domain*2.0*M_PI/period);
	grid[index][0] = value;
      }
    }
    std::cout<<(double)i/(double)n*domain<<"->"<<grid[i*n*n][0]<<" "<<grid[i*n*n][1]<<std::endl;
  }

  dtk::PowerSpectrum ps = dtk::calc_power_spectrum(grid,n,domain);
  
  std::ofstream myfile;
  myfile.open ("ps_data.txt");
  std::ofstream myfile2;
  myfile2.open("array_data.txt");
  for(int i =0;i<ps.size;++i){
    std::cout<<i<<": \t"<<ps.k[i]<<"\t"<<ps.ps[i]<<std::endl;
    myfile<<i<<"\t"<<ps.k[i]<<"\t"<<ps.ps[i]<<std::endl;
  }
  for(int i =0;i<n;++i){
    myfile2<<(double)i/(double) n*domain<<"\t"<<grid[i*n][0]<<std::endl;
  }
  myfile2.close();
  myfile.close();
  delete [] grid;
}

void test_correlation(){
  dtk::Param param("ps.param");
  int n = param.get<int>("n_size");
  float domain = param.get<float>("domain");
  float period = param.get<float>("period");
  fftwf_complex* grid =  (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * n*n*n);
  for(int i =0;i<n;++i){
    for(int j=0;j<n;++j){
      for(int k=0;k<n;++k){
	int index = i*n*n+j*n+k;
	double value = std::cos((double)i/(double) n*domain*2.0*M_PI/period);
	grid[index][0] = 0;//value;
      }
    }
        std::cout<<(double)i/(double)n*domain<<"->"<<grid[i*n*n][0]<<" "<<grid[i*n*n][1]<<std::endl;
  }
  //grid[0][0]=1;
  //grid[10][0]=1;
   std::cout<<"calcing corr..."<<std::endl;
  dtk::Correlation corr = dtk::calc_correlation(grid,n,domain);
  std::cout<<"done"<<std::endl;
  std::ofstream myfile;
  myfile.open ("corr_data.txt");
  std::ofstream myfile2;
  myfile2.open("array_data2.txt");

  for(int i =0;i<corr.size;++i){
    std::cout<<i<<": \t"<<corr.r[i]<<"\t"<<corr.corr[i]<<std::endl;
    myfile<<i<<"\t"<<corr.r[i]<<"\t"<<corr.corr[i]<<std::endl;
  }
  for(int i =0;i<n;++i){
    myfile2<<(double)i/(double) n*domain<<"\t"<<grid[i*n][0]<<std::endl;
  }
  myfile2.close();
  myfile.close();
  delete [] grid;
}
bool test_subsorter(){
  int size = 10;
  int* data1 = new int[size];
  int* data2 = new int[size];
  int* srt   = dtk::new_sort_array<int>(size);
  for(int i =0;i<size;++i){
    data1[i]=0;
    data2[i]=size-i;
  }
  

}
