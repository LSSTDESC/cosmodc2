#ifndef CHAINING_MESH_HPP
#define CHAINING_MESH_HPP

#include <iostream>
#include <vector>
namespace dtk{
  class ChainingMeshIndex{
    float* length_xyz;
    size_t* length_ijk;
    size_t* cell_ijk_offset;
    float*  cell_length_xyz;
    int dim_num;
    size_t* element_index;
    size_t* element_cell;
    size_t  element_num;
    size_t* cell_start;
    size_t* cell_size;
    size_t  cell_num;
  public:
    ChainingMeshIndex();
    ChainingMeshIndex(float* len_xyz, size_t* len_ijk, int dim_num);
    ~ChainingMeshIndex();
    void clear();
    void set_grid(float* len_xyz, size_t* len_ijk, int dim_num);
    size_t get_num_cells(){return cell_num;}
    size_t get_maxium_cell_size();
    void place_onto_mesh(float** xyz, size_t size);
    size_t get_cell_id_from_position(float* xyz);
    size_t get_cell_id_from_indexes(size_t* ijk);
    void get_indexes_from_cell_id(size_t cell_id, size_t* output_ijk);
    void get_indexes_from_position(float* xyz, size_t* output_ijk);
    void get_cell_element_indexes(size_t cell_index, size_t*& cell_element_start, size_t& cell_element_size);

    template<typename T>
    size_t get_cell_id_from_indexes(T* ijk);
    template<typename T, typename R>
    void get_indexes_from_position_no_wrap(T* xyz, R* outpu_ijk);


    //Vector based returns
    std::vector<size_t> get_neighbor_cell_id_list(size_t cell_index, size_t length);
    std::vector<size_t> get_neighbor_cell_id_list(size_t* ijk, size_t length);
    std::vector<size_t> get_neighbor_cell_id_list(float* xyz, size_t length);
    std::vector<size_t> get_region_cell_id_list(size_t cell_index, size_t length);
    std::vector<size_t> get_region_cell_id_list(float* xyz, size_t length);
    std::vector<size_t> get_region_cell_id_list(float* xyz, float radius);
    //vector based inputs
    std::vector<size_t> get_indexs_from_cell_id(size_t cell_id);
  private:
    void assign_element_cell(float** xyz, size_t size);
    void group_cells();
    
  };
  template<typename T,typename R>
  void ChainingMeshIndex::get_indexes_from_position_no_wrap(T* xyz, R* output_ijk){
     for(int i =0;i<dim_num;++i){
      //convert any ijk value outside the bounds to be in the bounds
      //including negative values
      //std::cout<<ijk[i]<<"->"<<ijk[i]%int(length_ijk[i])<<std::endl;

       output_ijk[i] = static_cast<R>(xyz[i]/cell_length_xyz[i]);
    }
  }
  template<typename T>
  size_t ChainingMeshIndex::get_cell_id_from_indexes(T* ijk){
    size_t result;
    size_t wrapped_ijk[dim_num];
    for(int i =0;i<dim_num;++i){
      //convert any ijk value outside the bounds to be in the bounds
      //including negative values
      //std::cout<<ijk[i]<<"->"<<ijk[i]%int(length_ijk[i])<<std::endl;
      T len = static_cast<T>(length_ijk[i]);
      wrapped_ijk[i] = ((ijk[i]%len)+len)%len;
    }
    return get_cell_id_from_indexes(wrapped_ijk);
  }

}
#endif//CHAINING_MESH_HPP
