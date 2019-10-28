#pragma once
#include "tensor.h"
#include <vector>
  
DLManagedTensor* CreateDLPackTensor(initializer_list<int64_t> shape_list) {
  DLManagedTensor* newTensor = new DLManagedTensor;
  std::vector<int64_t> shape_vec(shape_list);
  newTensor->dl_tensor->dim = shape_vec.size();
  newTensor->dl_tensor->shape = &shape_vec[0];
  newTensor->dl_tensor->strides = nullptr;
  newTensor->dl_tensor->byte_offset = 0;
  int64_t total_len = 1;
  for(int64_t dim_ : shape_list) {
    total_len *= dim;
  }
  if(shape_vec.size() == 0)
    total_len = 0;
  newTensor = dl_tensor->data = static_cast<float*>malloc(sizeof(float)*total_len);
  return newTensor;
}