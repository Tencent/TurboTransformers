#pragma once
#include "fast_transformers/core/enforce.h"
#include <dlpack/dlpack.h>
#include <memory>
#include <iostream>

namespace fast_transformers {

enum class DeviceType{ GPU, CPU };

namespace core {
namespace details {

struct DLPackManagedTensorDeleter {
  void operator()(DLManagedTensor *tensor) const {
    if (tensor == nullptr) {
      return;
    }
    tensor->deleter(tensor);
  }
};

template <typename T, DeviceType kDevice> struct DataTypeTrait;

template <> struct DataTypeTrait<float, DeviceType::CPU> {
  inline static bool CheckDataType(DLDataType data_type) {
    return data_type.code == kDLFloat && data_type.bits == 32;
  }
  
  static DLDataType getDLDataType() {
    static DLDataType dlDataType;
    dlDataType.code = kDLFloat;
    dlDataType.bits = sizeof(float) * 8;
    dlDataType.lanes = 0;
    return dlDataType;
  }

  static DLContext getDLContext() {
    static DLContext dlCxt;
    dlCxt.device_type = kDLCPU;
    dlCxt.device_id = 0; //dev id of GPU
  }
  
};

template<typename T, DeviceType kDev>
DLManagedTensor* CreateDLPackTensor(std::initializer_list<int64_t> shape_list) {
  DLManagedTensor* newTensor = new DLManagedTensor;
  std::vector<int64_t> shape_vec(shape_list);

  int64_t* shape_ptr = new int64_t [shape_vec.size()];
  for(int i = 0; i < shape_vec.size(); ++i)
    shape_ptr[i] = shape_vec[i];

  newTensor->dl_tensor.ctx = DataTypeTrait<T, kDev>::getDLContext();
  newTensor->dl_tensor.ndim = shape_vec.size();

  newTensor->dl_tensor.dtype = DataTypeTrait<T, kDev>::getDLDataType();
  newTensor->dl_tensor.shape = shape_ptr;

  newTensor->dl_tensor.strides = nullptr;
  newTensor->dl_tensor.byte_offset = 0;
  
  size_t numel_;
  if(shape_vec.size() == 0)
    numel_ = 0;
  numel_ = 1;
  for(int i = 0; i < shape_vec.size(); ++i)
    numel_ *= shape_vec[i];
  //TODO allocator interface: allocator<DeviceType kDev>(size_t size_)
  newTensor->dl_tensor.data = static_cast<T*>(malloc(sizeof(T) * numel_));

  newTensor->deleter = [](struct DLManagedTensor * self) { 
    //TDOO allocator interface: freer<DeviceType kDev>(T* data_);
    free(self->dl_tensor.data);
    delete self->dl_tensor.shape;
  };
  return newTensor;
}

} // namespace details

class Tensor {
public:
  explicit Tensor(DLManagedTensor *tensor) : tensor_(tensor) {}

  DLManagedTensor *ToDLPack() {
    FT_ENFORCE_NE(tensor_, nullptr, "The Tensor must contain data");
    return tensor_.release();
  }

  size_t n_dim() const { return tensor_->dl_tensor.ndim; }
  
  int64_t shape(size_t pos) const { return tensor_->dl_tensor.shape[pos]; }

  size_t numel() const {
    if(n_dim() == 0)
      return 0;
    size_t numel_ = 1;
    for(int i = 0; i < n_dim(); ++i)
      numel_ *= shape(i);
    return numel_;
  }


  template <typename T> const T *data() const {
    FT_ENFORCE_EQ(tensor_->dl_tensor.strides, nullptr,
                  "strides must be nullptr");
    FT_ENFORCE_EQ(tensor_->dl_tensor.byte_offset, 0,
                  "byte_offset must be zero");
    /*
    FT_ENFORCE(
        details::DataTypeTrait<T, details::DeviceType::CPU>::CheckDataType(tensor_->dl_tensor.dtype),
        "data type mismatch");
    */
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  template <typename T> T *mutableData() {
    FT_ENFORCE_EQ(tensor_->dl_tensor.strides, nullptr,
                  "strides must be nullptr");
    FT_ENFORCE_EQ(tensor_->dl_tensor.byte_offset, 0,
                  "byte_offset must be zero");
    /* 
    FT_ENFORCE(
        details::DataTypeTrait<T>::CheckDataType(tensor_->dl_tensor.dtype),
        "data type mismatch");
        */
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  template <typename T> void print() const {
    std::cout << "numel " << numel() << std::endl;
    for(int i = 0; i < numel(); ++i) {
      std::cout << data<T>()[i] << ", ";
    }
    std::cout << std::endl;
  }

private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

} // namespace core

} // namespace fast_transformers
