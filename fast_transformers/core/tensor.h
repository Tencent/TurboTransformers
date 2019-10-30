#pragma once
#include "fast_transformers/core/enforce.h"
#include <dlpack/dlpack.h>
#include <memory>
#include <iostream>
#include "fast_transformers/core/common.h"
#include "fast_transformers/core/blas.h"

namespace fast_transformers {

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

template <typename T> struct DataTypeTrait;

template <> struct DataTypeTrait<float> {
  inline static bool CheckDataType(DLDataType data_type) {
    return data_type.code == kDLFloat && data_type.bits == 32;
  }
  enum {
    DLPackTypeCode = kDLFloat
  };
};

template <> struct DataTypeTrait<int> {
  inline static bool CheckDataType(DLDataType data_type) {
    return data_type.code == kDLInt && data_type.bits == 32;
  }
  enum {
    DLPackTypeCode = kDLInt
  };
};

template<typename T, DeviceType kDev>
inline DLManagedTensor* CreateDLPackTensor(std::initializer_list<int64_t> shape_list) {
  DLManagedTensor* newTensor = new DLManagedTensor;
  std::vector<int64_t> shape_vec(shape_list);

  newTensor->dl_tensor.shape = new int64_t [shape_list.size()];
  std::copy(shape_list.begin(), shape_list.end(), newTensor->dl_tensor.shape);

  newTensor->dl_tensor.ctx = {kDLCPU, 0}; //device_type, device_id
  newTensor->dl_tensor.ndim = shape_list.size(); 

  newTensor->dl_tensor.dtype = {DataTypeTrait<T>::DLPackTypeCode, 32, 1}; //code, bits, lanes

  newTensor->dl_tensor.strides = nullptr; //TODO
  newTensor->dl_tensor.byte_offset = 0;

  size_t numel;
  if(shape_vec.size() == 0)
    numel = 0;
  else
    numel = std::accumulate(shape_list.begin(), shape_list.end(), 1, std::multiplies<int64_t>());
  newTensor->dl_tensor.data = static_cast<T*>(cpu_allocater(numel * sizeof(T), 64, false)); //new T [numel];

  newTensor->deleter = [](struct DLManagedTensor * self) { 
    if(self->dl_tensor.data)
      cpu_freer(self->dl_tensor.data);
      //delete [] (T*)self->dl_tensor.data;
    if(self->dl_tensor.shape)
      delete [] self->dl_tensor.shape;
    //if(self->manager_ctx)
    //  free(self->manager_ctx); //warning: deleting 'void*' is undefined
    delete self;
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
    FT_ENFORCE_EQ(tensor_->dl_tensor.byte_offset, 0,
                  "byte_offset must be zero");
    
    //FT_ENFORCE(details::DataTypeTrait<T>::CheckDataType(tensor_->dl_tensor.dtype),
    //    "data type mismatch");
    
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  template <typename T> T *mutableData() {
    FT_ENFORCE_EQ(tensor_->dl_tensor.byte_offset, 0,
                  "byte_offset must be zero");
    
    //FT_ENFORCE(details::DataTypeTrait<T>::CheckDataType(tensor_->dl_tensor.dtype),
    //    "data type mismatch");
        
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  DLDataTypeCode GetDataTypeCode() const {
    return static_cast<DLDataTypeCode>(tensor_->dl_tensor.dtype.code);
  }

  DLDeviceType GetDeviceType() const {
    return tensor_->dl_tensor.ctx.device_type;
  }
  //if stride is NULL, indicating tensor is compact and row-majored.
  int64_t GetStride() const {
    if(tensor_->dl_tensor.strides != nullptr)
      return *(tensor_->dl_tensor.strides);
    else
      return 0;
  }
  template <typename T> void Print(std::ostream& os) const {
    switch (GetDataTypeCode()) {
      case kDLInt:
        os << "type: int" <<std::endl;
        break;
      case kDLUInt:
        os << "type: unsigned" <<std::endl;
        break;
      case kDLFloat:
        os << "type float" << std::endl;
        break;
      default:
        os << "unrecoginized type" << std::endl;
    }
    
    os << "numel: " << numel() << std::endl;
    os << "stride: " << GetStride() << std::endl;
    os << "n_dim: " << n_dim() << ", shape: ";
    for(int i = 0; i < n_dim(); ++i)
      os << shape(i) << ", ";
    os << std::endl;

    int cnt = 10;
    double sum = 0.;
    for(int i = 0; i < numel(); ++i) {
      sum += data<T>()[i];
      if(cnt-- >= 0)
        os << data<T>()[i] << ", ";
    }
    os << "sum is " << sum << std::endl;
  }

private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

} // namespace core

} // namespace fast_transformers
