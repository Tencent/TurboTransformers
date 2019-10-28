#pragma once
#include "fast_transformers/core/enforce.h"
#include <dlpack/dlpack.h>
#include <memory>
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
};

} // namespace details

class Tensor {
public:
  explicit Tensor(DLManagedTensor *tensor) : tensor_(tensor) {}

  DLManagedTensor *ToDLPack() {
    FT_ENFORCE_NE(tensor_, nullptr, "The Tensor must contain data");
    return tensor_.release();
  }

  size_t length() const {
    size_t length_ = 1;
    for(int i = 0; i < n_dim(); ++i) {
      length_ *= shape(i);
    }
    return length_;
  }
  size_t n_dim() const { return tensor_->dl_tensor.ndim; }

  int64_t shape(size_t pos) const { return tensor_->dl_tensor.shape[pos]; }

  template <typename T> const T *data() const {
    FT_ENFORCE_EQ(tensor_->dl_tensor.strides, nullptr,
                  "strides must be nullptr");
    FT_ENFORCE_EQ(tensor_->dl_tensor.byte_offset, 0,
                  "byte_offset must be zero");
    FT_ENFORCE(
        details::DataTypeTrait<T>::CheckDataType(tensor_->dl_tensor.dtype),
        "data type mismatch");
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  static DLManagedTensor* CreateDLPackTensor(initializer_list<int64_t> shape_list) {
    DLManagedTensor* newTensor = new DLManagedTensor;
    vector<int64_t> shape_vec(shape_list);
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
    tensor = dl_tensor->data = static_cast<float*>malloc(sizeof(float)*total_len);
  }

private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

} // namespace core

} // namespace fast_transformers
