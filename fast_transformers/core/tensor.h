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

private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

} // namespace core

} // namespace fast_transformers
