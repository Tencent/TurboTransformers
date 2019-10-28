#pragma once
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

  DLManagedTensor *ToDLPack() { return tensor_.release(); }

  size_t n_dim() const { return tensor_->dl_tensor.ndim; }

  int64_t shape(size_t pos) const { return tensor_->dl_tensor.shape[pos]; }

  template <typename T> const T *data() const {
    if (tensor_->dl_tensor.strides != nullptr) {
      throw std::runtime_error("strides must be nullptr");
    }
    if (tensor_->dl_tensor.byte_offset != 0) {
      throw std::runtime_error("byte_offset must be zero");
    }
    if (details::DataTypeTrait<T>::CheckDataType(tensor_->dl_tensor.dtype)) {
      throw std::runtime_error("data type mismatch");
    }
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

} // namespace core

} // namespace fast_transformers
