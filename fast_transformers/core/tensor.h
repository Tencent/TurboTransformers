#pragma once
#include <dlpack/dlpack.h>

#include <iostream>
#include <memory>
#include <numeric>

#include "fast_transformers/core/blas.h"
#include "fast_transformers/core/enforce.h"
#include "fast_transformers/core/memory.h"

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

template <typename T>
struct DataTypeTrait;

template <>
struct DataTypeTrait<float> {
  enum { DLPackTypeCode = kDLFloat };
};

template <>
struct DataTypeTrait<int> {
  enum { DLPackTypeCode = kDLInt };
};

template <>
struct DataTypeTrait<int64_t> {
  enum { DLPackTypeCode = kDLInt };
};

template <typename T>
static inline bool IsDataType(DLDataType dt) {
  return DataTypeTrait<T>::DLPackTypeCode == dt.code &&
         (dt.bits == 0 || dt.bits == sizeof(T) * 8);
}

}  // namespace details
extern DLManagedTensor *NewDLPackTensor(
    std::initializer_list<int64_t> shape_list, DLDeviceType device,
    int device_id, uint8_t data_type_code, size_t bits, size_t lanes);

template <typename T>
inline DLManagedTensor *NewDLPackTensorT(
    std::initializer_list<int64_t> shape_list, DLDeviceType device = kDLCPU,
    int device_id = 0) {
  return NewDLPackTensor(shape_list, device, device_id,
                         details::DataTypeTrait<T>::DLPackTypeCode,
                         sizeof(T) * 8, 1);
}

class Tensor {
 public:
  explicit Tensor(DLManagedTensor *tensor) : tensor_(tensor) {}

  DLManagedTensor *ToDLPack() {
    // FT_ENFORCE_NE(tensor_, nullptr, "The Tensor must contain data");
    return tensor_.release();
  }

  size_t n_dim() const { return tensor_->dl_tensor.ndim; }

  const int64_t &shape(size_t pos) const {
    FT_ENFORCE_LT(pos, tensor_->dl_tensor.ndim,
                  "The index(%d) is out of the range[0...%d]", pos,
                  tensor_->dl_tensor.ndim);
    return tensor_->dl_tensor.shape[pos];
  }

  size_t numel() const {
    return std::accumulate(tensor_->dl_tensor.shape,
                           tensor_->dl_tensor.shape + tensor_->dl_tensor.ndim,
                           1, std::multiplies<int64_t>());
  }

  std::vector<int64_t> Dims() const {
    std::vector<int64_t> dims;
    dims.reserve(tensor_->dl_tensor.ndim);
    for (size_t i = 0; i < tensor_->dl_tensor.ndim; ++i) {
      dims.emplace_back(tensor_->dl_tensor.shape[i]);
    }
    return dims;
  }

  void Reshape(std::initializer_list<int64_t> dims) {
    FT_ENFORCE(tensor_, "This tensor is not initialized.)");
    auto t_dims = Dims();
    bool is_same_shape = false;
    if (t_dims.size() == dims.size()) {
      is_same_shape = std::equal(std::begin(dims), std::end(dims),
                                 std::begin(t_dims), std::end(t_dims));
    }
    if (!is_same_shape) {
      tensor_.reset(core::NewDLPackTensorT<float>(dims));
    }
  }

  bool is_same_shape(const std::vector<int64_t> &shape) const {
    if (shape.size() != tensor_->dl_tensor.ndim) return false;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != tensor_->dl_tensor.shape[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  const T *data() const {
    EnforceDataType<T>(tensor_->dl_tensor);
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  template <typename T>
  T *mutableData() {
    EnforceDataType<T>(tensor_->dl_tensor);
    return reinterpret_cast<T *>(tensor_->dl_tensor.data);
  }

  DLDeviceType device_type() const {
    return tensor_->dl_tensor.ctx.device_type;
  }

  template <typename T>
  void Print(std::ostream &os) const {
    os << "type " << tensor_->dl_tensor.dtype.code << std::endl;
    os << "bits " << tensor_->dl_tensor.dtype.bits << std::endl;
    os << "numel: " << numel() << std::endl;
    os << "n_dim: " << n_dim() << std::endl;
    os << "stride: ";
    if (tensor_->dl_tensor.strides != nullptr) {
      PrintArray(os, tensor_->dl_tensor.strides, tensor_->dl_tensor.ndim);
    } else {
      os << "null";
    }
    os << "\n";
    os << "shape: ";
    PrintArray(os, tensor_->dl_tensor.shape, tensor_->dl_tensor.ndim);
    os << "\n";
    os << "first 10 elems: (";
    int cnt = 10;
    double sum = 0.;
    for (int i = 0; i < numel(); ++i) {
      sum += data<T>()[i];
      if (cnt-- >= 0) os << data<T>()[i] << ", ";
    }
    os << ")\n";
    os << "sum is " << sum << std::endl;
  }

 private:
  template <typename T>
  static void PrintArray(std::ostream &os, const T *data, size_t n) {
    os << "(";
    for (size_t i = 0; i < n; ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << data[i];
    }
    os << ")";
  }

  template <typename T>
  static void EnforceDataType(DLTensor t) {
    FT_ENFORCE_EQ(t.byte_offset, 0, "byte_offset must be zero");

    FT_ENFORCE(details::IsDataType<T>(t.dtype),
               "data type mismatch, request %s, actual (%d,%d)",
               typeid(T).name(), t.dtype.code, t.dtype.bits);
  }

 private:
  std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter> tensor_;
};

}  // namespace core

}  // namespace fast_transformers
