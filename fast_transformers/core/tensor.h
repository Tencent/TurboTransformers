#pragma once
#include <dlpack/dlpack.h>

#include <iostream>
#include <memory>
#include <numeric>

#include "absl/types/variant.h"
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

using DLTensorPtr =
    std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter>;
using TensorPayload =
    absl::variant<absl::monostate, DLTensorPtr, DLManagedTensor>;

struct VisitDLTensor {
  const DLManagedTensor &operator()(const DLTensorPtr &ptr) const {
    return *ptr;
  }
  const DLManagedTensor &operator()(const DLManagedTensor &t) const {
    return t;
  }
  const DLManagedTensor &operator()(absl::monostate) const {
    FT_THROW("Tensor is null");
  }
};

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
  explicit Tensor(DLManagedTensor *tensor) {
    if (tensor == nullptr) {
      tensor_ = absl::monostate();
    } else {
      tensor_ = details::DLTensorPtr(tensor);
    }
  }

  DLManagedTensor *ToDLPack() {
    FT_ENFORCE(absl::holds_alternative<details::DLTensorPtr>(tensor_),
               "Must own dltensor");
    return absl::get<details::DLTensorPtr>(tensor_).release();
  }

  size_t n_dim() const {
    auto &dl_tensor = to_dl_tensor();
    return dl_tensor.ndim;
  }

  const int64_t &shape(int pos) const {
    auto &dl_tensor = to_dl_tensor();
    FT_ENFORCE_LT(pos, dl_tensor.ndim,
                  "The index(%d) is out of the range[0...%d]", pos,
                  dl_tensor.ndim);
    return dl_tensor.shape[pos];
  }

  int64_t numel() const {
    auto &dl_tensor = to_dl_tensor();
    return std::accumulate(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim, 1,
                           std::multiplies<int64_t>());
  }

  int64_t rows() const {
    auto &dl_tensor = to_dl_tensor();
    FT_ENFORCE_GE(dl_tensor.ndim, 2, "n_dims() >= 2");
    return std::accumulate(dl_tensor.shape,
                           dl_tensor.shape + dl_tensor.ndim - 1, 1,
                           std::multiplies<int64_t>());
  }
  int64_t cols() const {
    auto &dl_tensor = to_dl_tensor();
    FT_ENFORCE_GE(dl_tensor.ndim, 2, "n_dims() >= 2");
    return dl_tensor.shape[dl_tensor.ndim - 1];
  }

  // FIXME(florianzhao): Maybe this func should not be named Reshape.
  template <typename T>
  T *Reshape(std::initializer_list<int64_t> shape_list) {
    // if Need Realloc
    if (absl::visit(ReshapeNeedRealloc(shape_list), tensor_)) {
      tensor_ = details::DLTensorPtr(NewDLPackTensorT<T>(shape_list));
    }
    return this->template mutableData<T>();
  }

  template <typename T>
  const T *data() const {
    auto &dltensor = to_dl_tensor();
    EnforceDataType<T>(dltensor);
    return reinterpret_cast<T *>(dltensor.data);
  }

  template <typename T>
  T *mutableData() {
    return absl::visit(GetMutableData<T>(), tensor_);
  }

  DLDeviceType device_type() const {
    auto &dltensor = to_dl_tensor();
    return dltensor.ctx.device_type;
  }
  bool is_null() const {
    return absl::holds_alternative<absl::monostate>(tensor_);
  }

  template <typename T>
  void Print(std::ostream &os) const {
    auto &dl_tensor = to_dl_tensor();
    os << "type " << dl_tensor.dtype.code << std::endl;
    os << "bits " << dl_tensor.dtype.bits << std::endl;
    os << "numel: " << numel() << std::endl;
    os << "n_dim: " << n_dim() << std::endl;
    os << "stride: ";
    if (dl_tensor.strides != nullptr) {
      PrintArray(os, dl_tensor.strides, dl_tensor.ndim);
    } else {
      os << "null";
    }
    os << "\n";
    os << "shape: ";
    PrintArray(os, dl_tensor.shape, dl_tensor.ndim);
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
  struct ReshapeNeedRealloc {
   public:
    ReshapeNeedRealloc(const std::initializer_list<int64_t> &shape_list)
        : shape_list_(shape_list) {}

    bool operator()(details::DLTensorPtr &ptr) const {
      int64_t numel = std::accumulate(
          ptr->dl_tensor.shape, ptr->dl_tensor.shape + ptr->dl_tensor.ndim, 1,
          std::multiplies<>());
      if (numel >= std::accumulate(shape_list_.begin(), shape_list_.end(), 1,
                                   std::multiplies<>())) {
        if (ptr->dl_tensor.ndim != static_cast<int>(shape_list_.size())) {
          ptr->dl_tensor.ndim = shape_list_.size();
          delete[] ptr->dl_tensor.shape;
          ptr->dl_tensor.shape = new int64_t[shape_list_.size()];
        }
        std::copy(shape_list_.begin(), shape_list_.end(), ptr->dl_tensor.shape);
        ptr->dl_tensor.ndim = shape_list_.size();
        return false;
      }
      return true;
    }

    template <typename T>
    bool operator()(T &) const {
      return true;
    }

   private:
    const std::initializer_list<int64_t> &shape_list_;
  };

  template <typename T>
  struct GetMutableData {
    T *operator()(details::DLTensorPtr &ptr) const {
      EnforceDataType<T>(ptr->dl_tensor);
      return reinterpret_cast<T *>(ptr->dl_tensor.data);
    }
    template <typename Other>
    T *operator()(Other &) const {
      return nullptr;
    }
  };

  const DLTensor &to_dl_tensor() const {
    return absl::visit(details::VisitDLTensor(), tensor_).dl_tensor;
  }

  details::TensorPayload tensor_;
};

}  // namespace core

}  // namespace fast_transformers
