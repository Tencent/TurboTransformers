// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#pragma once
#include <dlpack/dlpack.h>

#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/types/variant.h"
#include "turbo_transformers/core/allocator/allocator_api.h"
#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/enforce.h"
#include "turbo_transformers/core/half.h"
#include "turbo_transformers/core/memory.h"
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif
namespace turbo_transformers {

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

template <>
struct DataTypeTrait<core::Half> {
  enum { DLPackTypeCode = kDLFloat };
};

template <typename T>
static inline bool IsDataType(DLDataType dt) {
  return DataTypeTrait<T>::DLPackTypeCode == dt.code &&
         (dt.bits == 0 || dt.bits == sizeof(T) * 8);
}

using DLManagedTensorPtr =
    std::unique_ptr<DLManagedTensor, details::DLPackManagedTensorDeleter>;

struct DLTensorDimDeleter {
  void operator()(DLTensor *tensor) const {
    if (tensor) {
      delete[] tensor->shape;
      delete tensor;
    }
  }
};

using DLTensorPtr = std::unique_ptr<DLTensor, DLTensorDimDeleter>;

using TensorPayload =
    absl::variant<absl::monostate, DLManagedTensorPtr, DLTensorPtr>;

struct VisitDLTensor {
  const DLTensor &operator()(const DLManagedTensorPtr &ptr) const {
    return ptr->dl_tensor;
  }
  const DLTensor &operator()(const DLTensorPtr &t) const { return *t; }
  const DLTensor &operator()(absl::monostate) const {
    TT_THROW("Tensor is null");
  }
};

}  // namespace details
extern DLManagedTensor *NewDLPackTensor(const std::vector<int64_t> &shape_list,
                                        DLDeviceType device, int device_id,
                                        uint8_t data_type_code, size_t bits,
                                        size_t lanes, const std::string &name);

template <typename T>
inline DLManagedTensor *NewDLPackTensorT(const std::vector<int64_t> &shape_list,
                                         DLDeviceType device = kDLCPU,
                                         int device_id = 0,
                                         const std::string &name = "") {
  return NewDLPackTensor(shape_list, device, device_id,
                         details::DataTypeTrait<T>::DLPackTypeCode,
                         sizeof(T) * 8, 1, name);
}

class Tensor {
 public:
  explicit Tensor(DLManagedTensor *tensor) {
    if (tensor == nullptr) {
      tensor_ = absl::monostate();
    } else {
      tensor_ = details::DLManagedTensorPtr(tensor);
    }
  }

  Tensor(Tensor &&o) noexcept : tensor_(std::move(o.tensor_)){};
  Tensor &operator=(Tensor &&o) noexcept {
    if (this == &o) {
      TT_THROW("Tensor can not be assigned to itself!");
    }
    tensor_ = std::move(o.tensor_);
    return *this;
  }

  DLManagedTensor *ToDLPack() {
    TT_ENFORCE(absl::holds_alternative<details::DLManagedTensorPtr>(tensor_),
               "Must own dltensor");
    return absl::get<details::DLManagedTensorPtr>(tensor_).release();
  }

  size_t n_dim() const {
    auto &dl_tensor = to_dl_tensor();
    return dl_tensor.ndim;
  }

  const int64_t &shape(int pos) const {
    auto &dl_tensor = to_dl_tensor();
    if (pos < 0) {
      pos = dl_tensor.ndim + pos;
    }
    TT_ENFORCE_LT(pos, dl_tensor.ndim,
                  "The index(%d) is out of the range[0...%d]", pos,
                  dl_tensor.ndim - 1);
    return dl_tensor.shape[pos];
  }

  int64_t numel() const {
    auto &dl_tensor = to_dl_tensor();
    return std::accumulate(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim, 1,
                           std::multiplies<int64_t>());
  }

  // FIXME(florianzhao): Maybe this func should not be named Reshape.
  template <typename T>
  T *Reshape(std::vector<int64_t> shape_list, DLDeviceType device_type,
             int device_id, const std::string &name = "Reshape") {
    // if Need Realloc
#ifdef WITH_PERFTOOLS
    auto &profile_ctx = core::Profiler::GetInstance();
    profile_ctx.start_profile(name, device_type);
#endif
    if (absl::visit(ReshapeNeedRealloc(shape_list), tensor_)) {
      tensor_ = details::DLManagedTensorPtr(
          NewDLPackTensorT<T>(shape_list, device_type, device_id, name));
    }
#ifdef WITH_PERFTOOLS
    profile_ctx.end_profile(name, device_type);
#endif
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
    return const_cast<T *>(data<T>());
  }

  DLDeviceType device_type() const {
    auto &dltensor = to_dl_tensor();
    return dltensor.ctx.device_type;
  }

  int device_id() const {
    auto &dltensor = to_dl_tensor();
    return dltensor.ctx.device_id;
  }
  DLContext device_ctx() const {
    auto &dltensor = to_dl_tensor();
    return dltensor.ctx;
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
    os << "first and last 10 elems: (";
    int cnt = 10;
    double sum = 0.;

    if (device_type() == kDLCPU) {
      os << "CPU\n";
      for (int i = 0; i < numel(); ++i) {
        sum += data<T>()[i];
        if (cnt-- >= 0 || numel() - i <= 10) os << data<T>()[i] << ", ";
      }
    } else if (device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
      os << "GPU\n";
      auto n = numel();
      std::unique_ptr<T[]> cpu_data(new T[n]);
      Memcpy(cpu_data.get(), data<T>(), n * sizeof(T), MemcpyFlag::kGPU2CPU);
      for (int i = 0; i < n; ++i) {
        sum += cpu_data[i];
        if (cnt-- >= 0 || n - i <= 10) os << cpu_data[i] << ", ";
      }
#else
      TT_THROW("No CUDA supported, Please Compile with TT_WITH_CUDA");
#endif
    }
    os << ")\n";
    os << "sum is " << sum << std::endl;
  }

  Tensor operator[](int64_t n) {
    auto &dl_tensor = to_dl_tensor();
    TT_ENFORCE_GT(dl_tensor.ndim, 1, "operator[] needs ndim > 1");
    details::DLTensorPtr result(new DLTensor());
    result->dtype = dl_tensor.dtype;
    result->byte_offset = 0;
    result->strides = nullptr;
    result->ctx = dl_tensor.ctx;
    if (n == 0) {
      result->data = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(dl_tensor.data) + dl_tensor.byte_offset);
    } else {
      int64_t offset =
          std::accumulate(dl_tensor.shape + 1, dl_tensor.shape + dl_tensor.ndim,
                          1, std::multiplies<int64_t>());

      int type_byte_size = dl_tensor.dtype.bits / 8;
      result->data = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(dl_tensor.data) + dl_tensor.byte_offset +
          offset * n * type_byte_size);
    }

    result->ndim = dl_tensor.ndim - 1;
    result->shape = new int64_t[result->ndim];
    std::copy(dl_tensor.shape + 1, dl_tensor.shape + dl_tensor.ndim,
              result->shape);

    Tensor r(nullptr);
    r.tensor_ = std::move(result);
    return r;
  }

  const Tensor operator[](int64_t n) const {
    return const_cast<Tensor *>(this)->operator[](n);
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
    TT_ENFORCE_EQ(t.byte_offset, 0, "byte_offset must be zero");

    TT_ENFORCE(details::IsDataType<T>(t.dtype),
               "data type mismatch, request %s, actual (%d,%d)",
               typeid(T).name(), t.dtype.code, t.dtype.bits);
  }

 private:
  struct ReshapeNeedRealloc {
   public:
    ReshapeNeedRealloc(const std::vector<int64_t> &shape_list)
        : shape_list_(shape_list) {}

    bool operator()(details::DLManagedTensorPtr &ptr) const {
      int64_t numel = std::accumulate(
          ptr->dl_tensor.shape, ptr->dl_tensor.shape + ptr->dl_tensor.ndim, 1,
          std::multiplies<int64_t>());
      if (numel >= std::accumulate(shape_list_.begin(), shape_list_.end(), 1,
                                   std::multiplies<int64_t>())) {
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
    const std::vector<int64_t> &shape_list_;
  };

  const DLTensor &to_dl_tensor() const {
    return absl::visit(details::VisitDLTensor(), tensor_);
  }

  details::TensorPayload tensor_;
};

}  // namespace core
}  // namespace turbo_transformers
