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
#include <string>
#include <utility>
#include "cnpy.h"
#include "turbo_transformers/core/tensor_copy.h"

namespace turbo_transformers {
namespace loaders {

class NPZMapView {
 public:
  NPZMapView(std::string prefix, cnpy::npz_t *npz)
      : prefix_(std::move(prefix)), npz_(npz) {}

  cnpy::NpyArray &operator[](const std::string &key) {
    auto actualKey = prefix_ + key;
    auto it = npz_->find(actualKey);
    TT_ENFORCE(it != npz_->end(), "cannot find parameter %s in npz file",
               actualKey);
    return it->second;
  }

  NPZMapView Sub(const std::string &subview) {
    return NPZMapView(prefix_ + subview + ".", npz_);
  }

  bool IsExist(const std::string &subprefix) {
    auto actualPrefix = prefix_ + subprefix;
    for (auto it = npz_->begin(); it != npz_->end(); ++it) {
      if (it->first.rfind(actualPrefix, 0) == 0) return true;
    }
    return false;
  }

 private:
  std::string prefix_;
  cnpy::npz_t *npz_;
};

class NPZLoader {
 public:
  NPZLoader(NPZMapView view, DLDeviceType device)
      : view_(std::move(view)), device_(device) {}

  template <typename T>
  core::Tensor LoadT(const std::string &name) {
    auto &array = view_[name];
    std::vector<int64_t> shape;
    shape.resize(array.shape.size());
    std::copy(array.shape.begin(), array.shape.end(), shape.begin());
    core::Tensor tensor(core::NewDLPackTensorT<T>(shape, device_));
    core::Copy(array.data<T>(), tensor.numel(), DLDeviceType::kDLCPU, tensor);
    return tensor;
  }

  core::Tensor LoadFloat(const std::string &name) { return LoadT<float>(name); }
  core::Tensor operator[](const std::string &name) { return LoadFloat(name); }

 private:
  NPZMapView view_;
  DLDeviceType device_;
};

}  // namespace loaders
}  // namespace turbo_transformers
