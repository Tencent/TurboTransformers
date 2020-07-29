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
#include <memory>
#include <utility>
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {

class BertIntermediate {
 public:
  BertIntermediate(core::Tensor dense_weight, core::Tensor dense_bias)
      : dense_weight_(std::move(dense_weight)),
        dense_bias_(std::move(dense_bias)) {
    EnforceShapeAndType();
  }

  void EnforceShapeAndType() const;
  void operator()(const core::Tensor& input_tensor, core::Tensor* output,
                  int64_t idx = 0) const;

 private:
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
