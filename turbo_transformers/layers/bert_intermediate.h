// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <memory>

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
  void operator()(const core::Tensor& input_tensor, core::Tensor* output) const;

 private:
  core::Tensor dense_weight_;
  core::Tensor dense_bias_;
};

}  // namespace layers
}  // namespace turbo_transformers
