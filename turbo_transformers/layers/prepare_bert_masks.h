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

#include "turbo_transformers/core/tensor.h"
namespace turbo_transformers {
namespace layers {

class PrepareBertMasks {
 public:
  void operator()(const core::Tensor& inputs, core::Tensor* att_mask,
                  core::Tensor* seq_type, core::Tensor* position_ids,
                  core::Tensor* extended_attention_mask) const;
};

}  // namespace layers
}  // namespace turbo_transformers
