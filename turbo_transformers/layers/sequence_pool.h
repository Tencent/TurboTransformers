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
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/seq_pool.h"
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {

class SequencePool {
 public:
  explicit SequencePool(const std::string &pool_type) {
    pool_type_ = kernels::GetPoolType(pool_type);
  }
  explicit SequencePool(layers::types::PoolType pt) : pool_type_(pt) {}

  void operator()(const core::Tensor &input_tensor, core::Tensor *output) const;

 private:
  layers::types::PoolType pool_type_;
};

}  // namespace layers
}  // namespace turbo_transformers
