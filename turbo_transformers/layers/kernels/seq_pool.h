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

#include <string>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/core/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

// The input's shape is (batch_size, seq_len, hidden_size)
// and the output's shape is (batch_size, hidden_size)
// The pool_type could be max, mean, first, last.
template <typename T>
void SeqPool(const core::Tensor &input,
             turbo_transformers::core::types::PoolType pool_type,
             core::Tensor *output);

turbo_transformers::core::types::PoolType GetPoolType(
    const std::string &pool_type);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
