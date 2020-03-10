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

#include "fast_transformers/layers/sequence_pool.h"

#include "fast_transformers/layers/kernels/seq_pool.h"

namespace fast_transformers {
namespace layers {

void SequencePool::operator()(const core::Tensor &input,
                              core::Tensor *output) const {
  kernels::SeqPool<float>(input, pool_type_, output);
}

}  // namespace layers
}  // namespace fast_transformers
