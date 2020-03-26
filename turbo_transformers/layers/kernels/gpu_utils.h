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
#include <stdint.h>
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T, layers::types::PoolType t>
void GPUReduceAxisOne(const T* input, T* output, int batch_size, int seq_len,
                      int hidden_size);

template <typename T>
void gpu_copy(const T* src, T* dst, int64_t size);

template <typename T>
void gpu_sequence(T* data_ptr, int64_t size);

template <typename T>
void gpu_fill(T* data_ptr, int64_t size, T val);

extern void gpu_transform(int64_t* src_data_ptr, float* dst_data_ptr,
                          const int64_t size);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
