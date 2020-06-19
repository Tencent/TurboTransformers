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
#include <stdint.h>

#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T, layers::types::PoolType t>
void GPUReduceAxisOne(const T* input, T* output, int batch_size, int seq_len,
                      int hidden_size);

template <typename T>
void GPUSequence(T* data_ptr, int64_t size);

template <typename T>
void GPUFill(T* data_ptr, int64_t size, T val);

extern void GPUTransform(int64_t* src_data_ptr, float* dst_data_ptr,
                         const int64_t size);
template <bool AddInput, typename T>
void GPUAddBias(const T* input1, const T* input2, const T* bias, int64_t m,
                int64_t n, cudaStream_t stream, T* out);

template <typename Dtype>
void GPUConcat(const Dtype* t1, const Dtype* t2, const int64_t high_dim,
               const int64_t t1_mid_size, const int64_t t2_mid_size,
               const int64_t low_dim, cudaStream_t stream, Dtype* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
