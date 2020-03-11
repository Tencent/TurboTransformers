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
#include <dlpack/dlpack.h>
#include "easy_transformers/core/tensor.h"

#ifdef FT_WITH_CUDA
#include "easy_transformers/layers/kernels/gpu_utils.h"
#endif

namespace easy_transformers {
namespace core {
namespace common {

extern bool is_same_device_ctx(DLContext t1, DLContext t2);

extern bool is_same_shape(const Tensor& t1, const Tensor& t2);

template <typename T>
void ft_seqence(T* data, int64_t size, DLDeviceType device);

template <typename T>
void ft_fill(T* data, int64_t size, T val, DLDeviceType device);

// TODO(jiaruifang): this function should better pass a function in.
// how can we pass a lambda function as __device__ to cuda?
void ft_transform(int64_t* src_data, float* dst_data, int64_t size,
                  DLDeviceType device);

}  // namespace common
}  // namespace core
}  // namespace easy_transformers
