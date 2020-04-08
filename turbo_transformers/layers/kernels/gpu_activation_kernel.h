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
#include <cuda_runtime.h>
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
using types::ActivationType;
template <typename T, ActivationType ActType>
void GPUAddBiasActKernel(const T* bias_data, int64_t batch_size,
                         int64_t feature_dim, cudaStream_t stream, T* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
