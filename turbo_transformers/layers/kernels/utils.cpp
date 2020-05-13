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

#include "utils.h"

#include "common.h"
#ifdef TT_WITH_CUDA
#include <cuda.h>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

void AddBias(const core::Tensor& bias, core::Tensor* output) {
  auto dim1 = bias.shape(0);
  auto dim0 = output->numel() / dim1;
  auto output_data = output->mutableData<float>();
  const auto bias_data = bias.data<float>();
#pragma omp parallel for
  for (int64_t i = 0; i < dim0; ++i) {
#pragma omp simd
    for (int64_t j = 0; j < dim1; ++j) {
      output_data[i * dim1 + j] += bias_data[j];
    }
  }
}

void AddInputBias(const core::Tensor& input1, const core::Tensor& input2,
                  const core::Tensor& bias, core::Tensor* output) {
  TT_ENFORCE_EQ(input1.numel(), input2.numel(),
                "Tensor input1 and Tensor input2 should have the same numel.");
  auto dim1 = bias.shape(0);
  auto dim0 = output->numel() / dim1;
  auto output_data = output->mutableData<float>();
  const auto bias_data = bias.data<float>();
  const auto input1_data = input1.data<float>();
  const auto input2_data = input2.data<float>();
#pragma omp parallel for
  for (int64_t i = 0; i < dim0; ++i) {
#pragma omp simd
    for (int64_t j = 0; j < dim1; ++j) {
      output_data[i * dim1 + j] =
          bias_data[j] + input1_data[i * dim1 + j] + input2_data[i * dim1 + j];
    }
  }
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
