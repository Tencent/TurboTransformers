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
#include <numeric>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
extern void LayerNorm(const core::Tensor& gamma, const core::Tensor& beta,
                      core::Tensor* out_tensor, T eps = 1e-12,
                      const std::string name = "LayerNorm");

template <typename T>
extern void AddBiasLayerNorm(const core::Tensor& input_tensor,
                             const core::Tensor& bias_tensor,
                             const core::Tensor& gamma_tensor,
                             const core::Tensor& beta_tensor,
                             core::Tensor* out_tensor, T eps = 1e-12,
                             const std::string name = "AddBiasLayerNorm");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
