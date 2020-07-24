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

#include "turbo_transformers/layers/addbias_layernorm.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"

namespace turbo_transformers {
namespace layers {

void FusedAddBiasLayerNorm::operator()(const core::Tensor &input_tensor, core::Tensor* output_tensor) const {
  kernels::AddBiasLayerNorm<float>(
      input_tensor, bias, norm_weight, norm_bias,
      output_tensor, 1e-12, "AddBiasLayerNorm");
}

}  // namespace layers
}  // namespace turbo_transformers
