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

#include "turbo_transformers/layers/positionwise_ffn.h"

#include <loguru.hpp>

#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"

namespace turbo_transformers {
namespace layers {

namespace details {
// static void Add(const core::Tensor& A, const core::Tensor& B, core::Tensor*
// C) {
//   TT_ENFORCE_EQ(A.numel(), B.numel(), "Tensor A and Tensor B should have the
//   same numel."); auto size = A.numel(); vsAdd(size, A.data<float>(),
//   B.data<float>(), C->mutableData<float>());
// }

// output += bias
static void AddInputBias(const core::Tensor& input1, const core::Tensor& input2,
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
}  // namespace details

void PositonWiseFFN::operator()(core::Tensor& input_tensor,
                                core::Tensor* output_tensor) const {
  TT_ENFORCE_EQ(
      kernels::common::is_same_device_ctx(input_tensor.device_ctx(),
                                          output_tensor->device_ctx()),
      true,
      "PositonWiseFFN: The input_tensor and hidden_states should have "
      "the same device type and device id.");
  auto devType = input_tensor.device_type();
  auto devId = input_tensor.device_id();

  // (batch_size, input_len, model_dim)
  auto batch_size = input_tensor.shape(0);
  auto input_len = input_tensor.shape(1);
  auto model_dim = input_tensor.shape(2);
  core::Tensor input_tensor_copy(nullptr);
  input_tensor_copy.Reshape<float>({batch_size, input_len, model_dim}, devType,
                                   devId);

  core::Copy<float>(input_tensor, input_tensor_copy);

  output_tensor->Reshape<float>({batch_size, input_len, model_dim}, devType,
                                devId);

  kernels::LayerNorm<float>(layer_norm_weight_, layer_norm_bias_,
                            &input_tensor);
  kernels::MatMul(input_tensor, false, dense_weight_1_, false, 1.0,
                  output_tensor, 0.0);
  kernels::AddBiasAct<float, types::ActivationType::Relu>(dense_bias_1_,
                                                          output_tensor);
  kernels::MatMul(*output_tensor, false, dense_weight_2_, false, 1.0,
                  &input_tensor, 0.0);
  details::AddInputBias(input_tensor_copy, input_tensor, dense_bias_2_,
                        output_tensor);
}

void PositonWiseFFN::EnforceShapeAndType() const {
  if (loguru::current_verbosity_cutoff() >= 3) {
    std::stringstream ss;
    LOG_S(3) << ss.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers
