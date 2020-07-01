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

#include "turbo_transformers/layers/albert_layer.h"

#include <loguru.hpp>

#include "turbo_transformers/core/blas.h"
#include "turbo_transformers/core/memory.h"
#include "turbo_transformers/layers/kernels/activation.h"
#include "turbo_transformers/layers/kernels/layer_norm.h"
#include "turbo_transformers/layers/kernels/mat_mul.h"
#include "turbo_transformers/layers/kernels/softmax.h"
#include "turbo_transformers/layers/kernels/transpose.h"
#include "turbo_transformers/layers/kernels/common.h"


namespace turbo_transformers {
namespace layers {

void AlbertLayer::operator()(const core::Tensor& input_tensor,
                             core::Tensor* hidden_output,
                             core::Tensor* output_tensor) const {
  hidden_output->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id());

  kernels::MatMul(input_tensor, false, dense_weight_, false, 1.0, hidden_output,
                  0.0);
  kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(dense_bias_,
                                                            hidden_output);
  output_tensor->Reshape<float>(
      {input_tensor.shape(0), input_tensor.shape(1), dense_output_weight_.shape(1)},
      input_tensor.device_type(), input_tensor.device_id());
  kernels::MatMul(*hidden_output, false, dense_output_weight_, false, 1.0,
                  output_tensor, 0.0);
  kernels::AddBiasLayerNorm<float>(input_tensor, dense_output_bias_,
                                   layer_norm_weight_, layer_norm_bias_,
                                   output_tensor);
}

void AlbertLayer::EnforceShapeAndType() const {
  TT_ENFORCE_EQ(dense_weight_.n_dim(), 2, "dense weight must be matrix");
  TT_ENFORCE_EQ(dense_bias_.n_dim(), 1, "dense bias must be vector");
  TT_ENFORCE_EQ(dense_weight_.shape(1), dense_bias_.shape(0),
                "weight and bias shape mismatch %d, %d", dense_weight_.shape(0),
                dense_bias_.shape(0));

  if (loguru::current_verbosity_cutoff() >= 3) {
    std::ostringstream os;
    os << ">>>>>>>>>>>> query_weight <<<<<<<<<<<<" << std::endl;
    dense_weight_.Print<float>(os);
    os << ">>>>>>>>>>>> query_bias <<<<<<<<<<<<" << std::endl;
    dense_bias_.Print<float>(os);
    os << "<<<<<<<< dense_weight_ <<<<<<<<<<";
    dense_output_weight_.Print<float>(os);
    os << "<<<<<<<< dense_bias <<<<<<<<<<";
    dense_output_bias_.Print<float>(os);
    os << "<<<<<<<< layer_norm_weight <<<<<<<<<<";
    layer_norm_weight_.Print<float>(os);
    os << "<<<<<<<< layer_norm_bias <<<<<<<<<<";
    layer_norm_bias_.Print<float>(os);
    LOG_S(3) << os.str();
  }
}

}  // namespace layers
}  // namespace turbo_transformers