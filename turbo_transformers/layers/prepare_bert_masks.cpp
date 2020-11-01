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

#include "prepare_bert_masks.h"

#include "turbo_transformers/layers/kernels/common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/layers/kernels/gpu_utils.h"
#endif

namespace turbo_transformers {
namespace layers {

void PrepareBertMasks::operator()(const core::Tensor& inputs,
                                  core::Tensor* att_mask,
                                  core::Tensor* seq_type,
                                  core::Tensor* position_ids,
                                  core::Tensor* extended_attention_mask) const {
  if (position_ids->is_null()) {
    auto pos_ids_ptr = position_ids->Reshape<int64_t>(
        {inputs.shape(0), inputs.shape(1)}, inputs.device_type(),
        inputs.device_id(), "PrepareBertMasks/possitionids/Reshape");

    // fill range
    for (int64_t row_id = 0; row_id < inputs.shape(0); ++row_id) {
      kernels::common::Sequence(pos_ids_ptr, inputs.shape(1),
                                inputs.device_type());
      pos_ids_ptr += inputs.shape(1);
    }
  }

  if (seq_type->is_null()) {
    // seq_type.zeros_like(inputs)
    seq_type->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)},
                               inputs.device_type(), inputs.device_id(),
                               "PrepareBertMasks/seqids/Reshape");
    kernels::common::Fill(seq_type->mutableData<int64_t>(), seq_type->numel(),
                          static_cast<int64_t>(0), inputs.device_type());
  }

  if (att_mask->is_null()) {
    att_mask->Reshape<int64_t>({inputs.shape(0), inputs.shape(1)},
                               inputs.device_type(), inputs.device_id(),
                               "PrepareBertMasks/attmask/Reshape");
    kernels::common::Fill(att_mask->mutableData<int64_t>(), att_mask->numel(),
                          static_cast<int64_t>(1), inputs.device_type());
  }

  // cast att_mask to float
  extended_attention_mask->Reshape<float>(
      {att_mask->shape(0), 1, 1, att_mask->shape(1)}, inputs.device_type(),
      inputs.device_id(), "PrepareBertMasks/extendedattnmask/Reshape");
  kernels::common::Transform(att_mask->mutableData<int64_t>(),
                             extended_attention_mask->mutableData<float>(),
                             att_mask->numel(), inputs.device_type());
}

}  // namespace layers
}  // namespace turbo_transformers
