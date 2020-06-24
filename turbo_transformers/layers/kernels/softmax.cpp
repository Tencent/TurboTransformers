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

#include "turbo_transformers/layers/kernels/softmax.h"

#include <cmath>
#include <numeric>
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_softmax_kernel.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {
// attr_mask's shape could be (batch, from_len, to_len),  (batch, 1, to_len) or
// nullptr is2D is used to distinguish the two scenarios.
void SoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                 int64_t head_num, int64_t from_seq_len, int64_t to_seq_len,
                 float scale, bool is2D = true) {
  int64_t M = batch_size * head_num * from_seq_len;
  int64_t N = to_seq_len;
#pragma omp parallel for
  for (int64_t i = 0; i < M; ++i) {
    auto* qk_buf_ptr = qk_buf + i * N;
    if (attr_mask != nullptr) {
      const float* attr_mask_ptr;
      auto batch_idx = i / (head_num * from_seq_len);
      auto from_seq_idx = i % from_seq_len;
      attr_mask_ptr =
          attr_mask +
          (is2D ? batch_idx * to_seq_len
                : (batch_idx * from_seq_len + from_seq_idx) * to_seq_len);
// max-trick
#pragma omp simd
      for (int64_t j = 0; j < N; ++j) {
        auto qk_val = qk_buf_ptr[j];
        auto mask_val = attr_mask_ptr[j];
        qk_val = qk_val * scale + mask_val;
        qk_buf_ptr[j] = qk_val;
      }
    } else {
// max-trick
#pragma omp simd
      for (int64_t j = 0; j < N; ++j) {
        auto qk_val = qk_buf_ptr[j];
        qk_val = qk_val * scale;
        qk_buf_ptr[j] = qk_val;
      }
    }
    float max_val = std::numeric_limits<float>::lowest();
#pragma omp simd reduction(max : max_val)
    for (int64_t j = 0; j < N; ++j) {
      max_val = std::max(max_val, qk_buf_ptr[j]);
    }
#pragma omp simd
    for (int64_t j = 0; j < N; ++j) {
      qk_buf_ptr[j] = std::exp(qk_buf_ptr[j] - max_val);
    }
    float sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int64_t j = 0; j < N; ++j) {
      sum += qk_buf_ptr[j];
    }
    auto coef = 1.0f / sum;
#pragma omp simd
    for (int64_t j = 0; j < N; ++j) {
      qk_buf_ptr[j] *= coef;
    }
  }
}

void ApplyMaskAndSoftmax(core::Tensor* inout, const core::Tensor& att_mask,
                         float scale, const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, inout->device_type());
#endif
  auto batch_size = inout->shape(0);
  auto num_att_heads = inout->shape(1);
  auto from_seq_len = inout->shape(2);
  auto to_seq_len = inout->shape(3);
  bool is_2D = false;
  if (!att_mask.is_null()) {
    if (att_mask.n_dim() == 2 ||
        (att_mask.n_dim() == 3 && att_mask.shape(1) == 1) ||
        (att_mask.n_dim() == 4 && att_mask.shape(2) == 1)) {
      is_2D = true;
    } else {
      is_2D = false;
    }
  }
  const float* att_mask_data = nullptr;
  if (!att_mask.is_null()) {
    att_mask_data = att_mask.data<float>();
  }
  if (inout->device_type() == kDLCPU) {
    SoftmaxMask(inout->mutableData<float>(), att_mask_data, batch_size,
                num_att_heads, from_seq_len, to_seq_len, scale, is_2D);
  } else if (inout->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSoftmaxMask(inout->mutableData<float>(), att_mask_data, batch_size,
                   num_att_heads, from_seq_len, to_seq_len, scale, is_2D,
                   cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, inout->device_type());
#endif
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
