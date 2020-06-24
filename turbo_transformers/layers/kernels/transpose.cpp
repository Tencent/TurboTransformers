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

#include "turbo_transformers/layers/kernels/transpose.h"

#include <cstring>

#include "common.h"
#include "turbo_transformers/layers/kernels/common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_transpose_kernel.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

// tranpose(2,3)
// input (B x seq_len x head_num * hidden_size)
// bias (head * hidden_size)
static void AddBiasTransposeForScoreImpl(const float* input, const float* bias,
                                         int64_t dim0, int64_t dim1,
                                         int64_t dim2, int64_t dim3,
                                         float* output) {
#pragma omp parallel for
  for (int64_t idx = 0; idx < dim0 * dim1; ++idx) {
    int64_t dim0_idx = idx / dim1;
    int64_t dim1_idx = idx % dim1;
    for (int64_t dim2_idx = 0; dim2_idx < dim2; ++dim2_idx) {
      const float* bias_ptr = bias + dim2_idx * dim3;
      auto* src = input + dim0_idx * (dim1 * dim2 * dim3) +
                  dim1_idx * dim2 * dim3 + dim2_idx * dim3;
      auto* dst = output + dim0_idx * (dim1 * dim2 * dim3) +
                  dim2_idx * dim1 * dim3 + dim1_idx * dim3;
#pragma omp simd
      for (int64_t dim3_idx = 0; dim3_idx < dim3; ++dim3_idx) {
        dst[dim3_idx] = src[dim3_idx] + bias_ptr[dim3_idx];
      }
    }
  }
}

// tranpose(2,3)
static void TransposeForScoreImpl(float* output, const float* input,
                                  int64_t batch_size, int64_t seq_length,
                                  int64_t num_attention_heads, int64_t width) {
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * seq_length; ++idx) {
    int64_t batch_idx = idx / seq_length;
    int64_t seq_idx = idx % seq_length;
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      auto* src = input +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  seq_idx * width + head_idx * seq_length * width;
      auto* dst = output +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  seq_idx * num_attention_heads * width + head_idx * width;
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx];
      }
    }
  }
}

void TransposeForScore(core::Tensor* output, const core::Tensor& input,
                       const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input.device_type());
#endif
  if (input.device_type() == kDLCPU && output->device_type() == kDLCPU) {
    TransposeForScoreImpl(output->mutableData<float>(), input.data<float>(),
                          output->shape(0), output->shape(1), input.shape(1),
                          input.shape(3));
  } else if (input.device_type() == kDLGPU && output->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto batch_size = output->shape(0);
    auto seq_length = output->shape(1);
    auto num_attention_heads = input.shape(1);
    auto width = input.shape(3);
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    const float* dummy = nullptr;
    GPUTransposeForScore<float, false>(
        input.data<float>(), dummy, batch_size, seq_length, num_attention_heads,
        width, cuda_ctx.stream(), output->mutableData<float>());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input.device_type());
#endif
}

// add bias and transpose(2,3)
void AddBiasTransposeForScore(const core::Tensor& input,
                              const core::Tensor& bias, core::Tensor* output,
                              const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input.device_type());
#endif
  TT_ENFORCE_EQ(input.n_dim(), 4, "input should be a 4-D tensor");
  TT_ENFORCE_EQ(bias.numel(), input.shape(2) * input.shape(3),
                "bias shape %d should be %d x %d", bias.n_dim(), input.shape(2),
                input.shape(3));
  auto dim0 = input.shape(0);
  auto dim1 = input.shape(1);
  auto dim2 = input.shape(2);
  auto dim3 = input.shape(3);
  if (input.device_type() == kDLCPU && output->device_type() == kDLCPU) {
    AddBiasTransposeForScoreImpl(input.data<float>(), bias.data<float>(), dim0,
                                 dim1, dim2, dim3,
                                 output->mutableData<float>());
  } else if (input.device_type() == kDLGPU && output->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUTransposeForScore<float, true>(input.data<float>(), bias.data<float>(),
                                      dim0, dim1, dim2, dim3, cuda_ctx.stream(),
                                      output->mutableData<float>());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input.device_type());
#endif
}

void SplitAddBiasTransposeForScore(core::Tensor* output_tensor,
                                   const core::Tensor& input_tensor,
                                   const core::Tensor& bias_tensor,
                                   const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input_tensor.device_type());
#endif

  TT_ENFORCE_EQ(output_tensor->n_dim(), 5,
                "output_tensor should be (weight_num, batch_size, seq_length, "
                "num_attention_heads, size_per_head)");
  // TT_ENFORCE_EQ(bias_tensor.n_dim(), 1,
  //               "output_tensor should be (weight_num * num_attention_heads, "
  //               "size_per_head)");

  auto batch_size = output_tensor->shape(1);
  auto seq_length = output_tensor->shape(3);
  auto weight_num = output_tensor->shape(0);
  auto num_attention_heads = output_tensor->shape(2);
  auto width = output_tensor->shape(4);
  auto input = input_tensor.data<float>();
  auto bias = bias_tensor.data<float>();
  auto output = output_tensor->mutableData<float>();

  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           bias_tensor.device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and bias_tensor "
                "should have the same device type and device id.");
  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           output_tensor->device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and output_tensor "
                "should have the same device type and device id.");

  if (output_tensor->device_type() == kDLCPU &&
      input_tensor.device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t idx = 0; idx < batch_size * weight_num * seq_length; ++idx) {
      auto batch_idx = idx / (seq_length * weight_num);
      auto seq_idx = idx / weight_num % seq_length;
      auto weight_idx = idx % weight_num;

      for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
        auto* src_ptr =
            input +
            batch_idx *
                (seq_length * weight_num * num_attention_heads * width) +
            seq_idx * weight_num * num_attention_heads * width +
            weight_idx * (num_attention_heads * width) + head_idx * width;
        auto* dst_ptr = output +
                        weight_idx * (batch_size * num_attention_heads *
                                      seq_length * width) +
                        batch_idx * (num_attention_heads * seq_length * width) +
                        head_idx * seq_length * width + seq_idx * width;
        auto* bias_ptr =
            bias + weight_idx * width * num_attention_heads + head_idx * width;
#pragma omp simd
        for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
          dst_ptr[width_idx] = src_ptr[width_idx] + bias_ptr[width_idx];
        }
      }
    }  // end for
  } else if (output_tensor->device_type() == kDLGPU &&
             input_tensor.device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSplitAddBiasTransposeForScore<float>(
        input, bias, output, batch_size, seq_length, weight_num,
        num_attention_heads, width, cuda_ctx.stream());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input_tensor.device_type());
#endif
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
