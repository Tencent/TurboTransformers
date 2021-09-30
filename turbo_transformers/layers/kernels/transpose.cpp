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

#include <algorithm>
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

// Transpose the 2,3 dim of a 4-D tensor after adding a bias.
// It should work as a general function. Do not care with dim is the seq_len
// dim. Because the application scenerio of this function is unique. input (B x
// seq_len x head_num * hidden_size) bias (head * hidden_size)
static void AddBiasTransposeForScoreImpl(const float* input, const float* bias,
                                         int64_t batch_size, int64_t seq_length,
                                         int64_t num_attention_heads,
                                         int64_t width, float* output) {
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * seq_length; ++idx) {
    int64_t batch_idx = idx / seq_length;
    int64_t seq_idx = idx % seq_length;
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const float* bias_ptr = bias + head_idx * width;
      auto* src = input +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  seq_idx * num_attention_heads * width + head_idx * width;
      auto* dst = output +
                  batch_idx * (seq_length * num_attention_heads * width) +
                  head_idx * seq_length * width + seq_idx * width;
#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx] + bias_ptr[width_idx];
      }
    }
  }
}

// Transpose the 2,3 dim of a 4-D tensor after adding a bias with smart padding.
// Therefore you have to pay attention to which dim is seq_len.
// This function is not a general function. You should care with dime is
// seq_len. input (1 x sum_seq_len x head_num * hidden_size) bias (head *
// hidden_size) output (batch, head, max_seq_len, hidden_size)
static void AddBiasTransposeForScorePadImpl(
    const float* input, const float* bias, int64_t num_attention_heads,
    int64_t width, float* output, const std::vector<int64_t>& seq_len_list) {
  int64_t batch_size = seq_len_list.size();
  int64_t max_seq_length =
      *std::max_element(seq_len_list.begin(), seq_len_list.end());

  memset(output, 0,
         batch_size * max_seq_length * num_attention_heads * width *
             sizeof(float));

#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * max_seq_length; ++idx) {
    int64_t batch_idx = idx / max_seq_length;
    int64_t seq_idx = idx % max_seq_length;
    if (seq_idx >= seq_len_list[batch_idx]) {
      continue;
    }

    int64_t acc_seq_len = std::accumulate(seq_len_list.begin(),
                                          seq_len_list.begin() + batch_idx, 0);
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const float* bias_ptr = bias + head_idx * width;
      auto* src = input +
                  (acc_seq_len + seq_idx) * num_attention_heads * width +
                  head_idx * width;
      auto* dst = output +
                  batch_idx * (num_attention_heads * max_seq_length * width) +
                  head_idx * max_seq_length * width + seq_idx * width;

#pragma omp simd
      for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
        dst[width_idx] = src[width_idx] + bias_ptr[width_idx];
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

// tranpose(2,3)
static void TransposeForScorePadImpl(float* output, const float* input,
                                     int64_t batch_size,           // 0
                                     int64_t max_seq_len,          // 2
                                     int64_t num_attention_heads,  // 1
                                     int64_t width,                // 3
                                     const std::vector<int64_t>& seq_len_list) {
#pragma omp parallel for
  for (int64_t idx = 0; idx < batch_size * max_seq_len; ++idx) {
    int64_t batch_idx = idx / max_seq_len;
    int64_t seq_idx = idx % max_seq_len;
    if (seq_idx >= seq_len_list[batch_idx]) {
      continue;
    }
    int64_t acc_seq_len = std::accumulate(seq_len_list.begin(),
                                          seq_len_list.begin() + batch_idx, 0);
    for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
      const float* src = input +
                         batch_idx * num_attention_heads * max_seq_len * width +
                         head_idx * max_seq_len * width + seq_idx * width;
      float* dst = output +
                   (acc_seq_len + seq_idx) * num_attention_heads * width +
                   head_idx * width;
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
  TT_ENFORCE_EQ(input.n_dim(), 4, "input should be a 4-D tensor");
  TT_ENFORCE_GE(output->n_dim(), 3,
                "output tensor dim should be greater than 3");
  TT_ENFORCE_EQ(input.numel(), output->numel(),
                "input.numel() and output.numel() should be the same");
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

// deprecated!
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

// forward transpose
// input_tensor: 4D array (batch_size, seq_length, 3, head_num * size_per_head)
void SplitAddBiasTransposeForScore(const core::Tensor& input_tensor,
                                   const core::Tensor& bias_tensor,
                                   core::Tensor& q_out_tensor,
                                   core::Tensor& k_out_tensor,
                                   core::Tensor& v_out_tensor,
                                   const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input_tensor.device_type());
#endif

  TT_ENFORCE_EQ(input_tensor.n_dim(), 4,
                "input_tensor should be (batch_size, seq_length, 3, "
                "num_attention_heads * size_per_head)");

  auto batch_size = input_tensor.shape(0);
  auto seq_length = input_tensor.shape(1);
  auto weight_num = 3;
  auto num_attention_heads = q_out_tensor.shape(1);
  auto width = input_tensor.shape(3) / num_attention_heads;
  auto input = input_tensor.data<float>();
  auto bias = bias_tensor.data<float>();
  auto q_out = q_out_tensor.mutableData<float>();
  auto k_out = k_out_tensor.mutableData<float>();
  auto v_out = v_out_tensor.mutableData<float>();

  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           bias_tensor.device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and bias_tensor "
                "should have the same device type and device id.");
  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           q_out_tensor.device_ctx()),
                true,
                "SplitAddBiasTransposeForScore: input_tensor and q_out_tensor "
                "should have the same device type and device id.");

  if (q_out_tensor.device_type() == kDLCPU &&
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
        float* dst_ptr = nullptr;
        switch (weight_idx) {
          case 0:
            dst_ptr = q_out +
                      batch_idx * (num_attention_heads * seq_length * width) +
                      head_idx * seq_length * width + seq_idx * width;
            break;
          case 1:
            dst_ptr = k_out +
                      batch_idx * (num_attention_heads * seq_length * width) +
                      head_idx * seq_length * width + seq_idx * width;
            break;
          case 2:
            dst_ptr = v_out +
                      batch_idx * (num_attention_heads * seq_length * width) +
                      head_idx * seq_length * width + seq_idx * width;
            break;
          default:
            break;
        }
        auto* bias_ptr =
            bias + weight_idx * width * num_attention_heads + head_idx * width;
#pragma omp simd
        for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
          dst_ptr[width_idx] = src_ptr[width_idx] + bias_ptr[width_idx];
        }
      }
    }  // end for
  } else if (q_out_tensor.device_type() == kDLGPU &&
             input_tensor.device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSplitAddBiasTransposeForScoreThreeOutput<float>(
        input, bias, batch_size, seq_length, weight_num, num_attention_heads,
        width, cuda_ctx.stream(), q_out, k_out, v_out);
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input_tensor.device_type());
#endif
}

// add bias and transpose(2,3) used for forward
void AddBiasTransposeForScorePad(const core::Tensor& input,
                                 const core::Tensor& bias, core::Tensor* output,
                                 const std::vector<int64_t>& seq_list,
                                 const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input.device_type());
#endif
  TT_ENFORCE_EQ(input.n_dim(), 4, "input should be a 4-D tensor");
  TT_ENFORCE_EQ(bias.numel(), input.shape(2) * input.shape(3),
                "bias shape %d should be %d x %d", bias.n_dim(), input.shape(2),
                input.shape(3));
  auto num_head = input.shape(2);
  auto hidden_size = input.shape(3);
  if (input.device_type() == kDLCPU && output->device_type() == kDLCPU) {
    AddBiasTransposeForScorePadImpl(input.data<float>(), bias.data<float>(),
                                    num_head, hidden_size,
                                    output->mutableData<float>(), seq_list);
  } else if (input.device_type() == kDLGPU && output->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasTransposeForScorePad<float>(
        input.data<float>(), bias.data<float>(), seq_list, num_head,
        hidden_size, cuda_ctx.stream(), output->mutableData<float>());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input.device_type());
#endif
}

/**
 * add bias, transpose as well as padding for a batch of variable length
 * requests. When this function is called, it is indicated that we use a
 * ByteDance smart padding method for batch inference. The batch is set as 1 and
 * seq_len is the summation of seq_list.
 *
 * @param input_tensor: 4D float tensor
 * `(1, sum(seq_len), 3, head_num * size_per_head)`
 * @param bias_tensor: float tensor of size `3 * head_num  * size_per_head`
 * @param q_out_tensor: batch, head_num, max(seq_len), size_per_head
 * @param k_out_tensor: batch, head_num, max(seq_len), size_per_head
 * @param v_out_tensor: batch, head_num, max(seq_len), size_per_head
 * @param seq_list : a batch of requests with variable seq_len for
 * example three request of sizes (4, 6, 3). sum(seq_len) = 4 + 6 + 3 = 13
 * batch_size = 3
 * @param name
 */
void SplitAddBiasTransposeForScorePad(const core::Tensor& input_tensor,
                                      const core::Tensor& bias_tensor,
                                      core::Tensor& q_out_tensor,
                                      core::Tensor& k_out_tensor,
                                      core::Tensor& v_out_tensor,
                                      const std::vector<int64_t>& seq_list,
                                      const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input_tensor.device_type());
#endif

  TT_ENFORCE_EQ(
      input_tensor.n_dim(), 4,
      "input_tensor should be 4D tensor of size (1, sum(seq_list), 3, "
      "num_attention_heads * size_per_head)");

  auto weight_num = 3;
  auto num_attention_heads = q_out_tensor.shape(1);
  auto width = input_tensor.shape(3) / num_attention_heads;
  auto input = input_tensor.data<float>();
  auto bias = bias_tensor.data<float>();
  auto q_out = q_out_tensor.mutableData<float>();
  auto k_out = k_out_tensor.mutableData<float>();
  auto v_out = v_out_tensor.mutableData<float>();

  int64_t out_batch_size = static_cast<int64_t>(seq_list.size());
  int64_t out_max_seq_length =
      *std::max_element(seq_list.begin(), seq_list.end());
  TT_ENFORCE_EQ(
      common::is_same_device_ctx(input_tensor.device_ctx(),
                                 bias_tensor.device_ctx()),
      true,
      "SplitAddBiasTransposeForScorePad: input_tensor and bias_tensor "
      "should have the same device type and device id.");
  TT_ENFORCE_EQ(
      common::is_same_device_ctx(input_tensor.device_ctx(),
                                 q_out_tensor.device_ctx()),
      true,
      "SplitAddBiasTransposeForScorePad: input_tensor and q_out_tensor "
      "should have the same device type and device id.");

  if (q_out_tensor.device_type() == kDLCPU &&
      input_tensor.device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU) {
    memset(q_out, 0,
           out_batch_size * out_max_seq_length * num_attention_heads * width *
               sizeof(float));
    memset(k_out, 0,
           out_batch_size * out_max_seq_length * num_attention_heads * width *
               sizeof(float));
    memset(v_out, 0,
           out_batch_size * out_max_seq_length * num_attention_heads * width *
               sizeof(float));

#pragma omp parallel for
    for (int64_t idx = 0;
         idx < out_batch_size * weight_num * out_max_seq_length; ++idx) {
      auto batch_idx = idx / (out_max_seq_length * weight_num);
      auto seq_idx = idx / weight_num % out_max_seq_length;

      if (seq_idx >= seq_list[batch_idx]) {
        continue;
      }

      auto weight_idx = idx % weight_num;
      int64_t acc_seq_length =
          std::accumulate(seq_list.begin(), seq_list.begin() + batch_idx, 0);

      for (int64_t head_idx = 0; head_idx < num_attention_heads; ++head_idx) {
        auto* src_ptr = input +
                        (acc_seq_length + seq_idx) *
                            (weight_num * num_attention_heads * width) +
                        weight_idx * (num_attention_heads * width) +
                        head_idx * width;
        float* dst_ptr = nullptr;
        switch (weight_idx) {
          case 0:
            dst_ptr =
                q_out +
                batch_idx * (num_attention_heads * out_max_seq_length * width) +
                head_idx * out_max_seq_length * width + seq_idx * width;
            break;
          case 1:
            dst_ptr =
                k_out +
                batch_idx * (num_attention_heads * out_max_seq_length * width) +
                head_idx * out_max_seq_length * width + seq_idx * width;
            break;
          case 2:
            dst_ptr =
                v_out +
                batch_idx * (num_attention_heads * out_max_seq_length * width) +
                head_idx * out_max_seq_length * width + seq_idx * width;
            break;
          default:
            break;
        }
        auto* bias_ptr =
            bias + weight_idx * width * num_attention_heads + head_idx * width;
#pragma omp simd
        for (int64_t width_idx = 0; width_idx < width; ++width_idx) {
          dst_ptr[width_idx] = src_ptr[width_idx] + bias_ptr[width_idx];
        }
      }
    }  // end for
  } else if (q_out_tensor.device_type() == kDLGPU &&
             input_tensor.device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUSplitAddBiasTransposeForScoreThreeOutputPad<float>(
        input, bias, seq_list, weight_num, num_attention_heads, width,
        cuda_ctx.stream(), q_out, k_out, v_out);
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input_tensor.device_type());
#endif
}

/***
 * Tranpose the input tensor back
 * @param output 3D/4D tensor (1, sum_seq_len, head_num * model_dim)
 * @param input 4D (batch, head_num, max_seq_len, model_dim)
 * @param seq_list: list contains seq_len of each request
 * @param name
 */
void TransposeForScorePad(core::Tensor* output, const core::Tensor& input,
                          const std::vector<int64_t>& seq_list,
                          const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input.device_type());
#endif
  TT_ENFORCE_EQ(input.n_dim(), 4,
                "input should be a 4-D tensor, size (batch, num_heads, "
                "max_seq_len, width)");
  TT_ENFORCE_GE(output->n_dim(), 3,
                "output tensor dim should be greater than 3, size (1, "
                "sum_seq_len, num_heads * width)");
  if (input.device_type() == kDLCPU && output->device_type() == kDLCPU) {
    TransposeForScorePadImpl(output->mutableData<float>(), input.data<float>(),
                             input.shape(0), input.shape(2), input.shape(1),
                             input.shape(3), seq_list);
  } else if (input.device_type() == kDLGPU && output->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto batch_size = input.shape(0);
    // auto max_seq_len = input.shape(2);
    auto num_attention_heads = input.shape(1);
    auto width = input.shape(3);
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUTransposeForScorePad<float>(
        input.data<float>(), batch_size, seq_list, num_attention_heads, width,
        cuda_ctx.stream(), output->mutableData<float>());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input.device_type());
#endif
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
