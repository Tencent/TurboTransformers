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

#include "turbo_transformers/layers/kernels/layer_norm.h"

#include "common.h"
#include "turbo_transformers/layers/kernels/common.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/common.h"
#include "turbo_transformers/layers/kernels/gpu_layer_norm_kernel.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {
// static constexpr float g_epsilon = 1e-12;

template <typename T>
void LayerNorm(const core::Tensor& gamma, const core::Tensor& beta,
               core::Tensor* out_tensor, T eps, const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, out_tensor->device_type());
#endif
  TT_ENFORCE_EQ(
      common::is_same_device_ctx(gamma.device_ctx(), beta.device_ctx()), true,
      "LayerNorm gamma and beta must be on the same device context.");

  auto feature_dim = out_tensor->shape(out_tensor->n_dim() - 1);
  int64_t batch_size = std::accumulate(
      &out_tensor->shape(0), &out_tensor->shape(0) + out_tensor->n_dim() - 1, 1,
      std::multiplies<int64_t>());

  auto out = out_tensor->mutableData<T>();
  const auto gamma_ptr = gamma.data<T>();
  const auto beta_ptr = beta.data<T>();

  if (out_tensor->device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      T mean = 0;
      T var = 0;
      auto start_idx = batch_idx * feature_dim;
      auto end_idx = (batch_idx + 1) * feature_dim;
#pragma omp simd reduction(+ : mean, var)
      for (int64_t i = start_idx; i < end_idx; i++) {
        mean += out[i];
        var += out[i] * out[i];
      }
      mean = mean / feature_dim;
      var = var / feature_dim - mean * mean;

      var = 1.f / sqrtf(var + eps);

#pragma omp simd
      for (int64_t i = 0; i < feature_dim; ++i) {
        int64_t j = batch_idx * feature_dim + i;
        out[j] = beta_ptr[i] + gamma_ptr[i] * var * (out[j] - mean);
      }
    }
  } else if (out_tensor->device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    auto& cuda_ctx = core::CUDADeviceContext::GetInstance();
    T* bias = nullptr;
    GPULayerNorm</*AddBias*/ false>(out, out, bias, gamma_ptr, beta_ptr,
                                    batch_size, feature_dim, cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("AddBiasLayerNorm device_type %d is not supported",
             out_tensor->device_type());
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, out_tensor->device_type());
#endif
}

template void LayerNorm<float>(const core::Tensor& gamma,
                               const core::Tensor& beta,
                               core::Tensor* out_tensor, float eps,
                               const std::string name);

template <typename T>
void AddBiasLayerNorm(const core::Tensor& input_tensor,
                      const core::Tensor& bias_tensor,
                      const core::Tensor& gamma_tensor,
                      const core::Tensor& beta_tensor, core::Tensor* out_tensor,
                      T eps, const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input_tensor.device_type());
#endif
  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           bias_tensor.device_ctx()),
                true,
                "AddBiasLayerNorm input_tensor and bias_tensor"
                "should have the same device context.");
  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           gamma_tensor.device_ctx()),
                true,
                "AddBiasLayerNorm input_tensor and gamma_tensor"
                "should have the same device context.");
  TT_ENFORCE_EQ(common::is_same_device_ctx(input_tensor.device_ctx(),
                                           beta_tensor.device_ctx()),
                true,
                "AddBiasLayerNorm input_tensor and beta_tensor"
                "should have the same device context.");

  T* out = out_tensor->mutableData<T>();
  const T* input = input_tensor.data<T>();
  const T* bias = bias_tensor.data<T>();
  const T* gamma = gamma_tensor.data<T>();
  const T* beta = beta_tensor.data<T>();

  int64_t n = input_tensor.shape(-1);
  int64_t m = input_tensor.numel() / n;
  // TODO(florianzhao): Check the dim of bias_tensor, gamma_tensor, beta_tensor,
  // out_tensor
  TT_ENFORCE_EQ(input_tensor.shape(-1), gamma_tensor.shape(0),
                "AddBiasLayerNorm input_tensor.shape(-1) should be the same as gamma_tensor.shape(0)");
  TT_ENFORCE_EQ(input_tensor.shape(-1), beta_tensor.shape(0),
                "AddBiasLayerNorm input_tensor.shape(-1) should be the same as beta_tensor.shape(0)");
  TT_ENFORCE_EQ(input_tensor.shape(-1), bias_tensor.shape(0),
                "AddBiasLayerNorm input_tensor.shape(-1) should be the same as bias_tensor.shape(0)");
  TT_ENFORCE(common::is_same_shape(input_tensor, *out_tensor),
             "The shape of the input_tensor and out_tensor is not equal.");
  if (input_tensor.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t batch_idx = 0; batch_idx < m; ++batch_idx) {
      T mean = 0;
      T var = 0;
#pragma omp simd reduction(+ : mean, var)
      for (int64_t i = batch_idx * n; i < (batch_idx + 1) * n; i++) {
        int64_t j = i - batch_idx * n;
        T t = out[i] = out[i] + input[i] + bias[j];
        mean += t;
        var += t * t;
      }
      mean = mean / n;
      var = var / n - mean * mean;

      var = 1.f / sqrtf(var + eps);

#pragma omp simd
      for (int64_t i = 0; i < n; ++i) {
        int64_t j = batch_idx * n + i;
        out[j] = beta[i] + gamma[i] * var * (out[j] - mean);
      }
    }
  } else if (input_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPULayerNorm</*AddBias*/ true>(out, input, bias, gamma, beta, m, n,
                                   cuda_ctx.stream());
#else
    TT_THROW("The current code is not compiled with CUDA.");
#endif
  } else {
    TT_THROW("LayerNorm device_type %d is not supported",
             input_tensor.device_type());
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input_tensor.device_type());
#endif
}

template void AddBiasLayerNorm<float>(const core::Tensor& input_tensor,
                                      const core::Tensor& bias_tensor,
                                      const core::Tensor& gamma_tensor,
                                      const core::Tensor& beta_tensor,
                                      core::Tensor* out_tensor, float eps,
                                      const std::string name);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
