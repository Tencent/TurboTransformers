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
#include "turbo_transformers/layers/kernels/activation.h"

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_activation_kernel.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

namespace {
template <typename T, ActivationType ActType>
void CPUAddBiasActKernel(const T *bias, int64_t batch_size, int64_t feature_dim,
                         T *out);

template <>
void CPUAddBiasActKernel<float, ActivationType::Gelu>(const float *bias,
                                                      int64_t batch_size,
                                                      int64_t feature_dim,
                                                      float *out) {
  core::Tensor temp_tensor(nullptr);
  auto *buff =
      temp_tensor.Reshape<float>({batch_size * feature_dim}, kDLCPU, 0);
#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      float tmp_ = out[j] + bias[k++];
      buff[j] = (0.7978845608028654f * (tmp_ + 0.044715f * tmp_ * tmp_ * tmp_));
    }
    vsTanh(feature_dim, &buff[i * feature_dim], &buff[i * feature_dim]);
    k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      out[j] = (out[j] + bias[k++]) * 0.5f * (1.0f + buff[j]);
    }
  }
}

template <>
void CPUAddBiasActKernel<float, ActivationType::Tanh>(const float *bias,
                                                      int64_t batch_size,
                                                      int64_t feature_dim,
                                                      float *out) {
#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      out[j] = out[j] + bias[k++];
    }
    vsTanh(feature_dim, &out[i * feature_dim], &out[i * feature_dim]);
  }
}

template <>
void CPUAddBiasActKernel<float, ActivationType::Relu>(const float *bias,
                                                      int64_t batch_size,
                                                      int64_t feature_dim,
                                                      float *out) {
#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t k = 0;
#pragma omp simd
    for (int64_t j = feature_dim * i; j < feature_dim * (i + 1); ++j) {
      out[j] = out[j] + bias[k++];
      out[j] = out[j] > 0. ? out[j] : 0.;
    }
  }
}
}  // namespace

template <typename T, ActivationType ActType>
void AddBiasAct(const core::Tensor &bias_tensor, core::Tensor *out_tensor,
                const std::string name) {
#ifdef WITH_PERFTOOLS
  auto &profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, bias_tensor.device_type());
#endif
  auto *out = out_tensor->mutableData<T>();
  auto *bias = bias_tensor.data<T>();

  int64_t n = out_tensor->shape(-1);
  int64_t m = out_tensor->numel() / n;

  if (out_tensor->device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU) {
    CPUAddBiasActKernel<T, ActType>(bias, m, n, out);
  } else if (out_tensor->device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasActKernel<T, ActType>(bias, m, n, cuda_ctx.stream(), out);
#endif
  } else {
    TT_THROW("device_type %d is not supported for AddBiasAct",
             out_tensor->device_type());
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, bias_tensor.device_type());
#endif
}

template void AddBiasAct<float, ActivationType::Tanh>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor,
    const std::string name);

template void AddBiasAct<float, ActivationType::Gelu>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor,
    const std::string name);

template void AddBiasAct<float, ActivationType::Relu>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor,
    const std::string name);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
