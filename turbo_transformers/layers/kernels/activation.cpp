// Copyright 2020 Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "turbo_transformers/layers/kernels/activation.h"

#include "turbo_transformers/core/aligned_scratchpad.h"
#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_activation_kernel.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
static void CPUAddBiasGeLUActKernel(const T *bias, T *out, int64_t batch_size,
                                    int64_t feature_dim) {
  static core::AlignedScratchpad<float> scratchpad;
  float *buff = scratchpad.mutable_data(batch_size * feature_dim);
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

template <typename T>
static void CPUAddBiasTanhActKernel(const T *bias, T *out, int64_t batch_size,
                                    int64_t feature_dim) {
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

static void AddBiasAndGeluFunc(const DLDeviceType &dev_type, const float *bias,
                               int64_t m, int64_t n, float *out) {
  if (dev_type == kDLCPU) {
    CPUAddBiasGeLUActKernel(bias, out, m, n);
  } else if (dev_type == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasActKernel<float, ActivationType::Gelu>(bias, out, m, n,
                                                     cuda_ctx.stream());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}

static void AddBiasAndTanhFunc(const DLDeviceType &dev_type, const float *bias,
                               int64_t m, int64_t n, float *out) {
  if (dev_type == kDLCPU) {
    CPUAddBiasTanhActKernel(bias, out, m, n);
  } else if (dev_type == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasActKernel<float, ActivationType::Tanh>(bias, out, m, n,
                                                     cuda_ctx.stream());
#endif
  } else {
    TT_THROW("device_type is not supported");
  }
}

template <>
void AddBiasAct<float, ActivationType::Gelu>(const core::Tensor &bias_tensor,
                                             core::Tensor *out_tensor) {
  auto *out = out_tensor->mutableData<float>();
  auto *bias = bias_tensor.data<float>();

  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();

  AddBiasAndGeluFunc(out_tensor->device_type(), bias, m, n, out);
}

template <>
void AddBiasAct<float, ActivationType::Tanh>(const core::Tensor &bias_tensor,
                                             core::Tensor *out_tensor) {
  auto *out = out_tensor->mutableData<float>();
  auto *bias = bias_tensor.data<float>();

  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();

  AddBiasAndTanhFunc(out_tensor->device_type(), bias, m, n, out);
}

#ifdef TT_WITH_CUDA
template <>
void AddBiasAct<core::Half, ActivationType::Gelu>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor) {
  TT_ENFORCE_EQ(bias_tensor.device_type(), kDLGPU, "The device should be GPU.");
  core::Half *out = out_tensor->mutableData<core::Half>();
  const core::Half *bias = bias_tensor.data<core::Half>();
  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();
  core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
  GPUAddBiasActKernel<half, ActivationType::Gelu>(
      reinterpret_cast<const half *>(bias), reinterpret_cast<half *>(out), m, n,
      cuda_ctx.stream());
}

template <>
void AddBiasAct<core::Half, ActivationType::Tanh>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor) {
  TT_ENFORCE_EQ(bias_tensor.device_type(), kDLGPU, "The device should be GPU.");
  core::Half *out = out_tensor->mutableData<core::Half>();
  const core::Half *bias = bias_tensor.data<core::Half>();
  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();
  core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
  GPUAddBiasActKernel<half, ActivationType::Tanh>(
      reinterpret_cast<const half *>(bias), reinterpret_cast<half *>(out), m, n,
      cuda_ctx.stream());
}
#endif
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
