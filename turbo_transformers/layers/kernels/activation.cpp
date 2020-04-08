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

#ifdef TT_WITH_CUDA
#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/layers/kernels/gpu_activation_kernel.h"
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
void CPUAddBiasActKernel<core::Half, ActivationType::Tanh>(
    const core::Half *bias, int64_t batch_size, int64_t feature_dim,
    core::Half *out) {
  TT_THROW("CPUAddBiasActKernel Tanh FP16 is not supported");
}

template <>
void CPUAddBiasActKernel<core::Half, ActivationType::Gelu>(
    const core::Half *bias, int64_t batch_size, int64_t feature_dim,
    core::Half *out) {
  TT_THROW("CPUAddBiasActKernel Gelu FP16 is not supported");
}

template <typename T, DLDeviceType deviceType>
struct ActivationTypeTrait {
  typedef T Type;
};

template <>
struct ActivationTypeTrait<core::Half, kDLGPU> {
  typedef half Type;
};

}  // namespace

template <typename T, ActivationType ActType>
void AddBiasAct(const core::Tensor &bias_tensor, core::Tensor *out_tensor) {
  auto *out = out_tensor->mutableData<T>();
  auto *bias = bias_tensor.data<T>();

  int64_t m = out_tensor->rows();
  int64_t n = out_tensor->cols();

  if (out_tensor->device_type() == kDLCPU &&
      bias_tensor.device_type() == kDLCPU) {
    CPUAddBiasActKernel<typename ActivationTypeTrait<T, kDLCPU>::Type, ActType>(
        reinterpret_cast<const typename ActivationTypeTrait<T, kDLCPU>::Type *>(
            bias),
        m, n,
        reinterpret_cast<typename ActivationTypeTrait<T, kDLCPU>::Type *>(out));
  } else if (out_tensor->device_type() == kDLGPU &&
             bias_tensor.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext &cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBiasActKernel<typename ActivationTypeTrait<T, kDLGPU>::Type, ActType>(
        reinterpret_cast<const typename ActivationTypeTrait<T, kDLGPU>::Type *>(
            bias),
        m, n, cuda_ctx.stream(),
        reinterpret_cast<typename ActivationTypeTrait<T, kDLGPU>::Type *>(out));
#endif
  } else {
    TT_THROW("device_type %d is not supported for AddBiasAct",
             out_tensor->device_type());
  }
}

template void AddBiasAct<float, ActivationType::Tanh>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor);

template void AddBiasAct<float, ActivationType::Gelu>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor);

template void AddBiasAct<core::Half, ActivationType::Tanh>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor);

template void AddBiasAct<core::Half, ActivationType::Gelu>(
    const core::Tensor &bias_tensor, core::Tensor *out_tensor);
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
