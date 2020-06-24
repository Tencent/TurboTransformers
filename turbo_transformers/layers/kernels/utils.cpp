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

#include "utils.h"

#include "common.h"
#ifdef TT_WITH_CUDA
#include <cuda.h>

#include "turbo_transformers/core/cuda_device_context.h"
#include "turbo_transformers/core/cuda_enforce.cuh"
#include "turbo_transformers/layers/kernels/gpu_utils.h"
#endif
#ifdef WITH_PERFTOOLS
#include "turbo_transformers/core/profiler.h"
#endif

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
void Concat(const core::Tensor& t1, const core::Tensor& t2, size_t dim,
            core::Tensor* output, const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, t1.device_type());
#endif
  TT_ENFORCE(t1.n_dim() >= dim && t2.n_dim() >= dim,
             "concatation of two tensors with dim as %d and %d is illegal.",
             t1.n_dim(), t2.n_dim());

  auto t1_size = t1.shape(dim);
  auto t2_size = t2.shape(dim);

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < t1.n_dim(); i++) {
    if (i != dim) {
      TT_ENFORCE(
          t1.shape(i) == t2.shape(i),
          "concatation of two tensors illegal, at dim %d size is %d vs %d", i,
          t1.shape(i), t2.shape(i));
      output_shape.push_back(t1.shape(i));
    } else {
      output_shape.push_back(t1_size + t2_size);
    }
  }

  int64_t high_dim = 1;
  for (size_t i = 0; i < dim; i++) {
    high_dim *= t1.shape(i);
  }

  size_t low_dim = 1;
  for (size_t i = t1.n_dim() - 1; i > dim; i--) {
    low_dim *= t2.shape(i);
  }

  output->Reshape<T>(output_shape, t1.device_type(), t1.device_id(),
                     "Concat/Reshape");
  if (t1.device_type() == kDLGPU) {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUConcat<T>(t1.data<T>(), t2.data<T>(), high_dim, t1_size, t2_size,
                 low_dim, cuda_ctx.stream(), output->mutableData<T>());
#endif
  } else if (t1.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < high_dim; ++i) {
      for (int64_t j = 0; j < t1_size; ++j) {
        core::Copy(
            t1.data<T>() + (i * t1_size + j) * low_dim, low_dim,
            t1.device_type(), output->device_type(),
            output->mutableData<T>() + (i * (t1_size + t2_size) + j) * low_dim);
      }
      for (int64_t j = 0; j < t2_size; ++j) {
        core::Copy(t2.data<T>() + (i * t2_size + j) * low_dim, low_dim,
                   t1.device_type(), output->device_type(),
                   output->mutableData<T>() +
                       (i * (t1_size + t2_size) + t1_size + j) * low_dim);
      }
    }
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, t1.device_type());
#endif
}

template void Concat<float>(const core::Tensor& t1, const core::Tensor& t2,
                            size_t dim, core::Tensor* output,
                            const std::string name);

void AddBias(const core::Tensor& bias, core::Tensor* output,
             const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, bias.device_type());
#endif
  auto dim1 = bias.shape(0);
  auto dim0 = output->numel() / dim1;
  auto output_data = output->mutableData<float>();
  const auto bias_data = bias.data<float>();
  if (bias.device_type() == kDLCPU && output->device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < dim0; ++i) {
#pragma omp simd
      for (int64_t j = 0; j < dim1; ++j) {
        output_data[i * dim1 + j] += bias_data[j];
      }
    }
  } else {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    const float* dummy{nullptr};
    kernels::GPUAddBias<false>(output_data, dummy, bias_data, dim0, dim1,
                               cuda_ctx.stream(), output_data);
#endif
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, bias.device_type());
#endif
}

void AddInputBias(const core::Tensor& input1, const core::Tensor& input2,
                  const core::Tensor& bias, core::Tensor* output,
                  const std::string name) {
#ifdef WITH_PERFTOOLS
  auto& profile_ctx = core::Profiler::GetInstance();
  profile_ctx.start_profile(name, input1.device_type());
#endif
  TT_ENFORCE_EQ(input1.numel(), input2.numel(),
                "Tensor input1 and Tensor input2 should have the same numel.");
  auto dim1 = bias.shape(0);
  auto dim0 = output->numel() / dim1;
  auto output_data = output->mutableData<float>();
  const auto bias_data = bias.data<float>();
  const auto input1_data = input1.data<float>();
  const auto input2_data = input2.data<float>();

  if (input1.device_type() == kDLCPU && output->device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < dim0; ++i) {
#pragma omp simd
      for (int64_t j = 0; j < dim1; ++j) {
        output_data[i * dim1 + j] = bias_data[j] + input1_data[i * dim1 + j] +
                                    input2_data[i * dim1 + j];
      }
    }
  } else {
#ifdef TT_WITH_CUDA
    core::CUDADeviceContext& cuda_ctx = core::CUDADeviceContext::GetInstance();
    GPUAddBias<true>(input1_data, input2_data, bias_data, dim0, dim1,
                     cuda_ctx.stream(), output_data);
#endif
  }
#ifdef WITH_PERFTOOLS
  profile_ctx.end_profile(name, input1.device_type());
#endif
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
