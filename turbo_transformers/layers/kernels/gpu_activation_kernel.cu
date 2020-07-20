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

#include <numeric>

#include "ide_macro.h"
#include "turbo_transformers/layers/kernels/gpu_activation_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
namespace {

template <typename T, ActivationType ActType>
__inline__ __device__ T ActivationOp(const T& x);

template <>
__inline__ __device__ float ActivationOp<float, ActivationType::Gelu>(
    const float& x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

template <>
__inline__ __device__ float ActivationOp<float, ActivationType::Tanh>(
    const float& x) {
  return tanhf(x);
}

template <>
__inline__ __device__ float ActivationOp<float, ActivationType::Relu>(
    const float& x) {
  return (x > 0) ? x : 0;
}

}  // namespace

template <typename T, ActivationType ActType>
static __global__ void add_bias_act(const T* bias, int batch_size,
                                    int feature_dim, T* out) {
  T val, reg_bias;

  int row_id;
  int elem_per_thread = (feature_dim + blockDim.x - 1) / blockDim.x;
  int tid = threadIdx.x;

  for (int i = 0; i < elem_per_thread; ++i) {
    int offset = i * blockDim.x + tid;
    if (offset < feature_dim) {
      reg_bias = bias[offset];
      row_id = blockIdx.x;
      val = out[offset + row_id * feature_dim] + reg_bias;
      out[offset + row_id * feature_dim] = ActivationOp<T, ActType>(val);
    }
  }
}

template <typename T, ActivationType ActType>
void GPUAddBiasActKernel(const T* bias_data, int64_t batch_size,
                         int64_t feature_dim, cudaStream_t stream,
                         T* out_data) {
  dim3 grid(batch_size);
  int block_size = min(1024, (int)(feature_dim / 4));
  dim3 block(block_size);
  add_bias_act<T, ActType><<<grid, block, 0, stream>>>(bias_data, batch_size,
                                                       feature_dim, out_data);
}

template void GPUAddBiasActKernel<float, ActivationType::Gelu>(
    const float* bias_data, int64_t batch_size, int64_t feature_dim,
    cudaStream_t stream, float* out_data);

template void GPUAddBiasActKernel<float, ActivationType::Tanh>(
    const float* bias_data, int64_t batch_size, int64_t feature_dim,
    cudaStream_t stream, float* out_data);

template void GPUAddBiasActKernel<float, ActivationType::Relu>(
    const float* bias_data, int64_t batch_size, int64_t feature_dim,
    cudaStream_t stream, float* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
