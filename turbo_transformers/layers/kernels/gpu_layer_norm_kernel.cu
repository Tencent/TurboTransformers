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

#include <cuda_runtime.h>

#include <numeric>

#include "turbo_transformers/layers/kernels/gpu_block_reduce.cuh"
#include "turbo_transformers/layers/kernels/gpu_layer_norm_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool AddBias>
static __global__ void layer_norm_kernel_32x_le_1024(
    float* out, const float* input, const float* bias, const float* gamma,
    const float* beta, int m, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * n + tid;
  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = 0.0f;
  if (AddBias) {
    local_out = out[offset] + input[offset] + bias[tid];
  } else {
    local_out = out[offset];
  }

  float sum_list[2] = {local_out, local_out * local_out};
  blockReduce<ReduceType::kSum, 2>(sum_list);

  if (tid == 0) {
    float mean = sum_list[0] / n;
    float mean_2 = sum_list[1] / n;
    s_mean = mean;
    s_variance = rsqrtf(mean_2 - mean * mean + 1e-6f);
  }
  __syncthreads();
  out[offset] = (local_out - s_mean) * s_variance * gamma[tid] + beta[tid];
}

// TODO(jiaruifang) if the lowese dimension is not 32x and <= 1024,
// implementation is not optimized
template <bool AddBias>
static __global__ void layer_norm_kernel(float* out, const float* input,
                                         const float* bias, const float* gamma,
                                         const float* beta, int m, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * n;
  int block_dim_x = blockDim.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  // local reduce
  int idx = tid;
  float local_sum_out = 0.;
  float local_sum_square_out = 0.;

  if (AddBias) {
    idx = tid;
    while (idx < n) {
      float tmp = (bias[idx] + input[idx + offset] + out[idx + offset]);
      local_sum_out += tmp;
      local_sum_square_out += tmp * tmp;
      idx += block_dim_x;
    }
  } else {
    idx = tid;
    while (idx < n) {
      float tmp = out[idx + offset];
      local_sum_out += tmp;
      local_sum_square_out += tmp * tmp;
      idx += block_dim_x;
    }
  }

  // remote reduce
  float sum_list[2] = {local_sum_out, local_sum_square_out};

  blockReduce<ReduceType::kSum, 2>(sum_list);

  if (tid == 0) {
    float mean = sum_list[0] / n;
    float mean_2 = sum_list[1] / n;
    s_mean = mean;
    s_variance = rsqrtf(mean_2 - mean * mean + 1e-6f);
  }
  __syncthreads();

  idx = tid;
  if (AddBias) {
    while (idx < n) {
      out[idx + offset] =
          (out[idx + offset] + input[idx + offset] + bias[idx] - s_mean) *
              s_variance * gamma[idx] +
          beta[idx];
      idx += block_dim_x;
    }
  } else {
    while (idx < n) {
      out[idx + offset] =
          (out[idx + offset] - s_mean) * s_variance * gamma[idx] + beta[idx];
      idx += block_dim_x;
    }
  }
}

template <bool AddBias, typename T>
void GPULayerNorm(T* out, const T* input, const T* bias, const T* gamma,
                  const T* beta, int m, int n, cudaStream_t stream) {
  dim3 grid(m);
  if (n <= 1024 && n % 32 == 0) {
    dim3 block(n);
    layer_norm_kernel_32x_le_1024<AddBias>
        <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
  } else {
    int block_size = min(1024, (int)((n + 31) / 32 * 32));
    dim3 block(block_size);
    layer_norm_kernel<AddBias>
        <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
  }
}

template void GPULayerNorm<true>(float* out, const float* input,
                                 const float* bias, const float* gamma,
                                 const float* beta, int m, int n,
                                 cudaStream_t stream);
template void GPULayerNorm<false>(float* out, const float* input,
                                  const float* bias, const float* gamma,
                                  const float* beta, int m, int n,
                                  cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
