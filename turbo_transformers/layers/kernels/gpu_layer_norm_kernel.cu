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

#include <cuda_runtime.h>
#include <immintrin.h>

#include <numeric>

#include "turbo_transformers/layers/kernels/gpu_block_reduce.h"
#include "turbo_transformers/layers/kernels/gpu_layer_norm_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool AddBias>
static __global__ void layer_norm_kernel(float* out, const float* input,
                                         const float* bias, const float* gamma,
                                         const float* beta, int m, int n) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * n + tid;
  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out = 0.0f;
  if (AddBias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 300
    local_out = out[offset] + input[offset] + __ldg(&bias[tid]);
#else
    local_out = out[offset] + input[offset] + bias[tid];
#endif
  } else {
    local_out = out[offset];
  }

  float sum_list[2] = {local_out, local_out * local_out};
  blockReduce<ReduceType, ReduceType::kSum, 2>(sum_list);

  if (tid == 0) {
    float mean = sum_list[0] / n;
    float mean_2 = sum_list[1] / n;
    s_mean = mean;
    s_variance = rsqrtf(mean_2 - mean * mean + 1e-6f);
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 300
  out[offset] = (local_out - s_mean) * s_variance * __ldg(&gamma[tid]) +
                __ldg(&beta[tid]);
#else
  out[offset] = (local_out - s_mean) * s_variance * gamma[tid] + beta[tid];
#endif
}

template <bool AddBias, typename T>
void GPULayerNorm(T* out, const T* input, const T* bias, const T* gamma,
                  const T* beta, int m, int n, cudaStream_t stream) {
  dim3 block(n);
  if (block.x > 1024) {
    throw std::runtime_error("GPULayerNorm thread block size large than 1024");
  }
  dim3 grid(m);
  layer_norm_kernel<AddBias>
      <<<grid, block, 0, stream>>>(out, input, bias, gamma, beta, m, n);
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
