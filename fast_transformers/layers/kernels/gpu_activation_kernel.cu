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

#include <numeric>

#include "fast_transformers/layers/kernels/gpu_activation_kernel.h"
#include "ide_macro.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

static __inline__ __device__ float gelu(float x) {
  float cdf =
      0.5f *
      (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x))));
  return x * cdf;
}

static __global__ void add_bias_act(float* out, const float* bias,
                                    int batch_size, int feature_dim) {
  float val, reg_bias;

  int row_id = blockIdx.x;
  int ite = feature_dim / blockDim.x;
  int tid = threadIdx.x;

  for (int i = 0; i < ite; ++i) {
    reg_bias = __ldg(&bias[i * blockDim.x + tid]);
    row_id = blockIdx.x;

    while (row_id < batch_size) {
      val = out[tid + i * blockDim.x + row_id * feature_dim] + reg_bias;
      out[tid + i * blockDim.x + row_id * feature_dim] = gelu(val);
      row_id += gridDim.x;
    }
  }
}
template <>
void GPUAddBiasGeLUActKernel(const float* bias_data, float* out_data,
                             int64_t batch_size, int64_t feature_dim,
                             cudaStream_t stream) {
  dim3 grid(batch_size / 4);
  dim3 block(feature_dim / 4);
  if (feature_dim / 4 > 1024) {
    throw std::runtime_error(
        "GPUAddBiasGeLUActKernel thread block size large than 1024");
  }
  add_bias_act<<<grid, block, 0, stream>>>(out_data, bias_data, batch_size,
                                           feature_dim);
}

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
