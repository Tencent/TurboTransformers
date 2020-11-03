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
#include <stdexcept>

#include "turbo_transformers/layers/kernels/gpu_embedding_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool IsAdd>
static __global__ void lookup(float* dst, const float* embedding_table,
                              const int64_t* ids, int64_t vocab_size) {
  int64_t id = ids[blockIdx.x];
  int hidden_idx = threadIdx.x;
  int hidden_size = blockDim.x;
  // TODO(jiaruifang): There should have a checker to check the range of id.
  if (id >= vocab_size) {
    asm("trap;");
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 300
  float val = __ldg(&embedding_table[id * hidden_size + hidden_idx]);
#else
  float val = embedding_table[id * hidden_size + hidden_idx];
#endif
  if (IsAdd) {
    dst[blockIdx.x * hidden_size + hidden_idx] += val;
  } else {
    dst[blockIdx.x * hidden_size + hidden_idx] = val;
  }
}

template <bool Add>
void GPULookupKernel(float* dst, const float* embedding_table,
                     const int64_t* ids, int64_t vocab_size,
                     int64_t hidden_size, int64_t num_ids,
                     cudaStream_t stream) {
  dim3 grid(num_ids);
  dim3 block(hidden_size);
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPULookupKernel currently does not support a hidden_size larger than "
        "1024");
  }
  lookup<Add>
      <<<grid, block, 0, stream>>>(dst, embedding_table, ids, vocab_size);
}

template void GPULookupKernel<true>(float* dst, const float* embedding_table,
                                    const int64_t* ids, int64_t vocab_size,
                                    int64_t hidden_size, int64_t num_ids,
                                    cudaStream_t stream);
template void GPULookupKernel<false>(float* dst, const float* embedding_table,
                                     const int64_t* ids, int64_t vocab_size,
                                     int64_t hidden_size, int64_t num_ids,
                                     cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
