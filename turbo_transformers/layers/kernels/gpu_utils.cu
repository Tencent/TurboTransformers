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

#include "turbo_transformers/layers/kernels/gpu_utils.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace turbo_transformers {
namespace layers {
namespace kernels {

#define max(a, b) ((a) > (b)) ? (a) : (b)

template <typename T, t>
static __inline__ T __device__ ReduceOp(const T* target, int start_idx,
                                        int stride, int len);

template <>
static __inline__ float __device__
ReduceOp<float, layers::types::PoolType::kMax>(const float* input,
                                               int start_idx, int stride,
                                               int len) {
  float res = input[start_idx];
  for (int k = 1; k < len; ++k) {
    res = max(res, input[start_idx + stride * k]);
  }
  return res;
}

template <>
static __inline__ float __device__
ReduceOp<float, layers::types::PoolType::kMean>(const float* input,
                                                int start_idx, int stride,
                                                int len) {
  float res = input[start_idx];
  for (int k = 1; k < len; ++k) {
    res += input[start_idx + stride * k];
  }
  return res / len;
}

//[batch, seq_len, hidden_size] -> [batch, hidden_size]
template <typename T, layers::types::PoolType t>
static __global__ void ReduceAixsOne(const T* input, T* output, int batch_size,
                                     int seq_len, int hidden_size) {
  int tid = threadIdx.x;  // hidden_size idx
  int gid = blockIdx.x;   // batch_size idx
  if (tid >= hidden_size || gid >= batch_size) return;
  for (int i = gid; i < batch_size; i += gridDim.x) {
    for (int j = tid; j < hidden_size; j += blockDim.x) {
      int output_idx = j + i * hidden_size;
      int input_idx = j + i * hidden_size * seq_len;
      output[output_idx] =
          ReduceOp<T, t>(input, input_idx, hidden_size, seq_len);
    }
  }
}

template static __global__ void
ReduceAixsOne<float, layers::types::PoolType::kMax>(const float* input,
                                                    float* output,
                                                    int batch_size, int seq_len,
                                                    int hidden_size);
template static __global__ void
ReduceAixsOne<float, layers::types::PoolType::kMean>(const float* input,
                                                     float* output,
                                                     int batch_size,
                                                     int seq_len,
                                                     int hidden_size);

template <typename T, layers::types::PoolType t>
static void GPUReduceAxisOne(const T* input, T* output, int batch_size,
                             int seq_len, int hidden_size) {
  dim3 grid_size(batch_size);
  dim3 block_size(max(1024, hidden_size));
  ReduceAixsOne<T, t><<<grid_size, block_size>>>(input, output, batch_size,
                                                 seq_len, hidden_size);
}

template static void GPUReduceAxisOne<float, layers::types::PoolType::kMax>(
    const float* input, float* output, int batch_size, int seq_len,
    int hidden_size);

template static void GPUReduceAxisOne<float, layers::types::PoolType::kMean>(
    const float* input, float* output, int batch_size, int seq_len,
    int hidden_size);

template <typename T>
void gpu_sequence(T* data_ptr, int64_t size) {
  thrust::device_ptr<T> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::sequence(thrust::device, data_dev_ptr, data_dev_ptr + size);
}

template void gpu_sequence<int64_t>(int64_t* data_ptr, int64_t size);
template void gpu_sequence<float>(float* data_ptr, int64_t size);

template <typename T>
void GPUFill(T* data_ptr, int64_t size, T val) {
  thrust::device_ptr<T> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::fill(thrust::device, data_dev_ptr, data_dev_ptr + size, val);
}

template void GPUFill<int64_t>(int64_t* data_ptr, int64_t size, int64_t val);
template void GPUFill<float>(float* data_ptr, int64_t size, float val);

struct negative_functor {
  __host__ __device__ float operator()(const int64_t& v) const {
    return -10000.0f * (1 - v);
  }
};
void GPUTransform(int64_t* src_data_ptr, float* dst_data_ptr,
                  const int64_t size) {
  negative_functor func;
  thrust::device_ptr<int64_t> src_data_ptr_dev_ptr =
      thrust::device_pointer_cast(src_data_ptr);
  thrust::device_ptr<float> dst_data_ptr_dev_ptr =
      thrust::device_pointer_cast(dst_data_ptr);
  thrust::transform(src_data_ptr_dev_ptr, src_data_ptr_dev_ptr + size,
                    dst_data_ptr_dev_ptr, func);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
