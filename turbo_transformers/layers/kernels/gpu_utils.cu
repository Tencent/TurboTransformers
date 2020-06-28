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

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "turbo_transformers/layers/kernels/gpu_utils.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

#define max(a, b) ((a) > (b)) ? (a) : (b)

template <typename T, types::PoolType t>
__inline__ T __device__ ReduceOp(const T* target, int start_idx, int stride,
                                 int len);

template <>
__inline__ float __device__ ReduceOp<float, types::PoolType::kMax>(
    const float* input, int start_idx, int stride, int len) {
  float res = input[start_idx];
  for (int k = 1; k < len; ++k) {
    res = max(res, input[start_idx + stride * k]);
  }
  return res;
}

template <>
__inline__ float __device__ ReduceOp<float, types::PoolType::kMean>(
    const float* input, int start_idx, int stride, int len) {
  float res = input[start_idx];
  for (int k = 1; k < len; ++k) {
    res += input[start_idx + stride * k];
  }
  return res / len;
}

//[batch, seq_len, hidden_size] -> [batch, hidden_size]
template <typename T, types::PoolType t>
__global__ void ReduceAixsOne(const T* input, T* output, int batch_size,
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

template __global__ void ReduceAixsOne<float, types::PoolType::kMax>(
    const float* input, float* output, int batch_size, int seq_len,
    int hidden_size);
template __global__ void ReduceAixsOne<float, types::PoolType::kMean>(
    const float* input, float* output, int batch_size, int seq_len,
    int hidden_size);

template <typename T, types::PoolType t>
void GPUReduceAxisOne(const T* input, T* output, int batch_size, int seq_len,
                      int hidden_size) {
  dim3 grid_size(batch_size);
  dim3 block_size(max(1024, hidden_size));
  ReduceAixsOne<T, t><<<grid_size, block_size>>>(input, output, batch_size,
                                                 seq_len, hidden_size);
}

template void GPUReduceAxisOne<float, types::PoolType::kMax>(const float* input,
                                                             float* output,
                                                             int batch_size,
                                                             int seq_len,
                                                             int hidden_size);

template void GPUReduceAxisOne<float, types::PoolType::kMean>(
    const float* input, float* output, int batch_size, int seq_len,
    int hidden_size);

template <typename T>
void GPUSequence(T* data_ptr, int64_t size) {
  thrust::device_ptr<T> data_dev_ptr = thrust::device_pointer_cast(data_ptr);
  thrust::sequence(thrust::device, data_dev_ptr, data_dev_ptr + size);
}

template void GPUSequence<int64_t>(int64_t* data_ptr, int64_t size);
template void GPUSequence<float>(float* data_ptr, int64_t size);

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

// TODO(jiaruifang) if the lowese dimension is not 32x and <= 1024,
// implementation is not optimized
template <bool AddInput>
static __global__ void add_bias(const float* input1, const float* input2,
                                const float* bias, int m, int n,
                                float* output) {
  int offset = blockIdx.x * n;
  int block_dim_x = blockDim.x;

  int idx = threadIdx.x;
  if (AddInput) {
    while (idx < n) {
      output[idx + offset] =
          input1[idx + offset] + input2[idx + offset] + bias[idx];
      idx += block_dim_x;
    }
  } else {
    while (idx < n) {
      output[idx + offset] = input1[idx + offset] + bias[idx];
      idx += block_dim_x;
    }
  }
}

template <bool AddInput, typename T>
void GPUAddBias(const T* input1, const T* input2, const T* bias, int64_t m,
                int64_t n, cudaStream_t stream, T* output) {
  dim3 grid(m);
  int block_size = min(1024, (int)((n + 31) / 32 * 32));
  dim3 block(block_size);
  add_bias<AddInput><<<grid, block, 0, stream>>>(
      input1, input2, bias, m, n, output);  // m : high dim, n : low dim
}

template void GPUAddBias<true>(const float* input1, const float* input2,
                               const float* bias, int64_t m, int64_t n,
                               cudaStream_t stream, float* output);
template void GPUAddBias<false>(const float* input1, const float* input2,
                                const float* bias, int64_t m, int64_t n,
                                cudaStream_t stream, float* output);

template <typename Dtype>
__global__ void concat_kernel(const Dtype* t1, const Dtype* t2,
                              int64_t high_dim, int64_t t1_mid_size,
                              int64_t t2_mid_size, int64_t low_dim,
                              Dtype* out_data) {
  int tid = threadIdx.x;  // hidden_size idx
  int gid = blockIdx.x;   // batch_size idx
  int out_mid_dim = t1_mid_size + t2_mid_size;
  int out_high_idx = gid / out_mid_dim;
  int out_mid_idx = gid % out_mid_dim;
  int out_low_dix = tid;

  if (out_mid_idx < t1_mid_size) {
    // copy from t1
    out_data[out_high_idx * out_mid_dim * low_dim + out_mid_idx * low_dim +
             out_low_dix] = t1[out_high_idx * t1_mid_size * low_dim +
                               out_mid_idx * low_dim + out_low_dix];
  } else {
    // copy from t2
    out_data[out_high_idx * out_mid_dim * low_dim + out_mid_idx * low_dim +
             out_low_dix] =
        t2[out_high_idx * t2_mid_size * low_dim +
           (out_mid_idx - t1_mid_size) * low_dim + out_low_dix];
  }
}

template <typename Dtype>
void GPUConcat(const Dtype* t1, const Dtype* t2, const int64_t high_dim,
               const int64_t t1_mid_size, const int64_t t2_mid_size,
               const int64_t low_dim, cudaStream_t stream, Dtype* out_data) {
  assert(low_dim < 1024);
  dim3 grid(high_dim * (t1_mid_size + t2_mid_size));
  int block_size = std::min((int)low_dim, 1024);
  dim3 block(block_size);
  concat_kernel<<<grid, block, 0, stream>>>(
      t1, t2, high_dim, t1_mid_size, t2_mid_size, low_dim,
      out_data);  // m : high dim, n : low dim
}

template void GPUConcat<float>(const float* t1, const float* t2,
                               const int64_t high_dim,
                               const int64_t t1_mid_size,
                               const int64_t t2_mid_size, const int64_t low_dim,
                               cudaStream_t stream, float* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
