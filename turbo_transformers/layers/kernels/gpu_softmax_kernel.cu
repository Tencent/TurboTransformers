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

#include <cub/cub.cuh>
#include <numeric>

#include "turbo_transformers/layers/kernels/gpu_softmax_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

namespace {
template <typename T, int Len>
struct Array {
  __device__ __forceinline__ Array() {}
  __device__ __forceinline__ Array(T* inputs) {
    for (int i = 0; i < Len; ++i) {
      data[i] = inputs[i];
    }
  }
  T data[Len];
};

template <typename T, int Len>
struct ArrayAddFunc {
  __device__ __forceinline__ Array<T, Len> operator()(const Array<T, Len>& p1,
                                                      const Array<T, Len>& p2) {
    Array<T, Len> result;
    for (int i = 0; i < Len; ++i) {
      result.data[i] = p1.data[i] + p2.data[i];
    }
    return result;
  }
};

template <typename T, int Len>
struct ArrayMaxFunc {
  __device__ __forceinline__ Array<T, Len> operator()(const Array<T, Len>& p1,
                                                      const Array<T, Len>& p2) {
    Array<T, Len> result;
    for (int i = 0; i < Len; ++i) {
      result.data[i] = p1.data[i] > p2.data[i] ? p1.data[i] : p2.data[i];
    }
    return result;
  }
};

template <int BlockDim, int K>
__global__ void cub_softmax_kernel_k(float* qk_buf_, const float* attr_mask,
                                     const int batch_size, const int head_num,
                                     const int from_seq_len,
                                     const int to_seq_len, const float scaler,
                                     bool is_2D) {
  __shared__ typename cub::BlockReduce<Array<float, K>, BlockDim>::TempStorage
      temp_storage;
  __shared__ float s_sum[K], s_max[K];
  float tmp[K];
  int qk_offset = blockIdx.x * K * to_seq_len;

  if (threadIdx.x < to_seq_len) {
    float mask_val = 0.;
    for (int i = 0; i < K; ++i) {
      float qk = qk_buf_[threadIdx.x + qk_offset + to_seq_len * i];
      if (attr_mask != nullptr) {
        int batch_id = (blockIdx.x * K + i) / (head_num * from_seq_len);
        int from_seq_id = (blockIdx.x * K + i) % from_seq_len;
        mask_val = attr_mask[threadIdx.x +
                            (is_2D ? (batch_id * to_seq_len)
                                    : (batch_id * from_seq_len + from_seq_id) *
                                          to_seq_len)];
      } else {
        mask_val = 0.0f;
      }
      // mask_val = (1.0f - mask_val) * -10000.0f;
      tmp[i] = qk * scaler + mask_val;
    }
  } else {
    for (int i = 0; i < K; ++i) {
      tmp[i] = -1e20f;
    }
  }

  Array<float, K> max_val =
      cub::BlockReduce<Array<float, K>, BlockDim>(temp_storage)
          .Reduce(Array<float, K>(tmp), ArrayMaxFunc<float, K>());

  if (threadIdx.x == 0) {
    for (int i = 0; i < K; ++i) {
      s_max[i] = max_val.data[i];
    }
  }
  __syncthreads();

  float qk_tmp[K];
  for (int i = 0; i < K; ++i) {
    qk_tmp[i] = threadIdx.x < to_seq_len ? __expf((tmp[i] - s_max[i])) : 0.0f;
  }

  Array<float, K> sum_val =
      cub::BlockReduce<Array<float, K>, BlockDim>(temp_storage)
          .Reduce(Array<float, K>(qk_tmp), ArrayAddFunc<float, K>());

  if (threadIdx.x == 0) {
    for (int i = 0; i < K; ++i) {
      s_sum[i] = sum_val.data[i] + 1e-6f;
    }
  }
  __syncthreads();

  if (threadIdx.x < to_seq_len) {
    for (int i = 0; i < K; ++i) {
      qk_buf_[threadIdx.x + qk_offset + to_seq_len * i] =
          (qk_tmp[i] / s_sum[i]);
    }
  }
}
}  // namespace

#define SOFTMAX_KERNEL_CASE(BlockDim, ...)                 \
  case (BlockDim):                                         \
    if (row_per_thread_block == RowsPerThreadBlock) {      \
      cub_softmax_kernel_k<BlockDim, RowsPerThreadBlock>   \
          <<<grid, block, 0, stream>>>(__VA_ARGS__);       \
    } else {                                               \
      cub_softmax_kernel_k<BlockDim, OneRowPerThreadBlock> \
          <<<grid, block, 0, stream>>>(__VA_ARGS__);       \
    }                                                      \
    break

#define RUN_KERNEL(...)                                         \
  do {                                                          \
    switch (block.x) {                                          \
      SOFTMAX_KERNEL_CASE(32, __VA_ARGS__);                     \
      SOFTMAX_KERNEL_CASE(64, __VA_ARGS__);                     \
      SOFTMAX_KERNEL_CASE(96, __VA_ARGS__);                     \
      SOFTMAX_KERNEL_CASE(128, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(160, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(192, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(224, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(256, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(288, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(320, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(352, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(384, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(416, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(448, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(480, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(512, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(544, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(576, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(608, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(640, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(672, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(704, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(736, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(768, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(800, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(832, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(864, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(896, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(928, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(960, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(992, __VA_ARGS__);                    \
      SOFTMAX_KERNEL_CASE(1024, __VA_ARGS__);                   \
      default:                                                  \
        throw std::runtime_error("The block.x should be 32x."); \
    }                                                           \
  } while (0)

template <>
void GPUSoftmaxMask(float* qk_buf, const float* attr_mask, int64_t batch_size,
                    int64_t head_num, int64_t from_seq_len, int64_t to_seq_len,
                    float scale, bool is_2D, cudaStream_t stream) {
  dim3 block, grid;
  int high_dim_size = batch_size * head_num * from_seq_len;
  const int OneRowPerThreadBlock = 1;
  const int RowsPerThreadBlock = 2;
  int row_per_thread_block = OneRowPerThreadBlock;
  if ((head_num * from_seq_len) % RowsPerThreadBlock == 0) {
    row_per_thread_block = RowsPerThreadBlock;
  }
  // block size must be 32x, so warp reduce can work
  block.x = (to_seq_len + 31) / 32 * 32;
  grid.x = high_dim_size / row_per_thread_block;
  // Because there are many function templates, the compilation speed may be
  // slow.
  RUN_KERNEL(qk_buf, attr_mask, batch_size, head_num, from_seq_len, to_seq_len,
             scale, is_2D);
}
#undef RUN_KERNEL
#undef SOFTMAX_KERNEL_CASE
}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
