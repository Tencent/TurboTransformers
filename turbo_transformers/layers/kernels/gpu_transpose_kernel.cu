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

#include <cstdio>
#include <numeric>

#include "turbo_transformers/layers/kernels/gpu_transpose_kernel.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

/*
   (batch_size, seq_len, weight_num, num_attention_heads, size_per_head) ->
   (weight_num, batch_size, head_num, seq_len, size_per_head)
   */
static __global__ void split_add_bias_transpose_for_score(
    const float* input_data, const float* bias_data, float* output_data,
    const int batch_size, const int seq_len, const int head_num,
    const int size_per_head, const int word_per_block) {
  const float* data_ptr;
  float* buf_ptr;
  const float* bias_ptr;

  int m = batch_size * seq_len;
  int n = head_num * size_per_head;

  int col_offset = blockIdx.x % 3;
  int row_offset = blockIdx.x / 3 * word_per_block % m;

  data_ptr = input_data + row_offset * 3 * n + col_offset * n;
  buf_ptr = output_data + col_offset * m * n;
  bias_ptr = bias_data + col_offset * n;

  int head_id = threadIdx.x % n / size_per_head;
  int hidden_id = threadIdx.x % size_per_head;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 300
  float bias = __ldg(&bias_ptr[threadIdx.x]);
#else
  float bias = bias_ptr[threadIdx.x];
#endif

  for (int i = row_offset; i < row_offset + word_per_block; ++i) {
    if (i >= m) break;
    int seq_id = i % seq_len;
    int batch_id = i / seq_len;

    int target_id = batch_id * (seq_len * head_num * size_per_head) +
                    head_id * seq_len * size_per_head + seq_id * size_per_head +
                    hidden_id;
    float tmp = data_ptr[threadIdx.x] + bias;

    buf_ptr[target_id] = tmp;
    data_ptr += 3 * n;
  }
}

template <>
void GPUSplitAddBiasTransposeForScore(
    const float* input_data, const float* bias_data, float* out_data,
    int64_t batch_size, int64_t seq_len, int64_t weight_num,
    int64_t num_attention_heads, int64_t size_per_head, cudaStream_t stream) {
  const int word_per_block = 32;
  const int n = num_attention_heads * size_per_head;
  const int m = batch_size * seq_len;
  if (n > 1024) {
    throw std::runtime_error(
        "GPUSplitAddBiasTransposeForScore thread block size large than 1024");
  }
  // x, y
  dim3 grid(3 * (m + word_per_block - 1) / word_per_block);
  dim3 block(n);
  split_add_bias_transpose_for_score<<<grid, block, 0, stream>>>(
      input_data, bias_data, out_data, batch_size, seq_len, num_attention_heads,
      size_per_head, word_per_block);
}

// copyright nvidia
static __global__ void transpose(const float* src, float* dst,
                                 const int batch_size, const int seq_len,
                                 const int head_num, const int size_per_head) {
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
  dst[batch_id * (head_num * seq_len * size_per_head) +
      seq_id * head_num * size_per_head + head_id * size_per_head +
      threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template <>
void GPUTransposeForScore(const float* input_data, float* output_data,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream) {
  const int seq_per_block = 1;
  dim3 grid, block;
  grid.x = batch_size * num_attention_heads * seq_len / seq_per_block;
  block.x = seq_per_block * size_per_head;
  if (block.x > 1024) {
    throw std::runtime_error(
        "GPUTransposeForScore thread block size large than 1024");
  }
  transpose<<<grid, block, 0, stream>>>(input_data, output_data, batch_size,
                                        seq_len, num_attention_heads,
                                        size_per_head);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
