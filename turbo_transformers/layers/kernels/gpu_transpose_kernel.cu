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
   input : (batch_size, seq_len, weight_num, head_num, size_per_head) ->
   output : (weight_num, batch_size, head_num, seq_len, size_per_head)
   bias (weight_num, head_num, size_per_head)
   */
static __global__ void split_add_bias_transpose_for_score(
    const float* input_data, const float* bias_data, const int batch_size,
    const int seq_len, const int head_num, const int weight_num,
    const int size_per_head, float* output_data) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid;
  int batch_id = bid / (seq_len * weight_num * head_num);
  int seq_id =
      bid % (seq_len * weight_num * head_num) / (weight_num * head_num);
  int weight_id = bid % (weight_num * head_num) / head_num;
  int head_id = bid % head_num;

  int head_num_size_per_head = head_num * size_per_head;
  int weight_id_head_num_size_per_head = weight_id * head_num_size_per_head;
  int head_id_size_per_head = head_id * size_per_head;
  while (idx < size_per_head) {
    float bias_val = bias_data[weight_id_head_num_size_per_head +
                               head_id_size_per_head + idx];
    output_data[weight_id * batch_size * seq_len * head_num_size_per_head +
                batch_id * seq_len * head_num_size_per_head +
                head_id * +seq_id * size_per_head + idx] =
        input_data[batch_id * seq_len * weight_num * head_num_size_per_head +
                   seq_id * weight_num * head_num_size_per_head +
                   weight_id_head_num_size_per_head + head_id_size_per_head +
                   idx] +
        bias_val;
    idx += blockDim.x;
  }
}

template <>
void GPUSplitAddBiasTransposeForScore(
    const float* input_data, const float* bias_data, float* out_data,
    int64_t batch_size, int64_t seq_len, int64_t weight_num,
    int64_t num_attention_heads, int64_t size_per_head, cudaStream_t stream) {
  const int n = size_per_head;
  const int m = batch_size * seq_len * num_attention_heads * weight_num;
  dim3 grid(m);
  dim3 block(min(n, 1024));
  split_add_bias_transpose_for_score<<<grid, block, 0, stream>>>(
      input_data, bias_data, batch_size, seq_len, num_attention_heads,
      weight_num, size_per_head, out_data);
}

static __global__ void transpose(const float* src, float* dst,
                                 const int batch_size, const int seq_len,
                                 const int head_num, const int size_per_head) {
  int tid = threadIdx.x;
  int batch_id = blockIdx.x / (head_num * seq_len);
  int seq_id = blockIdx.x % seq_len;
  int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;

  int idx = tid;
  while (idx < size_per_head) {
    dst[batch_id * (head_num * seq_len * size_per_head) +
        seq_id * head_num * size_per_head + head_id * size_per_head + idx] =
        src[blockIdx.x * size_per_head + idx];
    idx += blockDim.x;
  }
}

/*
   (batch_size, seq_len, num_attention_heads, size_per_head) ->
   (batch_size, head_num, seq_len, size_per_head)
   */
template <>
void GPUTransposeForScore(const float* input_data, float* output_data,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream) {
  dim3 grid, block;
  grid.x = batch_size * num_attention_heads * seq_len;
  block.x = min(1024, int(size_per_head));
  transpose<<<grid, block, 0, stream>>>(input_data, output_data, batch_size,
                                        seq_len, num_attention_heads,
                                        size_per_head);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
