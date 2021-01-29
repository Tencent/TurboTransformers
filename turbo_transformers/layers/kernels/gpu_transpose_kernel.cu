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

#include <algorithm>
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
                head_id * seq_len * size_per_head + seq_id * size_per_head +
                idx] =
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

/*
 Output transpose results into three tensors
*/
static __global__ void split_add_bias_transpose_for_score_3output(
    const float* input_data, const float* bias_data, const int batch_size,
    const int seq_len, const int head_num, const int weight_num,
    const int size_per_head, float* q_output_data, float* k_output_data,
    float* v_output_data) {
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

  float* output_data = nullptr;
  if (weight_id == 0) {
    output_data = q_output_data;
  } else if (weight_id == 1) {
    output_data = k_output_data;
  } else if (weight_id == 2) {
    output_data = v_output_data;
  }

  while (idx < size_per_head) {
    float bias_val = bias_data[weight_id_head_num_size_per_head +
                               head_id_size_per_head + idx];
    output_data[batch_id * seq_len * head_num_size_per_head +
                head_id * seq_len * size_per_head + seq_id * size_per_head +
                idx] =
        input_data[batch_id * seq_len * weight_num * head_num_size_per_head +
                   seq_id * weight_num * head_num_size_per_head +
                   weight_id_head_num_size_per_head + head_id_size_per_head +
                   idx] +
        bias_val;
    idx += blockDim.x;
  }
}

/*
 foward transpose (1, sum_seq_len, 3, head, hidden) -> 3 X (batch, head,
 max_seq_len, hidden) Output transpose results into three tensors with pad block
 dim size_per_head grid dim product of the rest dim
*/
static __global__ void split_add_bias_transpose_for_score_3output_pad(
    const float* input_data, const float* bias_data, const int batch_size,
    const int max_seq_len, const int64_t* seq_len_list, const int head_num,
    const int weight_num, const int size_per_head, float* q_output_data,
    float* k_output_data, float* v_output_data) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = tid;
  int batch_id = bid / (max_seq_len * weight_num * head_num);
  int seq_id =
      bid % (max_seq_len * weight_num * head_num) / (weight_num * head_num);
  int weight_id = bid % (weight_num * head_num) / head_num;
  int head_id = bid % head_num;

  // if (seq_id  >= seq_len_list[batch_id]) {
  //   return;
  // }

  int head_num_size_per_head = head_num * size_per_head;
  int weight_id_head_num_size_per_head = weight_id * head_num_size_per_head;
  int head_id_size_per_head = head_id * size_per_head;

  float* output_data = nullptr;
  if (weight_id == 0) {
    output_data = q_output_data;
  } else if (weight_id == 1) {
    output_data = k_output_data;
  } else if (weight_id == 2) {
    output_data = v_output_data;
  }

  int acc_seq_len = 0;  // std::accumulate(seq_len_list.begin(),
                        //               seq_len_list.begin() + batch_id, 0);
  for (size_t i = 0; i < batch_id; ++i) {
    acc_seq_len += seq_len_list[i];
  }
  while (idx < size_per_head) {
    if (seq_id >= seq_len_list[batch_id]) {
      output_data[batch_id * max_seq_len * head_num_size_per_head +
                  head_id * max_seq_len * size_per_head +
                  seq_id * size_per_head + idx] = 0.f;
    } else {
      float bias_val = bias_data[weight_id_head_num_size_per_head +
                                 head_id_size_per_head + idx];
      output_data[batch_id * max_seq_len * head_num_size_per_head +
                  head_id * max_seq_len * size_per_head +
                  seq_id * size_per_head + idx] =
          input_data[(acc_seq_len + seq_id) * weight_num *
                         head_num_size_per_head +
                     weight_id_head_num_size_per_head + head_id_size_per_head +
                     idx] +
          bias_val;
    }
    idx += blockDim.x;
  }
}

template <>
void GPUSplitAddBiasTransposeForScoreThreeOutput(
    const float* input_data, const float* bias_data, int64_t batch_size,
    int64_t seq_len, int64_t weight_num, int64_t num_attention_heads,
    int64_t size_per_head, cudaStream_t stream, float* q_out_data,
    float* k_out_data, float* v_out_data) {
  const int n = size_per_head;
  const int m = batch_size * seq_len * num_attention_heads * weight_num;
  dim3 grid(m);
  dim3 block(min(n, 1024));
  split_add_bias_transpose_for_score_3output<<<grid, block, 0, stream>>>(
      input_data, bias_data, batch_size, seq_len, num_attention_heads,
      weight_num, size_per_head, q_out_data, k_out_data, v_out_data);
}

template <>
void GPUSplitAddBiasTransposeForScoreThreeOutputPad(
    const float* input_data, const float* bias_data,
    const std::vector<int64_t>& seq_len_list, int64_t weight_num,
    int64_t num_attention_heads, int64_t size_per_head, cudaStream_t stream,
    float* q_out_data, float* k_out_data, float* v_out_data) {
  const int n = size_per_head;
  int64_t batch_size = seq_len_list.size();
  int64_t max_seq_length =
      *std::max_element(seq_len_list.begin(), seq_len_list.end());
  const int m = batch_size * max_seq_length * num_attention_heads * weight_num;
  dim3 grid(m);
  dim3 block(min(n, 1024));

  int64_t* d_seq_len_list;
  cudaMalloc((void**)&(d_seq_len_list), batch_size * sizeof(int64_t));
  cudaMemcpy(d_seq_len_list, seq_len_list.data(), batch_size * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  split_add_bias_transpose_for_score_3output_pad<<<grid, block, 0, stream>>>(
      input_data, bias_data, batch_size, max_seq_length,
      d_seq_len_list,  // device vector
      num_attention_heads, weight_num, size_per_head, q_out_data, k_out_data,
      v_out_data);
  cudaFree(d_seq_len_list);
}

namespace {

// backward
// batch, head, seq, size_per_head -> batch seq head size_per_head
template <bool AddBias>
__global__ void transpose(const float* src, const float* bias,
                          const int batch_size, const int seq_len,
                          const int head_num, const int size_per_head,
                          float* dst) {
  int tid = threadIdx.x;
  int idx = tid;
  if (AddBias) {
    int batch_id = blockIdx.x / (seq_len * head_num);
    int seq_id = blockIdx.x / head_num % seq_len;
    int head_id = blockIdx.x % head_num;
    while (idx < size_per_head) {
      dst[batch_id * (head_num * seq_len * size_per_head) +
          head_id * seq_len * size_per_head + seq_id * size_per_head + idx] =
          src[blockIdx.x * size_per_head + idx] +
          bias[head_id * size_per_head + idx];
      idx += blockDim.x;
    }
  } else {
    //(batch, head, seq_len, size_per_head) -> (batch, seq_len, head,
    // size_per_head)
    int batch_id = blockIdx.x / (head_num * seq_len);
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
    int seq_id = blockIdx.x % seq_len;

    while (idx < size_per_head) {
      dst[batch_id * (head_num * seq_len * size_per_head) +
          seq_id * head_num * size_per_head + head_id * size_per_head + idx] =
          src[blockIdx.x * size_per_head + idx];
      idx += blockDim.x;
    }
  }
}

// batch, head, max_seq, size_per_head -> 1, sum_seq, head, size_per_head
__global__ void transpose_back_pad(const float* src, const int batch_size,
                                   const int max_seq_len,
                                   const int64_t* seq_len_list,
                                   const int head_num, const int size_per_head,
                                   float* dst) {
  int tid = threadIdx.x;
  int idx = tid;
  //(batch, head, max_seq_len, size_per_head) -> (batch, seq_len, head,
  // size_per_head)
  int batch_id = blockIdx.x / (head_num * max_seq_len);
  int head_id = (blockIdx.x % (head_num * max_seq_len)) / max_seq_len;
  int seq_id = blockIdx.x % max_seq_len;

  if (seq_id >= seq_len_list[batch_id]) {
    return;
  }
  // int64_t acc_seq_len = std::accumulate(seq_len_list.begin(),
  // seq_len_list.begin() + batch_idx, 0);
  int64_t acc_seq_len = 0;
  for (size_t i = 0; i < batch_id; ++i) {
    acc_seq_len += seq_len_list[i];
  }
  while (idx < size_per_head) {
    // set the invalid elements to 0.
    dst[(acc_seq_len + seq_id) * (head_num * size_per_head) +
        head_id * size_per_head + idx] = src[blockIdx.x * size_per_head + idx];
    idx += blockDim.x;
  }
}

// 1, sum_seq, head size_per_head -> batch, head, max_seq, size_per_head
__global__ void add_bias_transpose_forward_pad(
    const float* src, const float* bias, const int batch_size,
    const int max_seq_len, const int64_t* seq_len_list, const int head_num,
    const int size_per_head, float* dst) {
  int tid = threadIdx.x;
  int idx = tid;

  int batch_id = blockIdx.x / (head_num * max_seq_len);
  int head_id = (blockIdx.x % (head_num * max_seq_len)) / max_seq_len;
  int seq_id = blockIdx.x % max_seq_len;

  int64_t acc_seq_len = 0;
  for (size_t i = 0; i < batch_id; ++i) {
    acc_seq_len += seq_len_list[i];
  }
  while (idx < size_per_head) {
    // set the invalid elements to 0.
    if (seq_id >= seq_len_list[batch_id]) {
      dst[blockIdx.x * size_per_head + idx] = 0.f;
    } else {
      dst[blockIdx.x * size_per_head + idx] =
          src[(acc_seq_len + seq_id) * (head_num * size_per_head) +
              head_id * size_per_head + idx] +
          bias[head_id * size_per_head + idx];
    }
    idx += blockDim.x;
  }
}
}  // namespace
/*
   (batch_size, seq_len, num_attention_heads, size_per_head) ->
   (batch_size, head_num, seq_len, size_per_head)
   */
template <typename T, bool AddBias>
void GPUTransposeForScore(const T* input_data, const T* bias,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream, T* output_data) {
  dim3 grid, block;
  grid.x = batch_size * num_attention_heads * seq_len;
  block.x = min(1024, int(size_per_head));
  transpose<AddBias><<<grid, block, 0, stream>>>(input_data, bias, batch_size,
                                                 seq_len, num_attention_heads,
                                                 size_per_head, output_data);
}

template void GPUTransposeForScore<float, true>(
    const float* input_data, const float* bias, int64_t batch_size,
    int64_t seq_len, int64_t num_attention_heads, int64_t size_per_head,
    cudaStream_t stream, float* output_data);

template void GPUTransposeForScore<float, false>(
    const float* input_data, const float* bias, int64_t batch_size,
    int64_t seq_len, int64_t num_attention_heads, int64_t size_per_head,
    cudaStream_t stream, float* output_data);

/*
(1, sum_seq_len, num_attention_heads, size_per_head) ->
(batch_size, head_num, max_seq_len, size_per_head)
*/
template <>
void GPUTransposeForScorePad(const float* input_data, int64_t batch_size,
                             const std::vector<int64_t>& seq_len_list,
                             int64_t num_attention_heads, int64_t size_per_head,
                             cudaStream_t stream, float* output_data) {
  dim3 grid, block;
  int64_t max_seq_length =
      *std::max_element(seq_len_list.begin(), seq_len_list.end());
  grid.x = batch_size * num_attention_heads * max_seq_length;
  block.x = min(1024, int(size_per_head));
  int64_t* d_seq_len_list;
  cudaMalloc((void**)&(d_seq_len_list), batch_size * sizeof(int64_t));
  cudaMemcpy(d_seq_len_list, seq_len_list.data(), batch_size * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  transpose_back_pad<<<grid, block, 0, stream>>>(
      input_data, batch_size, max_seq_length, d_seq_len_list,
      num_attention_heads, size_per_head, output_data);
  cudaFree(d_seq_len_list);
}

// (1, sum_seq_len, head, hidden_size) -> (batch, head, max_seq_len,
// hidden_size)
template <>
void GPUAddBiasTransposeForScorePad(const float* input_data,
                                    const float* bias_data,
                                    const std::vector<int64_t>& seq_len_list,
                                    int64_t num_attention_heads,
                                    int64_t size_per_head, cudaStream_t stream,
                                    float* output_data) {
  dim3 grid, block;
  int64_t batch_size = seq_len_list.size();
  int64_t max_seq_length =
      *std::max_element(seq_len_list.begin(), seq_len_list.end());
  grid.x = batch_size * num_attention_heads * max_seq_length;
  block.x = min(1024, int(size_per_head));
  int64_t* d_seq_len_list;
  cudaMalloc((void**)&(d_seq_len_list), batch_size * sizeof(int64_t));
  cudaMemcpy(d_seq_len_list, seq_len_list.data(), batch_size * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  add_bias_transpose_forward_pad<<<grid, block, 0, stream>>>(
      input_data, bias_data, batch_size, max_seq_length, d_seq_len_list,
      num_attention_heads, size_per_head, output_data);
  cudaFree(d_seq_len_list);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
