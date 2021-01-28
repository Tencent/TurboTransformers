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

#pragma once
#include <vector>

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUSplitAddBiasTransposeForScore(const T* input_data, const T* bias_data,
                                      T* out_data, int64_t batch_size,
                                      int64_t seq_len, int64_t weight_num,
                                      int64_t num_attention_heads,
                                      int64_t size_per_head,
                                      cudaStream_t stream);

template <typename T>
void GPUSplitAddBiasTransposeForScoreThreeOutput(
    const T* input_data, const T* bias_data, int64_t batch_size,
    int64_t seq_len, int64_t weight_num, int64_t num_attention_heads,
    int64_t size_per_head, cudaStream_t stream, T* q_out_data, T* k_out_data,
    T* v_out_data);

/****
 * input_data 4D (1, sum_query_seq_len, 3, num_attention_heads * size_per_head)
 * q, k, v out 4D (batch, num_attention_heads, max_query_seq_len, size_per_head)
 * */
template <typename T>
void GPUSplitAddBiasTransposeForScoreThreeOutputPad(
    const T* input_data, const T* bias_data,
    const std::vector<int64_t>& seq_len_list, int64_t weight_num,
    int64_t num_attention_heads, int64_t size_per_head, cudaStream_t stream,
    T* q_out_data, T* k_out_data, T* v_out_data);

template <typename T, bool AddBias>
void GPUTransposeForScore(const T* input_data, const T* bias,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream, T* output_data);

template <typename T>
void GPUTransposeForScorePad(const T* input_data, int64_t batch_size,
                             const std::vector<int64_t>& seq_len_list,
                             int64_t num_attention_heads, int64_t size_per_head,
                             cudaStream_t stream, T* output_data);

// used for context attn. forward transpose
template <typename T>
void GPUAddBiasTransposeForScorePad(const T* input_data, const T* bias_data,
                                    const std::vector<int64_t>& seq_len_list,
                                    int64_t num_attention_heads,
                                    int64_t size_per_head, cudaStream_t stream,
                                    T* output_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
