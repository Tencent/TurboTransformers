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

#pragma once

namespace fast_transformers {
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
void GPUTransposeForScore(const T* input_data, T* output_data,
                          int64_t batch_size, int64_t seq_len,
                          int64_t num_attention_heads, int64_t size_per_head,
                          cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
