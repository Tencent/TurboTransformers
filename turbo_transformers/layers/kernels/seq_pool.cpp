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

#include "turbo_transformers/layers/kernels/seq_pool.h"
#ifdef FT_WITH_CUDA
#include "turbo_transformers/layers/kernels/gpu_utils.h"
#endif

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "turbo_transformers/core/enforce.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

namespace {

template <typename T, PoolType t>
inline void ProcessEle(const T* in_ptr, T* out_ptr, int64_t seq_len,
                       int64_t hidden_size);
template <>
inline void ProcessEle<float, PoolType::kMean>(const float* in_ptr,
                                               float* out_ptr, int64_t seq_len,
                                               int64_t hidden_size) {
  if (hidden_size < 1)
    FT_THROW(
        "Avg Pooling on tensor whose leading dimension should be larger than "
        "0");
  for (int64_t i = 0; i < hidden_size; ++i) {
    out_ptr[i] = 0.;
    for (int64_t j = i; j < seq_len * hidden_size; j += hidden_size) {
      out_ptr[i] += in_ptr[j];
    }
    out_ptr[i] /= seq_len;
  }
};

template <>
inline void ProcessEle<float, PoolType::kMax>(const float* in_ptr,
                                              float* out_ptr, int64_t seq_len,
                                              int64_t hidden_size) {
  if (hidden_size < 1)
    FT_THROW(
        "Max Pooling on tensor whose leading dimension should be larger than "
        "0");
  for (int64_t i = 0; i < hidden_size; ++i) {
    out_ptr[i] = std::numeric_limits<float>::lowest();
    for (int64_t j = i; j < seq_len * hidden_size; j += hidden_size) {
      out_ptr[i] = std::max(out_ptr[i], in_ptr[j]);
    }
  }
};

template <typename T, PoolType t>
void SeqPoolWithProcess(const core::Tensor& input, core::Tensor* output) {
  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);

  const T* in_ptr = input.data<T>();
  T* out_ptr = output->mutableData<T>();

  if (input.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < batch_size; ++i) {
      ProcessEle<T, t>(in_ptr + i * hidden_size * seq_len,
                       out_ptr + i * hidden_size, seq_len, hidden_size);
    }
  } else {
#ifdef FT_WITH_CUDA
    gpu_reduce_axis_one<T, t>(in_ptr, out_ptr, batch_size, seq_len,
                              hidden_size);
#endif
  }
}

template <typename T>
void SeqPoolWithIdx(const core::Tensor& input, int64_t idx,
                    core::Tensor* output) {
  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);
  FT_ENFORCE(idx >= 0 && idx < seq_len, "The idx should be in [0, %d].",
             seq_len);

  const T* in_ptr = input.data<T>();
  T* out_ptr = output->mutableData<T>();
  int64_t stride = seq_len * hidden_size;
  if (input.device_type() == kDLCPU) {
#pragma omp parallel for
    for (int64_t i = 0; i < batch_size; ++i) {
      const T* sub_in_ptr = in_ptr + i * stride + idx * hidden_size;
      T* sub_out_ptr = out_ptr + i * hidden_size;
      std::copy(sub_in_ptr, sub_in_ptr + hidden_size, sub_out_ptr);
    }
  } else if (input.device_type() == kDLGPU) {
#ifdef FT_WITH_CUDA
    for (int64_t i = 0; i < batch_size; ++i) {
      const T* sub_in_ptr = in_ptr + i * stride + idx * hidden_size;
      T* sub_out_ptr = out_ptr + i * hidden_size;
      turbo_transformers::layers::kernels::gpu_copy(sub_in_ptr, sub_out_ptr,
                                                    hidden_size);
    }
#endif
  } else {
    FT_THROW("device_type %d is not supported for SeqPoolWithIdx",
             input.device_type());
  }
}
}  // namespace

template <typename T>
void SeqPool(const core::Tensor& input, PoolType pool_type,
             core::Tensor* output) {
  FT_ENFORCE_EQ(input.n_dim(), 3,
                "The input's dim should be 3, but the input's dim is %d",
                input.n_dim());

  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);

  output->Reshape<T>({batch_size, hidden_size}, input.device_type(),
                     input.device_id());

  switch (pool_type) {
    case PoolType::kMax:
      SeqPoolWithProcess<T, PoolType::kMax>(input, output);
      break;
    case PoolType::kMean:
      SeqPoolWithProcess<T, PoolType::kMean>(input, output);
      break;
    case PoolType::kFirst:
      SeqPoolWithIdx<T>(input, 0, output);
      break;
    case PoolType::kLast:
      SeqPoolWithIdx<T>(input, seq_len - 1, output);
      break;
  }
}

template void SeqPool<float>(const core::Tensor& input, PoolType pool_type,
                             core::Tensor* output);

PoolType GetPoolType(const std::string& pool_type) {
#define _EnumCase(EnumValue)         \
  do {                               \
    if (pool_type == #EnumValue) {   \
      return PoolType::k##EnumValue; \
    }                                \
  } while (0)

  _EnumCase(First);
  _EnumCase(Last);
  _EnumCase(Mean);
  _EnumCase(Max);
  FT_THROW(
      "The input pool_type(%s) is not int ['First', 'Last', 'Mean', 'Max'].",
      pool_type);
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
