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

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "turbo_transformers/core/enforce.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

namespace {
template <typename T>
struct AvgProcess {
  static void InitValue(T* ptr, int64_t len) {
    memset(ptr, 0, len * sizeof(T));
  }

  static void ProcessEle(T* ptr, int64_t idx, T ele) { ptr[idx] += ele; }

  static void Finalize(T* ptr, int64_t len, int64_t seq_len) {
#pragma omp simd
    for (int64_t i = 0; i < len; ++i) {
      ptr[i] /= seq_len;
    }
  }
};

template <typename T>
struct MaxProcess {
  static void InitValue(T* ptr, int64_t len) {
#pragma omp simd
    for (int64_t i = 0; i < len; ++i) {
      ptr[i] = std::numeric_limits<T>::lowest();
    }
  }

  static void ProcessEle(T* ptr, int64_t idx, T ele) {
    if (ptr[idx] < ele) {
      ptr[idx] = ele;
    }
  }

  static void Finalize(T* ptr, int64_t len, int64_t seq_len) {}
};

template <typename T, typename Process>
void SeqPoolWithProcess(const core::Tensor& input, core::Tensor* output) {
  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);

  const T* in_ptr = input.data<T>();
  T* out_ptr = output->mutableData<T>();

#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    T* sub_out_ptr = out_ptr + i * hidden_size;
    Process::InitValue(sub_out_ptr, hidden_size);
    int64_t stride = i * seq_len * hidden_size;
#pragma omp simd
    for (int64_t j = 0; j < seq_len; ++j) {
      const T* sub_in_ptr = in_ptr + stride + j * hidden_size;
      for (int64_t k = 0; k < hidden_size; k++) {
        Process::ProcessEle(sub_out_ptr, k, sub_in_ptr[k]);
      }
    }

    Process::Finalize(sub_out_ptr, hidden_size, seq_len);
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
#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    const T* sub_in_ptr = in_ptr + i * stride + idx * hidden_size;
    T* sub_out_ptr = out_ptr + i * hidden_size;
    std::copy(sub_in_ptr, sub_in_ptr + hidden_size, sub_out_ptr);
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
      SeqPoolWithProcess<T, MaxProcess<T>>(input, output);
      break;
    case PoolType::kMean:
      SeqPoolWithProcess<T, AvgProcess<T>>(input, output);
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
