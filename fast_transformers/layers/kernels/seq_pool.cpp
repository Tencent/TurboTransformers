#include "fast_transformers/layers/kernels/seq_pool.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

#include "fast_transformers/core/enforce.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

namespace {
template <typename T>
struct AvgProcess {
  static inline void InitValue(T* ptr, int64_t len) {
    memset(ptr, 0, len * sizeof(T));
  }

  static inline int ProcessEle(T* ptr, int64_t idx, T ele) { ptr[idx] += ele; }

  static inline void Finalize(T* ptr, int64_t len, int64_t seq_len) {
#pragma omp parallel simd
    for (int64_t i = 0; i < len; ++i) {
      ptr[i] /= seq_len;
    }
  }
};

template <typename T>
struct MaxProcess {
  static inline void InitValue(T* ptr, int64_t len) {
#pragma omp parallel simd
    for (int64_t i = 0; i < len; ++i) {
      ptr[i] = -std::numeric_limits<T>::max();
    }
  }

  static inline int ProcessEle(T* ptr, int64_t idx, T ele) {
    if (ptr[idx] < ele) {
      ptr[idx] = ele;
    }
  }

  static inline void Finalize(T* ptr, int64_t len, int64_t seq_len) {}
};
}  // namespace

template <typename T, typename Process>
static void SeqPoolWithProcess(const core::Tensor& input,
                               core::Tensor* output) {
  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);

  const T* in_ptr = input.data<T>();
  T* out_ptr = output->mutableData<T>();

#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    T* sub_out_ptr = out_ptr + i * hidden_size;
    Process::InitValue(sub_out_ptr, hidden_size);
#pragma omp parallel simd
    for (int64_t j = 0; j < seq_len; ++j) {
      const T* sub_in_ptr = in_ptr + i * seq_len * hidden_size + j;

#pragma omp parallel simd
      for (int64_t k = 0; k < hidden_size; k++) {
        Process::ProcessEle(sub_out_ptr, k, sub_in_ptr[k]);
      }
    }

    Process::Finalize(sub_out_ptr, hidden_size, seq_len);
  }
}

template <typename T>
static void SeqPoolWithIdx(const core::Tensor& input, int64_t idx,
                           core::Tensor* output) {
  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);
  FT_ENFORCE(idx >= 0 && idx < seq_len, "The idx should be in [0, %d].",
             seq_len);

  const T* in_ptr = input.data<T>();
  T* out_ptr = output->mutableData<T>();

#pragma omp parallel for
  for (int64_t i = 0; i < batch_size; ++i) {
    const T* sub_in_ptr = in_ptr + i * seq_len * hidden_size + idx;
    T* sub_out_ptr = out_ptr + i * hidden_size;
    std::copy(sub_in_ptr, sub_in_ptr + hidden_size, sub_out_ptr);
  }
}

template <typename T>
void SeqPool(const core::Tensor& input, PoolType pool_type,
             core::Tensor* output) {
  FT_ENFORCE_EQ(input.n_dim(), 3,
                "The input's dim should be 3, but the input's dim is %d",
                input.n_dim());

  auto batch_size = input.shape(0);
  auto seq_len = input.shape(1);
  auto hidden_size = input.shape(2);

  output->Reshape<T>({batch_size, hidden_size});

  switch (pool_type) {
    case PoolType::kMax:
      SeqPoolWithProcess<T, MaxProcess>(input, output);
      break;
    case PoolType::kAvg:
      SeqPoolWithProcess<T, AvgProcess>(input, output);
      break;
    case PoolType::kFirst:
      SeqPoolWithIdx<T>(input, output, 0);
      break;
    case PoolType::kLast:
      SeqPoolWithIdx<T>(input, output, seq_len - 1);
      break;
  }
}

PoolType GetPoolType(const std::string& pool_type) {
  static std::unordered_map<std::string, PoolType> pool_type_map(
      {{"First", PoolType::kFirst},
       {"Last", PoolType::kLast},
       {"Mean", PoolType::kAvg},
       {"Max", PoolType::kMax}});
  auto iter = pool_type_map.find(pool_type);
  if (iter == pool_type_map.end()) {
    FT_THROW(
        "The input pool_type(%s) is not int ['First', 'Last', 'Mean', 'Max'].",
        pool_type);
  }
  return iter->second;
}
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
