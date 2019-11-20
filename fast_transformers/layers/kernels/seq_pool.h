#pragma once
#include <stdint.h>

#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

enum PoolType { kMax, kAvg, kFirst, kLast };

// The input's shape is (batch_size, seq_len, hidden_size)
// and the output's shape is (batch_size, hidden_size)
// The pool_type could be max, mean, first, last.
template <typename T>
void SeqPool(const core::Tensor& input, const PoolType& pool_type,
             core::Tensor* output);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
