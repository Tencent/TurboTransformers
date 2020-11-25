

#pragma once

#include <string>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

// The input's shape is (batch_size, seq_len, hidden_size)
// and the output's shape is (batch_size, hidden_size)
// The pool_type could be max, mean, first, last.
template <typename T>
void SeqPool(const core::Tensor &input, layers::types::PoolType pool_type,
             core::Tensor *output);

layers::types::PoolType GetPoolType(const std::string &pool_type);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
