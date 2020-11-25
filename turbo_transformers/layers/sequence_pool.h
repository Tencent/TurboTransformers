

#pragma once
#include <string>

#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/kernels/seq_pool.h"
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {

class SequencePool {
 public:
  explicit SequencePool(const std::string &pool_type) {
    pool_type_ = kernels::GetPoolType(pool_type);
  }
  explicit SequencePool(layers::types::PoolType pt) : pool_type_(pt) {}

  void operator()(const core::Tensor &input_tensor, core::Tensor *output) const;

 private:
  layers::types::PoolType pool_type_;
};

}  // namespace layers
}  // namespace turbo_transformers
