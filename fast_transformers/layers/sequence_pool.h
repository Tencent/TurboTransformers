#pragma once
#include "fast_transformers/core/tensor.h"
#include "fast_transformers/layers/kernels/seq_pool.h"

namespace fast_transformers {
namespace layers {

class SequencePool {
 public:
  explicit SequencePool(const std::string &pool_type) {
    pool_type_ = kernels::GetPoolType(pool_type);
  }

  void operator()(const core::Tensor &input_tensor, core::Tensor *output) const;

 private:
  kernels::PoolType pool_type_;
};

}  // namespace layers
}  // namespace fast_transformers
