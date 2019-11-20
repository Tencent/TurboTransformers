#include "fast_transformers/layers/sequence_pool.h"

#include "fast_transformers/layers/kernels/seq_pool.h"

namespace fast_transformers {
namespace layers {

void SequencePool::operator()(const core::Tensor &input,
                              core::Tensor *output) const {
  kernels::SeqPool(input, pool_type_, output);
}

}  // namespace layers
}  // namespace fast_transformers
