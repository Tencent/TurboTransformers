

#include "turbo_transformers/layers/sequence_pool.h"

#include "turbo_transformers/layers/kernels/seq_pool.h"

namespace turbo_transformers {
namespace layers {

void SequencePool::operator()(const core::Tensor &input,
                              core::Tensor *output) const {
  kernels::SeqPool<float>(input, pool_type_, output);
}

}  // namespace layers
}  // namespace turbo_transformers
