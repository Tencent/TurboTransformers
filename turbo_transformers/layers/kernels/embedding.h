

#pragma once
#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool Add>
void LookupEmbedding(core::Tensor *out_tensor,
                     const core::Tensor &embedding_table,
                     const core::Tensor &ids_tensor,
                     const std::string name = "LookupEmbedding");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
