

#include "turbo_transformers/layers/addbias_layernorm.h"

#include "turbo_transformers/layers/kernels/layer_norm.h"

namespace turbo_transformers {
namespace layers {

void FusedAddBiasLayerNorm::operator()(const core::Tensor& input_tensor,
                                       core::Tensor* output_tensor) const {
  kernels::AddBiasLayerNorm<float>(input_tensor, bias, norm_weight, norm_bias,
                                   output_tensor, 1e-12, "AddBiasLayerNorm");
}

}  // namespace layers
}  // namespace turbo_transformers
