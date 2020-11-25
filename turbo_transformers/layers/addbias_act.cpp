

#include "turbo_transformers/layers/addbias_act.h"

#include "turbo_transformers/layers/kernels/activation.h"

namespace turbo_transformers {
namespace layers {

void FusedAddBiasGELU::operator()(core::Tensor* output_tensor) const {
  kernels::AddBiasAct<float, kernels::ActivationType::Gelu>(bias, output_tensor,
                                                            "AddBiasAct");
}

}  // namespace layers
}  // namespace turbo_transformers
