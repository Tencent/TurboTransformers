

#pragma once
#include "turbo_transformers/core/tensor.h"
#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

using types::ActivationType;
template <typename T, ActivationType ActType>
void AddBiasAct(const core::Tensor& bias, core::Tensor* out,
                const std::string name = "AddBiasAct");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
