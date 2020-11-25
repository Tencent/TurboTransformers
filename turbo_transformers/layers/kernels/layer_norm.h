

#pragma once
#include <numeric>

#include "turbo_transformers/core/tensor.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <typename T>
extern void LayerNorm(const core::Tensor& gamma, const core::Tensor& beta,
                      core::Tensor* out_tensor, T eps = 1e-12,
                      const std::string name = "LayerNorm");

template <typename T>
extern void AddBiasLayerNorm(const core::Tensor& input_tensor,
                             const core::Tensor& bias_tensor,
                             const core::Tensor& gamma_tensor,
                             const core::Tensor& beta_tensor,
                             core::Tensor* out_tensor, T eps = 1e-12,
                             const std::string name = "AddBiasLayerNorm");

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
