#pragma once
#include <numeric>

#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
extern void LayerNorm(const core::Tensor& gamma, const core::Tensor& beta,
                      core::Tensor* out_tensor);

template <typename T>
extern void AddBiasLayerNorm(const core::Tensor& input_tensor,
                             const core::Tensor& bias_tensor,
                             const core::Tensor& gamma_tensor,
                             const core::Tensor& beta_tensor,
                             core::Tensor* out_tensor);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
