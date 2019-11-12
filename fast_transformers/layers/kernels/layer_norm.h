#pragma once
#include <numeric>
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

extern void LayerNorm(core::Tensor& out_tensor, const core::Tensor& gamma,
                      const core::Tensor& beta);

extern void AddBiasLayerNorm(float* out, const float* input, const float* bias,
                             const float* gamma, const float* beta, int m,
                             int n);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
