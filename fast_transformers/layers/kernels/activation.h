#pragma once
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

void AddBiasGeLUAct(float* out, const float* bias, int64_t m, int64_t n);

void AddBiasGeLUAct(const core::Tensor& bias, core::Tensor* inout);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
