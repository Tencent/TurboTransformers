#pragma once
#include "fast_transformers/core/tensor.h"

namespace fast_transformers {
namespace layers {
namespace kernels {

void AddBiasGeLUAct(float *out, const float *bias, int64_t m, int64_t n);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
