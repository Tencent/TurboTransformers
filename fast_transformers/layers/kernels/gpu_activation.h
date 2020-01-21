#pragma once
#include "cub/cub.cuh"
#include "fast_transformers/core/tensor.h"
namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUAddBiasGeLUAct(const core::Tensor& bias, core::Tensor* out,
                       cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
