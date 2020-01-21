#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

template <bool AddBias, typename T>
void GPULayerNorm(T* out, const T* input, const T* bias, const T* gamma,
                  const T* beta, int m, int n, cudaStream_t stream);
}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
