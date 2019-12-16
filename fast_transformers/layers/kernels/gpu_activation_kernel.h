#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUAddBiasGeLUActKernel(const T* bias_data, T* out_data, int64_t m, int64_t n, cudaStream_t stream); 

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
