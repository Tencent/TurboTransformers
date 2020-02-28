#pragma once
#include <cuda_runtime.h>

namespace fast_transformers {
namespace layers {
namespace kernels {

template <typename T>
void GPUAddBiasGeLUActKernel(
    const T* bias_data, T* out_data, int64_t batch_size, int64_t feature_dim,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
