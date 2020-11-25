

#pragma once
#include <cuda_runtime.h>

#include "turbo_transformers/layers/types.h"

namespace turbo_transformers {
namespace layers {
namespace kernels {
using types::ActivationType;
template <typename T, ActivationType ActType>
void GPUAddBiasActKernel(const T* bias_data, int64_t batch_size,
                         int64_t feature_dim, cudaStream_t stream, T* out_data);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
