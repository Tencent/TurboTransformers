

#pragma once

namespace turbo_transformers {
namespace layers {
namespace kernels {

template <bool Add>
void GPULookupKernel(float* dst, const float* embedding_table,
                     const int64_t* ids, int64_t vocab_size,
                     int64_t hidden_size, int64_t num_ids, cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers
