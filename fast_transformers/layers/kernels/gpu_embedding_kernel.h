#pragma once

namespace fast_transformers {
namespace layers {
namespace kernels {

void GPULookupKernel(float* dst, const float* embedding_table,
                     const int64_t* ids, int64_t vocab_size,
                     int64_t hidden_size, int64_t num_ids, bool is_add,
                     cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace fast_transformers
